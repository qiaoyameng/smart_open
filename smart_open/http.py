#
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Implements file-like objects for reading from http; ``kerberos=True`` needs the ``requests-kerberos`` package."""

from __future__ import annotations

import io
import logging
import posixpath
import queue
import re
import threading
import urllib.parse
from typing import TYPE_CHECKING, TypedDict

try:
    import requests
except ImportError:
    MISSING_DEPS = True

import smart_open.utils
from smart_open import bytebuffer, constants

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import WriteableBuffer

    from smart_open._typing import TransportParams

DEFAULT_BUFFER_SIZE = 128 * 1024
DEFAULT_READAHEAD_SIZE = 4 * DEFAULT_BUFFER_SIZE
SCHEMES = ("http", "https")

logger = logging.getLogger(__name__)


class _HTTPUri(TypedDict):
    scheme: str
    uri_path: str


_HEADERS = {"Accept-Encoding": "identity"}
"""The headers we send to the server with every HTTP request.

For now, we ask the server to send us the files as they are.
Sometimes, servers compress the file for more efficient transfer, in which case
the client (us) has to decompress them with the appropriate algorithm.
"""

_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


def _content_length(response: requests.Response) -> int:
    """Return the Content-Length, or -1 when it is missing or malformed, which urllib3 also reads as unknown."""
    value = response.headers.get("Content-Length", "")
    return int(value) if value.isdecimal() else -1


def _satisfied_range(content_range: str, start: int) -> tuple[int, int | None] | None:
    """Return the end and total of a Content-Range answering ``bytes=start-``, or None if it does not."""
    match = _CONTENT_RANGE.fullmatch(content_range)
    if match is None:
        return None
    first, end, total = int(match[1]), int(match[2]) + 1, None if match[3] == "*" else int(match[3])
    # RFC 9110 14.4 forbids recombining an invalid range, and an end before the start never advances
    if first != start or end <= start or (total is not None and end > total):
        return None
    return end, total


class _ReadaheadBuffer:
    """Download response bytes in a worker thread, bounded by a total byte budget.

    The worker reads one ``block_size`` block at a time from the response that the
    reader owns and hands finished blocks over through a queue.  A block still being
    downloaded counts towards ``readahead_size`` as well as the queued blocks, so
    when the budget is reached the worker blocks on the queue instead of issuing
    more reads: a consumer that stops reading therefore stalls the download too.

    Read errors stay with the block that hit them and are raised when that block is
    consumed, never from the background thread itself.  A new response replaces the
    old one atomically, so late blocks from a previous response are discarded.
    """

    def __init__(self, block_size: int, readahead_size: int) -> None:
        self.block_size = block_size
        self.readahead_size = readahead_size
        self._continue: Callable[[], requests.Response | None] | None = None
        self._blocks: queue.Queue[tuple[bytes, BaseException | None]] = queue.Queue()
        self._wakeup = threading.Event()
        self._lock = threading.Lock()
        self._response: requests.Response | None = None
        self._generation = 0
        self._closing = False
        self._in_flight = 0
        self._thread = threading.Thread(target=self._run, name="smart_open-http-readahead", daemon=True)
        self._thread.start()

    def set_response(self, response: requests.Response | None) -> None:
        """Point the worker at a new (or None at EOF) response, dropping its old queue."""
        with self._lock:
            self._generation += 1
            self._response = response
            self._in_flight = 0
            self._drain_locked()
            self._wakeup.set()

    def set_continue(self, callback: Callable[[], requests.Response | None] | None) -> None:
        """Register how the worker requests the rest when a server caps a partial response."""
        with self._lock:
            self._continue = callback

    def get(self) -> tuple[bytes, BaseException | None]:
        """Block until the next block for the current response is ready and return it."""
        data, error = self._blocks.get()
        self._consumed(data)
        return data, error

    def get_nowait(self) -> tuple[bytes, BaseException | None] | None:
        """Return an already queued block, or None without waiting for the worker."""
        try:
            data, error = self._blocks.get_nowait()
        except queue.Empty:
            return None
        self._consumed(data)
        return data, error

    def _consumed(self, data: bytes) -> None:
        """Release the budget of a consumed block; empty EOF/error blocks were never counted."""
        if not data:
            return
        with self._lock:
            self._in_flight -= len(data)  # the queued bytes left the buffer, freeing their budget
            self._wakeup.set()

    def close(self) -> None:
        """Stop issuing reads, drop queued blocks and wait for the worker to finish."""
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._generation += 1
            self._response = None
            self._in_flight = 0
            self._drain_locked()
            self._wakeup.set()
        self._thread.join()

    def _drain_locked(self) -> None:
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._closing:
                    return
                response, generation = self._response, self._generation
                if response is None or self.readahead_size - self._in_flight < self.block_size:
                    self._wakeup.clear()
                    should_wait = True
                else:
                    self._in_flight += self.block_size
                    should_wait = False
            if should_wait:
                self._wakeup.wait()
                continue
            data, error = self._read_block(response)
            if not data and error is None:
                response = self._continue_response(generation, response)
                if response is not None:
                    continue  # the server capped the partial body; read on from the next range
                self._finish(generation)  # idle the worker; the reader closes on consuming this block
            self._deliver(generation, data, error)

    def _continue_response(
        self, generation: int, response: requests.Response
    ) -> requests.Response | None:
        """Request the next range after a capped partial body, if one is still needed."""
        with self._lock:
            if self._closing or generation != self._generation:
                return None  # a seek or close superseded this response chain
            callback = self._continue
        if callback is None:
            return None
        try:
            continuation = callback()
        except BaseException as exc:  # delivered to the consumer instead of killing the thread
            self._deliver(generation, b"", exc)
            return None
        with self._lock:
            if self._closing or generation != self._generation:
                if continuation is not None:
                    continuation.close()
                return None
            self._in_flight -= self.block_size  # the empty block was never delivered
            self._response = continuation
            return continuation

    def _finish(self, generation: int) -> None:
        """Idle the worker once the response chain is exhausted."""
        with self._lock:
            if self._closing or generation != self._generation:
                return
            self._response = None  # keep the worker idle until a seek points it at another response

    def _read_block(self, response: requests.Response) -> tuple[bytes, BaseException | None]:
        try:
            return response.raw.read(self.block_size), None
        except BaseException as exc:  # delivered to the consumer instead of killing the thread
            logger.debug("readahead read failed: %r", exc)
            return b"", exc

    def _deliver(self, generation: int, data: bytes, error: BaseException | None) -> None:
        with self._lock:
            if self._closing or generation != self._generation:
                return  # a seek or close superseded this response; the block is stale
            if error is None:
                self._in_flight -= self.block_size - len(data)  # free only the unfilled rest of a short read
            self._blocks.put_nowait((data, error))
            self._wakeup.set()

    @property
    def in_flight(self) -> int:
        """Bytes currently buffered or being downloaded against the budget."""
        with self._lock:
            return self._in_flight


class _ReadaheadCursor:
    """Consume readahead blocks for a reader, keeping its logical position.

    All of the reader's read, readinto and readline calls share this cursor, so
    they advance one position, and bytes past the end of a line stay buffered for
    the next call.  A short block from the server is passed on like a short read
    from a stream: read(size) returns fewer bytes, and readline keeps consuming
    blocks until the line ends.  Blocked read errors surface here, when the data
    is actually consumed.
    """

    def __init__(self, reader: BufferedInputBase, blocks: _ReadaheadBuffer) -> None:
        self._reader = reader
        self._blocks = blocks
        self._current = b""  # the block being consumed; bytes before _offset are spent
        self._offset = 0
        self._eof = False

    def reset(self) -> None:
        """Drop the partial block; called when the reader seeks to another position."""
        self._current = b""
        self._offset = 0
        self._eof = False

    def discard(self, size: int) -> int:
        """Skip up to size buffered bytes for a forward seek, returning how many were skipped."""
        skipped = 0
        while skipped < size:
            data = self._pull_nowait()
            if not data:
                break  # the buffered blocks are exhausted; a seek falls back to a range request
            take = min(len(data), size - skipped)
            self._offset += take
            skipped += take
        self._reader._current_pos += skipped
        return skipped

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes (all remaining bytes when negative), stream semantics."""
        if size == 0:
            self._reader._check_failed()
            return b""
        self._reader._check_failed()  # a failed range or continuation request keeps raising until a seek
        if self._reader.response is None:
            return b""
        parts: list[bytes] = []
        total = 0
        while size < 0 or total < size:
            data = self._pull()
            if not data:
                break  # the empty block marks EOF of the current response chain
            want = len(data) if size < 0 else min(len(data), size - total)
            parts.append(data[:want])
            self._offset += want
            total += want
        retval = b"".join(parts)
        self._reader._current_pos += len(retval)
        return retval

    def readline(self, size: int | None = -1) -> bytes:
        """Read one line, consuming blocks until the terminator, the limit or EOF."""
        if size is None:
            size = -1
        if size == 0:
            self._reader._check_failed()
            return b""
        self._reader._check_failed()  # a failed range or continuation request keeps raising until a seek
        if self._reader.response is None:
            return b""
        parts: list[bytes] = []
        limit = size
        while limit != 0:
            data = self._pull()
            if not data:
                break  # EOF
            index = data.find(b"\n")
            if index != -1:
                end = index + 1
                if limit >= 0:
                    end = min(end, limit)
                parts.append(data[:end])
                self._offset += end
                if limit >= 0:
                    limit -= end
                break  # the terminator or the size limit ends the line inside this block
            if limit >= 0 and limit <= len(data):
                parts.append(data[:limit])
                self._offset += limit
                limit = 0
                break  # the size limit ends the line inside this block
            parts.append(data)
            self._offset += len(data)
            if limit >= 0:
                limit -= len(data)
        line = b"".join(parts)
        self._reader._current_pos += len(line)
        return line

    def _pull(self) -> bytes:
        """Return the unconsumed part of the current block, blocking for the next one when spent."""
        if self._offset == len(self._current):
            if self._eof:
                return b""
            data, error = self._blocks.get()
            if error is not None:
                raise error  # a failed block does not mark EOF; the reader's failed state is checked next
            self._current, self._offset = data, 0
            if not data:
                self._eof = True  # the empty block marks EOF of the current response chain
                self._reader._on_readahead_eof()
        return self._current[self._offset :]

    def _pull_nowait(self) -> bytes:
        """Like _pull, but return empty bytes when no block is buffered yet."""
        if self._offset == len(self._current):
            if self._eof:
                return b""
            item = self._blocks.get_nowait()
            if item is None:
                return b""
            data, error = item
            if error is not None:
                raise error  # a failed block does not mark EOF; the reader's failed state is checked next
            self._current, self._offset = data, 0
            if not data:
                self._eof = True
                self._reader._on_readahead_eof()
        return self._current[self._offset :]


def parse_uri(uri_as_string: str) -> _HTTPUri:
    """Parse an ``http://`` or ``https://`` URI into its path component."""
    split_uri = urllib.parse.urlsplit(uri_as_string)
    assert split_uri.scheme in SCHEMES  # noqa: S101  # internal precondition; misuse should crash loudly

    uri_path = split_uri.netloc + split_uri.path
    uri_path = "/" + uri_path.lstrip("/")
    return {"scheme": split_uri.scheme, "uri_path": uri_path}


def open_uri(uri: str, mode: str, transport_params: TransportParams) -> BufferedInputBase:
    """Open an HTTP/HTTPS URI using the given mode and transport params."""
    kwargs = smart_open.utils.check_kwargs(open, transport_params)
    return open(uri, mode, **kwargs)


def open(  # noqa: PLR0913  # legacy public API; refactor in a dedicated PR
    uri: str,
    mode: str,
    kerberos: bool = False,  # noqa: FBT001, FBT002  # public API
    user: str | None = None,
    password: str | None = None,
    cert: str | tuple[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    session: requests.Session | None = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    readahead: bool = False,  # noqa: FBT001, FBT002  # public API
    readahead_size: int = DEFAULT_READAHEAD_SIZE,
) -> BufferedInputBase:
    """Implement streamed reader from a web site.

    Supports Kerberos and Basic HTTP authentication.

    Args:
        uri: The URL to open.
        mode: The mode to open using.
        kerberos: If True, will attempt to use the local Kerberos credentials.
        user: The username for authenticating over HTTP.
        password: The password for authenticating over HTTP.
        cert: If a string, path to ssl client cert file (``.pem``).
            If a tuple, ``('cert', 'key')``.
        headers: Any headers to send in the request. If ``None``, the default headers
            are sent: ``{'Accept-Encoding': 'identity'}``. To use no headers at all,
            set this variable to an empty dict, ``{}``. Range requests, from seeks and from reads
            past a range the server cut short, send ``If-Match`` or ``If-Unmodified-Since`` for
            the version first read, so they fail with 412 when it changed; send
            ``{'Accept-Encoding': 'identity', 'If-Match': '*'}`` to read regardless.
        timeout: Request timeout in seconds.
        session: The ``requests.Session`` object to use with HTTP GET requests.
            Can be used for OAuth2 clients. The session stays owned by the caller;
            the reader never closes it.
        buffer_size: The buffer size to use when performing I/O. With readahead enabled,
            this is the size of each block the background worker downloads.
        readahead: If True, download upcoming blocks in a background worker thread so
            the next read does not wait for a network round trip. Off by default, and
            only ``"rb"`` reads are affected. Pausing consumption pauses the download
            once the readahead budget fills up.
        readahead_size: The total readahead budget in bytes, covering both the blocks
            already buffered and the block currently being downloaded. Must be at least
            ``buffer_size`` so one block can always be in flight. Ignored unless
            ``readahead`` is True.

    Returns:
        A file-like object opened for reading.

    Raises:
        NotImplementedError: If ``mode`` is anything other than ``"rb"``.
        ValueError: If ``readahead_size`` is smaller than ``buffer_size``.

    Note:
        If neither ``kerberos`` nor ``(user, password)`` are set, will connect
        unauthenticated, unless set separately in headers.
    """
    if mode == constants.READ_BINARY:
        fobj = SeekableBufferedInputBase(
            uri,
            mode,
            buffer_size=buffer_size,
            kerberos=kerberos,
            user=user,
            password=password,
            cert=cert,
            headers=headers,
            session=session,
            timeout=timeout,
            readahead=readahead,
            readahead_size=readahead_size,
        )
        fobj.name = posixpath.basename(urllib.parse.urlparse(uri).path)
        return fobj
    msg = f"http support for mode {mode!r} not implemented"
    raise NotImplementedError(msg)


class BufferedInputBase(io.BufferedIOBase):
    """Buffered HTTP reader implementing the `io.BufferedIOBase` interface."""

    name: str
    response: requests.Response | None = None
    _read_buffer: bytebuffer.ByteBuffer | None = None  # so `closed` works if __init__ fails and __del__ runs
    _readahead: _ReadaheadBuffer | None = None
    _readahead_cursor: _ReadaheadCursor | None = None

    def __init__(  # noqa: PLR0913  # legacy public API; refactor in a dedicated PR
        self,
        url: str,
        mode: str = "r",
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        kerberos: bool = False,  # noqa: FBT001, FBT002  # public API
        user: str | None = None,
        password: str | None = None,
        cert: str | tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        session: requests.Session | None = None,
        timeout: float | None = None,
        readahead: bool = False,  # noqa: FBT001, FBT002  # public API
        readahead_size: int = DEFAULT_READAHEAD_SIZE,
    ) -> None:

        self.url = url
        self.cert = cert
        self.session = session or requests

        if kerberos:
            import requests_kerberos  # ty: ignore[unresolved-import]  # optional, install requests-kerberos for kerberos=True

            self.auth = requests_kerberos.HTTPKerberosAuth()
        elif user is not None and password is not None:
            self.auth = (user, password)
        else:
            self.auth = None

        self.buffer_size = buffer_size
        self.mode = mode

        if headers is None:
            self.headers = _HEADERS.copy()
        else:
            self.headers = headers

        self.timeout = timeout
        self.readahead = readahead
        self.readahead_size = readahead_size
        if readahead and readahead_size < buffer_size:
            msg = f"readahead_size must be >= buffer_size ({buffer_size}), got {readahead_size}"
            raise ValueError(msg)

        self.response = self.session.get(
            self.url,
            auth=self.auth,
            cert=self.cert,
            stream=True,
            headers=self.headers,
            timeout=self.timeout,
        )

        if not self.response.ok:
            self.response.close()
            self.response.raise_for_status()

        self._read_buffer = bytebuffer.ByteBuffer(buffer_size)
        self._current_pos = 0
        self._failed = False  # a range request failed, so reads raise instead of reading as EOF
        if readahead:
            self._readahead = _ReadaheadBuffer(buffer_size, readahead_size)
            self._readahead.set_response(self.response)
            self._readahead_cursor = _ReadaheadCursor(self, self._readahead)

    #
    # Override some methods from io.IOBase.
    #
    def close(self) -> None:
        """Flush and close this stream."""
        logger.debug("close: called")
        if not self.closed:
            self._close_response()
            if self._readahead is not None:
                self._readahead.close()  # joins the worker thread after the response releases it
                self._readahead = None
                self._readahead_cursor = None
            self._read_buffer = None

    def _close_response(self) -> None:
        if self._readahead is not None:
            self._readahead.set_response(None)  # stop issuing reads and drop queued blocks
        if self.response is not None:
            self.response.close()
            self.response = None

    def _fill(self) -> int:
        assert self._read_buffer is not None  # noqa: S101  # only called while reading
        assert self.response is not None  # noqa: S101  # only called while reading
        filled = self._read_buffer.fill(self.response.raw)
        if not filled and self._next_response():
            return self._fill()
        return filled

    def _read_rest(self) -> bytes:
        assert self.response is not None  # noqa: S101  # only called while reading
        parts = [self.response.raw.read()]
        while self._next_response():
            parts.append(self.response.raw.read())
        return b"".join(parts)

    def _next_response(self) -> bool:
        """Continue a response that ended early with the next one, which only range readers need."""
        return False

    def _on_readahead_eof(self) -> None:
        """Release the exhausted response; further reads return empty bytes until a seek."""
        if self.response is not None:
            self.response.close()
            self.response = None

    def _check_failed(self) -> None:
        if self._failed:
            msg = f"an earlier range request for {self.url} failed, seek to retry"
            raise OSError(msg)

    @property
    def closed(self) -> bool:
        """Return True if the stream is closed."""
        return self._read_buffer is None  # response is also None at EOF after a seek

    def readable(self) -> bool:
        """Return True if the stream can be read from."""
        return True

    def seekable(self) -> bool:
        """Return False; the base HTTP reader does not support seeking."""
        return False

    #
    # io.BufferedIOBase methods.
    #
    def detach(self) -> io.RawIOBase:
        """Unsupported."""
        raise io.UnsupportedOperation

    def read(self, size: int | None = -1) -> bytes:
        """Mimic the read call to a filehandle object."""
        self._checkClosed()
        if size is None:
            size = -1
        if size < -1:
            msg = f"size must be >= -1, got {size}"
            raise ValueError(msg)

        logger.debug("reading with size: %d", size)
        if self._readahead_cursor is not None:
            return self._readahead_cursor.read(size)
        buf, response = self._read_buffer, self.response
        if buf is None or response is None or size == 0:
            self._check_failed()
            return b""

        if size == -1:
            if len(buf):  # noqa: SIM108  # avoid the unnecessary + when the buffer is empty
                retval = buf.read() + self._read_rest()
            else:
                retval = self._read_rest()
        else:
            # Fill _read_buffer until it contains enough bytes
            while len(buf) < size:
                if self._fill() == 0:
                    break  # EOF reached
            retval = buf.read(size)

        self._current_pos += len(retval)
        return retval

    def read1(self, size: int | None = -1) -> bytes:
        """This is the same as read()."""
        return self.read(size=size)

    def readline(self, size: int | None = -1) -> bytes:
        """Read and return one line, refilling the buffer from the response in chunks."""
        self._checkClosed()
        if self._readahead_cursor is not None:
            return self._readahead_cursor.readline(size)
        buf, response = self._read_buffer, self.response
        if buf is None or response is None:
            self._check_failed()
            return b""
        line = buf.readline(b"\n", size, self._fill)
        self._current_pos += len(line)
        return line

    def readinto(self, b: WriteableBuffer) -> int:
        """Read up to ``len(b)`` bytes into ``b``, and return the number of bytes read."""
        mv = memoryview(b).cast("B")
        data = self.read(len(mv))
        if not data:
            return 0
        mv[: len(data)] = data
        return len(data)


class SeekableBufferedInputBase(BufferedInputBase):
    """Seekable streamed reader from a web site.

    Supports Kerberos, client certificate and Basic HTTP authentication.
    If ``kerberos`` is True, will attempt to use the local Kerberos credentials.
    If ``cert`` is set, will try to use a client certificate. Otherwise, will try
    to use "basic" HTTP authentication via username/password. If none of those are
    set, will connect unauthenticated.
    """

    def __init__(  # noqa: PLR0913  # legacy public API; refactor in a dedicated PR
        self,
        url: str,
        mode: str = "r",
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        kerberos: bool = False,  # noqa: FBT001, FBT002  # public API
        user: str | None = None,
        password: str | None = None,
        cert: str | tuple[str, str] | None = None,
        headers: dict[str, str] | None = None,
        session: requests.Session | None = None,
        timeout: float | None = None,
        readahead: bool = False,  # noqa: FBT001, FBT002  # public API
        readahead_size: int = DEFAULT_READAHEAD_SIZE,
    ) -> None:
        super().__init__(
            url,
            mode,
            buffer_size,
            kerberos,
            user,
            password,
            cert,
            headers,
            session,
            timeout,
            readahead,
            readahead_size,
        )
        assert self.response is not None  # noqa: S101  # set by super().__init__
        self.content_length = _content_length(self.response)
        #
        # We assume the HTTP stream is seekable unless the server explicitly
        # tells us it isn't.  It's better to err on the side of "seekable"
        # because we don't want to prevent users from seeking a stream that
        # does not appear to be seekable but really is.
        #
        self._seekable = self.response.headers.get("Accept-Ranges", "").lower() != "none"
        #
        # Range requests carry a precondition on the version of this first response,
        # so a resource replaced in between fails with 412 instead of mixing versions.
        #
        etag = self.response.headers.get("ETag", "")
        last_modified = self.response.headers.get("Last-Modified")
        if etag and not etag.startswith("W/"):  # If-Match compares strongly, so a weak ETag never matches
            self._precondition = {"If-Match": etag}
        elif last_modified:
            self._precondition = {"If-Unmodified-Since": last_modified}
        else:
            self._precondition = {}
        self._partial: tuple[int, int, int | None] | None = None  # start, end, total from Content-Range
        if self._readahead is not None:
            self._readahead.set_continue(self._continue_readahead)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to the specified position.

        Args:
            offset: The offset in bytes.
            whence: Where the offset is from.

        Returns:
            The position after seeking.

        Raises:
            ValueError: If ``whence`` is not one of ``WHENCE_CHOICES``.
            io.UnsupportedOperation: If the stream is not seekable or the server ignores a nonzero range request.
        """
        logger.debug("seeking to offset: %r whence: %r", offset, whence)
        self._checkClosed()
        if whence not in constants.WHENCE_CHOICES:
            msg = f"invalid whence, expected one of {constants.WHENCE_CHOICES!r}"
            raise ValueError(msg)

        if not self.seekable():
            msg = "stream is not seekable"
            raise io.UnsupportedOperation(msg)

        buf = self._read_buffer
        assert buf is not None  # noqa: S101  # not closed implies buf is set

        if whence == constants.WHENCE_START:
            new_pos = offset
        elif whence == constants.WHENCE_CURRENT:
            new_pos = self._current_pos + offset
        elif self.content_length == -1:  # constants.WHENCE_END without a Content-Length
            msg = "cannot seek from the end of a response of unknown length"
            raise io.UnsupportedOperation(msg)
        else:  # constants.WHENCE_END
            new_pos = self.content_length + offset

        new_pos = smart_open.utils.clamp(
            new_pos, maxval=None if self.content_length == -1 else self.content_length
        )

        if self._current_pos == new_pos and self.response is not None and not self._failed:
            return self._current_pos  # an exhausted or failed response still needs a range request to retry

        # Check if we can satisfy the seek from buffer (forward seek within buffered data)
        if new_pos > self._current_pos:
            distance = new_pos - self._current_pos
            if self._readahead_cursor is not None:
                # Readahead blocks are held by the worker and the cursor, not by _read_buffer
                if self._seek_readahead_forward(distance):
                    return self._current_pos
            elif distance <= len(buf):
                buf.read(distance)
                self._current_pos = new_pos
                return self._current_pos

        logger.debug("http seeking from current_pos: %d to new_pos: %d", self._current_pos, new_pos)

        self._close_response()  # with readahead on, this discards late blocks from the old position
        buf.empty()

        if new_pos != self.content_length:
            self._failed = True  # until the range request succeeds
            self.response = self._range_response(new_pos)
            if self._readahead is not None:
                assert self._readahead_cursor is not None  # noqa: S101  # the two are created together
                self._readahead_cursor.reset()
                self._readahead.set_response(self.response)  # None past EOF leaves the worker idle
        self._failed = False

        self._current_pos = new_pos
        return self._current_pos

    def _seek_readahead_forward(self, distance: int) -> bool:
        """Advance the shared readahead cursor by distance buffered bytes without a new request."""
        assert self._readahead_cursor is not None  # noqa: S101  # only called when readahead is enabled
        skipped = self._readahead_cursor.discard(distance)
        if skipped < distance:
            return False  # not enough buffered bytes yet; the caller falls back to a range request
        return True

    def tell(self) -> int:
        """Return the current stream position."""
        self._checkClosed()
        return self._current_pos

    def seekable(self, *args: object, **kwargs: object) -> bool:
        """Return True if the server reports it accepts byte-range requests."""
        return self._seekable

    def truncate(self, size: int | None = None) -> int:
        """Unsupported."""
        raise io.UnsupportedOperation

    def _range_response(self, start_pos: int) -> requests.Response | None:
        self._partial = None
        response = self._partial_request(start_pos)
        if start_pos and response.status_code == requests.codes.ok:
            response.close()
            if 0 <= _content_length(response) <= start_pos:
                return None  # the whole body ends before the requested range, which reads as EOF like a 416
            msg = "server ignored the range request"
            raise io.UnsupportedOperation(msg)
        content_range = response.headers.get("Content-Range")
        if response.status_code == requests.codes.partial_content and content_range:
            span = _satisfied_range(content_range, start_pos)
            if span is None:
                response.close()
                msg = f"server answered bytes={start_pos}- with Content-Range: {content_range}"
                raise OSError(msg)
            self._partial = (start_pos, *span)
        if response.ok:
            return response
        response.close()
        if response.status_code == requests.codes.precondition_failed:
            msg = f"412 Precondition Failed: {self.url} changed since it was opened, send If-Match: * along with Accept-Encoding: identity to read regardless"
            raise requests.HTTPError(msg, response=response)
        # past EOF of a stream of unknown length, which reads as EOF like io
        if response.status_code != requests.codes.range_not_satisfiable:
            response.raise_for_status()
        return None

    def _continue_readahead(self) -> requests.Response | None:
        """Replace a capped partial response with the next range for the readahead worker."""
        if self._partial is None:
            return None
        start, end, total = self._partial
        old = self.response
        assert old is not None  # noqa: S101  # the worker only continues a live response
        received = old.raw.tell()
        if received != end - start:  # a longer body would repeat bytes in the next request
            msg = f"partial body for bytes {start}-{end - 1} has {received} bytes"
            raise OSError(msg)
        if total is not None and end >= total:
            return None
        old.close()
        self._failed = True  # until the rest arrives
        response = self._range_response(end)
        if response is None and total is not None:  # only an unknown total ends with a 416
            msg = f"server answered 416 to bytes={end}- before the end at {total}"
            raise OSError(msg)
        self._failed = False
        self.response = response
        return response

    def _next_response(self) -> bool:
        """Check the end of a partial body, and request the rest when the server capped the range."""
        if self._partial is None:
            return False
        assert self.response is not None  # noqa: S101  # only called while reading
        start, end, total = self._partial
        received = self.response.raw.tell()
        if received != end - start:  # a longer body would repeat bytes in the next request
            msg = f"partial body for bytes {start}-{end - 1} has {received} bytes"
            raise OSError(msg)
        if total is not None and end >= total:
            return False
        self._close_response()
        self._failed = True  # until the rest arrives
        self.response = self._range_response(end)
        if self._readahead is not None:
            assert self._readahead_cursor is not None  # noqa: S101  # the two are created together
            self._readahead_cursor.reset()
            self._readahead.set_response(self.response)
        if self.response is None and total is not None:  # only an unknown total ends with a 416
            msg = f"server answered 416 to bytes={end}- before the end at {total}"
            raise OSError(msg)
        self._failed = False
        return self.response is not None

    def _partial_request(self, start_pos: int | None = None) -> requests.Response:
        headers = {**self._precondition, **self.headers}
        if start_pos is not None:
            headers["range"] = smart_open.utils.make_range_string(start_pos)

        return self.session.get(
            self.url,
            auth=self.auth,
            stream=True,
            cert=self.cert,
            headers=headers,
            timeout=self.timeout,
        )

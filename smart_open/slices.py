"""Read-only views over half-open byte ranges of seekable sources.

The ranges address the bytes produced by the transport layer, before any
compression or text decoding.  A :class:`SliceView` is therefore usable as a
small, independent binary file: callers that want to decompress or decode a
slice can wrap the view explicitly, but only when the slice itself contains
complete, valid content.
"""

from __future__ import annotations

import builtins
import io
import os
import threading
import urllib.parse
from typing import IO, TYPE_CHECKING, Any

import smart_open.local_file as so_file
import smart_open.transport as so_transport

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

    from smart_open._typing import TransportParams

_SOURCE_LOCK = threading.RLock()
"""Serialize seek-and-read pairs on shared sources.

Slice views keep their own cursors, but two views of the same underlying
handle must not interleave their physical seeks and reads.
"""

_READ_CHUNK_SIZE = 64 * 1024
_HTTP_DEFAULT_BUFFER_SIZE = 128 * 1024


def _sniff_scheme(uri: str) -> str:
    if os.name == "nt" and "://" not in uri:
        uri = "file://" + uri
    return urllib.parse.urlsplit(uri).scheme


def _known_length(source: Any) -> int | None:
    content_length = getattr(source, "content_length", None)
    if isinstance(content_length, int) and content_length >= 0:
        return content_length
    return None


def _local_length(uri: str) -> int | None:
    try:
        return os.path.getsize(so_file.extract_local_path(uri))
    except OSError:
        return None


def _open_source(
    uri: str,
    transport_params: TransportParams | None,
    start: int,
    stop: int | None,
) -> tuple[IO[bytes], int | None, str | None]:
    """Open ``uri`` at the transport layer, without compression or decoding."""
    scheme = _sniff_scheme(uri)
    params: dict[str, Any] = dict(transport_params or {})
    length_hint = None

    if scheme in ("", so_file.SCHEME):
        length_hint = _local_length(uri)

    if scheme in ("http", "https"):
        if stop == start:
            return io.BytesIO(), 0, uri
        #
        # Ask for exactly the slice on the first request.  Never fetch more
        # than the slice needs into the HTTP read buffer.  Later range
        # requests are version-pinned by the HTTP transport.
        #
        params["range_start"] = start
        if stop is not None:
            params["range_stop"] = stop
            if "buffer_size" not in params:
                params["buffer_size"] = max(1, min(_HTTP_DEFAULT_BUFFER_SIZE, stop - start))

    submodule = so_transport.get_transport(scheme)
    source = submodule.open_uri(uri, "rb", params)
    name = getattr(source, "name", None)
    if scheme in ("http", "https"):
        length_hint = _known_length(source)
    return source, length_hint, name if isinstance(name, str) else uri


def _coerce_range(start: int | builtins.slice | tuple[int, int], stop: int | None) -> tuple[int, int | None]:
    if isinstance(start, builtins.slice):
        if start.step not in (None, 1):
            msg = "slice views only support a step of 1"
            raise ValueError(msg)
        start, stop = start.start, start.stop
        if start is None:
            start = 0
    elif isinstance(start, tuple):
        if stop is not None or len(start) != 2:
            msg = "a byte range tuple must contain exactly (start, stop)"
            raise ValueError(msg)
        start, stop = start

    if isinstance(start, bool) or not isinstance(start, int):
        msg = "start must be an integer"
        raise TypeError(msg)
    if start < 0:
        msg = "start must be >= 0"
        raise ValueError(msg)
    if stop is None:
        return start, None
    if isinstance(stop, bool) or not isinstance(stop, int):
        msg = "stop must be an integer or None"
        raise TypeError(msg)
    if stop < start:
        msg = "stop must be >= start"
        raise ValueError(msg)
    return start, stop


class SliceView(io.BufferedIOBase):
    """A read-only, independently positioned view of a byte range.

    Args:
        source: A URI/path, an already-open binary file-like object, or another
            :class:`SliceView`.  Locatable transports are opened at their raw
            byte layer; no compressor or text wrapper is applied.
        start: First byte of the half-open range.  A :class:`slice` object or
            ``(start, stop)`` tuple may be passed instead.
        stop: One past the last byte of the range, or ``None`` for EOF.
        transport_params: Forwarded to the transport when ``source`` is a URI.
        closefd: Whether the view owns and closes ``source``.  Defaults to
            ``True`` for URIs and ``False`` for already-open objects, so an
            external handle is left open when its view is closed.

    The view's ``read``, ``seek``, ``tell``, ``readinto`` and ``readline``
    operations use offsets relative to ``start``.  Each view keeps its own
    cursor, so interleaved views of one handle do not disturb each other.
    """

    name: str | None

    def __init__(  # noqa: PLR0913  # public, file-like constructor
        self,
        source: str | os.PathLike[str] | IO[Any] | SliceView,
        start: int | builtins.slice | tuple[int, int],
        stop: int | None = None,
        transport_params: TransportParams | None = None,
        closefd: bool | None = None,
        *,
        end: int | None = None,
    ) -> None:
        if end is not None:
            if stop is not None:
                msg = "stop and end are mutually exclusive"
                raise TypeError(msg)
            stop = end
        start, stop = _coerce_range(start, stop)

        self._closed = False
        self._pos = 0
        self._pending = b""
        self._source_length: int | None = None
        self._start = start
        self._stop = stop
        self.mode = "rb"

        if isinstance(source, os.PathLike):
            source = os.fspath(source)

        if isinstance(source, str):
            self._source, length_hint, name = _open_source(source, transport_params, start, stop)
            self._owns_source = closefd is not False
            self.name = name
        else:
            self._source = source
            self._owns_source = bool(closefd)
            name = getattr(source, "name", None)
            self.name = name if isinstance(name, str) else None
            length_hint = _known_length(source)

        if length_hint is not None:
            self._source_length = length_hint

    #
    # io.IOBase interface.
    #
    @property
    def closed(self) -> bool:
        """Return True if this view or its underlying source is closed."""
        return self._closed or bool(getattr(self._source, "closed", False))

    def _check_open(self) -> None:
        if self._closed:
            msg = "I/O operation on closed file."
            raise ValueError(msg)
        if getattr(self._source, "closed", False):
            msg = "I/O operation on closed file."
            raise ValueError(msg)

    def readable(self) -> bool:
        """Return True if the view is open for reading."""
        self._check_open()
        return True

    def writable(self) -> bool:
        """Return False; slice views are read-only."""
        self._check_open()
        return False

    def seekable(self) -> bool:
        """Return True; slice views are positioned by byte offset."""
        self._check_open()
        return True

    def close(self) -> None:
        """Close the view, and the transport only when this view owns it."""
        if self._closed:
            return
        self._closed = True
        self._pending = b""
        if self._owns_source:
            self._source.close()

    def __enter__(self) -> SliceView:
        """Enter the runtime context for this view."""
        self._check_open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the view when leaving its runtime context."""
        self.close()

    def __iter__(self) -> SliceView:
        """Iterate over the lines in the slice."""
        return self

    def __next__(self) -> bytes:
        """Return the next line, or raise StopIteration at EOF."""
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def fileno(self) -> int:
        """Return the underlying file descriptor, when there is one."""
        self._check_open()
        return self._source.fileno()

    def flush(self) -> None:
        """Do nothing; there is nothing to flush in a read-only view."""
        self._check_open()

    def detach(self) -> io.RawIOBase:
        """Unsupported."""
        raise io.UnsupportedOperation

    #
    # Slicing helpers.
    #
    def _infer_source_length(self) -> int | None:
        if self._source_length is not None:
            return self._source_length

        content_length = _known_length(self._source)
        if content_length is not None:
            self._source_length = content_length
            return content_length

        try:
            with _SOURCE_LOCK:
                physical_pos = self._source.tell()
                self._source.seek(0, os.SEEK_END)
                source_length = self._source.tell()
                self._source.seek(physical_pos)
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            return None

        if source_length >= 0:
            self._source_length = source_length
        return self._source_length

    def _end(self) -> int | None:
        if self._stop is None:
            return self._infer_source_length()
        if self._source_length is None:
            return self._stop
        return min(self._stop, self._source_length)

    def _length(self) -> int | None:
        end = self._end()
        if end is None:
            return None
        return max(0, end - self._start)

    def _remaining(self) -> int | None:
        length = self._length()
        if length is None:
            return None
        return max(0, length - self._pos)

    def _note_eof(self, absolute_pos: int) -> None:
        if self._source_length is None or absolute_pos < self._source_length:
            self._source_length = absolute_pos

    #
    # File-like reading methods.
    #
    def read(self, size: int | None = -1) -> bytes:
        """Read up to ``size`` bytes from the slice, or all remaining bytes."""
        self._check_open()
        if size is None:
            size = -1
        if not isinstance(size, int):
            msg = "size must be an integer or None"
            raise TypeError(msg)
        if size < -1:
            msg = f"size must be >= -1, got {size}"
            raise ValueError(msg)
        if size == 0:
            return b""

        remaining = self._remaining()
        if remaining is not None and remaining == 0:
            return b""
        want = remaining if size == -1 else size if remaining is None else min(size, remaining)

        parts: list[bytes] = []
        consumed = 0
        if self._pending:
            take = len(self._pending) if want is None else min(want, len(self._pending))
            parts.append(self._pending[:take])
            self._pending = self._pending[take:]
            consumed += take

        if want is None or consumed < want:
            read_size = None if want is None else want - consumed
            data = self._read_source(read_size, self._pos + consumed)
            parts.append(data)
            consumed += len(data)

        self._pos += consumed
        return b"".join(parts)

    def read1(self, size: int | None = -1) -> bytes:
        """Read up to ``size`` bytes, like :meth:`read`."""
        return self.read(size)

    def readinto(self, b: WriteableBuffer) -> int:
        """Read up to ``len(b)`` bytes into ``b`` and return the count."""
        view = memoryview(b).cast("B")
        data = self.read(len(view))
        if not data:
            return 0
        view[: len(data)] = data
        return len(data)

    def readline(self, size: int | None = -1) -> bytes:
        """Read and return one line, bounded by the slice and ``size``."""
        self._check_open()
        if size is None:
            size = -1
        if not isinstance(size, int):
            msg = "size must be an integer or None"
            raise TypeError(msg)
        if size < -1:
            msg = f"size must be >= -1, got {size}"
            raise ValueError(msg)
        if size == 0:
            return b""

        remaining = self._remaining()
        if remaining is not None and remaining == 0:
            return b""
        max_read = remaining if size == -1 else size if remaining is None else min(size, remaining)

        chunks: list[bytes] = []
        total = 0
        while max_read is None or total < max_read:
            chunk_size = _READ_CHUNK_SIZE if max_read is None else min(_READ_CHUNK_SIZE, max_read - total)
            data = self.read(chunk_size)
            if not data:
                break
            newline = data.find(b"\n")
            if newline != -1:
                line_end = newline + 1
                leftover = data[line_end:]
                if leftover:
                    self._pos -= len(leftover)
                    self._pending = leftover + self._pending
                chunks.append(data[:line_end])
                return b"".join(chunks)
            chunks.append(data)
            total += len(data)
        return b"".join(chunks)

    def _read_source(self, size: int | None, position: int | None = None) -> bytes:
        """Seek the shared source to this view's cursor and read raw bytes."""
        absolute_pos = self._start + (self._pos if position is None else position)
        with _SOURCE_LOCK:
            self._source.seek(absolute_pos)
            if size is None:
                data = self._source.read()
                if not isinstance(data, (bytes, bytearray)):
                    msg = "slice sources must return bytes"
                    raise TypeError(msg)
                data = bytes(data)
                self._note_eof(absolute_pos + len(data))
                return data

            parts: list[bytes] = []
            remaining = size
            read_pos = absolute_pos
            while remaining:
                data = self._source.read(remaining)
                if not isinstance(data, (bytes, bytearray)):
                    msg = "slice sources must return bytes"
                    raise TypeError(msg)
                if not data:
                    self._note_eof(read_pos)
                    break
                data = bytes(data)
                if len(data) > remaining:
                    extra = data[remaining:]
                    self._pending = extra + self._pending
                    data = data[:remaining]
                parts.append(data)
                remaining -= len(data)
                read_pos += len(data)
            return b"".join(parts)

    def slice(self, start: int | builtins.slice | tuple[int, int], stop: int | None = None) -> SliceView:
        """Return another read-only view nested inside this one."""
        return SliceView(self, start, stop)

    #
    # Positioning.
    #
    def tell(self) -> int:
        """Return the current position, relative to the slice start."""
        self._check_open()
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Seek within the slice and return the new relative position.

        Offsets are clamped to ``[0, slice length]``, just like seeking in
        ordinary Python files.  ``whence`` is one of ``SEEK_SET``,
        ``SEEK_CUR`` or ``SEEK_END``.
        """
        self._check_open()
        if not isinstance(offset, int):
            msg = "offset must be an integer"
            raise TypeError(msg)
        if whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            msg = f"invalid whence, expected one of {(os.SEEK_SET, os.SEEK_CUR, os.SEEK_END)!r}"
            raise ValueError(msg)

        if whence == os.SEEK_SET:
            new_pos = offset
        elif whence == os.SEEK_CUR:
            new_pos = self._pos + offset
        else:
            self._infer_source_length()
            length = self._length()
            if length is None:
                msg = "cannot seek from the end of a source of unknown length"
                raise io.UnsupportedOperation(msg)
            new_pos = length + offset

        if self._stop is not None and self._source_length is None:
            self._infer_source_length()
        length = self._length()
        if length is None:
            new_pos = max(0, new_pos)
        else:
            new_pos = min(length, max(0, new_pos))

        delta = new_pos - self._pos
        if 0 <= delta <= len(self._pending):
            self._pending = self._pending[delta:]
        else:
            self._pending = b""
        self._pos = new_pos
        return self._pos


def open_slice(
    source: str | os.PathLike[str] | IO[Any] | SliceView,
    start: int | builtins.slice | tuple[int, int],
    stop: int | None = None,
    transport_params: TransportParams | None = None,
    closefd: bool | None = None,
    *,
    end: int | None = None,
) -> SliceView:
    """Open a read-only view of the half-open byte range ``[start, stop)``.

    Args:
        source: A URI/path, an open binary file-like object, or another
            :class:`SliceView`.  URIs are opened at the transport byte layer,
            bypassing compression and text decoding.
        start: First byte of the range.  A :class:`slice` object or
            ``(start, stop)`` tuple may be passed instead.
        stop: One past the last byte, or ``None`` for EOF.
        transport_params: Forwarded to the transport for URI sources.
        closefd: Whether the view owns the source.  Defaults to ``True`` for
            URIs and ``False`` for already-open objects.

    Returns:
        A :class:`SliceView` whose offsets start at zero.
    """
    return SliceView(
        source,
        start,
        stop,
        transport_params=transport_params,
        closefd=closefd,
        end=end,
    )


#
# Compatibility aliases for the names used by callers.
#
Slice = SliceView
SliceReader = SliceView
SlicedReader = SliceView
ReadOnlySlice = SliceView
ReadonlySlice = SliceView
open_range = open_slice
open_byte_range = open_slice
read_slice = open_slice
slice_view = open_slice

__all__ = [
    "ReadOnlySlice",
    "ReadonlySlice",
    "Slice",
    "SliceReader",
    "SlicedReader",
    "SliceView",
    "open_byte_range",
    "open_range",
    "open_slice",
    "read_slice",
    "slice_view",
]

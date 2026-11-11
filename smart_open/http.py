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
import re
import urllib.parse
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

try:
    import requests
except ImportError:
    MISSING_DEPS = True

import smart_open.utils
from smart_open import bytebuffer, constants

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer

    from smart_open._typing import TransportParams

DEFAULT_BUFFER_SIZE = 128 * 1024
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


class RedirectInfo(TypedDict):
    """One redirect offered to a :data:`RedirectPolicy`, before the next request is sent."""

    source: str
    """The URL that produced the redirect response."""

    target: str
    """The resolved absolute URL from the ``Location`` header."""

    status_code: int
    """The 3xx status code of the redirect response."""

    chain: tuple[str, ...]
    """The URLs requested so far, starting with the originally opened URL."""


RedirectPolicy = Callable[[RedirectInfo], bool]
"""A callback deciding whether a redirect may be followed.

It is called with the source URL, the resolved target URL, the redirect status
code and the current redirect chain. Returning ``True`` follows the redirect,
returning ``False`` raises :class:`RedirectRejected`. The callback runs before
the request for the target is sent, so a rejected target never receives a
request. Every GET the reader issues -- the initial one, seek range requests and
continuations of truncated ranges -- goes through the same policy.
"""

DEFAULT_MAX_REDIRECTS = 30
"""The default limit on followed redirects, matching ``requests``."""


class RedirectRejected(OSError):
    """Raised when a :data:`RedirectPolicy` refuses a redirect or the chain exceeds its limits."""


# requests strips these on cross-origin redirects, including the default Authorization
# that HTTPBasicAuth adds, so we apply the same rule ourselves before each hop
_SENSITIVE_HEADERS = frozenset({"authorization", "cookie", "cookie2", "netrc", "www-authenticate"})


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlsplit(url)
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname or "", port


def _headers_for_hop(headers: dict[str, str], source: str, target: str) -> dict[str, str]:
    """Drop credentials and sensitive headers when a redirect crosses an origin."""
    if _origin(source) == _origin(target):
        return headers
    return {name: value for name, value in headers.items() if name.lower() not in _SENSITIVE_HEADERS}


def _strip_sensitive_headers(prepared: requests.PreparedRequest) -> None:
    """Remove sensitive headers a Session may have merged in from its defaults or cookie jar."""
    for name in list(prepared.headers):
        if name.lower() in _SENSITIVE_HEADERS:
            del prepared.headers[name]


class _RedirectChain:
    """Follow 3xx responses manually, consulting a policy before every hop."""

    def __init__(
        self,
        session: Any,  # requests.Session or the requests module when no Session was given
        policy: RedirectPolicy | None,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
    ) -> None:
        self.session = session
        self.policy = policy
        self.max_redirects = max_redirects

    def _send(
        self,
        url: str,
        headers: dict[str, str],
        auth: object,
        cert: str | tuple[str, str] | None,
        timeout: float | None,
        strip_sensitive: bool = False,
    ) -> requests.Response:
        """Prepare and send one GET straight to the adapter, so requests never follows redirects itself."""
        if isinstance(self.session, requests.Session):
            request = self.session.prepare_request(requests.Request("GET", url, headers=headers, auth=auth))
            if strip_sensitive:
                _strip_sensitive_headers(request)
            adapter = self.session.get_adapter(url=url)
            response = adapter.send(
                request,
                stream=True,
                timeout=timeout,
                verify=self.session.verify,
                cert=cert,
                proxies=self.session.proxies,
            )
        else:  # the bare requests module keeps its own temporary session internally
            response = self.session.get(
                url,
                auth=auth,
                cert=cert,
                stream=True,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
            )
        return response

    def get(
        self,
        url: str,
        headers: dict[str, str],
        auth: object,
        cert: str | tuple[str, str] | None,
        timeout: float | None,
    ) -> requests.Response:
        """GET ``url`` and follow its 3xx responses, consulting the policy before every hop."""
        chain = [url]
        hop_headers, hop_auth = headers, auth
        response = self._send(url, hop_headers, hop_auth, cert, timeout)
        while response.is_redirect:
            source = chain[-1]
            location = response.headers.get("location", "")
            if not location:
                break
            target = urllib.parse.urljoin(source, location)
            cross_origin = _origin(source) != _origin(target)
            if cross_origin:
                hop_headers = _headers_for_hop(hop_headers, source, target)
                hop_auth = None
            if self.policy is not None:
                try:
                    allowed = self.policy(
                        RedirectInfo(
                            source=source, target=target, status_code=response.status_code, chain=tuple(chain)
                        )
                    )
                except BaseException:
                    response.close()
                    raise
                if not allowed:
                    response.close()
                    msg = f"redirect policy refused {response.status_code} redirect from {source} to {target}"
                    raise RedirectRejected(msg)
            if target in chain:
                response.close()
                msg = f"redirect loop detected at {target} in chain {chain!r}"
                raise RedirectRejected(msg)
            if len(chain) > self.max_redirects:
                response.close()
                msg = f"too many redirects (max {self.max_redirects}) in chain {chain!r}"
                raise RedirectRejected(msg)
            response.close()
            response = self._send(target, hop_headers, hop_auth, cert, timeout, strip_sensitive=cross_origin)
            chain.append(target)
        return response


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
    redirect_policy: RedirectPolicy | None = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
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
            Can be used for OAuth2 clients.
        redirect_policy: A callback consulted before every redirect, on the initial
            GET and on every range request. It receives the source URL, the resolved
            target URL, the status code and the redirect chain so far, and returns
            ``True`` to follow or ``False`` to raise :class:`RedirectRejected`. When
            ``None``, redirects are followed automatically, up to
            :data:`DEFAULT_MAX_REDIRECTS`; relative ``Location`` headers are resolved
            and loops are rejected. On a cross-origin redirect, authentication, cookies
            and other sensitive headers are not sent to the target origin.
        buffer_size: The buffer size to use when performing I/O.

    Returns:
        A file-like object opened for reading.

    Raises:
        NotImplementedError: If ``mode`` is anything other than ``"rb"``.

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
            redirect_policy=redirect_policy,
            timeout=timeout,
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
        redirect_policy: RedirectPolicy | None = None,
        timeout: float | None = None,
    ) -> None:

        self.url = url
        self.cert = cert
        self.session = session or requests
        self.redirect_policy = redirect_policy
        self._redirects = _RedirectChain(self.session, redirect_policy)

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

        self.response = self._redirects.get(
            self.url,
            headers=self.headers,
            auth=self.auth,
            cert=self.cert,
            timeout=self.timeout,
        )

        if not self.response.ok:
            self.response.close()
            self.response.raise_for_status()

        self._read_buffer = bytebuffer.ByteBuffer(buffer_size)
        self._current_pos = 0
        self._failed = False  # a range request failed, so reads raise instead of reading as EOF

    #
    # Override some methods from io.IOBase.
    #
    def close(self) -> None:
        """Flush and close this stream."""
        logger.debug("close: called")
        if not self.closed:
            self._close_response()
            self._read_buffer = None

    def _close_response(self) -> None:
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
        redirect_policy: RedirectPolicy | None = None,
        timeout: float | None = None,
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
            redirect_policy,
            timeout,
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

        if self._current_pos == new_pos and self.response is not None:
            return self._current_pos

        # Check if we can satisfy the seek from buffer (forward seek within buffered data)
        if new_pos > self._current_pos and new_pos - self._current_pos <= len(buf):
            buf.read(new_pos - self._current_pos)
            self._current_pos = new_pos
            return self._current_pos

        logger.debug("http seeking from current_pos: %d to new_pos: %d", self._current_pos, new_pos)

        self._close_response()
        buf.empty()

        if new_pos != self.content_length:
            self._failed = True  # until the range request succeeds
            self.response = self._range_response(new_pos)
        self._failed = False

        self._current_pos = new_pos
        return self._current_pos

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
        if self.response is None and total is not None:  # only an unknown total ends with a 416
            msg = f"server answered 416 to bytes={end}- before the end at {total}"
            raise OSError(msg)
        self._failed = False
        return self.response is not None

    def _partial_request(self, start_pos: int | None = None) -> requests.Response:
        headers = {**self._precondition, **self.headers}
        if start_pos is not None:
            headers["range"] = smart_open.utils.make_range_string(start_pos)

        return self._redirects.get(
            self.url,
            headers=headers,
            auth=self.auth,
            cert=self.cert,
            timeout=self.timeout,
        )

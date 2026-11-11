#
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
import functools
import gzip
import io
import re
import threading
import time
import unittest

import pytest
import requests
import responses

import smart_open.constants
import smart_open.http
import smart_open.s3

BYTES = b"i tried so hard and got so far but in the end it doesn't even matter"
GZIPPED_BYTES = gzip.compress(BYTES)
URL = "http://localhost"
HTTPS_URL = "https://localhost"
HEADERS = {
    "Accept-Ranges": "bytes",
}


def request_callback(request, headers=HEADERS, data=BYTES):
    """Request callback."""
    headers = headers.copy()
    range_string = request.headers.get("range", "bytes=0-")

    start, end = range_string.replace("bytes=", "", 1).split("-", 1)
    start = int(start)
    end = int(end) if end else len(data)

    data = data[start:end]
    headers["Content-Length"] = str(len(data))

    return (206 if "range" in request.headers else 200, headers, data)


class HttpTest(unittest.TestCase):
    """Tests for Http."""

    @responses.activate
    def test_read_all(self):
        """Read all."""
        responses.add(responses.GET, URL, body=BYTES)
        reader = smart_open.http.SeekableBufferedInputBase(URL)
        read_bytes = reader.read()
        assert read_bytes == BYTES

    @responses.activate
    def test_seek_from_start(self):
        """Seek from start."""
        responses.add_callback(responses.GET, URL, callback=request_callback)
        reader = smart_open.http.SeekableBufferedInputBase(URL)

        reader.seek(10)
        assert reader.tell() == 10  # test uses inline magic value
        read_bytes = reader.read(size=10)
        assert reader.tell() == 20  # test uses inline magic value
        assert BYTES[10:20] == read_bytes

        reader.seek(20)
        read_bytes = reader.read(size=10)
        assert BYTES[20:30] == read_bytes

        reader.seek(0)
        read_bytes = reader.read(size=10)
        assert BYTES[:10] == read_bytes

    @responses.activate
    def test_seek_from_current(self):
        """Seek from current."""
        responses.add_callback(responses.GET, URL, callback=request_callback)
        reader = smart_open.http.SeekableBufferedInputBase(URL)

        reader.seek(10)
        read_bytes = reader.read(size=10)
        assert BYTES[10:20] == read_bytes

        assert reader.tell() == 20  # test uses inline magic value
        reader.seek(10, whence=smart_open.constants.WHENCE_CURRENT)
        assert reader.tell() == 30  # test uses inline magic value
        read_bytes = reader.read(size=10)
        assert reader.tell() == 40  # test uses inline magic value
        assert BYTES[30:40] == read_bytes

    @responses.activate
    def test_seek_forward_within_buffer(self):
        """Does forward seeking within buffered data avoid additional GET requests?"""
        responses.add_callback(responses.GET, URL, callback=request_callback)
        reader = smart_open.http.SeekableBufferedInputBase(URL, buffer_size=32)

        assert reader.read(5) == BYTES[:5]
        initial_calls = len(responses.calls)

        # Forward seek within buffer using WHENCE_CURRENT - no new GET request
        reader.seek(1, whence=smart_open.constants.WHENCE_CURRENT)
        assert reader.tell() == 6  # test uses inline magic value
        assert reader.read(5) == BYTES[6:11]

        # Forward seek within buffer using WHENCE_START - no new GET request
        reader.seek(13, whence=smart_open.constants.WHENCE_START)
        assert reader.tell() == 13  # test uses inline magic value
        assert reader.read(3) == BYTES[13:16]

        assert len(responses.calls) == initial_calls

    @responses.activate
    def test_seek_from_end(self):
        """Seek from end."""
        responses.add_callback(responses.GET, URL, callback=request_callback)
        reader = smart_open.http.SeekableBufferedInputBase(URL)

        reader.seek(-10, whence=smart_open.constants.WHENCE_END)
        assert reader.tell() == len(BYTES) - 10
        read_bytes = reader.read(size=10)
        assert reader.tell() == len(BYTES)
        assert BYTES[-10:] == read_bytes

    @responses.activate
    def test_headers_are_as_assigned(self):
        """Headers are as assigned."""
        responses.add_callback(responses.GET, URL, callback=request_callback)

        # use default _HEADERS
        x = smart_open.http.BufferedInputBase(URL)
        # set different ones
        x.headers["Accept-Encoding"] = "compress, gzip"
        x.headers["Other-Header"] = "value"

        # use default again, global shoudn't overwritten from x
        y = smart_open.http.BufferedInputBase(URL)
        # should be default headers
        assert y.headers == {"Accept-Encoding": "identity"}
        # should be assigned headers
        assert x.headers == {"Accept-Encoding": "compress, gzip", "Other-Header": "value"}

    @responses.activate
    def test_headers(self):
        """Does the top-level http.open function handle headers correctly?"""
        responses.add_callback(responses.GET, URL, callback=request_callback)
        reader = smart_open.http.open(URL, "rb", headers={"Foo": "bar"})
        assert reader.headers["Foo"] == "bar"

    @responses.activate
    def test_https_seek_start(self):
        """Did the seek start over HTTPS work?"""
        responses.add_callback(responses.GET, HTTPS_URL, callback=request_callback)

        with smart_open.open(HTTPS_URL, "rb") as fin:
            read_bytes_1 = fin.read(size=10)
            fin.seek(0)
            read_bytes_2 = fin.read(size=10)
            assert read_bytes_1 == read_bytes_2

    @responses.activate
    def test_https_seek_forward(self):
        """Did the seek forward over HTTPS work?"""
        responses.add_callback(responses.GET, HTTPS_URL, callback=request_callback)

        with smart_open.open(HTTPS_URL, "rb") as fin:
            fin.seek(10)
            read_bytes = fin.read(size=10)
            assert BYTES[10:20] == read_bytes

    @responses.activate
    def test_https_seek_reverse(self):
        """Did the seek in reverse over HTTPS work?"""
        responses.add_callback(responses.GET, HTTPS_URL, callback=request_callback)

        with smart_open.open(HTTPS_URL, "rb") as fin:
            read_bytes_1 = fin.read(size=10)
            fin.seek(-10, whence=smart_open.constants.WHENCE_CURRENT)
            read_bytes_2 = fin.read(size=10)
            assert read_bytes_1 == read_bytes_2

    @responses.activate
    def test_timeout_attribute(self):
        """Timeout attribute."""
        timeout = 1
        responses.add_callback(responses.GET, URL, callback=request_callback)
        reader = smart_open.open(URL, "rb", transport_params={"timeout": timeout})
        assert hasattr(reader, "timeout")
        assert reader.timeout == timeout

    @responses.activate
    def test_session_attribute(self):
        """Session attribute."""
        session = requests.Session()
        responses.add_callback(responses.GET, URL, callback=request_callback)
        reader = smart_open.open(URL, "rb", transport_params={"session": session})
        assert hasattr(reader, "session")
        assert reader.session == session
        assert reader.read() == BYTES


@responses.activate
def test_seek_implicitly_enabled():
    """Can we seek even if the server hasn't explicitly allowed it?"""
    numbytes = 10
    callback = functools.partial(request_callback, headers={})
    responses.add_callback(responses.GET, HTTPS_URL, callback=callback)
    with smart_open.open(HTTPS_URL, "rb") as fin:
        assert fin.seekable()
        first = fin.read(size=numbytes)
        fin.seek(-numbytes, whence=smart_open.constants.WHENCE_CURRENT)
        second = fin.read(size=numbytes)
        assert first == second


@responses.activate
def test_seek_implicitly_disabled():
    """Does seeking fail when the server has explicitly disabled it?"""
    callback = functools.partial(request_callback, headers={"Accept-Ranges": "none"})
    responses.add_callback(responses.GET, HTTPS_URL, callback=callback)
    with smart_open.open(HTTPS_URL, "rb") as fin:
        assert not fin.seekable()
        fin.read()
        with pytest.raises(OSError, match="stream is not seekable"):
            fin.seek(0)


@responses.activate
def test_gzip_encoding_default_headers():
    """Does Accept-Encoding: identity prevent gzip compression?"""

    def callback(request):  # noqa: ARG001  # unused fn arg in fixture
        # Server respects Accept-Encoding: identity and sends uncompressed
        headers = HEADERS.copy()
        headers["Content-Length"] = str(len(BYTES))
        return (200, headers, BYTES)

    responses.add_callback(responses.GET, URL, callback=callback)
    reader = smart_open.http.SeekableBufferedInputBase(URL)
    read_bytes = reader.read()
    assert read_bytes == BYTES


@responses.activate
def test_gzip_encoding_explicit_request():
    """Does Accept-Encoding: gzip properly decompress via response.raw?"""

    def callback(request):
        # Server sees gzip in Accept-Encoding and returns compressed data
        if "gzip" in request.headers.get("Accept-Encoding", ""):
            headers = HEADERS.copy()
            headers["Content-Encoding"] = "gzip"
            headers["Content-Length"] = str(len(GZIPPED_BYTES))
            return (200, headers, GZIPPED_BYTES)
        headers = HEADERS.copy()
        headers["Content-Length"] = str(len(BYTES))
        return (200, headers, BYTES)

    responses.add_callback(responses.GET, URL, callback=callback)
    # Explicitly request gzip encoding
    reader = smart_open.http.SeekableBufferedInputBase(URL, headers={"Accept-Encoding": "gzip"})
    read_bytes = reader.read()
    assert read_bytes == BYTES  # Should be decompressed by requests/urllib3

    # Combining multiple read calls also works
    reader.seek(0)
    partial = reader.read(2) + reader.read(1000)
    assert partial == BYTES


@responses.activate
def test_read_after_read_to_eof():
    """Reading after reading to EOF should return empty bytes."""
    responses.add_callback(responses.GET, URL, callback=request_callback)
    reader = smart_open.http.SeekableBufferedInputBase(URL)

    # Read to EOF
    result = reader.read(-1)
    assert len(result) == len(BYTES)
    assert reader.tell() == len(BYTES)

    # Read should return empty bytes
    result = reader.read()
    assert result == b""

    # Read with size should also return empty bytes
    result = reader.read(10)
    assert result == b""


@responses.activate
def test_read_after_seek_to_eof():
    """Reading after seeking to EOF should return empty bytes."""
    responses.add_callback(responses.GET, URL, callback=request_callback)
    reader = smart_open.http.SeekableBufferedInputBase(URL)

    # Seek to EOF
    reader.seek(0, whence=smart_open.constants.WHENCE_END)
    assert reader.tell() == len(BYTES)

    # Read should return empty bytes
    result = reader.read()
    assert result == b""


@responses.activate
def test_readline_after_seek_to_eof():
    """Readline after seeking to EOF should return empty bytes, like read()."""
    responses.add_callback(responses.GET, URL, callback=request_callback)
    reader = smart_open.http.SeekableBufferedInputBase(URL)
    reader.seek(0, whence=smart_open.constants.WHENCE_END)
    assert reader.readline() == b""
    assert reader.readline(10) == b""

    # Read with size should also return empty bytes
    result = reader.read(10)
    assert result == b""


@responses.activate
def test_read_with_invalid_size():
    """Read with size < -1 should raise ValueError."""
    responses.add_callback(responses.GET, URL, callback=request_callback)
    reader = smart_open.http.SeekableBufferedInputBase(URL)

    with pytest.raises(ValueError, match="size must be >= -1"):
        reader.read(-2)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/a/b.gz", "b.gz"),
        ("/foo/", ""),  # trailing slash -> empty, not "foo" (regression: PurePosixPath.name diverges)
        ("/", ""),
        ("", ""),
    ],
)
@responses.activate
def test_open_sets_fobj_name(path, expected):
    """`http.open` derives `fobj.name` from the URL path, posix-basename style."""
    url = HTTPS_URL + path
    responses.add_callback(responses.GET, url, callback=request_callback)
    fobj = smart_open.http.open(url, "rb")
    assert fobj.name == expected


@pytest.mark.parametrize(
    "reader_class", [smart_open.http.BufferedInputBase, smart_open.http.SeekableBufferedInputBase]
)
@responses.activate
def test_close_releases_response(reader_class):
    """Release a partially read response when closing its reader."""
    responses.add(responses.GET, URL, body=BYTES)
    reader = reader_class(URL, buffer_size=1)
    response = reader.response
    assert reader.read(1) == BYTES[:1]
    assert not response.raw.closed
    reader.close()
    assert response.raw.closed
    assert reader.closed
    reader.close()


@responses.activate
def test_failed_open_releases_response():
    """Raise on and release a response that cannot be used for reading."""
    responses.add(responses.GET, URL, status=404, body=b"not found")
    with pytest.raises(requests.HTTPError, match="404"):
        smart_open.http.SeekableBufferedInputBase(URL)
    assert responses.calls[-1].response.raw.closed


@pytest.mark.parametrize("offset", [10, len(BYTES)])
@responses.activate
def test_seek_releases_response(offset):
    """Release the previous response before a range request or a seek to EOF."""
    responses.add_callback(responses.GET, URL, callback=request_callback)
    reader = smart_open.http.SeekableBufferedInputBase(URL, buffer_size=1)
    response = reader.response
    assert reader.read(1) == BYTES[:1]
    assert not response.raw.closed
    reader.seek(offset)
    assert response.raw.closed
    assert reader.read(1) == BYTES[offset : offset + 1]
    reader.close()


@responses.activate
def test_buffered_seek_keeps_response_open():
    """A seek within buffered data keeps the current response usable."""
    responses.add_callback(responses.GET, URL, callback=request_callback)
    with smart_open.http.SeekableBufferedInputBase(URL, buffer_size=32) as reader:
        response = reader.response
        assert reader.read(1) == BYTES[:1]
        reader.seek(1)
        reader.seek(5)
        assert not response.raw.closed
        assert reader.read() == BYTES[5:]
        assert len(responses.calls) == 1


@responses.activate
def test_seek_releases_failed_response():
    """Raise on and release a range response that cannot be used for reading."""
    responses.add(responses.GET, URL, body=BYTES)
    responses.add(responses.GET, URL, status=503, body=b"unavailable")
    responses.add(responses.GET, URL, status=416, body=b"not satisfiable")
    with smart_open.http.SeekableBufferedInputBase(URL, buffer_size=1) as reader:
        with pytest.raises(requests.HTTPError, match="503"):
            reader.seek(10)
        assert responses.calls[-1].response.raw.closed
        assert reader.tell() == 0
        assert reader.seek(10) == 10  # 416 means past EOF of a stream of unknown length
        assert responses.calls[-1].response.raw.closed
        assert reader.read() == b""


@responses.activate
def test_failed_seek_keeps_position_and_drops_stale_data():
    """Keep the position and drop stale buffered data when a range request fails."""
    responses.add(responses.GET, URL, body=BYTES)
    responses.add(responses.GET, URL, body=requests.ConnectionError("range failed"))
    responses.add_callback(responses.GET, URL, callback=request_callback)
    with smart_open.http.SeekableBufferedInputBase(URL, buffer_size=8) as reader:
        assert reader.read(1) == BYTES[:1]
        with pytest.raises(requests.ConnectionError):
            reader.seek(20)
        assert reader.tell() == 1
        assert reader.seek(1) == 1
        assert reader.read(3) == BYTES[1:4]
        assert reader.seek(20) == 20
        assert reader.read(3) == BYTES[20:23]


@responses.activate
def test_seek_rejects_ignored_range():
    """Raise when a nonzero range request gets the full body, and still allow rewinding."""
    responses.add(responses.GET, URL, body=BYTES)
    with smart_open.http.SeekableBufferedInputBase(URL) as reader:
        with pytest.raises(io.UnsupportedOperation, match="ignored the range request"):
            reader.seek(10)
        assert responses.calls[-1].response.raw.closed
        assert reader.tell() == 0
        assert reader.seek(0) == 0
        assert reader.read() == BYTES


@responses.activate
def test_seek_from_end_needs_length():
    """Refuse to seek from the end of a response of unknown length."""
    responses.add(responses.GET, URL, body=BYTES)
    with (
        smart_open.http.SeekableBufferedInputBase(URL) as reader,
        pytest.raises(io.UnsupportedOperation, match="unknown length"),
    ):
        reader.seek(-3, smart_open.constants.WHENCE_END)


@pytest.mark.parametrize(
    "content_range",
    [
        f"bytes 8-{len(BYTES) - 1}/{len(BYTES)}",
        f"bytes 10-9/{len(BYTES)}",
        f"bytes 10-{len(BYTES)}/{len(BYTES)}",
        "invalid",
        f"bytes */{len(BYTES)}",
        f"bytes 10-{len(BYTES) - 1}/invalid",
    ],
    ids=["wrong_start", "end_before_start", "end_past_total", "malformed", "unsatisfied", "malformed_total"],
)
@responses.activate
def test_seek_rejects_other_range(content_range):
    """Raise on and release a partial response that starts elsewhere than the requested offset, or is invalid."""
    partial = (206, {"Content-Range": content_range}, BYTES[10:])
    responses.add_callback(
        responses.GET, URL, callback=lambda r: partial if "range" in r.headers else (200, {}, BYTES)
    )
    with (
        smart_open.http.SeekableBufferedInputBase(URL) as reader,
        pytest.raises(OSError, match=re.escape(content_range)),
    ):
        reader.seek(10)
    assert responses.calls[-1].response.raw.closed


@responses.activate
def test_malformed_content_length_reads_as_unknown():
    """Read a response with a malformed Content-Length like one of unknown length."""
    responses.add(responses.GET, URL, body=BYTES, headers={"Content-Length": "invalid"})
    with smart_open.http.SeekableBufferedInputBase(URL) as reader:
        assert reader.content_length == -1
        assert reader.read() == BYTES


@pytest.mark.parametrize(
    ("read", "total", "at_end", "calls"),
    [
        (lambda reader: reader.read(), len(BYTES), None, 8),
        (lambda reader: reader.read(len(BYTES)), len(BYTES), None, 8),
        (lambda reader: b"".join(iter(reader.readline, b"")), len(BYTES), None, 8),
        (lambda reader: reader.read(), "*", (416, {}, b""), 9),
        (
            lambda reader: reader.read(),
            "*",
            (200, {"Content-Length": str(len(BYTES))}, BYTES),
            9,
        ),  # Range ignored
    ],
    ids=["read", "read_size", "readline", "unknown_total_416", "unknown_total_200"],
)
@responses.activate
def test_read_continues_after_capped_range(read, total, at_end, calls):
    """Request the rest from where a server cut a partial response short."""

    def callback(request):
        first = int(request.headers.get("range", "bytes=0-").removeprefix("bytes=").partition("-")[0])
        if not first:  # also a rewind, answered without Content-Range
            return 200, {"ETag": '"v1"'}, BYTES
        assert request.headers["If-Match"] == '"v1"'  # every range pins the first version
        if first >= len(BYTES):
            return at_end
        last = min(first + 10, len(BYTES)) - 1  # the server caps every range at 10 bytes
        return 206, {"Content-Range": f"bytes {first}-{last}/{total}"}, BYTES[first : last + 1]

    responses.add_callback(responses.GET, URL, callback=callback)
    with smart_open.http.SeekableBufferedInputBase(URL) as reader:
        reader.seek(5)
        assert read(reader) == BYTES[5:]
        assert len(responses.calls) == calls  # no request past a known total
        reader.seek(0)
        assert reader.read() == BYTES


@pytest.mark.parametrize(
    ("version", "precondition"),
    [
        ({"ETag": '"v1"'}, {"If-Match": '"v1"'}),
        (
            {"ETag": 'W/"v1"', "Last-Modified": "Tue, 27 Oct 2026 10:00:00 GMT"},
            {"If-Unmodified-Since": "Tue, 27 Oct 2026 10:00:00 GMT"},
        ),
    ],
)
@responses.activate
def test_seek_requires_same_version(version, precondition):
    """Fail a seek with 412 when the resource changed since the first response."""

    def callback(request):
        if "range" not in request.headers:
            return 200, version, BYTES
        name, stale = next(iter(precondition.items()))
        if request.headers.get(name) == stale:  # the resource was replaced in between
            return 412, {}, b""
        return request_callback(request)

    responses.add_callback(responses.GET, URL, callback=callback)
    with (
        smart_open.http.SeekableBufferedInputBase(URL) as reader,
        pytest.raises(requests.HTTPError, match=r"412.*If-Match: \*.*Accept-Encoding: identity"),
    ):
        reader.seek(10)


@pytest.mark.parametrize(
    ("body", "match"),
    [(BYTES[5:8], "has 3 bytes"), (BYTES[5:], "has 63 bytes"), (BYTES[5:15], "416")],
    ids=["short", "long", "416_before_total"],
)
@responses.activate
def test_read_rejects_inconsistent_partial_response(body, match):
    """Raise instead of skipping, repeating or truncating bytes when a capped response does not add up."""

    def callback(request):
        if "range" not in request.headers:
            return request_callback(request)
        if request.headers["range"] == "bytes=5-":  # declares 10 bytes, no Content-Length to catch it
            return 206, {"Content-Range": f"bytes 5-14/{len(BYTES)}"}, body
        return 416, {}, b""  # the rest is gone before the declared total

    responses.add_callback(responses.GET, URL, callback=callback)
    with smart_open.http.SeekableBufferedInputBase(URL) as reader:
        reader.seek(5)
        with pytest.raises(OSError, match=match):
            reader.read()


@pytest.mark.parametrize("read", [lambda r: r.read(), lambda r: r.readline()], ids=["read", "readline"])
@responses.activate
def test_read_after_failed_seek_raises(read):
    """Raise on reads after a failed range request instead of reading as EOF, until a seek succeeds."""
    responses.add(responses.GET, URL, body=BYTES)
    responses.add(responses.GET, URL, status=503, body=b"unavailable")
    responses.add_callback(responses.GET, URL, callback=request_callback)
    with smart_open.http.SeekableBufferedInputBase(URL) as reader:
        with pytest.raises(requests.HTTPError, match="503"):
            reader.seek(10)
        with pytest.raises(OSError, match="seek to retry"):
            read(reader)
        assert reader.seek(reader.tell()) == 0
        assert read(reader) == BYTES


@responses.activate
def test_read_after_failed_continuation_raises():
    """Keep raising after a failed request for the rest of a capped range, until a seek succeeds."""
    failures = [503]  # the first request for the rest fails, a retry succeeds

    def callback(request):
        first = int(request.headers.get("range", "bytes=0-").removeprefix("bytes=").partition("-")[0])
        if not first:
            return 200, {}, BYTES
        if first == 15 and failures:
            return failures.pop(), {}, b"unavailable"
        last = min(first + 10, len(BYTES)) - 1  # the server caps every range at 10 bytes
        return 206, {"Content-Range": f"bytes {first}-{last}/{len(BYTES)}"}, BYTES[first : last + 1]

    responses.add_callback(responses.GET, URL, callback=callback)
    with smart_open.http.SeekableBufferedInputBase(URL) as reader:
        reader.seek(5)
        with pytest.raises(requests.HTTPError, match="503"):
            reader.read(len(BYTES))
        with pytest.raises(OSError, match="seek to retry"):
            reader.read(len(BYTES))
        assert reader.seek(reader.tell()) == 5
        assert reader.read() == BYTES[5:]


class BlockingRaw:
    """Raw stream whose read() blocks on a gate while recording downloaded bytes."""

    def __init__(self, data, gate=None, error=None):
        self._data = data
        self._pos = 0
        self._gate = gate
        self._error = error
        self.closed = False
        self.reads = []  # requested sizes, in call order
        self.downloaded = 0
        self.read_threads = []

    def read(self, size=-1):
        self.read_threads.append(threading.get_ident())
        self.reads.append(size)  # register the in-flight read before blocking on the gate
        if self._gate is not None:
            self._gate.wait()
        if self._error is not None:
            raise self._error
        if size is None or size < 0:
            chunk = self._data[self._pos :]
        else:
            chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        self.downloaded += len(chunk)
        return chunk

    def tell(self):
        return self._pos

    def close(self):
        self.closed = True
        if self._gate is not None:
            self._gate.set()  # a closed response unblocks a read in flight, like urllib3


class FakeSession:
    """Minimal requests.Session stand-in handing out crafted responses."""

    def __init__(self, factory):
        self._factory = factory
        self.closed = False
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return self._factory(url, kwargs)

    def close(self):
        self.closed = True


def _blocking_response(data, gate=None, error=None, headers=None):
    """Wrap a BlockingRaw in a requests.Response without touching the network."""
    response = requests.Response()
    response.status_code = requests.codes.ok
    response.headers = requests.structures.CaseInsensitiveDict(headers or HEADERS)
    response.raw = BlockingRaw(data, gate, error)
    return response


def _wait_until(predicate, timeout=5.0, interval=0.01):
    """Poll predicate until it holds or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for condition")
        time.sleep(interval)


class ReadaheadTest(unittest.TestCase):
    """Explicit readahead mode: worker, byte budget, shared cursor, seek and close."""

    URL = "http://localhost/readahead"

    def _reader(self, data, block_size=8, readahead_size=24, factory=None, session=None):
        session = session or FakeSession(factory or (lambda url, kwargs: _blocking_response(data)))
        reader = smart_open.http.SeekableBufferedInputBase(
            self.URL,
            buffer_size=block_size,
            readahead=True,
            readahead_size=readahead_size,
            session=session,
        )
        return reader, session

    def test_read_readinto_readline_share_cursor(self):
        """read, readinto and readline advance one logical position."""
        data = b"first line\nsecond line\nthird line\n"
        reader, _ = self._reader(data, block_size=8, readahead_size=24)

        assert reader.read(6) == b"first "
        assert reader.tell() == 6

        target = bytearray(5)
        assert reader.readinto(target) == 5
        assert bytes(target) == b"line\n"
        assert reader.tell() == 11

        assert reader.readline() == b"second line\n"
        assert reader.tell() == 23

        assert reader.read() == b"third line\n"
        assert reader.tell() == len(data)
        assert reader.read() == b""
        assert reader.readline() == b""
        reader.close()

    def test_readline_limit_and_block_boundary(self):
        """readline(size) stops at the limit, keeping the rest for the next call."""
        data = b"abcdefgh\nij\n"
        reader, _ = self._reader(data, block_size=4, readahead_size=12)

        assert reader.readline(3) == b"abc"
        assert reader.tell() == 3
        assert reader.readline() == b"defgh\n"  # the line continues across blocks
        assert reader.tell() == 9
        assert reader.readline() == b"ij\n"
        assert reader.readline() == b""
        reader.close()

    def test_read_all_and_eof(self):
        """read(-1) consumes everything, then reads return empty bytes."""
        data = b"x" * 30
        reader, _ = self._reader(data, block_size=8, readahead_size=16)
        assert reader.read() == data
        assert reader.tell() == 30
        assert reader.read() == b""
        assert reader.read(5) == b""
        reader.close()

    def test_read_zero_size(self):
        """read(0) returns empty bytes without touching the worker."""
        reader, _ = self._reader(b"data", block_size=4, readahead_size=8)
        assert reader.read(0) == b""
        assert reader.tell() == 0
        reader.close()

    def test_budget_caps_buffered_and_inflight_bytes(self):
        """The worker stops downloading once queued plus in-flight blocks hit the budget."""
        gate = threading.Event()
        data = b"0123456789abcdef" * 4  # 64 bytes
        response = _blocking_response(data, gate)
        reader, _ = self._reader(data, block_size=8, readahead_size=24, factory=lambda url, kwargs: response)

        # The first block is blocked inside raw.read, charging the full block size
        _wait_until(lambda: response.raw.reads)
        gate.set()
        _wait_until(lambda: response.raw.downloaded == 24)  # three blocks fill the budget
        time.sleep(0.05)  # prove the worker is stalled, not merely scheduled later
        assert response.raw.downloaded == 24
        assert reader._readahead.in_flight == 24

        # Consuming one block frees its budget and lets exactly one more block download
        assert reader.read(8) == b"01234567"
        _wait_until(lambda: response.raw.downloaded == 32)
        time.sleep(0.05)
        assert response.raw.downloaded == 32

        assert reader.read() == data[8:]
        _wait_until(lambda: response.raw.downloaded == 64)
        reader.close()

    def test_paused_consumer_pauses_download(self):
        """When the caller stops reading, no bytes download past the budget."""
        gate = threading.Event()
        data = b"z" * 40
        response = _blocking_response(data, gate)
        reader, _ = self._reader(data, block_size=8, readahead_size=16, factory=lambda url, kwargs: response)
        gate.set()

        _wait_until(lambda: response.raw.downloaded == 16)  # two blocks at most
        time.sleep(0.1)
        assert response.raw.downloaded == 16
        assert reader.tell() == 0  # the consumer paused before reading anything
        reader.close()
        assert response.raw.closed

    def test_invalid_readahead_size_rejected(self):
        """A readahead budget smaller than the block size cannot schedule any block."""
        with pytest.raises(ValueError, match="readahead_size must be >= buffer_size"):
            smart_open.http.SeekableBufferedInputBase(
                self.URL,
                buffer_size=16,
                readahead=True,
                readahead_size=8,
                session=FakeSession(lambda u, k: None),
            )

    def test_seek_drops_late_blocks_from_old_position(self):
        """Blocks downloaded for the old position never leak into reads after a seek."""
        gate = threading.Event()
        data = b"abcdefghijklmnopqrstuvwxyz" * 2  # 52 bytes
        first = _blocking_response(data, gate)
        second = _blocking_response(data)

        def factory(url, kwargs):
            if "range" in kwargs["headers"]:
                start = int(kwargs["headers"]["range"].removeprefix("bytes=").rstrip("-"))
                return _blocking_response(data[start:])
            return first

        reader, _ = self._reader(data, block_size=8, readahead_size=24, factory=factory)
        gate.set()
        _wait_until(lambda: first.raw.downloaded == 24)  # old position buffered ahead

        assert reader.read(4) == b"abcd"
        assert reader.tell() == 4
        reader.seek(20)
        assert reader.tell() == 20
        assert reader.read(8) == data[20:28]  # exactly the new position, no stale bytes
        assert reader.tell() == 28
        reader.close()
        assert first.raw.closed

    def test_seek_backward_restarts_stream(self):
        """A backward seek issues a range request and re-reads from there."""
        data = b"0123456789" * 5  # 50 bytes

        def factory(url, kwargs):
            if "range" in kwargs["headers"]:
                start = int(kwargs["headers"]["range"].removeprefix("bytes=").rstrip("-"))
                return _blocking_response(data[start:])
            return _blocking_response(data)

        reader, _ = self._reader(data, block_size=8, readahead_size=16, factory=factory)
        assert reader.read(10) == data[:10]
        reader.seek(0)
        assert reader.tell() == 0
        assert reader.read(10) == data[:10]
        reader.close()

    def test_close_stops_worker_and_releases_response(self):
        """close() stops new downloads, closes the response and joins the worker thread."""
        gate = threading.Event()
        data = b"q" * 64
        response = _blocking_response(data, gate)
        reader, _ = self._reader(data, block_size=8, readahead_size=16, factory=lambda url, kwargs: response)
        worker = reader._readahead._thread

        _wait_until(lambda: response.raw.reads)  # the worker is blocked inside raw.read
        reader.close()
        assert worker.ident is not None
        worker.join(timeout=2.0)
        assert not worker.is_alive()  # the blocked read returned and the thread exited
        assert response.raw.closed
        assert reader.closed
        reader.close()  # closing twice is a no-op

    def test_close_keeps_caller_session_open(self):
        """The caller-owned session is never closed by the reader."""
        session = FakeSession(lambda url, kwargs: _blocking_response(b"data"))
        reader, _ = self._reader(b"data", session=session)
        assert reader.read() == b"data"
        reader.close()
        assert not session.closed

    def test_error_delivered_on_consumption(self):
        """A failing read surfaces the original error to the consumer, not the worker."""
        error = OSError("connection reset by peer")
        response = _blocking_response(b"abcdefgh" * 4, error=error)
        reader, _ = self._reader(b"", factory=lambda url, kwargs: response)

        with pytest.raises(OSError, match="connection reset by peer") as excinfo:
            reader.read(4)
        assert excinfo.value is error  # the original exception object is re-raised
        reader.close()

    def test_continuation_error_recovered_by_seek(self):
        """A failed capped-range continuation is delivered, then a seek retries successfully."""
        failures = [503]

        def factory(url, kwargs):
            first = int(kwargs["headers"].get("range", "bytes=0-").removeprefix("bytes=").partition("-")[0])
            if first == 15 and failures:
                failures.pop()
                response = _blocking_response(b"")
                response.status_code = 503
                response.headers = requests.structures.CaseInsensitiveDict(HEADERS)
                return response
            last = min(first + 10, len(BYTES))
            payload = BYTES[first:last]
            response = _blocking_response(payload)
            response.status_code = requests.codes.partial_content
            response.headers = requests.structures.CaseInsensitiveDict(
                {**HEADERS, "Content-Range": f"bytes {first}-{last - 1}/{len(BYTES)}"}
            )
            return response

        reader, _ = self._reader(BYTES, block_size=8, readahead_size=24, factory=factory)
        reader.seek(5)
        with pytest.raises(requests.HTTPError, match="503"):
            reader.read(len(BYTES))
        with pytest.raises(OSError, match="seek to retry"):
            reader.read(10)
        with pytest.raises(OSError, match="seek to retry"):
            reader.readline()
        assert reader.seek(5) == 5
        assert reader.read() == BYTES[5:]
        reader.close()

    def test_worker_runs_in_background_and_exits_on_close(self):
        """The worker thread downloads ahead and is gone after close."""
        data = b"m" * 32
        response = _blocking_response(data)
        reader, _ = self._reader(data, block_size=8, readahead_size=24, factory=lambda url, kwargs: response)
        worker = reader._readahead._thread
        assert worker.is_alive()
        assert worker.name == "smart_open-http-readahead"
        _wait_until(lambda: response.raw.read_threads)  # the worker started downloading
        assert set(response.raw.read_threads) == {worker.ident}
        reader.close()
        worker.join(timeout=2.0)
        assert not worker.is_alive()

    def test_forward_seek_within_buffered_blocks_skips_request(self):
        """A short forward seek consumes buffered blocks without a new GET request."""
        data = b"a" * 40
        session = FakeSession(lambda url, kwargs: _blocking_response(data))
        reader, session = self._reader(data, block_size=8, readahead_size=24, session=session)
        _wait_until(lambda: reader._readahead.in_flight == 24)  # three blocks are buffered
        initial_calls = session.calls

        reader.seek(10, whence=smart_open.constants.WHENCE_CURRENT)
        assert reader.tell() == 10
        assert reader.read(4) == b"aaaa"
        assert session.calls == initial_calls  # no new GET request for a buffered seek
        reader.close()

    def test_forward_seek_past_buffer_falls_back_to_range(self):
        """A forward seek beyond buffered blocks issues a range request."""
        data = b"a" * 40

        def factory(url, kwargs):
            start = int(kwargs["headers"].get("range", "bytes=0-").removeprefix("bytes=").rstrip("-"))
            response = _blocking_response(data[start:])
            response.status_code = requests.codes.partial_content
            response.headers = requests.structures.CaseInsensitiveDict(
                {**HEADERS, "Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"}
            )
            return response

        session = FakeSession(factory)
        reader, session = self._reader(data, block_size=8, readahead_size=16, session=session)
        _wait_until(lambda: reader._readahead.in_flight == 16)  # only two blocks fit the budget
        calls_before = session.calls
        reader.seek(30)  # beyond the 16 buffered bytes
        assert session.calls > calls_before
        assert reader.tell() == 30
        assert reader.read(4) == data[30:34]
        reader.close()

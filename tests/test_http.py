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


class RecordingAdapter(requests.adapters.BaseAdapter):
    """A fake adapter that answers GETs from a routing table and records every request it receives."""

    def __init__(self, routes):
        self.routes = routes  # url -> (status, headers, body)
        self.requests = []  # the PreparedRequests this adapter received
        self.responses = []  # the Responses it handed out, to check they get closed

    def send(self, request, **kwargs):  # noqa: ARG002  # requests passes stream/timeout/etc.
        self.requests.append(request)
        route = self.routes[request.url]
        status, headers, body = route(request) if callable(route) else route
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response.url = request.url
        response.request = request
        response.raw = io.BytesIO(body)
        self.responses.append(response)
        return response

    def close(self):
        pass


def redirect_chain_session(routes):
    """A Session whose RecordingAdapter serves the given routes, mounted for both schemes."""
    session = requests.Session()
    adapter = RecordingAdapter(routes)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session, adapter


def test_redirect_policy_follows_three_hop_chain():
    """Consult the policy before every hop of a chain, then serve the final body."""
    routes = {
        "http://example.com/start": (302, {"Location": "/next"}, b""),
        "http://example.com/next": (301, {"Location": "http://other.org/final"}, b""),
        "http://other.org/final": (200, {}, BYTES),
    }
    session, adapter = redirect_chain_session(routes)
    offers = []

    def policy(info):
        offers.append(info)
        return True

    with smart_open.http.open(
        "http://example.com/start", "rb", session=session, redirect_policy=policy
    ) as reader:
        assert reader.read() == BYTES
        assert all(response.raw.closed for response in adapter.responses[:-1])  # intermediates closed on the way
        assert not adapter.responses[-1].raw.closed  # the body stays open until the reader closes

    assert [request.url for request in adapter.requests] == list(routes)
    assert offers == [
        {
            "source": "http://example.com/start",
            "target": "http://example.com/next",
            "status_code": 302,
            "chain": ("http://example.com/start",),
        },
        {
            "source": "http://example.com/next",
            "target": "http://other.org/final",
            "status_code": 301,
            "chain": ("http://example.com/start", "http://example.com/next"),
        },
    ]
    assert all(response.raw.closed for response in adapter.responses)  # closing the reader closes the final body
    assert session.get("http://example.com/start", allow_redirects=False).status_code == 302  # session still usable


def test_redirect_policy_rejects_cross_origin():
    """Never send a request to a target the policy refuses, and close the redirect response."""
    routes = {
        "http://example.com/start": (302, {"Location": "http://evil.org/final"}, b""),
        "http://evil.org/final": (200, {}, BYTES),
    }
    session, adapter = redirect_chain_session(routes)

    def policy(info):
        return info["target"].startswith("http://example.com/")

    with pytest.raises(smart_open.http.RedirectRejected, match="evil.org"):
        smart_open.http.open(
            "http://example.com/start", "rb", session=session, redirect_policy=policy
        )

    assert [request.url for request in adapter.requests] == ["http://example.com/start"]  # target never requested
    assert adapter.responses[0].raw.closed
    assert session.get("http://example.com/start", allow_redirects=False).status_code == 302  # session still usable


def test_redirect_policy_same_origin_different_port():
    """Treat a different port as a different origin, so a policy can allow it explicitly."""
    routes = {
        "http://example.com/start": (302, {"Location": "http://example.com:8080/final"}, b""),
        "http://example.com:8080/final": (200, {}, BYTES),
    }
    session, adapter = redirect_chain_session(routes)
    offers = []

    def policy(info):
        offers.append(info)
        return True

    with smart_open.http.open(
        "http://example.com/start", "rb", session=session, redirect_policy=policy
    ) as reader:
        assert reader.read() == BYTES

    assert offers[0]["target"] == "http://example.com:8080/final"
    assert [request.url for request in adapter.requests] == list(routes)


def test_redirect_strips_sensitive_headers_cross_origin():
    """Send auth, cookies and sensitive headers only to the origin they belong to."""
    routes = {
        "http://example.com/start": (302, {"Location": "http://other.org/final"}, b""),
        "http://other.org/final": (200, {}, BYTES),
    }
    session, adapter = redirect_chain_session(routes)
    session.headers["Cookie"] = "session=secret"
    headers = {
        "Accept-Encoding": "identity",
        "X-Api-Key": "secret",  # a plain custom header follows the request across origins
        "WWW-Authenticate": "Bearer token",  # a sensitive custom header is stripped
    }

    with smart_open.http.open(
        "http://example.com/start",
        "rb",
        session=session,
        user="user",
        password="pass",
        headers=headers,
        redirect_policy=lambda info: True,
    ) as reader:
        assert reader.read() == BYTES

    first, second = adapter.requests
    assert first.headers["Authorization"].startswith("Basic ")
    assert first.headers["Cookie"] == "session=secret"
    assert first.headers["WWW-Authenticate"] == "Bearer token"
    assert "Authorization" not in second.headers
    assert "Cookie" not in second.headers
    assert "WWW-Authenticate" not in second.headers
    assert second.headers["X-Api-Key"] == "secret"  # only the explicit sensitive set is stripped


def test_redirect_rejects_loop_and_missing_location():
    """Reject a redirect chain that loops, and pass a 3xx without Location through."""
    routes = {
        "http://example.com/a": (302, {"Location": "http://example.com/b"}, b""),
        "http://example.com/b": (302, {"Location": "http://example.com/a"}, b""),
    }
    session, adapter = redirect_chain_session(routes)
    with pytest.raises(smart_open.http.RedirectRejected, match="loop"):
        smart_open.http.open("http://example.com/a", "rb", session=session)
    assert len(adapter.requests) == 2  # the loop target is never requested again
    assert all(response.raw.closed for response in adapter.responses)

    routes = {"http://example.com/a": (302, {}, b"")}  # a redirect status without a Location header
    session, adapter = redirect_chain_session(routes)
    with smart_open.http.open("http://example.com/a", "rb", session=session) as reader:
        assert reader.read() == b""  # not followed, read as a plain response
    assert len(adapter.requests) == 1


def test_redirect_policy_applies_to_seek_and_continuation():
    """Run seek range requests and continuations of a truncated range through the same policy."""
    offers = []

    def final(request):
        if "range" not in request.headers:
            return 200, {}, BYTES
        start = int(request.headers["range"].removeprefix("bytes=").partition("-")[0])
        end = min(start + 10, len(BYTES))  # the server caps every range at 10 bytes
        return 206, {"Content-Range": f"bytes {start}-{end - 1}/{len(BYTES)}"}, BYTES[start:end]

    routes = {
        "http://example.com/start": (302, {"Location": "/final"}, b""),
        "http://example.com/final": final,
    }
    session, adapter = redirect_chain_session(routes)

    def policy(info):
        offers.append(info)
        return True

    with smart_open.http.open(
        "http://example.com/start", "rb", session=session, redirect_policy=policy
    ) as reader:
        reader.seek(5)
        assert reader.read() == BYTES[5:]

    # the initial GET, the seek and every truncated-range continuation redirect again
    num_gets = 1 + (len(BYTES) - 5 + 9) // 10
    hops = ["http://example.com/start", "http://example.com/final"]
    assert [request.url for request in adapter.requests] == hops * num_gets
    assert all(offer["target"] == "http://example.com/final" for offer in offers)
    assert len(offers) == len(adapter.requests) // 2  # one offer per GET, before each hop
    assert all(response.raw.closed for response in adapter.responses[:-1])


def test_redirect_does_not_mix_versions():
    """Fail a seek with 412 when the redirect target serves a new version, instead of mixing bytes."""

    def serve(request):
        if "range" not in request.headers:
            return 302, {"Location": "/final"}, b""
        assert request.headers["If-Match"] == '"v2"'  # the range pins the version first read
        return 412, {}, b""  # the target was replaced in between

    routes = {
        "http://example.com/start": serve,
        "http://example.com/final": lambda request: (200, {"ETag": '"v2"'}, BYTES),
    }
    session, _ = redirect_chain_session(routes)
    with (
        smart_open.http.open("http://example.com/start", "rb", session=session) as reader,
        pytest.raises(requests.HTTPError, match="412"),
    ):
        reader.seek(10)

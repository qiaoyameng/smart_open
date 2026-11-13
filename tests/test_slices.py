import gzip
import io
import os

import pytest
import requests

import smart_open
from smart_open import slices


class CountingBytesIO(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.read_sizes = []
        self.bytes_read = 0

    def read(self, size=-1):
        self.read_sizes.append(size)
        data = super().read(size)
        self.bytes_read += len(data)
        return data


DATA = b"0123456789abcdefghij\nklmnopqrstuvwxyz\n"


def test_reads_exact_interval_and_counts_bytes():
    source = CountingBytesIO(DATA)
    view = slices.open_slice(source, 5, 25)
    assert view.tell() == 0
    assert view.read(9) == DATA[5:14]
    assert view.tell() == 9
    assert view.read() == DATA[14:25]
    assert view.read() == b""
    assert view.readinto(bytearray(4)) == 0
    assert source.bytes_read == 20
    view.close()
    assert not source.closed


def test_empty_and_out_of_bounds_slices():
    source = CountingBytesIO(DATA)

    empty = slices.open_slice(source, 5, 5)
    assert empty.read() == b""
    assert empty.tell() == 0
    assert empty.seek(0, os.SEEK_END) == 0
    assert source.bytes_read == 0

    truncated = slices.open_slice(source, 20, len(DATA) + 100)
    assert truncated.read() == DATA[20:]

    past = slices.open_slice(source, len(DATA) + 10, len(DATA) + 20)
    assert past.read() == b""
    assert past.seek(0, os.SEEK_END) == 0
    assert source.bytes_read == len(DATA) - 20


def test_seek_tell_readinto_and_readline():
    view = slices.open_slice(io.BytesIO(DATA), 5, 25)
    assert view.readable()
    assert not view.writable()
    assert view.seekable()

    assert view.seek(6) == 6
    assert view.tell() == 6
    assert view.seek(-2, os.SEEK_CUR) == 4
    assert view.seek(-1, os.SEEK_END) == 19
    assert view.read(1) == DATA[24:25]
    assert view.seek(100) == 20

    view.seek(0)
    target = bytearray(6)
    assert view.readinto(target) == 6
    assert target == bytearray(DATA[5:11])

    view.seek(0)
    assert view.readline(4) == DATA[5:9]
    assert view.readline() == DATA[9:21]
    assert view.readline() == DATA[21:25]
    assert view.readline() == b""
    with pytest.raises(StopIteration):
        next(iter(view))

    view.close()
    with pytest.raises(ValueError):
        view.read()
    with pytest.raises(ValueError):
        view.tell()
    with pytest.raises(ValueError):
        view.seek(0)


def test_interleaved_views_keep_independent_cursors():
    source = CountingBytesIO(DATA)
    left = slices.open_slice(source, 0, 10)
    right = slices.open_slice(source, 10, 20)

    assert left.read(3) == DATA[0:3]
    assert right.read(3) == DATA[10:13]
    assert left.read(3) == DATA[3:6]
    assert right.read(3) == DATA[13:16]
    assert left.tell() == right.tell() == 6
    assert left.read() == DATA[6:10]
    assert right.read() == DATA[16:20]
    assert source.bytes_read == 20


def test_external_handle_ownership_and_parent_close():
    source = io.BytesIO(DATA)
    view = slices.open_slice(source, 0, 10)
    view.close()
    assert not source.closed
    assert source.tell() == 0

    borrowed = slices.open_slice(source, 0, 10)
    source.close()
    with pytest.raises(ValueError):
        borrowed.read()
    borrowed.close()
    assert source.closed


def test_slice_object_and_nested_view():
    view = smart_open.open_slice(io.BytesIO(DATA), slice(5, 10))
    assert view.read() == DATA[5:10]
    nested = smart_open.open_slice(view, 2, 4)
    assert nested.read() == DATA[7:9]
    nested.close()
    assert not view.closed
    view.close()


def test_local_file_uri_is_raw_and_owned(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(DATA)
    view = smart_open.open_slice(str(path), 3, 8)
    assert view.read() == DATA[3:8]
    view.close()

    view = smart_open.open_slice(path, (3, 8))
    assert view.read() == DATA[3:8]
    view.close()


def test_compression_is_only_applied_explicitly_by_caller():
    payload = b"hello compressed world\n" * 10

    compressed = gzip.compress(payload)
    # The gzip member is complete, so the caller can opt in to decompression.
    view = smart_open.open_slice(io.BytesIO(compressed), 0, len(compressed))
    with gzip.GzipFile(fileobj=view, mode="rb") as decompressed:
        assert decompressed.read() == payload
    view.close()

    # Slicing a compressed file at transport offsets does not decompress it.
    view = smart_open.open_slice(io.BytesIO(compressed), 10, 20)
    assert view.read() == compressed[10:20]
    view.close()


class FakeHTTPAdapter(requests.adapters.BaseAdapter):
    def __init__(self, data, *, cap=None, version="v1", changed=None):
        super().__init__()
        self.data = data
        self.cap = cap
        self.version = version
        self.changed = changed
        self.requests = []
        self.body_bytes_constructed = 0

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        del stream, timeout, verify, cert, proxies
        self.requests.append(request)
        headers = request.headers
        range_header = headers.get("Range")
        if range_header:
            start, _, end = range_header.removeprefix("bytes=").partition("-")
            start = int(start)
            end = int(end) + 1 if end else len(self.data)
            if self.changed is not None and headers.get("If-Match") not in (None, '"v1"'):
                body = self.changed[start:end]
            else:
                body = self.data[start:end]
            if self.cap is not None:
                body = body[: self.cap]
            response = requests.Response()
            response.status_code = 206
            response.headers["Content-Range"] = f"bytes {start}-{start + len(body) - 1}/{len(self.data)}"
        else:
            body = self.data
            response = requests.Response()
            response.status_code = 200
        if range_header:
            self.body_bytes_constructed += len(body)
        response.headers["Content-Length"] = str(len(body))
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["ETag"] = f'"{self.version}"'
        response._content = body
        response.raw = io.BytesIO(body)
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass


def _http_session(adapter):
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def test_http_small_slice_does_not_fetch_the_whole_body():
    url = "http://example.test/data"
    data = b"x" * 1_000_000
    adapter = FakeHTTPAdapter(data)
    session = _http_session(adapter)
    try:
        view = slices.open_slice(url, 100, 110, transport_params={"session": session})
        assert view.read() == data[100:110]
        assert view.tell() == 10
        view.close()
    finally:
        session.close()

    range_requests = [request for request in adapter.requests if "Range" in request.headers]
    assert range_requests
    assert all(request.headers["Range"].startswith("bytes=100-") for request in range_requests)
    # Streamed responses only expose the bytes the view asks for.
    assert adapter.body_bytes_constructed == 10


def test_http_versions_are_not_spliced_within_one_view():
    url = "http://example.test/versioned"
    v1 = b"11111111111111111111"
    v2 = b"22222222222222222222"
    adapter = FakeHTTPAdapter(v1, cap=4, changed=v2)
    session = _http_session(adapter)
    try:
        view = slices.open_slice(url, 0, 12, transport_params={"session": session})
        assert view.read() == v1[:12]
        view.close()
    finally:
        session.close()

    range_requests = [request for request in adapter.requests if "Range" in request.headers]
    assert range_requests
    # The first request establishes the version; every continuation pins it.
    assert all(request.headers.get("If-Match") == '"v1"' for request in range_requests[1:])

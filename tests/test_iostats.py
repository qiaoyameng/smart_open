#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Per-handle I/O accounting: hand-checked counters across the read stack."""
import dataclasses
import gzip
import io

import pytest
import responses

import smart_open
from smart_open.iostats import IOStatsRecorder, IOStatsSnapshot, MeteredReader

PLAIN = b"alpha\nbeta\ngamma\ndelta\n"
GZIPPED = gzip.compress(PLAIN)
TEXT = "alpha\nмир\n"
TEXT_BYTES = TEXT.encode()
TEXT_GZIPPED = gzip.compress(TEXT_BYTES)
URL = "http://host/file.gz"


def gzip_callback(request, data=GZIPPED):
    """Answer range requests against one fixed gzip body, like test_http.request_callback."""
    range_string = request.headers.get("range", "bytes=0-")
    start = int(range_string.removeprefix("bytes=").partition("-")[0])
    if not start:
        return 200, {"Accept-Ranges": "bytes", "Content-Length": str(len(data))}, data
    return (
        206,
        {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}", "Accept-Ranges": "bytes"},
        data[start:],
    )


def _readinto(reader, size):
    buffer = bytearray(size)
    written = reader.readinto(buffer)
    return bytes(buffer[:written])


#
# Snapshot and recorder semantics.
#

def test_snapshot_is_immutable_and_copied():
    """Snapshots are frozen, slotted and independent of later accounting."""
    recorder = IOStatsRecorder()
    recorder.add_transport(5)
    first = recorder.snapshot()
    recorder.add_transport(7)
    second = recorder.snapshot()

    assert first == IOStatsSnapshot(5, 0, 0)
    assert second == IOStatsSnapshot(12, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.transport_bytes = 99
    with pytest.raises(AttributeError):
        first.unexpected = 1


def test_recorder_holds_no_contents():
    """The recorder stores integer counters only, never the bytes themselves."""
    recorder = IOStatsRecorder()
    MeteredReader(io.BytesIO(b"some payload"), recorder, recorder.add_transport).read()

    assert set(vars(recorder)) == {
        "_transport_bytes",
        "_decompressed_bytes",
        "_text_chars",
        "_on_update",
    }
    assert recorder.transport_bytes == len(b"some payload")


#
# MeteredReader: all read methods share one counter.
#

def test_mixed_reads_share_one_counter():
    """read, read1, readinto, readline and iteration meter one data flow."""
    recorder = IOStatsRecorder()
    reader = MeteredReader(io.BytesIO(PLAIN), recorder, recorder.add_decompressed)

    assert reader.read(3) == b"alp"
    assert recorder.decompressed_bytes == 3
    assert reader.read1(4) == b"ha\nb"
    assert recorder.decompressed_bytes == 7
    assert reader.readline() == b"eta\n"
    assert recorder.decompressed_bytes == 11
    assert reader.readline(3) == b"gam"
    assert recorder.decompressed_bytes == 14
    assert _readinto(reader, 5) == b"ma\nde"
    assert recorder.decompressed_bytes == 19
    assert list(reader) == [b"lta\n"]
    assert recorder.decompressed_bytes == len(PLAIN)


def test_readlines_and_iteration_are_metered():
    """Both line consumers record every line, including a final partial line."""
    recorder = IOStatsRecorder()
    reader = MeteredReader(io.BytesIO(PLAIN), recorder, recorder.add_transport)

    assert reader.readlines() == PLAIN.splitlines(keepends=True)
    assert recorder.transport_bytes == len(PLAIN)

    other = IOStatsRecorder()
    another = MeteredReader(io.BytesIO(b"one\ntwo"), other, other.add_transport)
    assert list(another) == [b"one\n", b"two"]
    assert another.io_stats.transport_bytes == len(b"one\ntwo")


def test_empty_and_zero_sized_reads_do_not_count():
    """Short reads at EOF and read(0) leave every counter untouched."""
    recorder = IOStatsRecorder()
    reader = MeteredReader(io.BytesIO(PLAIN), recorder, recorder.add_transport)
    reader.read()
    settled = recorder.snapshot()

    assert reader.read() == b""
    assert reader.read(10) == b""
    assert reader.readline() == b""
    assert reader.read(0) == b""
    assert recorder.snapshot() == settled


def test_failed_read_keeps_only_earned_bytes():
    """A read that raises counts nothing, while earlier successful bytes stay."""

    class Flaky(io.RawIOBase):
        def __init__(self):
            super().__init__()
            self._reads = 0

        def readable(self):
            return True

        def read(self, size=-1):
            self._reads += 1
            if self._reads > 1:
                raise OSError("connection reset")
            return b"ok"

    recorder = IOStatsRecorder()
    reader = MeteredReader(Flaky(), recorder, recorder.add_transport)
    assert reader.read(2) == b"ok"
    with pytest.raises(OSError, match="connection reset"):
        reader.read(2)
    assert recorder.snapshot() == IOStatsSnapshot(2, 0, 0)


def test_callback_failure_does_not_break_or_repeat_the_read():
    """A raising callback is swallowed; the wrapped read runs exactly once."""

    class Counting(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.read_calls = 0

        def read(self, *args, **kwargs):
            self.read_calls += 1
            return super().read(*args, **kwargs)

    stream = Counting(b"hello")
    events = []

    def bad_callback(snapshot):
        events.append(snapshot)
        raise RuntimeError("callback exploded")

    recorder = IOStatsRecorder(bad_callback)
    reader = MeteredReader(stream, recorder, recorder.add_transport)

    assert reader.read() == b"hello"
    assert stream.read_calls == 1
    assert recorder.transport_bytes == len(b"hello")
    assert events == [IOStatsSnapshot(len(b"hello"), 0, 0)]


def test_callback_receives_a_snapshot_per_update():
    """Every metered delivery fires the callback once with an immutable snapshot."""
    events = []
    recorder = IOStatsRecorder(events.append)
    reader = MeteredReader(io.BytesIO(PLAIN), recorder, recorder.add_decompressed)

    reader.read(3)
    reader.readline()
    assert events == [IOStatsSnapshot(0, 3, 0), IOStatsSnapshot(0, 6, 0)]


#
# Full open() stack: hand-checked counters per layer.
#

@responses.activate
def test_gzip_over_http_binary_counts_each_layer():
    """A small gzip body over a fake HTTP source: transport bytes, decompressed bytes, no text."""
    responses.add_callback(responses.GET, URL, callback=gzip_callback)
    with smart_open.open(URL, "rb", transport_params={"io_stats": True, "buffer_size": 8}) as fin:
        assert fin.read() == PLAIN
        snapshot = fin.io_stats.snapshot()
    assert snapshot == IOStatsSnapshot(len(GZIPPED), len(PLAIN), 0)


@responses.activate
def test_gzip_over_http_text_counts_delivered_characters():
    """Text mode adds the decoding layer: characters delivered, not bytes."""
    responses.add(responses.GET, URL, body=TEXT_GZIPPED)
    with smart_open.open(URL, "r", encoding="utf-8", transport_params={"io_stats": True}) as fin:
        assert fin.read() == TEXT
        snapshot = fin.io_stats.snapshot()
    assert snapshot == IOStatsSnapshot(len(TEXT_GZIPPED), len(TEXT_BYTES), len(TEXT))
    assert snapshot.text_chars != snapshot.decompressed_bytes  # мир is 3 chars but 6 bytes


@responses.activate
def test_mixed_reads_over_http_gzip_add_up():
    """Mixed read methods over the full stack deliver each decompressed byte once."""
    responses.add_callback(responses.GET, URL, callback=gzip_callback)
    with smart_open.open(URL, "rb", transport_params={"io_stats": True, "buffer_size": 8}) as fin:
        pieces = [fin.read(3), fin.read1(4), fin.readline(), fin.readline(3), _readinto(fin, 5)]
        pieces.extend(fin)
        snapshot = fin.io_stats.snapshot()
    assert b"".join(pieces) == PLAIN
    assert snapshot == IOStatsSnapshot(len(GZIPPED), len(PLAIN), 0)


@responses.activate
def test_plain_http_binary_has_no_decompressed_layer():
    """Without compression, the decompressed counter stays at zero."""
    responses.add(responses.GET, "http://host/plain", body=PLAIN)
    with smart_open.open("http://host/plain", "rb", transport_params={"io_stats": True}) as fin:
        assert fin.read() == PLAIN
        assert fin.io_stats.snapshot() == IOStatsSnapshot(len(PLAIN), 0, 0)


@pytest.mark.parametrize("consume", [lambda fin: fin.readlines(), list], ids=["readlines", "iteration"])
@responses.activate
def test_text_line_consumers_count_delivered_characters(consume):
    """Both text line consumers report characters delivered, distinct from the byte count."""
    multiline = "line one\nлиния два\nthird line\n"
    responses.add(responses.GET, URL, body=gzip.compress(multiline.encode()))
    with smart_open.open(URL, "r", encoding="utf-8", transport_params={"io_stats": True}) as fin:
        assert consume(fin) == multiline.splitlines(keepends=True)
        snapshot = fin.io_stats.snapshot()
    assert snapshot == IOStatsSnapshot(
        len(gzip.compress(multiline.encode())), len(multiline.encode()), len(multiline)
    )


@responses.activate
def test_seek_refetch_counts_as_network_but_buffered_reread_does_not():
    """A rewind refetches over the network; re-reading buffered bytes is free."""
    body = b"0123456789abcdefghijklmnopqrstuvwxyz" * 2000

    def callback(request):
        start = int(request.headers.get("range", "bytes=0-").removeprefix("bytes=").partition("-")[0])
        if not start:
            return 200, {"Accept-Ranges": "bytes", "Content-Length": str(len(body))}, body
        return (
            206,
            {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}", "Accept-Ranges": "bytes"},
            body[start:],
        )

    responses.add_callback(responses.GET, "http://host/big", callback=callback)
    with smart_open.open(
        "http://host/big", "rb", transport_params={"io_stats": True, "buffer_size": 128}
    ) as fin:
        fin.read(10)
        first_fill = fin.io_stats.transport_bytes  # one 128-byte chunk already left the wire
        assert first_fill == 128

        # A forward seek inside the buffered window consumes memory, not network.
        fin.seek(15)
        fin.read(5)
        assert fin.io_stats.transport_bytes == 128

        # A rewind cannot be served from the buffer: a range request refetches.
        fin.seek(0)
        fin.read(3)
        assert fin.io_stats.transport_bytes == 256

        # Reads past EOF are short reads and add nothing.
        fin.seek(len(body))
        assert fin.read(10) == b""
        assert fin.io_stats.transport_bytes == 256


@responses.activate
def test_capped_range_continuations_count_every_fetched_chunk():
    """Bytes fetched through several capped range responses all leave the wire and get counted."""
    body = b"0123456789abcdefghij"

    def callback(request):
        first = int(request.headers.get("range", "bytes=0-").removeprefix("bytes=").partition("-")[0])
        if not first:
            return 200, {"ETag": '"v1"'}, body
        if first >= len(body):
            return 416, {}, b""
        last = min(first + 4, len(body)) - 1  # the server caps every range at four bytes
        return 206, {"Content-Range": f"bytes {first}-{last}/{len(body)}"}, body[first : last + 1]

    responses.add_callback(responses.GET, "http://host/capped", callback=callback)
    with smart_open.open(
        "http://host/capped", "rb", transport_params={"io_stats": True, "buffer_size": 4}
    ) as fin:
        fin.seek(5)
        assert fin.read() == body[5:]
        assert fin.io_stats.transport_bytes == len(body) - 5


@responses.activate
def test_decode_failure_keeps_lower_layer_counts():
    """When decoding fails, bytes pulled and decompressed stay counted; no text was delivered."""
    bad = b"good\n" + b"\xff\xfe bad\n"
    responses.add(responses.GET, URL, body=gzip.compress(bad))
    with smart_open.open(URL, "r", encoding="utf-8", transport_params={"io_stats": True}) as fin:
        with pytest.raises(UnicodeDecodeError):
            fin.readline()
        snapshot = fin.io_stats.snapshot()
        # Like the built-in TextIOWrapper, the error fires once; later reads are empty
        # and do not move any counter.
        assert fin.read() == ""
        assert fin.readline() == ""
        assert fin.io_stats.snapshot() == snapshot
    assert snapshot == IOStatsSnapshot(len(gzip.compress(bad)), len(bad), 0)


@responses.activate
def test_default_off_leaves_reader_and_cursor_unchanged():
    """Without io_stats, no metering proxy is created and reads/seeks/close behave as before."""
    responses.add_callback(responses.GET, URL, callback=gzip_callback)
    fin = smart_open.open(URL, "rb", transport_params={"buffer_size": 8})
    try:
        assert fin.io_stats is None
        assert not isinstance(fin.__wrapped__, MeteredReader)
        assert fin.read(5) == PLAIN[:5]
        assert fin.tell() == 5
        fin.seek(0)
        assert fin.tell() == 0
        assert fin.read() == PLAIN
    finally:
        fin.close()
    assert fin.closed


@responses.activate
def test_io_stats_false_is_equivalent_to_default_off():
    """An explicit False disables metering."""
    responses.add(responses.GET, URL, body=GZIPPED)
    fin = smart_open.open(URL, "rb", transport_params={"io_stats": False})
    try:
        assert fin.io_stats is None
        assert not isinstance(fin.__wrapped__, MeteredReader)
        assert fin.read() == PLAIN
    finally:
        fin.close()


def test_invalid_io_stats_option_raises_type_error():
    """Only booleans, recorders and callbacks are accepted."""
    with pytest.raises(TypeError, match="invalid io_stats option"):
        smart_open.open("http://host/x", "rb", transport_params={"io_stats": "nonsense"})


@responses.activate
def test_recorder_instance_and_callback_are_accepted():
    """A supplied recorder is reused; a callable is wrapped as a callback."""
    responses.add(responses.GET, URL, body=GZIPPED)
    recorder = IOStatsRecorder()
    with smart_open.open(URL, "rb", transport_params={"io_stats": recorder}) as fin:
        fin.read()
        assert fin.io_stats is recorder
    assert recorder.snapshot() == IOStatsSnapshot(len(GZIPPED), len(PLAIN), 0)

    events = []
    responses.add(responses.GET, URL, body=GZIPPED)
    with smart_open.open(URL, "rb", transport_params={"io_stats": events.append}) as fin:
        fin.read()
    assert events and isinstance(events[0], IOStatsSnapshot)


@responses.activate
def test_snapshot_after_close_holds_final_state():
    """The recorder survives close and reports the final counters."""
    responses.add(responses.GET, URL, body=GZIPPED)
    recorder = IOStatsRecorder()
    with smart_open.open(URL, "rb", transport_params={"io_stats": recorder}) as fin:
        fin.read()
    final = recorder.snapshot()
    assert final == IOStatsSnapshot(len(GZIPPED), len(PLAIN), 0)


def test_local_gzip_file_uses_fallback_transport_metering(tmp_path):
    """Transports without native support still get counted via the wrapping fallback."""
    path = tmp_path / "data.gz"
    path.write_bytes(GZIPPED)
    with smart_open.open(str(path), "rb", transport_params={"io_stats": True}) as fin:
        assert fin.read() == PLAIN
        snapshot = fin.io_stats.snapshot()
    assert snapshot.decompressed_bytes == len(PLAIN)
    assert 0 < snapshot.transport_bytes <= path.stat().st_size


def test_local_text_shortcut_counts_only_characters(tmp_path):
    """The uncompressed local shortcut has a single layer: delivered characters."""
    path = tmp_path / "data.txt"
    path.write_text(TEXT, encoding="utf-8")
    with smart_open.open(str(path), "r", transport_params={"io_stats": True}) as fin:
        assert fin.read() == TEXT
        snapshot = fin.io_stats.snapshot()
    assert snapshot == IOStatsSnapshot(0, 0, len(TEXT))

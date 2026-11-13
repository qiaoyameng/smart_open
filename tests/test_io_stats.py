#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Tests for optional per-handle I/O accounting (``smart_open.io_stats``)."""

import gzip
import io

import pytest
import responses

import smart_open
import smart_open.constants
from smart_open.io_stats import IOStats, StatsSnapshot

URL = "http://localhost/data.gz"
PLAIN_URL = "http://localhost/data.bin"
TEXT = "hello world\nline two\n" * 20
RAW = TEXT.encode("utf-8")
GZIPPED = gzip.compress(RAW)


def _range_callback(data):
    """Build a fake HTTP source that answers open-ended byte-range requests."""

    def callback(request):
        range_string = request.headers.get("range", "bytes=0-")
        start = int(range_string.replace("bytes=", "").split("-", 1)[0])
        body = data[start:]
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(len(body))}
        if start:
            headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
            return (206, headers, body)
        return (200, headers, body)

    return callback


@responses.activate
def test_gzip_three_layers_under_mixed_reads():
    """Transport, decompressed and text counters each match their own layer."""
    responses.add_callback(responses.GET, URL, callback=_range_callback(GZIPPED))
    stats = IOStats()
    with smart_open.open(URL, "r", transport_params={"io_stats": stats, "buffer_size": 32}) as fin:
        head = fin.read(7)
        line = fin.readline()
        rest = "".join(fin)

    assert head + line + rest == TEXT
    snapshot = stats.snapshot()
    assert snapshot == StatsSnapshot(len(GZIPPED), len(RAW), len(TEXT))
    assert snapshot.transport_bytes == len(GZIPPED)  # compressed bytes on the wire
    assert snapshot.decompressed_bytes == len(RAW)  # gzip expansion
    assert snapshot.text_chars == len(TEXT)


@responses.activate
def test_binary_gzip_has_zero_text_chars():
    """Binary reads never move the text-character counter."""
    responses.add_callback(responses.GET, URL, callback=_range_callback(GZIPPED))
    stats = IOStats()
    with smart_open.open(URL, "rb", transport_params={"io_stats": stats, "buffer_size": 32}) as fin:
        data = fin.read(10) + fin.read1(10) + fin.read()

    assert data == RAW
    snapshot = stats.snapshot()
    assert snapshot.transport_bytes == len(GZIPPED)
    assert snapshot.decompressed_bytes == len(RAW)
    assert snapshot.text_chars == 0


@responses.activate
def test_no_compression_transport_equals_decompressed():
    """Without compression the two byte counters are identical and exact per read."""
    responses.add_callback(responses.GET, PLAIN_URL, callback=_range_callback(RAW))
    stats = IOStats()
    with smart_open.open(
        PLAIN_URL, "rb", compression="disable", transport_params={"io_stats": stats, "buffer_size": 16}
    ) as fin:
        first = fin.read(5)
        second = fin.read(11)

    assert first + second == RAW[:16]
    snapshot = stats.snapshot()
    assert snapshot.decompressed_bytes == 16
    # One 16-byte transport fill satisfied both reads; buffered reuse is free.
    assert snapshot.transport_bytes == 16


@responses.activate
def test_readinto_meters_only_copied_bytes():
    """``readinto`` counts the bytes actually written into the caller buffer."""
    responses.add_callback(responses.GET, PLAIN_URL, callback=_range_callback(RAW))
    stats = IOStats()
    with smart_open.open(
        PLAIN_URL, "rb", compression="disable", transport_params={"io_stats": stats, "buffer_size": 64}
    ) as fin:
        buffer = bytearray(12)
        num_bytes = fin.readinto(buffer)

    assert num_bytes == 12
    assert bytes(buffer) == RAW[:12]
    assert stats.snapshot().decompressed_bytes == 12


@responses.activate
def test_seek_refetch_charges_transport_but_buffer_seek_does_not():
    """A network range re-read is metered; an in-buffer forward seek is not."""
    responses.add_callback(responses.GET, PLAIN_URL, callback=_range_callback(RAW))
    stats = IOStats()
    with smart_open.open(
        PLAIN_URL, "rb", compression="disable", transport_params={"io_stats": stats, "buffer_size": 64}
    ) as fin:
        assert fin.read(10) == RAW[:10]
        after_first_read = stats.snapshot().transport_bytes

        # Backward seek cannot use the buffer, so it issues a fresh range request.
        fin.seek(0)
        assert fin.read(10) == RAW[:10]
        after_refetch = stats.snapshot().transport_bytes
        assert after_refetch > after_first_read

        # Forward seek inside the buffered window pulls nothing new.
        position = fin.tell()
        fin.seek(position + 5)
        buffered = stats.snapshot().transport_bytes
        assert fin.read(5) == RAW[15:20]
        assert stats.snapshot().transport_bytes == buffered


@responses.activate
def test_snapshot_is_immutable_and_survives_close():
    """The snapshot copies integers, cannot be mutated, and outlives the handle."""
    responses.add_callback(responses.GET, PLAIN_URL, callback=_range_callback(RAW))
    stats = IOStats()
    with smart_open.open(PLAIN_URL, "rb", compression="disable", transport_params={"io_stats": stats}) as fin:
        fin.read(8)
        snapshot = stats.snapshot()

    assert isinstance(snapshot, StatsSnapshot)
    with pytest.raises(AttributeError):
        snapshot.transport_bytes = 999
    # Closing the handle does not retroactively change a snapshot.
    assert stats.snapshot() == snapshot


def test_on_update_failure_does_not_repeat_read():
    """A raising callback surfaces after one successful read, which is never re-executed."""
    from smart_open.io_stats import _CountingBinaryReader

    underlying_reads = 0

    class OneShotSource(io.RawIOBase):
        def readable(self):
            return True

        def readinto(self, buffer):
            nonlocal underlying_reads
            underlying_reads += 1
            data = b"hello"
            view = memoryview(buffer)
            view[: len(data)] = data
            return len(data)

    def boom():
        raise RuntimeError("callback failed")

    stats = IOStats(on_update=boom)
    reader = _CountingBinaryReader(io.BufferedReader(OneShotSource()), stats.record_decompressed)
    with pytest.raises(RuntimeError, match="callback failed"):
        reader.read(5)

    # The source was read exactly once and the bytes were still accounted for.
    assert underlying_reads == 1
    assert stats.snapshot().decompressed_bytes == 5


class _ShortSource(io.RawIOBase):
    """Raw source that returns at most ``cap`` bytes per ``readinto``."""

    def __init__(self, data, cap):
        self._buffer = io.BytesIO(data)
        self._cap = cap

    def readable(self):
        return True

    def readinto(self, buffer):
        view = memoryview(buffer)
        data = self._buffer.read(min(self._cap, len(view)))
        view[: len(data)] = data
        return len(data)


class _FailingSource(io.RawIOBase):
    """Yields one chunk, then raises on every subsequent read."""

    def __init__(self):
        self._calls = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        self._calls += 1
        if self._calls == 1:
            view = memoryview(buffer)
            view[:5] = b"12345"
            return 5
        raise OSError("boom after first chunk")


def test_short_read_meters_each_delivered_chunk():
    """Repeated short reads accumulate exactly the bytes handed to the caller."""
    from smart_open.io_stats import _CountingBinaryReader

    stats = IOStats()
    reader = _CountingBinaryReader(io.BufferedReader(_ShortSource(b"abcdefghij" * 3, 4)), stats.record_decompressed)
    buffer = bytearray(7)
    assert reader.readinto(buffer) == 7
    assert bytes(buffer) == b"abcdefg"
    assert stats.snapshot().decompressed_bytes == 7


def test_failed_whole_read_counts_no_partial_delivery():
    """A ``read`` that cannot fill the request and raises delivers nothing to count."""
    from smart_open.io_stats import _CountingBinaryReader

    stats = IOStats()
    reader = _CountingBinaryReader(io.BufferedReader(_FailingSource()), stats.record_decompressed)
    with pytest.raises(OSError, match="boom"):
        reader.read(100)
    assert stats.snapshot().decompressed_bytes == 0


def test_read1_surfaces_bytes_obtained_before_failure():
    """``read1`` returns the buffered chunk (and counts it) before the next read fails."""
    from smart_open.io_stats import _CountingBinaryReader

    stats = IOStats()
    reader = _CountingBinaryReader(io.BufferedReader(_FailingSource()), stats.record_decompressed)
    assert reader.read1(100) == b"12345"
    assert stats.snapshot().decompressed_bytes == 5
    with pytest.raises(OSError, match="boom"):
        reader.read1(100)


@responses.activate
def test_truncated_gzip_keeps_transport_count_on_decompress_failure():
    """Bytes pulled before a decompression error stay charged to the transport."""
    truncated = GZIPPED[: len(GZIPPED) // 2]
    responses.add(
        responses.GET,
        URL,
        body=truncated,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(len(truncated))},
    )
    stats = IOStats()
    with smart_open.open(URL, "rb", transport_params={"io_stats": stats, "buffer_size": 16}) as fin:
        with pytest.raises((OSError, EOFError)):
            fin.read()

    snapshot = stats.snapshot()
    assert snapshot.transport_bytes == len(truncated)
    assert snapshot.decompressed_bytes == 0


@responses.activate
def test_on_update_fires_through_full_stack():
    """The update callback observes each layer's counters during a real streamed read."""
    responses.add_callback(responses.GET, URL, callback=_range_callback(GZIPPED))
    events = []
    stats = IOStats(on_update=lambda: events.append(stats.snapshot()))
    with smart_open.open(URL, "r", transport_params={"io_stats": stats, "buffer_size": 32}) as fin:
        fin.read(4)

    assert events  # at least one layer reported progress
    final = stats.snapshot()
    assert final == events[-1]
    assert final.text_chars == 4
    assert final.decompressed_bytes >= 4
    assert final.transport_bytes > 0


@responses.activate
def test_default_off_does_not_change_reader_behavior():
    """Without ``io_stats``, reads, the cursor and closing behave exactly as before."""
    responses.add_callback(responses.GET, PLAIN_URL, callback=_range_callback(RAW))
    with smart_open.open(PLAIN_URL, "rb", compression="disable", transport_params={"buffer_size": 32}) as fin:
        assert fin.read(5) == RAW[:5]
        assert fin.tell() == 5
        fin.seek(0, whence=smart_open.constants.WHENCE_END)
        assert fin.tell() == len(RAW)
        assert fin.read() == b""
    assert fin.closed


def test_invalid_io_stats_type_rejected():
    """A non-IOStats value in transport_params fails fast with a clear type error."""
    with pytest.raises(TypeError, match="IOStats"):
        smart_open.open(PLAIN_URL, "rb", transport_params={"io_stats": object()})

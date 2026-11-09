"""Tests for the session-only gzip decompression checkpoint index."""

from __future__ import annotations

import gzip
import io
import os
import random
import struct
import zlib

import pytest

from smart_open import open as smart_open
from smart_open.compression import compression_wrapper
from smart_open.gzip_index import (
    DEFAULT_INDEX_INTERVAL,
    DEFAULT_MAX_CHECKPOINTS,
    IndexedGzipReader,
    open_indexed_gzip,
)


class CountingRawFile(io.FileIO):
    """FileIO subclass counting physical reads and seeks on the real file."""

    def __init__(self, path: str) -> None:
        super().__init__(path, "rb")
        self.read_count = 0
        self.bytes_read = 0
        self.seek_count = 0

    def read(self, size=-1):
        self.read_count += 1
        data = super().read(size)
        self.bytes_read += len(data)
        return data

    def readinto(self, buffer):
        self.read_count += 1
        result = super().readinto(buffer)
        if result and result > 0:
            self.bytes_read += result
        return result

    def seek(self, offset, whence=io.SEEK_SET):
        self.seek_count += 1
        return super().seek(offset, whence)


def make_membered_gzip(path: str, chunks: list[bytes]) -> bytes:
    """Write concatenated gzip members (some empty) and return plain data."""
    plain = b"".join(chunks)
    with open(path, "wb") as handle:
        handle.write(gzip.compress(chunks[0]))
        handle.write(gzip.compress(b""))
        handle.write(gzip.compress(chunks[1]))
        handle.write(gzip.compress(b""))
        handle.write(gzip.compress(chunks[2]))
    return plain


@pytest.fixture()
def gzip_file(tmp_path):
    rng = random.Random(1234)
    chunks = [
        bytes(rng.randrange(256) for _ in range(40_000)),
        b"".join(bytes([i % 256]) * 50 for i in range(2000)),
        bytes(rng.randrange(256) for _ in range(60_000)),
    ]
    path = tmp_path / "sample.gz"
    plain = make_membered_gzip(str(path), chunks)
    return path, plain


def test_full_read_matches_gzip(gzip_file):
    path, plain = gzip_file
    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(handle)
        assert indexed.read() == plain

    with gzip.open(path, "rb") as reference:
        assert reference.read() == plain


def test_random_operations_match_gzip(gzip_file):
    path, plain = gzip_file
    rng = random.Random(99)
    with (
        open(path, "rb") as indexed_handle,
        open(path, "rb") as reference_handle,
    ):
        indexed = open_indexed_gzip(
            indexed_handle, index_interval=8192, max_checkpoints=16
        )
        reference = gzip.open(reference_handle, "rb")

        operations = 0
        while operations < 400:
            choice = rng.randrange(6)
            if choice < 3:
                size = rng.randint(0, 700)
                assert indexed.read(size) == reference.read(size)
            elif choice == 3:
                target = rng.randrange(0, len(plain))
                assert indexed.seek(target) == reference.seek(target)
            elif choice == 4:
                delta = rng.randint(-500, 500)
                low = max(0, indexed.tell() + delta)
                if low <= len(plain):
                    assert indexed.seek(delta, io.SEEK_CUR) == reference.seek(
                        delta, io.SEEK_CUR
                    )
            else:
                delta = rng.randint(-64, 0)
                assert indexed.seek(delta, io.SEEK_END) == reference.seek(
                    delta, io.SEEK_END
                )
            assert indexed.tell() == reference.tell()
            operations += 1


def test_readinto_matches_gzip(gzip_file):
    path, plain = gzip_file
    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(handle, index_interval=4096)
        output = bytearray()
        stage = bytearray(333)
        while True:
            count = indexed.readinto(stage)
            if count == 0:
                break
            output += stage[:count]
        assert bytes(output) == plain


def test_seek_end_scans_total_length(gzip_file):
    path, plain = gzip_file
    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(handle)
        assert indexed.seek(0, io.SEEK_END) == len(plain)
        with gzip.open(path, "rb") as reference:
            assert reference.seek(0, io.SEEK_END) == len(plain)


def test_backward_seek_reuses_checkpoint_without_rereading(gzip_file):
    path, plain = gzip_file
    counter = CountingRawFile(str(path))
    indexed = open_indexed_gzip(counter, index_interval=4096, max_checkpoints=32)

    # Walk far enough forward to create several checkpoints.
    indexed.seek(30_000)
    indexed.read(2_000)
    warmup_reads = counter.read_count
    warmup_bytes = counter.bytes_read
    seek_offsets = []
    counter.seek = (  # type: ignore[method-assign]
        lambda offset, whence=io.SEEK_SET, original=counter.seek: (
            seek_offsets.append(offset) or original(offset, whence)
        )
    )

    for _ in range(20):
        indexed.seek(12_000)
        assert indexed.read(256) == plain[12_000:12_256]
        indexed.seek(25_000)
        assert indexed.read(128) == plain[25_000:25_128]

    # No compressed bytes are re-read from disk while hopping between
    # retained checkpoints: the saved decompressor state covers everything.
    assert counter.read_count == warmup_reads
    assert counter.bytes_read == warmup_bytes
    # Seeks only re-align to checkpoint positions, never restart at offset 0.
    assert seek_offsets
    assert all(offset > 0 for offset in seek_offsets)
    indexed.close()


def test_indexed_reader_beats_gzip_redecompress_counter(gzip_file):
    """Same repeated hops: stdlib re-reads compressed prefix, the index does not."""
    path, plain = gzip_file

    indexed_counter = CountingRawFile(str(path))
    indexed = open_indexed_gzip(
        indexed_counter, index_interval=4096, max_checkpoints=32
    )
    indexed.seek(30_000)
    indexed.read(2_000)
    indexed_hot_bytes = indexed_counter.bytes_read

    gzip_counter = CountingRawFile(str(path))
    reference = gzip.open(gzip_counter, "rb")
    reference.seek(30_000)
    reference.read(2_000)
    gzip_hot_bytes = gzip_counter.bytes_read

    hops = 15
    for _ in range(hops):
        indexed.seek(12_000)
        indexed.read(256)
        indexed.seek(25_000)
        indexed.read(128)

        reference.seek(12_000)
        assert reference.read(256) == plain[12_000:12_256]
        reference.seek(25_000)
        assert reference.read(128) == plain[25_000:25_128]

    # The indexed reader adds zero physical bytes; gzip re-pulls a growing
    # compressed prefix on every backwards seek.
    assert indexed_counter.bytes_read == indexed_hot_bytes
    assert gzip_counter.bytes_read > gzip_hot_bytes
    assert gzip_counter.bytes_read > indexed_counter.bytes_read
    indexed.close()
    reference.close()


def test_checkpoint_eviction_is_bounded(gzip_file):
    path, _ = gzip_file
    with open(path, "rb") as handle:
        reader = IndexedGzipReader(
            handle, index_interval=2048, max_checkpoints=4
        )
        reader.read(40_000)
        assert len(reader._checkpoints) <= 4
        assert 0 in reader._checkpoints
        # Pending compressed state is only buffer-sized, never the full file.
        largest_pending = max(
            len(checkpoint.pending)
            for checkpoint in reader._checkpoints.values()
        )
        assert largest_pending <= 128 * 1024


def test_evicted_checkpoint_falls_back_to_older_anchor(gzip_file):
    path, plain = gzip_file
    with open(path, "rb") as handle:
        indexed = io.BufferedReader(
            IndexedGzipReader(handle, index_interval=4096, max_checkpoints=3)
        )
        indexed.read(30_000)
        indexed.seek(5_000)
        assert indexed.read(999) == plain[5_000:5_999]


def test_zero_anchor_rewinds_underlying_file(gzip_file):
    path, plain = gzip_file
    counter = CountingRawFile(str(path))
    indexed = open_indexed_gzip(counter, index_interval=4096)
    indexed.read(20_000)
    indexed.seek(0)
    assert counter.seek_count >= 1
    assert indexed.read(123) == plain[:123]
    indexed.close()


def test_crc_failure_survives_checkpoint_restore(tmp_path):
    plain = b"checkpoint me " * 1000
    path = tmp_path / "bad.gz"
    blob = bytearray(gzip.compress(plain))
    blob[-8] ^= 0x01  # corrupt stored CRC
    path.write_bytes(bytes(blob))

    counter = CountingRawFile(str(path))
    indexed = open_indexed_gzip(counter, index_interval=256, max_checkpoints=8)
    indexed.seek(1_000)
    indexed.seek(500)
    with pytest.raises(gzip.BadGzipFile, match="CRC"):
        indexed.read()
    indexed.close()


def test_length_failure_still_raised(tmp_path):
    plain = b"isize must match" * 100
    path = tmp_path / "badlen.gz"
    blob = bytearray(gzip.compress(plain))
    blob[-1] = (blob[-1] + 1) & 0xFF  # corrupt stored ISIZE
    path.write_bytes(bytes(blob))

    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(handle, index_interval=128)
        with pytest.raises(gzip.BadGzipFile, match="length"):
            indexed.read()


def test_truncated_stream_raises(tmp_path):
    path = tmp_path / "trunc.gz"
    path.write_bytes(gzip.compress(b"x" * 5000)[:-20])
    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(handle)
        with pytest.raises((EOFError, gzip.BadGzipFile)):
            indexed.read()


def test_rejects_non_seekable_stream():
    sink = io.BytesIO(gzip.compress(b"anything"))
    class NonSeekable(io.RawIOBase):
        def readable(self):
            return True

        def seekable(self):
            return False

        def read(self, size=-1):
            return sink.read(size)

    with pytest.raises(ValueError, match="seekable"):
        open_indexed_gzip(NonSeekable())


def test_rejects_in_memory_stream():
    with pytest.raises(ValueError, match="local"):
        open_indexed_gzip(io.BytesIO(gzip.compress(b"in memory")))


def test_rejects_text_stream(tmp_path, gzip_file):
    path, _ = gzip_file
    with open(path, "r") as text_handle:
        with pytest.raises(ValueError, match="binary"):
            open_indexed_gzip(text_handle)


def test_rejects_pipe(tmp_path):
    read_end, write_end = os.pipe()
    try:
        with os.fdopen(read_end, "rb", closefd=True) as pipe:
            with pytest.raises(ValueError, match="seekable|local regular"):
                open_indexed_gzip(pipe)
    finally:
        os.close(write_end)


def test_invalid_configuration(tmp_path, gzip_file):
    path, _ = gzip_file
    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="index_interval"):
            IndexedGzipReader(handle, index_interval=0)
        with pytest.raises(ValueError, match="max_checkpoints"):
            IndexedGzipReader(handle, max_checkpoints=0)


def test_smart_open_opt_in_path(gzip_file):
    path, plain = gzip_file
    with smart_open(
        str(path),
        "rb",
        compression=".gz",
        compression_kwargs={
            "indexed_seek": True,
            "index_interval": 8192,
            "max_checkpoints": 10,
        },
    ) as indexed:
        indexed.seek(1_000)
        assert indexed.read(500) == plain[1_000:1_500]
        assert indexed.tell() == 1_500


def test_smart_open_default_path_unchanged(gzip_file):
    path, plain = gzip_file
    with smart_open(str(path), "rb", compression=".gz") as handle:
        assert type(handle.raw).__name__ != "IndexedGzipReader"
        assert handle.read() == plain


def test_indexed_seek_rejected_for_other_compression(gzip_file):
    path, _ = gzip_file
    with open(path, "rb") as handle:
        with pytest.raises(ValueError, match="only supported for gzip"):
            compression_wrapper(
                handle,
                "rb",
                compression=".bz2",
                compression_kwargs={"indexed_seek": True},
            )


def test_defaults_exposed():
    assert DEFAULT_INDEX_INTERVAL > 0
    assert DEFAULT_MAX_CHECKPOINTS > 0


def _flagged_member(data, flags, extra=b"", name=b""):
    header = b"\x1f\x8b\x08" + bytes([flags]) + b"\x00\x00\x00\x00\x00\xff"
    if flags & 4:
        header += struct.pack("<H", len(extra)) + extra
    if flags & 8:
        header += name + b"\x00"
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    body = compressor.compress(data) + compressor.flush()
    trailer = struct.pack(
        "<II", zlib.crc32(data) & 0xFFFFFFFF, len(data) & 0xFFFFFFFF
    )
    return header + body + trailer


def test_flagged_headers_and_cross_member_checkpoints(tmp_path):
    first = b"A" * 7000
    second = b"B" * 5000
    blob = _flagged_member(
        first, flags=12, extra=b"XY", name=b"first.txt"
    ) + gzip.compress(b"") + _flagged_member(second, flags=8, name=b"second")
    plain = first + second
    assert gzip.decompress(blob) == plain
    path = tmp_path / "flagged.gz"
    path.write_bytes(blob)

    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(
            handle, index_interval=2000, max_checkpoints=64
        )
        assert indexed.read() == plain
        positions = sorted(indexed.raw._checkpoints)
        assert any(0 < pos < len(first) for pos in positions)
        assert any(pos > len(first) for pos in positions)

        indexed.seek(0)
        with gzip.open(io.BytesIO(blob), "rb") as reference:
            for target in (0, 6999, len(first), len(first) + 1, 9000):
                assert indexed.seek(target) == reference.seek(target)
                assert indexed.read(137) == reference.read(137)
                assert indexed.tell() == reference.tell()


def test_indexed_mode_rejected_for_write(tmp_path):
    path = tmp_path / "out.gz"
    with open(path, "wb") as handle:
        with pytest.raises(ValueError, match="reading"):
            compression_wrapper(
                handle,
                "wb",
                compression=".gz",
                compression_kwargs={"indexed_seek": True},
            )


def test_round_trip_does_not_reread_compressed_prefix(gzip_file):
    path, plain = gzip_file
    counter = CountingRawFile(str(path))
    indexed = open_indexed_gzip(
        counter, index_interval=4096, max_checkpoints=32
    )
    indexed.seek(40_000)
    indexed.seek(8_000)
    indexed.read(1_000)
    bytes_after_warmup = counter.bytes_read

    for _ in range(10):
        indexed.seek(9_000)
        indexed.read(512)
        indexed.seek(20_000)
        indexed.read(256)

    assert counter.bytes_read == bytes_after_warmup
    assert indexed.tell() == 20_256
    assert indexed.seek(9_000) == 9_000
    assert indexed.read(512) == plain[9_000:9_512]
    indexed.close()


def test_member_end_during_capped_decompress(tmp_path):
    """Regression: EOF reached while max_length truncates the feed.

    In that state ``unconsumed_tail`` contains the member trailer and the
    following member; mishandling it re-feeds framing bytes as deflate data.
    """
    first = b"A" * 100_000
    second = b"B" * 50_000
    path = tmp_path / "capped_eof.gz"
    path.write_bytes(gzip.compress(first) + gzip.compress(second))
    plain = first + second

    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(
            handle, index_interval=256, max_checkpoints=8
        )
        rng = random.Random(11)
        for _ in range(500):
            target = rng.randrange(len(plain))
            count = rng.randint(1, 200)
            indexed.seek(target)
            assert indexed.read(count) == plain[target:target + count]
        indexed.seek(0)
        assert indexed.read() == plain


def test_checkpoint_restores_underlying_file_position(tmp_path):
    """A checkpoint must re-align the physical file, not just the zlib state.

    After later reads have advanced the underlying file, restoring an older
    checkpoint whose decompressor buffer does not cover the whole gap would
    otherwise silently skip compressed bytes.
    """
    rng = random.Random(77)
    plain = bytes(rng.randrange(256) for _ in range(150_000))
    members = [
        gzip.compress(plain[:40_000]),
        gzip.compress(plain[40_000:95_000]),
        gzip.compress(plain[95_000:]),
    ]
    path = tmp_path / "realign.gz"
    path.write_bytes(b"".join(members))

    counter = CountingRawFile(str(path))
    indexed = open_indexed_gzip(
        counter, index_interval=16_384, max_checkpoints=2
    )
    indexed.seek(0, io.SEEK_END)
    # Aggressive eviction leaves very few checkpoints; hop across the file.
    for target in (140_000, 20_000, 130_000, 5_000, 100_000, 0):
        indexed.seek(target)
        assert indexed.read(333) == plain[target:target + 333]
        assert indexed.tell() == target + 333
    indexed.close()


def test_retained_state_does_not_grow_with_file_length(tmp_path):
    """The index footprint stays bounded by max_checkpoints, not file size."""
    rng = random.Random(5)
    snapshots = []
    for size in (30_000, 120_000, 240_000):
        plain = bytes(rng.randrange(256) for _ in range(size))
        path = tmp_path / f"grow_{size}.gz"
        path.write_bytes(gzip.compress(plain))
        with open(path, "rb") as handle:
            reader = IndexedGzipReader(
                handle, index_interval=4096, max_checkpoints=4
            )
            reader.seek(0, io.SEEK_END)
            assert len(reader._checkpoints) <= 4
            footprint = sum(
                len(cp.pending)
                + len(cp.decompressor.unconsumed_tail)
                + len(cp.decompressor.unused_data)
                for cp in reader._checkpoints.values()
            )
            snapshots.append(footprint)
    # An 8x larger file must not produce an 8x larger index.
    assert snapshots[-1] <= snapshots[0] * 2
    # No checkpoint may hold the whole decompressed payload.
    assert all(footprint < 30_000 for footprint in snapshots)


@pytest.mark.parametrize(
    ("blob", "plain"),
    [
        (gzip.compress(b""), b""),
        (gzip.compress(b"") + gzip.compress(b""), b""),
        (gzip.compress(b"") + gzip.compress(b"hello"), b"hello"),
        (gzip.compress(b"abc") + b"\x00\x00\x00", b"abc"),
    ],
)
def test_empty_members_and_padding(tmp_path, blob, plain):
    path = tmp_path / "edge.gz"
    path.write_bytes(blob)
    with gzip.open(path, "rb") as reference, open(path, "rb") as handle:
        indexed = open_indexed_gzip(handle)
        assert indexed.read() == reference.read() == plain
        assert indexed.seek(0, io.SEEK_END) == len(plain)
        indexed.seek(0)
        assert indexed.read(100) == plain


def test_readline_and_iteration(tmp_path):
    text = b"line one\nline two\nline three\n" * 500
    path = tmp_path / "lines.gz"
    path.write_bytes(gzip.compress(text))
    with open(path, "rb") as handle:
        indexed = open_indexed_gzip(handle, index_interval=512)
        assert indexed.readline() == b"line one\n"
        indexed.seek(0)
        assert b"".join(indexed) == text

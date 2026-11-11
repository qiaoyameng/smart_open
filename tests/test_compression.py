#
# Copyright (C) 2020 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
import bz2
import gzip
import io
import logging
import lzma
import sys

import pytest

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd
import lz4.frame

import smart_open
import smart_open.compression

plain = "доброе утро планета!".encode()
multiline = ("первая строка\nвторая строка\nтретья, длинная " + "строка " * 200).encode()


def label(thing, name):
    """Label."""
    thing.name = name
    return thing


@pytest.mark.parametrize(
    ("fileobj", "compression", "filename"),
    [
        (io.BytesIO(plain), "disable", None),
        (io.BytesIO(plain), "disable", ""),
        (io.BytesIO(plain), "infer_from_extension", "file.txt"),
        (io.BytesIO(plain), "infer_from_extension", "file.TXT"),
        (io.BytesIO(plain), ".unknown", ""),
        (io.BytesIO(gzip.compress(plain)), "infer_from_extension", "file.gz"),
        (io.BytesIO(gzip.compress(plain)), "infer_from_extension", "file.GZ"),
        (label(io.BytesIO(gzip.compress(plain)), "file.gz"), "infer_from_extension", ""),
        (io.BytesIO(gzip.compress(plain)), ".gz", "file.gz"),
        (io.BytesIO(zstd.compress(plain)), "infer_from_extension", "file.zst"),
        (io.BytesIO(zstd.compress(plain)), "infer_from_extension", "file.ZST"),
        (label(io.BytesIO(zstd.compress(plain)), "file.zst"), "infer_from_extension", ""),
        (io.BytesIO(zstd.compress(plain)), ".zst", "file.zst"),
        (io.BytesIO(lzma.compress(plain)), "infer_from_extension", "file.xz"),
        (io.BytesIO(lzma.compress(plain)), "infer_from_extension", "file.XZ"),
        (label(io.BytesIO(lzma.compress(plain)), "file.xz"), "infer_from_extension", ""),
        (io.BytesIO(lzma.compress(plain)), ".xz", "file.xz"),
        (io.BytesIO(bz2.compress(plain)), "infer_from_extension", "file.bz2"),
        (io.BytesIO(bz2.compress(plain)), "infer_from_extension", "file.BZ2"),
        (label(io.BytesIO(bz2.compress(plain)), "file.bz2"), "infer_from_extension", ""),
        (io.BytesIO(bz2.compress(plain)), ".bz2", "file.bz2"),
        (io.BytesIO(lz4.frame.compress(plain)), "infer_from_extension", "file.lz4"),
        (io.BytesIO(lz4.frame.compress(plain)), "infer_from_extension", "file.LZ4"),
        (label(io.BytesIO(lz4.frame.compress(plain)), "file.lz4"), "infer_from_extension", ""),
        (io.BytesIO(lz4.frame.compress(plain)), ".lz4", "file.lz4"),
    ],
)
def test_compression_wrapper_read(fileobj, compression, filename):
    """Compression wrapper read."""
    wrapped = smart_open.compression.compression_wrapper(fileobj, "rb", compression, filename)
    assert wrapped.read() == plain


class OneByteReader(io.RawIOBase):
    """Non-seekable source that hands out exactly one byte per read call."""

    def __init__(self, data):
        super().__init__()
        self._buffer = io.BytesIO(data)
        self.read_calls = 0

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, b):
        byte = self._buffer.read(1)
        if not byte:
            return 0
        self.read_calls += 1
        b[:1] = byte
        return 1


DETECT_BLOBS = {
    ".gz": gzip.compress(multiline),
    ".bz2": bz2.compress(multiline),
    ".xz": lzma.compress(multiline),
    "plain": multiline,
}


@pytest.mark.parametrize("kind", sorted(DETECT_BLOBS))
def test_detect_full_roundtrip(kind):
    """Detect mode reads each codec end to end from a non-seekable one-byte source."""
    source = OneByteReader(DETECT_BLOBS[kind])
    wrapped = smart_open.compression.compression_wrapper(
        source, "rb", "detect", filename="download-without-ext"
    )
    first_line, rest = wrapped.readline(), wrapped.read()
    assert first_line == "первая строка\n".encode()
    assert first_line + rest == multiline
    assert source.read_calls > len(DETECT_BLOBS[kind]) // 2  # genuinely one byte at a time


@pytest.mark.parametrize("kind", sorted(DETECT_BLOBS))
def test_detect_byte_by_byte_consumption(kind):
    """Reading one output byte at a time still reproduces the full payload."""
    source = OneByteReader(DETECT_BLOBS[kind])
    wrapped = smart_open.compression.compression_wrapper(source, "rb", "detect", filename="stream")
    chunks = []
    while True:
        chunk = wrapped.read(1)
        if not chunk:
            break
        assert len(chunk) == 1
        chunks.append(chunk)
    assert b"".join(chunks) == multiline


# Closing the outer wrapper closes the raw source whenever the chosen codec's
# own handler does: bz2/xz and the plain replay close it; GzipFile never closes
# its fileobj, matching the pre-existing explicit-.gz ownership behavior.
_EXPECTED_SOURCE_CLOSED = {".gz": False, ".bz2": True, ".xz": True, "plain": True}


@pytest.mark.parametrize("kind", sorted(DETECT_BLOBS))
def test_detect_early_close_finalizes_source(kind):
    """Closing before EOF follows the same ownership rules as an explicit codec."""
    source = OneByteReader(DETECT_BLOBS[kind])
    wrapped = smart_open.compression.compression_wrapper(source, "rb", "detect", filename="stream")
    head = wrapped.read(8)
    assert head == multiline[: len(head)]
    wrapped.close()
    wrapped.close()  # closing twice is a no-op
    assert source.closed is _EXPECTED_SOURCE_CLOSED[kind]


def test_detect_empty_stream_is_plain():
    """An empty stream is ordinary data, not a truncated container."""
    source = OneByteReader(b"")
    wrapped = smart_open.compression.compression_wrapper(source, "rb", "detect", filename="empty")
    assert wrapped.read() == b""
    wrapped.close()
    assert source.closed


# Every proper (incomplete) prefix of each magic signature, with no further
# bytes available: detection must flag a truncated container rather than
# swallow the data as plain.
_TRUNCATED_PREFIXES = [
    prefix
    for magic, _ in smart_open.compression._MAGIC_SIGNATURES
    for prefix in (magic[:length] for length in range(1, len(magic)))
]


@pytest.mark.parametrize("prefix", _TRUNCATED_PREFIXES)
def test_detect_truncated_magic_prefix(prefix):
    """A stream ending inside a magic signature is diagnosed as truncated."""
    with pytest.raises(EOFError, match="truncated"):
        smart_open.compression.compression_wrapper(io.BytesIO(prefix), "rb", "detect", filename="cut")


@pytest.mark.parametrize("prefix", [b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00"])
def test_detect_full_magic_but_truncated_container(prefix):
    """Complete magic followed by EOF selects the codec; the decoder reports truncation."""
    wrapped = smart_open.compression.compression_wrapper(io.BytesIO(prefix), "rb", "detect", filename="cut")
    with pytest.raises(EOFError):
        wrapped.read()


@pytest.mark.parametrize(
    "data",
    [
        b"\x1fX",
        b"\x1f\x8c",
        b"BZx",
        b"Bhello\n",
        b"\xfd8",
        b"\xfd7y",
        b"\xfd7zXZY",
        b"\x1fXXXXX ordinary bytes",
        b"plain text that happens to start oddly\n",
    ],
)
def test_detect_plain_data_with_magic_like_prefix(data):
    """Ordinary data that merely resembles a magic prefix is returned byte for byte."""
    source = OneByteReader(data)
    wrapped = smart_open.compression.compression_wrapper(source, "rb", "detect", filename="noext")
    assert wrapped.read() == data
    wrapped.close()
    assert source.closed


def test_detect_waits_for_more_bytes():
    """A prefix that is still ambiguous while more bytes arrive is not misclassified."""
    # Two bytes in, the data could still become gzip; the third byte diverges.
    source = OneByteReader(b"\x1f\x8cordinary data follows")
    wrapped = smart_open.compression.compression_wrapper(source, "rb", "detect", filename="noext")
    assert wrapped.read() == b"\x1f\x8cordinary data follows"


def test_detect_filename_conflict_lenient(caplog):
    """By default an extension/content mismatch is diagnosed but content wins."""
    with caplog.at_level(logging.WARNING, logger="smart_open.compression"):
        wrapped = smart_open.compression.compression_wrapper(
            io.BytesIO(b"plain bytes\n"), "rb", "detect", filename="archive.gz"
        )
        assert wrapped.read() == b"plain bytes\n"
    assert ".gz" in caplog.text and "detected" in caplog.text

def test_detect_filename_conflict_content_codec_wins(caplog):
    """Content behind a conflicting registered extension wins, with a warning."""
    blob = gzip.compress(plain)
    with caplog.at_level(logging.WARNING, logger="smart_open.compression"):
        wrapped = smart_open.compression.compression_wrapper(
            io.BytesIO(blob), "rb", "detect", filename="archive.lz4"
        )
        assert wrapped.read() == plain
    assert "archive.lz4" in caplog.text and "detected '.gz'" in caplog.text


def test_detect_strict_rejects_conflict():
    """Strict mode raises on extension/content mismatch and finalizes the raw stream."""
    source = io.BytesIO(b"plain bytes\n")
    with pytest.raises(ValueError, match="implies .gz compression"):
        smart_open.compression.compression_wrapper(source, "rb", "detect_strict", filename="archive.gz")
    assert source.closed


def test_detect_strict_accepts_matching_content():
    """Strict mode stays quiet when extension and content agree."""
    blob = gzip.compress(plain)
    wrapped = smart_open.compression.compression_wrapper(
        io.BytesIO(blob), "rb", "detect_strict", filename="archive.gz"
    )
    assert wrapped.read() == plain


def test_detect_strict_without_extension():
    """With no registered extension to conflict, strict mode equals plain detect."""
    for kind, data in DETECT_BLOBS.items():
        wrapped = smart_open.compression.compression_wrapper(
            OneByteReader(data), "rb", "detect_strict", filename="no-extension"
        )
        assert wrapped.read() == multiline


def test_detect_init_failure_closes_source():
    """If the decompressor fails to initialize, the detect layer owns the cleanup."""
    source = io.BytesIO(gzip.compress(plain))
    with pytest.raises(TypeError):
        smart_open.compression.compression_wrapper(
            source,
            "rb",
            "detect",
            filename="stream",
            compression_kwargs={"definitely_not_a_real_kwarg": 1},
        )
    assert source.closed


@pytest.mark.parametrize("mode", ["wb", "ab", "wb+", "rb+"])
def test_detect_unsupported_modes(mode):
    """Detection is read-only; write and update modes are rejected."""
    with pytest.raises(ValueError, match="only supported for reading|unsupported for mode"):
        smart_open.compression.compression_wrapper(io.BytesIO(), mode, "detect")


def test_explicit_codec_takes_precedence():
    """An explicit codec never goes through content detection."""
    blob = gzip.compress(plain)
    wrapped = smart_open.compression.compression_wrapper(
        io.BytesIO(blob), "rb", ".gz", filename="totally-unrelated.txt"
    )
    assert wrapped.read() == plain


def test_detect_is_a_supported_compression_type():
    """Both new modes are advertised by the registry."""
    types = smart_open.compression.get_supported_compression_types()
    assert "detect" in types and "detect_strict" in types
    # legacy modes keep their position and meaning
    assert types[:2] == ["disable", "infer_from_extension"]


def test_detect_via_smart_open(tmp_path):
    """The public open() API routes extensionless files through detection."""
    payload = "первая строка\nвторая строка\n".encode()
    for name, blob in [
        ("gunzip-me", gzip.compress(payload)),
        ("bunzip-me", bz2.compress(payload)),
        ("unxz-me", lzma.compress(payload)),
        ("read-me", payload),
    ]:
        path = tmp_path / name
        path.write_bytes(blob)
        with smart_open.open(path, "rb", compression="detect") as fin:
            assert fin.read() == payload
        with smart_open.open(path, "rt", compression="detect", encoding="utf-8") as fin:
            assert fin.readline() == "первая строка\n"


def test_infer_from_extension_remains_default_and_compatible():
    """The legacy extension-based path is untouched, including its fallback."""
    # unknown extension: raw passthrough
    wrapped = smart_open.compression.compression_wrapper(
        io.BytesIO(plain), "rb", "infer_from_extension", "file.unknown"
    )
    assert wrapped.read() == plain
    # known extension still decompresses
    wrapped = smart_open.compression.compression_wrapper(
        io.BytesIO(gzip.compress(plain)), "rb", "infer_from_extension", "file.gz"
    )
    assert wrapped.read() == plain
    # disable stays a raw passthrough even for gzip-looking content
    wrapped = smart_open.compression.compression_wrapper(
        io.BytesIO(gzip.compress(plain)), "rb", "disable", "file.gz"
    )
    assert wrapped.read() == gzip.compress(plain)


class ChunkedReader(io.RawIOBase):
    """Non-seekable source that emits a leading chunk, then one byte per read."""

    def __init__(self, head, tail):
        super().__init__()
        self._head = head
        self._tail = io.BytesIO(tail)

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, b):
        if self._head:
            chunk, self._head = self._head[: len(b)], self._head[len(b) :]
            b[: len(chunk)] = chunk
            return len(chunk)
        byte = self._tail.read(1)
        if not byte:
            return 0
        b[:1] = byte
        return 1


_MAGIC_CUT_CASES = [
    (b"\x1f\x8b", ".gz", gzip),
    (b"BZh", ".bz2", bz2),
    (b"\xfd7zXZ\x00", ".xz", lzma),
]


@pytest.mark.parametrize(("magic", "extension", "module"), _MAGIC_CUT_CASES)
def test_detect_every_magic_cut_point(magic, extension, module):
    """Split each stream at every byte offset of its magic and across it.

    The source serves the bytes before the cut in one chunk and everything
    after it one byte at a time, so detection observes the signature split at
    every possible boundary.  The first line and the full payload must both
    survive intact.
    """
    payload = multiline
    blob = module.compress(payload)
    first_line = payload.split(b"\n", 1)[0] + b"\n"
    for cut in range(0, len(magic) + 2):
        source = ChunkedReader(blob[:cut], blob[cut:])
        wrapped = smart_open.compression.compression_wrapper(
            source, "rb", "detect", filename="extensionless-download"
        )
        assert wrapped.readline() == first_line, (extension, cut)
        assert wrapped.read() == payload[len(first_line) :], (extension, cut)
        wrapped.close()
        # and once more with line-oriented reads all the way through
        source = ChunkedReader(blob[:cut], blob[cut:])
        wrapped = smart_open.compression.compression_wrapper(source, "rb", "detect", filename="stream")
        lines = []
        while True:
            line = wrapped.readline()
            if not line:
                break
            lines.append(line)
        assert b"".join(lines) == payload, (extension, cut)
        wrapped.close()

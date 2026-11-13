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

import smart_open
import smart_open.compression

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd
import lz4.frame

plain = "доброе утро планета!".encode()


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


DETECT = smart_open.compression.DETECT_FROM_CONTENT
MAGIC = {
    ".gz": b"\x1f\x8b",
    ".bz2": b"BZh",
    ".xz": b"\xfd7zXZ\x00",
}
CODECS = {
    ".gz": gzip,
    ".bz2": bz2,
    ".xz": lzma,
}
detect_payload = (b"first line\n" + plain + b"\nlast line\n") * 3


class OneByteStream:
    """Non-seekable source that hands out exactly one byte per read()."""

    def __init__(self, data):
        self._buf = io.BytesIO(data)
        self.closed = False
        self.name = "one-byte-stream"

    def readable(self):
        return True

    def seekable(self):
        return False

    def read(self, size=-1):
        if self.closed:
            msg = "read from closed file"
            raise ValueError(msg)
        return self._buf.read(1)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("ext", [".gz", ".bz2", ".xz"])
def test_detect_on_one_byte_stream(ext):
    """Every magic cut point is crossed one byte at a time; full body survives."""
    data = CODECS[ext].compress(detect_payload)
    source = OneByteStream(data)
    wrapped = smart_open.compression.compression_wrapper(source, "rb", DETECT, "no-extension")
    assert wrapped.readline() == b"first line\n"
    assert wrapped.read() == detect_payload[len(b"first line\n") :]


@pytest.mark.parametrize("ext", [".gz", ".bz2", ".xz"])
def test_detect_on_seekable_stream(ext):
    """Seekable sources are rewound and decoded without a replay wrapper."""
    data = CODECS[ext].compress(detect_payload)
    source = io.BytesIO(data)
    wrapped = smart_open.compression.compression_wrapper(source, "rb", DETECT, None)
    assert wrapped.read() == detect_payload


def test_detect_plain_data():
    """Ordinary data, including magic-like short prefixes, is returned as-is."""
    for payload in (
        b"",
        b"plain text without any magic",
        b"\x1f" + b" not really gzip",
        b"BZ" + b" not really bz2",
        b"\xfd7" + b" not really xz",
    ):
        source = OneByteStream(payload)
        wrapped = smart_open.compression.compression_wrapper(
            source, "rb", DETECT, None, detect_truncated="plain"
        )
        assert wrapped.read() == payload


@pytest.mark.parametrize(("ext", "magic"), list(MAGIC.items()))
def test_detect_truncated_signature_every_cut(ext, magic):
    """A stream ending inside a magic is a truncated container, not plain data."""
    for cut in range(1, len(magic)):
        source = io.BytesIO(magic[:cut])
        with pytest.raises(EOFError):
            smart_open.compression.compression_wrapper(source, "rb", DETECT, None)

        source = OneByteStream(magic[:cut])
        with pytest.raises(EOFError):
            smart_open.compression.compression_wrapper(source, "rb", DETECT, None)

        # opt-in: treat the unfinished signature as plain data
        source = io.BytesIO(magic[:cut])
        wrapped = smart_open.compression.compression_wrapper(
            source, "rb", DETECT, None, detect_truncated="plain"
        )
        assert wrapped.read() == magic[:cut]


def test_detect_extension_conflict_warns_and_trusts_content(caplog):
    """Default conflict policy: warn, but decode using the detected content."""
    data = gzip.compress(detect_payload)
    source = io.BytesIO(data)
    with caplog.at_level(logging.WARNING, logger="smart_open.compression"):
        wrapped = smart_open.compression.compression_wrapper(source, "rb", DETECT, "actually.xz")
    assert wrapped.read() == detect_payload
    assert "mismatch" in caplog.text


def test_detect_extension_conflict_strict_raises():
    """Strict conflict policy raises and leaves the source open."""
    data = gzip.compress(detect_payload)
    source = io.BytesIO(data)
    with pytest.raises(ValueError, match="mismatch"):
        smart_open.compression.compression_wrapper(
            source, "rb", DETECT, "actually.xz", detect_conflict="strict"
        )
    assert not source.closed


def test_detect_strict_conflict_non_seekable_keeps_source_open():
    source = OneByteStream(gzip.compress(detect_payload))
    with pytest.raises(ValueError, match="mismatch"):
        smart_open.compression.compression_wrapper(
            source, "rb", DETECT, "actually.xz", detect_conflict="strict"
        )
    assert not source.closed


def test_detect_explicit_codec_takes_precedence():
    """An explicit codec bypasses detection entirely."""
    source = io.BytesIO(gzip.compress(detect_payload))
    wrapped = smart_open.compression.compression_wrapper(source, "rb", ".gz", "weird")
    assert wrapped.read() == detect_payload


def test_detect_rejects_invalid_policies():
    source = io.BytesIO(plain)
    with pytest.raises(ValueError, match="detect_conflict"):
        smart_open.compression.compression_wrapper(source, "rb", DETECT, None, detect_conflict="nope")
    source = io.BytesIO(plain)
    with pytest.raises(ValueError, match="detect_truncated"):
        smart_open.compression.compression_wrapper(source, "rb", DETECT, None, detect_truncated="nope")


def test_detect_rejects_plus_mode():
    with pytest.raises(ValueError, match="unsupported"):
        smart_open.compression.compression_wrapper(io.BytesIO(plain), "rb+", DETECT, None)


def test_detect_write_falls_back_to_extension():
    """There is nothing to sniff when writing; keep extension inference."""
    source = io.BytesIO()
    source.name = "out.gz"
    wrapped = smart_open.compression.compression_wrapper(source, "wb", DETECT, "out.gz")
    wrapped.write(plain)
    wrapped.close()
    source.seek(0)
    assert gzip.decompress(source.read()) == plain


def test_detect_codec_failure_keeps_source_open():
    """A valid magic followed by garbage fails in the codec; ownership stays with the caller."""
    broken = MAGIC[".gz"] + b"\x00" * 32
    for source in (io.BytesIO(broken), OneByteStream(broken)):
        wrapped = smart_open.compression.compression_wrapper(source, "rb", DETECT, None)
        with pytest.raises(OSError):  # gzip.BadGzipFile is an OSError subclass
            wrapped.read()
        assert not source.closed


def test_detect_early_close():
    """Closing after a partial read must not raise or close the source twice."""
    source = OneByteStream(gzip.compress(detect_payload))
    wrapped = smart_open.compression.compression_wrapper(source, "rb", DETECT, None)
    assert wrapped.read(7) == detect_payload[:7]
    wrapped.close()
    wrapped.close()  # idempotent
    # the codec never owned the source, mirroring the extension-inference path
    assert not source.closed
    source.close()


def test_detect_registered_as_supported_type():
    assert DETECT in smart_open.compression.get_supported_compression_types()


def test_open_invalid_compression_still_rejected(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"whatever")
    with pytest.raises(ValueError, match="invalid compression type"):
        smart_open.open(str(path), "rb", compression="not-a-codec")


@pytest.mark.parametrize("ext", [".gz", ".bz2", ".xz"])
def test_open_detect_local_file_without_extension(tmp_path, ext):
    """End-to-end: a local file with no extension is recognized from content."""
    path = tmp_path / "blob"
    path.write_bytes(CODECS[ext].compress(detect_payload))
    with smart_open.open(str(path), "rb", compression=DETECT) as fin:
        assert fin.readline() == b"first line\n"
        assert fin.read() == detect_payload[len(b"first line\n") :]

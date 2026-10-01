#
# Copyright (C) 2020 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
import bz2
import gzip
import io
import lzma
import os
import struct
import sys
import zlib

import pytest

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd
import lz4.frame

import smart_open
import smart_open.compression
import smart_open.utils

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


#
# Reproducible gzip: identical input must produce identical output bytes,
# regardless of the file name, the wall-clock time, or the write chunking.
#

PAYLOAD = (b"the quick brown fox jumps over the lazy dog\n" * 1024) + bytes(range(256)) * 64


class FakeS3Writer(io.RawIOBase):
    """A minimal fake of smart_open.s3.SinglepartWriter.

    Buffers written bytes in memory.  ``close`` commits the buffered bytes as
    the remote object; ``terminate`` discards them.  Mirroring the real S3
    writers, ``__exit__`` terminates the upload when the with-block raises.
    """

    def __init__(self, bucket="bucket", key="key"):
        super().__init__()
        self.name = f"s3://{bucket}/{key}"
        self._buf = io.BytesIO()
        self.committed = None
        self.terminated = False

    def writable(self):
        return True

    def write(self, b):
        return self._buf.write(b)

    def close(self):
        if not self.closed:
            if not self.terminated:
                self.committed = self._buf.getvalue()
            self._buf.close()
        super().close()

    def terminate(self):
        """Discard the buffered bytes without committing the object."""
        if not self.closed:
            self.terminated = True
            self._buf.close()
            super().close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            self.terminate()


def _wrap_remote(writer, mode="wb", **compression_kwargs):
    """Wrap a writer the same way smart_open.open wraps remote binary streams."""
    decompressed = smart_open.compression.compression_wrapper(
        writer, mode, ".gz", filename=writer.name, compression_kwargs=compression_kwargs
    )
    return smart_open.utils.FileLikeProxy(decompressed, writer)


def _write_local(path, payload, mode="wb", encoding=None, newline=None, **compression_kwargs):
    with smart_open.open(
        str(path), mode, encoding=encoding, newline=newline, compression_kwargs=compression_kwargs
    ) as fout:
        fout.write(payload)
    with open(path, "rb") as fin:
        return fin.read()


def test_gzip_reproducible_fixes_header_metadata(tmp_path):
    """The gzip header carries no filename and no timestamp."""
    data = _write_local(tmp_path / "some-file.gz", PAYLOAD, reproducible=True)
    assert data[:3] == b"\x1f\x8b\x08"  # magic + deflate
    assert data[3] == 0  # FLG: no FNAME, FCOMMENT, FHCRC, FEXTRA
    assert data[4:8] == b"\x00\x00\x00\x00"  # MTIME: zeroed
    assert gzip.decompress(data) == PAYLOAD


def test_gzip_reproducible_same_input_same_output(tmp_path):
    """Identical input written twice (to different names) has identical digests."""
    first = _write_local(tmp_path / "first.gz", PAYLOAD, reproducible=True)
    second = _write_local(tmp_path / "second.gz", PAYLOAD, reproducible=True)
    assert first == second


def test_gzip_normal_mode_keeps_current_defaults(tmp_path):
    """Without the reproducible option, the header still varies (name, mtime)."""
    first = _write_local(tmp_path / "first.gz", PAYLOAD)
    second = _write_local(tmp_path / "second.gz", PAYLOAD)
    assert first != second  # different FNAME embedded
    assert first[3] & gzip.FNAME
    assert int.from_bytes(first[4:8], "little") != 0  # wall-clock MTIME


@pytest.mark.parametrize("chunk_sizes", [(1, 7, 4096), (len(PAYLOAD), 3, 1)])
def test_gzip_reproducible_chunking_invariant(tmp_path, chunk_sizes):
    """Splitting the same input into different write chunks changes nothing."""
    outputs = []
    for index, chunk_size in enumerate(chunk_sizes):
        path = tmp_path / f"chunked-{index}.gz"
        with smart_open.open(str(path), "wb", compression_kwargs={"reproducible": True}) as fout:
            for offset in range(0, len(PAYLOAD), chunk_size):
                fout.write(PAYLOAD[offset : offset + chunk_size])
        with open(path, "rb") as fin:
            outputs.append(fin.read())
    assert outputs[0] == outputs[1] == outputs[2]


def test_gzip_reproducible_compresslevel_is_explicit(tmp_path):
    """The level defaults to 9, is overridable, and affects the output bytes."""
    default = _write_local(tmp_path / "default.gz", PAYLOAD, reproducible=True)
    nine = _write_local(tmp_path / "nine.gz", PAYLOAD, reproducible=True, compresslevel=9)
    one = _write_local(tmp_path / "one.gz", PAYLOAD, reproducible=True, compresslevel=1)
    assert default == nine
    assert default != one
    assert gzip.decompress(one) == PAYLOAD


def test_gzip_reproducible_text_mode_stable(tmp_path, monkeypatch):
    """Text mode defaults to UTF-8 and '\\n' newlines, not locale/platform."""
    monkeypatch.setattr("smart_open.smart_open_lib.DEFAULT_ENCODING", "cp1251")
    text = "доброе утро\nпланета!\n"

    reproducible = _write_local(tmp_path / "text.gz", text, mode="w", reproducible=True)
    explicit = _write_local(
        tmp_path / "explicit.gz", text, mode="w", encoding="utf-8", newline="\n", reproducible=True
    )
    assert reproducible == explicit
    assert gzip.decompress(reproducible) == text.encode("utf-8")

    normal = _write_local(tmp_path / "normal.gz", text, mode="w")
    assert gzip.decompress(normal) == text.encode("cp1251")


def test_gzip_reproducible_read_unaffected(tmp_path):
    """The option is accepted (and ignored) when reading."""
    _write_local(tmp_path / "data.gz", PAYLOAD, reproducible=True)
    with smart_open.open(str(tmp_path / "data.gz"), "rb", compression_kwargs={"reproducible": True}) as fin:
        assert fin.read() == PAYLOAD


def test_gzip_reproducible_local_and_remote_bytes_identical(tmp_path):
    """A local file and a remote (fake S3) object hold byte-identical content."""
    local = _write_local(tmp_path / "local.gz", PAYLOAD, reproducible=True)

    remote = FakeS3Writer()
    with _wrap_remote(remote, reproducible=True) as fout:
        fout.write(PAYLOAD)

    assert remote.committed is not None
    assert remote.committed == local
    assert gzip.decompress(remote.committed) == PAYLOAD


def test_gzip_reproducible_close_finalizes_trailer_and_upload():
    """Nothing is uploaded before close; close writes the trailer and commits."""
    remote = FakeS3Writer()
    fout = _wrap_remote(remote, reproducible=True)
    fout.write(PAYLOAD)
    assert remote.committed is None

    fout.close()
    assert remote.committed is not None
    assert remote.committed[:2] == b"\x1f\x8b"
    crc, isize = struct.unpack("<II", remote.committed[-8:])
    assert crc == zlib.crc32(PAYLOAD)
    assert isize == len(PAYLOAD) % (2**32)
    assert gzip.decompress(remote.committed) == PAYLOAD


@pytest.mark.parametrize("mode", ["wb", "w"])
def test_gzip_reproducible_exception_does_not_commit(mode):
    """An abnormal exit must not commit an incomplete gzip stream as an object."""
    remote = FakeS3Writer()
    if "b" in mode:
        fout = _wrap_remote(remote, reproducible=True)
        payload = PAYLOAD
    else:
        binary = _wrap_remote(remote, reproducible=True)
        decoded = smart_open.smart_open_lib._encoding_wrapper(binary, mode, encoding="utf-8", newline="\n")
        fout = smart_open.utils.FileLikeProxy(decoded, remote)
        payload = "the quick brown fox jumps over the lazy dog\n" * 1024

    with pytest.raises(RuntimeError, match="boom"):
        with fout:
            fout.write(payload)
            raise RuntimeError("boom")

    assert remote.committed is None
    assert remote.terminated

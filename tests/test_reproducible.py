#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Tests for smart_open.open(..., reproducible=True)."""

import gzip
import hashlib
import io
import os

import boto3
import botocore.exceptions
import pytest

# See https://github.com/piskvorky/smart_open/issues/800
# This supports moto 4 & 5 until v4 is no longer used by distros.
try:
    from moto import mock_s3
except ImportError:
    from moto import mock_aws as mock_s3

import smart_open

#
# This is a hack to keep moto happy
# See https://github.com/spulec/moto/issues/1941
#
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"  # fixture password literal

BUCKET_NAME = "test-smartopen-reproducible"
KEY_NAME = "test-key.gz"

TEXT = "доброе утро планета!\nsecond line\n"
DATA = TEXT.encode("utf-8") * 500


@pytest.fixture
def s3():
    with mock_s3():
        boto3.resource("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET_NAME)
        yield boto3.client("s3", region_name="us-east-1")


def get_object_bytes(client, key=KEY_NAME):
    return client.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()


def object_exists(client, key=KEY_NAME):
    try:
        client.head_object(Bucket=BUCKET_NAME, Key=key)
    except botocore.exceptions.ClientError:
        return False
    return True


def parse_gzip_header(data):
    """Return the (flags, mtime, filename) of a gzip stream."""
    assert data[:3] == b"\x1f\x8b\x08"
    flags = data[3]
    mtime = int.from_bytes(data[4:8], "little")
    filename = None
    if flags & 0x08:
        end = data.index(b"\x00", 10)
        filename = data[10:end].decode("latin-1")
    return flags, mtime, filename


def write_local(path, chunks, **kwargs):
    with smart_open.open(path, "wb", **kwargs) as fout:
        for chunk in chunks:
            fout.write(chunk)
    with open(path, "rb") as fin:
        return fin.read()


def test_local_write_is_reproducible(tmp_path):
    """Writing the same content twice yields byte-identical files."""
    first = write_local(str(tmp_path / "first.gz"), [DATA], reproducible=True)
    second = write_local(str(tmp_path / "second.gz"), [DATA], reproducible=True)
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
    assert gzip.decompress(first) == DATA


def test_local_write_chunking_does_not_matter(tmp_path):
    """Splitting the same input into different write chunks yields identical bytes."""
    whole = write_local(str(tmp_path / "whole.gz"), [DATA], reproducible=True)
    chunked = write_local(
        str(tmp_path / "chunked.gz"),
        [DATA[i : i + 7] for i in range(0, len(DATA), 7)],
        reproducible=True,
    )
    assert whole == chunked


def test_local_write_text_mode_is_stable(tmp_path):
    """Text mode uses utf-8 and \\n regardless of locale and platform."""
    path = str(tmp_path / "text.gz")
    with smart_open.open(path, "wt", reproducible=True) as fout:
        fout.write(TEXT)
    with open(path, "rb") as fin:
        assert gzip.decompress(fin.read()) == TEXT.encode("utf-8")
    with smart_open.open(path, "rt", reproducible=True) as fin:
        assert fin.read() == TEXT


def test_local_write_compresslevel_is_explicit(tmp_path):
    """The compression level is fixed by default and overridable."""
    default = write_local(str(tmp_path / "default.gz"), [DATA], reproducible=True)
    assert default[8] == 2  # XFL: 2 => maximum compression (level 9)
    fast = write_local(
        str(tmp_path / "fast.gz"),
        [DATA],
        reproducible=True,
        compression_kwargs={"compresslevel": 1},
    )
    assert fast != default
    assert gzip.decompress(fast) == DATA


def test_default_mode_keeps_metadata(tmp_path):
    """Without reproducible=True, gzip keeps its filename/timestamp metadata."""
    path = str(tmp_path / "plain.gz")
    data = write_local(path, [DATA])
    flags, mtime, filename = parse_gzip_header(data)
    assert flags & 0x08  # FNAME is set
    assert filename == "plain"  # gzip strips the .gz suffix from FNAME
    assert mtime != 0


def test_reproducible_mode_pins_metadata(tmp_path):
    """With reproducible=True, the gzip header carries no filename/timestamp."""
    data = write_local(str(tmp_path / "pinned.gz"), [DATA], reproducible=True)
    flags, mtime, filename = parse_gzip_header(data)
    assert not flags & 0x08  # FNAME is not set
    assert filename is None
    assert mtime == 0


def test_s3_write_matches_local_bytes(tmp_path, s3):
    """A fake-S3 write produces the exact same bytes as a local write."""
    local = write_local(str(tmp_path / "local.gz"), [DATA], reproducible=True)
    uri = f"s3://{BUCKET_NAME}/{KEY_NAME}"
    with smart_open.open(uri, "wb", reproducible=True) as fout:
        for i in range(0, len(DATA), 13):
            fout.write(DATA[i : i + 13])
    remote = get_object_bytes(s3)
    assert remote == local
    assert gzip.decompress(remote) == DATA


def test_s3_write_completes_upload_on_close(s3):
    """Closing finalizes the gzip trailer and completes the remote upload."""
    uri = f"s3://{BUCKET_NAME}/{KEY_NAME}"
    with smart_open.open(uri, "wb", reproducible=True) as fout:
        fout.write(DATA)
    remote = get_object_bytes(s3)
    #
    # A missing gzip trailer (CRC32 + ISIZE) would make decompression fail.
    #
    assert gzip.decompress(remote) == DATA
    crc32, isize = remote[-8:-4], int.from_bytes(remote[-4:], "little")
    assert isize == len(DATA) % (2**32)
    assert len(crc32) == 4


@pytest.mark.parametrize("transport_params", [{}, {"multipart_upload": False}])
def test_s3_abnormal_exit_commits_nothing(s3, transport_params):
    """An exception must not commit an incomplete gzip stream as an object."""
    uri = f"s3://{BUCKET_NAME}/{KEY_NAME}"
    with pytest.raises(RuntimeError, match="boom"):
        with smart_open.open(uri, "wb", reproducible=True, transport_params=transport_params) as fout:
            fout.write(DATA)
            raise RuntimeError("boom")
    assert not object_exists(s3)
    assert s3.list_objects_v2(Bucket=BUCKET_NAME).get("KeyCount", 0) == 0

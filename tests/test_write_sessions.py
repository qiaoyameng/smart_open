#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#

import gzip
import os
import tempfile

import pytest

import smart_open


class FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.uploads = {}
        self._next_upload = 1
        self.complete_calls = 0
        self.abort_calls = 0

    def create_multipart_upload(self, Bucket, Key):
        upload_id = f"upload-{self._next_upload}"
        self._next_upload += 1
        self.uploads[upload_id] = {
            "Bucket": Bucket,
            "Key": Key,
            "Parts": [],
            "aborted": False,
        }
        return {"UploadId": upload_id}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        Body.seek(0)
        upload = self.uploads[UploadId]
        upload["Parts"].append((PartNumber, Body.read()))
        return {"ETag": f'"etag-{UploadId}-{PartNumber}"'}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        upload = self.uploads[UploadId]
        assert upload["Bucket"] == Bucket
        assert upload["Key"] == Key
        assert len(MultipartUpload["Parts"]) == len(upload["Parts"])
        self.objects[(Bucket, Key)] = b"".join(part for _, part in upload["Parts"])
        del self.uploads[UploadId]
        self.complete_calls += 1
        return {}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        upload = self.uploads.pop(UploadId)
        upload["aborted"] = True
        self.abort_calls += 1
        return {}

    def put_object(self, Bucket, Key, Body):
        body = Body.read() if hasattr(Body, "read") else Body
        self.objects[(Bucket, Key)] = body
        return {}


@pytest.fixture
def small_s3_parts(monkeypatch):
    monkeypatch.setattr(smart_open.s3, "MIN_PART_SIZE", 10)


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp(dir=os.getcwd())
    try:
        yield path
    finally:
        for name in os.listdir(path):
            os.unlink(os.path.join(path, name))
        os.rmdir(path)


def test_s3_multipart_session_commits_only_after_commit(small_s3_parts):
    client = FakeS3Client()
    key = ("bucket", "object.gz")
    payload = b"visible only after commit " * 8

    session = smart_open.open(
        "s3://bucket/object.gz",
        "wb",
        transport_params={"client": client, "min_part_size": 10},
        write_session=True,
    )
    session.write(payload)

    assert key not in client.objects
    assert client.uploads
    assert client.complete_calls == 0

    session.finish()

    assert key not in client.objects
    assert client.uploads
    assert client.complete_calls == 0
    session.finish()

    session.commit()

    assert key in client.objects
    assert gzip.decompress(client.objects[key]) == payload
    assert client.complete_calls == 1
    session.commit()
    assert client.complete_calls == 1
    with pytest.raises(ValueError, match="committed"):
        session.finish()
    with pytest.raises(ValueError, match="committed"):
        session.abort()


def test_s3_multipart_session_abort_after_finish(small_s3_parts):
    client = FakeS3Client()
    key = ("bucket", "object.gz")
    client.objects[key] = b"previous object"

    session = smart_open.open(
        "s3://bucket/object.gz",
        "wb",
        transport_params={"client": client, "min_part_size": 10},
        write_session=True,
    )
    session.write(b"new object that must be discarded" * 4)
    session.finish()
    session.abort()

    assert client.objects[key] == b"previous object"
    assert not client.uploads
    assert client.complete_calls == 0
    assert client.abort_calls == 1
    session.abort()
    assert client.abort_calls == 1
    with pytest.raises(ValueError, match="aborted"):
        session.commit()


def test_s3_rejects_single_part_write_session():
    client = FakeS3Client()

    with pytest.raises(ValueError, match="multipart_upload"):
        smart_open.open(
            "s3://bucket/object",
            "wb",
            transport_params={"client": client, "multipart_upload": False},
            write_session=True,
        )


def test_s3_multipart_empty_session_commits_empty_object(small_s3_parts):
    client = FakeS3Client()
    key = ("bucket", "empty")

    session = smart_open.open(
        "s3://bucket/empty",
        "wb",
        transport_params={"client": client, "min_part_size": 10},
        write_session=True,
    )
    session.finish()

    assert key not in client.objects
    assert client.uploads

    session.commit()

    assert client.objects[key] == b""
    assert client.complete_calls == 0


def test_s3_multipart_text_session_flushes_encoder(small_s3_parts):
    client = FakeS3Client()

    session = smart_open.open(
        "s3://bucket/text",
        "w",
        encoding="utf-8",
        transport_params={"client": client, "min_part_size": 10},
        write_session=True,
    )
    session.write("enc\u00f6ded \u00e9")
    session.finish()
    session.commit()

    assert client.objects[("bucket", "text")] == "enc\u00f6ded \u00e9".encode("utf-8")


def test_s3_multipart_commit_finishes_pending_layers(small_s3_parts):
    client = FakeS3Client()

    session = smart_open.open(
        "s3://bucket/auto-commit.gz",
        "wb",
        transport_params={"client": client, "min_part_size": 10},
        write_session=True,
    )
    session.write(b"pending compressor trailer")
    session.commit()

    assert gzip.decompress(client.objects[("bucket", "auto-commit.gz")]) == b"pending compressor trailer"


def test_local_session_stages_finishes_then_commits(temp_dir):
    target = os.path.join(temp_dir, "object.gz")
    payload = b"local staged compressed payload" * 5

    session = smart_open.open(target, "wb", write_session=True)
    session.write(payload)

    assert not os.path.exists(target)
    temporary_files = [name for name in os.listdir(temp_dir) if name.endswith(".tmp")]
    assert temporary_files

    session.finish()

    assert not os.path.exists(target)
    assert os.listdir(temp_dir) == temporary_files
    session.finish()

    session.commit()

    assert os.path.exists(target)
    assert gzip.decompress(open(target, "rb").read()) == payload  # noqa: PTH123, SIM115
    assert not [name for name in os.listdir(temp_dir) if name.endswith(".tmp")]
    session.commit()
    with pytest.raises(ValueError, match="committed"):
        session.abort()


def test_local_session_abort_removes_temp_and_preserves_destination(temp_dir):
    target = os.path.join(temp_dir, "object.txt")
    with open(target, "wb") as existing:  # noqa: PTH123
        existing.write(b"old")

    session = smart_open.open(
        target,
        "w",
        encoding="utf-8",
        write_session=True,
    )
    session.write("new value with unicode \u00e9")
    session.finish()
    session.abort()

    assert open(target, "rb").read() == b"old"  # noqa: PTH123, SIM115
    assert not [name for name in os.listdir(temp_dir) if name.endswith(".tmp")]
    session.abort()
    with pytest.raises(ValueError, match="aborted"):
        session.commit()


def test_write_sessions_are_opted_in_and_capability_checked():
    with pytest.raises(ValueError, match="write mode"):
        smart_open.open("s3://bucket/object", "rb", write_session=True)

    with pytest.raises(NotImplementedError, match="http"):
        smart_open.open("http://example.invalid/object", "wb", write_session=True)

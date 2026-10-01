import gzip
import io

import botocore.exceptions
import pytest

import smart_open


def test_local_session_visibility_and_compression(tmp_path):
    path = tmp_path / "result.gz"

    with smart_open.open_write_session(path) as session:
        session.write(b"payload")
        assert not path.exists()

        session.finish()
        assert not path.exists()
        with pytest.raises(ValueError, match="finished"):
            session.write(b"more")

        session.commit()

    assert gzip.decompress(path.read_bytes()) == b"payload"


def test_local_session_abort_retains_existing_file(tmp_path):
    path = tmp_path / "result.txt"
    path.write_bytes(b"existing")

    session = smart_open.open_write_session(path)
    session.write(b"replacement")
    session.finish()
    session.abort()

    assert path.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".result.txt.tmp-*"))

    with pytest.raises(ValueError, match="aborted"):
        session.commit()
    session.abort()
    session.finish()


def test_local_session_close_aborts_uncommitted_data(tmp_path):
    path = tmp_path / "result.txt"

    session = smart_open.open_write_session(path)
    session.write(b"payload")
    session.close()

    assert not path.exists()
    session.close()


def test_local_session_context_without_commit_aborts(tmp_path):
    path = tmp_path / "result.txt"

    with smart_open.open_write_session(path) as session:
        session.write(b"payload")

    assert not path.exists()


def test_local_text_session_flushes_encoder(tmp_path):
    path = tmp_path / "result.txt"

    with smart_open.open_write_session(path, "w", encoding="utf-8") as session:
        session.write("héllo")
        session.finish()
        assert not path.exists()
        session.commit()

    assert path.read_text(encoding="utf-8") == "héllo"


def test_local_session_commits_through_symlink(tmp_path):
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_bytes(b"existing")
    link.symlink_to(target)

    session = smart_open.open_write_session(link)
    session.write(b"updated")
    session.commit()

    assert link.is_symlink()
    assert link.read_bytes() == b"updated"


class StubS3Client:
    def __init__(self):
        self.objects = {}
        self.uploads = {}
        self._next_upload = 0

    def _key(self, bucket, key):
        return bucket, key

    def create_multipart_upload(self, Bucket, Key, **kwargs):
        upload_id = f"upload-{self._next_upload}"
        self._next_upload += 1
        self.uploads[upload_id] = {"bucket": Bucket, "key": Key, "parts": {}}
        return {"UploadId": upload_id}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        upload = self.uploads[UploadId]
        upload["parts"][PartNumber] = Body.read()
        return {"ETag": f'"{PartNumber}"'}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        try:
            upload = self.uploads.pop(UploadId)
        except KeyError as exc:
            raise botocore.exceptions.ClientError(
                {"Error": {"Code": "NoSuchUpload", "Message": "upload not found"}},
                "CompleteMultipartUpload",
            ) from exc

        parts = [upload["parts"][part["PartNumber"]] for part in MultipartUpload["Parts"]]
        self.objects[self._key(Bucket, Key)] = b"".join(parts)
        return {}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.uploads.pop(UploadId, None)
        return {}

    def put_object(self, Bucket, Key, Body):
        self.objects[self._key(Bucket, Key)] = Body.read()
        return {}


def test_s3_multipart_session_visibility(monkeypatch):
    monkeypatch.setattr(smart_open.s3, "MIN_PART_SIZE", 1)
    client = StubS3Client()
    transport_params = {"client": client, "min_part_size": 5}

    session = smart_open.open_write_session("s3://bucket/key", transport_params=transport_params)
    assert ("bucket", "key") not in client.objects

    session.write(b"abc")
    assert ("bucket", "key") not in client.objects
    assert len(client.uploads) == 1

    session.write(b"def")
    assert ("bucket", "key") not in client.objects
    upload = next(iter(client.uploads.values()))
    assert len(upload["parts"]) == 1

    session.finish()
    assert ("bucket", "key") not in client.objects
    assert len(upload["parts"]) == 2

    session.commit()
    assert client.objects[("bucket", "key")] == b"abcdef"
    assert not client.uploads
    session.commit()
    session.finish()


def test_s3_session_abort_after_finish(monkeypatch):
    monkeypatch.setattr(smart_open.s3, "MIN_PART_SIZE", 1)
    client = StubS3Client()
    client.objects[("bucket", "key")] = b"existing"

    session = smart_open.open_write_session(
        "s3://bucket/key",
        transport_params={"client": client, "min_part_size": 5},
    )
    session.write(b"abcdef")
    session.finish()
    session.abort()

    assert client.objects[("bucket", "key")] == b"existing"
    assert not client.uploads
    with pytest.raises(ValueError, match="aborted"):
        session.commit()
    session.abort()


def test_s3_compressed_text_session_flushes_compressor(monkeypatch):
    monkeypatch.setattr(smart_open.s3, "MIN_PART_SIZE", 1)
    client = StubS3Client()

    with smart_open.open_write_session(
        "s3://bucket/key.gz",
        "w",
        transport_params={"client": client, "min_part_size": 1024},
    ) as session:
        session.write("compressed text")
        session.finish()
        assert ("bucket", "key.gz") not in client.objects
        session.commit()

    assert gzip.decompress(client.objects[("bucket", "key.gz")]) == b"compressed text"


def test_s3_singlepart_session_rejected_at_open():
    client = StubS3Client()

    with pytest.raises(io.UnsupportedOperation, match="multipart"):
        smart_open.open_write_session(
            "s3://bucket/key",
            transport_params={"client": client, "multipart_upload": False},
        )

    assert not client.uploads


def test_unsupported_transport_session_rejected_at_open():
    with pytest.raises(io.UnsupportedOperation, match="does not support"):
        smart_open.open_write_session("http://example.com/object")

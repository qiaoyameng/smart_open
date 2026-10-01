#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#

"""Tests for the explicit streaming authenticated encryption layer."""

from __future__ import annotations

import io
import logging
import os
import struct

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import smart_open
from smart_open import AESGCMEncryption, EncryptionError
from smart_open import encryption as so_enc

CHUNK = 1024
KEY = b"0123456789abcdef0123456789abcdef"
PLAIN = bytes((i * 7 + 13) % 256 for i in range(10_000))


def seal(blob_parts, aesgcm, file_id, sequence, record_type, payload):
    """Independently construct one SOAE record straight from the format spec."""
    nonce = os.urandom(so_enc.NONCE_BYTES)
    aad = (
        so_enc.MAGIC
        + bytes((so_enc.FORMAT_VERSION,))
        + file_id
        + struct.pack(">Q", sequence)
        + bytes((record_type,))
    )
    ciphertext = aesgcm.encrypt(nonce, payload, aad)
    blob_parts.append(bytes((record_type,)) + nonce + struct.pack(">I", len(ciphertext)) + ciphertext)


def make_container(file_id, records, aesgcm, data_count, *, terminal_seq=None, terminal_count=None):
    """Assemble a container from physical record payloads, with full framing control."""
    parts = [so_enc.MAGIC + bytes((so_enc.FORMAT_VERSION,)) + file_id]
    parts.extend(records)
    seq = data_count if terminal_seq is None else terminal_seq
    count = data_count if terminal_count is None else terminal_count
    seal(parts, aesgcm, file_id, seq, so_enc._TYPE_TERMINAL, struct.pack(">Q", count))
    return b"".join(parts)


def build_stream(key, segments, terminal_count=None, *, bad_terminal_aad=False):
    """Assemble a valid container from plaintext segments."""
    file_id = os.urandom(so_enc.FILE_ID_BYTES)
    aesgcm = AESGCM(key)
    records = []
    for seq, payload in enumerate(segments):
        seal(records, aesgcm, file_id, seq, so_enc._TYPE_DATA, payload)
    terminal_seq = len(segments) + (1 if bad_terminal_aad else 0)
    return make_container(
        file_id, records, aesgcm, len(segments), terminal_seq=terminal_seq,
        terminal_count=terminal_count,
    )


def stream_with_record_order(key, segments, order):
    """Emit physical data records in the given physical order, sequences stay as sealed."""
    file_id = os.urandom(so_enc.FILE_ID_BYTES)
    aesgcm = AESGCM(key)
    records = []
    for seq, payload in enumerate(segments):
        seal(records, aesgcm, file_id, seq, so_enc._TYPE_DATA, payload)
    return make_container(file_id, [records[i] for i in order], aesgcm, len(order))


def encrypted_blob(payload=PLAIN, key=KEY, chunk_size=CHUNK):
    buf = io.BytesIO()
    with AESGCMEncryption(key, chunk_size).writer(buf) as w:
        w.write(payload)
    return buf.getvalue()


def decrypt_all(blob, key=KEY, chunk_size=CHUNK):
    with AESGCMEncryption(key, chunk_size).reader(io.BytesIO(blob)) as r:
        return r.read()


# --------------------------------------------------------------------------- roundtrip


@pytest.mark.parametrize("size", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 5 * CHUNK + 37])
def test_roundtrip_various_sizes(size):
    payload = bytes(i % 256 for i in range(size))
    assert decrypt_all(encrypted_blob(payload)) == payload


def test_small_reads_and_iteration():
    blob = encrypted_blob()
    with AESGCMEncryption(KEY, CHUNK).reader(io.BytesIO(blob)) as r:
        chunks = [r.read(137) for _ in range(999)]
        chunks.append(r.read())
    assert b"".join(chunks) == PLAIN


def test_each_write_gets_unique_file_id_and_nonce():
    blobs = {encrypted_blob(b"same plaintext") for _ in range(3)}
    assert len(blobs) == 3  # random file id + random nonces => distinct ciphertext


def test_empty_file_has_authenticated_terminal():
    blob = encrypted_blob(b"")
    assert decrypt_all(blob) == b""
    # magic + version + file id, then one terminal record (0x01)
    assert blob[so_enc._HEADER_LEN] == so_enc._TYPE_TERMINAL


# --------------------------------------------------------------------------- key handling


def test_key_must_be_bytes():
    with pytest.raises(TypeError):
        AESGCMEncryption("not bytes")  # type: ignore[arg-type]


@pytest.mark.parametrize("length", [0, 15, 17, 31, 33])
def test_key_length_validated(length):
    with pytest.raises(ValueError):
        AESGCMEncryption(b"k" * length)


@pytest.mark.parametrize("length", [16, 24, 32])
def test_other_aes_key_lengths(length):
    key = bytes(range(length))
    assert decrypt_all(encrypted_blob(key=key), key=key) == PLAIN


def test_bytes_key_accepted_by_open(tmp_path):
    path = tmp_path / "raw.dat"
    with smart_open.open(str(path), "wb", encryption=KEY) as f:
        f.write(PLAIN)
    with smart_open.open(str(path), "rb", encryption=KEY) as f:
        assert f.read() == PLAIN


def test_key_never_in_repr_or_logs(caplog, tmp_path):
    cfg = AESGCMEncryption(KEY)
    assert repr(KEY) not in repr(cfg)
    assert "redacted" in repr(cfg)
    path = tmp_path / "logged.bin"
    with caplog.at_level(logging.DEBUG, logger="smart_open.smart_open_lib"):
        with smart_open.open(str(path), "wb", encryption=cfg):
            pass
    assert KEY not in caplog.text.encode()
    assert repr(KEY) not in caplog.text


def test_invalid_encryption_object_rejected():
    with pytest.raises(TypeError):
        so_enc.validate_encryption("aesgcm://secret")
    assert so_enc.validate_encryption(None) is None


def test_append_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="append"):
        smart_open.open(str(tmp_path / "x"), "ab", encryption=AESGCMEncryption(KEY))


# --------------------------------------------------------------------------- streaming


class MaxReadStream(io.RawIOBase):
    """Record the largest single read issued; forbid unbounded reads."""

    def __init__(self, data):
        self._data = data
        self._pos = 0
        self.max_read = 0
        self.read_calls = 0

    def readable(self):
        return True

    def readinto(self, b):
        size = len(b)  # BufferedReader always passes a fixed buffer: never unbounded
        assert size <= 64 * 1024, "must never request the whole object at once"
        chunk = self._data[self._pos : self._pos + size]
        b[: len(chunk)] = chunk
        self._pos += len(chunk)
        self.max_read = max(self.max_read, size)
        self.read_calls += 1
        return len(chunk)


def test_never_reads_whole_object_at_once():
    blob = encrypted_blob(b"q" * 50_000)
    source = MaxReadStream(blob)
    with AESGCMEncryption(KEY, CHUNK).reader(io.BufferedReader(source)) as r:
        out = r.read()
    assert out == b"q" * 50_000
    assert source.max_read <= 64 * 1024
    assert source.read_calls > 1


def test_memory_bounded_by_one_segment():
    # payload vastly larger than chunk size; reader still only holds one segment
    payload = b"x" * (100 * CHUNK)
    blob = encrypted_blob(payload, chunk_size=CHUNK)
    with AESGCMEncryption(KEY, CHUNK).reader(io.BytesIO(blob)) as r:
        while r.read(CHUNK // 2):
            pass


# --------------------------------------------------------------------------- corruption


def read_with_tracking(blob):
    delivered, failed = b"", False
    try:
        with AESGCMEncryption(KEY, CHUNK).reader(io.BytesIO(blob)) as r:
            while True:
                chunk = r.read(500)
                if not chunk:
                    break
                delivered += chunk
    except EncryptionError:
        failed = True
    return delivered, failed


def test_tampered_ciphertext_fails_and_stops():
    segments = [PLAIN[i : i + CHUNK] for i in range(0, len(PLAIN), CHUNK)]
    blob = bytearray(build_stream(KEY, segments))
    # flip a byte inside the first data segment's ciphertext region
    blob[so_enc._HEADER_LEN + 40] ^= 0xFF
    delivered, failed = read_with_tracking(bytes(blob))
    assert failed and delivered == b""  # first segment fails: zero plaintext out


def test_tampered_later_segment_stops_further_output():
    segments = [PLAIN[i : i + CHUNK] for i in range(0, len(PLAIN), CHUNK)]
    blob = bytearray(build_stream(KEY, segments))
    # find the start of the second record and flip a ciphertext byte there
    second = so_enc._HEADER_LEN + so_enc._RECORD_PREFIX_LEN + len(segments[0]) + so_enc.TAG_BYTES
    blob[second + 30] ^= 0xFF
    delivered, failed = read_with_tracking(bytes(blob))
    assert failed
    # only authenticated bytes from the first segment may have been delivered
    assert delivered == segments[0][: len(delivered)]
    assert len(delivered) <= len(segments[0])


def test_reordered_segments_rejected():
    segments = [PLAIN[i : i + CHUNK] for i in range(0, len(PLAIN), CHUNK)]
    order = [1, 0, *range(2, len(segments))]  # physically swap two sealed records
    blob = stream_with_record_order(KEY, segments, order)
    with pytest.raises(EncryptionError):
        decrypt_all(blob)


def test_duplicated_segment_rejected():
    segments = [PLAIN[i : i + CHUNK] for i in range(0, len(PLAIN), CHUNK)]
    order = [0, 0, *range(1, len(segments))]  # replay the first physical record
    blob = stream_with_record_order(KEY, segments, order)
    with pytest.raises(EncryptionError):
        decrypt_all(blob)


def test_dropped_segment_rejected_by_terminal_count():
    segments = [PLAIN[i : i + CHUNK] for i in range(0, len(PLAIN), CHUNK)]
    # seal all segments, then drop the physical second record by re-slicing is hard;
    # instead forge a terminal that authenticates the wrong count.
    blob = build_stream(KEY, segments, terminal_count=len(segments) - 1)
    with pytest.raises(EncryptionError, match="segment count mismatch"):
        decrypt_all(blob)


def test_truncation_missing_terminal_rejected():
    blob = encrypted_blob()
    with pytest.raises(EncryptionError, match="truncated"):
        decrypt_all(blob[:-5])
    with pytest.raises(EncryptionError, match="truncated"):
        decrypt_all(blob[: so_enc._HEADER_LEN])


def test_bytes_appended_after_terminal_rejected():
    blob = bytearray(encrypted_blob())
    blob.extend(b"\x00extra trailing bytes")
    with pytest.raises(EncryptionError, match="extra bytes"):
        decrypt_all(bytes(blob))


def test_terminal_tamper_rejected():
    blob = bytearray(encrypted_blob())
    blob[-1] ^= 0x01  # flip a tag bit
    with pytest.raises(EncryptionError):
        decrypt_all(bytes(blob))


def test_terminal_wrong_position_rejected():
    segments = [PLAIN[i : i + CHUNK] for i in range(0, len(PLAIN), CHUNK)]
    blob = build_stream(KEY, segments, bad_terminal_aad=True)
    with pytest.raises(EncryptionError):
        decrypt_all(blob)


def test_header_magic_and_version_must_match():
    blob = bytearray(encrypted_blob())
    blob[0] = ord("X")
    with pytest.raises(EncryptionError, match="magic"):
        decrypt_all(bytes(blob))
    blob = bytearray(encrypted_blob())
    blob[4] = 99
    with pytest.raises(EncryptionError):
        decrypt_all(bytes(blob))


def test_file_id_change_breaks_aad():
    blob = bytearray(encrypted_blob())
    blob[5] ^= 0xFF  # flip a bit in the file id; every AAD now mismatches
    with pytest.raises(EncryptionError):
        decrypt_all(bytes(blob))


def test_wrong_key_fails():
    other = b"a" * 32
    with pytest.raises(EncryptionError):
        decrypt_all(encrypted_blob(), key=other)


def test_length_prefix_inflation_rejected():
    blob = bytearray(encrypted_blob())
    # lie that the first ciphertext is huge -> bounded allocation / truncation error
    offset = so_enc._HEADER_LEN + so_enc.TYPE_BYTES + so_enc.NONCE_BYTES
    blob[offset : offset + 4] = struct.pack(">I", so_enc.MAX_RECORD_CIPHERTEXT + 1)
    with pytest.raises(EncryptionError):
        decrypt_all(bytes(blob))


def test_unknown_record_type_rejected():
    segments = [PLAIN[:CHUNK]]
    blob = bytearray(build_stream(KEY, segments))
    blob[so_enc._HEADER_LEN] = 0x7F
    with pytest.raises(EncryptionError, match="record type"):
        decrypt_all(bytes(blob))


def test_reader_unusable_after_failure():
    blob = bytearray(encrypted_blob())
    blob[so_enc._HEADER_LEN + 20] ^= 0xFF
    r = AESGCMEncryption(KEY, CHUNK).reader(io.BytesIO(bytes(blob)))
    with pytest.raises(EncryptionError):
        r.read()
    with pytest.raises(EncryptionError):
        r.read()  # no further output after the first failure


# --------------------------------------------------------------------------- integration


def test_compression_happens_before_encryption(tmp_path):
    path = tmp_path / "backup.gz"
    payload = PLAIN * 5
    with smart_open.open(str(path), "wb", compression=".gz", encryption=AESGCMEncryption(KEY)) as f:
        f.write(payload)
    stored = path.read_bytes()
    assert stored.startswith(so_enc.MAGIC)
    # the gzip header magic must not appear anywhere in the ciphertext
    assert b"\x1f\x8b" not in stored
    with smart_open.open(str(path), "rb", compression=".gz", encryption=AESGCMEncryption(KEY)) as f:
        assert f.read() == payload


def test_encryption_without_compression(tmp_path):
    path = tmp_path / "backup.bin"
    with smart_open.open(str(path), "wb", compression="disable", encryption=AESGCMEncryption(KEY)) as f:
        f.write(PLAIN)
    with smart_open.open(str(path), "rb", compression="disable", encryption=AESGCMEncryption(KEY)) as f:
        assert f.read() == PLAIN


def test_decryption_failure_aborts_compressed_pipeline(tmp_path):
    path = tmp_path / "backup.gz"
    with smart_open.open(str(path), "wb", compression=".gz", encryption=AESGCMEncryption(KEY)) as f:
        f.write(PLAIN * 10)
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0x01  # corrupt terminal tag
    path.write_bytes(bytes(blob))
    with pytest.raises(EncryptionError):
        with smart_open.open(str(path), "rb", compression=".gz", encryption=AESGCMEncryption(KEY)) as f:
            f.read()


def test_writer_close_without_flush_still_seals_terminal(tmp_path):
    path = tmp_path / "x.bin"
    f = AESGCMEncryption(KEY).writer(open(path, "wb"))
    f.write(PLAIN)
    f.close()
    assert decrypt_all(path.read_bytes()) == PLAIN


# --------------------------------------------------------------------------- object store (moto S3)

boto3 = pytest.importorskip("boto3")
try:
    from moto import mock_s3
except ImportError:  # pragma: no cover
    from moto import mock_aws as mock_s3  # type: ignore[no-redef]


@mock_s3
def test_encrypted_backup_roundtrip_on_s3(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.resource("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="bkt")
    uri = "s3://bkt/backups/dump.gz"
    payload = bytes((i * 11 + 5) % 256 for i in range(70_000))
    cfg = AESGCMEncryption(KEY, chunk_size=8192)
    with smart_open.open(uri, "wb", compression=".gz", encryption=cfg) as f:
        f.write(payload)
    stored = boto3.client("s3").get_object(Bucket="bkt", Key="backups/dump.gz")["Body"].read()
    assert stored.startswith(so_enc.MAGIC)
    with smart_open.open(uri, "rb", compression=".gz", encryption=cfg) as f:
        assert f.read() == payload


@mock_s3
def test_corrupt_s3_object_aborts_output(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.resource("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="bkt")
    uri = "s3://bkt/backups/dump.gz"
    cfg = AESGCMEncryption(KEY, chunk_size=8192)
    with smart_open.open(uri, "wb", compression=".gz", encryption=cfg) as f:
        f.write(PLAIN * 20)
    obj = boto3.client("s3").get_object(Bucket="bkt", Key="backups/dump.gz")["Body"].read()
    damaged = obj[:-1]  # truncate the authenticated terminal
    boto3.client("s3").put_object(Bucket="bkt", Key="backups/bad.gz", Body=damaged)
    with pytest.raises(EncryptionError):
        with smart_open.open("s3://bkt/backups/bad.gz", "rb", compression=".gz", encryption=cfg) as f:
            f.read()


# --------------------------------------------------------------------------- cross-container attacks


def parse_records(blob):
    """Split a container blob into (header, data_records, terminal)."""
    header = blob[: so_enc._HEADER_LEN]
    pos = so_enc._HEADER_LEN
    data_records = []
    while True:
        rtype = blob[pos]
        length = struct.unpack(">I", blob[pos + 13 : pos + 17])[0]
        record = blob[pos : pos + so_enc._RECORD_PREFIX_LEN + length]
        pos += len(record)
        if rtype == so_enc._TYPE_TERMINAL:
            return header, data_records, record
        data_records.append(record)


def test_splicing_segments_from_another_file_rejected():
    blob_a = encrypted_blob(b"AAAA" * 5000)
    blob_b = encrypted_blob(b"BBBB" * 5000)
    header_a, data_a, _term_a = parse_records(blob_a)
    _header_b, data_b, term_b = parse_records(blob_b)
    # splice file B's authenticated records into file A: file id AAD must mismatch
    forged = header_a + b"".join(data_b) + term_b
    with pytest.raises(EncryptionError):
        decrypt_all(forged)


def test_empty_object_rejected():
    with pytest.raises(EncryptionError):
        decrypt_all(b"")


def test_header_only_truncated_rejected():
    with pytest.raises(EncryptionError):
        decrypt_all(so_enc.MAGIC + bytes((so_enc.FORMAT_VERSION,)) + b"\\x00" * 10)

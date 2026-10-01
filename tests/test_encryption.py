#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
import io
import logging
import os
import struct

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import smart_open
import smart_open.encryption
from smart_open.encryption import EncryptionError

#
# Independently constructed references to the documented on-the-wire format.
# These deliberately duplicate the format constants instead of importing them,
# so the tests validate the format itself rather than the implementation.
#
MAGIC = b"SOAESGCM"
VERSION = 1
HEADER_SIZE = 29
RECORD_HEADER_SIZE = 25
TAG_SIZE = 16
FLAG_FINAL = 0x01

KEY = bytes(range(32))
PLAINTEXT = ("доброе утро планета!\n" * 1000).encode()


def build_record(key, header, seq, nonce, payload, flags):
    """Build a single segment record from scratch, per the documented format."""
    aad = header + struct.pack(">Q", seq) + bytes([flags])
    ciphertext = AESGCM(key).encrypt(nonce, payload, aad)
    return struct.pack(">Q", seq) + nonce + bytes([flags]) + struct.pack(">I", len(ciphertext)) + ciphertext


def build_stream(key, chunks, segment_size=1024, file_id=None):
    """Build a complete encrypted stream from scratch, per the documented format."""
    file_id = file_id if file_id is not None else os.urandom(16)
    header = MAGIC + struct.pack(">B", VERSION) + struct.pack(">I", segment_size) + file_id
    records = []
    for seq, chunk in enumerate(chunks):
        flags = FLAG_FINAL if seq == len(chunks) - 1 else 0
        records.append(build_record(key, header, seq, os.urandom(12), chunk, flags))
    return header + b"".join(records)


def split_stream(data):
    """Split a stream into (header, [record, ...]) using the documented layout."""
    header, rest = data[:HEADER_SIZE], data[HEADER_SIZE:]
    records = []
    while rest:
        (length,) = struct.unpack(">I", rest[21:25])
        records.append(rest[: RECORD_HEADER_SIZE + length])
        rest = rest[RECORD_HEADER_SIZE + length :]
    return header, records


def encrypt(plaintext, key=KEY, segment_size=1024):
    """Encrypt via the library under test."""
    buf = io.BytesIO()
    with smart_open.encryption.encryption_wrapper(
        buf, "wb", "aesgcm", {"key": key, "segment_size": segment_size}
    ) as fout:
        fout.write(plaintext)
    return buf.getvalue()


def decrypt(data, key=KEY):
    """Decrypt via the library under test."""
    with smart_open.encryption.encryption_wrapper(io.BytesIO(data), "rb", "aesgcm", {"key": key}) as fin:
        return fin.read()


class TrackingReader:
    """A minimal read-only stream that counts how many bytes were consumed."""

    def __init__(self, data):
        self._stream = io.BytesIO(data)
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = self._stream.read(size)
        self.bytes_read += len(chunk)
        return chunk


def test_roundtrip_multiple_segments():
    data = encrypt(PLAINTEXT, segment_size=100)
    _, records = split_stream(data)
    assert len(records) > 10  # sanity: really segmented
    assert decrypt(data) == PLAINTEXT


def test_roundtrip_empty_stream():
    data = encrypt(b"")
    _, records = split_stream(data)
    assert len(records) == 1  # just the authenticated final record
    assert decrypt(data) == b""


def test_roundtrip_exact_segment_boundary():
    plaintext = os.urandom(3000)
    assert decrypt(encrypt(plaintext, segment_size=1000)) == plaintext


def test_ciphertext_is_randomized():
    first, second = encrypt(PLAINTEXT), encrypt(PLAINTEXT)
    assert first != second  # random file identifier and nonces
    assert decrypt(first) == decrypt(second) == PLAINTEXT


def test_fixed_test_vector():
    """A hardcoded stream, generated independently, must decrypt correctly."""
    vector = bytes.fromhex(
        "534f41455347434d0100000010000102030405060708090a0b0c0d0e0f"
        "0000000000000000000102030405060708090a0b0100000027"
        "2f67ba77aac9e27ee322e5f2c19d1d09a3a1e8469c1f7eba2311c436b7b7117e42d8dc4b37b5be"
    )
    assert decrypt(vector) == b"hello, encrypted world!"


def test_independently_constructed_stream():
    """A stream built from scratch per the documented format must decrypt."""
    chunks = [PLAINTEXT[:100], PLAINTEXT[100:5000], PLAINTEXT[5000:]]
    data = build_stream(KEY, chunks, segment_size=4096)
    assert decrypt(data) == PLAINTEXT


def test_library_output_matches_documented_layout():
    data = encrypt(PLAINTEXT, segment_size=100)
    header, records = split_stream(data)
    assert header[:8] == MAGIC
    assert header[8] == VERSION
    (segment_size,) = struct.unpack(">I", header[9:13])
    assert segment_size == 100
    for seq, record in enumerate(records):
        (record_seq,) = struct.unpack(">Q", record[:8])
        assert record_seq == seq
    assert records[-1][20] & FLAG_FINAL
    assert not any(record[20] & FLAG_FINAL for record in records[:-1])


def test_reordered_segments_detected():
    header, records = split_stream(encrypt(PLAINTEXT, segment_size=100))
    records[1], records[2] = records[2], records[1]
    with pytest.raises(EncryptionError, match="sequence"):
        decrypt(header + b"".join(records))


def test_duplicated_segment_detected():
    header, records = split_stream(encrypt(PLAINTEXT, segment_size=100))
    records.insert(1, records[1])
    with pytest.raises(EncryptionError, match="sequence"):
        decrypt(header + b"".join(records))


def test_dropped_segment_detected():
    header, records = split_stream(encrypt(PLAINTEXT, segment_size=100))
    del records[1]
    with pytest.raises(EncryptionError, match="sequence"):
        decrypt(header + b"".join(records))


def test_truncated_record_detected():
    data = encrypt(PLAINTEXT, segment_size=100)
    with pytest.raises(EncryptionError, match="truncated"):
        decrypt(data[:-10])


def test_missing_final_record_detected():
    header, records = split_stream(encrypt(PLAINTEXT, segment_size=100))
    with pytest.raises(EncryptionError, match="truncated"):
        decrypt(header + b"".join(records[:-1]))


def test_trailing_garbage_detected():
    data = encrypt(PLAINTEXT, segment_size=100)
    with pytest.raises(EncryptionError, match="trailing"):
        decrypt(data + b"garbage")


def test_bit_flip_detected():
    data = bytearray(encrypt(PLAINTEXT, segment_size=100))
    data[HEADER_SIZE + RECORD_HEADER_SIZE + 5] ^= 0x01  # inside the first ciphertext
    with pytest.raises(EncryptionError, match="authentication"):
        decrypt(bytes(data))


def test_tampered_sequence_number_detected():
    data = bytearray(encrypt(PLAINTEXT, segment_size=100))
    data[HEADER_SIZE + 7] ^= 0x01  # low byte of the first record's sequence number
    with pytest.raises(EncryptionError):
        decrypt(bytes(data))


def test_tampered_flags_detected():
    data = bytearray(encrypt(PLAINTEXT, segment_size=100))
    data[HEADER_SIZE + 20] |= FLAG_FINAL  # mark the first record as final
    with pytest.raises(EncryptionError):
        decrypt(bytes(data))


def test_wrong_key_detected():
    data = encrypt(PLAINTEXT)
    with pytest.raises(EncryptionError, match="authentication"):
        decrypt(data, key=os.urandom(32))


def test_bad_magic_detected():
    data = bytearray(encrypt(PLAINTEXT))
    data[0] ^= 0xFF
    with pytest.raises(EncryptionError, match="magic"):
        decrypt(bytes(data))


def test_bad_version_detected():
    data = bytearray(encrypt(PLAINTEXT))
    data[8] = 0xFF
    with pytest.raises(EncryptionError, match="version"):
        decrypt(bytes(data))


def test_foreign_file_id_detected():
    """Mixing records from two different streams must fail authentication."""
    _, other_records = split_stream(encrypt(PLAINTEXT, segment_size=100))
    header, records = split_stream(encrypt(PLAINTEXT, segment_size=100))
    records[1] = other_records[1]
    with pytest.raises(EncryptionError):
        decrypt(header + b"".join(records))


def test_decryption_failure_stops_output():
    """Once a record fails, no further plaintext is ever produced."""
    data = bytearray(encrypt(PLAINTEXT, segment_size=100))
    header, records = split_stream(bytes(data))
    second_record_offset = len(header) + len(records[0])
    data[second_record_offset + RECORD_HEADER_SIZE] ^= 0x01  # corrupt the second record
    reader = smart_open.encryption.SegmentedAesGcmReader(io.BytesIO(bytes(data)), KEY)
    first = reader.read(100)
    assert first == PLAINTEXT[:100]  # authentic prefix was delivered
    with pytest.raises(EncryptionError):
        reader.read(100)
    with pytest.raises(EncryptionError):  # the failure is latched
        reader.read(100)


def test_reader_is_streaming():
    """The reader must not consume the whole object before delivering plaintext."""
    data = encrypt(PLAINTEXT, segment_size=100)
    tracked = TrackingReader(data)
    reader = smart_open.encryption.encryption_wrapper(tracked, "rb", "aesgcm", {"key": KEY})
    assert reader.read(10) == PLAINTEXT[:10]
    assert tracked.bytes_read < len(data) // 2
    assert reader.read() == PLAINTEXT[10:]


def test_key_sizes():
    for size in (16, 24, 32):
        key = os.urandom(size)
        assert decrypt(encrypt(PLAINTEXT, key=key), key=key) == PLAINTEXT
    with pytest.raises(ValueError, match="key"):
        encrypt(PLAINTEXT, key=os.urandom(17))
    with pytest.raises(TypeError, match="key"):
        encrypt(PLAINTEXT, key="not-bytes")


def test_key_required():
    with pytest.raises(ValueError, match="key"):
        smart_open.encryption.encryption_wrapper(io.BytesIO(), "wb", "aesgcm")
    with pytest.raises(ValueError, match="key"):
        smart_open.encryption.encryption_wrapper(io.BytesIO(), "wb", "aesgcm", {})


def test_invalid_encryption_type():
    with pytest.raises(ValueError, match="encryption"):
        smart_open.encryption.encryption_wrapper(io.BytesIO(), "wb", "rot13", {"key": KEY})


def test_disabled_encryption_passthrough():
    stream = io.BytesIO()
    assert smart_open.encryption.encryption_wrapper(stream, "wb", "disable") is stream


def test_unsupported_modes_rejected():
    for mode in ("ab", "rb+", "wb+"):
        with pytest.raises(ValueError, match="mode"):
            smart_open.encryption.encryption_wrapper(io.BytesIO(), mode, "aesgcm", {"key": KEY})


def test_invalid_segment_size_rejected():
    with pytest.raises(ValueError, match="segment_size"):
        smart_open.encryption.encryption_wrapper(
            io.BytesIO(), "wb", "aesgcm", {"key": KEY, "segment_size": 0}
        )


@pytest.fixture(params=["wb", "wt"])
def write_mode(request):
    return request.param


def test_open_roundtrip(tmp_path, write_mode):
    path = str(tmp_path / "backup.bin")
    plaintext = PLAINTEXT.decode() if write_mode == "wt" else PLAINTEXT
    with smart_open.open(
        path, write_mode, encryption="aesgcm", encryption_kwargs={"key": KEY, "segment_size": 512}
    ) as fout:
        fout.write(plaintext)
    read_mode = write_mode.replace("w", "r")
    with smart_open.open(path, read_mode, encryption="aesgcm", encryption_kwargs={"key": KEY}) as fin:
        assert fin.read() == plaintext


def test_open_compression_happens_before_encryption(tmp_path):
    """The ciphertext must wrap the compressed (gzip) byte stream."""
    path = tmp_path / "backup.txt.gz"
    with smart_open.open(str(path), "wb", encryption="aesgcm", encryption_kwargs={"key": KEY}) as fout:
        fout.write(PLAINTEXT)
    raw = path.read_bytes()
    assert raw[:8] == MAGIC  # the outermost layer is the encrypted stream
    decrypted = decrypt(raw)
    assert decrypted[:2] == b"\x1f\x8b"  # gzip magic: compression happened first
    with smart_open.open(str(path), "rb", encryption="aesgcm", encryption_kwargs={"key": KEY}) as fin:
        assert fin.read() == PLAINTEXT


def test_open_corrupted_object_detected(tmp_path):
    path = tmp_path / "backup.bin"
    path.write_bytes(encrypt(PLAINTEXT, segment_size=100))
    raw = bytearray(path.read_bytes())
    raw[-5] ^= 0x01  # corrupt the final record's tag
    path.write_bytes(bytes(raw))
    with pytest.raises(EncryptionError):
        with smart_open.open(str(path), "rb", encryption="aesgcm", encryption_kwargs={"key": KEY}) as fin:
            fin.read()


def test_open_invalid_encryption_type(tmp_path):
    with pytest.raises(ValueError, match="encryption"):
        smart_open.open(str(tmp_path / "f.bin"), "wb", encryption="rot13")


def test_open_append_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        smart_open.open(str(tmp_path / "f.bin"), "ab", encryption="aesgcm", encryption_kwargs={"key": KEY})


def test_open_encryption_disabled_by_default(tmp_path):
    path = str(tmp_path / "f.bin")
    with smart_open.open(path, "wb") as fout:
        fout.write(PLAINTEXT)
    with smart_open.open(path, "rb") as fin:
        assert fin.read() == PLAINTEXT


def test_key_not_logged(tmp_path, caplog):
    key = b"supersecret-key-do-not-log!!1234"[:32]
    assert len(key) == 32
    with caplog.at_level(logging.DEBUG):
        with smart_open.open(
            str(tmp_path / "f.bin"), "wb", encryption="aesgcm", encryption_kwargs={"key": key}
        ) as fout:
            fout.write(b"x")
        with smart_open.open(
            str(tmp_path / "f.bin"), "rb", encryption="aesgcm", encryption_kwargs={"key": key}
        ) as fin:
            fin.read()
    assert repr(key) not in caplog.text
    assert key.hex() not in caplog.text
    assert "<redacted>" in caplog.text

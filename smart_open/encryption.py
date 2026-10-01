#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#

"""Explicit opt-in, streaming authenticated encryption for backup objects.

The layer encrypts data *after* compression and *before* it reaches the object
store (S3, GCS, Azure, ...), and transparently decrypts it again on the way
back.  Plaintext is only delivered to the caller **after** each segment has
been authenticated with AES-GCM.  Objects are never buffered whole: at most one
segment (see ``DEFAULT_CHUNK_SIZE``) is held in memory.

Enable it explicitly through :func:`smart_open.open`::

    from smart_open import AESGCMEncryption, open

    with open("s3://bucket/backup.gz", "wb",
              compression=".gz", encryption=AESGCMEncryption(key)) as f:
        f.write(payload)

    with open("s3://bucket/backup.gz", "rb", encryption=AESGCMEncryption(key)) as f:
        assert f.read() == payload

The key is caller-supplied bytes and must never be embedded in the URI or
logged; it is passed separately via the ``encryption`` parameter.

Minimum interoperable format -- "SOAE" (Smart-Open Authenticated Encryption),
version 1.

All multi-byte integers are unsigned, big-endian.

Container layout::

    stream  := header record* terminal
    header  := b"SOAE" version(=0x01) file_id(16 random bytes)
    record  := type(1) nonce(12) length(4) ciphertext(length)
    type    := 0x00 data segment | 0x01 authenticated terminal record

Each segment is sealed independently with AES-GCM using a fresh random 96-bit
nonce.  ``ciphertext`` is the encrypted plaintext followed by the 16-byte GCM
authentication tag.  The authenticated associated data (AAD) of *every*
record binds it to the container and to its exact position::

    aad := b"SOAE" || version(1) || file_id(16) || sequence(8) || type(1)

``sequence`` starts at 0 and increments by one for each record, including the
terminal.  The terminal record seals 8 bytes containing the number of data
segments (big-endian uint64).

Consequences, all verified by the reader before plaintext is handed out:

* a reordered, duplicated or dropped data segment fails the strict sequence
  check or the terminal's authenticated segment count;
* a truncated stream (missing terminal) is rejected;
* a replayed/forged terminal, or any extra bytes following the terminal, is
  rejected;
* any tampering with the magic, version, file id, sequence, type, ciphertext
  or tag fails GCM authentication (the header fields are part of every AAD);
* the decryption key is never read from the URI; ``repr`` and logging never
  expose it.
"""

from __future__ import annotations

import io
import os
import struct
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from smart_open._typing import Encryption

MAGIC = b"SOAE"
"""Container magic bytes."""

FORMAT_VERSION = 1
"""Only format version implemented by this module."""

FILE_ID_BYTES = 16
"""Per-file random identifier length, in bytes."""

NONCE_BYTES = 12
"""AES-GCM nonce (IV) length, in bytes."""

TAG_BYTES = 16
"""AES-GCM authentication tag length, in bytes."""

LENGTH_BYTES = 4
"""Big-endian ciphertext-length prefix length, in bytes."""

SEQUENCE_BYTES = 8
"""Big-endian segment-sequence length, in bytes."""

TYPE_BYTES = 1

_TYPE_DATA = 0
_TYPE_TERMINAL = 1

DEFAULT_CHUNK_SIZE = 64 * 1024
"""Plaintext bytes buffered before one data segment is sealed."""

MIN_CHUNK_SIZE = 1024
MAX_CHUNK_SIZE = 64 * 1024**2
# Guard against malformed length prefixes forcing huge allocations.
MAX_RECORD_CIPHERTEXT = MAX_CHUNK_SIZE + TAG_BYTES

_HEADER_LEN = len(MAGIC) + 1 + FILE_ID_BYTES
_RECORD_PREFIX_LEN = TYPE_BYTES + NONCE_BYTES + LENGTH_BYTES
_TERMINAL_PAYLOAD = SEQUENCE_BYTES


class EncryptionError(Exception):
    """Raised when an encrypted stream fails verification or cannot be parsed.

    Plaintext that fails authentication is never delivered; once this error is
    raised the reader stops and produces no further output.
    """


def _aesgcm() -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - exercised without optional dep
        msg = (
            "streaming encryption needs the cryptography package; install it "
            "via `pip install smart_open[crypto]`"
        )
        raise ImportError(msg) from exc
    return AESGCM


def _aad(file_id: bytes, sequence: int, record_type: int) -> bytes:
    return MAGIC + bytes((FORMAT_VERSION,)) + file_id + struct.pack(">Q", sequence) + bytes((record_type,))


class AESGCMEncryption:
    """Configuration and factory for the SOAE v1 AES-GCM encryption layer.

    Args:
        key: AES key supplied by the caller.  32 bytes (AES-256) by default;
            16 or 24 bytes (AES-128/AES-192) are also accepted.  The key is
            never embedded in a URI or logged.
        chunk_size: Maximum plaintext bytes per sealed data segment.

    Raises:
        TypeError: If ``key`` is not bytes-like.
        ValueError: If the key length is not a valid AES key length, or
            ``chunk_size`` is out of range.
    """

    algorithm = "AES-GCM"
    version = FORMAT_VERSION

    def __init__(self, key: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if not isinstance(key, (bytes, bytearray)):
            msg = "encryption key must be bytes supplied out-of-band (never a URI component)"
            raise TypeError(msg)
        if len(key) not in (16, 24, 32):
            msg = "encryption key must be 16, 24 or 32 bytes for AES-128/192/256"
            raise ValueError(msg)
        if not isinstance(chunk_size, int) or not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
            msg = f"chunk_size must be an int in [{MIN_CHUNK_SIZE}, {MAX_CHUNK_SIZE}]"
            raise ValueError(msg)
        self._key = bytes(key)
        self.chunk_size = chunk_size

    @property
    def key(self) -> bytes:
        """The raw key; callers should avoid copying it into URIs or logs."""
        return self._key

    def writer(self, raw: BinaryIO) -> BinaryIO:
        """Return a buffered binary writer that encrypts into ``raw``."""
        return io.BufferedWriter(_EncryptingWriter(raw, self), buffer_size=self.chunk_size)

    def reader(self, raw: BinaryIO) -> BinaryIO:
        """Return a buffered binary reader that decrypts from ``raw``."""
        return io.BufferedReader(_DecryptingReader(raw, self))

    def __repr__(self) -> str:
        return f"AESGCMEncryption(algorithm={self.algorithm!r}, version={self.version}, key=<redacted>)"


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if not isinstance(data, (bytes, bytearray)):
        msg = "underlying stream did not return bytes"
        raise EncryptionError(msg)
    if len(data) != size:
        msg = f"truncated encrypted stream: wanted {size} bytes, got {len(data)}"
        raise EncryptionError(msg)
    return bytes(data)


class _EncryptingWriter(io.RawIOBase):
    """Raw stream: buffer-independent AES-GCM segments written to ``raw``."""

    def __init__(self, raw: BinaryIO, config: AESGCMEncryption) -> None:
        super().__init__()
        self._raw = raw
        self._aesgcm = _aesgcm()(config.key)
        self._chunk_size = config.chunk_size
        self._file_id = os.urandom(FILE_ID_BYTES)
        self._sequence = 0
        self._data_segments = 0
        self._header_written = False
        self._terminated = False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def _write_header(self) -> None:
        self._raw.write(MAGIC + bytes((FORMAT_VERSION,)) + self._file_id)
        self._header_written = True

    def _seal(self, plaintext: bytes, record_type: int) -> None:
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = self._aesgcm.encrypt(
            nonce, plaintext, _aad(self._file_id, self._sequence, record_type)
        )
        self._raw.write(
            bytes((record_type,)) + nonce + struct.pack(">I", len(ciphertext)) + ciphertext
        )
        self._sequence += 1

    def write(self, b: Any) -> int:
        if self.closed or self._terminated:
            msg = "write to closed/terminated encryption stream"
            raise ValueError(msg)
        data = bytes(memoryview(b))
        if not self._header_written:
            self._write_header()
        offset = 0
        while offset < len(data):
            chunk = data[offset : offset + self._chunk_size]
            self._seal(chunk, _TYPE_DATA)
            self._data_segments += 1
            offset += len(chunk)
        return len(data)

    def close(self) -> None:
        if not self.closed:
            try:
                if not self._terminated:
                    if not self._header_written:
                        self._write_header()
                    self._seal(struct.pack(">Q", self._data_segments), _TYPE_TERMINAL)
                    self._terminated = True
                    flush = getattr(self._raw, "flush", None)
                    if flush is not None:
                        flush()
            finally:
                super().close()


class _DecryptingReader(io.RawIOBase):
    """Raw stream: verify one segment, then release only that segment's bytes."""

    def __init__(self, raw: BinaryIO, config: AESGCMEncryption) -> None:
        super().__init__()
        self._raw = raw
        self._aesgcm = _aesgcm()(config.key)
        header = _read_exact(raw, _HEADER_LEN)
        if header[: len(MAGIC)] != MAGIC:
            msg = "not a smart_open encrypted stream (bad magic)"
            raise EncryptionError(msg)
        if header[len(MAGIC)] != FORMAT_VERSION:
            msg = f"unsupported encryption format version: {header[len(MAGIC)]}"
            raise EncryptionError(msg)
        self._file_id = header[-FILE_ID_BYTES:]
        self._sequence = 0
        self._pending = b""
        self._terminal_seen = False
        self._poisoned = False

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def _fail(self, message: str) -> None:
        self._poisoned = True
        raise EncryptionError(message)

    def _read_record(self) -> tuple[int, bytes]:
        prefix = _read_exact(self._raw, _RECORD_PREFIX_LEN)
        record_type = prefix[0]
        nonce = prefix[TYPE_BYTES : TYPE_BYTES + NONCE_BYTES]
        (length,) = struct.unpack(
            ">I", prefix[TYPE_BYTES + NONCE_BYTES : TYPE_BYTES + NONCE_BYTES + LENGTH_BYTES]
        )
        if record_type not in (_TYPE_DATA, _TYPE_TERMINAL):
            self._fail(f"unknown record type: {record_type}")
        if length > MAX_RECORD_CIPHERTEXT:
            self._fail(f"record length {length} exceeds maximum {MAX_RECORD_CIPHERTEXT}")
        ciphertext = _read_exact(self._raw, length)
        return record_type, self._open(nonce, ciphertext, record_type)

    def _open(self, nonce: bytes, ciphertext: bytes, record_type: int) -> bytes:
        try:
            return self._aesgcm.decrypt(
                nonce, ciphertext, _aad(self._file_id, self._sequence, record_type)
            )
        except Exception:
            self._fail("segment authentication failed")

    def readinto(self, b: Any) -> int:
        view = memoryview(b)
        if self._poisoned:
            msg = "stream aborted after an authentication failure"
            raise EncryptionError(msg)
        if self._terminal_seen:
            return 0
        if len(self._pending) == 0:
            try:
                record_type, plaintext = self._read_record()
                if record_type == _TYPE_DATA:
                    if len(plaintext) > MAX_CHUNK_SIZE:
                        self._fail(f"data segment {len(plaintext)} exceeds chunk limit")
                    self._sequence += 1
                    self._pending = plaintext
                else:
                    if len(plaintext) != _TERMINAL_PAYLOAD:
                        self._fail("malformed terminal record")
                    (declared,) = struct.unpack(">Q", plaintext)
                    if declared != self._sequence:
                        self._fail(
                            f"segment count mismatch: terminal authenticated {declared}, seen {self._sequence}"
                        )
                    self._terminal_seen = True
                    # The terminal is the final record: anything after it is a forgery.
                    trailing = self._raw.read(1)
                    if trailing:
                        self._fail("extra bytes after authenticated terminal")
                    return 0
            except EncryptionError:
                # Authentication/framing failure: stop, deliver nothing more.
                self._poisoned = True
                raise
        take = min(len(view), len(self._pending))
        view[:take] = self._pending[:take]
        self._pending = self._pending[take:]
        return take


def validate_encryption(encryption: Any) -> Encryption | None:
    """Validate an explicitly supplied ``encryption`` argument.

    Args:
        encryption: None (disabled) or an object exposing ``reader(raw)`` and
            ``writer(raw)`` factories, e.g. :class:`AESGCMEncryption`.  Raw key
            bytes are accepted and wrapped in :class:`AESGCMEncryption`.

    Returns:
        The validated config, or None when encryption is disabled.

    Raises:
        TypeError: If ``encryption`` is not a recognized encryption config or
            key.
    """
    if encryption is None:
        return None
    if isinstance(encryption, (bytes, bytearray)):
        return AESGCMEncryption(bytes(encryption))
    if all(hasattr(encryption, attr) for attr in ("reader", "writer")):
        return encryption
    msg = "encryption must be an AESGCMEncryption (or key bytes), not a URI component"
    raise TypeError(msg)

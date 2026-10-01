#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Implements the transparent segmented encryption layer of the `smart_open` library.

Encryption is always opt-in: pass ``encryption="aesgcm"`` and
``encryption_kwargs={"key": ...}`` to :func:`smart_open.open`.  The key is
supplied out-of-band by the caller; it never appears in URIs, and smart_open
never logs it.

Compression (if any) happens before encryption when writing, and after
decryption when reading, so ciphertext always wraps the compressed bytes.

On-the-wire format (version 1)
------------------------------

All integers are big-endian.  The stream consists of one header followed by
one or more segment records, the last of which carries the FINAL flag.

Stream header (29 bytes)::

    offset  size  field
    0       8     magic: b"SOAESGCM"
    8       1     format version: 0x01
    9       4     segment payload size in bytes (>= 1)
    13      16    random file identifier (unique per encrypted stream)

Segment record (repeated until and including a record with the FINAL flag)::

    offset  size  field
    0       8     segment sequence number, starting at 0, incrementing by 1
    8       12    random 96-bit GCM nonce
    20      1     flags: bit 0 (0x01) = FINAL, other bits reserved (must be 0)
    21      4     record length N: byte length of the following field
    25      N     AES-GCM ciphertext || 16-byte authentication tag

The additional authenticated data (AAD) of every record is
``header || sequence number || flags`` (29 + 8 + 1 = 38 bytes), binding each
segment to its file, its position and its finality.  A record payload is at
most ``segment payload size`` bytes of plaintext; the FINAL record may carry
a zero-length payload (N == 16, tag only).  A well-formed stream ends with
exactly one FINAL record and no trailing bytes.

This layout lets a reader detect reordered, duplicated, truncated, extended
and bit-flipped segments, and lets a writer stream without buffering more
than one segment in memory.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smart_open._typing import EncryptionKwargs

logger = logging.getLogger(__name__)

NO_ENCRYPTION = "disable"
"""Use no encryption.  Read/write the data as-is."""

AES_GCM = "aesgcm"
"""Segmented AES-GCM authenticated encryption (see the module docstring)."""

MAGIC = b"SOAESGCM"
VERSION = 1

_VERSION_STRUCT = struct.Struct(">B")
_SEGMENT_SIZE_STRUCT = struct.Struct(">I")
_FILE_ID_SIZE = 16
_NONCE_SIZE = 12
_TAG_SIZE = 16

HEADER_SIZE = len(MAGIC) + _VERSION_STRUCT.size + _SEGMENT_SIZE_STRUCT.size + _FILE_ID_SIZE
"""Total size of the stream header in bytes."""

_RECORD_HEADER_STRUCT = struct.Struct(">Q12sBI")
RECORD_HEADER_SIZE = _RECORD_HEADER_STRUCT.size
"""Size of a segment record header (sequence, nonce, flags, length) in bytes."""

FLAG_FINAL = 0x01
"""Record flag marking the authenticated final record of the stream."""

DEFAULT_SEGMENT_SIZE = 4 * 1024 * 1024
"""Default plaintext segment size in bytes."""

_KEY_SIZES = (16, 24, 32)
"""Valid AES key sizes in bytes."""

_REDACTED = "<redacted>"


class EncryptionError(ValueError):
    """Raised when an encrypted stream is malformed, corrupt or inauthentic."""


def get_supported_encryption_types() -> list[str]:
    """Return the list of supported encryption types available to open.

    See the encryption parameter to smart_open.open().
    """
    return [NO_ENCRYPTION, AES_GCM]


def redact_kwargs(encryption_kwargs: EncryptionKwargs | None) -> EncryptionKwargs | None:
    """Return a copy of `encryption_kwargs` with the secret key redacted, safe for logging."""
    if encryption_kwargs is None:
        return None
    return {name: (_REDACTED if name == "key" else value) for name, value in encryption_kwargs.items()}


def _make_aesgcm(key: bytes) -> Any:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not isinstance(key, (bytes, bytearray, memoryview)):
        msg = f"encryption key must be bytes-like, not {type(key).__name__}"
        raise TypeError(msg)
    key = bytes(key)
    if len(key) not in _KEY_SIZES:
        msg = f"encryption key must be {_KEY_SIZES} bytes long, not {len(key)}"
        raise ValueError(msg)
    return AESGCM(key)


def _read_exactly(file_obj: IO[bytes], size: int) -> bytes:
    """Read exactly `size` bytes from `file_obj`, raising EncryptionError on EOF."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = file_obj.read(remaining)
        if not chunk:
            msg = "truncated stream: unexpected end of file before the authenticated final record"
            raise EncryptionError(msg)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class SegmentedAesGcmWriter(io.RawIOBase):
    """Encrypt plaintext into segmented AES-GCM records, streaming to `file_obj`.

    Buffers at most one segment of plaintext in memory.  The authenticated
    FINAL record is written on close; closing the underlying `file_obj` is
    left to the caller.
    """

    def __init__(self, file_obj: IO[bytes], key: bytes, segment_size: int = DEFAULT_SEGMENT_SIZE) -> None:
        if segment_size < 1:
            msg = f"segment_size must be a positive integer, not {segment_size!r}"
            raise ValueError(msg)
        self._file_obj = file_obj
        self._buffer = bytearray()
        self._seq = 0
        self._final_written = False
        self._aesgcm = _make_aesgcm(key)
        self._segment_size = segment_size
        self._header = (
            MAGIC
            + _VERSION_STRUCT.pack(VERSION)
            + _SEGMENT_SIZE_STRUCT.pack(segment_size)
            + os.urandom(_FILE_ID_SIZE)
        )
        self._file_obj.write(self._header)

    def writable(self) -> bool:  # noqa: D102
        return True

    def _emit_record(self, payload: bytes | bytearray, flags: int) -> None:
        nonce = os.urandom(_NONCE_SIZE)
        aad = self._header + struct.pack(">Q", self._seq) + bytes([flags])
        ciphertext = self._aesgcm.encrypt(nonce, bytes(payload), aad)
        self._file_obj.write(_RECORD_HEADER_STRUCT.pack(self._seq, nonce, flags, len(ciphertext)))
        self._file_obj.write(ciphertext)
        self._seq += 1

    def write(self, b: bytes | bytearray | memoryview) -> int:  # noqa: D102
        if self.closed:
            msg = "I/O operation on closed file."
            raise ValueError(msg)
        self._buffer += b
        while len(self._buffer) >= self._segment_size:
            self._emit_record(self._buffer[: self._segment_size], 0)
            del self._buffer[: self._segment_size]
        return len(b)

    def flush(self) -> None:  # noqa: D102
        if self.closed:
            msg = "I/O operation on closed file."
            raise ValueError(msg)
        if self._buffer:
            self._emit_record(self._buffer, 0)
            self._buffer.clear()
        self._file_obj.flush()

    def close(self) -> None:  # noqa: D102
        if self.closed:
            return
        try:
            if not getattr(self, "_final_written", True):
                self._emit_record(self._buffer, FLAG_FINAL)
                self._buffer.clear()
                self._final_written = True
        finally:
            super().close()


class SegmentedAesGcmReader(io.RawIOBase):
    """Decrypt segmented AES-GCM records from `file_obj`, streaming plaintext.

    Verifies each record's authentication tag before releasing its plaintext.
    Reordered, duplicated, truncated or tampered records raise
    :class:`EncryptionError`; once a record has failed, all subsequent reads
    fail too, so no output is ever produced past the corruption point.
    """

    def __init__(self, file_obj: IO[bytes], key: bytes) -> None:
        self._file_obj = file_obj
        self._aesgcm = _make_aesgcm(key)
        self._header = _read_exactly(self._file_obj, HEADER_SIZE)
        self._parse_header()
        self._seq = 0
        self._plaintext = bytearray()
        self._final_seen = False
        self._failure: Exception | None = None

    def _parse_header(self) -> None:
        magic = self._header[: len(MAGIC)]
        if magic != MAGIC:
            msg = f"not an encrypted smart_open stream: bad magic {magic!r}"
            raise EncryptionError(msg)
        (version,) = _VERSION_STRUCT.unpack(self._header[len(MAGIC) : len(MAGIC) + 1])
        if version != VERSION:
            msg = f"unsupported encryption format version: {version}"
            raise EncryptionError(msg)
        (self._segment_size,) = _SEGMENT_SIZE_STRUCT.unpack(self._header[len(MAGIC) + 1 : len(MAGIC) + 5])
        if self._segment_size < 1:
            msg = "corrupt stream header: segment size must be positive"
            raise EncryptionError(msg)

    def readable(self) -> bool:  # noqa: D102
        return True

    def _read_next_record(self) -> None:
        record_header = _read_exactly(self._file_obj, RECORD_HEADER_SIZE)
        seq, nonce, flags, length = _RECORD_HEADER_STRUCT.unpack(record_header)
        if flags & ~FLAG_FINAL:
            msg = f"corrupt segment record: unknown flags {flags:#04x}"
            raise EncryptionError(msg)
        if seq != self._seq:
            msg = f"out-of-order, duplicated or missing segment: expected sequence {self._seq}, got {seq}"
            raise EncryptionError(msg)
        if length < _TAG_SIZE:
            msg = f"corrupt segment record: length {length} smaller than the authentication tag"
            raise EncryptionError(msg)
        ciphertext = _read_exactly(self._file_obj, length)
        aad = self._header + struct.pack(">Q", seq) + bytes([flags])
        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, aad)
        except Exception as exc:
            msg = f"segment {seq} failed authentication"
            raise EncryptionError(msg) from exc
        self._seq += 1
        if flags & FLAG_FINAL:
            self._final_seen = True
            if self._file_obj.read(1):
                msg = "trailing data after the authenticated final record"
                raise EncryptionError(msg)
        self._plaintext = bytearray(plaintext)

    def readinto(self, b: bytearray | memoryview) -> int:  # noqa: D102
        if self._failure is not None:
            raise self._failure
        try:
            while not self._plaintext:
                if self._final_seen:
                    return 0
                self._read_next_record()
        except Exception as exc:
            self._failure = exc
            raise
        size = min(len(b), len(self._plaintext))
        b[:size] = self._plaintext[:size]
        del self._plaintext[:size]
        return size


def encryption_wrapper(
    file_obj: IO[bytes],
    mode: str,
    encryption: str = NO_ENCRYPTION,
    encryption_kwargs: EncryptionKwargs | None = None,
) -> IO[Any]:
    """Wrap `file_obj` with transparent segmented AES-GCM [de]encryption.

    If `encryption` is "disable", simply return `file_obj` unchanged.

    `encryption_kwargs` must contain a ``key`` entry holding the caller's
    secret AES key (16, 24 or 32 bytes).  The optional ``segment_size`` entry
    (write mode only) sets the plaintext segment size in bytes.  The key is
    never logged or embedded in the URI by smart_open.

    Only the "rb" and "wb" modes are supported: encrypted streams are
    sequential and cannot be appended to or updated in place.
    """
    if encryption == NO_ENCRYPTION:
        return file_obj
    if encryption != AES_GCM:
        msg = f"invalid encryption type: {encryption}"
        raise ValueError(msg)
    if mode not in ("rb", "wb"):
        msg = f"transparent encryption unsupported for mode {mode!r}"
        raise ValueError(msg)

    kwargs = dict(encryption_kwargs or {})
    key = kwargs.pop("key", None)
    if key is None:
        msg = "encryption_kwargs must contain the caller-provided secret 'key'"
        raise ValueError(msg)

    if mode == "rb":
        if kwargs:
            msg = f"unsupported encryption_kwargs for reading: {sorted(kwargs)}"
            raise ValueError(msg)
        return io.BufferedReader(SegmentedAesGcmReader(file_obj, key))
    return io.BufferedWriter(SegmentedAesGcmWriter(file_obj, key, **kwargs))

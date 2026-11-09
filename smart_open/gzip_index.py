#
# Copyright (C) 2026
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Session-only decompression checkpoints for seekable local gzip files.

The standard-library :class:`gzip.GzipFile` supports ``seek`` on a seekable
underlying stream, but every backwards jump re-opens the compressed file at
its very beginning and re-decompresses everything up to the requested
position. Workloads that repeatedly sample the same local gzip file pay that
cost once per jump.

This module adds an explicitly opt-in index of decompression checkpoints that
lives only for the duration of a single open. A checkpoint stores the full
:mod:`zlib` decompressor state (via :meth:`zlib.Decompress.copy`) together
with the compressed bytes already read from disk but not yet consumed by the
decompressor. Backwards seeks resume from the closest retained checkpoint
instead of restarting at offset zero, while forward seeks keep decompressing
ahead as usual.

The index never caches decompressed output and never writes anything to disk,
so the total retained state is bounded by the configured checkpoint limit
rather than by the length of the file.

The implementation deliberately mirrors :mod:`gzip`'s member, CRC and ISIZE
handling so that checkpoints cannot bypass stream validation:
:class:`gzip.BadGzipFile` is still raised on corrupt members.
"""

from __future__ import annotations

import gzip
import io
import os
import stat
import struct
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import IO, Any

READ_BUFFER_SIZE = 128 * 1024

GZIP_MAGIC = b"\037\213"
GZIP_FLAG_FEXTRA = 4
GZIP_FLAG_FNAME = 8
GZIP_FLAG_FCOMMENT = 16
GZIP_FLAG_FHCRC = 2

DEFAULT_INDEX_INTERVAL = 1 << 20
DEFAULT_MAX_CHECKPOINTS = 32


class _RawBuffer:
    """Compressed bytes read from the underlying file but not yet parsed.

    The buffer keeps the raw (still compressed) stream in front of both the
    gzip header/trailer parser and the zlib decompressor. ``read`` only pulls
    more bytes from the underlying file when the pending data is exhausted,
    which makes restoring a checkpoint free of extra disk reads.
    """

    def __init__(self, fp: IO[bytes]) -> None:
        self._fp = fp
        self._pending = b""

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            data = self._pending + self._fp.read()
            self._pending = b""
            return data
        if size == 0:
            return b""
        if len(self._pending) >= size:
            data = self._pending[:size]
            self._pending = self._pending[size:]
            return data
        data = self._pending
        self._pending = b""
        if len(data) < size:
            data += self._fp.read(size - len(data))
        return data

    def prepend(self, data: bytes) -> None:
        if data:
            self._pending = data + self._pending

    def snapshot(self) -> bytes:
        return self._pending

    def restore(self, pending: bytes) -> None:
        self._pending = pending


@dataclass(frozen=True)
class _Checkpoint:
    """A resumable point in the decompressed stream.

    Args:
        pos: Decompressed byte position the checkpoint resumes at.
        decompressor: Independent copy of the zlib decompressor, including any
            bytes buffered internally by zlib.
        pending: Compressed bytes already read from disk but not yet handed to
            the decompressor or the gzip header/trailer parser.
        raw_position: Physical offset of the underlying file at the moment the
            pending bytes and the decompressor's internal buffer were formed.
            Restoring it keeps the file aligned with the saved decompressor
            state; it is only meaningful together with that state, which is
            why the index stores both rather than a bare file offset.
        new_member: Whether a gzip header must be parsed before the next read.
        crc: Running CRC32 of the currently open member.
        stream_size: Decompressed byte count of the currently open member.
    """

    pos: int
    decompressor: zlib.Decompress
    pending: bytes
    raw_position: int
    new_member: bool
    crc: int
    stream_size: int


def _read_exact(raw: _RawBuffer, size: int) -> bytes:
    """Read exactly *size* bytes, raising EOFError on a truncated stream."""
    data = raw.read(size)
    while len(data) < size:
        chunk = raw.read(size - len(data))
        if not chunk:
            raise EOFError(
                "Compressed file ended before the end-of-stream marker was reached"
            )
        data += chunk
    return data


def _read_gzip_header(raw: _RawBuffer) -> int | None:
    """Parse one gzip member header, returning its mtime or None at EOF."""
    magic = raw.read(2)
    if magic == b"":
        return None
    if magic != GZIP_MAGIC:
        raise gzip.BadGzipFile(f"Not a gzipped file ({magic!r})")

    method, flag, mtime = struct.unpack("<BBIxx", _read_exact(raw, 8))
    if method != 8:
        raise gzip.BadGzipFile("Unknown compression method")

    if flag & GZIP_FLAG_FEXTRA:
        (extra_len,) = struct.unpack("<H", _read_exact(raw, 2))
        _read_exact(raw, extra_len)
    if flag & GZIP_FLAG_FNAME:
        while True:
            char = raw.read(1)
            if not char or char == b"\000":
                break
    if flag & GZIP_FLAG_FCOMMENT:
        while True:
            char = raw.read(1)
            if not char or char == b"\000":
                break
    if flag & GZIP_FLAG_FHCRC:
        _read_exact(raw, 2)
    return mtime


class IndexedGzipReader(io.RawIOBase):
    """Seekable raw gzip reader with a bounded, session-only checkpoint index.

    Args:
        fp: A binary, seekable, local file positioned at the start of the gzip
            stream. The stream is assumed not to change while open.
        index_interval: Create a checkpoint roughly every this many
            decompressed bytes.
        max_checkpoints: Maximum number of checkpoints retained, including the
            pinned checkpoint at position zero. Older checkpoints are evicted
            first when the limit is exceeded.

    Raises:
        ValueError: If an interval or limit argument is non-positive.
    """

    def __init__(
        self,
        fp: IO[bytes],
        index_interval: int = DEFAULT_INDEX_INTERVAL,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
    ) -> None:
        if index_interval <= 0:
            msg = f"index_interval must be positive, got {index_interval!r}"
            raise ValueError(msg)
        if max_checkpoints <= 0:
            msg = f"max_checkpoints must be positive, got {max_checkpoints!r}"
            raise ValueError(msg)
        self._fp = fp
        self._raw = _RawBuffer(fp)
        self._index_interval = index_interval
        self._max_checkpoints = max_checkpoints

        self._pos = 0
        self._size = -1
        self._new_member = True
        self._crc = zlib.crc32(b"")
        self._stream_size = 0
        self._decompressor = self._new_decompressor()

        # Ordered by decompressed position. The zero checkpoint is pinned and
        # doubles as the equivalent of gzip's full-rewind fallback.
        self._checkpoints: OrderedDict[int, _Checkpoint] = OrderedDict()
        self._checkpoints[0] = self._snapshot()
        self._last_checkpoint_pos = 0

    @staticmethod
    def _new_decompressor() -> zlib.Decompress:
        return zlib.decompressobj(-zlib.MAX_WBITS)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def name(self) -> Any:
        """Name of the underlying file, when available."""
        return getattr(self._fp, "name", None)

    def _snapshot(self) -> _Checkpoint:
        return _Checkpoint(
            pos=self._pos,
            decompressor=self._decompressor.copy(),
            pending=self._raw.snapshot(),
            raw_position=self._fp.tell(),
            new_member=self._new_member,
            crc=self._crc,
            stream_size=self._stream_size,
        )

    def _restore(self, checkpoint: _Checkpoint) -> None:
        # Re-align the physical file with the saved decompressor state. The
        # state still owns every byte between this offset and the current one,
        # so no decompressed work is repeated.
        self._fp.seek(checkpoint.raw_position)
        self._pos = checkpoint.pos
        self._decompressor = checkpoint.decompressor.copy()
        self._raw.restore(checkpoint.pending)
        self._new_member = checkpoint.new_member
        self._crc = checkpoint.crc
        self._stream_size = checkpoint.stream_size

    def _record_checkpoint(self) -> None:
        """Snapshot at interval boundaries, independent of caller read sizes.

        A single read may span several intervals, so decompression is capped
        to the next boundary whenever a checkpoint is due. Capping keeps the
        checkpoint consistent: every byte produced up to it has already been
        accounted for and no output past it exists yet. Checkpoints are only
        taken from a live member: at decompressor EOF the member trailer has
        not yet been validated, so the next read must finish the member before
        any resumable state can be retained.
        """
        if self._pos - self._last_checkpoint_pos < self._index_interval:
            return
        if self._decompressor.eof or self._new_member:
            return
        if self._pos in self._checkpoints:
            self._last_checkpoint_pos = self._pos
            return
        self._checkpoints[self._pos] = self._snapshot()
        self._last_checkpoint_pos = self._pos
        # Position zero is pinned, so the evictable budget is one slot lower.
        while len(self._checkpoints) > self._max_checkpoints:
            oldest = next(iter(self._checkpoints))
            if oldest != 0:
                self._checkpoints.pop(oldest)
                continue
            if len(self._checkpoints) <= 1:
                break
            # The pinned zero checkpoint occupies the first slot; evict the
            # next oldest instead.
            second = next(iter(list(self._checkpoints.keys())[1:2]))
            self._checkpoints.pop(second)

    def _next_checkpoint_cap(self, remaining: int) -> int:
        """Limit a decompress call so an interval boundary stays observable."""
        next_boundary = self._last_checkpoint_pos + self._index_interval
        gap = next_boundary - self._pos
        if 0 < gap <= remaining:
            return gap
        return remaining

    def _discard_checkpoints_after(self, pos: int) -> None:
        # OrderedDict is keyed by ascending position, so pop from the end.
        while self._checkpoints and next(reversed(self._checkpoints)) > pos:
            self._checkpoints.popitem()
        self._last_checkpoint_pos = max(
            key for key in self._checkpoints if key <= pos
        )

    def _finish_member(self) -> None:
        """Validate the trailer of a completed member and skip padding zeros."""
        crc32, isize = struct.unpack("<II", _read_exact(self._raw, 8))
        if crc32 != self._crc:
            raise gzip.BadGzipFile(
                f"CRC check failed {hex(crc32)} != {hex(self._crc)}"
            )
        if isize != (self._stream_size & 0xFFFFFFFF):
            raise gzip.BadGzipFile("Incorrect length of data produced")
        # gzip allows zero padding between concatenated members.
        padding = b"\x00"
        while padding == b"\x00":
            padding = self._raw.read(1)
        if padding:
            self._raw.prepend(padding)

    def read(self, size: int = -1) -> bytes:  # noqa: C901, PLR0912
        """Read up to *size* decompressed bytes, mirroring gzip semantics."""
        self._checkClosed()
        if size < 0:
            return self.readall()
        if size == 0 or self._size == self._pos:
            return b""

        chunks: list[bytes] = []
        remaining = size
        made_progress = False
        while remaining > 0:
            if self._decompressor.eof:
                # At EOF two buffers can hold unconsumed compressed bytes:
                # unused_data holds everything past the deflate end-of-stream
                # marker (member trailer and following members), while
                # unconsumed_tail additionally retains input that zlib never
                # reached because a max_length cap triggered EOF early. In the
                # latter case unused_data is a prefix of unconsumed_tail, so
                # returning both would duplicate bytes; return the tail only,
                # otherwise return the framed unused_data.
                if self._decompressor.unconsumed_tail:
                    self._raw.prepend(self._decompressor.unconsumed_tail)
                else:
                    self._raw.prepend(self._decompressor.unused_data)
                self._finish_member()
                self._new_member = True
                self._decompressor = self._new_decompressor()

            if self._new_member:
                self._crc = zlib.crc32(b"")
                self._stream_size = 0
                if _read_gzip_header(self._raw) is None:
                    self._size = self._pos
                    break
                self._new_member = False

            if self._decompressor.unconsumed_tail:
                compressed = self._decompressor.unconsumed_tail
            else:
                compressed = self._raw.read(READ_BUFFER_SIZE)
            output_limit = self._next_checkpoint_cap(remaining)
            try:
                uncompressed = self._decompressor.decompress(
                    compressed, output_limit
                )
            except zlib.error as exc:
                raise gzip.BadGzipFile(str(exc)) from exc

            if uncompressed:
                chunks.append(uncompressed)
                produced = len(uncompressed)
                remaining -= produced
                self._pos += produced
                self._crc = zlib.crc32(uncompressed, self._crc)
                self._stream_size += produced
                made_progress = True
                self._record_checkpoint()
                continue

            if not compressed and not self._decompressor.unconsumed_tail:
                if made_progress:
                    break
                raise EOFError(
                    "Compressed file ended before the "
                    "end-of-stream marker was reached"
                )

        return b"".join(chunks)

    def readinto(self, buffer: Any) -> int:
        """Read decompressed bytes into a pre-allocated writable buffer."""
        with memoryview(buffer) as view, view.cast("B") as byte_view:
            data = self.read(len(byte_view))
            byte_view[: len(data)] = data
        return len(data)

    def tell(self) -> int:
        """Return the current decompressed byte position."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:  # noqa: C901
        """Seek in the decompressed stream, reusing retained checkpoints."""
        self._checkClosed()
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._pos + offset
        elif whence == io.SEEK_END:
            if self._size < 0:
                while self.read(io.DEFAULT_BUFFER_SIZE):
                    pass
            target = self._size + offset
        else:
            msg = f"Invalid value for whence: {whence!r}"
            raise ValueError(msg)
        # Match gzip's DecompressReader: out-of-range targets are clamped to
        # the stream bounds instead of raising.
        if target < 0:
            target = 0
        if self._size >= 0 and target > self._size:
            target = self._size

        if target < self._pos:
            anchor = max(pos for pos in self._checkpoints if pos <= target)
            checkpoint = self._checkpoints[anchor]
            self._restore(checkpoint)
            self._discard_checkpoints_after(anchor)

        while self._pos < target:
            skipped = self.read(
                min(io.DEFAULT_BUFFER_SIZE, target - self._pos)
            )
            if not skipped:
                break
        return self._pos


def _validate_local_seekable_binary(fp: Any) -> None:
    """Reject anything outside the supported local immutable-file scope."""
    if not (
        hasattr(fp, "read")
        and hasattr(fp, "seek")
        and hasattr(fp, "tell")
    ):
        msg = (
            "indexed gzip mode requires a seekable binary stream with "
            "read(), seek() and tell()"
        )
        raise ValueError(msg)
    try:
        seekable = fp.seekable()
    except (AttributeError, OSError, ValueError) as exc:
        msg = "indexed gzip mode requires a seekable underlying stream"
        raise ValueError(msg) from exc
    if not seekable:
        msg = "indexed gzip mode requires a seekable underlying stream"
        raise ValueError(msg)
    mode = getattr(fp, "mode", None)
    if isinstance(mode, str) and "b" not in mode:
        msg = "indexed gzip mode requires a binary stream, got text mode"
        raise ValueError(msg)
    try:
        fileno = fp.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation) as exc:
        msg = "indexed gzip mode is limited to local on-disk gzip files"
        raise ValueError(msg) from exc
    try:
        is_regular = stat.S_ISREG(os.fstat(fileno).st_mode)
    except OSError as exc:
        msg = "indexed gzip mode is limited to local regular gzip files"
        raise ValueError(msg) from exc
    if not is_regular:
        msg = "indexed gzip mode is limited to local regular gzip files"
        raise ValueError(msg)


def open_indexed_gzip(
    fp: IO[bytes],
    index_interval: int = DEFAULT_INDEX_INTERVAL,
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
) -> io.BufferedReader:
    """Open a local seekable gzip stream with the session checkpoint index.

    Args:
        fp: Binary, seekable local file positioned at the first gzip member.
        index_interval: Target spacing, in decompressed bytes, between
            checkpoints.
        max_checkpoints: Hard upper bound on retained checkpoints.

    Returns:
        A buffered binary reader exposing the usual ``read``, ``readinto``,
        ``seek`` and ``tell`` interface.

    Raises:
        ValueError: If *fp* is not a seekable local binary file.
    """
    _validate_local_seekable_binary(fp)
    reader = IndexedGzipReader(
        fp,
        index_interval=index_interval,
        max_checkpoints=max_checkpoints,
    )
    return io.BufferedReader(reader)

#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Optional per-handle I/O accounting for streamed reads.

The counters separate the three layers a byte travels through on its way to
the caller, so a download quota can be charged on transport bytes alone,
without double counting buffered re-reads or decompression expansion:

* :attr:`IOStats.transport_bytes` - raw bytes pulled from the network
  (compressed, when the server compresses the body);
* :attr:`IOStats.decompressed_bytes` - bytes handed out by the (de)compression
  layer, i.e. what the application-level binary stream returns;
* :attr:`IOStats.text_chars` - characters produced by text decoding.

Every counter only advances when *new* data leaves the corresponding layer.
Re-reading bytes already held in a buffer, including after a seek that the
buffer can satisfy, never moves a counter; a seek that forces a fresh range
request moves :attr:`transport_bytes` again, because real bytes cross the
network again.

An :class:`IOStats` instance never retains the data itself, only three
integers. :meth:`IOStats.snapshot` returns an immutable copy of those
integers, safe to keep while the handle keeps streaming.
"""

from __future__ import annotations

import io
from collections import namedtuple
from collections.abc import Callable
from typing import IO, Any


class StatsSnapshot(
    namedtuple("_StatsSnapshot", ("transport_bytes", "decompressed_bytes", "text_chars"))
):
    """Immutable point-in-time copy of the three I/O counters.

    Attributes:
        transport_bytes: Raw bytes read from the underlying transport.
        decompressed_bytes: Bytes returned by the decompression layer; equals
            the transport count when no (de)compression is in use.
        text_chars: Characters returned by text decoding; zero for binary reads.
    """

    __slots__ = ()


class IOStats:
    """Mutable per-handle I/O counters shared by every layer of one open handle.

    A single instance is meant to be threaded through the transport,
    compression and text wrappers created for one ``open()`` call. Each layer
    increments the counter it owns, so all read styles (``read``, ``read1``,
    ``readinto``, ``readline`` and iteration) are metered by the same code
    path and cannot disagree.

    Args:
        on_update: Optional zero-argument callback invoked after a successful
            read advances at least one counter. The callback receives no
            arguments; read :meth:`snapshot` inside it. A callback that
            raises propagates its exception to the reader *after* the read
            already completed and the counters advanced, so the read is never
            re-executed and the bytes are not lost or double counted.
    """

    def __init__(self, on_update: Callable[[], None] | None = None) -> None:
        self._transport_bytes = 0
        self._decompressed_bytes = 0
        self._text_chars = 0
        self._on_update = on_update

    @property
    def transport_bytes(self) -> int:
        """Return the cumulative raw bytes pulled from the transport."""
        return self._transport_bytes

    @property
    def decompressed_bytes(self) -> int:
        """Return the cumulative bytes produced by the decompression layer."""
        return self._decompressed_bytes

    @property
    def text_chars(self) -> int:
        """Return the cumulative characters produced by text decoding."""
        return self._text_chars

    def snapshot(self) -> StatsSnapshot:
        """Return an immutable copy of the current counters.

        The snapshot does not reference any streamed content and stays valid
        after the handle is closed or keeps reading.
        """
        return StatsSnapshot(
            self._transport_bytes, self._decompressed_bytes, self._text_chars
        )

    def record_transport(self, num_bytes: int) -> None:
        """Account for ``num_bytes`` newly read from the transport, if positive."""
        if num_bytes > 0:
            self._transport_bytes += num_bytes
            self._notify()

    def record_decompressed(self, num_bytes: int) -> None:
        """Account for ``num_bytes`` newly produced by decompression, if positive."""
        if num_bytes > 0:
            self._decompressed_bytes += num_bytes
            self._notify()

    def record_text(self, num_chars: int) -> None:
        """Account for ``num_chars`` newly produced by text decoding, if positive."""
        if num_chars > 0:
            self._text_chars += num_chars
            self._notify()

    def _notify(self) -> None:
        if self._on_update is not None:
            self._on_update()


class _CountingReaderBase(io.IOBase):
    """Common machinery for a counting wrapper around a file-like stream.

    Initializing :class:`io.IOBase` gives the wrapper the standard context
    manager (``__enter__`` returns the wrapper, ``__exit__`` calls ``close``)
    and the ``closed`` bookkeeping, so it behaves like a real stream inside
    ``with`` blocks and :class:`~smart_open.utils.FileLikeProxy`.
    """

    def __init__(self, raw: IO[Any], counter: Callable[[int], None]) -> None:
        super().__init__()
        self._raw = raw
        self._counter = counter

    def close(self) -> None:
        """Close the wrapped stream exactly once, like ``io.IOBase.close``."""
        if not self.closed:
            super().close()
            self._raw.close()

    def readable(self) -> bool:
        return self._raw.readable()

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return self._raw.seekable()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    @property
    def name(self) -> Any:
        return getattr(self._raw, "name", None)

    def __getattr__(self, attr: str) -> Any:
        # Only invoked when normal lookup fails; delegate the rest (isatty,
        # fileno, ...) to the wrapped stream. __getattr__ runs before
        # self._raw exists during __init__, guard against that recursion.
        try:
            raw = self.__dict__["_raw"]
        except KeyError as exc:
            raise AttributeError(attr) from exc
        return getattr(raw, attr)


class _CountingBinaryReader(_CountingReaderBase):
    """Wrap a binary stream, counting the bytes each successful read delivers.

    Every read style (``read``, ``read1``, ``readinto``, ``readline`` and
    iteration) increments the same counter by the number of bytes actually
    handed out, so they cannot disagree. Bytes re-read from a buffer below
    this wrapper are not seen here and therefore never move the counter.
    """

    def __iter__(self) -> Any:
        return self

    def read(self, size: int | None = -1) -> bytes:
        data = self._raw.read(size)
        self._counter(len(data))
        return data

    def read1(self, size: int | None = -1) -> bytes:
        read1 = getattr(self._raw, "read1", self._raw.read)
        data = read1(size)
        self._counter(len(data))
        return data

    def readinto(self, buffer: Any) -> int:
        num_bytes = self._raw.readinto(buffer)
        if num_bytes:
            self._counter(num_bytes)
        return num_bytes

    def readline(self, size: int | None = -1) -> bytes:
        line = self._raw.readline(size)
        self._counter(len(line))
        return line

    def __next__(self) -> bytes:
        line = self._raw.__next__()
        self._counter(len(line))
        return line


class _CountingTextReader(_CountingReaderBase):
    """Wrap a text stream, counting the characters each successful read delivers.

    Mirrors :class:`_CountingBinaryReader` but measures ``len(str)`` and has
    no ``readinto`` (text streams do not support it).
    """

    def read(self, size: int | None = -1) -> str:
        text = self._raw.read(size)
        self._counter(len(text))
        return text

    def readline(self, size: int | None = -1) -> str:
        line = self._raw.readline(size)
        self._counter(len(line))
        return line

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> str:
        line = self._raw.__next__()
        self._counter(len(line))
        return line


def wrap_binary_stream(stream: IO[bytes], stats: IOStats) -> IO[bytes]:
    """Count every binary byte ``stream`` delivers to the layer above it.

    Used for the bytes produced by the (de)compression layer, so the counter
    sees post-decompression sizes even when gzip expands the payload.
    """
    return _CountingBinaryReader(stream, stats.record_decompressed)


def wrap_text_stream(stream: IO[str], stats: IOStats) -> IO[str]:
    """Count every text character ``stream`` delivers to the caller."""
    return _CountingTextReader(stream, stats.record_text)  # type: ignore[return-value]

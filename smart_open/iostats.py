#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Optional per-handle I/O accounting.

The counters live outside the normal read path: nothing here is instantiated
unless the caller explicitly opts in through ``transport_params``, so an
ordinary :func:`smart_open.open` keeps its exact read, seek and close behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import wrapt

if TYPE_CHECKING:
    from collections.abc import Callable

    from smart_open._typing import FileObj

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IOStatsSnapshot:
    """Immutable point-in-time view of the three metered counters.

    Attributes:
        transport_bytes: Bytes pulled from the transport, including refetches
            after a seek but excluding bytes re-served from an in-memory buffer.
        decompressed_bytes: Bytes handed out by the (de)compression layer,
            which equals ``transport_bytes`` when no compression is involved.
        text_chars: Characters delivered by the text decoding layer; stays
            zero for binary handles.
    """

    transport_bytes: int = 0
    decompressed_bytes: int = 0
    text_chars: int = 0


class IOStatsRecorder:
    """Accumulates the I/O counters of one open handle.

    A single recorder is shared by the transport, decompression and decoding
    wrappers of the same handle, so the three counters describe one data flow.
    Only counter values are retained: the recorder never holds file contents.

    Args:
        on_update: Optional callback invoked with an immutable snapshot after
            every metered read. Exceptions raised by the callback are logged
            and swallowed, so a faulty callback cannot turn a successful read
            into a failure or make the read run twice.
    """

    def __init__(self, on_update: Callable[[IOStatsSnapshot], object] | None = None) -> None:
        self._transport_bytes = 0
        self._decompressed_bytes = 0
        self._text_chars = 0
        self._on_update = on_update

    @property
    def transport_bytes(self) -> int:
        """Cumulative bytes fetched from the transport layer."""
        return self._transport_bytes

    @property
    def decompressed_bytes(self) -> int:
        """Cumulative decompressed bytes produced for the handle."""
        return self._decompressed_bytes

    @property
    def text_chars(self) -> int:
        """Cumulative characters delivered in text mode."""
        return self._text_chars

    def add_transport(self, amount: int) -> None:
        """Account for ``amount`` bytes fetched from the transport."""
        if amount:
            self._transport_bytes += amount
            self._notify()

    def add_decompressed(self, amount: int) -> None:
        """Account for ``amount`` decompressed bytes produced."""
        if amount:
            self._decompressed_bytes += amount
            self._notify()

    def add_text(self, amount: int) -> None:
        """Account for ``amount`` delivered text characters."""
        if amount:
            self._text_chars += amount
            self._notify()

    def snapshot(self) -> IOStatsSnapshot:
        """Return an immutable copy of the current counter values."""
        return IOStatsSnapshot(self._transport_bytes, self._decompressed_bytes, self._text_chars)

    def _notify(self) -> None:
        if self._on_update is None:
            return
        try:
            self._on_update(self.snapshot())
        except Exception:
            logger.exception("io_stats callback failed; the read itself is unaffected")


class MeteredReader(wrapt.ObjectProxy):
    """Count the data delivered through every read method of one layer.

    Wraps a single stream in the open-handle stack and records the size of
    everything it hands to the caller. ``read``, ``read1``, ``readinto``,
    ``readline``, ``readlines`` and iteration share one counter, so repeated
    reads of buffered data are never counted again: only bytes actually
    crossing this layer are added, and nothing is counted when the wrapped
    read returns nothing or raises.

    Args:
        wrapped: The stream to meter; it is not otherwise altered.
        recorder: The handle-wide recorder that receives the counts.
        add: The recorder method that receives the size of each delivery.
    """

    _recorder: IOStatsRecorder = ...  # initialized before wrapt would store it on the wrapped stream
    _add: Callable[[int], None] = ...

    def __init__(
        self,
        wrapped: FileObj,
        recorder: IOStatsRecorder,
        add: Callable[[int], None],
    ) -> None:
        super().__init__(wrapped)
        self._recorder = recorder
        self._add = add

    @property
    def io_stats(self) -> IOStatsRecorder:
        """The recorder shared by every metered layer of the same handle."""
        return self._recorder

    def _record(self, amount: int) -> None:
        self._add(amount)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        """Read through the wrapped stream and count what is delivered."""
        data = self.__wrapped__.read(*args, **kwargs)
        if data:
            self._record(len(data))
        return data

    def read1(self, *args: Any, **kwargs: Any) -> Any:
        """Read through the wrapped stream and count what is delivered."""
        data = self.__wrapped__.read1(*args, **kwargs)
        if data:
            self._record(len(data))
        return data

    def readinto(self, buffer: Any) -> int:
        """Read into ``buffer`` and count the bytes the wrapped call wrote."""
        written = self.__wrapped__.readinto(buffer)
        if written:
            self._record(written)
        return written

    def readline(self, *args: Any, **kwargs: Any) -> Any:
        """Read one line through the wrapped stream and count it."""
        line = self.__wrapped__.readline(*args, **kwargs)
        if line:
            self._record(len(line))
        return line

    def readlines(self, *args: Any, **kwargs: Any) -> list[Any]:
        """Read all lines through the wrapped stream and count each one."""
        lines = self.__wrapped__.readlines(*args, **kwargs)
        for line in lines:
            self._record(len(line))
        return lines

    def __iter__(self) -> MeteredReader:
        """Iterate lines through this proxy so iteration is metered too."""
        return self

    def __next__(self) -> Any:
        """Return the next metered line, or stop at the end of the stream."""
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self) -> MeteredReader:
        """Enter the wrapped stream but keep the metering proxy in place."""
        self.__wrapped__.__enter__()
        return self

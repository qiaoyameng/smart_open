#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Explicit write sessions with separate finalization and publication steps."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Any, Protocol

import wrapt

if TYPE_CHECKING:
    from types import TracebackType


class _StagedBinaryIO(Protocol):
    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...

    def finish(self) -> None: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


class WriteSession(wrapt.ObjectProxy):
    """A writable file-like object whose destination is not published on flush.

    ``finish()`` closes encoding and compression wrappers, while ``commit()``
    makes the completed object visible in the underlying transport.
    """

    __binary: _StagedBinaryIO
    __state: str

    def __init__(self, outer: IO[Any], binary: _StagedBinaryIO) -> None:
        super().__init__(outer)
        self.__binary = binary
        self.__state = "active"

    @property
    def session_state(self) -> str:
        """Return the current session state."""
        return self.__state

    def finish(self) -> None:
        """Flush and close encoding and compression layers without committing."""
        if self.__state in ("finished", "committed", "aborted"):
            return
        self.__wrapped__.close()
        self.__state = "finished"

    def commit(self) -> None:
        """Finish the stream if needed and publish the completed object."""
        if self.__state == "committed":
            return
        if self.__state == "aborted":
            msg = "cannot commit an aborted write session"
            raise ValueError(msg)

        self.finish()
        self.__binary.commit()
        self.__state = "committed"

    def abort(self) -> None:
        """Discard the staged object without flushing encoding wrappers."""
        if self.__state == "aborted":
            return
        if self.__state == "committed":
            msg = "cannot abort a committed write session"
            raise ValueError(msg)

        self.__binary.abort()
        self.__state = "aborted"

    def close(self) -> None:
        """Discard an unresolved session; committed sessions are no-ops."""
        if self.__state == "committed":
            return
        self.abort()

    def writable(self) -> bool:
        """Return whether more data may be written to the session."""
        return self.__state == "active"

    def write(self, data: Any) -> Any:
        """Write through to the active stream."""
        if self.__state != "active":
            msg = f"cannot write to write session in {self.__state!r} state"
            raise ValueError(msg)
        return self.__wrapped__.write(data)

    def writelines(self, lines: Any) -> None:
        """Write lines through to the active stream."""
        if self.__state != "active":
            msg = f"cannot write to write session in {self.__state!r} state"
            raise ValueError(msg)
        self.__wrapped__.writelines(lines)

    def __enter__(self) -> WriteSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

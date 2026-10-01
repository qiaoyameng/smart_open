#
# Copyright (C) 2020 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Implements the transport for the file:// schema."""

from __future__ import annotations

import builtins
import contextlib
import io
import os.path
import tempfile
from typing import IO, TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from types import TracebackType

    from smart_open._typing import TransportParams

SCHEME = "file"
SUPPORTS_WRITE_SESSION = True

URI_EXAMPLES = (
    "./local/path/file",
    "~/local/path/file",
    "local/path/file",
    "./local/path/file.gz",
    "file:///home/user/file",
    "file:///home/user/file.bz2",
)


class _LocalUri(TypedDict):
    scheme: str
    uri_path: str


open = io.open


class StagedFile(io.BufferedIOBase):
    """Write to a temporary file and atomically publish it on commit."""

    name: str

    def __init__(self, destination: str) -> None:
        self._destination = destination
        directory = os.path.dirname(destination) or "."
        prefix = f".{os.path.basename(destination)}.tmp-"
        self._file = tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=directory,
            prefix=prefix,
            delete=False,
        )
        self._staging_path = self._file.name
        current_umask = os.umask(0)
        os.umask(current_umask)
        os.chmod(self._staging_path, 0o666 & ~current_umask)
        self.name = destination
        self._state = "active"

    def finish(self) -> None:
        """Flush and close the staging file without publishing it."""
        if self._state in ("finished", "committed", "aborted"):
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._state = "finished"

    def commit(self) -> None:
        """Publish the staging file at the destination path."""
        if self._state == "committed":
            return
        if self._state == "aborted":
            msg = "cannot commit an aborted staged file"
            raise ValueError(msg)

        self.finish()
        if os.path.islink(self._destination):
            destination = os.path.realpath(self._destination)
            os.replace(self._staging_path, destination)
        else:
            os.replace(self._staging_path, self._destination)
        self._state = "committed"

    def abort(self) -> None:
        """Close and remove the staging file without publishing it."""
        if self._state == "aborted":
            return
        if self._state == "committed":
            msg = "cannot abort a committed staged file"
            raise ValueError(msg)

        if not self._file.closed:
            self._file.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._staging_path)
        self._state = "aborted"

    def close(self) -> None:
        """Finish staging without publishing or removing the temporary file."""
        self.finish()

    @property
    def closed(self) -> bool:
        """Return whether the underlying staging file is closed."""
        return self._file.closed

    def writable(self) -> bool:
        """Return whether the staging file supports writing."""
        return self._state == "active"

    def write(self, b: Any) -> int:
        """Write bytes to the staging file."""
        if self._state == "aborted":
            return len(memoryview(b))
        return self._file.write(b)

    def flush(self) -> None:
        """Flush staging-file buffers without publishing."""
        self._file.flush()

    def tell(self) -> int:
        """Return the staging file position."""
        return self._file.tell()

    def seekable(self) -> bool:
        """Return whether the staging file supports seeking."""
        return self._file.seekable()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        """Seek in the staging file."""
        return self._file.seek(offset, whence)

    def __del__(self) -> None:
        if getattr(self, "_state", "aborted") == "active":
            with contextlib.suppress(Exception):
                self.abort()

    def __enter__(self) -> StagedFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._state != "committed":
            self.abort()


def parse_uri(uri_as_string: str) -> _LocalUri:
    """Parse a ``file://`` URI (or bare local path) into its path component."""
    local_path = extract_local_path(uri_as_string)
    return {"scheme": SCHEME, "uri_path": local_path}


def open_uri(  # noqa: ARG001  # interface conformance
    uri_as_string: str,
    mode: str,
    transport_params: TransportParams,
    *,
    write_session: bool = False,
) -> IO[Any]:
    """Open a local file URI using the given mode."""
    parsed_uri = parse_uri(uri_as_string)
    if write_session:
        if mode != "wb":
            msg = "write sessions require write mode"
            raise ValueError(msg)
        return StagedFile(parsed_uri["uri_path"])
    return builtins.open(parsed_uri["uri_path"], mode)  # noqa: PTH123  # mirrors builtins.open signature exactly


def extract_local_path(uri_as_string: str) -> str:
    """Return the user-expanded local filesystem path from `uri_as_string`."""
    if uri_as_string.startswith("file://"):
        local_path = uri_as_string.replace("file://", "", 1)
    else:
        local_path = uri_as_string
    return os.path.expanduser(local_path)  # noqa: PTH111  # pathlib collapses leading double slashes; preserve os.path semantics

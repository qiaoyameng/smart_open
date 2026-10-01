#
# Copyright (C) 2020 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Implements the transport for the file:// schema."""

from __future__ import annotations

import builtins
import io
import os
import os.path
import tempfile
from typing import IO, TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from smart_open._typing import TransportParams

SCHEME = "file"

SUPPORTS_WRITE_SESSIONS = True

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


def parse_uri(uri_as_string: str) -> _LocalUri:
    """Parse a ``file://`` URI (or bare local path) into its path component."""
    local_path = extract_local_path(uri_as_string)
    return {"scheme": SCHEME, "uri_path": local_path}


def open_uri(uri_as_string: str, mode: str, transport_params: TransportParams) -> IO[Any]:  # noqa: ARG001  # interface conformance
    """Open a local file URI using the given mode."""
    parsed_uri = parse_uri(uri_as_string)
    return builtins.open(parsed_uri["uri_path"], mode)  # noqa: PTH123  # mirrors builtins.open signature exactly


class StagedFileWriter(io.BufferedIOBase):
    """Write to a temporary file and publish it only when the session commits."""

    def __init__(self, path: str) -> None:
        self._target_path = path
        directory = os.path.dirname(path) or "."
        prefix = f".{os.path.basename(path)}."
        fd, self._temporary_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
        self._file = os.fdopen(fd, "wb")
        self._state = "open"
        self._bytes_written = 0

    @property
    def name(self) -> str:
        """Return the destination path, even while bytes are still in the temporary file."""
        return self._target_path

    @property
    def temporary_name(self) -> str:
        """Return the path of the temporary staging file."""
        return self._temporary_path

    @property
    def closed(self) -> bool:
        """Return whether the staged stream is closed."""
        return self._file.closed

    def writable(self) -> bool:
        """Return whether the staging file accepts writes."""
        return self._file.writable()

    def seekable(self) -> bool:
        """Propagate seek support from the staging file."""
        return self._file.seekable()

    def tell(self) -> int:
        """Propagate the current staging-file position."""
        if self._file.closed:
            return self._bytes_written
        return self._file.tell()

    def write(self, b: Any) -> int:
        """Write bytes to the staging file."""
        if self._state != "open":
            msg = "write to finished write session"
            raise ValueError(msg)
        bytes_written = self._file.write(b)
        self._bytes_written += bytes_written
        return bytes_written

    def flush(self) -> None:
        """Flush Python buffers to the staging file."""
        self._file.flush()

    def finish(self) -> None:
        """Flush and close the staging file without publishing it."""
        if self._state == "finished":
            return
        if self._state in ("committed", "aborted"):
            msg = f"cannot finish a {self._state} write session"
            raise ValueError(msg)
        self._file.flush()
        self._file.close()
        self._state = "finished"

    def commit(self) -> None:
        """Atomically replace the destination with the staged file."""
        if self._state == "committed":
            return
        if self._state == "aborted":
            msg = "cannot commit an aborted write session"
            raise ValueError(msg)
        if self._state == "open":
            self.finish()
        if self._state != "finished":
            msg = "cannot commit a write session before it is finished"
            raise ValueError(msg)
        os.replace(self._temporary_path, self._target_path)
        self._state = "committed"

    def abort(self) -> None:
        """Discard and remove the staging file."""
        if self._state == "aborted":
            return
        if self._state == "committed":
            msg = "cannot abort a committed write session"
            raise ValueError(msg)
        temporary_path = self._temporary_path
        if not self._file.closed:
            self._file.close()
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        self._state = "aborted"


def open_write_session_uri(
    uri_as_string: str,
    mode: str,
    transport_params: TransportParams,  # noqa: ARG001  # interface conformance
) -> IO[Any]:
    """Open a local write session backed by a temporary file."""
    if mode != "wb":
        msg = f"write sessions require write mode, got {mode!r}"
        raise ValueError(msg)
    parsed_uri = parse_uri(uri_as_string)
    return StagedFileWriter(parsed_uri["uri_path"])


def extract_local_path(uri_as_string: str) -> str:
    """Return the user-expanded local filesystem path from `uri_as_string`."""
    if uri_as_string.startswith("file://"):
        local_path = uri_as_string.replace("file://", "", 1)
    else:
        local_path = uri_as_string
    return os.path.expanduser(local_path)  # noqa: PTH111  # pathlib collapses leading double slashes; preserve os.path semantics

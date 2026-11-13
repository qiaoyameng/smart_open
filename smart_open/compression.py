#
# Copyright (C) 2020 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#
"""Implements the compression layer of the `smart_open` library."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smart_open._typing import CompressionKwargs, Compressor

logger = logging.getLogger(__name__)

_COMPRESSOR_REGISTRY: dict[str, Compressor] = {}

NO_COMPRESSION = "disable"
"""Use no compression. Read/write the data as-is."""
INFER_FROM_EXTENSION = "infer_from_extension"
"""Determine the compression to use from the file extension.

See get_supported_extensions().
"""
DETECT_FROM_CONTENT = "detect"
"""Determine the compression to use by sniffing the first bytes of the stream.

The stream does not need to be seekable: every byte consumed while sniffing is
replayed to the (de)compression layer.  See :func:`compression_wrapper`.
"""

#
# Policies controlling what happens when detection finds something unexpected.
#
CONFLICT_WARN = "warn"
"""Trust the detected content and log a warning when it contradicts the extension."""
CONFLICT_STRICT = "strict"
"""Raise ``ValueError`` when the detected content contradicts the extension."""
TRUNCATED_ERROR = "error"
"""Raise ``EOFError`` when the stream ends in the middle of a compression signature."""
TRUNCATED_PLAIN = "plain"
"""Treat an unfinished compression signature as ordinary, uncompressed data."""

#
# Magic-byte signatures of the registered formats, mapped to their registry
# extensions.  Signatures must be pairwise unambiguous: no signature may be a
# strict prefix of another one.  Keeping this table separate from the registry
# means externally registered custom extensions simply never get sniffed.
#
_MAGIC_SIGNATURES: dict[str, bytes] = {
    ".gz": b"\x1f\x8b",
    ".bz2": b"BZh",
    ".xz": b"\xfd7zXZ\x00",
    ".zst": b"\x28\xb5\x2f\xfd",
    ".lz4": b"\x04\x22\x4d\x18",
}


def get_supported_compression_types() -> list[str]:
    """Return the list of supported compression types available to open.

    See compression paratemeter to smart_open.open().
    """
    return [NO_COMPRESSION, INFER_FROM_EXTENSION, DETECT_FROM_CONTENT, *get_supported_extensions()]


def get_supported_extensions() -> list[str]:
    """Return the list of file extensions for which we have registered compressors."""
    return sorted(_COMPRESSOR_REGISTRY.keys())


def register_compressor(ext: str, callback: Compressor) -> None:
    """Register a callback for transparently decompressing files with a specific extension.

    Args:
        ext: The extension.  Must include the leading period, e.g. `.gz`.
        callback: The callback.  It must accept two positional arguments, file_obj and mode,
            and is recommended to also accept **kwargs so that whatever the caller passes
            via smart_open.open(..., compression_kwargs={...}) reaches the underlying
            library unchanged.  Callbacks with the legacy (file_obj, mode) signature still
            work, but will raise TypeError if the caller supplies compression_kwargs
            that the callback doesn't declare.

    Raises:
        ValueError: If `ext` does not start with a period.

    Example:
        Instruct smart_open to use the `lzma` module whenever opening a file
        with a .xz extension (see README.md for the complete example showing I/O):

        >>> def _handle_xz(file_obj, mode, **kwargs):
        ...     import lzma
        ...
        ...     return lzma.open(filename=file_obj, mode=mode, **kwargs)
        >>>
        >>> register_compressor(".xz", _handle_xz)

        This is just an example: `lzma` is in the standard library and is registered by default.
    """
    if not (ext and ext[0] == "."):
        msg = f"ext must be a string starting with ., not {ext!r}"
        raise ValueError(msg)
    ext = ext.lower()
    if ext in _COMPRESSOR_REGISTRY:
        logger.warning("overriding existing compression handler for %r", ext)
    _COMPRESSOR_REGISTRY[ext] = callback


def _maybe_wrap_buffered(file_obj: Any, mode: str) -> IO[bytes]:
    # https://github.com/piskvorky/smart_open/issues/760#issuecomment-1553971657
    result = file_obj
    if "b" in mode and "w" in mode:
        result = io.BufferedWriter(result)
    elif "b" in mode and "r" in mode:
        result = io.BufferedReader(result)
    return result


def _handle_bz2(file_obj: IO[bytes], mode: str, **kwargs: Any) -> IO[Any]:
    import bz2

    result = bz2.open(filename=file_obj, mode=mode, **kwargs)  # noqa: SIM115  # returns the file object to caller
    return _maybe_wrap_buffered(result, mode)


def _handle_gzip(file_obj: IO[bytes], mode: str, **kwargs: Any) -> IO[Any]:
    import gzip

    result = gzip.open(filename=file_obj, mode=mode, **kwargs)  # noqa: SIM115  # returns the file object to caller
    return _maybe_wrap_buffered(result, mode)


def _handle_zstd(file_obj: IO[bytes], mode: str, **kwargs: Any) -> IO[Any]:
    import sys

    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        from backports import zstd
    # dynamic **kwargs cannot be matched against zstd.open()'s overloads, so go through Any
    zstd_open: Any = zstd.open
    result = zstd_open(file_obj, mode=mode, **kwargs)
    return _maybe_wrap_buffered(result, mode)


def _handle_xz(file_obj: IO[bytes], mode: str, **kwargs: Any) -> IO[Any]:
    import lzma

    result = lzma.open(filename=file_obj, mode=mode, **kwargs)  # noqa: SIM115  # returns the file object to caller
    return _maybe_wrap_buffered(result, mode)


def _handle_lz4(file_obj: IO[bytes], mode: str, **kwargs: Any) -> IO[Any]:
    import lz4.frame

    result = lz4.frame.open(file_obj, mode=mode, **kwargs)
    return _maybe_wrap_buffered(result, mode)


class _ReplayStream(io.RawIOBase):
    """Re-emit the sniffed prefix before transparently forwarding to `source`.

    The wrapped source is allowed to be non-seekable.  Bytes already consumed
    during magic sniffing are buffered here and handed back, byte for byte,
    before reads fall through to the underlying stream, so no input is ever
    lost.

    Closing this stream closes `source` too.  :meth:`detach_source` opts out of
    that on failure paths, leaving the original source with its caller.
    """

    def __init__(self, source: IO[bytes], prefix: bytes) -> None:
        super().__init__()
        self._source = source
        self._prefix = bytearray(prefix)
        self._detached = False
        self._name = getattr(source, "name", None)

    @property
    def name(self) -> Any:
        return self._name

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return bool(getattr(self._source, "seekable", lambda: False)())

    def detach_source(self) -> IO[bytes]:
        """Return the underlying source and drop the buffered prefix.

        Used on failure paths so that closing this helper cannot reach into a
        source the caller still owns.
        """
        self._prefix = bytearray()
        self._detached = True
        return self._source

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            msg = "read from closed file"
            raise ValueError(msg)
        if size is None or size < 0:
            head = bytes(self._prefix)
            self._prefix = bytearray()
            tail = self._source.read(-1)
            return head + (tail or b"")
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        if self._prefix:
            head = bytes(self._prefix[:remaining])
            del self._prefix[: len(head)]
            chunks.append(head)
            remaining -= len(head)
        if remaining:
            tail = self._source.read(remaining)
            if tail:
                chunks.append(tail)
        return b"".join(chunks)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        count = len(data)
        buffer[:count] = data
        return count

    def close(self) -> None:
        if not self.closed:
            source = None if self._detached else self._source
            self._prefix = bytearray()
            try:
                if source is not None:
                    source.close()
            finally:
                super().close()


def _is_seekable(file_obj: IO[bytes]) -> bool:
    """Best-effort check whether `file_obj` supports seeking."""
    seekable = getattr(file_obj, "seekable", None)
    if seekable is None:
        return False
    try:
        return bool(seekable())
    except (OSError, ValueError):
        return False


def _detect_from_content(
    file_obj: IO[bytes],
    filename: str | None,
    detect_conflict: str,
    detect_truncated: str,
) -> tuple[str, IO[bytes]]:
    """Sniff the leading bytes of `file_obj` and decide how to decode it.

    Returns a tuple ``(extension, stream)`` where ``extension`` is the registry
    key of the detected compressor (or an empty string for plain data) and
    ``stream`` is a readable object that yields every byte of the input,
    including the bytes consumed during sniffing.

    The sniff loop distinguishes three situations once the data stops coming:

    * a complete magic signature matched: the corresponding compressor;
    * the buffered bytes are a strict prefix of one or more signatures but no
      signature matched: a truncated container, handled per `detect_truncated`;
    * anything else (including an empty stream): plain, uncompressed data.
    """
    sniffed = bytearray()
    extension = ""
    max_signature = max(map(len, _MAGIC_SIGNATURES.values()))
    while len(sniffed) < max_signature:
        chunk = file_obj.read(1)
        if not chunk:
            break
        sniffed += chunk
        prefix = bytes(sniffed)
        for candidate_extension, magic in _MAGIC_SIGNATURES.items():
            if prefix == magic:
                # A complete, unambiguous signature: decide immediately.
                extension = candidate_extension
                break
            if magic.startswith(prefix):
                # Still a possible container: keep reading before deciding.
                break
        else:
            # The bytes seen so far cannot start any known container, so no
            # later byte can rescue the decision: this is plain data.
            break

    if (
        sniffed
        and not extension
        and any(magic.startswith(bytes(sniffed)) for magic in _MAGIC_SIGNATURES.values())
    ):
        #
        # The stream ended while the buffered bytes were still a strict prefix
        # of a signature: a truncated container rather than plain data.
        #
        if detect_truncated == TRUNCATED_PLAIN:
            extension = ""
        else:
            msg = (
                f"compression detection hit end of stream after {len(sniffed)} byte(s); "
                f"{bytes(sniffed)!r} is an incomplete compression signature"
            )
            raise EOFError(msg)

    if _is_seekable(file_obj):
        #
        # Rewind: the codec then starts exactly where detection did, with no
        # extra wrapper in the way.
        #
        file_obj.seek(0)
        stream = file_obj
    else:
        #
        # The source cannot rewind.  Hand back the bytes already consumed and
        # continue reading from the source's current position.
        #
        stream = _ReplayStream(file_obj, bytes(sniffed))

    if filename:
        try:
            name_extension = Path(filename).suffix.lower()
        except (AttributeError, TypeError):
            name_extension = ""
        if name_extension in _COMPRESSOR_REGISTRY and name_extension != extension:
            detail = extension or "uncompressed data"
            message = (
                f"compression mismatch: {filename!r} implies {name_extension}, "
                f"but the content looks like {detail}"
            )
            if detect_conflict == CONFLICT_STRICT:
                if isinstance(stream, _ReplayStream):
                    # Non-seekable source: leave it open for the caller.
                    stream.detach_source()
                    stream.close()
                raise ValueError(message)
            logger.warning(message)

    return extension, stream


def compression_wrapper(
    file_obj: IO[Any],
    mode: str,
    compression: str = INFER_FROM_EXTENSION,
    filename: str | None = None,
    compression_kwargs: CompressionKwargs | None = None,
    detect_conflict: str = CONFLICT_WARN,
    detect_truncated: str = TRUNCATED_ERROR,
) -> IO[Any]:
    """Wrap `file_obj` with an appropriate [de]compression mechanism based on its file extension.

    If the filename extension isn't recognized, simply return the original `file_obj` unchanged.

    `file_obj` must either be a filehandle object, or a class which behaves like one.

    If `filename` is specified, it will be used to extract the extension.
    If not, the `file_obj.name` attribute is used as the filename.

    If `compression_kwargs` is specified, its contents are forwarded as keyword
    arguments to the registered compressor callback.

    When `compression` is :data:`DETECT_FROM_CONTENT`, the format is identified
    from the leading bytes of `file_obj` rather than the filename.  Seekable
    sources are rewound; non-seekable sources get a replay stream so that the
    sniffed bytes still reach the codec.  `detect_conflict` and
    `detect_truncated` tune the mismatch and truncated-signature policies.
    """
    if compression == NO_COMPRESSION:
        return file_obj

    if compression == DETECT_FROM_CONTENT:
        if detect_conflict not in (CONFLICT_WARN, CONFLICT_STRICT):
            msg = f"invalid detect_conflict policy: {detect_conflict!r}"
            raise ValueError(msg)
        if detect_truncated not in (TRUNCATED_ERROR, TRUNCATED_PLAIN):
            msg = f"invalid detect_truncated policy: {detect_truncated!r}"
            raise ValueError(msg)
        if "r" not in mode:
            #
            # Nothing to sniff before writing.  Fall back to extension-based
            # behavior rather than silently writing uncompressed data.
            #
            logger.warning(
                "compression='detect' only applies when reading; inferring from the extension instead"
            )
            compression = INFER_FROM_EXTENSION
        else:
            if mode.endswith("+"):
                msg = f"transparent (de)compression unsupported for mode {mode!r}"
                raise ValueError(msg)
            detected_extension, replay_stream = _detect_from_content(
                file_obj,
                filename,
                detect_conflict,
                detect_truncated,
            )
            if not detected_extension:
                # Plain data: still route through a buffered reader so the
                # returned object matches the buffered behavior of codec paths.
                return _maybe_wrap_buffered(replay_stream, mode)
            try:
                callback = _COMPRESSOR_REGISTRY[detected_extension]
                return callback(replay_stream, mode, **(compression_kwargs or {}))
            except BaseException:
                #
                # The codec failed to take ownership (e.g. a malformed header
                # during initialization).  Keep the original source open and
                # let the caller apply the usual close-ownership rules.
                #
                if isinstance(replay_stream, _ReplayStream):
                    replay_stream.detach_source()
                    replay_stream.close()
                raise

    if compression == INFER_FROM_EXTENSION:
        try:
            inferred_name = (filename or file_obj.name).lower()
        except (AttributeError, TypeError):
            logger.warning(
                "unable to transparently decompress %r because it seems to lack a string-like .name", file_obj
            )
            return file_obj
        compression = Path(inferred_name).suffix

    if compression in _COMPRESSOR_REGISTRY and mode.endswith("+"):
        msg = f"transparent (de)compression unsupported for mode {mode!r}"
        raise ValueError(msg)

    try:
        callback = _COMPRESSOR_REGISTRY[compression]
    except KeyError:
        return file_obj

    return callback(file_obj, mode, **(compression_kwargs or {}))


#
# NB. avoid using lambda here to make stack traces more readable.
#
register_compressor(".bz2", _handle_bz2)
register_compressor(".gz", _handle_gzip)
register_compressor(".zst", _handle_zstd)
register_compressor(".xz", _handle_xz)
register_compressor(".lz4", _handle_lz4)

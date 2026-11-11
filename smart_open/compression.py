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
"""Determine the compression from a limited prefix of the data itself.

The first few bytes are inspected for the magic signatures of gzip, bz2 and
xz.  If none matches, the data is read as-is.  Only supported for reading.
"""
DETECT_STRICT = "detect_strict"
"""Like :data:`DETECT_FROM_CONTENT`, but reject an extension/content mismatch.

Raise :class:`ValueError` when the filename extension names a registered
codec whose magic signature is not found in the data (or vice versa).
"""

#
# Magic signatures used by DETECT_FROM_CONTENT.  Keep the tuples ordered from
# longest to shortest so prefix matching is unambiguous.
#
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xfd7zXZ\x00", ".xz"),
    (b"BZh", ".bz2"),
    (b"\x1f\x8b", ".gz"),
)
#: Prefix long enough to distinguish every supported magic signature.
_DETECT_PREFIX_SIZE = max(len(magic) for magic, _ in _MAGIC_SIGNATURES)


def get_supported_compression_types() -> list[str]:
    """Return the list of supported compression types available to open.

    See compression paratemeter to smart_open.open().
    """
    return [
        NO_COMPRESSION,
        INFER_FROM_EXTENSION,
        DETECT_FROM_CONTENT,
        DETECT_STRICT,
        *get_supported_extensions(),
    ]


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


class _PrefixReplay(io.RawIOBase):
    """Replay bytes already consumed from a non-seekable stream.

    Detection reads a bounded prefix of the underlying stream, but every
    probed byte still has to reach the final (de)compression layer.  This
    wrapper serves the buffered prefix first and then transparently falls
    through to the underlying stream, so no byte is lost and no seeking is
    required.
    """

    def __init__(self, stream: IO[bytes], prefix: bytes) -> None:
        super().__init__()
        self._stream = stream
        self._prefix = bytes(prefix)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        if not self._prefix:
            chunk = self._stream.read(len(buffer))
            if not chunk:
                return 0
            buffer[: len(chunk)] = chunk
            return len(chunk)
        chunk = self._prefix[: len(buffer)]
        size = len(chunk)
        buffer[:size] = chunk
        self._prefix = self._prefix[size:]
        return size

    def close(self) -> None:
        if not self.closed:
            super().close()
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()


def _identify_prefix(prefix: bytes, stream_exhausted: bool) -> str | None:
    """Classify a probed prefix.

    Returns the registered extension of a positively identified container,
    or None for ordinary data / data still too short to decide.

    Raises:
        EOFError: If the stream ended while its prefix could only be the
            beginning of a container magic signature, i.e. a truncated
            container rather than ordinary data.
    """
    for magic, extension in _MAGIC_SIGNATURES:
        if prefix.startswith(magic):
            return extension
        if magic.startswith(prefix) and prefix:
            # The prefix is a proper prefix of a magic signature.  More
            # bytes could complete it; if there are no more bytes, this is a
            # truncated container, not ordinary data.
            if stream_exhausted:
                msg = (
                    f"stream ended after {len(prefix)} byte(s) that only match the beginning "
                    f"of a {extension} container (magic {magic!r}); the data looks truncated"
                )
                raise EOFError(msg)
            return None
    return None


def _detect_compression(
    file_obj: IO[bytes],
    filename: str | None,
    strict: bool,  # noqa: FBT001  # internal dispatch flag, not user-facing config
) -> tuple[str, IO[bytes]]:
    """Probe up to ``_DETECT_PREFIX_SIZE`` bytes and pick a read codec.

    Returns the extension of the codec to use (possibly one that is not
    registered, in which case the caller reads the data as-is) and a stream
    that replays every probed byte before falling through to ``file_obj``.
    """
    prefix = b""
    while len(prefix) < _DETECT_PREFIX_SIZE:
        chunk = file_obj.read(_DETECT_PREFIX_SIZE - len(prefix))
        if not chunk:
            break
        prefix += chunk
    stream_exhausted = len(prefix) < _DETECT_PREFIX_SIZE

    detected = _identify_prefix(prefix, stream_exhausted)
    if detected is None:
        detected = NO_COMPRESSION
    replay: IO[bytes] = _PrefixReplay(file_obj, prefix)

    expected: str | None = None
    if filename is not None:
        suffix = Path(filename).suffix.lower()
        if suffix in _COMPRESSOR_REGISTRY:
            expected = suffix

    if expected is not None and expected != detected:
        msg = (
            f"filename {filename!r} implies {expected} compression, but the content begins "
            f"with {prefix[:_DETECT_PREFIX_SIZE]!r}; detected {detected!r}"
        )
        if strict:
            replay.close()
            raise ValueError(msg)
        logger.warning("%s; using the detected codec instead", msg)
    elif detected != NO_COMPRESSION and detected not in _COMPRESSOR_REGISTRY:
        logger.warning(
            "detected %s compression but no handler is registered; reading the data as-is", detected
        )

    return detected, replay


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


def compression_wrapper(
    file_obj: IO[Any],
    mode: str,
    compression: str = INFER_FROM_EXTENSION,
    filename: str | None = None,
    compression_kwargs: CompressionKwargs | None = None,
) -> IO[Any]:
    """Wrap `file_obj` with an appropriate [de]compression mechanism based on its file extension.

    If the filename extension isn't recognized, simply return the original `file_obj` unchanged.

    `file_obj` must either be a filehandle object, or a class which behaves like one.

    If `filename` is specified, it will be used to extract the extension.
    If not, the `file_obj.name` attribute is used as the filename.

    With `compression` set to "detect" (or "detect_strict"), the extension is
    ignored and the codec is chosen by inspecting a bounded prefix of the
    stream instead (see DETECT_FROM_CONTENT).  The stream does not need to be
    seekable: every probed byte is replayed to the decompression layer.

    If `compression_kwargs` is specified, its contents are forwarded as keyword
    arguments to the registered compressor callback.
    """
    if compression == NO_COMPRESSION:
        return file_obj
    if compression in (DETECT_FROM_CONTENT, DETECT_STRICT):
        if "r" not in mode:
            msg = f"compression {compression!r} is only supported for reading, not mode {mode!r}"
            raise ValueError(msg)
        if mode.endswith("+"):
            msg = f"transparent (de)compression unsupported for mode {mode!r}"
            raise ValueError(msg)
        compression, file_obj = _detect_compression(file_obj, filename, strict=(compression == DETECT_STRICT))
        if compression == NO_COMPRESSION:
            return file_obj
        try:
            return _COMPRESSOR_REGISTRY[compression](file_obj, mode, **(compression_kwargs or {}))
        except BaseException:
            # The detect layer already consumed (and owns) the raw stream;
            # make sure it is finalized if the decompressor fails to start.
            file_obj.close()
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

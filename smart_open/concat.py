#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#

"""Read multiple objects as one concatenated text stream.

A log (or any large text dataset) is often split into many objects, e.g.
``s3://bucket/logs/0001`` ... ``s3://bucket/logs/9999``.  This module lets
you read such a collection as if it were a single file:

- the path list may be produced lazily (any iterable, e.g. a generator);
- the next object is opened only after the previous one is fully consumed;
- empty objects are skipped automatically;
- ``readline``/iteration join text across object boundaries when an object
  does not end with a newline;
- ``tell`` reports the logical offset: the number of characters returned
  so far, counted across all objects.

Decoding at object boundaries
-----------------------------

By default (``reset_decoder=True``) every object is decoded independently:
the incremental decoder is flushed and reset at each object boundary.  A
multi-byte character that is truncated at the end of an object is *not*
completed by bytes from the next object -- with ``errors='strict'`` this
raises :class:`UnicodeDecodeError`, otherwise the ``errors`` handler
applies.  Pass ``reset_decoder=False`` to share a single decoder across
the whole concatenation, allowing a character split across two objects to
be reassembled.
"""

from __future__ import annotations

import codecs
import io
import locale
import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import smart_open.compression as so_compression
from smart_open.smart_open_lib import open as so_open

if TYPE_CHECKING:
    from smart_open._typing import CompressionKwargs, TransportParams, Uri

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = locale.getpreferredencoding(do_setlocale=False)

_CHUNK_SIZE = io.DEFAULT_BUFFER_SIZE


class ConcatError(OSError):
    """Raised when an object in the concatenation cannot be opened.

    Attributes:
        index: 0-based position of the failing object in the path iterable.
        offset: logical offset (characters already returned to the caller)
            at the moment of the failure.
        path: the URI that failed to open.
    """

    def __init__(self, index: int, offset: int, path: Any, cause: BaseException) -> None:
        self.index = index
        self.offset = offset
        self.path = path
        super().__init__(
            f"failed to open object #{index} ({path!r}) at logical offset {offset}: {cause}"
        )


class ConcatReader(io.TextIOBase):
    """Read a sequence of objects as one concatenated text stream.

    Instances are usually created via :func:`concat`.  The stream is
    forward-only: ``seek`` is not supported.  ``tell`` returns the number
    of characters returned to the caller so far, so ``read`` and line
    iteration may be interleaved freely -- both consume the same logical
    position.
    """

    def __init__(
        self,
        paths: Iterable[Uri],
        encoding: str | None = None,
        errors: str | None = None,
        reset_decoder: bool = True,  # noqa: FBT001,FBT002  # public API
        compression: str = so_compression.INFER_FROM_EXTENSION,
        compression_kwargs: CompressionKwargs | None = None,
        transport_params: TransportParams | None = None,
    ) -> None:
        super().__init__()
        self._path_iter: Iterator[Uri] = iter(paths)
        self._encoding = encoding or DEFAULT_ENCODING
        self._errors = errors or "strict"
        self._reset_decoder = reset_decoder
        self._compression = compression
        self._compression_kwargs = compression_kwargs
        self._transport_params = transport_params

        self._decoder = codecs.getincrementaldecoder(self._encoding)(self._errors)
        self._raw: io.IOBase | None = None  # currently open object, binary mode
        self._index = -1  # 0-based index of the currently open object
        self._offset = 0  # logical offset: characters returned to the caller
        self._buffer = ""  # decoded but not yet returned text
        self._exhausted = False  # all paths consumed

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding

    @property
    def errors(self) -> str:  # type: ignore[override]
        return self._errors

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        """Return the logical offset: characters returned so far."""
        self._check_closed()
        return self._offset

    def read(self, size: int = -1) -> str:
        self._check_closed()
        if size is None or size < 0:
            while not self._exhausted:
                self._fill()
            result, self._buffer = self._buffer, ""
        else:
            while len(self._buffer) < size and not self._exhausted:
                self._fill()
            result, self._buffer = self._buffer[:size], self._buffer[size:]
        self._offset += len(result)
        return result

    def readline(self, size: int = -1) -> str:
        self._check_closed()
        limit = size if (size is not None and size >= 0) else None
        while True:
            newline_at = self._buffer.find("\n")
            if (
                newline_at >= 0
                or self._exhausted
                or (limit is not None and len(self._buffer) >= limit)
            ):
                end = newline_at + 1 if newline_at >= 0 else len(self._buffer)
                if limit is not None:
                    end = min(end, limit)
                result, self._buffer = self._buffer[:end], self._buffer[end:]
                self._offset += len(result)
                return result
            self._fill()

    def close(self) -> None:
        """Close the stream.

        Only the currently open object is closed.  Remaining paths are
        never pulled from the iterable, so later objects are never opened.
        """
        if not self.closed:
            raw, self._raw = self._raw, None
            if raw is not None:
                raw.close()
        super().close()

    def _check_closed(self) -> None:
        if self.closed:
            msg = "I/O operation on closed ConcatReader"
            raise ValueError(msg)

    def _fill(self) -> None:
        """Perform one read/advance step: decode a chunk or move on."""
        if self._raw is None:
            self._open_next()
            if self._exhausted:
                return
        assert self._raw is not None
        chunk = self._raw.read(_CHUNK_SIZE)
        if chunk:
            self._buffer += self._decoder.decode(chunk)
            return
        #
        # End of the current object.  With reset_decoder=True each object
        # is decoded independently: flush the decoder (raising on a
        # truncated character under errors='strict') and start the next
        # object with a fresh decoder state.
        #
        if self._reset_decoder:
            self._buffer += self._decoder.decode(b"", final=True)
            self._decoder.reset()
        self._raw.close()
        self._raw = None

    def _open_next(self) -> None:
        try:
            path = next(self._path_iter)
        except StopIteration:
            if not self._reset_decoder:
                self._buffer += self._decoder.decode(b"", final=True)
            self._exhausted = True
            return
        self._index += 1
        logger.debug("opening object #%d: %r", self._index, path)
        try:
            self._raw = so_open(
                path,
                "rb",
                compression=self._compression,
                compression_kwargs=self._compression_kwargs,
                transport_params=self._transport_params,
            )
        except Exception as exc:
            raise ConcatError(self._index, self._offset, path, exc) from exc


def concat(
    paths: Iterable[Uri],
    encoding: str | None = None,
    errors: str | None = None,
    reset_decoder: bool = True,  # noqa: FBT001,FBT002  # public API
    compression: str = so_compression.INFER_FROM_EXTENSION,
    compression_kwargs: CompressionKwargs | None = None,
    transport_params: TransportParams | None = None,
) -> ConcatReader:
    """Open a sequence of objects for reading as one concatenated text stream.

    Args:
        paths: URIs/paths of the objects to concatenate.  Any iterable is
            accepted; items are pulled lazily, so a generator works and the
            next object is opened only after the previous one is exhausted.
        encoding: Decode each object with this encoding.  Defaults to the
            locale's preferred encoding, like :func:`smart_open.open`.
        errors: Decoding error handler, e.g. ``'strict'`` (default),
            ``'replace'``, ``'ignore'``.
        reset_decoder: If True (default), decode each object independently:
            the decoder is flushed and reset at every object boundary, so a
            character truncated at the end of an object is never completed
            by bytes from the next object.  If False, share one decoder
            across the whole concatenation.
        compression: Compression handling, as in :func:`smart_open.open`.
        compression_kwargs: Forwarded to the compressor callback.
        transport_params: Forwarded to the transport layer.

    Returns:
        A forward-only text stream supporting ``read``, ``readline``,
        iteration, ``tell`` and ``close``.  ``tell`` returns the logical
        offset (characters returned so far across all objects).

    Raises:
        ConcatError: If an object fails to open mid-stream.  Carries the
            object ``index`` and the logical ``offset`` at the failure.
    """
    return ConcatReader(
        paths,
        encoding=encoding,
        errors=errors,
        reset_decoder=reset_decoder,
        compression=compression,
        compression_kwargs=compression_kwargs,
        transport_params=transport_params,
    )

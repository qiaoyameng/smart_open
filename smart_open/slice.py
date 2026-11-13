"""Compatibility wrapper for :mod:`smart_open.slices`."""

from smart_open.slices import (  # noqa: F401  # compatibility re-exports
    ReadOnlySlice,
    ReadonlySlice,
    Slice,
    SliceReader,
    SlicedReader,
    SliceView,
    open_byte_range,
    open_range,
    open_slice,
    read_slice,
    slice_view,
)

__all__ = [
    "ReadOnlySlice",
    "ReadonlySlice",
    "Slice",
    "SliceReader",
    "SlicedReader",
    "SliceView",
    "open_byte_range",
    "open_range",
    "open_slice",
    "read_slice",
    "slice_view",
]

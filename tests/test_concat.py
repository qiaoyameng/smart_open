#
# Copyright (C) 2026 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).
#

"""Tests for smart_open.concat: reading many objects as one text stream."""

import gzip

import pytest

import smart_open
from smart_open.concat import ConcatError, concat


def write(tmp_path, name, data):
    path = tmp_path / name
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.write_bytes(data)
    return str(path)


@pytest.fixture
def three_parts(tmp_path):
    return [
        write(tmp_path, "part1", "hello "),
        write(tmp_path, "part2", "world\n"),
        write(tmp_path, "part3", "bye\n"),
    ]


class TrackingPaths:
    """An iterable that records how many paths were pulled from it."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.pulled = 0

    def __iter__(self):
        for path in self.paths:
            self.pulled += 1
            yield path


def test_read_all_concatenates(three_parts):
    with concat(three_parts, encoding="utf-8") as fin:
        assert fin.read() == "hello world\nbye\n"


def test_paths_are_lazy(three_parts):
    paths = TrackingPaths(three_parts)
    fin = concat(paths, encoding="utf-8")
    assert paths.pulled == 0
    assert fin.read(6) == "hello "
    assert paths.pulled == 1
    assert fin.read(6) == "world\n"
    assert paths.pulled == 2
    fin.close()


def test_next_object_opened_only_after_previous_exhausted(three_parts):
    paths = TrackingPaths(three_parts)
    with concat(paths, encoding="utf-8") as fin:
        assert fin.readline() == "hello world\n"
        assert paths.pulled == 2
        assert fin.readline() == "bye\n"
        assert paths.pulled == 3
        assert fin.readline() == ""
        assert paths.pulled == 3


def test_readline_joins_across_boundary(tmp_path):
    paths = [
        write(tmp_path, "a", "no newline at"),
        write(tmp_path, "b", " the end\nnext\n"),
    ]
    with concat(paths, encoding="utf-8") as fin:
        assert fin.readline() == "no newline at the end\n"
        assert fin.readline() == "next\n"
        assert fin.readline() == ""


def test_empty_objects_are_skipped(tmp_path):
    paths = [
        write(tmp_path, "a", ""),
        write(tmp_path, "b", "one\n"),
        write(tmp_path, "c", ""),
        write(tmp_path, "d", ""),
        write(tmp_path, "e", "two\n"),
        write(tmp_path, "f", ""),
    ]
    with concat(paths, encoding="utf-8") as fin:
        assert list(fin) == ["one\n", "two\n"]


def test_decoder_reset_by_default_strict(tmp_path):
    # '€' is b'\xe2\x82\xac' in utf-8: split it across two objects.
    paths = [
        write(tmp_path, "a", "ab".encode() + b"\xe2\x82"),
        write(tmp_path, "b", b"\xac" + "cd".encode()),
    ]
    with concat(paths, encoding="utf-8") as fin:
        with pytest.raises(UnicodeDecodeError):
            fin.read()


def test_decoder_reset_by_default_replace(tmp_path):
    paths = [
        write(tmp_path, "a", "ab".encode() + b"\xe2\x82"),
        write(tmp_path, "b", b"\xac" + "cd".encode()),
    ]
    with concat(paths, encoding="utf-8", errors="replace") as fin:
        # the truncated char is NOT completed by the next object's bytes
        assert fin.read() == "ab\ufffd\ufffdcd"


def test_decoder_shared_when_reset_disabled(tmp_path):
    paths = [
        write(tmp_path, "a", "ab".encode() + b"\xe2\x82"),
        write(tmp_path, "b", b"\xac" + "cd".encode()),
    ]
    with concat(paths, encoding="utf-8", reset_decoder=False) as fin:
        assert fin.read() == "ab€cd"


def test_open_failure_carries_index_and_offset(tmp_path):
    paths = [
        write(tmp_path, "a", "first\n"),
        str(tmp_path / "missing"),
        write(tmp_path, "b", "never reached\n"),
    ]
    with concat(paths, encoding="utf-8") as fin:
        assert fin.readline() == "first\n"
        with pytest.raises(ConcatError) as ctx:
            fin.readline()
    assert ctx.value.index == 1
    assert ctx.value.offset == len("first\n")
    assert ctx.value.path == paths[1]


def test_close_does_not_open_later_objects(three_parts):
    paths = TrackingPaths(three_parts)
    fin = concat(paths, encoding="utf-8")
    assert fin.read(3) == "hel"
    fin.close()
    assert paths.pulled == 1
    assert fin.closed
    with pytest.raises(ValueError, match="closed"):
        fin.read(1)


def test_interleaved_read_and_iteration_position(three_parts):
    with concat(three_parts, encoding="utf-8") as fin:
        assert fin.tell() == 0
        assert fin.read(6) == "hello "
        assert fin.tell() == 6
        # iteration continues exactly where read() stopped
        assert next(fin) == "world\n"
        assert fin.tell() == 12
        assert fin.read(2) == "by"
        assert fin.tell() == 14
        assert next(fin) == "e\n"
        assert fin.tell() == 16
        assert next(fin, None) is None
        assert fin.read() == ""
        assert fin.tell() == 16


def test_readline_size_limit(tmp_path):
    paths = [write(tmp_path, "a", "abcdef\ngh")]
    with concat(paths, encoding="utf-8") as fin:
        assert fin.readline(3) == "abc"
        assert fin.readline(3) == "def"
        assert fin.readline(3) == "\n"
        assert fin.readline(10) == "gh"
        assert fin.readline(10) == ""


def test_read_size_and_read_zero(three_parts):
    with concat(three_parts, encoding="utf-8") as fin:
        assert fin.read(0) == ""
        assert fin.tell() == 0
        assert fin.read(100) == "hello world\nbye\n"
        assert fin.read(100) == ""


def test_compressed_objects(tmp_path):
    gz_path = tmp_path / "part.gz"
    with gzip.open(gz_path, "wb") as fout:
        fout.write("compressed\n".encode())
    plain = write(tmp_path, "plain", "plain\n")
    with concat([str(gz_path), plain], encoding="utf-8") as fin:
        assert list(fin) == ["compressed\n", "plain\n"]


def test_exported_at_package_level():
    assert smart_open.concat is concat
    assert "concat" in smart_open.__all__

import pytest

import smart_open
from smart_open import smart_open_lib


def write_files(tmp_path, names_and_data, kind="t"):
    paths = []
    for name, data in names_and_data:
        path = tmp_path / name
        if kind == "b":
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")
        paths.append(str(path))
    return paths


def test_open_many_reads_text_and_skips_empty_objects(tmp_path):
    paths = write_files(
        tmp_path,
        [("0", "ab"), ("1", ""), ("2", "cd"), ("3", ""), ("4", "ef")],
    )

    with smart_open.open_many(paths, "rt", encoding="utf-8") as fin:
        assert fin.read(3) == "abc"
        assert fin.tell() == 3
        assert fin.read() == "def"
        assert fin.tell() == 6
        assert fin.read() == ""


def test_open_many_opens_lazily_and_closes_objects(tmp_path, monkeypatch):
    paths = write_files(tmp_path, [("0", "ab"), ("1", ""), ("2", "cd")])
    opened = []
    original_open = smart_open_lib.open

    def tracking_open(uri, mode, **kwargs):
        opened.append(uri)
        return original_open(uri, mode, **kwargs)

    monkeypatch.setattr(smart_open_lib, "open", tracking_open)

    def uris():
        yield paths[0]
        yield paths[1]
        yield paths[2]

    with smart_open.open_many(uris(), "rt", encoding="utf-8") as fin:
        assert opened == []
        assert fin.read(2) == "ab"
        assert opened == [paths[0]]

        assert fin.read(1) == "c"
        assert opened == [paths[0], paths[1], paths[2]]

    assert opened == [paths[0], paths[1], paths[2]]


def test_open_many_readline_joins_without_newline(tmp_path):
    paths = write_files(tmp_path, [("0", "ab"), ("1", ""), ("2", "cd\nef"), ("3", "\n")])

    with smart_open.open_many(paths, "rt", encoding="utf-8") as fin:
        assert fin.readline() == "abcd\n"
        assert fin.tell() == 5
        assert fin.readline(2) == "ef"
        assert fin.tell() == 7
        assert fin.readline() == "\n"
        assert fin.readline() == ""


def test_open_many_readline_stops_at_carriage_return_boundary(tmp_path):
    paths = write_files(tmp_path, [("0", "ab\r"), ("1", "cd")])

    with smart_open.open_many(paths, "rt", encoding="utf-8", newline="") as fin:
        assert fin.readline() == "ab\r"
        assert fin.tell() == 3
        assert fin.readline() == "cd"


def test_open_many_interleaves_read_and_readline_by_logical_position(tmp_path):
    paths = write_files(tmp_path, [("0", "ab\ncd"), ("1", "ef\ngh")])

    with smart_open.open_many(paths, "rt", encoding="utf-8") as fin:
        assert fin.read(1) == "a"
        assert fin.tell() == 1
        assert fin.readline() == "b\n"
        assert fin.tell() == 3
        assert fin.read(2) == "cd"
        assert fin.tell() == 5
        assert fin.readline() == "ef\n"
        assert fin.tell() == 8
        assert fin.readline() == "gh"
        assert fin.tell() == 10


def test_open_many_each_object_has_an_independent_decoder(tmp_path, monkeypatch):
    paths = write_files(
        tmp_path,
        [("0", b"\xc3"), ("1", b"\xa9\n")],
        "b",
    )
    opened = []
    original_open = smart_open_lib.open

    def tracking_open(uri, mode, **kwargs):
        opened.append(uri)
        return original_open(uri, mode, **kwargs)

    monkeypatch.setattr(smart_open_lib, "open", tracking_open)

    with smart_open.open_many(paths, "rt", encoding="utf-8", errors="strict") as fin:
        with pytest.raises(UnicodeDecodeError):
            fin.readline()

    assert opened == [paths[0]]


def test_open_many_binary_still_concatenates_raw_boundaries(tmp_path):
    paths = write_files(tmp_path, [("0", b"\xc3"), ("1", b"\xa9")], "b")

    with smart_open.open_many(paths, "rb") as fin:
        assert fin.read() == b"\xc3\xa9"


def test_open_many_open_failure_reports_index_and_offset(tmp_path, monkeypatch):
    paths = write_files(tmp_path, [("0", "ab"), ("1", "missing"), ("2", "cd")])
    original_open = smart_open_lib.open

    def failing_open(uri, mode, **kwargs):
        if uri == paths[1]:
            raise FileNotFoundError(uri)
        return original_open(uri, mode, **kwargs)

    monkeypatch.setattr(smart_open_lib, "open", failing_open)

    with smart_open.open_many(paths, "rt", encoding="utf-8") as fin:
        assert fin.read(2) == "ab"
        with pytest.raises(smart_open.OpenManyError) as excinfo:
            fin.read(1)

    error = excinfo.value
    assert error.object_index == 1
    assert error.logical_offset == 2
    assert error.uri == paths[1]


def test_open_many_early_close_does_not_open_later_objects(tmp_path, monkeypatch):
    paths = write_files(tmp_path, [("0", "ab"), ("1", "cd")])
    opened = []
    original_open = smart_open_lib.open

    def tracking_open(uri, mode, **kwargs):
        opened.append(uri)
        return original_open(uri, mode, **kwargs)

    monkeypatch.setattr(smart_open_lib, "open", tracking_open)

    fin = smart_open.open_many(paths, "rt", encoding="utf-8")
    assert fin.read(1) == "a"
    fin.close()

    assert opened == [paths[0]]
    with pytest.raises(ValueError, match="closed"):
        fin.read()


def test_open_many_rejects_single_uri(tmp_path):
    path = tmp_path / "0"
    path.write_text("ab", encoding="utf-8")

    with pytest.raises(TypeError, match="iterable"):
        smart_open.open_many(str(path), "rt", encoding="utf-8")


def test_open_many_rejects_write_mode(tmp_path):
    with pytest.raises(ValueError, match="read modes"):
        smart_open.open_many([str(tmp_path / "0")], "wt", encoding="utf-8")

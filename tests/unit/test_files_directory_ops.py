"""Merging a directory into another and removing it, without Qt.

This ran only inside the Qt file-operations controller and had no tests, while
moving every file out of a directory and then deleting it. The rule worth
pinning is the collision behaviour: an entry whose destination name already
exists is skipped and left behind -- Utils.move_file would raise instead --
so the source directory is not necessarily empty when it is removed.
"""

import os

import pytest

from files.directory_ops import move_directory_contents_then_delete
from utils.config import config


@pytest.fixture(autouse=True)
def _delete_instantly(monkeypatch):
    # Removal goes through Utils.remove_path with the real config; pin this so
    # it uses os.remove/rmtree rather than send2trash or a trash folder.
    monkeypatch.setattr(config, "delete_instantly", True)


@pytest.fixture
def dirs(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    return source, target


def _file(path, text="x"):
    path.write_text(text, encoding="utf-8")
    return path


class TestMoveContentsThenDelete:
    def test_moves_every_entry_and_removes_the_directory(self, dirs):
        source, target = dirs
        _file(source / "a.txt")
        _file(source / "b.txt")

        result = move_directory_contents_then_delete(str(source), str(target))

        assert result.moved == 2
        assert result.skipped == 0
        assert sorted(p.name for p in target.iterdir()) == ["a.txt", "b.txt"]
        assert not source.exists()

    def test_colliding_entry_is_skipped_and_left_behind(self, dirs):
        source, target = dirs
        _file(source / "dup.txt", "from source")
        _file(target / "dup.txt", "already here")
        _file(source / "fresh.txt")

        result = move_directory_contents_then_delete(str(source), str(target))

        assert result.moved == 1
        assert result.skipped == 1
        # The existing target file is untouched, not replaced.
        assert (target / "dup.txt").read_text(encoding="utf-8") == "already here"
        assert (target / "fresh.txt").exists()

    def test_directory_is_removed_even_when_something_was_skipped(self, dirs):
        # The skipped entry stays in the source, so the directory is not empty
        # -- it is removed anyway, taking the leftover with it.
        source, target = dirs
        _file(source / "dup.txt", "from source")
        _file(target / "dup.txt", "already here")

        result = move_directory_contents_then_delete(str(source), str(target))

        assert result.skipped == 1
        assert not source.exists()

    def test_subdirectories_are_moved_whole(self, dirs):
        source, target = dirs
        nested = source / "nested"
        nested.mkdir()
        _file(nested / "inner.txt")

        result = move_directory_contents_then_delete(str(source), str(target))

        assert result.moved == 1
        assert (target / "nested" / "inner.txt").exists()

    def test_empty_directory_is_just_removed(self, dirs):
        source, target = dirs

        result = move_directory_contents_then_delete(str(source), str(target))

        assert (result.moved, result.skipped) == (0, 0)
        assert not source.exists()
        assert list(target.iterdir()) == []

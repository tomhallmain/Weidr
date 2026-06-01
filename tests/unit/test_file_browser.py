"""Unit tests for files/file_browser.py — gather, extensions, recursive, sort."""

import os
import time

import pytest
from PIL import Image

from files.file_browser import FileBrowser
from utils.config import config
from utils.constants import Sort, SortBy
from utils.utils import Utils


def _make_png(path: str) -> None:
    Image.new("RGB", (4, 4), (128, 128, 128)).save(path, format="PNG")


def _touch(path: str, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


@pytest.fixture
def browser_tree(tmp_path, monkeypatch):
    """Root + nested PNGs, a .txt file, and a .jpg excluded by file_types."""
    monkeypatch.setattr(config, "file_types", [".png"])
    root = tmp_path
    nested = root / "nested"
    nested.mkdir()
    _make_png(root / "aaa.png")
    _make_png(root / "zzz.png")
    _make_png(nested / "mid.png")
    (root / "readme.txt").write_text("skip", encoding="utf-8")
    Image.new("RGB", (4, 4), (200, 200, 200)).save(root / "other.jpg", format="JPEG")
    return str(root)


@pytest.fixture(autouse=True)
def _clear_confirmed_dirs():
    FileBrowser.have_confirmed_directories.clear()
    yield
    FileBrowser.have_confirmed_directories.clear()


class TestFileBrowserGather:
    def test_non_recursive_lists_top_level_only(self, browser_tree):
        fb = FileBrowser(browser_tree, recursive=False)
        fb.set_directory(browser_tree)
        names = [os.path.basename(p) for p in fb.get_files()]
        assert names == ["aaa.png", "zzz.png"]

    def test_recursive_includes_subdirectory(self, browser_tree):
        fb = FileBrowser(browser_tree, recursive=False)
        fb.set_recursive(True)
        names = [os.path.basename(p) for p in fb.get_files()]
        assert names == ["aaa.png", "mid.png", "zzz.png"]

    def test_extension_gating_excludes_non_configured_types(self, browser_tree):
        fb = FileBrowser(browser_tree, recursive=True)
        fb.set_directory(browser_tree)
        paths = fb.get_files()
        assert all(p.lower().endswith(".png") for p in paths)
        assert not any(p.endswith("readme.txt") for p in paths)
        assert not any(p.endswith("other.jpg") for p in paths)

    def test_set_filter_limits_glob_matches(self, browser_tree):
        fb = FileBrowser(browser_tree, recursive=True)
        fb.set_filter("aaa")
        fb.refresh()
        assert fb.count() == 1
        assert os.path.basename(fb.get_files()[0]) == "aaa.png"


class TestParseFilter:
    def test_empty_string(self):
        assert FileBrowser._parse_filter("") == ([], [])

    def test_single_inclusion(self):
        assert FileBrowser._parse_filter("2024") == (["2024"], [])

    def test_multiple_inclusions(self):
        assert FileBrowser._parse_filter("2024;vacation") == (["2024", "vacation"], [])

    def test_single_exclusion(self):
        assert FileBrowser._parse_filter("!_edit") == ([], ["_edit"])

    def test_multiple_exclusions(self):
        assert FileBrowser._parse_filter("!_edit;!_proxy") == ([], ["_edit", "_proxy"])

    def test_mixed_inclusion_and_exclusion(self):
        assert FileBrowser._parse_filter("2024;!_edit") == (["2024"], ["_edit"])

    def test_whitespace_around_terms_is_stripped(self):
        assert FileBrowser._parse_filter(" 2024 ; vacation ") == (["2024", "vacation"], [])

    def test_whitespace_after_bang_is_stripped(self):
        assert FileBrowser._parse_filter("! _edit") == ([], ["_edit"])

    def test_bare_bang_is_ignored(self):
        assert FileBrowser._parse_filter("!") == ([], [])

    def test_empty_tokens_from_adjacent_semicolons_are_ignored(self):
        assert FileBrowser._parse_filter("a;;b") == (["a", "b"], [])

    def test_only_semicolons(self):
        assert FileBrowser._parse_filter(";;") == ([], [])


@pytest.fixture
def filter_tree(tmp_path, monkeypatch):
    """Tree with files spread across named subdirectories for filter testing."""
    monkeypatch.setattr(config, "file_types", [".png"])
    root = tmp_path
    for d in ("cats", "vacation", "cats_edit"):
        (root / d).mkdir()
    _make_png(root / "aaa.png")
    _make_png(root / "zzz.png")
    _make_png(root / "cats" / "cats_file.png")
    _make_png(root / "vacation" / "vacation_file.png")
    _make_png(root / "cats_edit" / "cats_edit_file.png")
    return str(root)


class TestFileBrowserFilter:
    def _names(self, fb):
        return {os.path.basename(p) for p in fb.get_files()}

    def test_single_inclusion_unchanged(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=True)
        fb.set_filter("cats")
        fb.refresh()
        assert self._names(fb) == {"cats_file.png", "cats_edit_file.png"}

    def test_multi_inclusion_unions_results(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=True)
        fb.set_filter("cats;vacation")
        fb.refresh()
        assert self._names(fb) == {"cats_file.png", "cats_edit_file.png", "vacation_file.png"}

    def test_duplicate_inclusion_terms_deduplicate(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=True)
        fb.set_filter("cats;cats")
        fb.refresh()
        assert self._names(fb) == {"cats_file.png", "cats_edit_file.png"}

    def test_exclusion_only_top_level(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=False)
        fb.set_filter("!zzz")
        fb.refresh()
        assert self._names(fb) == {"aaa.png"}

    def test_exclusion_only_recursive(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=True)
        fb.set_filter("!cats")
        fb.refresh()
        assert self._names(fb) == {"aaa.png", "zzz.png", "vacation_file.png"}

    def test_inclusion_with_exclusion(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=True)
        fb.set_filter("cats;!_edit")
        fb.refresh()
        assert self._names(fb) == {"cats_file.png"}

    def test_multi_inclusion_with_exclusion(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=True)
        fb.set_filter("cats;vacation;!_edit")
        fb.refresh()
        assert self._names(fb) == {"cats_file.png", "vacation_file.png"}

    def test_empty_filter_clears_to_full_set(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=False)
        fb.set_filter("")
        fb.refresh()
        assert self._names(fb) == {"aaa.png", "zzz.png"}

    def test_bare_bang_is_noop(self, filter_tree):
        fb = FileBrowser(filter_tree, recursive=True)
        fb.set_filter("!")
        fb.refresh()
        assert self._names(fb) == {"aaa.png", "zzz.png", "cats_file.png", "vacation_file.png", "cats_edit_file.png"}


class TestIsInvalidFileFilter:
    """Tests for the multi-pattern filter logic in Utils.is_invalid_file()."""

    def test_multi_inclusion_first_term_matches(self):
        assert Utils.is_invalid_file("/some/cats/file.jpg", 1, False, "cats;vacation") is False

    def test_multi_inclusion_second_term_matches(self):
        assert Utils.is_invalid_file("/some/vacation/file.jpg", 1, False, "cats;vacation") is False

    def test_multi_inclusion_no_term_matches(self):
        assert Utils.is_invalid_file("/some/dogs/file.jpg", 1, False, "cats;vacation") is True

    def test_exclusion_only_path_not_excluded(self):
        assert Utils.is_invalid_file("/some/cats/file.jpg", 1, False, "!_edit") is False

    def test_exclusion_only_path_is_excluded(self):
        assert Utils.is_invalid_file("/some/cats_edit/file.jpg", 1, False, "!_edit") is True

    def test_inclusion_matches_exclusion_does_not(self):
        assert Utils.is_invalid_file("/some/cats/file.jpg", 1, False, "cats;!_edit") is False

    def test_inclusion_matches_exclusion_also_matches(self):
        assert Utils.is_invalid_file("/some/cats_edit/file.jpg", 1, False, "cats;!_edit") is True

    def test_inclusion_does_not_match_with_exclusion(self):
        assert Utils.is_invalid_file("/some/dogs/file.jpg", 1, False, "cats;!_edit") is True

    def test_bare_bang_acts_as_no_filter(self):
        assert Utils.is_invalid_file("/some/file.jpg", 1, False, "!") is False


class TestFileBrowserSort:
    def test_sort_by_name_ascending(self, browser_tree):
        fb = FileBrowser(browser_tree, recursive=True)
        fb.set_sort_by(SortBy.NAME)
        names = [os.path.basename(p) for p in fb.get_files()]
        assert names == ["aaa.png", "mid.png", "zzz.png"]

    def test_sort_by_modify_time_ascending(self, browser_tree):
        root = browser_tree
        old = os.path.join(root, "aaa.png")
        new = os.path.join(root, "zzz.png")
        base = time.time() - 1000
        _touch(old, base)
        _touch(new, base + 500)

        fb = FileBrowser(root, recursive=False)
        fb.set_sort_by(SortBy.MODIFY_TIME)
        files = fb.get_files()
        assert files[0] == old
        assert files[-1] == new

    def test_descending_name_order_via_operation_sort(self, browser_tree):
        fb = FileBrowser(browser_tree, recursive=True)
        fb.set_directory(browser_tree)
        desc = fb.get_files_sorted_for_operation(SortBy.NAME, Sort.DESC)
        names = [os.path.basename(p) for p in desc]
        assert names == ["zzz.png", "mid.png", "aaa.png"]

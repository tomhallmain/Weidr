"""Unit tests for files/file_browser.py — gather, extensions, recursive, sort."""

import os
import time

import pytest
from PIL import Image

from files.file_browser import FileBrowser
from utils.config import config
from utils.constants import Sort, SortBy


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

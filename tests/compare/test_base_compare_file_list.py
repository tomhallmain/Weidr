"""
Unit tests for compare-mode get_files() honoring an explicit CompareArgs.file_list.

Lets FileActionsWindow's "Search in New Window" feature build a search corpus
from action-history destination files instead of scanning a directory. See
docs/file-actions-search-in-new-window.md.

BaseCompare.get_files() carries the file_list shortcut, but several compare
modes (CompareSize, CompareModels, ComparePrompts, ComparePromptsExact) have
their own full get_files() override that never calls super().get_files() --
a pre-existing duplication in this codebase. The fix therefore had to be
applied to each override independently, and each is parametrized here so a
future change to one doesn't silently stop covering the others (exactly the
gap that slipped through when this was tested via CompareSize alone and
CompareSize's own override didn't have the shortcut yet).
"""
from __future__ import annotations

import pytest

from compare.compare_args import CompareArgs
from compare.compare_colors import CompareColors
from compare.compare_models import CompareModels
from compare.compare_prompts import ComparePrompts
from compare.compare_prompts_exact import ComparePromptsExact
from compare.compare_size import CompareSize


def _raise_if_called(**kwargs):
    raise AssertionError("gather_files_func should not be called when args.file_list is set")


# name -> compare class. CompareColors uses BaseCompare.get_files() unmodified;
# the rest each have their own override.
COMPARE_CLASSES = {
    "CompareColors (base impl)": CompareColors,
    "CompareSize": CompareSize,
    "CompareModels": CompareModels,
    "ComparePrompts": ComparePrompts,
    "ComparePromptsExact": ComparePromptsExact,
}


@pytest.mark.parametrize("compare_cls", COMPARE_CLASSES.values(), ids=COMPARE_CLASSES.keys())
class TestGetFilesWithExplicitFileList:
    def test_file_list_bypasses_directory_scan(self, tmp_path, compare_cls):
        args = CompareArgs(base_dir=str(tmp_path))
        args.file_list = ["/z.jpg", "/a.jpg", "/m.jpg"]
        compare = compare_cls(args=args, gather_files_func=_raise_if_called)

        compare.get_files()  # must not raise -- proves the scan was skipped

        assert compare.files == ["/a.jpg", "/m.jpg", "/z.jpg"]

    def test_empty_file_list_falls_back_to_directory_scan(self, tmp_path, compare_cls):
        called = []

        def _gather(**kwargs):
            called.append(kwargs)
            return []

        args = CompareArgs(base_dir=str(tmp_path))
        compare = compare_cls(args=args, gather_files_func=_gather)

        compare.get_files()

        assert len(called) == 1

    def test_search_media_path_still_placed_first(self, tmp_path, compare_cls):
        """Edge case from the design doc: the active media is itself one of
        the file_list entries -- get_files() already handles this via its
        existing search_media_path placement, unchanged by the file_list path."""
        args = CompareArgs(base_dir=str(tmp_path), search_media_path="/m.jpg")
        args.file_list = ["/z.jpg", "/a.jpg", "/m.jpg"]
        compare = compare_cls(args=args, gather_files_func=_raise_if_called)

        compare.get_files()

        assert compare.files[0] == "/m.jpg"
        assert sorted(compare.files) == ["/a.jpg", "/m.jpg", "/z.jpg"]

    def test_base_dir_still_used_for_compare_data_cache_path(self, tmp_path, compare_cls):
        """base_dir is still used for the embedding/data cache path even when
        an explicit file_list is supplied."""
        args = CompareArgs(base_dir=str(tmp_path))
        args.file_list = ["/a.jpg"]
        compare = compare_cls(args=args, gather_files_func=_raise_if_called)

        assert compare.compare_data.base_dir == str(tmp_path)

"""
Regression tests for CompareWrapper._sync_result_after_deletion.

Bug report: moving marks while in search mode crashed with
    IndexError: invalid index to scalar variable.
inside compare_wrapper._sync_result_after_deletion -> `if v[0] in active_group_indexes`.

Root cause: in GROUP mode, CompareResult.files_grouped maps
{file_index: (group_index, score)}, so `v[0]` is the group index. But in
SEARCH mode, files_grouped maps {filepath: score} -- a flat dict keyed by
path with a plain (often numpy) scalar score as the value -- so `v[0]`
indexes into the scalar itself and raises.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from compare.compare_result import CompareResult
from compare.compare_wrapper import CompareWrapper
from utils.constants import CompareMode


def _stored_compare_result(tmp_path, files, files_grouped, file_groups):
    compare_result = CompareResult(
        base_dir=str(tmp_path), files=files, mode=CompareMode.CLIP_EMBEDDING
    )
    compare_result.files_grouped = files_grouped
    compare_result.file_groups = file_groups
    compare_result.is_complete = True
    compare_result.store()
    assert os.path.exists(CompareResult.cache_path(str(tmp_path), CompareMode.CLIP_EMBEDDING))
    return compare_result


def _wrapper_for(tmp_path, compare_result, wrapper_file_groups, is_run_search):
    wrapper = CompareWrapper(master=None, compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=None)
    wrapper.file_groups = wrapper_file_groups
    wrapper._compare = SimpleNamespace(
        compare_result=compare_result,
        base_dir=str(tmp_path),
        COMPARE_MODE=CompareMode.CLIP_EMBEDDING,
        is_run_search=is_run_search,
    )
    return wrapper


class TestSyncResultAfterDeletionSearchMode:
    def test_numpy_scalar_score_does_not_raise(self, tmp_path):
        """The reported crash: a numpy scalar score raises on v[0] indexing."""
        files_grouped = {"/a.jpg": np.float32(0.9), "/b.jpg": np.float32(0.5)}
        file_groups = {0: dict(files_grouped)}
        compare_result = _stored_compare_result(
            tmp_path, list(files_grouped.keys()), files_grouped, file_groups
        )
        wrapper = _wrapper_for(tmp_path, compare_result, file_groups, is_run_search=True)

        wrapper._sync_result_after_deletion("/a.jpg")  # must not raise

    def test_deleted_file_removed_from_files_grouped(self, tmp_path):
        files_grouped = {"/a.jpg": 0.9, "/b.jpg": 0.5}
        file_groups = {0: dict(files_grouped)}
        compare_result = _stored_compare_result(
            tmp_path, list(files_grouped.keys()), files_grouped, file_groups
        )
        wrapper = _wrapper_for(tmp_path, compare_result, file_groups, is_run_search=True)

        wrapper._sync_result_after_deletion("/a.jpg")

        assert "/a.jpg" not in compare_result.files_grouped
        assert compare_result.files_grouped["/b.jpg"] == 0.5

    def test_missing_filepath_is_a_noop(self, tmp_path):
        """Deleting a path that isn't in files_grouped shouldn't raise (pop default)."""
        files_grouped = {"/a.jpg": 0.9}
        file_groups = {0: dict(files_grouped)}
        compare_result = _stored_compare_result(
            tmp_path, list(files_grouped.keys()), files_grouped, file_groups
        )
        wrapper = _wrapper_for(tmp_path, compare_result, file_groups, is_run_search=True)

        wrapper._sync_result_after_deletion("/not-present.jpg")

        assert compare_result.files_grouped == {"/a.jpg": 0.9}


class TestSyncResultAfterDeletionGroupMode:
    def test_group_index_filtering_still_applies(self, tmp_path):
        """Non-search (group) mode keeps the original (group_index, score)
        tuple-based pruning: entries whose group no longer exists are dropped."""
        files_grouped = {0: (0, 0.9), 1: (1, 0.5)}
        file_groups = {0: {"/a.jpg": 0.9}, 1: {"/c.jpg": 0.5}}
        compare_result = _stored_compare_result(
            tmp_path, ["/a.jpg", "/b.jpg", "/c.jpg"], files_grouped, file_groups
        )
        # Simulate group 1 having already been pruned from the live wrapper state.
        wrapper = _wrapper_for(tmp_path, compare_result, {0: {"/a.jpg": 0.9}}, is_run_search=False)

        wrapper._sync_result_after_deletion("/b.jpg")

        assert compare_result.files_grouped == {0: (0, 0.9)}

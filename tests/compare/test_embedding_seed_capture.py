"""
Unit tests for CompareWrapper.get_current_supergroup_seed_data
(docs/embedding-seed-library.md, section 5.1).

The single-media capture path (section 5.4) is deliberately independent of
CompareWrapper -- see compare/embedding_capture.py and
tests/compare/test_embedding_capture.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from compare.compare_args import CompareArgs
from compare.compare_embeddings_clip import CompareEmbeddingClip
from compare.compare_wrapper import CompareWrapper
from utils.constants import CompareMode, Mode


def _unit(vec):
    arr = np.array(vec, dtype=float)
    return arr / np.linalg.norm(arr)


def _make_compare(tmp_path, threshold=0.9):
    args = CompareArgs(base_dir=str(tmp_path), compare_threshold=threshold)
    return CompareEmbeddingClip(args=args)


def _seed(compare, files_to_embeddings: dict, file_groups: dict):
    """Populate files_found / _file_embeddings / compare_result.file_groups."""
    files = list(files_to_embeddings.keys())
    compare.compare_data.files_found = files
    compare._file_embeddings = np.array([files_to_embeddings[f] for f in files])
    compare.compare_result.file_groups = file_groups


def _wrapper(app=None):
    return CompareWrapper(
        master=app if app is not None else MagicMock(),
        compare_mode=CompareMode.CLIP_EMBEDDING,
        app_actions=MagicMock(),
    )


class TestGetCurrentSupergroupSeedData:
    def test_no_compare_returns_none(self):
        assert _wrapper().get_current_supergroup_seed_data() is None

    def test_no_group_indexes_returns_none(self, tmp_path):
        wrapper = _wrapper()
        wrapper._compare = _make_compare(tmp_path)
        wrapper.group_indexes = []
        assert wrapper.get_current_supergroup_seed_data() is None

    def test_group_complement_mode_returns_none(self, tmp_path):
        app = MagicMock()
        app.mode = Mode.GROUP_COMPLEMENT
        wrapper = _wrapper(app)
        wrapper._compare = _make_compare(tmp_path)
        wrapper.group_indexes = [0]
        assert wrapper.get_current_supergroup_seed_data() is None

    def test_no_meaningful_supergroups_returns_none(self, tmp_path):
        wrapper = _wrapper()
        compare = _make_compare(tmp_path)
        _seed(compare, {"a": _unit([1, 0]), "b": _unit([0, 1])}, {0: ["a"], 1: ["b"]})
        compare.compare_result.supergroups = []
        wrapper._compare = compare
        wrapper.group_indexes = [0, 1]
        wrapper.current_group_index = 0
        assert wrapper.get_current_supergroup_seed_data() is None

    def test_resolves_centroid_for_current_groups_supergroup(self, tmp_path):
        wrapper = _wrapper()
        compare = _make_compare(tmp_path)
        _seed(
            compare,
            {"a": _unit([1, 0]), "b": _unit([0.99, 0.05]), "c": _unit([0, 1])},
            {0: ["a"], 1: ["b"], 2: ["c"]},
        )
        compare.compare_result.supergroups = [[0, 1], [2]]
        wrapper._compare = compare
        wrapper.group_indexes = [0, 1, 2]
        wrapper.current_group_index = 0  # actual_group_index() -> 0, in cluster [0, 1]

        data = wrapper.get_current_supergroup_seed_data()

        assert data is not None
        assert sorted(data["group_indexes"]) == [0, 1]
        assert data["member_count"] == 2
        assert data["compare_mode"] == "CLIP_EMBEDDING"
        assert np.isclose(np.linalg.norm(data["vector"]), 1.0)

    def test_current_group_outside_any_supergroup_returns_none(self, tmp_path):
        """Group 2 is navigable (in group_indexes) but was never clustered at
        all -- e.g. it had no resolvable centroid when compute_supergroups()
        last ran. Distinct from a *singleton* cluster like [2] on its own,
        which _supergroup_label_suffix's existing precedent treats as valid
        membership (matched by this same "in cluster" lookup) as long as
        has_meaningful_supergroups() is True overall -- this test is instead
        about group 2 not appearing in the supergroups list at all."""
        wrapper = _wrapper()
        compare = _make_compare(tmp_path)
        _seed(
            compare,
            {"a": _unit([1, 0]), "b": _unit([0.99, 0.05]), "c": _unit([0, 1])},
            {0: ["a"], 1: ["b"], 2: ["c"]},
        )
        compare.compare_result.supergroups = [[0, 1]]  # group 2 absent entirely
        wrapper._compare = compare
        wrapper.group_indexes = [0, 1, 2]
        wrapper.current_group_index = 2  # actual_group_index() -> 2, not in any cluster

        assert wrapper.get_current_supergroup_seed_data() is None

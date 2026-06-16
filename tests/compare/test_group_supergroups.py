"""
Unit tests for group mean-embedding centroids and supergroup clustering --
mean-pooling each compare group's per-file embeddings into a centroid, then
clustering those centroids into "supergroups" of related groups.

Covers BaseCompareEmbedding.compute_group_centroids / compute_supergroups,
the module-level cluster_group_indexes helper, and CompareWrapper's
supergroup navigation / label suffix.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from compare.base_compare_embedding import cluster_group_indexes
from compare.compare_args import CompareArgs
from compare.compare_embeddings_clip import CompareEmbeddingClip
from compare.compare_result import CompareResult
from compare.compare_wrapper import CompareWrapper
from utils.constants import CompareMode


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


# ---------------------------------------------------------------------------
# cluster_group_indexes (pure clustering algorithm)
# ---------------------------------------------------------------------------

class TestClusterGroupIndexes:
    def test_single_group_forms_singleton_cluster(self):
        centroids = {1: _unit([1.0, 0.0])}
        assert cluster_group_indexes(centroids, threshold=0.9) == [[1]]

    def test_two_similar_centroids_merge(self):
        centroids = {1: _unit([1.0, 0.0]), 2: _unit([0.99, 0.05])}
        clusters = cluster_group_indexes(centroids, threshold=0.9)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [1, 2]

    def test_two_dissimilar_centroids_stay_separate(self):
        centroids = {1: _unit([1.0, 0.0]), 2: _unit([0.0, 1.0])}
        clusters = cluster_group_indexes(centroids, threshold=0.9)
        assert sorted(sorted(c) for c in clusters) == [[1], [2]]

    def test_transitive_chain_merges_via_single_link(self):
        """A~B and B~C (but not A~C directly) still end up in one cluster."""
        a = _unit([1.0, 0.0, 0.0])
        b = _unit([0.7, 0.7, 0.0])
        c = _unit([0.0, 1.0, 0.0])
        centroids = {1: a, 2: b, 3: c}
        # a.b and b.c are both >= 0.7 similarity; a.c is 0 -- only transitive
        # (single-link) clustering merges all three into one cluster.
        clusters = cluster_group_indexes(centroids, threshold=0.69)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [1, 2, 3]

    def test_empty_centroids_returns_empty(self):
        assert cluster_group_indexes({}, threshold=0.9) == []


# ---------------------------------------------------------------------------
# BaseCompareEmbedding.compute_group_centroids
# ---------------------------------------------------------------------------

class TestComputeGroupCentroids:
    def test_mean_of_two_member_group_is_normalized_average(self, tmp_path):
        compare = _make_compare(tmp_path)
        _seed(
            compare,
            {"/a.jpg": [1.0, 0.0], "/b.jpg": [0.0, 1.0]},
            {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}},
        )
        centroids = compare.compute_group_centroids()
        expected = _unit([0.5, 0.5])
        assert centroids[0] == pytest.approx(expected)
        assert np.linalg.norm(centroids[0]) == pytest.approx(1.0)

    def test_singleton_stranded_group_centroid_is_its_own_embedding(self, tmp_path):
        """A single-member ("stranded") group's centroid is just that file's
        own embedding -- included deliberately, not excluded, so a stranded
        file can still land in a supergroup with a related group."""
        compare = _make_compare(tmp_path)
        _seed(
            compare,
            {"/a.jpg": [1.0, 0.0]},
            {0: {"/a.jpg": 0.0}},
        )
        centroids = compare.compute_group_centroids()
        assert centroids[0] == pytest.approx(_unit([1.0, 0.0]))

    def test_member_missing_from_files_found_is_skipped(self, tmp_path):
        compare = _make_compare(tmp_path)
        _seed(
            compare,
            {"/a.jpg": [1.0, 0.0], "/b.jpg": [0.0, 1.0]},
            {0: {"/a.jpg": 0.0, "/missing.jpg": 0.0, "/b.jpg": 0.0}},
        )
        # /missing.jpg isn't in files_found -- centroid uses only a and b.
        centroids = compare.compute_group_centroids()
        assert centroids[0] == pytest.approx(_unit([0.5, 0.5]))

    def test_group_with_no_resolvable_members_omitted(self, tmp_path):
        compare = _make_compare(tmp_path)
        _seed(
            compare,
            {"/a.jpg": [1.0, 0.0]},
            {0: {"/missing1.jpg": 0.0, "/missing2.jpg": 0.0}},
        )
        assert compare.compute_group_centroids() == {}

    def test_multiple_groups_each_get_own_centroid(self, tmp_path):
        compare = _make_compare(tmp_path)
        _seed(
            compare,
            {
                "/a.jpg": [1.0, 0.0], "/b.jpg": [1.0, 0.0],
                "/c.jpg": [0.0, 1.0], "/d.jpg": [0.0, 1.0],
            },
            {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}, 1: {"/c.jpg": 0.0, "/d.jpg": 0.0}},
        )
        centroids = compare.compute_group_centroids()
        assert set(centroids.keys()) == {0, 1}
        assert centroids[0] == pytest.approx(_unit([1.0, 0.0]))
        assert centroids[1] == pytest.approx(_unit([0.0, 1.0]))


# ---------------------------------------------------------------------------
# BaseCompareEmbedding.compute_supergroups
# ---------------------------------------------------------------------------

class TestComputeSupergroups:
    def test_stranded_singleton_joins_supergroup_with_its_related_group(self, tmp_path):
        """The scenario stranding is meant to recover from: /a.jpg was abandoned
        by its original groupmate (now alone in group 0), but it's still close
        enough to group 1's content to land in the same supergroup -- no longer
        looking orphaned."""
        compare = _make_compare(tmp_path, threshold=0.9)
        _seed(
            compare,
            {
                "/a.jpg": [1.0, 0.0],
                "/c.jpg": [0.99, 0.05], "/d.jpg": [0.99, 0.05],
            },
            {0: {"/a.jpg": 0.0}, 1: {"/c.jpg": 0.0, "/d.jpg": 0.0}},
        )
        supergroups = compare.compute_supergroups()
        assert len(supergroups) == 1
        assert sorted(supergroups[0]) == [0, 1]

    def test_similar_groups_become_one_supergroup(self, tmp_path):
        compare = _make_compare(tmp_path, threshold=0.9)
        _seed(
            compare,
            {
                "/a.jpg": [1.0, 0.0], "/b.jpg": [1.0, 0.0],
                "/c.jpg": [0.99, 0.05], "/d.jpg": [0.99, 0.05],
            },
            {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}, 1: {"/c.jpg": 0.0, "/d.jpg": 0.0}},
        )
        supergroups = compare.compute_supergroups()
        assert supergroups == [[0, 1]] or supergroups == [[1, 0]]
        assert compare.compare_result.supergroups == supergroups

    def test_dissimilar_groups_stay_in_separate_supergroups(self, tmp_path):
        compare = _make_compare(tmp_path, threshold=0.9)
        _seed(
            compare,
            {
                "/a.jpg": [1.0, 0.0], "/b.jpg": [1.0, 0.0],
                "/c.jpg": [0.0, 1.0], "/d.jpg": [0.0, 1.0],
            },
            {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}, 1: {"/c.jpg": 0.0, "/d.jpg": 0.0}},
        )
        supergroups = compare.compute_supergroups()
        assert sorted(sorted(c) for c in supergroups) == [[0], [1]]

    def test_ascending_sort_by_total_member_file_count(self, tmp_path):
        compare = _make_compare(tmp_path, threshold=0.5)
        # Group 0: 2 files, isolated. Groups 1+2: 2 similar groups of 3 files
        # each (6 total) -- the bigger supergroup must sort after the smaller one.
        _seed(
            compare,
            {
                "/a.jpg": [1.0, 0.0], "/b.jpg": [1.0, 0.0],
                "/c.jpg": [0.0, 1.0], "/d.jpg": [0.0, 1.0], "/e.jpg": [0.0, 1.0],
                "/f.jpg": [0.0, 0.99], "/g.jpg": [0.0, 0.99], "/h.jpg": [0.0, 0.99],
            },
            {
                0: {"/a.jpg": 0.0, "/b.jpg": 0.0},
                1: {"/c.jpg": 0.0, "/d.jpg": 0.0, "/e.jpg": 0.0},
                2: {"/f.jpg": 0.0, "/g.jpg": 0.0, "/h.jpg": 0.0},
            },
        )
        supergroups = compare.compute_supergroups()
        sizes = [sum(len(compare.compare_result.file_groups[g]) for g in cluster) for cluster in supergroups]
        assert sizes == sorted(sizes)
        assert sizes == [2, 6]

    def test_below_min_viable_threshold_skips_supergrouping(self, tmp_path):
        compare = _make_compare(tmp_path, threshold=0.3)  # below SUPERGROUP_MIN_VIABLE_THRESHOLD
        _seed(
            compare,
            {"/a.jpg": [1.0, 0.0], "/b.jpg": [1.0, 0.0], "/c.jpg": [0.0, 1.0], "/d.jpg": [0.0, 1.0]},
            {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}, 1: {"/c.jpg": 0.0, "/d.jpg": 0.0}},
        )
        assert compare.compute_supergroups() == []
        assert compare.compare_result.supergroups == []

    def test_fewer_than_two_centroids_skips_supergrouping(self, tmp_path):
        compare = _make_compare(tmp_path, threshold=0.9)
        _seed(
            compare,
            {"/a.jpg": [1.0, 0.0], "/b.jpg": [1.0, 0.0]},
            {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}},
        )
        assert compare.compute_supergroups() == []

    def test_threshold_is_derived_from_run_threshold(self, tmp_path, monkeypatch):
        """Two centroids at exactly the run threshold's similarity must merge
        when scaled down by SUPERGROUP_THRESHOLD_RATIO, but not at the raw
        (unscaled) run threshold -- proves the ratio is actually applied."""
        compare = _make_compare(tmp_path, threshold=0.99)
        # cos similarity between these two unit vectors is ~0.95, which is
        # below 0.99 (the raw run threshold) but above 0.99 * 0.9 = 0.891.
        a = _unit([1.0, 0.0])
        b = _unit([0.95, 0.31])
        sim = float(np.dot(a, b))
        assert 0.891 < sim < 0.99

        _seed(
            compare,
            {"/a.jpg": a.tolist(), "/b.jpg": a.tolist(), "/c.jpg": b.tolist(), "/d.jpg": b.tolist()},
            {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}, 1: {"/c.jpg": 0.0, "/d.jpg": 0.0}},
        )
        supergroups = compare.compute_supergroups()
        assert len(supergroups) == 1
        assert sorted(supergroups[0]) == [0, 1]


# ---------------------------------------------------------------------------
# CompareResult.supergroups field
# ---------------------------------------------------------------------------

class TestCompareResultSupergroupsField:
    def test_defaults_to_empty_list(self):
        assert CompareResult().supergroups == []

    def test_is_independent_per_instance(self):
        """Mutating one instance's list must not leak into a fresh one (the
        classic mutable-default-argument trap)."""
        r1 = CompareResult()
        r1.supergroups.append([1, 2])
        r2 = CompareResult()
        assert r2.supergroups == []


# ---------------------------------------------------------------------------
# CompareWrapper supergroup navigation / label
# ---------------------------------------------------------------------------

def _make_wrapper(supergroups, group_indexes):
    wrapper = CompareWrapper(master=MagicMock(), compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=MagicMock())
    wrapper._compare = SimpleNamespace(compare_result=SimpleNamespace(supergroups=supergroups))
    wrapper.group_indexes = group_indexes
    wrapper.file_groups = {g: {f"/{g}.jpg": 0.0} for g in group_indexes}
    return wrapper


class TestCompareWrapperGetSupergroups:
    def test_no_compare_instance_returns_empty(self):
        wrapper = CompareWrapper(master=MagicMock(), compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=MagicMock())
        assert wrapper._get_supergroups() == []

    def test_old_pickle_without_attribute_returns_empty(self):
        """getattr default handles a CompareResult unpickled from before this
        feature existed, which won't have a `supergroups` attribute at all."""
        wrapper = CompareWrapper(master=MagicMock(), compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=MagicMock())
        wrapper._compare = SimpleNamespace(compare_result=SimpleNamespace())
        assert wrapper._get_supergroups() == []

    def test_returns_compare_result_supergroups(self):
        wrapper = _make_wrapper(supergroups=[[0], [1, 2]], group_indexes=[0, 1, 2])
        assert wrapper._get_supergroups() == [[0], [1, 2]]


class TestCompareWrapperSupergroupNavigation:
    def test_no_supergroups_toasts_and_does_not_change_index(self):
        wrapper = _make_wrapper(supergroups=[], group_indexes=[0, 1])
        wrapper.current_group_index = 0
        wrapper.show_next_supergroup()
        wrapper._app_actions.toast.assert_called_once()
        assert wrapper.current_group_index == 0

    def test_next_supergroup_jumps_to_first_group_of_next_cluster(self):
        wrapper = _make_wrapper(supergroups=[[0], [1, 2]], group_indexes=[0, 1, 2])
        wrapper.current_supergroup_index = 0
        wrapper.show_next_supergroup()
        assert wrapper.current_supergroup_index == 1
        # group_indexes.index(1) == 1 -- group 1 is the within-supergroup-order-first
        # member of cluster [1, 2] (since group_indexes lists 1 before 2).
        assert wrapper.current_group_index == 1
        wrapper._app_actions.create_media.assert_called_once()

    def test_prev_supergroup_wraps_around(self):
        wrapper = _make_wrapper(supergroups=[[0], [1, 2]], group_indexes=[0, 1, 2])
        wrapper.current_supergroup_index = 0
        wrapper.show_prev_supergroup()
        assert wrapper.current_supergroup_index == 1

    def test_picks_lowest_within_group_order_when_multiple_candidates_survive(self):
        """If the first listed member of a cluster isn't in group_indexes order-first,
        the lowest-positioned surviving candidate is still chosen, not cluster[0] blindly."""
        wrapper = _make_wrapper(supergroups=[[5, 2]], group_indexes=[2, 5])
        wrapper.current_supergroup_index = -1  # so next() lands on index 0
        wrapper.show_next_supergroup()
        assert wrapper.current_group_index == wrapper.group_indexes.index(2)

    def test_stale_group_index_is_skipped_defensively(self):
        """A supergroup referencing a group_index no longer in group_indexes
        (per docs §2.5 -- not re-clustered after purge/remove) doesn't crash;
        it falls back to whichever member still exists."""
        wrapper = _make_wrapper(supergroups=[[99, 1]], group_indexes=[1])
        wrapper.current_supergroup_index = -1
        wrapper.show_next_supergroup()
        assert wrapper.current_group_index == 0

    def test_all_members_stale_toasts_without_crashing(self):
        wrapper = _make_wrapper(supergroups=[[99]], group_indexes=[1])
        wrapper.current_supergroup_index = -1
        wrapper.show_next_supergroup()
        wrapper._app_actions.toast.assert_called_once()


class TestSupergroupLabelSuffix:
    def test_empty_when_no_supergroups(self):
        wrapper = _make_wrapper(supergroups=[], group_indexes=[0])
        assert wrapper._supergroup_label_suffix(0) == ""

    def test_reports_membership_by_lookup_not_cursor(self):
        """Looked up by membership, not current_supergroup_index, since plain
        group navigation can move into a different supergroup than the cursor
        last pointed at (see docstring in compare_wrapper.py)."""
        wrapper = _make_wrapper(supergroups=[[0], [1, 2]], group_indexes=[0, 1, 2])
        wrapper.current_supergroup_index = 0  # stale relative to actual_group_index below
        suffix = wrapper._supergroup_label_suffix(2)
        assert "2/2" in suffix

    def test_group_not_in_any_supergroup_returns_empty(self):
        wrapper = _make_wrapper(supergroups=[[0]], group_indexes=[0, 1])
        assert wrapper._supergroup_label_suffix(1) == ""

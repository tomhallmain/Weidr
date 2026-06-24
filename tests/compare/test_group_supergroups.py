"""
Unit tests for group mean-embedding centroids and supergroup clustering --
mean-pooling each compare group's per-file embeddings into a centroid, then
clustering those centroids into "supergroups" of related groups.

Covers BaseCompareEmbedding.compute_group_centroids / compute_supergroups,
the module-level cluster_group_indexes helper, and CompareWrapper's
supergroup navigation / label suffix.
"""
from __future__ import annotations

import os
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

    def test_transitive_chain_does_not_merge_with_complete_link(self):
        """A~B and B~C (but not A~C directly): complete-link keeps A+B together
        but refuses to absorb C because dot(A, C) is below the threshold."""
        a = _unit([1.0, 0.0, 0.0])
        b = _unit([0.7, 0.7, 0.0])
        c = _unit([0.0, 1.0, 0.0])
        centroids = {1: a, 2: b, 3: c}
        # dot(a,b) ≈ 0.7 >= 0.69 → {1,2} merge.
        # dot(a,c) = 0 < 0.69 → complete-link blocks {1,2} from absorbing 3.
        clusters = cluster_group_indexes(centroids, threshold=0.69)
        assert len(clusters) == 2
        flat = [sorted(c) for c in clusters]
        assert sorted(flat) == [[1, 2], [3]]

    def test_all_mutually_similar_centroids_merge(self):
        """When every pair exceeds the threshold, complete-link produces one cluster."""
        a = _unit([1.0, 0.0, 0.0])
        b = _unit([0.99, 0.1, 0.0])
        c = _unit([0.98, 0.15, 0.0])
        centroids = {1: a, 2: b, 3: c}
        clusters = cluster_group_indexes(centroids, threshold=0.9)
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


class TestPruneStaleSupergroups:
    def test_removed_group_dropped_from_its_cluster(self):
        """Group 1 was removed entirely (e.g. shrank to one file and got
        deleted by _update_groups_for_removed_file) -- file_groups no longer
        has it, so it must disappear from the cluster it used to share with 2."""
        result = CompareResult()
        result.file_groups = {0: {"/a.jpg": 0.0}, 2: {"/c.jpg": 0.0}}
        result.supergroups = [[0], [1, 2]]
        result.prune_stale_supergroups()
        assert result.supergroups == [[0], [2]]

    def test_cluster_emptied_entirely_is_removed(self):
        result = CompareResult()
        result.file_groups = {0: {"/a.jpg": 0.0}}
        result.supergroups = [[0], [1, 2]]
        result.prune_stale_supergroups()
        assert result.supergroups == [[0]]

    def test_noop_when_supergroups_already_empty(self):
        result = CompareResult()
        result.file_groups = {0: {"/a.jpg": 0.0}}
        result.prune_stale_supergroups()
        assert result.supergroups == []

    def test_noop_when_attribute_absent_old_pickle(self):
        """A CompareResult unpickled from before this feature existed won't
        have the attribute at all -- must not raise."""
        result = CompareResult()
        del result.supergroups
        result.file_groups = {0: {"/a.jpg": 0.0}}
        result.prune_stale_supergroups()  # should not raise
        assert not hasattr(result, "supergroups")

    def test_nothing_stale_leaves_supergroups_unchanged(self):
        result = CompareResult()
        result.file_groups = {0: {"/a.jpg": 0.0}, 1: {"/b.jpg": 0.0}, 2: {"/c.jpg": 0.0}}
        result.supergroups = [[0], [1, 2]]
        result.prune_stale_supergroups()
        assert result.supergroups == [[0], [1, 2]]

    def test_explicit_active_indexes_takes_precedence_over_file_groups(self):
        """_update_groups_for_removed_file deletes from its own self.file_groups
        before compare_result.file_groups is synced.  Passing active_group_indexes
        explicitly lets the live wrapper state win."""
        result = CompareResult()
        # file_groups still has group 1 (not yet synced)
        result.file_groups = {0: {"/a.jpg": 0.0}, 1: {"/b.jpg": 0.0}, 2: {"/c.jpg": 0.0}}
        result.supergroups = [[0, 1], [2]]
        # But the wrapper already deleted group 1 from its own dict
        result.prune_stale_supergroups(active_group_indexes={0, 2})
        assert result.supergroups == [[0], [2]]


class TestHasMeaningfulSupergroups:
    def test_false_when_empty(self):
        assert CompareResult().has_meaningful_supergroups() is False

    def test_false_when_all_clusters_are_singletons(self):
        result = CompareResult()
        result.supergroups = [[0], [1], [2]]
        assert result.has_meaningful_supergroups() is False

    def test_true_when_one_cluster_has_multiple_groups(self):
        result = CompareResult()
        result.supergroups = [[0, 1], [2]]
        assert result.has_meaningful_supergroups() is True

    def test_true_when_all_clusters_have_multiple_groups(self):
        result = CompareResult()
        result.supergroups = [[0, 1], [2, 3]]
        assert result.has_meaningful_supergroups() is True

    def test_false_when_attribute_absent(self):
        result = CompareResult()
        del result.supergroups
        assert result.has_meaningful_supergroups() is False


class TestClearSupergroups:
    def test_wipes_to_empty_list(self):
        result = CompareResult()
        result.supergroups = [[0], [1, 2]]
        result.clear_supergroups()
        assert result.supergroups == []


class TestRandomPurgeWipesSupergroups:
    def test_supergroups_cleared_after_purge(self, tmp_path):
        """No groups survive a random purge, so any existing supergroups
        (which reference group_index values that all just stopped existing)
        must be wiped, not left stale."""
        from PIL import Image

        paths = [str(tmp_path / f"{i}.png") for i in range(4)]
        for p in paths:
            Image.new("RGB", (10, 10), (100, 100, 100)).save(p)

        app_actions = MagicMock()
        app_actions.alert.return_value = True
        app_actions.delete.side_effect = lambda path, **kw: os.remove(path)

        wrapper = CompareWrapper(master=None, compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=app_actions)
        wrapper.file_groups = {0: {paths[0]: 0.0, paths[1]: 0.0}, 1: {paths[2]: 0.0, paths[3]: 0.0}}
        wrapper.current_supergroup_index = 1
        compare_result = CompareResult()
        compare_result.supergroups = [[0, 1]]
        wrapper._compare = SimpleNamespace(
            compare_result=compare_result, base_dir=str(tmp_path), COMPARE_MODE=CompareMode.CLIP_EMBEDDING
        )

        wrapper.random_purge_groups()

        assert wrapper._compare.compare_result.supergroups == []
        assert wrapper.current_supergroup_index == 0


class TestCompositeRecomputesSupergroupsAgainstRebuiltGroups:
    def test_supergroups_reference_new_renumbered_indices_not_old_ones(self, tmp_path):
        """Group 1 ({c, g}) loses g to the composite filter and drops below 2
        members, so it's discarded entirely -- group 2 ({d, e}) shifts down to
        new index 1. Supergroups recomputed afterward must reflect that shift:
        the merged cluster should be [0, 1] (new indices), never referencing
        the stale old index 2."""
        from compare.compare_manager import CompareManager

        a, b = [1.0, 0.0], [1.0, 0.0]
        d, e = [0.99, 0.05], [0.99, 0.05]
        c, g = [0.0, 1.0], [0.0, 1.0]

        compare = _make_compare(tmp_path, threshold=0.9)
        compare.compare_data.files_found = ["/a.jpg", "/b.jpg", "/c.jpg", "/g.jpg", "/d.jpg", "/e.jpg"]
        compare._file_embeddings = np.array([a, b, c, g, d, e])

        mgr = CompareManager(master=MagicMock(), app_actions=MagicMock())
        mgr.set_primary_mode(CompareMode.CLIP_EMBEDDING)
        wrapper = mgr._primary_wrapper()
        wrapper._compare = compare
        wrapper.file_groups = {
            0: {"/a.jpg": 0.0, "/b.jpg": 0.0},
            1: {"/c.jpg": 0.0, "/g.jpg": 0.0},
            2: {"/d.jpg": 0.0, "/e.jpg": 0.0},
        }
        mgr._combined_results = {p: 1.0 for p in ("/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg", "/e.jpg")}  # /g.jpg excluded

        mgr._apply_combined_results_to_primary(is_group_mode=True)

        assert wrapper.file_groups == {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}, 1: {"/d.jpg": 0.0, "/e.jpg": 0.0}}
        supergroups = compare.compare_result.supergroups
        assert len(supergroups) == 1
        assert sorted(supergroups[0]) == [0, 1]

    def test_no_surviving_groups_clears_supergroups(self, tmp_path):
        from compare.compare_manager import CompareManager

        compare = _make_compare(tmp_path, threshold=0.9)
        compare.compare_result.supergroups = [[0, 1]]

        mgr = CompareManager(master=MagicMock(), app_actions=MagicMock())
        mgr.set_primary_mode(CompareMode.CLIP_EMBEDDING)
        wrapper = mgr._primary_wrapper()
        wrapper._compare = compare
        wrapper.file_groups = {0: {"/a.jpg": 0.0, "/b.jpg": 0.0}}
        mgr._combined_results = {"/a.jpg": 1.0}  # /b.jpg excluded -> group drops to 1 member -> discarded

        mgr._apply_combined_results_to_primary(is_group_mode=True)

        assert compare.compare_result.supergroups == []

    def test_non_embedding_primary_with_embedding_secondary_skips_supergroups(self, tmp_path):
        """Composite bundle has one embedding instance and one non-embedding
        instance, with the non-embedding one as primary. supports_supergrouping()
        must gate the recompute so this doesn't crash (CompareColors has no
        per-file vector to build a centroid from); supergroups stay empty."""
        from compare.compare_colors import CompareColors
        from compare.compare_manager import CompareManager

        mgr = CompareManager(master=MagicMock(), app_actions=MagicMock())
        mgr.set_primary_mode(CompareMode.COLOR_MATCHING)
        mgr.add_mode_instance(CompareMode.CLIP_EMBEDDING)

        primary_compare = CompareColors(args=CompareArgs(base_dir=str(tmp_path)))
        wrapper = mgr._primary_wrapper()
        wrapper._compare = primary_compare
        wrapper.file_groups = {0: {"/a.png": 0.0, "/b.png": 0.0}}
        mgr._combined_results = {"/a.png": 1.0, "/b.png": 1.0}

        mgr._apply_combined_results_to_primary(is_group_mode=True)  # must not raise

        assert wrapper.file_groups == {0: {"/a.png": 0.0, "/b.png": 0.0}}
        assert primary_compare.compare_result.supergroups == []

    def test_both_instances_non_embedding_skips_supergroups(self, tmp_path):
        """Neither instance in the composite bundle is embedding-based --
        recompute must be skipped entirely (no centroid math possible) and the
        rebuild must still complete without crashing."""
        from compare.compare_colors import CompareColors
        from compare.compare_manager import CompareManager

        mgr = CompareManager(master=MagicMock(), app_actions=MagicMock())
        mgr.set_primary_mode(CompareMode.COLOR_MATCHING)
        mgr.add_mode_instance(CompareMode.SIZE)

        primary_compare = CompareColors(args=CompareArgs(base_dir=str(tmp_path)))
        wrapper = mgr._primary_wrapper()
        wrapper._compare = primary_compare
        wrapper.file_groups = {0: {"/a.png": 0.0, "/b.png": 0.0}}
        mgr._combined_results = {"/a.png": 1.0, "/b.png": 1.0}

        mgr._apply_combined_results_to_primary(is_group_mode=True)  # must not raise

        assert wrapper.file_groups == {0: {"/a.png": 0.0, "/b.png": 0.0}}
        assert primary_compare.compare_result.supergroups == []


# ---------------------------------------------------------------------------
# CompareWrapper supergroup navigation / label
# ---------------------------------------------------------------------------

def _make_wrapper(supergroups, group_indexes):
    wrapper = CompareWrapper(master=MagicMock(), compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=MagicMock())
    compare_result = CompareResult()
    compare_result.supergroups = supergroups
    wrapper._compare = SimpleNamespace(compare_result=compare_result)
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
        (accepted limitation: supergroups are not re-clustered after purge/remove) doesn't crash;
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
        # Cluster [0, 2] is meaningful (2 members) but group 1 is not a member.
        wrapper = _make_wrapper(supergroups=[[0, 2]], group_indexes=[0, 1, 2])
        assert wrapper._supergroup_label_suffix(1) == ""

    def test_empty_when_all_clusters_singleton(self):
        """Every cluster has exactly one group -- the partition mirrors the base
        groups and the label must be suppressed entirely."""
        wrapper = _make_wrapper(supergroups=[[0], [1]], group_indexes=[0, 1])
        assert wrapper._supergroup_label_suffix(0) == ""
        assert wrapper._supergroup_label_suffix(1) == ""

    def test_label_suppressed_after_deletion_collapses_last_multi_group_cluster(self):
        """Deleting a group that leaves every remaining cluster as a singleton
        should cause the label to disappear (total count changed AND trivial)."""
        wrapper = _make_wrapper(supergroups=[[0, 1]], group_indexes=[0, 1])
        # group 1 is still present — label is active
        assert wrapper._supergroup_label_suffix(0) != ""

        # Simulate _update_groups_for_removed_file removing group 1
        del wrapper.file_groups[1]
        wrapper._compare.compare_result.prune_stale_supergroups(
            active_group_indexes=set(wrapper.file_groups.keys())
        )
        # Only cluster [0] survives — all singletons now, label must be suppressed
        assert wrapper._supergroup_label_suffix(0) == ""


# ---------------------------------------------------------------------------
# _requires_new_compare
# ---------------------------------------------------------------------------

def _make_wrapper_with_result(base_dir, compare_mode, applied_group_sort=None):
    """Return a CompareWrapper whose _compare stub has a matching base_dir,
    COMPARE_MODE, and compare_result with the given applied_group_sort."""
    from utils.constants import Sort
    wrapper = CompareWrapper(master=MagicMock(), compare_mode=compare_mode, app_actions=MagicMock())
    result = SimpleNamespace(applied_group_sort=applied_group_sort)
    wrapper._compare = SimpleNamespace(
        base_dir=base_dir,
        COMPARE_MODE=compare_mode,
        compare_result=result,
    )
    return wrapper


class TestRequiresNewCompare:
    def test_true_when_no_compare(self):
        wrapper = CompareWrapper(master=MagicMock(), compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=MagicMock())
        assert wrapper._requires_new_compare("/some/dir", is_group=True) is True

    def test_false_when_same_dir_mode_and_sort(self, monkeypatch):
        from utils.constants import Sort
        import utils.config as cfg_module
        monkeypatch.setattr(cfg_module.config, "compare_group_sort", Sort.DESC)
        wrapper = _make_wrapper_with_result("/dir", CompareMode.CLIP_EMBEDDING, applied_group_sort=Sort.DESC)
        assert wrapper._requires_new_compare("/dir", is_group=True) is False

    def test_true_when_base_dir_changed(self, monkeypatch):
        from utils.constants import Sort
        import utils.config as cfg_module
        monkeypatch.setattr(cfg_module.config, "compare_group_sort", Sort.DESC)
        wrapper = _make_wrapper_with_result("/old", CompareMode.CLIP_EMBEDDING, applied_group_sort=Sort.DESC)
        assert wrapper._requires_new_compare("/new", is_group=True) is True

    def test_true_when_compare_mode_changed(self, monkeypatch):
        from utils.constants import Sort
        import utils.config as cfg_module
        monkeypatch.setattr(cfg_module.config, "compare_group_sort", Sort.DESC)
        wrapper = _make_wrapper_with_result("/dir", CompareMode.COLOR_MATCHING, applied_group_sort=Sort.DESC)
        wrapper.compare_mode = CompareMode.CLIP_EMBEDDING
        assert wrapper._requires_new_compare("/dir", is_group=True) is True

    def test_true_when_group_sort_changed_in_group_mode(self, monkeypatch):
        from utils.constants import Sort
        import utils.config as cfg_module
        # Last group run used ASC; config is now DESC.
        monkeypatch.setattr(cfg_module.config, "compare_group_sort", Sort.DESC)
        wrapper = _make_wrapper_with_result("/dir", CompareMode.CLIP_EMBEDDING, applied_group_sort=Sort.ASC)
        assert wrapper._requires_new_compare("/dir", is_group=True) is True

    def test_false_when_group_sort_changed_but_not_group_mode(self, monkeypatch):
        """Sort change must not force a new compare for search requests — sort
        only affects group display ordering."""
        from utils.constants import Sort
        import utils.config as cfg_module
        monkeypatch.setattr(cfg_module.config, "compare_group_sort", Sort.DESC)
        wrapper = _make_wrapper_with_result("/dir", CompareMode.CLIP_EMBEDDING, applied_group_sort=Sort.ASC)
        assert wrapper._requires_new_compare("/dir", is_group=False) is False

    def test_false_when_no_applied_sort_stored(self, monkeypatch):
        """applied_group_sort=None means no group run has happened yet;
        sort change should not force a new compare in that case."""
        from utils.constants import Sort
        import utils.config as cfg_module
        monkeypatch.setattr(cfg_module.config, "compare_group_sort", Sort.DESC)
        wrapper = _make_wrapper_with_result("/dir", CompareMode.CLIP_EMBEDDING, applied_group_sort=None)
        assert wrapper._requires_new_compare("/dir", is_group=True) is False

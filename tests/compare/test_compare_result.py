"""
Tests for compare/compare_result.py.

Covers pure-logic methods that do not require ML models or GPU:
  - hash_dir_files / equals_hash
  - sort_groups
  - validate_indices
  - store / load round-trip (uses tmp_path)
"""

import pickle
import pytest

from compare.compare_result import CompareResult
from utils.translations import _


class TestHashDirFiles:
    def test_returns_list(self):
        result = CompareResult.hash_dir_files(["a.jpg", "b.jpg"])
        assert isinstance(result, list)

    def test_empty_list(self):
        assert CompareResult.hash_dir_files([]) == []

    def test_same_files_produce_same_hash(self):
        files = ["x.png", "y.png", "z.png"]
        assert CompareResult.hash_dir_files(files) == CompareResult.hash_dir_files(files)

    def test_different_files_produce_different_hash(self):
        assert CompareResult.hash_dir_files(["a.jpg"]) != CompareResult.hash_dir_files(["b.jpg"])

    def test_order_matters(self):
        h1 = CompareResult.hash_dir_files(["a.jpg", "b.jpg"])
        h2 = CompareResult.hash_dir_files(["b.jpg", "a.jpg"])
        assert h1 != h2


class TestEqualsHash:
    def test_same_files_equals(self, tmp_path):
        files = ["img1.png", "img2.png"]
        cr = CompareResult(str(tmp_path), files)
        assert cr.equals_hash(files) is True

    def test_different_files_not_equals(self, tmp_path):
        cr = CompareResult(str(tmp_path), ["a.png"])
        assert cr.equals_hash(["b.png"]) is False

    def test_empty_equals_empty(self, tmp_path):
        cr = CompareResult(str(tmp_path), [])
        assert cr.equals_hash([]) is True


class TestSortGroups:
    def test_sorted_ascending_by_group_size(self, tmp_path):
        cr = CompareResult(str(tmp_path))
        cr.file_groups = {
            0: {"a.jpg": 0.1, "b.jpg": 0.2, "c.jpg": 0.3},  # size 3
            1: {"x.jpg": 0.1},                                 # size 1
            2: {"m.jpg": 0.1, "n.jpg": 0.2},                  # size 2
        }
        order = cr.sort_groups(cr.file_groups)
        sizes = [len(cr.file_groups[i]) for i in order]
        assert sizes == sorted(sizes)

    def test_sorted_descending_by_group_size(self, tmp_path):
        cr = CompareResult(str(tmp_path))
        cr.file_groups = {
            0: {"a.jpg": 0.1, "b.jpg": 0.2, "c.jpg": 0.3},  # size 3
            1: {"x.jpg": 0.1},                                 # size 1
            2: {"m.jpg": 0.1, "n.jpg": 0.2},                  # size 2
        }
        order = cr.sort_groups(cr.file_groups, reverse=True)
        sizes = [len(cr.file_groups[i]) for i in order]
        assert sizes == sorted(sizes, reverse=True)

    def test_empty_groups(self, tmp_path):
        cr = CompareResult(str(tmp_path))
        assert cr.sort_groups({}) == []

    def test_single_group(self, tmp_path):
        cr = CompareResult(str(tmp_path))
        cr.file_groups = {0: {"a.jpg": 0.5}}
        assert list(cr.sort_groups(cr.file_groups)) == [0]


class TestValidateIndices:
    def test_valid_indices_returns_true(self, tmp_path):
        files = ["a.jpg", "b.jpg", "c.jpg"]
        cr = CompareResult(str(tmp_path), files)
        cr.files_grouped = {0: 0.9, 1: 0.8, 2: 0.7}
        assert cr.validate_indices(files) is True

    def test_negative_index_returns_false(self, tmp_path):
        """A negative index would silently read files from the end of the list."""
        files = ["a.jpg", "b.jpg"]
        cr = CompareResult(str(tmp_path), files)
        cr.files_grouped = {0: 0.9, -1: 0.8}
        assert cr.validate_indices(files) is False

    def test_out_of_range_index_returns_false(self, tmp_path):
        files = ["a.jpg", "b.jpg"]
        cr = CompareResult(str(tmp_path), files)
        cr.files_grouped = {0: 0.9, 5: 0.8}  # index 5 is out of range
        assert cr.validate_indices(files) is False

    def test_empty_files_grouped_returns_true(self, tmp_path):
        files = ["a.jpg"]
        cr = CompareResult(str(tmp_path), files)
        cr.files_grouped = {}
        assert cr.validate_indices(files) is True


class TestStoreLoad:
    def test_load_overwrite_returns_fresh(self, tmp_path):
        files = ["a.jpg"]
        result = CompareResult.load(str(tmp_path), files, overwrite=True)
        assert isinstance(result, CompareResult)
        assert result.files_grouped == {}

    def test_load_no_cache_returns_fresh(self, tmp_path):
        files = ["a.jpg"]
        result = CompareResult.load(str(tmp_path), files)
        assert isinstance(result, CompareResult)

    def test_store_and_load_roundtrip(self, tmp_path):
        files = ["a.jpg", "b.jpg"]
        cr = CompareResult(str(tmp_path), files)
        cr.files_grouped = {0: 0.95}
        cr.is_complete = True
        cr.store()

        loaded = CompareResult.load(str(tmp_path), files)
        assert loaded.files_grouped == {0: 0.95}
        assert loaded.is_complete is True

    def test_load_hash_mismatch_raises(self, tmp_path):
        files_original = ["a.jpg", "b.jpg"]
        cr = CompareResult(str(tmp_path), files_original)
        cr.store()

        files_changed = ["c.jpg", "d.jpg"]
        with pytest.raises(ValueError):
            CompareResult.load(str(tmp_path), files_changed)

    def test_load_hash_mismatch_names_only_the_changes(self, tmp_path):
        """The message lists a few changed basenames per direction, not the
        whole file list (which can run to 200k paths)."""
        kept = [f"/media/keep_{i:03}.jpg" for i in range(100)]
        removed = [f"/media/gone_{i}.jpg" for i in range(7)]
        CompareResult(str(tmp_path), kept + removed).store()

        with pytest.raises(ValueError) as excinfo:
            CompareResult.load(str(tmp_path), kept + ["/media/new.jpg"])
        message = str(excinfo.value)
        assert _("Removed ({0}): {1}").format(
            7, "gone_0.jpg, gone_1.jpg, gone_2.jpg, gone_3.jpg, gone_4.jpg"
            + " " + _("(and {0} more)").format(2)) in message
        assert _("Added ({0}): {1}").format(1, "new.jpg") in message
        assert "keep_000.jpg" not in message

    def test_load_reordered_file_list_says_so(self, tmp_path):
        CompareResult(str(tmp_path), ["a.jpg", "b.jpg"]).store()
        with pytest.raises(ValueError) as excinfo:
            CompareResult.load(str(tmp_path), ["b.jpg", "a.jpg"])
        assert _("No file was added or removed, but the file list is in a different order.") in str(excinfo.value)

    def test_load_invalid_indices_returns_fresh(self, tmp_path):
        files = ["a.jpg"]
        cr = CompareResult(str(tmp_path), files)
        cr.files_grouped = {99: 0.9}  # invalid index
        cr.store()

        loaded = CompareResult.load(str(tmp_path), files)
        assert loaded.files_grouped == {}


class TestFilteredCheckpoints:
    """A run with a data filter has its own checkpoint file, keyed by
    compare_filters.filter_signature(), so it never collides with the
    unfiltered (or differently filtered) checkpoint of the same directory/mode."""

    def test_cache_path_includes_filter_key(self, tmp_path):
        plain = CompareResult.cache_path(str(tmp_path), None)
        filtered = CompareResult.cache_path(str(tmp_path), None, "abc123")
        assert plain != filtered
        assert filtered.endswith("weidr_result_default_fabc123.pkl")

    def test_no_filter_key_keeps_existing_file_name(self, tmp_path):
        assert CompareResult.cache_path(str(tmp_path), None, None) == CompareResult.cache_path(str(tmp_path), None)

    def test_filtered_run_ignores_unfiltered_checkpoint(self, tmp_path):
        unfiltered = CompareResult(str(tmp_path), ["a.jpg", "b.jpg", "c.jpg"])
        unfiltered.is_complete = True
        unfiltered.store()

        loaded = CompareResult.load(str(tmp_path), ["a.jpg"], filter_key="abc123")
        assert loaded.is_complete is False

    def test_filtered_store_and_load_roundtrip(self, tmp_path):
        files = ["a.jpg"]
        cr = CompareResult(str(tmp_path), files, filter_key="abc123")
        cr.files_grouped = {0: 0.5}
        cr.is_complete = True
        cr.store()

        assert CompareResult.load(str(tmp_path), files).is_complete is False
        loaded = CompareResult.load(str(tmp_path), files, filter_key="abc123")
        assert loaded.is_complete is True
        assert loaded.files_grouped == {0: 0.5}

    def test_loaded_pickle_without_filter_key_attribute_stores_to_requested_path(self, tmp_path):
        files = ["a.jpg"]
        cr = CompareResult(str(tmp_path), files, filter_key="abc123")
        del cr._filter_key  # as in a pickle written before filtered checkpoints
        with open(CompareResult.cache_path(str(tmp_path), None, "abc123"), "wb") as f:
            pickle.dump(cr, f)

        loaded = CompareResult.load(str(tmp_path), files, filter_key="abc123")
        assert loaded._filter_key == "abc123"


class TestCheckpointThreshold:
    """A checkpoint holds groups formed at one threshold; a run at another
    must not resume or reuse it."""

    def _stored(self, tmp_path, files, threshold):
        cr = CompareResult(str(tmp_path), files, threshold=threshold)
        cr.files_grouped = {0: (0, 1.0), 1: (0, 2.0)}
        cr.file_groups = {0: {files[0]: 1.0, files[1]: 2.0}}
        cr.is_complete = True
        cr.store()

    def test_same_threshold_reuses_the_checkpoint(self, tmp_path):
        files = ["a.jpg", "b.jpg"]
        self._stored(tmp_path, files, 15)
        loaded = CompareResult.load(str(tmp_path), files, threshold=15)
        assert loaded.is_complete is True

    def test_other_threshold_starts_fresh(self, tmp_path):
        files = ["a.jpg", "b.jpg"]
        self._stored(tmp_path, files, 15)
        loaded = CompareResult.load(str(tmp_path), files, threshold=10)
        assert loaded.is_complete is False
        assert loaded.files_grouped == {}

    def test_checkpoint_without_a_threshold_is_kept_and_adopts_the_run_threshold(self, tmp_path):
        files = ["a.jpg", "b.jpg"]
        self._stored(tmp_path, files, None)
        loaded = CompareResult.load(str(tmp_path), files, threshold=15)
        assert loaded.is_complete is True
        assert loaded._threshold == 15

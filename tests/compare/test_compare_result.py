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

    def test_load_invalid_indices_returns_fresh(self, tmp_path):
        files = ["a.jpg"]
        cr = CompareResult(str(tmp_path), files)
        cr.files_grouped = {99: 0.9}  # invalid index
        cr.store()

        loaded = CompareResult.load(str(tmp_path), files)
        assert loaded.files_grouped == {}

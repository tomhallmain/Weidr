"""
Tests for compare/compare_args.py.

Pure logic — no ML models, no file I/O.
"""

import pytest

from compare.compare_args import CompareArgs
from utils.constants import CompareMode, Mode


class TestNotSearching:
    def test_all_empty_returns_true(self):
        args = CompareArgs()
        assert args.not_searching() is True

    def test_search_media_path_set_returns_false(self):
        args = CompareArgs(search_media_path="/some/image.jpg")
        assert args.not_searching() is False

    def test_search_text_set_returns_false(self):
        args = CompareArgs(search_text="cat")
        assert args.not_searching() is False

    def test_search_text_negative_set_returns_false(self):
        args = CompareArgs(search_text_negative="dog")
        assert args.not_searching() is False

    def test_whitespace_only_string_is_empty(self):
        args = CompareArgs(search_text="   ")
        assert args.not_searching() is True

    def test_negative_search_media_path_set_returns_false(self):
        args = CompareArgs()
        args.negative_search_media_path = "/neg.jpg"
        assert args.not_searching() is False


class TestClone:
    def test_clone_is_independent_copy(self):
        args = CompareArgs(search_text="hello")
        clone = args.clone()
        clone.search_text = "world"
        assert args.search_text == "hello"

    def test_clone_does_not_copy_listener(self):
        sentinel = object()
        args = CompareArgs(listener=sentinel)
        clone = args.clone()
        assert clone.listener is None

    def test_clone_does_not_copy_app_actions(self):
        sentinel = object()
        args = CompareArgs(app_actions=sentinel)
        clone = args.clone()
        assert clone.app_actions is None

    def test_clone_preserves_compare_mode(self):
        args = CompareArgs(compare_mode=CompareMode.COLOR_MATCHING)
        clone = args.clone()
        assert clone.compare_mode == CompareMode.COLOR_MATCHING


class TestIsNewDataRequestRequired:
    def _base(self):
        return CompareArgs()

    def test_identical_args_returns_false(self):
        a = self._base()
        b = self._base()
        assert a._is_new_data_request_required(b) is False

    def test_different_threshold_returns_true(self):
        a = self._base()
        b = self._base()
        b.threshold = a.threshold + 0.1
        assert a._is_new_data_request_required(b) is True

    def test_different_counter_limit_returns_true(self):
        a = self._base()
        b = self._base()
        b.counter_limit = a.counter_limit + 100
        assert a._is_new_data_request_required(b) is True

    def test_different_inclusion_pattern_returns_true(self):
        a = self._base()
        b = self._base()
        b.inclusion_pattern = "*.png"
        assert a._is_new_data_request_required(b) is True

    def test_different_recursive_returns_true(self):
        a = self._base()
        b = self._base()
        b.recursive = not a.recursive
        assert a._is_new_data_request_required(b) is True

    def test_overwrite_false_to_true_returns_true(self):
        a = self._base()
        a.overwrite = False
        b = self._base()
        b.overwrite = True
        assert a._is_new_data_request_required(b) is True

    def test_overwrite_true_to_false_returns_false(self):
        a = self._base()
        a.overwrite = True
        b = self._base()
        b.overwrite = False
        assert a._is_new_data_request_required(b) is False


class TestDefaults:
    def test_default_mode_is_group(self):
        assert CompareArgs().mode == Mode.GROUP

    def test_default_compare_mode_is_clip(self):
        assert CompareArgs().compare_mode == CompareMode.CLIP_EMBEDDING

    def test_default_recursive_is_true(self):
        assert CompareArgs().recursive is True

    def test_default_find_duplicates_is_false(self):
        assert CompareArgs().find_duplicates is False

"""
Unit tests for compare/compare_args.py.
CompareArgs is a plain data class with no I/O — only the config defaults
are pulled in at import time, which is fine for a working project.
"""

import pytest

from compare.compare_args import CompareArgs
from utils.constants import CompareMode, Mode


class TestNotSearching:
    def test_defaults_are_not_searching(self):
        args = CompareArgs()
        assert args.not_searching() is True

    def test_with_search_media_path(self):
        args = CompareArgs(search_media_path="/some/image.jpg")
        assert args.not_searching() is False

    def test_with_search_text(self):
        args = CompareArgs(search_text="a cat")
        assert args.not_searching() is False

    def test_with_search_text_negative(self):
        args = CompareArgs(search_text_negative="a dog")
        assert args.not_searching() is False

    def test_with_negative_search_media_path(self):
        args = CompareArgs()
        args.negative_search_media_path = "/some/negative.jpg"
        assert args.not_searching() is False

    def test_empty_string_search_text_is_not_searching(self):
        args = CompareArgs(search_text="   ")
        assert args.not_searching() is True

    def test_empty_string_search_media_path_is_not_searching(self):
        args = CompareArgs(search_media_path="")
        assert args.not_searching() is True


class TestClone:
    def test_clone_is_not_same_object(self):
        args = CompareArgs(base_dir="/foo")
        clone = args.clone()
        assert clone is not args

    def test_clone_copies_base_dir(self):
        args = CompareArgs(base_dir="/foo")
        clone = args.clone()
        assert clone.base_dir == "/foo"

    def test_clone_copies_mode(self):
        args = CompareArgs(mode=Mode.BROWSE)
        clone = args.clone()
        assert clone.mode == Mode.BROWSE

    def test_clone_deep_copies_list_field(self):
        # include_videos is a bool, but verify that mutable fields don't share references
        args = CompareArgs()
        clone = args.clone()
        # Mutating the clone's base_dir should not affect the original
        clone.base_dir = "/other"
        assert args.base_dir != "/other"

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


class TestIsNewDataRequestRequired:
    def _pair(self, **overrides):
        base = CompareArgs()
        other = base.clone()
        for k, v in overrides.items():
            setattr(other, k, v)
        return base, other

    def test_identical_args_no_new_data_required(self):
        base, other = self._pair()
        assert base._is_new_data_request_required(other) is False

    def test_changed_threshold_requires_new_data(self):
        base, other = self._pair(threshold=0.5)
        base.threshold = 0.9
        assert base._is_new_data_request_required(other) is True

    def test_changed_counter_limit_requires_new_data(self):
        base, other = self._pair()
        other.counter_limit = base.counter_limit + 1
        assert base._is_new_data_request_required(other) is True

    def test_changed_file_filter_requires_new_data(self):
        base, other = self._pair()
        other.file_filter = "cats"
        assert base._is_new_data_request_required(other) is True

    def test_changed_recursive_requires_new_data(self):
        base, other = self._pair()
        other.recursive = not base.recursive
        assert base._is_new_data_request_required(other) is True

    def test_overwrite_false_to_true_requires_new_data(self):
        base = CompareArgs(overwrite=False)
        other = base.clone()
        other.overwrite = True
        assert base._is_new_data_request_required(other) is True

    def test_overwrite_true_to_false_does_not_require_new_data(self):
        base = CompareArgs(overwrite=True)
        other = base.clone()
        other.overwrite = False
        assert base._is_new_data_request_required(other) is False


class TestCompareArgsInit:
    def test_default_base_dir(self):
        args = CompareArgs()
        assert args.base_dir == "."

    def test_custom_base_dir(self):
        args = CompareArgs(base_dir="/images")
        assert args.base_dir == "/images"

    def test_find_duplicates_default_false(self):
        assert CompareArgs().find_duplicates is False

    def test_use_matrix_comparison_default_true(self):
        assert CompareArgs().use_matrix_comparison is True

    def test_negative_search_media_path_default_none(self):
        assert CompareArgs().negative_search_media_path is None

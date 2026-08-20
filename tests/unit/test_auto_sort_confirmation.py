"""Unit tests for AutoSortConfirmation category gating."""

import os

import pytest

from files.auto_sort_confirmation import AutoSortConfirmation


@pytest.fixture(autouse=True)
def _clear_categories():
    saved = set(AutoSortConfirmation.confirm_categories)
    AutoSortConfirmation.confirm_categories = set()
    yield
    AutoSortConfirmation.confirm_categories = saved


def test_no_categories_means_no_confirmation():
    assert not AutoSortConfirmation.is_confirm_required("landscapes")


def test_confirmation_required_for_configured_category():
    AutoSortConfirmation.set_confirm_required("landscapes", True)
    assert AutoSortConfirmation.is_confirm_required("landscapes")


def test_matching_is_case_insensitive():
    AutoSortConfirmation.set_confirm_required("LandScapes", True)
    assert AutoSortConfirmation.is_confirm_required("landscapes")
    assert AutoSortConfirmation.is_confirm_required("LANDSCAPES")


def test_unsetting_removes_the_requirement():
    AutoSortConfirmation.set_confirm_required("portraits", True)
    AutoSortConfirmation.set_confirm_required("PORTRAITS", False)
    assert not AutoSortConfirmation.is_confirm_required("portraits")


def test_empty_and_none_categories_never_require_confirmation():
    AutoSortConfirmation.set_confirm_required("", True)
    assert not AutoSortConfirmation.is_confirm_required("")
    assert not AutoSortConfirmation.is_confirm_required(None)
    assert AutoSortConfirmation.confirm_categories == set()


def test_set_categories_replaces_the_whole_set():
    AutoSortConfirmation.set_confirm_required("old", True)
    AutoSortConfirmation.set_categories(["New", "  Other  ", ""])
    assert not AutoSortConfirmation.is_confirm_required("old")
    assert AutoSortConfirmation.is_confirm_required("new")
    # Only casefolding is applied, so surrounding whitespace is significant.
    assert not AutoSortConfirmation.is_confirm_required("other")
    assert AutoSortConfirmation.is_confirm_required("  other  ")


def test_get_categories_returns_sorted_casefolded_names():
    AutoSortConfirmation.set_categories(["Zebra", "apple"])
    assert AutoSortConfirmation.get_categories() == ["apple", "zebra"]


def test_category_for_target_dir_is_the_basename():
    target = os.path.join("base", "dir", "landscapes")
    assert AutoSortConfirmation.category_for_target_dir(target) == "landscapes"


def test_category_for_target_dir_ignores_trailing_separator():
    target = os.path.join("base", "landscapes") + os.sep
    assert AutoSortConfirmation.category_for_target_dir(target) == "landscapes"


def test_category_for_target_dir_handles_empty():
    assert AutoSortConfirmation.category_for_target_dir("") is None
    assert AutoSortConfirmation.category_for_target_dir(None) is None

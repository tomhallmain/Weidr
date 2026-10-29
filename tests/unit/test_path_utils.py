"""Tests for utils/path_utils.py: directory overlap for directory groupings."""

import os

import pytest

from utils import path_utils
from utils.path_utils import (
    canonical_path,
    groupings_overlap,
    overlapping_pairs,
    paths_overlap,
    same_directory_set,
)


def test_identical_directories_overlap(tmp_path):
    assert paths_overlap(str(tmp_path), str(tmp_path))


def test_containment_overlaps_both_ways(tmp_path):
    sub = tmp_path / "sub"
    assert paths_overlap(str(tmp_path), str(sub))
    assert paths_overlap(str(sub), str(tmp_path))


def test_sibling_sharing_a_name_prefix_does_not_overlap(tmp_path):
    assert not paths_overlap(str(tmp_path / "b"), str(tmp_path / "bc"))


def test_trailing_separator_and_dot_segments_are_ignored(tmp_path):
    a = str(tmp_path / "x") + os.sep
    b = os.path.join(str(tmp_path), "y", "..", "x")
    assert canonical_path(a) == canonical_path(b)
    assert paths_overlap(a, b)


def test_case_follows_platform(tmp_path):
    lower, upper = str(tmp_path / "dir"), str(tmp_path / "DIR")
    assert paths_overlap(lower, upper) == (os.path.normcase("A") == os.path.normcase("a"))


def test_commonpath_value_error_means_no_overlap(monkeypatch, tmp_path):
    def _raise(_paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(path_utils.os.path, "commonpath", _raise)
    assert not paths_overlap(str(tmp_path / "a"), str(tmp_path / "b"))


def test_overlapping_pairs_returns_raw_inputs(tmp_path):
    a = [str(tmp_path / "a"), str(tmp_path / "b")]
    b = [str(tmp_path / "a" / "deep"), str(tmp_path / "c")]
    assert overlapping_pairs(a, b) == [(a[0], b[0])]
    assert groupings_overlap(a, b)
    assert not groupings_overlap([a[1]], [b[1]])


def test_same_directory_set_ignores_order_and_spelling(tmp_path):
    a = [str(tmp_path / "a"), str(tmp_path / "b")]
    b = [str(tmp_path / "b") + os.sep, str(tmp_path / "a")]
    assert same_directory_set(a, b)
    assert not same_directory_set(a, a[:1])


@pytest.mark.parametrize("value", ["~"])
def test_user_home_is_expanded(value):
    assert canonical_path(value) == canonical_path(os.path.expanduser(value))

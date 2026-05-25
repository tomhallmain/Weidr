"""
Tests for duplicate detection via CompareColors._probable_duplicates.

During run_comparison(), pairs with diff_score < THRESHHOLD_POTENTIAL_DUPLICATE (50)
are added to _probable_duplicates.  get_probable_duplicates() retrieves the list.

COLOR_MATCHING on solid-color images:
  - Identical colors: ΔE76 = 0 per pixel → diff_score = 0 → flagged as duplicate.
  - Near-identical same-family colors: ΔE76 cast to int ≈ 0 → also flagged.
  - Cross-family colors: ΔE76 >> 15 → not even similar → never reach the
    duplicate check.
"""

import glob
import os

import pytest
from PIL import Image

from compare.compare_args import CompareArgs
from compare.compare_colors import CompareColors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_gather(base_dir, **kw):
    return sorted(glob.glob(os.path.join(base_dir, "*.png")))


def _write_png(directory, name, color):
    path = os.path.join(directory, name)
    Image.new("RGB", (48, 48), color).save(path, format="PNG")
    return path


def _run_compare(tmp_path, threshold=15):
    """Build and fully run a CompareColors over all PNGs in tmp_path."""
    args = CompareArgs(base_dir=str(tmp_path), compare_threshold=threshold)
    cc = CompareColors(args=args, use_thumb=True, gather_files_func=_png_gather)
    cc.get_files()
    cc.get_data()
    cc.run_comparison()
    return cc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_identical_one_distinct(tmp_path):
    """Two identical red images and one blue image."""
    red_a = _write_png(str(tmp_path), "red_a.png", (220, 0, 0))
    red_b = _write_png(str(tmp_path), "red_b.png", (220, 0, 0))
    blue_a = _write_png(str(tmp_path), "blue_a.png", (0, 0, 220))
    cc = _run_compare(tmp_path)
    return cc, red_a, red_b, blue_a


# ---------------------------------------------------------------------------
# Identical-image detection
# ---------------------------------------------------------------------------

class TestIdenticalImageDetection:
    def test_identical_images_appear_in_probable_duplicates(
        self, two_identical_one_distinct
    ):
        cc, red_a, red_b, _ = two_identical_one_distinct
        dupes = cc.get_probable_duplicates()
        assert any(
            set(pair) == {red_a, red_b} for pair in dupes
        ), f"Expected (red_a, red_b) pair in duplicates; got: {dupes}"

    def test_cross_family_pair_not_in_probable_duplicates(
        self, two_identical_one_distinct
    ):
        cc, red_a, _, blue_a = two_identical_one_distinct
        dupes = cc.get_probable_duplicates()
        assert not any(
            set(pair) == {red_a, blue_a} for pair in dupes
        ), f"Red and blue should not be duplicates; got: {dupes}"

    def test_get_probable_duplicates_returns_list_of_two_tuples(
        self, two_identical_one_distinct
    ):
        cc, *_ = two_identical_one_distinct
        dupes = cc.get_probable_duplicates()
        assert isinstance(dupes, list)
        assert all(
            isinstance(pair, tuple) and len(pair) == 2 for pair in dupes
        )

    def test_no_self_pairs_in_duplicates(self, two_identical_one_distinct):
        cc, *_ = two_identical_one_distinct
        for f1, f2 in cc.get_probable_duplicates():
            assert f1 != f2, f"Self-pair found: {f1}"


# ---------------------------------------------------------------------------
# No duplicates when images are all distinct
# ---------------------------------------------------------------------------

class TestNoDuplicatesForDistinctImages:
    def test_three_distinct_colors_produce_no_duplicates(self, tmp_path):
        _write_png(str(tmp_path), "red.png", (220, 0, 0))
        _write_png(str(tmp_path), "blue.png", (0, 0, 220))
        _write_png(str(tmp_path), "green.png", (0, 220, 0))
        cc = _run_compare(tmp_path)
        assert cc.get_probable_duplicates() == []


# ---------------------------------------------------------------------------
# Near-identical (same-family) images flagged as duplicates
# ---------------------------------------------------------------------------

class TestNearIdenticalDetection:
    def test_exact_copies_flagged_as_duplicate(self, tmp_path):
        # THRESHHOLD_POTENTIAL_DUPLICATE = 50.  With 225 thumbnail pixels, the
        # per-pixel ΔE76 must be < 1 after int truncation for diff_score < 50.
        # Only pixel-identical copies reliably satisfy this, because even a 5-unit
        # RGB delta (~1 LAB unit per pixel × 225 pixels = 225 >> 50).
        path_a = _write_png(str(tmp_path), "a.png", (220, 0, 0))
        path_b = _write_png(str(tmp_path), "b.png", (220, 0, 0))
        cc = _run_compare(tmp_path)
        dupes = cc.get_probable_duplicates()
        assert any(
            set(pair) == {path_a, path_b} for pair in dupes
        ), f"Pixel-identical images not flagged as duplicate; got: {dupes}"

    def test_duplicate_pairs_are_symmetric_free(self, tmp_path):
        """Each pair appears only once — (A,B) but not also (B,A)."""
        _write_png(str(tmp_path), "a.png", (220, 0, 0))
        _write_png(str(tmp_path), "b.png", (220, 0, 0))
        _write_png(str(tmp_path), "c.png", (220, 0, 0))
        cc = _run_compare(tmp_path)
        dupes = cc.get_probable_duplicates()
        # Normalise to frozensets and check no pair appears twice.
        pair_sets = [frozenset(p) for p in dupes]
        assert len(pair_sets) == len(set(pair_sets)), f"Duplicate pairs found: {dupes}"


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

class TestThresholdSensitivity:
    def test_high_threshold_includes_cross_family_as_similar_but_not_as_duplicate(
        self, tmp_path
    ):
        # With a very high similarity threshold (500), red and blue images
        # may be considered similar, but their diff_score is huge (>> 50)
        # so they still should NOT appear in probable_duplicates.
        _write_png(str(tmp_path), "red.png", (220, 0, 0))
        _write_png(str(tmp_path), "blue.png", (0, 0, 220))
        cc = _run_compare(tmp_path, threshold=500)
        dupes = cc.get_probable_duplicates()
        # Even if similarity passes, duplicate gate requires diff_score < 50.
        # Cross-family diff_score is >> 50, so no duplicates expected.
        assert dupes == []

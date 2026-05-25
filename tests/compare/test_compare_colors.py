"""
Unit tests for compare/compare_colors.py.

Covers pure helper functions (is_any_x_true_consecutive, is_any_x_true_weighted,
get_median_values, RGB2HEX, get_image_thumb_colors), CompareColors constructor
configuration, _compute_color_diff with synthetic numpy arrays, and the
get_data / run_comparison / find_similars_to_image pipeline with real PNG files
from the compare_colors_dir fixture.
"""

import glob
import os

import numpy as np
import pytest

from compare.compare_args import CompareArgs
from compare.compare_colors import (
    CompareColors,
    RGB2HEX,
    get_image_thumb_colors,
    get_median_values,
    is_any_x_true_consecutive,
    is_any_x_true_weighted,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _png_gather(base_dir, **kw):
    return sorted(glob.glob(os.path.join(base_dir, "*.png")))


def _make_compare(base_dir, threshold=15, use_thumb=True):
    args = CompareArgs(base_dir=str(base_dir), compare_threshold=threshold)
    return CompareColors(args=args, use_thumb=use_thumb, gather_files_func=_png_gather)


# ---------------------------------------------------------------------------
# Pure helpers: is_any_x_true_consecutive
# ---------------------------------------------------------------------------

class TestIsAnyXTrueConsecutive:
    def test_empty_list_returns_0(self):
        assert is_any_x_true_consecutive([], 5) == 0

    def test_all_false_returns_0(self):
        assert is_any_x_true_consecutive([False] * 225, 112) == 0

    def test_all_true_exceeding_thresholds_returns_1(self):
        # 225 all-True values: count_true exceeds 112 and consecutive_runs_true
        # exceeds 10, so both gate conditions are met → 1.
        assert is_any_x_true_consecutive([True] * 225, 112) == 1

    def test_insufficient_total_count_returns_0(self):
        # 20 consecutive trues: consecutive_runs_true grows past 10, but
        # count_true (21) never exceeds x_threshold=112 → 0.
        assert is_any_x_true_consecutive([True] * 20, 112) == 0

    def test_alternating_trues_no_consecutive_runs_returns_0(self):
        # Alternating True/False: prior_bool is always False when True is seen,
        # so consecutive_count_true stays at 1 → consecutive_runs_true never
        # increments → 0 even when total count exceeds x_threshold.
        alternating = [i % 2 == 0 for i in range(300)]
        assert is_any_x_true_consecutive(alternating, 112) == 0


# ---------------------------------------------------------------------------
# Pure helpers: is_any_x_true_weighted
# ---------------------------------------------------------------------------

class TestIsAnyXTrueWeighted:
    # NOTE: The current implementation has a known defect — the loop body
    # never reads bool_list[i] and _count is never updated.  Tests here
    # document actual behaviour, not the intended algorithm.

    def test_empty_list_returns_0(self):
        assert is_any_x_true_weighted([], 5) == 0

    def test_always_returns_0_for_positive_threshold(self):
        # _count is always 0; 0 == x_threshold is only True when x_threshold==0.
        assert is_any_x_true_weighted([True] * 100, 5) == 0

    def test_zero_threshold_nonempty_returns_1(self):
        # x_threshold=0: first iteration checks 0==0 and count_true(1) > 0 → 1.
        assert is_any_x_true_weighted([True], 0) == 1

    def test_zero_threshold_empty_returns_0(self):
        # Empty list: loop never executes.
        assert is_any_x_true_weighted([], 0) == 0


# ---------------------------------------------------------------------------
# Pure helpers: get_median_values
# ---------------------------------------------------------------------------

class TestGetMedianValues:
    def test_empty_array_returns_empty(self):
        assert get_median_values(np.array([])) == []

    def test_single_value_returns_one_element(self):
        result = get_median_values(np.array([7.0]))
        assert len(result) == 1
        assert result[0] == pytest.approx(7.0)

    def test_first_median_is_middle_value(self):
        arr = np.array([1.0, 5.0, 9.0])
        result = get_median_values(arr)
        assert result[0] == pytest.approx(5.0)

    def test_returns_at_most_three_values(self):
        arr = np.arange(100, dtype=float)
        assert len(get_median_values(arr)) <= 3


# ---------------------------------------------------------------------------
# Pure helpers: RGB2HEX
# ---------------------------------------------------------------------------

class TestRGB2HEX:
    def test_pure_red(self):
        assert RGB2HEX((255, 0, 0)) == "#ff0000"

    def test_pure_green(self):
        assert RGB2HEX((0, 255, 0)) == "#00ff00"

    def test_pure_blue(self):
        assert RGB2HEX((0, 0, 255)) == "#0000ff"

    def test_black(self):
        assert RGB2HEX((0, 0, 0)) == "#000000"

    def test_white(self):
        assert RGB2HEX((255, 255, 255)) == "#ffffff"


# ---------------------------------------------------------------------------
# Pure helpers: get_image_thumb_colors
# ---------------------------------------------------------------------------

class TestGetImageThumbColors:
    def test_returns_correct_shape(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = get_image_thumb_colors(image, thumb_dim=15)
        assert result.shape == (225, 3)

    def test_solid_red_produces_consistent_lab(self):
        image = np.full((48, 48, 3), [220, 0, 0], dtype=np.uint8)
        result = get_image_thumb_colors(image, thumb_dim=15)
        # All 225 pixels are identical in LAB; std across pixels should be ~0.
        assert result.std(axis=0).max() < 1e-6


# ---------------------------------------------------------------------------
# Constructor configuration
# ---------------------------------------------------------------------------

class TestCompareColorsConstructor:
    def test_use_thumb_true_attributes(self, tmp_path):
        cc = _make_compare(tmp_path, use_thumb=True)
        assert cc.use_thumb is True
        assert cc.thumb_dim == 15
        assert cc.n_colors == 225
        assert cc.colors_below_threshold == 112
        assert cc.color_diff_alg is is_any_x_true_consecutive
        assert cc.CACHE_FILENAME == CompareColors.CACHE_FILENAME_THUMB

    def test_use_thumb_false_attributes(self, tmp_path):
        cc = _make_compare(tmp_path, use_thumb=False)
        assert cc.use_thumb is False
        assert cc.n_colors == 8
        assert cc.colors_below_threshold == 4
        assert cc.color_diff_alg is is_any_x_true_weighted
        assert cc.CACHE_FILENAME == CompareColors.CACHE_FILENAME_TOP

    def test_threshold_from_args(self, tmp_path):
        cc = _make_compare(tmp_path, threshold=25)
        assert cc.color_diff_threshold == 25

    def test_none_threshold_defaults_to_15(self, tmp_path):
        args = CompareArgs(base_dir=str(tmp_path))
        args.threshold = None
        cc = CompareColors(args=args, use_thumb=True, gather_files_func=_png_gather)
        assert cc.color_diff_threshold == 15

    def test_probable_duplicates_starts_empty(self, tmp_path):
        cc = _make_compare(tmp_path)
        assert cc.get_probable_duplicates() == []


# ---------------------------------------------------------------------------
# _compute_color_diff
# ---------------------------------------------------------------------------

class TestComputeColorDiff:
    @pytest.fixture
    def cc(self, tmp_path):
        return _make_compare(tmp_path, threshold=15)

    def test_identical_arrays_match(self, cc):
        arr = np.zeros((1, cc.n_colors, 3), dtype=float)
        result = cc._compute_color_diff(arr, arr)
        assert result[0] == 1

    def test_very_different_arrays_no_match(self, cc):
        base = np.zeros((1, cc.n_colors, 3), dtype=float)
        other = np.full((1, cc.n_colors, 3), 100.0)
        result = cc._compute_color_diff(base, other)
        assert result[0] == 0

    def test_return_diff_scores_returns_tuple(self, cc):
        arr = np.zeros((1, cc.n_colors, 3), dtype=float)
        similars, scores = cc._compute_color_diff(arr, arr, return_diff_scores=True)
        assert similars[0] == 1
        assert scores[0] == 0

    def test_diff_scores_increase_with_distance(self, cc):
        base = np.zeros((1, cc.n_colors, 3), dtype=float)
        close = np.full((1, cc.n_colors, 3), 1.0)
        far = np.full((1, cc.n_colors, 3), 100.0)
        _, close_scores = cc._compute_color_diff(base, close, return_diff_scores=True)
        _, far_scores = cc._compute_color_diff(base, far, return_diff_scores=True)
        assert close_scores[0] < far_scores[0]

    def test_zero_threshold_rejects_identical(self, tmp_path):
        # ΔE76 = 0, but 0 < 0 is False → no match.
        cc = _make_compare(tmp_path, threshold=0)
        arr = np.zeros((1, cc.n_colors, 3), dtype=float)
        assert cc._compute_color_diff(arr, arr)[0] == 0

    def test_multi_file_comparison(self, cc):
        # Three files: [A, B, C] all identical → all three should match.
        arr = np.zeros((3, cc.n_colors, 3), dtype=float)
        result = cc._compute_color_diff(arr, arr)
        assert all(r == 1 for r in result)


# ---------------------------------------------------------------------------
# Pipeline: get_data, run_comparison, find_similars_to_image
# ---------------------------------------------------------------------------

@pytest.fixture
def loaded_compare(compare_colors_dir):
    """CompareColors with all 18 solid-color PNGs loaded and data extracted."""
    d = compare_colors_dir["dir"]
    cc = _make_compare(d, threshold=15)
    cc.get_files()
    cc.get_data()
    return cc, compare_colors_dir


class TestGetData:
    def test_file_colors_has_correct_shape(self, loaded_compare):
        cc, colors = loaded_compare
        n = len(colors["red"]) + len(colors["blue"]) + len(colors["green"]) + len(colors["outliers"])
        assert cc._file_colors.shape == (n, 225, 3)

    def test_files_found_matches_all_images(self, loaded_compare):
        cc, colors = loaded_compare
        n = len(colors["red"]) + len(colors["blue"]) + len(colors["green"]) + len(colors["outliers"])
        assert len(cc.compare_data.files_found) == n


class TestRunComparison:
    def test_produces_at_least_three_groups(self, loaded_compare):
        cc, _ = loaded_compare
        cc.run_comparison()
        assert len(cc.compare_result.file_groups) >= 3

    def test_each_group_is_pure_color_family(self, loaded_compare):
        cc, colors = loaded_compare
        cc.run_comparison()
        red_set = set(colors["red"])
        blue_set = set(colors["blue"])
        green_set = set(colors["green"])

        for group_idx, group in cc.compare_result.file_groups.items():
            group_files = set(group.keys())
            is_pure = (
                group_files.issubset(red_set)
                or group_files.issubset(blue_set)
                or group_files.issubset(green_set)
            )
            assert is_pure, f"Group {group_idx} mixes color families: {group_files}"


class TestFindSimilarsToImage:
    def test_red_query_returns_only_red_siblings(self, loaded_compare, monkeypatch):
        import compare.compare_colors as _cc_module
        monkeypatch.setattr(_cc_module.config, "search_only_return_closest", True)

        cc, colors = loaded_compare
        search_path = colors["red"][0]
        search_idx = cc.compare_data.files_found.index(search_path)
        result = cc.find_similars_to_image(search_path, search_idx)

        matched = set(result[0].keys())
        assert matched.issubset(set(colors["red"])), (
            f"Non-red files in results: {matched - set(colors['red'])}"
        )
        assert len(matched) >= 4, f"Expected ≥4 red siblings, got {len(matched)}"

    def test_compare_result_sorted_by_increasing_diff_score(self, loaded_compare, monkeypatch):
        """The side-effect on compare_result.files_grouped is sorted; the return
        value ordering follows file-discovery order, not diff score."""
        import compare.compare_colors as _cc_module
        monkeypatch.setattr(_cc_module.config, "search_only_return_closest", True)

        cc, colors = loaded_compare
        search_path = colors["red"][0]
        search_idx = cc.compare_data.files_found.index(search_path)
        cc.find_similars_to_image(search_path, search_idx)

        scores = list(cc.compare_result.files_grouped.values())
        assert scores == sorted(scores), "compare_result.files_grouped should be sorted"


# ---------------------------------------------------------------------------
# remove_from_groups
# ---------------------------------------------------------------------------

class TestRemoveFromGroups:
    def test_removes_file_from_files_found(self, loaded_compare):
        cc, colors = loaded_compare
        target = colors["red"][0]
        assert target in cc.compare_data.files_found
        cc.remove_from_groups([target])
        assert target not in cc.compare_data.files_found

    def test_reduces_file_colors_array_by_one(self, loaded_compare):
        cc, colors = loaded_compare
        original_count = len(cc._file_colors)
        cc.remove_from_groups([colors["red"][0]])
        assert len(cc._file_colors) == original_count - 1

    def test_remove_multiple_files(self, loaded_compare):
        cc, colors = loaded_compare
        original_count = len(cc._file_colors)
        targets = colors["red"][:2]
        cc.remove_from_groups(targets)
        assert len(cc._file_colors) == original_count - 2
        for t in targets:
            assert t not in cc.compare_data.files_found

    def test_remove_nonexistent_file_is_noop(self, loaded_compare):
        cc, _ = loaded_compare
        original_count = len(cc._file_colors)
        cc.remove_from_groups(["/nonexistent/file.png"])
        assert len(cc._file_colors) == original_count

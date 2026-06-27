"""
Unit and integration tests for compare/compare_color_histogram.py.

Covers:
- _l1_distance pure function (symmetry, identity, triangle inequality, range)
- CompareColorHistogram constructor (threshold, class attributes, initial state)
- _compute_signature static method (correct shape, media type gating, error handling)
- get_data / run_comparison / find_similars_to_media pipeline with synthetic PNGs
- remove_from_groups housekeeping
- CompareMode.COLOR_HISTOGRAM enum properties (is_embedding, threshold_vals,
  threshold_str, text_search_modes exclusion)
- CompareManager._normalize_score for COLOR_HISTOGRAM
"""

import glob
import os

import numpy as np
import pytest
from PIL import Image

from compare.compare_args import CompareArgs
from compare.compare_color_histogram import CompareColorHistogram, _l1_distance, _HIST_LEN
from utils.constants import CompareMode


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _png_gather(base_dir, **kw):
    return sorted(glob.glob(os.path.join(base_dir, "*.png")))


def _make_compare(base_dir, threshold=0.15):
    args = CompareArgs(base_dir=str(base_dir), compare_threshold=threshold)
    return CompareColorHistogram(args=args, gather_files_func=_png_gather)


def _solid_png(path: str, color: tuple) -> str:
    Image.new("RGB", (48, 48), color).save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Pure helper: _l1_distance
# ---------------------------------------------------------------------------

class TestL1Distance:
    def test_identical_histograms_zero(self):
        h = np.ones(_HIST_LEN, dtype=np.float64) / _HIST_LEN
        assert _l1_distance(h, h) == pytest.approx(0.0)

    def test_all_mass_in_first_vs_last_bin_is_one(self):
        h1 = np.zeros(_HIST_LEN)
        h2 = np.zeros(_HIST_LEN)
        h1[0] = 1.0
        h2[-1] = 1.0
        # L1 = |1 - 0| + |0 - 1| = 2; divided by 2 = 1.0
        assert _l1_distance(h1, h2) == pytest.approx(1.0)

    def test_result_in_unit_interval(self):
        # _l1_distance divides by 2, bounding the result to [0, 1] only for
        # unit-sum inputs (i.e. proper probability vectors).  Normalise before
        # calling so the property actually holds.
        rng = np.random.default_rng(0)
        for _ in range(20):
            h1 = rng.random(_HIST_LEN)
            h1 /= h1.sum()
            h2 = rng.random(_HIST_LEN)
            h2 /= h2.sum()
            d = _l1_distance(h1, h2)
            assert 0.0 <= d <= 1.0

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        h1 = rng.random(_HIST_LEN)
        h2 = rng.random(_HIST_LEN)
        assert _l1_distance(h1, h2) == pytest.approx(_l1_distance(h2, h1))

    def test_triangle_inequality(self):
        rng = np.random.default_rng(7)
        h1 = rng.random(_HIST_LEN)
        h2 = rng.random(_HIST_LEN)
        h3 = rng.random(_HIST_LEN)
        assert _l1_distance(h1, h3) <= _l1_distance(h1, h2) + _l1_distance(h2, h3) + 1e-10

    def test_partial_overlap_between_zero_and_one(self):
        h1 = np.zeros(_HIST_LEN)
        h2 = np.zeros(_HIST_LEN)
        h1[0] = 0.5
        h1[1] = 0.5
        h2[0] = 0.5
        h2[2] = 0.5
        # Difference: bins 1 and 2 each contribute 0.5 → sum = 1.0 → / 2 = 0.5
        assert _l1_distance(h1, h2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Constructor configuration
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_threshold_when_none(self, tmp_path):
        args = CompareArgs(base_dir=str(tmp_path))
        args.threshold = None
        cc = CompareColorHistogram(args=args, gather_files_func=_png_gather)
        assert cc.threshold == pytest.approx(0.2)

    def test_threshold_from_args(self, tmp_path):
        cc = _make_compare(tmp_path, threshold=0.3)
        assert cc.threshold == pytest.approx(0.3)

    def test_cache_filename_is_set(self, tmp_path):
        cc = _make_compare(tmp_path)
        assert cc.CACHE_FILENAME == "image_color_histogram.pkl"

    def test_compare_mode_class_attribute(self, tmp_path):
        assert CompareColorHistogram.COMPARE_MODE == CompareMode.COLOR_HISTOGRAM

    def test_file_histograms_starts_empty(self, tmp_path):
        cc = _make_compare(tmp_path)
        assert cc._file_histograms.shape == (0, _HIST_LEN)

    def test_get_set_similarity_threshold(self, tmp_path):
        cc = _make_compare(tmp_path, threshold=0.1)
        assert cc.get_similarity_threshold() == pytest.approx(0.1)
        cc.set_similarity_threshold(0.25)
        assert cc.get_similarity_threshold() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _compute_signature
# ---------------------------------------------------------------------------

class TestComputeSignature:
    def test_returns_correct_shape_for_png(self, tmp_path):
        path = _solid_png(str(tmp_path / "red.png"), (220, 0, 0))
        sig = CompareColorHistogram._compute_signature(path)
        assert sig is not None
        assert sig.shape == (_HIST_LEN,)
        assert sig.dtype == np.float64

    def test_identical_images_produce_identical_signatures(self, tmp_path):
        p1 = _solid_png(str(tmp_path / "a.png"), (100, 150, 200))
        p2 = _solid_png(str(tmp_path / "b.png"), (100, 150, 200))
        s1 = CompareColorHistogram._compute_signature(p1)
        s2 = CompareColorHistogram._compute_signature(p2)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_different_colors_produce_different_signatures(self, tmp_path):
        p_red = _solid_png(str(tmp_path / "red.png"), (220, 0, 0))
        p_blue = _solid_png(str(tmp_path / "blue.png"), (0, 0, 220))
        s_red = CompareColorHistogram._compute_signature(p_red)
        s_blue = CompareColorHistogram._compute_signature(p_blue)
        assert not np.allclose(s_red, s_blue)

    def test_signature_is_normalized(self, tmp_path):
        path = _solid_png(str(tmp_path / "img.png"), (80, 160, 40))
        sig = CompareColorHistogram._compute_signature(path)
        # Each channel (H: 36, S: 16, V: 16) should sum to 1.0
        assert sig[:36].sum() == pytest.approx(1.0, abs=1e-6)
        assert sig[36:52].sum() == pytest.approx(1.0, abs=1e-6)
        assert sig[52:68].sum() == pytest.approx(1.0, abs=1e-6)

    def test_returns_none_for_audio_media_type(self, tmp_path, monkeypatch):
        import utils.media_utils as mu
        from utils.constants import CompareMediaType
        monkeypatch.setattr(mu, "get_media_type_for_path",
                            lambda _path: CompareMediaType.AUDIO)
        path = str(tmp_path / "fake.mp3")
        assert CompareColorHistogram._compute_signature(path) is None

    def test_returns_none_for_unconfigured_media_type(self, tmp_path, monkeypatch):
        import utils.media_utils as mu
        from utils.constants import CompareMediaType
        monkeypatch.setattr(mu, "get_media_type_for_path",
                            lambda _path: CompareMediaType.UNCONFIGURED)
        path = str(tmp_path / "fake.xyz")
        assert CompareColorHistogram._compute_signature(path) is None

    def test_returns_none_on_corrupt_file(self, tmp_path, monkeypatch):
        # A file with .png extension but invalid contents
        import utils.media_utils as mu
        from utils.constants import CompareMediaType
        monkeypatch.setattr(mu, "get_media_type_for_path",
                            lambda _path: CompareMediaType.IMAGE)
        path = tmp_path / "bad.png"
        path.write_bytes(b"not an image")
        assert CompareColorHistogram._compute_signature(str(path)) is None


# ---------------------------------------------------------------------------
# Pipeline: get_data
# ---------------------------------------------------------------------------

@pytest.fixture
def loaded_compare(compare_colors_dir):
    """CompareColorHistogram with all 18 solid-color PNGs loaded."""
    d = compare_colors_dir["dir"]
    cc = _make_compare(d, threshold=0.15)
    cc.get_files()
    cc.get_data()
    return cc, compare_colors_dir


class TestGetData:
    def test_file_histograms_shape(self, loaded_compare):
        cc, colors = loaded_compare
        n = (len(colors["red"]) + len(colors["blue"])
             + len(colors["green"]) + len(colors["outliers"]))
        assert cc._file_histograms.shape == (n, _HIST_LEN)

    def test_files_found_count(self, loaded_compare):
        cc, colors = loaded_compare
        n = (len(colors["red"]) + len(colors["blue"])
             + len(colors["green"]) + len(colors["outliers"]))
        assert len(cc.compare_data.files_found) == n

    def test_cache_hit_reuses_stored_signatures(self, compare_colors_dir):
        d = compare_colors_dir["dir"]
        cc1 = _make_compare(d, threshold=0.15)
        cc1.get_files()
        cc1.get_data()
        # CompareData.save_data() clears file_data_dict to free memory after
        # persisting to disk.  Compare the in-memory histogram matrix instead.
        first_hists = cc1._file_histograms.copy()
        first_files = list(cc1.compare_data.files_found)

        cc2 = _make_compare(d, threshold=0.15)
        cc2.get_files()
        cc2.get_data()

        # _png_gather is deterministic (sorted), so order must match.
        assert cc2.compare_data.files_found == first_files
        np.testing.assert_array_almost_equal(first_hists, cc2._file_histograms)

    def test_each_histogram_is_float64(self, loaded_compare):
        cc, _ = loaded_compare
        assert cc._file_histograms.dtype == np.float64

    def test_red_vs_blue_distance_above_threshold(self, loaded_compare):
        cc, colors = loaded_compare
        red_idx = cc.compare_data.files_found.index(colors["red"][0])
        blue_idx = cc.compare_data.files_found.index(colors["blue"][0])
        dist = _l1_distance(cc._file_histograms[red_idx], cc._file_histograms[blue_idx])
        assert dist > cc.threshold

    def test_red_variants_distance_below_threshold(self, histogram_stable_loaded):
        # Uses bin-stable images (all reds in the same HSV V-bin) so that
        # intra-family L1 = 0.0, well within any reasonable threshold.
        cc, colors = histogram_stable_loaded
        idx0 = cc.compare_data.files_found.index(colors["red"][0])
        idx1 = cc.compare_data.files_found.index(colors["red"][1])
        dist = _l1_distance(cc._file_histograms[idx0], cc._file_histograms[idx1])
        assert dist < cc.threshold


# ---------------------------------------------------------------------------
# Bin-stable fixture for grouping / distance tests
#
# compare_colors_dir was designed for LAB-space comparison and uses red
# variants with R in [210, 230].  In HSV histogram space (16 V-bins) this
# range straddles the bin 13/14 boundary, giving cross-bin L1 = 1.0 for
# "similar" reds — defeating the grouping assertions below.
#
# histogram_stable_compare picks colors guaranteed to land in the same V-bin
# for every family member (R,G,B in [225, 233] → V-bin 14 for each family),
# so intra-family L1 = 0.0 and cross-family L1 ≥ 1.0 (different H-bins).
# ---------------------------------------------------------------------------

@pytest.fixture
def histogram_stable_compare(tmp_path):
    d = str(tmp_path / "hist_stable")
    os.makedirs(d)

    # All primaries in V-bin 14: R/255 (or G/255, B/255) in [224/255, 239/255)
    reds   = [(225, 0, 0), (227, 0, 0), (229, 0, 0), (231, 0, 0), (233, 0, 0)]
    blues  = [(0, 0, 225), (0, 0, 227), (0, 0, 229), (0, 0, 231), (0, 0, 233)]
    greens = [(0, 225, 0), (0, 227, 0), (0, 229, 0), (0, 231, 0), (0, 233, 0)]

    def write(name, color):
        p = os.path.join(d, name)
        Image.new("RGB", (48, 48), color).save(p, format="PNG")
        return p

    return {
        "dir": d,
        "red":   [write(f"red_{i:02d}.png", c)   for i, c in enumerate(reds)],
        "blue":  [write(f"blue_{i:02d}.png", c)  for i, c in enumerate(blues)],
        "green": [write(f"green_{i:02d}.png", c) for i, c in enumerate(greens)],
    }


@pytest.fixture
def histogram_stable_loaded(histogram_stable_compare):
    d = histogram_stable_compare["dir"]
    cc = _make_compare(d, threshold=0.15)
    cc.get_files()
    cc.get_data()
    return cc, histogram_stable_compare


# ---------------------------------------------------------------------------
# Pipeline: run_comparison
# ---------------------------------------------------------------------------

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
        outlier_set = set(colors["outliers"])

        for group_idx, group in cc.compare_result.file_groups.items():
            group_files = set(group.keys())
            is_pure = (
                group_files.issubset(red_set)
                or group_files.issubset(blue_set)
                or group_files.issubset(green_set)
                or group_files.issubset(outlier_set)
            )
            assert is_pure, (
                f"Group {group_idx} mixes color families: {group_files}"
            )

    def test_all_red_siblings_in_same_group(self, histogram_stable_loaded):
        # Uses bin-stable images so all reds are equidistant (L1=0.0) and
        # always collapse into a single group at threshold 0.15.
        cc, colors = histogram_stable_loaded
        cc.run_comparison()
        red_set = set(colors["red"])
        red_groups = {
            g_idx
            for g_idx, group in cc.compare_result.file_groups.items()
            if any(f in red_set for f in group)
        }
        assert len(red_groups) == 1, "Red variants should be in exactly one group"

    def test_group_scores_are_distances_in_unit_interval(self, loaded_compare):
        cc, _ = loaded_compare
        cc.run_comparison()
        for group in cc.compare_result.file_groups.values():
            for score in group.values():
                assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"


# ---------------------------------------------------------------------------
# Pipeline: find_similars_to_media (search mode)
# ---------------------------------------------------------------------------

class TestFindSimilarsToMedia:
    def test_red_query_returns_only_red_siblings(self, histogram_stable_loaded, monkeypatch):
        # Uses bin-stable images: all 4 non-query reds are at L1=0.0 and
        # therefore all returned; blues/greens differ in H-bin so L1≥1.0.
        import compare.compare_color_histogram as _mod
        monkeypatch.setattr(_mod.config, "search_only_return_closest", True)

        cc, colors = histogram_stable_loaded
        search_path = colors["red"][0]
        search_idx = cc.compare_data.files_found.index(search_path)
        result = cc.find_similars_to_media(search_path, search_idx)

        matched = set(result[0].keys())
        red_set = set(colors["red"])
        assert matched.issubset(red_set), (
            f"Non-red files in results: {matched - red_set}"
        )
        assert len(matched) >= 4, f"Expected ≥4 red siblings, got {len(matched)}"

    def test_results_sorted_by_increasing_distance(self, loaded_compare, monkeypatch):
        import compare.compare_color_histogram as _mod
        monkeypatch.setattr(_mod.config, "search_only_return_closest", True)

        cc, colors = loaded_compare
        search_path = colors["red"][0]
        search_idx = cc.compare_data.files_found.index(search_path)
        cc.find_similars_to_media(search_path, search_idx)

        scores = list(cc.compare_result.files_grouped.values())
        assert scores == sorted(scores), "files_grouped should be sorted by distance"

    def test_all_scores_at_or_below_threshold_when_closest_only(
        self, loaded_compare, monkeypatch
    ):
        import compare.compare_color_histogram as _mod
        monkeypatch.setattr(_mod.config, "search_only_return_closest", True)

        cc, colors = loaded_compare
        search_path = colors["red"][0]
        search_idx = cc.compare_data.files_found.index(search_path)
        result = cc.find_similars_to_media(search_path, search_idx)

        for score in result[0].values():
            assert score <= cc.threshold

    def test_max_results_respected_when_not_closest_only(
        self, loaded_compare, monkeypatch
    ):
        import compare.compare_color_histogram as _mod
        monkeypatch.setattr(_mod.config, "search_only_return_closest", False)
        monkeypatch.setattr(_mod.config, "max_search_results", 3)

        cc, colors = loaded_compare
        search_path = colors["red"][0]
        search_idx = cc.compare_data.files_found.index(search_path)
        result = cc.find_similars_to_media(search_path, search_idx)

        assert len(result[0]) <= 3

    def test_returns_dict_keyed_by_zero(self, loaded_compare, monkeypatch):
        import compare.compare_color_histogram as _mod
        monkeypatch.setattr(_mod.config, "search_only_return_closest", True)

        cc, colors = loaded_compare
        search_path = colors["red"][0]
        search_idx = cc.compare_data.files_found.index(search_path)
        result = cc.find_similars_to_media(search_path, search_idx)

        assert 0 in result
        assert isinstance(result[0], dict)

    def test_search_path_not_in_corpus_is_added(self, compare_colors_dir, monkeypatch):
        import compare.compare_color_histogram as _mod
        monkeypatch.setattr(_mod.config, "search_only_return_closest", True)

        d = compare_colors_dir["dir"]
        cc = _make_compare(d, threshold=0.15)
        cc.get_files()
        cc.get_data()

        # Create a new file not in the original scan
        new_path = _solid_png(
            str(os.path.join(d, "new_red.png")), (222, 0, 0)
        )
        assert new_path not in cc.compare_data.files_found
        cc._run_search_on_path_histogram(new_path)
        # Should have been inserted at index 0
        assert cc.compare_data.files_found[0] == new_path


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

    def test_reduces_histogram_array_by_one(self, loaded_compare):
        cc, colors = loaded_compare
        original_rows = len(cc._file_histograms)
        cc.remove_from_groups([colors["red"][0]])
        assert len(cc._file_histograms) == original_rows - 1

    def test_remove_multiple_files(self, loaded_compare):
        cc, colors = loaded_compare
        original_rows = len(cc._file_histograms)
        targets = colors["red"][:2]
        cc.remove_from_groups(targets)
        assert len(cc._file_histograms) == original_rows - 2
        for t in targets:
            assert t not in cc.compare_data.files_found

    def test_remove_nonexistent_file_is_noop(self, loaded_compare):
        cc, _ = loaded_compare
        original_rows = len(cc._file_histograms)
        cc.remove_from_groups(["/nonexistent/file.png"])
        assert len(cc._file_histograms) == original_rows

    def test_histogram_order_preserved_after_removal(self, loaded_compare):
        cc, colors = loaded_compare
        # Remove the first red file; the remaining histograms should still
        # correspond correctly to their files_found entries.
        target = colors["red"][0]
        target_idx = cc.compare_data.files_found.index(target)
        # Record the histogram of a file that comes after the removed one.
        next_file = cc.compare_data.files_found[target_idx + 1]
        next_hist_before = cc._file_histograms[target_idx + 1].copy()

        cc.remove_from_groups([target])

        new_idx = cc.compare_data.files_found.index(next_file)
        np.testing.assert_array_equal(cc._file_histograms[new_idx], next_hist_before)


# ---------------------------------------------------------------------------
# is_related (stub behaviour)
# ---------------------------------------------------------------------------

class TestIsRelated:
    def test_always_returns_false(self, tmp_path):
        p1 = _solid_png(str(tmp_path / "a.png"), (0, 0, 0))
        p2 = _solid_png(str(tmp_path / "b.png"), (0, 0, 0))
        assert CompareColorHistogram.is_related(p1, p2) is False


# ---------------------------------------------------------------------------
# CompareMode.COLOR_HISTOGRAM enum properties
# ---------------------------------------------------------------------------

class TestCompareModeColorHistogram:
    def test_is_not_embedding(self):
        assert CompareMode.COLOR_HISTOGRAM.is_embedding() is False

    def test_threshold_vals_returns_expected_list(self):
        vals = CompareMode.COLOR_HISTOGRAM.threshold_vals()
        assert vals == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    def test_threshold_str_is_non_empty_string(self):
        s = CompareMode.COLOR_HISTOGRAM.threshold_str()
        assert isinstance(s, str) and len(s) > 0

    def test_not_in_text_search_modes(self):
        assert CompareMode.COLOR_HISTOGRAM not in CompareMode.text_search_modes()

    def test_get_text_returns_string(self):
        assert isinstance(CompareMode.COLOR_HISTOGRAM.get_text(), str)

    def test_present_in_all_compare_modes(self):
        assert CompareMode.COLOR_HISTOGRAM in list(CompareMode)

    def test_not_in_embedding_modes(self):
        assert CompareMode.COLOR_HISTOGRAM not in CompareMode.embedding_modes()


# ---------------------------------------------------------------------------
# CompareManager._normalize_score for COLOR_HISTOGRAM
# ---------------------------------------------------------------------------

class TestNormalizeScoreColorHistogram:
    @staticmethod
    def _normalize(score):
        from compare.compare_manager import CompareManager
        return CompareManager._normalize_score(CompareMode.COLOR_HISTOGRAM, score)

    def test_zero_distance_maps_to_one(self):
        assert self._normalize(0.0) == pytest.approx(1.0)

    def test_half_distance_maps_to_half(self):
        assert self._normalize(0.5) == pytest.approx(0.5)

    def test_full_distance_maps_to_zero(self):
        assert self._normalize(1.0) == pytest.approx(0.0)

    def test_clamped_above_one_maps_to_zero(self):
        assert self._normalize(1.5) == pytest.approx(0.0)

    def test_negative_score_clamped_to_one(self):
        assert self._normalize(-0.1) == pytest.approx(1.0)

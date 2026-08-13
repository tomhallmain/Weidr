"""
Integration coverage for CompareArgs.file_filter across a real get_files() /
get_data() pipeline -- the scope that was missing when the following bugs
shipped and only surfaced in a live user session (see BaseCompare.get_files,
BaseCompareEmbedding._reset_run_accumulators, and Utils.is_invalid_file):

  1. get_data()'s per-run result containers (compare_data.files_found and
     each subclass's own array, e.g. _file_colors) were never cleared before
     being repopulated, so a second get_data() call on the same instance
     (e.g. switching from GROUP to a text search, or narrowing the filter)
     accumulated on top of the previous call's results instead of replacing
     them.
  2. Utils.is_invalid_file exempted index 0 of the candidate list from
     file_filter for ANY search (has_search_reference), not just a genuine
     media-path search -- so in a text-only search (no search_media_path),
     the first candidate file always bypassed the filter regardless of
     whether it actually matched.

Fixing (1) and (2) means a filter can now legitimately narrow the candidate
pool to zero files. CompareData.save_data() and
BaseCompareEmbedding._compute_multiembedding_diff() both deliberately raise
in that case (a hard "no image data"/"no results" signal) rather than
returning an empty result -- previously unreachable from a filtered search
because the index-0 leak in (2) guaranteed at least one (wrong) match slipped
through instead. That raise is intentional, not a bug; covered here so a
future change doesn't accidentally soften it.

Uses CompareColors (not an embedding mode) so get_data() runs for real
against real PNGs on disk, with no model to mock. The zero-candidates case
below additionally uses CompareEmbeddingClip directly (no get_data() call,
no model needed), since that's the exact class from the real CLIP
text-search bug report this was found from.
"""
from __future__ import annotations

import glob
import os

import pytest

from compare.compare_args import CompareArgs
from compare.compare_colors import CompareColors
from compare.compare_embeddings_clip import CompareEmbeddingClip


def _png_gather(base_dir, **kw):
    return sorted(glob.glob(os.path.join(base_dir, "*.png")))


def _make_compare(base_dir, file_filter=None, search_text=None, search_media_path=None):
    args = CompareArgs(base_dir=str(base_dir), compare_threshold=15)
    args.file_filter = file_filter
    args.search_text = search_text
    args.search_media_path = search_media_path
    return CompareColors(args=args, use_thumb=True, gather_files_func=_png_gather)


class TestFileFilterAppliesInTextSearch:
    """A text search (search_text set, no search_media_path) must apply
    file_filter to every candidate, including the first one."""

    def test_filter_narrows_results_to_matching_files(self, compare_colors_dir):
        cc = _make_compare(
            compare_colors_dir["dir"], file_filter="red", search_text="anything"
        )
        cc.get_files()
        cc.get_data()

        assert set(cc.compare_data.files_found) == set(compare_colors_dir["red"])
        assert len(cc._file_colors) == len(compare_colors_dir["red"])

    def test_filter_matching_nothing_raises_no_image_data(self, compare_colors_dir):
        """Regression: the first candidate must not bypass file_filter just
        because it's index 0 of a search run with no reference file -- a
        filter matching nothing must raise CompareData's "no image data"
        error (its intended, deliberate signal for zero candidates), not
        quietly succeed with the one wrongly-included leaked file."""
        cc = _make_compare(
            compare_colors_dir["dir"],
            file_filter="no_such_term_matches_anything",
            search_text="anything",
        )
        cc.get_files()

        with pytest.raises(AssertionError, match="No image data found"):
            cc.get_data()

        assert cc.compare_data.files_found == []
        assert len(cc._file_colors) == 0


class TestSearchReferenceFileStillProtectedFromFilter:
    """A genuine media-path search (search_media_path set) must still keep
    the reference file at index 0 even when file_filter would exclude it --
    this is the one case has_search_reference is meant to protect."""

    def test_reference_file_kept_despite_non_matching_filter(self, compare_colors_dir):
        reference = compare_colors_dir["blue"][0]
        cc = _make_compare(
            compare_colors_dir["dir"], file_filter="red", search_media_path=reference
        )
        cc.get_files()
        cc.get_data()

        assert reference in cc.compare_data.files_found
        # Every other result must still be filter-matching.
        others = set(cc.compare_data.files_found) - {reference}
        assert others == set(compare_colors_dir["red"])


class TestGetDataDoesNotAccumulateAcrossRuns:
    """Regression: repeated get_data() calls on the same instance (as
    CompareWrapper.run() does when reusing a Compare object across mode/
    filter changes) must fully replace the previous run's results, not
    accumulate on top of them."""

    def test_narrowing_filter_on_second_call_drops_prior_matches(self, compare_colors_dir):
        cc = _make_compare(
            compare_colors_dir["dir"], file_filter="red", search_text="anything"
        )
        cc.get_files()
        cc.get_data()
        assert set(cc.compare_data.files_found) == set(compare_colors_dir["red"])

        cc.args.file_filter = "blue"
        cc.get_files()
        cc.get_data()

        assert set(cc.compare_data.files_found) == set(compare_colors_dir["blue"])
        assert len(cc._file_colors) == len(compare_colors_dir["blue"])

    def test_result_arrays_stay_aligned_with_files_found_after_second_call(
        self, compare_colors_dir
    ):
        cc = _make_compare(compare_colors_dir["dir"], file_filter=None)
        cc.get_files()
        cc.get_data()

        cc.args.file_filter = "outlier"
        cc.get_files()
        cc.get_data()

        assert len(cc.compare_data.files_found) == len(compare_colors_dir["outliers"])
        assert len(cc._file_colors) == len(cc.compare_data.files_found)

    def test_removing_the_filter_on_second_call_restores_the_full_set(
        self, compare_colors_dir
    ):
        all_files = (
            compare_colors_dir["red"]
            + compare_colors_dir["blue"]
            + compare_colors_dir["green"]
            + compare_colors_dir["outliers"]
        )
        cc = _make_compare(compare_colors_dir["dir"], file_filter="red")
        cc.get_files()
        cc.get_data()
        assert set(cc.compare_data.files_found) == set(compare_colors_dir["red"])

        cc.args.file_filter = None
        cc.get_files()
        cc.get_data()

        assert set(cc.compare_data.files_found) == set(all_files)
        assert len(cc._file_colors) == len(all_files)


class TestZeroCandidatesIsADeliberateError:
    """Zero candidate files is an intentional hard-error signal in this
    codebase (not a "no results" outcome to swallow quietly) --
    save_data() raises "No image data found". Covered here so this contract
    stays locked in rather than getting "fixed" away.

    _compute_multiembedding_diff()'s own "No results found" raise for an
    empty _file_embeddings is NOT covered here: get_data() always calls
    save_data() first, which already raises before search_multimodal() could
    ever run with zero embeddings, so that branch is unreachable through the
    real pipeline -- a synthetic call straight into search_multimodal() with
    a hand-emptied _file_embeddings doesn't hit it either; it hits a numpy
    ValueError inside _compute_embedding_diff's np.vectorize call first
    ("cannot call vectorize on size 0 inputs unless otypes is set"), which
    only proves the same thing from another angle.
    """

    def test_save_data_raises_when_no_files_found(self, tmp_path):
        cc = CompareEmbeddingClip(args=CompareArgs(base_dir=str(tmp_path)))
        cc.compare_data.files_found = []
        cc.compare_data.has_new_file_data = False

        with pytest.raises(AssertionError, match="No image data found"):
            cc.compare_data.save_data(overwrite=False, verbose=False)

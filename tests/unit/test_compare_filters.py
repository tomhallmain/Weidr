"""
Unit tests for compare/compare_filters.py and the normalization helper in
compare/compare_manager.py.

All I/O is mocked — no real files are read.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from compare.compare_filters import (
    CompareFilter,
    CompareFilterGroup,
    FilterOperator,
    ModelFilter,
    SizeFilter,
    apply_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FILES = ["/img/a.png", "/img/b.png", "/img/c.png"]

def _size_reader(sizes: dict):
    """Return a patcher for extract_size_from_media keyed by file path."""
    return patch(
        "compare.compare_size.extract_size_from_media",
        side_effect=lambda fp: sizes.get(fp),
    )


def _model_reader(models: dict):
    """Fake metadata_reader whose get_models returns (models, loras) per file."""
    reader = MagicMock()
    reader.get_models.side_effect = lambda fp: models.get(fp, ([], []))
    return reader


# ===========================================================================
# SizeFilter — is_active
# ===========================================================================

class TestSizeFilterIsActive:
    def test_all_none_is_inactive(self):
        assert SizeFilter().is_active() is False

    def test_min_size_makes_active(self):
        assert SizeFilter(min_size=(512, 512)).is_active() is True

    def test_max_size_makes_active(self):
        assert SizeFilter(max_size=(1024, 1024)).is_active() is True

    def test_exact_size_makes_active(self):
        assert SizeFilter(exact_size=(512, 512)).is_active() is True

    def test_tolerance_alone_is_inactive(self):
        assert SizeFilter(size_tolerance=10).is_active() is False


# ===========================================================================
# ModelFilter — is_active
# ===========================================================================

class TestModelFilterIsActive:
    def test_none_models_is_inactive(self):
        assert ModelFilter().is_active() is False

    def test_empty_list_is_inactive(self):
        assert ModelFilter(models=[]).is_active() is False

    def test_nonempty_list_is_active(self):
        assert ModelFilter(models=["dreamshaper"]).is_active() is True


# ===========================================================================
# CompareFilterGroup — is_active and add
# ===========================================================================

class TestCompareFilterGroup:
    def test_empty_group_is_inactive(self):
        assert CompareFilterGroup().is_active() is False

    def test_group_with_inactive_children_is_inactive(self):
        g = CompareFilterGroup().add(SizeFilter()).add(ModelFilter())
        assert g.is_active() is False

    def test_group_with_one_active_child_is_active(self):
        g = CompareFilterGroup().add(SizeFilter(min_size=(1, 1)))
        assert g.is_active() is True

    def test_add_returns_self_for_chaining(self):
        g = CompareFilterGroup()
        result = g.add(SizeFilter())
        assert result is g

    def test_default_operator_is_and(self):
        assert CompareFilterGroup().operator == FilterOperator.AND


# ===========================================================================
# apply_filter — pass-through cases
# ===========================================================================

class TestApplyFilterPassthrough:
    def test_inactive_size_filter_returns_all_files(self):
        result = apply_filter(FILES[:], SizeFilter())
        assert result == FILES

    def test_inactive_model_filter_returns_all_files(self):
        result = apply_filter(FILES[:], ModelFilter())
        assert result == FILES

    def test_inactive_group_returns_all_files(self):
        g = CompareFilterGroup().add(SizeFilter())
        result = apply_filter(FILES[:], g)
        assert result == FILES

    def test_unknown_filter_type_returns_all_files(self):
        class _Alien(CompareFilter):
            def is_active(self):
                return True

        result = apply_filter(FILES[:], _Alien())
        assert result == FILES


# ===========================================================================
# SizeFilter — filtering logic
# ===========================================================================

class TestApplySizeFilter:
    def test_exact_match_passes(self):
        sizes = {"/img/a.png": (512, 512), "/img/b.png": (512, 512)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"],
                                  SizeFilter(exact_size=(512, 512)))
        assert result == ["/img/a.png", "/img/b.png"]

    def test_exact_mismatch_excluded(self):
        sizes = {"/img/a.png": (512, 512), "/img/b.png": (1024, 1024)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"],
                                  SizeFilter(exact_size=(512, 512)))
        assert result == ["/img/a.png"]

    def test_exact_within_tolerance_passes(self):
        sizes = {"/img/a.png": (514, 510)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(exact_size=(512, 512), size_tolerance=5))
        assert result == ["/img/a.png"]

    def test_exact_outside_tolerance_excluded(self):
        sizes = {"/img/a.png": (520, 512)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(exact_size=(512, 512), size_tolerance=5))
        assert result == []

    def test_min_size_too_small_excluded(self):
        sizes = {"/img/a.png": (256, 256)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(min_size=(512, 512)))
        assert result == []

    def test_min_size_exactly_at_boundary_passes(self):
        sizes = {"/img/a.png": (512, 512)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(min_size=(512, 512)))
        assert result == ["/img/a.png"]

    def test_max_size_too_large_excluded(self):
        sizes = {"/img/a.png": (2048, 2048)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(max_size=(1024, 1024)))
        assert result == []

    def test_max_size_exactly_at_boundary_passes(self):
        sizes = {"/img/a.png": (1024, 1024)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(max_size=(1024, 1024)))
        assert result == ["/img/a.png"]

    def test_unreadable_file_is_excluded(self):
        # extract_size_from_media returns None for unreadable files
        with _size_reader({}):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(min_size=(1, 1)))
        assert result == []

    def test_min_and_max_together(self):
        sizes = {
            "/img/small.png":  (256, 256),
            "/img/medium.png": (512, 512),
            "/img/large.png":  (2048, 2048),
        }
        f = SizeFilter(min_size=(512, 512), max_size=(1024, 1024))
        with _size_reader(sizes):
            result = apply_filter(list(sizes.keys()), f)
        assert result == ["/img/medium.png"]

    def test_only_width_axis_checked_for_min(self):
        # Width below min → excluded even if height is fine
        sizes = {"/img/a.png": (256, 1024)}
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png"],
                                  SizeFilter(min_size=(512, 512)))
        assert result == []


# ===========================================================================
# ModelFilter — filtering logic
# ===========================================================================

class TestApplyModelFilter:
    def test_include_matching_model_passes(self):
        reader = _model_reader({"/img/a.png": (["dreamshaper"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='include'),
                               metadata_reader=reader)
        assert result == ["/img/a.png"]

    def test_include_non_matching_excluded(self):
        reader = _model_reader({"/img/a.png": (["sdxl_base"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='include'),
                               metadata_reader=reader)
        assert result == []

    def test_exclude_matching_model_excluded(self):
        reader = _model_reader({"/img/a.png": (["dreamshaper"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='exclude'),
                               metadata_reader=reader)
        assert result == []

    def test_exclude_non_matching_passes(self):
        reader = _model_reader({"/img/a.png": (["sdxl_base"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='exclude'),
                               metadata_reader=reader)
        assert result == ["/img/a.png"]

    def test_match_is_substring_case_insensitive(self):
        reader = _model_reader({"/img/a.png": (["DreamShaper_v8"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='include'),
                               metadata_reader=reader)
        assert result == ["/img/a.png"]

    def test_match_any_true_one_name_sufficient(self):
        reader = _model_reader({"/img/a.png": (["dreamshaper"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper", "sdxl"], mode='include',
                                           match_any=True),
                               metadata_reader=reader)
        assert result == ["/img/a.png"]

    def test_match_any_false_all_names_required(self):
        # File only has one of the two required models
        reader = _model_reader({"/img/a.png": (["dreamshaper"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper", "sdxl"], mode='include',
                                           match_any=False),
                               metadata_reader=reader)
        assert result == []

    def test_match_any_false_all_present_passes(self):
        reader = _model_reader({"/img/a.png": (["dreamshaper", "sdxl_base"], [])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper", "sdxl"], mode='include',
                                           match_any=False),
                               metadata_reader=reader)
        assert result == ["/img/a.png"]

    def test_include_loras_true_lora_counts(self):
        reader = _model_reader({"/img/a.png": ([], ["detail_tweaker_lora"])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["detail_tweaker"], mode='include',
                                           include_loras=True),
                               metadata_reader=reader)
        assert result == ["/img/a.png"]

    def test_include_loras_false_lora_ignored(self):
        reader = _model_reader({"/img/a.png": ([], ["detail_tweaker_lora"])})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["detail_tweaker"], mode='include',
                                           include_loras=False),
                               metadata_reader=reader)
        assert result == []

    def test_no_metadata_treated_as_no_models_include_mode(self):
        # No metadata → no models → fails include filter
        reader = _model_reader({})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='include'),
                               metadata_reader=reader)
        assert result == []

    def test_no_metadata_treated_as_no_models_exclude_mode(self):
        # No metadata → no models → passes exclude filter (nothing to exclude)
        reader = _model_reader({})
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='exclude'),
                               metadata_reader=reader)
        assert result == ["/img/a.png"]

    def test_metadata_read_exception_treated_as_no_models(self):
        reader = MagicMock()
        reader.get_models.side_effect = Exception("corrupt metadata")
        result = apply_filter(["/img/a.png"],
                               ModelFilter(models=["dreamshaper"], mode='include'),
                               metadata_reader=reader)
        assert result == []


# ===========================================================================
# CompareFilterGroup — AND / OR / NOT operators
# ===========================================================================

class TestGroupOperators:
    """Test group logic using SizeFilter mocks so we control which files pass."""

    def _size_pass_a(self):
        """SizeFilter that passes only /img/a.png."""
        return SizeFilter(exact_size=(512, 512))

    def _size_pass_b(self):
        """SizeFilter that passes only /img/b.png."""
        return SizeFilter(exact_size=(1024, 1024))

    def test_and_both_must_pass(self):
        sizes = {
            "/img/a.png": (512, 512),
            "/img/b.png": (512, 512),
        }
        # Two independent size filters that both require 512×512 — all matching files pass
        g = CompareFilterGroup(
            operator=FilterOperator.AND,
            filters=[SizeFilter(min_size=(256, 256)), SizeFilter(max_size=(1024, 1024))],
        )
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"], g)
        assert sorted(result) == ["/img/a.png", "/img/b.png"]

    def test_and_one_fails_excluded(self):
        sizes = {"/img/a.png": (512, 512), "/img/b.png": (2048, 2048)}
        g = CompareFilterGroup(
            operator=FilterOperator.AND,
            filters=[SizeFilter(max_size=(1024, 1024))],
        )
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"], g)
        assert result == ["/img/a.png"]

    def test_or_either_may_pass(self):
        # f1 passes a (512), f2 passes b (1024) — OR should return both
        sizes = {"/img/a.png": (512, 512), "/img/b.png": (1024, 1024)}
        g = CompareFilterGroup(
            operator=FilterOperator.OR,
            filters=[
                SizeFilter(exact_size=(512, 512)),
                SizeFilter(exact_size=(1024, 1024)),
            ],
        )
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"], g)
        assert sorted(result) == ["/img/a.png", "/img/b.png"]

    def test_or_only_one_child_matches(self):
        sizes = {"/img/a.png": (512, 512), "/img/b.png": (999, 999)}
        g = CompareFilterGroup(
            operator=FilterOperator.OR,
            filters=[
                SizeFilter(exact_size=(512, 512)),
                SizeFilter(exact_size=(1024, 1024)),
            ],
        )
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"], g)
        assert result == ["/img/a.png"]

    def test_not_excludes_matches(self):
        sizes = {"/img/a.png": (512, 512), "/img/b.png": (1024, 1024)}
        # NOT filter: exclude files that are 512×512 → only b passes
        g = CompareFilterGroup(
            operator=FilterOperator.NOT,
            filters=[SizeFilter(exact_size=(512, 512))],
        )
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"], g)
        assert result == ["/img/b.png"]

    def test_empty_active_children_returns_all(self):
        g = CompareFilterGroup(
            operator=FilterOperator.AND,
            filters=[SizeFilter()],  # inactive
        )
        result = apply_filter(FILES[:], g)
        assert result == FILES


# ===========================================================================
# Nested group (tree depth > 1)
# ===========================================================================

class TestNestedGroup:
    def test_and_of_or_groups(self):
        """
        Outer AND of two OR groups:
          OR-1: exact 512×512
          OR-2: exact 1024×1024

        Only files satisfying BOTH ORs simultaneously can pass — impossible for
        a single file, so result is empty.
        """
        sizes = {"/img/a.png": (512, 512), "/img/b.png": (1024, 1024)}
        or1 = CompareFilterGroup(
            operator=FilterOperator.OR,
            filters=[SizeFilter(exact_size=(512, 512))],
        )
        or2 = CompareFilterGroup(
            operator=FilterOperator.OR,
            filters=[SizeFilter(exact_size=(1024, 1024))],
        )
        outer = CompareFilterGroup(operator=FilterOperator.AND, filters=[or1, or2])
        with _size_reader(sizes):
            result = apply_filter(["/img/a.png", "/img/b.png"], outer)
        assert result == []


# ===========================================================================
# CompareArgs — data_filter field
# ===========================================================================

class TestCompareArgsDataFilter:
    def test_default_is_none(self):
        from compare.compare_args import CompareArgs
        assert CompareArgs().data_filter is None

    def test_can_assign_size_filter(self):
        from compare.compare_args import CompareArgs
        args = CompareArgs()
        args.data_filter = SizeFilter(min_size=(512, 512))
        assert args.data_filter.is_active() is True

    def test_clone_copies_data_filter(self):
        from compare.compare_args import CompareArgs
        args = CompareArgs()
        args.data_filter = SizeFilter(exact_size=(512, 512))
        clone = args.clone()
        assert clone.data_filter is not None
        assert clone.data_filter.exact_size == (512, 512)


# ===========================================================================
# CompareManager._normalize_score
# ===========================================================================

class TestNormalizeScore:
    from utils.constants import CompareMode

    @pytest.fixture(autouse=True)
    def _import(self):
        from compare.compare_manager import CompareManager
        from utils.constants import CompareMode
        self.normalize = CompareManager._normalize_score
        self.Mode = CompareMode

    def test_color_matching_zero_diff_is_one(self):
        assert self.normalize(self.Mode.COLOR_MATCHING, 0.0) == pytest.approx(1.0)

    def test_color_matching_cutoff_is_zero(self):
        from compare.compare_colors import CompareColors
        assert self.normalize(self.Mode.COLOR_MATCHING,
                              float(CompareColors.THRESHHOLD_GROUP_CUTOFF)) == pytest.approx(0.0)

    def test_color_matching_half_cutoff_is_half(self):
        from compare.compare_colors import CompareColors
        half = CompareColors.THRESHHOLD_GROUP_CUTOFF / 2.0
        assert self.normalize(self.Mode.COLOR_MATCHING, half) == pytest.approx(0.5)

    def test_color_matching_over_cutoff_clamped_to_zero(self):
        from compare.compare_colors import CompareColors
        over = CompareColors.THRESHHOLD_GROUP_CUTOFF * 2
        assert self.normalize(self.Mode.COLOR_MATCHING, float(over)) == 0.0

    def test_color_matching_negative_diff_clamped_to_one(self):
        assert self.normalize(self.Mode.COLOR_MATCHING, -100.0) == 1.0

    def test_clip_embedding_passthrough(self):
        assert self.normalize(self.Mode.CLIP_EMBEDDING, 0.87) == pytest.approx(0.87)

    def test_size_mode_passthrough(self):
        assert self.normalize(self.Mode.SIZE, 0.5) == pytest.approx(0.5)

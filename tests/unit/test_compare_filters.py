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


# ===========================================================================
# ClassifierFilter
# ===========================================================================

from compare.compare_filters import (  # noqa: E402
    SELECTION_MODEL_STRATEGY,
    ClassifierFilter,
    ClassifierFilterError,
    contains_classifier_filter,
    filter_from_dict,
    filter_to_dict,
    validate_filter,
)


class _StubClassifier:
    """Stands in for ImageClassifierWrapper/AudioClassifierWrapper.

    predictions maps path -> (category, score)."""

    def __init__(self, predictions, model_categories=("photo", "drawing", "nsfw"),
                 positive_groups=(), fail=(), can_run=True):
        self.predictions = predictions
        self.model_categories = list(model_categories)
        self.positive_groups = [list(g) for g in positive_groups]
        self.fail = set(fail)
        self.can_run = can_run
        self.classified = []

    def classify_image(self, path):
        self.classified.append(path)
        if path in self.fail:
            raise RuntimeError("unreadable")
        return self.predictions[path][0]

    def predict_image(self, path):
        category, score = self.predictions[path]
        return {category: score}

    classify_audio = classify_image
    predict_audio = predict_image


PREDICTIONS = {
    "/img/a.png": ("photo", 0.9),
    "/img/b.png": ("drawing", 0.8),
    "/img/c.png": ("photo", 0.4),
}


def _resolver(stub):
    return lambda f: stub


class TestClassifierFilterIsActive:
    def test_no_model_is_inactive(self):
        assert ClassifierFilter(categories=["photo"]).is_active() is False

    def test_no_categories_is_inactive(self):
        assert ClassifierFilter(classifier_name="m").is_active() is False

    def test_model_and_categories_is_active(self):
        assert ClassifierFilter(classifier_name="m", categories=["photo"]).is_active() is True

    def test_model_strategy_needs_no_categories(self):
        f = ClassifierFilter(classifier_name="m", selection_mode=SELECTION_MODEL_STRATEGY)
        assert f.is_active() is True


class TestApplyClassifierFilter:
    def test_include_keeps_only_selected_categories(self):
        f = ClassifierFilter(classifier_name="m", categories=["photo"])
        result = apply_filter(FILES, f, classifier_resolver=_resolver(_StubClassifier(PREDICTIONS)))
        assert result == ["/img/a.png", "/img/c.png"]

    def test_exclude_drops_selected_categories(self):
        f = ClassifierFilter(classifier_name="m", categories=["photo"], mode="exclude")
        result = apply_filter(FILES, f, classifier_resolver=_resolver(_StubClassifier(PREDICTIONS)))
        assert result == ["/img/b.png"]

    def test_min_confidence_requires_score(self):
        f = ClassifierFilter(classifier_name="m", categories=["photo"], min_confidence=0.5)
        result = apply_filter(FILES, f, classifier_resolver=_resolver(_StubClassifier(PREDICTIONS)))
        assert result == ["/img/a.png"]

    def test_model_strategy_uses_positive_groups(self):
        stub = _StubClassifier(PREDICTIONS, positive_groups=[["drawing", "nsfw"]])
        f = ClassifierFilter(classifier_name="m", selection_mode=SELECTION_MODEL_STRATEGY)
        assert apply_filter(FILES, f, classifier_resolver=_resolver(stub)) == ["/img/b.png"]

    def test_model_strategy_without_positive_groups_raises(self):
        f = ClassifierFilter(classifier_name="m", selection_mode=SELECTION_MODEL_STRATEGY)
        with pytest.raises(ClassifierFilterError):
            apply_filter(FILES, f, classifier_resolver=_resolver(_StubClassifier(PREDICTIONS)))

    def test_unknown_category_raises(self):
        f = ClassifierFilter(classifier_name="m", categories=["landscape"])
        with pytest.raises(ClassifierFilterError):
            apply_filter(FILES, f, classifier_resolver=_resolver(_StubClassifier(PREDICTIONS)))

    def test_unclassifiable_file_dropped_by_include_kept_by_exclude(self):
        stub = _StubClassifier(PREDICTIONS, fail={"/img/a.png"})
        include = ClassifierFilter(classifier_name="m", categories=["photo"])
        exclude = ClassifierFilter(classifier_name="m", categories=["photo"], mode="exclude")
        assert apply_filter(FILES, include, classifier_resolver=_resolver(stub)) == ["/img/c.png"]
        assert apply_filter(FILES, exclude, classifier_resolver=_resolver(stub)) == [
            "/img/a.png", "/img/b.png"]

    def test_audio_file_not_applicable_to_image_classifier(self):
        stub = _StubClassifier(PREDICTIONS)
        files = FILES + ["/img/song.mp3"]
        include = ClassifierFilter(classifier_name="m", categories=["photo"])
        exclude = ClassifierFilter(classifier_name="m", categories=["photo"], mode="exclude")
        assert "/img/song.mp3" not in apply_filter(files, include, classifier_resolver=_resolver(stub))
        assert "/img/song.mp3" in apply_filter(files, exclude, classifier_resolver=_resolver(stub))
        assert "/img/song.mp3" not in stub.classified

    def test_image_file_not_applicable_to_audio_classifier(self):
        stub = _StubClassifier(PREDICTIONS)
        f = ClassifierFilter(classifier_name="m", domain="audio", categories=["photo"])
        assert apply_filter(FILES, f, classifier_resolver=_resolver(stub)) == []
        assert stub.classified == []

    def test_progress_reported_and_can_cancel(self):
        calls = []

        class _Cancelled(Exception):
            pass

        def progress(done, total):
            calls.append((done, total))
            if done == 1:
                raise _Cancelled()

        f = ClassifierFilter(classifier_name="m", categories=["photo"])
        with pytest.raises(_Cancelled):
            apply_filter(FILES, f, classifier_resolver=_resolver(_StubClassifier(PREDICTIONS)),
                         progress=progress)
        assert calls == [(0, 3), (1, 3)]


class TestClassifierFilterInGroups:
    def test_and_runs_size_filter_before_classifier(self):
        stub = _StubClassifier(PREDICTIONS)
        group = CompareFilterGroup(operator=FilterOperator.AND, filters=[
            ClassifierFilter(classifier_name="m", categories=["photo"]),
            SizeFilter(min_size=(512, 512)),
        ])
        sizes = {"/img/a.png": (1024, 1024), "/img/b.png": (1024, 1024), "/img/c.png": (100, 100)}
        with _size_reader(sizes):
            result = apply_filter(FILES, group, classifier_resolver=_resolver(stub))
        assert result == ["/img/a.png"]
        assert stub.classified == ["/img/a.png", "/img/b.png"]

    def test_not_group_excludes_classifier_matches(self):
        group = CompareFilterGroup(operator=FilterOperator.NOT, filters=[
            ClassifierFilter(classifier_name="m", categories=["drawing"]),
        ])
        result = apply_filter(FILES, group, classifier_resolver=_resolver(_StubClassifier(PREDICTIONS)))
        assert result == ["/img/a.png", "/img/c.png"]

    def test_contains_classifier_filter(self):
        cf = ClassifierFilter(classifier_name="m", categories=["photo"])
        assert contains_classifier_filter(cf) is True
        assert contains_classifier_filter(CompareFilterGroup(filters=[SizeFilter(min_size=(1, 1)), cf])) is True
        assert contains_classifier_filter(SizeFilter(min_size=(1, 1))) is False
        assert contains_classifier_filter(ClassifierFilter(classifier_name="m")) is False
        assert contains_classifier_filter(None) is False


class TestClassifierFilterSerialization:
    def test_round_trip(self):
        f = ClassifierFilter(classifier_name="m", domain="audio", categories=["photo", "nsfw"],
                             mode="exclude", min_confidence=0.3)
        assert filter_from_dict(filter_to_dict(f)) == f

    def test_round_trip_inside_group(self):
        group = CompareFilterGroup(operator=FilterOperator.OR, filters=[
            SizeFilter(min_size=(10, 10)),
            ClassifierFilter(classifier_name="m", selection_mode=SELECTION_MODEL_STRATEGY),
        ])
        assert filter_from_dict(filter_to_dict(group)) == group

    def test_invalid_values_fall_back_to_defaults(self):
        f = filter_from_dict({"type": "classifier", "classifier_name": "m", "domain": "video",
                              "selection_mode": "bogus", "mode": "bogus",
                              "min_confidence": "x", "categories": None})
        assert f == ClassifierFilter(classifier_name="m", categories=[])


class TestValidateFilter:
    def test_no_filter_is_valid(self):
        assert validate_filter(None) == []

    def test_size_only_filter_does_not_resolve_classifiers(self):
        def resolver(f):
            raise AssertionError("must not be called")
        assert validate_filter(SizeFilter(min_size=(1, 1)), classifier_resolver=resolver) == []

    def test_reports_each_failing_classifier(self):
        def resolver(f):
            if f.classifier_name == "missing":
                raise ClassifierFilterError("missing model")
            return _StubClassifier(PREDICTIONS)
        group = CompareFilterGroup(filters=[
            ClassifierFilter(classifier_name="missing", categories=["photo"]),
            ClassifierFilter(classifier_name="ok", categories=["landscape"]),
            ClassifierFilter(classifier_name="ok", categories=["photo"]),
        ])
        errors = validate_filter(group, classifier_resolver=resolver)
        assert len(errors) == 2
        assert errors[0] == "missing model"
        assert "landscape" in errors[1]


class TestResolveClassifier:
    def test_unregistered_model_raises(self, monkeypatch):
        from compare import compare_filters
        manager = MagicMock()
        manager.resolve_registered_model_name.return_value = None
        monkeypatch.setattr(compare_filters, "_classifier_manager", lambda domain: manager)
        with pytest.raises(ClassifierFilterError):
            compare_filters.resolve_classifier(ClassifierFilter(classifier_name="m", categories=["x"]))

    def test_not_runnable_model_raises(self, monkeypatch):
        from compare import compare_filters
        manager = MagicMock()
        manager.resolve_registered_model_name.return_value = "m"
        manager.get_classifier.return_value = _StubClassifier({}, can_run=False)
        monkeypatch.setattr(compare_filters, "_classifier_manager", lambda domain: manager)
        with pytest.raises(ClassifierFilterError):
            compare_filters.resolve_classifier(ClassifierFilter(classifier_name="m", categories=["x"]))

    def test_runnable_model_returned(self, monkeypatch):
        from compare import compare_filters
        stub = _StubClassifier({})
        manager = MagicMock()
        manager.resolve_registered_model_name.return_value = "m"
        manager.get_classifier.return_value = stub
        monkeypatch.setattr(compare_filters, "_classifier_manager", lambda domain: manager)
        assert compare_filters.resolve_classifier(
            ClassifierFilter(classifier_name="m", categories=["x"])) is stub


class TestFilterSignature:
    def test_none_and_inactive_have_no_signature(self):
        from compare.compare_filters import filter_signature
        assert filter_signature(None) is None
        assert filter_signature(SizeFilter()) is None

    def test_equal_filters_share_signature(self):
        from compare.compare_filters import filter_signature
        a = ClassifierFilter(classifier_name="m", categories=["photo"])
        b = ClassifierFilter(classifier_name="m", categories=["photo"])
        assert filter_signature(a) == filter_signature(b)

    def test_different_filters_differ(self):
        from compare.compare_filters import filter_signature
        a = ClassifierFilter(classifier_name="m", categories=["photo"])
        b = ClassifierFilter(classifier_name="m", categories=["photo"], mode="exclude")
        assert filter_signature(a) != filter_signature(b)


class TestClassifierFilterDynamicMedia:
    """Videos/GIFs/PDFs go through the same frame sampling as classifier actions."""

    @pytest.fixture
    def sampled(self, monkeypatch):
        """Treat /vid/* as dynamic media whose frames are /vid/<name>#<n>."""
        import utils.media_utils as mu
        from image.frame_cache import FrameCache
        monkeypatch.setattr(mu, "is_classifier_dynamic_media_path", lambda p: p.startswith("/vid/"))
        calls = []

        def stream(media_path, sample_ratio=0.1, detect_pseudostatic=False, max_samples=None):
            calls.append((media_path, sample_ratio))
            frames = [f"{media_path}#{i}" for i in range(4)]
            return len(frames), iter(frames)

        monkeypatch.setattr(FrameCache, "stream_frame_samples", stream)
        return calls

    def _stub(self, frame_categories):
        return _StubClassifier({p: (c, 1.0) for p, c in frame_categories.items()})

    def test_match_in_later_frames_counts(self, sampled):
        stub = self._stub({"/vid/a.mp4#0": "drawing", "/vid/a.mp4#1": "drawing",
                           "/vid/a.mp4#2": "photo", "/vid/a.mp4#3": "photo"})
        f = ClassifierFilter(classifier_name="m", categories=["photo"], positive_ratio=0.5)
        assert apply_filter(["/vid/a.mp4"], f, classifier_resolver=_resolver(stub)) == ["/vid/a.mp4"]

    def test_positive_ratio_not_reached_excludes(self, sampled):
        stub = self._stub({"/vid/a.mp4#0": "photo", "/vid/a.mp4#1": "drawing",
                           "/vid/a.mp4#2": "drawing", "/vid/a.mp4#3": "drawing"})
        f = ClassifierFilter(classifier_name="m", categories=["photo"], positive_ratio=0.5)
        assert apply_filter(["/vid/a.mp4"], f, classifier_resolver=_resolver(stub)) == []

    def test_sample_ratio_passed_to_frame_cache(self, sampled):
        stub = self._stub({f"/vid/a.mp4#{i}": "photo" for i in range(4)})
        f = ClassifierFilter(classifier_name="m", categories=["photo"], sample_ratio=0.3)
        apply_filter(["/vid/a.mp4"], f, classifier_resolver=_resolver(stub))
        assert sampled == [("/vid/a.mp4", 0.3)]

    def test_audio_domain_does_not_sample(self, sampled):
        stub = self._stub({})
        f = ClassifierFilter(classifier_name="m", domain="audio", categories=["photo"])
        assert apply_filter(["/vid/a.mp4"], f, classifier_resolver=_resolver(stub)) == []
        assert sampled == []

    def test_ratios_round_trip_and_clamp(self):
        f = ClassifierFilter(classifier_name="m", categories=["photo"], sample_ratio=0.25, positive_ratio=0.5)
        assert filter_from_dict(filter_to_dict(f)) == f
        clamped = filter_from_dict({"type": "classifier", "classifier_name": "m", "categories": ["x"],
                                    "sample_ratio": 5, "positive_ratio": "bad"})
        assert clamped.sample_ratio == 1.0
        assert clamped.positive_ratio == 0.1

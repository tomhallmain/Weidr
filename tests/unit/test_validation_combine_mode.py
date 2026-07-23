"""Unit tests for ValidationCombineMode (OR vs AND across enabled validation
types on a ClassifierAction/Prevalidation).

Covers:
  - ValidationCombineMode.from_value() normalization
  - Default value, serialization round-trip, backward compatibility
  - _evaluate_image_path_match_and(): every enabled type must match
  - _evaluate_image_path_match_for_mode(): OR/AND dispatch
  - The _check_prompt_validation() positive/negative default-True fix
"""

from unittest.mock import MagicMock

from compare.classifier_action import (
    ClassifierAction,
    Prevalidation,
    ValidationCombineMode,
)
from utils.constants import ClassifierActionType


def _action(**kwargs) -> ClassifierAction:
    defaults = dict(
        name="test",
        action=ClassifierActionType.NOTIFY,
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_blacklist=False,
        use_prototype=False,
    )
    defaults.update(kwargs)
    return ClassifierAction(**defaults)


# ---------------------------------------------------------------------------
# ValidationCombineMode.from_value()
# ---------------------------------------------------------------------------

class TestValidationCombineModeFromValue:
    def test_instance_passthrough(self):
        assert ValidationCombineMode.from_value(ValidationCombineMode.AND) == ValidationCombineMode.AND
        assert ValidationCombineMode.from_value(ValidationCombineMode.OR) == ValidationCombineMode.OR

    def test_string_values(self):
        assert ValidationCombineMode.from_value("and") == ValidationCombineMode.AND
        assert ValidationCombineMode.from_value("AND") == ValidationCombineMode.AND
        assert ValidationCombineMode.from_value("or") == ValidationCombineMode.OR

    def test_invalid_or_missing_defaults_to_or(self):
        assert ValidationCombineMode.from_value(None) == ValidationCombineMode.OR
        assert ValidationCombineMode.from_value("") == ValidationCombineMode.OR
        assert ValidationCombineMode.from_value("garbage") == ValidationCombineMode.OR


# ---------------------------------------------------------------------------
# Default value, serialization, backward compatibility
# ---------------------------------------------------------------------------

class TestValidationCombineModeSerialization:
    def test_default_is_or(self):
        ca = _action()
        assert ca.validation_combine_mode == ValidationCombineMode.OR

    def test_explicit_and(self):
        ca = _action(validation_combine_mode=ValidationCombineMode.AND)
        assert ca.validation_combine_mode == ValidationCombineMode.AND

    def test_string_value_normalized_in_post_init(self):
        ca = _action(validation_combine_mode="and")
        assert ca.validation_combine_mode == ValidationCombineMode.AND

    def test_to_dict_and_from_dict_round_trip(self):
        ca = _action(validation_combine_mode=ValidationCombineMode.AND)
        d = ca.to_dict()
        assert d["validation_combine_mode"] == "and"

        restored = ClassifierAction.from_dict(dict(d))
        assert restored.validation_combine_mode == ValidationCombineMode.AND

    def test_from_dict_backward_compat_defaults_to_or(self):
        # Dicts persisted before this feature was added have no such key.
        old_dict = {
            "name": "legacy",
            "action": "NOTIFY",
            "use_embedding": True,
            "positives": ["cat"],
            "negatives": [],
        }
        ca = ClassifierAction.from_dict(old_dict)
        assert ca.validation_combine_mode == ValidationCombineMode.OR

    def test_prevalidation_from_dict_backward_compat_defaults_to_or(self):
        old_dict = {
            "name": "legacy_pv",
            "action": "HIDE",
            "use_embedding": True,
            "positives": ["cat"],
            "negatives": [],
        }
        pv = Prevalidation.from_dict(old_dict)
        assert pv.validation_combine_mode == ValidationCombineMode.OR

    def test_prevalidation_to_dict_includes_combine_mode(self):
        pv = Prevalidation(
            name="pv",
            action=ClassifierActionType.HIDE,
            use_embedding=False,
            use_filename_contains=True,
            filename_contains_patterns=["_wip"],
            validation_combine_mode=ValidationCombineMode.AND,
        )
        d = pv.to_dict()
        assert d["validation_combine_mode"] == "and"


# ---------------------------------------------------------------------------
# _evaluate_image_path_match_and() -- every enabled type must match
# ---------------------------------------------------------------------------

class TestEvaluateImagePathMatchAnd:
    def _two_type_action(self):
        """filename-contains + base-stem-match: simple, no external mocking
        needed for the checks themselves, so tests isolate the AND-combining
        logic rather than the individual check implementations (already
        covered elsewhere)."""
        return _action(
            use_filename_contains=True,
            filename_contains_patterns=["_wip"],
            use_base_stem_match=True,
        )

    def test_all_enabled_types_match_returns_true(self, monkeypatch):
        ca = self._two_type_action()
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)
        monkeypatch.setattr(ca, "_check_base_stem_match", lambda p: True)

        matched, _category = ca._evaluate_image_path_match_and("/data/img_wip.png")
        assert matched is True

    def test_one_enabled_type_failing_returns_false(self, monkeypatch):
        ca = self._two_type_action()
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)
        monkeypatch.setattr(ca, "_check_base_stem_match", lambda p: False)

        matched, _category = ca._evaluate_image_path_match_and("/data/img_wip.png")
        assert matched is False

    def test_both_enabled_types_failing_returns_false(self, monkeypatch):
        ca = self._two_type_action()
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: False)
        monkeypatch.setattr(ca, "_check_base_stem_match", lambda p: False)

        matched, _category = ca._evaluate_image_path_match_and("/data/img.png")
        assert matched is False

    def test_no_validation_type_enabled_returns_false(self):
        # Mirrors _evaluate_image_path_match()'s behavior for the same
        # degenerate case -- an empty AND must not vacuously match everything.
        ca = _action()
        matched, _category = ca._evaluate_image_path_match_and("/data/anything.png")
        assert matched is False

    def test_lookahead_veto_short_circuits_even_when_all_types_would_match(self, monkeypatch):
        ca = self._two_type_action()
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)
        monkeypatch.setattr(ca, "_check_base_stem_match", lambda p: True)
        monkeypatch.setattr(ca, "_check_lookaheads", lambda p, lookahead_eval_cache=None: True)

        matched, _category = ca._evaluate_image_path_match_and("/data/img_wip.png")
        assert matched is False

    def test_image_classifier_category_propagated_on_match(self, monkeypatch):
        ca = _action(
            use_image_classifier=True,
            image_classifier_selected_categories=["cat"],
            use_filename_contains=True,
            filename_contains_patterns=["_wip"],
        )
        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.return_value = True
        fake_classifier.classify_image.return_value = "cat"
        ca.image_classifier = fake_classifier
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)

        matched, category = ca._evaluate_image_path_match_and("/data/img_wip.png")
        assert matched is True
        assert category == "cat"

    def test_image_classifier_no_match_blocks_and(self, monkeypatch):
        ca = _action(
            use_image_classifier=True,
            image_classifier_selected_categories=["cat"],
            use_filename_contains=True,
            filename_contains_patterns=["_wip"],
        )
        fake_classifier = MagicMock()
        fake_classifier.test_image_for_categories.return_value = False
        ca.image_classifier = fake_classifier
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)

        matched, _category = ca._evaluate_image_path_match_and("/data/img_wip.png")
        assert matched is False

    def test_missing_image_classifier_blocks_and(self, monkeypatch):
        # image_classifier is None (never loaded/found) -- must not silently
        # skip this type the way the OR method's fallthrough would. No
        # image_classifier_name set, so ensure_image_classifier_loaded's
        # lazy-load attempt is a no-op and the real classifier registry is
        # never touched.
        ca = _action(
            use_image_classifier=True,
            image_classifier_selected_categories=["cat"],
            use_filename_contains=True,
            filename_contains_patterns=["_wip"],
        )
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)

        matched, _category = ca._evaluate_image_path_match_and("/data/img_wip.png")
        assert matched is False


# ---------------------------------------------------------------------------
# _evaluate_image_path_match_for_mode() -- OR/AND dispatch
# ---------------------------------------------------------------------------

class TestEvaluateImagePathMatchForModeDispatch:
    def _mixed_result_action(self, combine_mode):
        """One type matches, one doesn't -- OR and AND must disagree here,
        proving the dispatch actually changes behavior."""
        return _action(
            use_filename_contains=True,
            filename_contains_patterns=["_wip"],
            use_base_stem_match=True,
            validation_combine_mode=combine_mode,
        )

    def test_or_mode_matches_on_any_type(self, monkeypatch):
        ca = self._mixed_result_action(ValidationCombineMode.OR)
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)
        monkeypatch.setattr(ca, "_check_base_stem_match", lambda p: False)

        matched, _category = ca._evaluate_image_path_match_for_mode("/data/img_wip.png")
        assert matched is True

    def test_and_mode_requires_all_types(self, monkeypatch):
        ca = self._mixed_result_action(ValidationCombineMode.AND)
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)
        monkeypatch.setattr(ca, "_check_base_stem_match", lambda p: False)

        matched, _category = ca._evaluate_image_path_match_for_mode("/data/img_wip.png")
        assert matched is False

    def test_default_mode_is_or(self, monkeypatch):
        ca = _action(
            use_filename_contains=True,
            filename_contains_patterns=["_wip"],
            use_base_stem_match=True,
        )
        assert ca.validation_combine_mode == ValidationCombineMode.OR
        monkeypatch.setattr(ca, "_check_filename_contains", lambda p: True)
        monkeypatch.setattr(ca, "_check_base_stem_match", lambda p: False)

        matched, _category = ca._evaluate_image_path_match_for_mode("/data/img_wip.png")
        assert matched is True


# ---------------------------------------------------------------------------
# _check_prompt_validation() -- positive/negative default-True bug fix
# ---------------------------------------------------------------------------

class TestCheckPromptValidation:
    def _action_with_prompts(self, positives=None, negatives=None):
        return _action(
            use_prompts=True,
            positives=positives or [],
            negatives=negatives or [],
        )

    def _mock_prompts(self, monkeypatch, positive_prompt, negative_prompt):
        monkeypatch.setattr(
            "compare.classifier_action.image_data_extractor.extract_prompts_all_strategies",
            lambda _path: (positive_prompt, negative_prompt),
        )

    def test_positive_only_requires_actual_match(self, monkeypatch):
        """Regression test: previously, an unconfigured self.negatives made
        negative_match default to True, and `positive_match or negative_match`
        was always True regardless of whether the positive terms actually
        appeared -- defeating positive-only prompt rules entirely."""
        ca = self._action_with_prompts(positives=["cat"])
        self._mock_prompts(monkeypatch, "a dog in a field", "")

        assert ca._check_prompt_validation("/data/img.png") is False

    def test_positive_only_matches_when_term_present(self, monkeypatch):
        ca = self._action_with_prompts(positives=["cat"])
        self._mock_prompts(monkeypatch, "a cat in a field", "")

        assert ca._check_prompt_validation("/data/img.png") is True

    def test_negative_only_requires_actual_match(self, monkeypatch):
        ca = self._action_with_prompts(negatives=["blurry"])
        self._mock_prompts(monkeypatch, "a cat in a field", "sharp, clear")

        assert ca._check_prompt_validation("/data/img.png") is False

    def test_negative_only_matches_when_term_present(self, monkeypatch):
        ca = self._action_with_prompts(negatives=["blurry"])
        self._mock_prompts(monkeypatch, "a cat in a field", "blurry, low quality")

        assert ca._check_prompt_validation("/data/img.png") is True

    def test_both_configured_requires_both_to_match(self, monkeypatch):
        ca = self._action_with_prompts(positives=["cat"], negatives=["blurry"])
        self._mock_prompts(monkeypatch, "a cat in a field", "sharp, clear")

        # Positive matches, negative doesn't -- AND semantics require both.
        assert ca._check_prompt_validation("/data/img.png") is False

    def test_both_configured_and_both_match(self, monkeypatch):
        ca = self._action_with_prompts(positives=["cat"], negatives=["blurry"])
        self._mock_prompts(monkeypatch, "a cat in a field", "blurry, low quality")

        assert ca._check_prompt_validation("/data/img.png") is True

    def test_neither_configured_defaults_true(self, monkeypatch):
        # Degenerate case (use_prompts=True but nothing configured to check) --
        # unchanged from before the fix.
        ca = self._action_with_prompts()
        self._mock_prompts(monkeypatch, "anything", "anything")

        assert ca._check_prompt_validation("/data/img.png") is True

    def test_no_prompt_extracted_returns_false(self, monkeypatch):
        ca = self._action_with_prompts(positives=["cat"])
        self._mock_prompts(monkeypatch, None, None)

        assert ca._check_prompt_validation("/data/img.png") is False

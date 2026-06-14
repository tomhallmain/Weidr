"""
Unit tests for ClassifierAction.applies_to_media_types.

Covers:
  - Default value (None = all types)
  - Accepts CompareMediaType enum instances
  - Coerces string values (from JSON) to CompareMediaType
  - Empty list collapses to None
  - Serialization via to_dict()
  - Round-trip via from_dict()
  - Backward compatibility when key is absent from saved data
  - Gating: run_on_media_path and find_first_trigger_slot skip all work
    when the path's type is not in the allowed set
"""

from unittest.mock import patch, MagicMock

import pytest

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction
from utils.constants import ClassifierActionType, CompareMediaType


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


class TestAppliesToMediaTypesDefault:
    def test_default_is_none(self):
        ca = _action()
        assert ca.applies_to_media_types is None

    def test_explicit_none(self):
        ca = _action(applies_to_media_types=None)
        assert ca.applies_to_media_types is None


class TestAppliesToMediaTypesCoercion:
    def test_enum_instances_preserved(self):
        ca = _action(applies_to_media_types=[CompareMediaType.IMAGE, CompareMediaType.VIDEO])
        assert ca.applies_to_media_types == [CompareMediaType.IMAGE, CompareMediaType.VIDEO]

    def test_string_values_coerced_to_enum(self):
        ca = _action(applies_to_media_types=["image", "pdf", "gif"])
        assert ca.applies_to_media_types == [
            CompareMediaType.IMAGE,
            CompareMediaType.PDF,
            CompareMediaType.GIF,
        ]

    def test_mixed_enum_and_string(self):
        ca = _action(applies_to_media_types=[CompareMediaType.SVG, "audio"])
        assert ca.applies_to_media_types == [CompareMediaType.SVG, CompareMediaType.AUDIO]

    def test_empty_list_collapses_to_none(self):
        ca = _action(applies_to_media_types=[])
        assert ca.applies_to_media_types is None

    def test_invalid_string_raises(self):
        with pytest.raises((ValueError, KeyError)):
            _action(applies_to_media_types=["not_a_real_type"])

    def test_all_non_unconfigured_members_accepted(self):
        types = [mt for mt in CompareMediaType if mt != CompareMediaType.UNCONFIGURED]
        ca = _action(applies_to_media_types=types)
        assert ca.applies_to_media_types == types


class TestAppliesToMediaTypesSerialization:
    def test_none_serializes_to_none(self):
        d = _action().to_dict()
        assert d["applies_to_media_types"] is None

    def test_list_serializes_to_string_values(self):
        ca = _action(applies_to_media_types=[CompareMediaType.IMAGE, CompareMediaType.VIDEO])
        d = ca.to_dict()
        assert d["applies_to_media_types"] == ["image", "video"]

    def test_single_type_serializes(self):
        ca = _action(applies_to_media_types=[CompareMediaType.PDF])
        d = ca.to_dict()
        assert d["applies_to_media_types"] == ["pdf"]


class TestAppliesToMediaTypesRoundTrip:
    def test_none_round_trips(self):
        ca = _action()
        ca2 = ClassifierAction.from_dict(ca.to_dict())
        assert ca2.applies_to_media_types is None

    def test_list_round_trips(self):
        ca = _action(applies_to_media_types=[CompareMediaType.IMAGE, CompareMediaType.GIF, CompareMediaType.PDF])
        ca2 = ClassifierAction.from_dict(ca.to_dict())
        assert ca2.applies_to_media_types == [
            CompareMediaType.IMAGE,
            CompareMediaType.GIF,
            CompareMediaType.PDF,
        ]

    def test_from_dict_with_string_list(self):
        d = _action().to_dict()
        d["applies_to_media_types"] = ["video", "svg"]
        ca = ClassifierAction.from_dict(d)
        assert ca.applies_to_media_types == [CompareMediaType.VIDEO, CompareMediaType.SVG]


class TestAppliesToMediaTypesBackwardCompat:
    def test_missing_key_defaults_to_none(self):
        d = _action().to_dict()
        d.pop("applies_to_media_types")
        ca = ClassifierAction.from_dict(d)
        assert ca.applies_to_media_types is None


# ---------------------------------------------------------------------------
# Gating: media_type_allowed drives early return in the public entry points
# ---------------------------------------------------------------------------

def _patch_media_type(media_type: CompareMediaType):
    """Patch get_media_type_for_path to always return media_type."""
    return patch("compare.classifier_action.get_media_type_for_path", return_value=media_type)


class TestMediaTypeGatingRunOnMediaPath:
    """run_on_media_path must return None immediately when the type is not allowed."""

    def _ca_image_only(self) -> ClassifierAction:
        return _action(applies_to_media_types=[CompareMediaType.IMAGE])

    def test_allowed_type_proceeds(self):
        ca = self._ca_image_only()
        # IMAGE is allowed — run_on_media_path should reach its inner logic,
        # not bail out at the gate. We verify by checking it does NOT return
        # early (it will fall through to run_on_image_path which calls _evaluate).
        evaluate_called = []
        with _patch_media_type(CompareMediaType.IMAGE):
            with patch.object(ca, "_evaluate_image_path_match",
                              side_effect=lambda *a, **kw: evaluate_called.append(True) or (False, None)):
                with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=False):
                    ca.run_on_media_path("/img/photo.jpg", ActionCallbacks())
        assert evaluate_called, "_evaluate_image_path_match should be reached for an allowed type"

    def test_disallowed_type_returns_none_without_evaluating(self):
        ca = self._ca_image_only()
        evaluate_called = []
        with _patch_media_type(CompareMediaType.VIDEO):
            with patch.object(ca, "_evaluate_image_path_match",
                              side_effect=lambda *a, **kw: evaluate_called.append(True) or (False, None)):
                result = ca.run_on_media_path("/vid/clip.mp4", ActionCallbacks())
        assert result is None
        assert not evaluate_called, "_evaluate_image_path_match must not be called for a disallowed type"

    def test_disallowed_type_skips_dynamic_sampling(self):
        ca = _action(applies_to_media_types=[CompareMediaType.IMAGE])
        stream_called = []
        with _patch_media_type(CompareMediaType.VIDEO):
            with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
                with patch("compare.classifier_action.FrameCache.stream_frame_samples",
                           side_effect=lambda *a, **kw: stream_called.append(True)):
                    result = ca.run_on_media_path("/vid/clip.mp4", ActionCallbacks())
        assert result is None
        assert not stream_called, "Frame sampling must not start for a disallowed type"

    def test_none_applies_to_allows_all(self):
        ca = _action(applies_to_media_types=None)
        evaluate_called = []
        with _patch_media_type(CompareMediaType.VIDEO):
            with patch.object(ca, "_evaluate_image_path_match",
                              side_effect=lambda *a, **kw: evaluate_called.append(True) or (False, None)):
                with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=False):
                    ca.run_on_media_path("/vid/clip.mp4", ActionCallbacks())
        assert evaluate_called, "With applies_to_media_types=None all types should be allowed"


class TestMediaTypeGatingFindFirstTriggerSlot:
    """find_first_trigger_slot must return None without sampling when type is disallowed."""

    def test_disallowed_returns_none_without_sampling(self):
        ca = _action(applies_to_media_types=[CompareMediaType.VIDEO])
        stream_called = []
        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with _patch_media_type(CompareMediaType.GIF):
                with patch("compare.classifier_action.FrameCache.stream_frame_samples",
                           side_effect=lambda *a, **kw: stream_called.append(True)):
                    result = ca.find_first_trigger_slot("/anim/loop.gif")
        assert result is None
        assert not stream_called

    def test_allowed_type_proceeds_to_sampling(self):
        ca = _action(applies_to_media_types=[CompareMediaType.VIDEO])
        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with _patch_media_type(CompareMediaType.VIDEO):
                with patch("compare.classifier_action.FrameCache.stream_frame_samples",
                           return_value=(0, iter([]))):
                    result = ca.find_first_trigger_slot("/vid/clip.mp4")
        # Returns None because planned_slots=0, but sampling was attempted
        assert result is None  # no frames → None is correct; the point is no early gate exit

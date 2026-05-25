"""
Unit tests for enum classes in utils/constants.py.
All tests are pure logic — no I/O, no mocking required.
"""

import pytest

from utils.constants import (
    ActionType,
    ClassifierActionClass,
    ClassifierActionType,
    CompareMediaType,
    CompareMode,
    Direction,
    ProtectedActions,
    SortBy,
)


class TestCompareMode:
    def test_get_by_name(self):
        assert CompareMode.get("CLIP_EMBEDDING") == CompareMode.CLIP_EMBEDDING

    def test_get_by_instance(self):
        assert CompareMode.get(CompareMode.COLOR_MATCHING) == CompareMode.COLOR_MATCHING

    def test_get_invalid_raises(self):
        with pytest.raises(Exception):
            CompareMode.get("NOT_A_MODE")

    def test_is_embedding_true_for_clip(self):
        assert CompareMode.CLIP_EMBEDDING.is_embedding() is True

    def test_is_embedding_false_for_color(self):
        assert CompareMode.COLOR_MATCHING.is_embedding() is False

    def test_is_embedding_false_for_size(self):
        assert CompareMode.SIZE.is_embedding() is False

    def test_is_embedding_false_for_models(self):
        assert CompareMode.MODELS.is_embedding() is False

    def test_embedding_modes_excludes_non_embedding(self):
        modes = CompareMode.embedding_modes()
        assert CompareMode.COLOR_MATCHING not in modes
        assert CompareMode.SIZE not in modes
        assert CompareMode.MODELS not in modes
        assert CompareMode.CLIP_EMBEDDING in modes

    def test_threshold_vals_non_empty_for_all_modes(self):
        for mode in CompareMode:
            vals = mode.threshold_vals()
            assert len(vals) > 0, f"threshold_vals() empty for {mode}"

    def test_threshold_str_non_empty_for_all_modes(self):
        for mode in CompareMode:
            s = mode.threshold_str()
            assert isinstance(s, str) and len(s) > 0, f"threshold_str() empty for {mode}"

    def test_get_translated_names_length(self):
        names = CompareMode.get_translated_names()
        assert len(names) == len(list(CompareMode))

    def test_get_text_returns_string_for_all(self):
        for mode in CompareMode:
            assert isinstance(mode.get_text(), str)

    def test_text_search_modes_excludes_size_and_color(self):
        modes = CompareMode.text_search_modes()
        assert CompareMode.SIZE not in modes
        assert CompareMode.COLOR_MATCHING not in modes


class TestSortBy:
    def test_get_by_text(self):
        assert SortBy.get("Name") == SortBy.NAME

    def test_get_invalid_raises(self):
        with pytest.raises(Exception):
            SortBy.get("NOT_A_SORT")

    def test_members_length(self):
        assert len(SortBy.members()) == len(list(SortBy))

    def test_get_text_non_empty_for_all(self):
        for s in SortBy:
            assert isinstance(s.get_text(), str) and len(s.get_text()) > 0


class TestCompareMediaType:
    def test_video_is_video(self):
        assert CompareMediaType.VIDEO.is_video() is True

    def test_image_is_not_video(self):
        assert CompareMediaType.IMAGE.is_video() is False

    def test_gif_is_gif(self):
        assert CompareMediaType.GIF.is_gif() is True

    def test_image_is_not_gif(self):
        assert CompareMediaType.IMAGE.is_gif() is False

    def test_audio_is_audio(self):
        assert CompareMediaType.AUDIO.is_audio() is True
        assert CompareMediaType.VIDEO.is_audio() is False

    def test_unconfigured_is_unconfigured(self):
        assert CompareMediaType.UNCONFIGURED.is_unconfigured() is True

    def test_image_is_not_unconfigured(self):
        assert CompareMediaType.IMAGE.is_unconfigured() is False

    def test_video_is_non_video_false(self):
        assert CompareMediaType.VIDEO.is_non_video() is False

    def test_image_is_non_video_true(self):
        assert CompareMediaType.IMAGE.is_non_video() is True

    def test_unconfigured_is_non_video_true(self):
        assert CompareMediaType.UNCONFIGURED.is_non_video() is True

    def test_supports_raster_image_details_true_for_image(self):
        assert CompareMediaType.IMAGE.supports_raster_image_details() is True

    def test_supports_raster_image_details_false_for_video(self):
        assert CompareMediaType.VIDEO.supports_raster_image_details() is False

    def test_supports_raster_image_details_false_for_unconfigured(self):
        assert CompareMediaType.UNCONFIGURED.supports_raster_image_details() is False

    def test_supports_raster_image_details_false_for_audio(self):
        assert CompareMediaType.AUDIO.supports_raster_image_details() is False

    def test_get_translation_non_empty_for_all(self):
        for media_type in CompareMediaType:
            t = media_type.get_translation()
            assert isinstance(t, str) and len(t) > 0


class TestActionType:
    def test_get_translation_non_empty_for_all(self):
        for action in ActionType:
            t = action.get_translation()
            assert isinstance(t, str) and len(t) > 0

    def test_all_members_present(self):
        names = {a.name for a in ActionType}
        assert "MOVE_FILE" in names
        assert "COPY_FILE" in names
        assert "REMOVE_FILE" in names
        assert "SYSTEM" in names


class TestClassifierActionClass:
    def test_from_key_classifier_action(self):
        assert ClassifierActionClass.from_key("classifier_action") == ClassifierActionClass.CLASSIFIER_ACTION

    def test_from_key_prevalidation(self):
        assert ClassifierActionClass.from_key("prevalidation") == ClassifierActionClass.PREVALIDATION

    def test_from_key_case_insensitive(self):
        assert ClassifierActionClass.from_key("CLASSIFIER_ACTION") == ClassifierActionClass.CLASSIFIER_ACTION

    def test_from_key_instance_passthrough(self):
        assert ClassifierActionClass.from_key(ClassifierActionClass.PREVALIDATION) == ClassifierActionClass.PREVALIDATION

    def test_from_key_invalid_raises(self):
        with pytest.raises(ValueError):
            ClassifierActionClass.from_key("not_a_class")

    def test_from_display_value_roundtrip(self):
        for member in ClassifierActionClass:
            assert ClassifierActionClass.from_display_value(member.get_display_value()) == member

    def test_from_display_value_invalid_raises(self):
        with pytest.raises(ValueError):
            ClassifierActionClass.from_display_value("Gibberish")

    def test_from_display_value_instance_passthrough(self):
        assert ClassifierActionClass.from_display_value(ClassifierActionClass.CLASSIFIER_ACTION) == ClassifierActionClass.CLASSIFIER_ACTION


class TestClassifierActionType:
    def test_get_action_by_name(self):
        assert ClassifierActionType.get_action("SKIP") == ClassifierActionType.SKIP

    def test_get_action_case_insensitive(self):
        assert ClassifierActionType.get_action("skip") == ClassifierActionType.SKIP

    def test_get_action_invalid_raises(self):
        with pytest.raises(Exception):
            ClassifierActionType.get_action("FLYING")

    def test_is_cache_type_true(self):
        assert ClassifierActionType.HIDE.is_cache_type() is True
        assert ClassifierActionType.NOTIFY.is_cache_type() is True
        assert ClassifierActionType.SKIP.is_cache_type() is True
        assert ClassifierActionType.ADD_MARK.is_cache_type() is True

    def test_is_cache_type_false(self):
        assert ClassifierActionType.MOVE.is_cache_type() is False
        assert ClassifierActionType.COPY.is_cache_type() is False
        assert ClassifierActionType.DELETE.is_cache_type() is False

    def test_get_translation_non_empty_for_all(self):
        for action in ClassifierActionType:
            t = action.get_translation()
            assert isinstance(t, str) and len(t) > 0


class TestDirection:
    def test_forward_correction(self):
        assert Direction.FORWARD.get_correction(backward_value=0) == -1

    def test_backward_correction(self):
        assert Direction.BACKWARD.get_correction(backward_value=5) == 5


class TestProtectedActions:
    def test_get_action_valid(self):
        result = ProtectedActions.get_action("start_application")
        assert result == ProtectedActions.OPEN_APPLICATION

    def test_get_action_invalid_returns_none(self):
        assert ProtectedActions.get_action("not_a_real_action") is None

    def test_get_description_non_empty_for_all(self):
        for action in ProtectedActions:
            desc = action.get_description()
            assert isinstance(desc, str) and len(desc) > 0

"""
Unit tests for utils.constants.HfHubModelTask (formerly HfHubVisualMediaTask,
renamed+expanded to cover audio-classification alongside the existing visual tasks;
see the Model Manager's Task dropdown in ui/compare/hf_model_manager_window_qt.py).
"""

import pytest

from utils.constants import HfHubModelTask


class TestHfHubModelTaskMembers:
    def test_audio_classification_present(self):
        assert HfHubModelTask.AUDIO_CLASSIFICATION.value == "audio-classification"

    def test_zero_shot_audio_classification_present(self):
        assert HfHubModelTask.ZERO_SHOT_AUDIO_CLASSIFICATION.value == "zero-shot-audio-classification"

    def test_image_classification_unchanged(self):
        # Confirms the rename/expansion didn't disturb pre-existing values.
        assert HfHubModelTask.IMAGE_CLASSIFICATION.value == "image-classification"

    def test_all_tasks_is_empty_string(self):
        assert HfHubModelTask.ALL_TASKS.value == ""


class TestApiValues:
    def test_excludes_all_tasks_sentinel(self):
        assert "" not in HfHubModelTask.api_values()

    def test_includes_audio_classification(self):
        assert "audio-classification" in HfHubModelTask.api_values()

    def test_count_matches_non_sentinel_members(self):
        assert len(HfHubModelTask.api_values()) == len(HfHubModelTask) - 1


class TestDisplay:
    def test_every_member_has_a_display_string(self):
        for member in HfHubModelTask:
            text = member.display()
            assert isinstance(text, str) and len(text) > 0

    def test_display_values_matches_member_count(self):
        assert len(HfHubModelTask.display_values()) == len(HfHubModelTask)


class TestGet:
    def test_get_by_value(self):
        assert HfHubModelTask.get("audio-classification") is HfHubModelTask.AUDIO_CLASSIFICATION

    def test_get_by_member_name(self):
        assert HfHubModelTask.get("AUDIO_CLASSIFICATION") is HfHubModelTask.AUDIO_CLASSIFICATION

    def test_get_by_display_string(self):
        assert HfHubModelTask.get(HfHubModelTask.AUDIO_CLASSIFICATION.display()) is HfHubModelTask.AUDIO_CLASSIFICATION

    def test_get_invalid_raises(self):
        with pytest.raises(Exception, match="Not a valid HF Hub model task"):
            HfHubModelTask.get("not-a-real-task")

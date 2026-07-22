"""
Unit tests for image.suggested_classifier_models -- specifically the
classifier_type=image/audio branching added alongside audio classifier support
(the pre-existing image-only behavior is exercised implicitly by every "image"
case below, since it's the default and unchanged in shape).
"""

import pytest

from image.suggested_classifier_models import (
    SuggestedClassifierModel,
    _parse_entry,
    _VALID_CLASSIFIER_TYPES,
)


def _image_raw(**overrides):
    d = {
        "display_name": "Test Image Classifier",
        "model_name": "test_image_clf",
        "hf_repo_id": "org/image-model",
        "hf_selected_filename": "model.safetensors",
        "model_categories": ["a", "b"],
    }
    d.update(overrides)
    return d


def _audio_raw(**overrides):
    d = {
        "display_name": "Test Audio Classifier",
        "model_name": "test_audio_clf",
        "hf_repo_id": "org/audio-model",
        "model_categories": ["safe", "unsafe"],
        "classifier_type": "audio",
    }
    d.update(overrides)
    return d


class TestParseEntryClassifierType:
    def test_default_classifier_type_is_image(self):
        m = _parse_entry(_image_raw())
        assert m.classifier_type == "image"

    def test_explicit_image_type(self):
        m = _parse_entry(_image_raw(classifier_type="image"))
        assert m.classifier_type == "image"

    def test_explicit_audio_type(self):
        m = _parse_entry(_audio_raw())
        assert m.classifier_type == "audio"

    def test_invalid_classifier_type_raises(self):
        with pytest.raises(ValueError, match="classifier_type"):
            _parse_entry(_audio_raw(classifier_type="video"))

    def test_valid_types_constant(self):
        assert _VALID_CLASSIFIER_TYPES == {"image", "audio"}

    def test_image_missing_hf_selected_filename_raises(self):
        raw = _image_raw()
        del raw["hf_selected_filename"]
        with pytest.raises(ValueError, match="hf_selected_filename"):
            _parse_entry(raw)

    def test_audio_does_not_require_hf_selected_filename(self):
        m = _parse_entry(_audio_raw())
        assert m.hf_selected_filename == ""


class TestParseEntryAudioFields:
    def test_sample_rate_parsed(self):
        m = _parse_entry(_audio_raw(sample_rate=22050))
        assert m.sample_rate == 22050

    def test_sample_rate_defaults_to_none(self):
        m = _parse_entry(_audio_raw())
        assert m.sample_rate is None

    def test_max_duration_seconds_parsed(self):
        m = _parse_entry(_audio_raw(max_duration_seconds=6.5))
        assert m.max_duration_seconds == 6.5


class TestToModelDetailsAudio:
    def test_shape_matches_audio_classifier_model_config(self):
        m = _parse_entry(_audio_raw(sample_rate=22050, max_duration_seconds=8.0))
        details = m.to_model_details("org/audio-model")
        assert details["model_name"] == "test_audio_clf"
        assert details["model_location"] == "org/audio-model"
        assert details["model_categories"] == ["safe", "unsafe"]
        assert details["hf_repo_id"] == "org/audio-model"
        assert details["sample_rate"] == 22050
        assert details["max_duration_seconds"] == 8.0
        # Image-only fields must not leak into the audio-shaped dict.
        assert "backend" not in details
        assert "input_shape" not in details
        assert "hf_selected_filename" not in details

    def test_omits_unset_sample_rate_and_duration(self):
        m = _parse_entry(_audio_raw())
        details = m.to_model_details("org/audio-model")
        assert "sample_rate" not in details
        assert "max_duration_seconds" not in details

    def test_includes_positive_groups_and_severity(self):
        raw = _audio_raw(
            model_categories=["safe", "mild", "explicit"],
            positive_groups=[["mild", "explicit"]],
            severity_order=["explicit", "mild"],
        )
        m = _parse_entry(raw)
        details = m.to_model_details("org/audio-model")
        assert details["positive_groups"] == [["mild", "explicit"]]
        assert details["severity_order"] == ["explicit", "mild"]

    def test_builds_a_valid_audio_classifier_model_config(self):
        """End-to-end: the dict this produces must actually satisfy
        AudioClassifierModelConfig.from_dict (what the Model Manager window calls it
        with) -- this is the contract that matters, not just the dict's shape."""
        from image.audio_classifier_model_config import AudioClassifierModelConfig

        m = _parse_entry(_audio_raw(sample_rate=22050))
        details = m.to_model_details("org/audio-model")
        cfg = AudioClassifierModelConfig.from_dict(details)
        assert cfg.model_name == "test_audio_clf"
        assert cfg.sample_rate == 22050


class TestToModelDetailsImage:
    def test_shape_unchanged_from_before_audio_support(self):
        m = _parse_entry(_image_raw(backend="pytorch", input_shape=[224, 224]))
        details = m.to_model_details("/local/path/model.safetensors")
        assert details["model_name"] == "test_image_clf"
        assert details["model_location"] == "/local/path/model.safetensors"
        assert details["backend"] == "pytorch"
        assert details["input_shape"] == [224, 224]
        assert details["hf_selected_filename"] == "model.safetensors"
        # Audio-only fields must not leak into the image-shaped dict.
        assert "sample_rate" not in details
        assert "max_duration_seconds" not in details

    def test_builds_a_valid_image_classifier_model_config(self):
        from image.image_classifier_model_config import ImageClassifierModelConfig

        m = _parse_entry(_image_raw())
        details = m.to_model_details("/local/path/model.safetensors")
        cfg = ImageClassifierModelConfig.from_dict(details)
        assert cfg.model_name == "test_image_clf"


class TestExistingConfiguredEntriesStillParse:
    """Guards against the classifier_type addition breaking the real,
    already-shipped configs/suggested_classifier_models.json entries -- the
    original three predate the classifier_type field entirely (must default to
    "image"); a real audio entry (AST/AudioSet, 527 categories) was added
    alongside them once verified against the model's actual config.json."""

    def test_real_file_parses_without_errors(self):
        from image.suggested_classifier_models import SUGGESTED_CLASSIFIER_MODELS

        assert len(SUGGESTED_CLASSIFIER_MODELS) >= 4
        for m in SUGGESTED_CLASSIFIER_MODELS:
            assert isinstance(m, SuggestedClassifierModel)
            assert m.classifier_type in ("image", "audio")

    def test_original_three_entries_are_image_type(self):
        from image.suggested_classifier_models import SUGGESTED_CLASSIFIER_MODELS

        original_names = {"image_orientation_detection", "nsfw_detection", "coherence_detection"}
        for m in SUGGESTED_CLASSIFIER_MODELS:
            if m.model_name in original_names:
                assert m.classifier_type == "image"

    def test_audioset_entry_has_527_unique_categories(self):
        from image.suggested_classifier_models import SUGGESTED_CLASSIFIER_MODELS

        audio_entries = [m for m in SUGGESTED_CLASSIFIER_MODELS if m.classifier_type == "audio"]
        assert len(audio_entries) >= 1
        audioset = next(m for m in audio_entries if "audioset" in m.model_name)
        assert len(audioset.model_categories) == 527
        assert len(set(audioset.model_categories)) == 527  # no duplicates

    def test_audioset_entry_builds_a_valid_audio_classifier_model_config(self):
        from image.audio_classifier_model_config import AudioClassifierModelConfig
        from image.suggested_classifier_models import SUGGESTED_CLASSIFIER_MODELS

        audioset = next(
            m for m in SUGGESTED_CLASSIFIER_MODELS
            if m.classifier_type == "audio" and "audioset" in m.model_name
        )
        details = audioset.to_model_details(audioset.hf_repo_id)
        cfg = AudioClassifierModelConfig.from_dict(details)
        assert len(cfg.model_categories) == 527

"""
Unit tests for image.audio_classifier_model_config.AudioClassifierModelConfig.

Pure dataclass validation/serialization -- no torch/transformers required.
"""

import pytest

from image.audio_classifier_model_config import AudioClassifierModelConfig


def _base_dict(**overrides):
    d = {
        "model_name": "nsfw_audio_v1",
        "model_location": "org/nsfw-audio-model",
        "model_categories": ["safe", "explicit"],
    }
    d.update(overrides)
    return d


class TestFromDictRequiredFields:
    def test_missing_model_name_raises(self):
        d = _base_dict()
        del d["model_name"]
        with pytest.raises(ValueError, match="Missing required"):
            AudioClassifierModelConfig.from_dict(d)

    def test_missing_model_location_raises(self):
        d = _base_dict()
        del d["model_location"]
        with pytest.raises(ValueError, match="Missing required"):
            AudioClassifierModelConfig.from_dict(d)

    def test_missing_model_categories_raises(self):
        d = _base_dict()
        del d["model_categories"]
        with pytest.raises(ValueError, match="Missing required"):
            AudioClassifierModelConfig.from_dict(d)

    def test_empty_model_categories_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            AudioClassifierModelConfig.from_dict(_base_dict(model_categories=[]))

    def test_blank_model_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            AudioClassifierModelConfig.from_dict(_base_dict(model_name="   "))

    def test_duplicate_categories_raise(self):
        with pytest.raises(ValueError, match="duplicates"):
            AudioClassifierModelConfig.from_dict(_base_dict(model_categories=["safe", "safe"]))


class TestFromDictDefaults:
    def test_default_sample_rate(self):
        c = AudioClassifierModelConfig.from_dict(_base_dict())
        assert c.sample_rate == 16000

    def test_default_max_duration(self):
        c = AudioClassifierModelConfig.from_dict(_base_dict())
        assert c.max_duration_seconds == 10.0

    def test_custom_sample_rate(self):
        c = AudioClassifierModelConfig.from_dict(_base_dict(sample_rate=44100))
        assert c.sample_rate == 44100

    def test_zero_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioClassifierModelConfig.from_dict(_base_dict(sample_rate=0))

    def test_negative_max_duration_raises(self):
        with pytest.raises(ValueError, match="max_duration_seconds must be positive"):
            AudioClassifierModelConfig.from_dict(_base_dict(max_duration_seconds=-1))


class TestCategoryReferenceValidation:
    def test_positive_groups_unknown_category_raises(self):
        d = _base_dict(positive_groups=[["safe", "made_up"]])
        with pytest.raises(ValueError, match="positive_groups.*made_up"):
            AudioClassifierModelConfig.from_dict(d)

    def test_neutral_categories_unknown_category_raises(self):
        d = _base_dict(neutral_categories=["made_up"])
        with pytest.raises(ValueError, match="neutral_categories.*made_up"):
            AudioClassifierModelConfig.from_dict(d)

    def test_severity_order_unknown_category_raises(self):
        d = _base_dict(severity_order=["made_up"])
        with pytest.raises(ValueError, match="severity_order.*made_up"):
            AudioClassifierModelConfig.from_dict(d)

    def test_valid_positive_groups_pass(self):
        d = _base_dict(
            model_categories=["safe", "mild", "explicit"],
            positive_groups=[["mild", "explicit"]],
        )
        c = AudioClassifierModelConfig.from_dict(d)
        assert c.positive_groups == [["mild", "explicit"]]


class TestRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        d = _base_dict(
            positive_groups=[["explicit"]],
            neutral_categories=["safe"],
            severity_order=["explicit"],
            sample_rate=22050,
            max_duration_seconds=6.0,
            hf_repo_id="org/nsfw-audio-model",
        )
        c = AudioClassifierModelConfig.from_dict(d)
        c2 = AudioClassifierModelConfig.from_dict(c.to_dict())
        assert c2 == c

    def test_to_dict_omits_empty_optional_fields(self):
        c = AudioClassifierModelConfig.from_dict(_base_dict())
        out = c.to_dict()
        assert "positive_groups" not in out
        assert "neutral_categories" not in out
        assert "severity_order" not in out
        assert "hf_repo_id" not in out

    def test_unknown_keys_do_not_raise(self):
        d = _base_dict(some_future_field="x")
        c = AudioClassifierModelConfig.from_dict(d)
        assert c.model_name == "nsfw_audio_v1"

"""
Unit tests for the object detection classifier backend (image/image_classifier.py):
BackendType.DETECTION parsing, category-to-label resolution, detections-to-scores
aggregation, the box-area and score filters, and the ImageClassifierWrapper path.

The transformers processor and model are replaced with stubs through
DetectionImageClassifier._load_components and _forward, so neither
transformers, torch nor a real model file is needed.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from image.image_classifier import (
    BackendType,
    DetectionImageClassifier,
    ImageClassifierWrapper,
    accepted_model_kwargs,
    model_kwargs_for_backend_change,
)
from image.image_classifier_model_config import ImageClassifierModelConfig

_ID2LABEL = {0: "person", 1: "bicycle", 2: "dog", 3: "cat", 4: "traffic_light"}


class _StubProcessor:
    size = {"height": 640, "width": 640}

    def __init__(self, detections):
        # (score, label_id, (x0, y0, x1, y1)) in pixels of the 100x50 test image
        self.detections = detections
        self.post_process_calls = []

    def __call__(self, images=None, return_tensors=None):
        return {"pixel_values": "pixels"}

    def post_process_object_detection(self, outputs, threshold=0.5, target_sizes=None):
        self.post_process_calls.append({"threshold": threshold, "target_sizes": target_sizes})
        kept = [d for d in self.detections if d[0] >= threshold]
        return [{
            "scores": [d[0] for d in kept],
            "labels": [d[1] for d in kept],
            "boxes": [list(d[2]) for d in kept],
        }]


class _StubModel:
    def __init__(self):
        self.config = type("Config", (), {"id2label": {str(k): v for k, v in _ID2LABEL.items()}})()

    def __call__(self, **inputs):
        return "outputs"


@pytest.fixture
def detections(monkeypatch):
    """Mutable detection list the stub processor returns; patches the loader and forward pass."""
    found: list = []
    state = {}

    def load(self, model_root):
        state["processor"] = _StubProcessor(found)
        state["model_root"] = model_root
        return state["processor"], _StubModel()

    monkeypatch.setattr(DetectionImageClassifier, "_load_components", load)
    monkeypatch.setattr(DetectionImageClassifier, "_forward", lambda self, inputs: self.model(**inputs))
    return found, state


@pytest.fixture
def model_dir(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"stub")
    (root / "config.json").write_text(json.dumps({"id2label": {str(k): v for k, v in _ID2LABEL.items()}}))
    return root


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (100, 50)).save(path)
    return str(path)


def _classifier(model_dir, categories=("person", "no person"), **kwargs):
    kwargs.setdefault("background_category", "no person")
    return DetectionImageClassifier(
        str(model_dir / "model.safetensors"), list(categories), device="cpu", **kwargs)


class TestBackendTypeParseDetection:
    @pytest.mark.parametrize("value", ["detection", " Detection ", "object_detection", "object-detection"])
    def test_parses_aliases(self, value):
        assert BackendType.parse(value) == BackendType.DETECTION

    def test_offered_as_config_value(self):
        assert "detection" in BackendType.config_values()

    def test_safetensors_is_not_auto_detected_as_detection(self, tmp_path):
        model_path = tmp_path / "model.safetensors"
        model_path.write_bytes(b"stub")
        wrapper = ImageClassifierWrapper(ImageClassifierModelConfig(
            model_name="m", model_location=str(model_path), model_categories=["a", "b"]))
        assert wrapper.backend == BackendType.PYTORCH


class TestCategoryResolution:
    def test_loads_from_weights_file_directory(self, detections, model_dir):
        _found, state = detections
        clf = _classifier(model_dir)
        assert clf.is_loaded
        assert state["model_root"] == str(model_dir)

    def test_hf_pretrained_path_overrides_directory(self, detections, model_dir, tmp_path):
        _found, state = detections
        other = tmp_path / "other"
        other.mkdir()
        _classifier(model_dir, hf_pretrained_path=str(other))
        assert state["model_root"] == str(other)

    def test_label_matching_ignores_case_and_separators(self, detections, model_dir):
        clf = _classifier(model_dir, categories=("Person", "Traffic Light", "none"), background_category="none")
        assert clf.is_loaded

    def test_unknown_label_fails_to_load(self, detections, model_dir):
        clf = _classifier(model_dir, categories=("persn", "no person"))
        assert clf.is_loaded is False

    def test_background_outside_categories_fails_to_load(self, detections, model_dir):
        clf = _classifier(model_dir, categories=("person", "empty"), background_category="no person")
        assert clf.is_loaded is False

    def test_aliases_map_one_category_to_several_labels(self, detections, model_dir, image_path):
        found, _state = detections
        found.extend([(0.6, 2, (0, 0, 50, 50)), (0.8, 3, (0, 0, 50, 50))])
        clf = _classifier(model_dir, categories=("animal", "no animal"), background_category="no animal",
                          label_aliases={"animal": ["dog", "cat"]})
        assert np.allclose(clf.predict_image(image_path), [[0.8, 0.2]])

    def test_input_shape_comes_from_processor(self, detections, model_dir):
        assert _classifier(model_dir).input_shape == (640, 640)


class TestScoring:
    def test_target_scores_its_best_detection(self, detections, model_dir, image_path):
        found, _state = detections
        found.extend([(0.4, 0, (0, 0, 50, 50)), (0.9, 0, (0, 0, 100, 50)), (0.95, 1, (0, 0, 10, 10))])
        assert np.allclose(_classifier(model_dir).predict_image(image_path), [[0.9, 0.1]])

    def test_no_detection_scores_background_one(self, detections, model_dir, image_path):
        assert np.allclose(_classifier(model_dir).predict_image(image_path), [[0.0, 1.0]])

    def test_score_threshold_and_image_size_are_forwarded(self, detections, model_dir, image_path):
        _found, state = detections
        _classifier(model_dir, score_threshold=0.45).predict_image(image_path)
        assert state["processor"].post_process_calls == [{"threshold": 0.45, "target_sizes": [(50, 100)]}]

    def test_small_boxes_are_dropped(self, detections, model_dir, image_path):
        found, _state = detections
        # 10x10 box on a 100x50 image covers 2% of it
        found.append((0.9, 0, (0, 0, 10, 10)))
        assert np.allclose(_classifier(model_dir, min_box_area_ratio=0.05).predict_image(image_path), [[0.0, 1.0]])
        assert np.allclose(_classifier(model_dir, min_box_area_ratio=0.01).predict_image(image_path), [[0.9, 0.1]])

    def test_multiple_targets_each_score_and_background_uses_best(self, detections, model_dir, image_path):
        found, _state = detections
        found.extend([(0.7, 0, (0, 0, 50, 50)), (0.8, 2, (0, 0, 50, 50))])
        clf = _classifier(model_dir, categories=("person", "dog", "none"), background_category="none")
        assert np.allclose(clf.predict_image(image_path), [[0.7, 0.8, 0.2]])

    def test_without_background_only_targets_are_scored(self, detections, model_dir, image_path):
        found, _state = detections
        found.append((0.6, 0, (0, 0, 50, 50)))
        clf = _classifier(model_dir, categories=("person",), background_category=None)
        assert np.allclose(clf.predict_image(image_path), [[0.6]])

    def test_detect_reports_filtered_detections(self, detections, model_dir, image_path):
        found, _state = detections
        found.extend([(0.9, 0, (10, 0, 60, 50)), (0.2, 0, (0, 0, 100, 50))])
        result = _classifier(model_dir).detect(image_path)
        assert result == [{
            "label": "person", "label_id": 0, "score": 0.9,
            "box": (10.0, 0.0, 60.0, 50.0), "area_ratio": pytest.approx(0.5),
        }]


def test_categories_for_label_id(detections, model_dir):
    clf = _classifier(model_dir, categories=("animal", "dog", "none"), background_category="none",
                      label_aliases={"animal": ["dog", "cat"]})
    assert clf.categories_for_label_id(2) == ["animal", "dog"]
    assert clf.categories_for_label_id(3) == ["animal"]
    assert clf.categories_for_label_id(0) == []


def test_last_detections_are_kept_per_image(detections, model_dir, image_path, tmp_path):
    found, _state = detections
    found.append((0.9, 0, (0, 0, 100, 50)))
    clf = _classifier(model_dir)
    assert clf.last_detections(image_path) is None
    clf.predict_image(image_path)
    assert [d["label"] for d in clf.last_detections(image_path)] == ["person"]
    assert clf.last_detections(str(tmp_path / "other.png")) is None


class TestBackendKwargs:
    @pytest.mark.parametrize("location, expected", [
        ("m.h5", BackendType.HDF5),
        ("m.SAFETENSORS", BackendType.PYTORCH),
        ("m.bin", BackendType.PYTORCH),
        ("m.onnx", BackendType.ONNX),
        ("m.tflite", BackendType.TFLITE),
        ("m.xyz", None),
    ])
    def test_from_model_location(self, location, expected):
        assert BackendType.from_model_location(location) == expected

    def test_accepted_model_kwargs_follow_constructor_arguments(self):
        assert "channels_first" in accepted_model_kwargs(BackendType.ONNX)
        assert "channels_first" not in accepted_model_kwargs(BackendType.PYTORCH)
        assert "label_aliases" in accepted_model_kwargs(BackendType.DETECTION)
        assert "model_categories" not in accepted_model_kwargs(BackendType.DETECTION)
        assert accepted_model_kwargs(BackendType.HDF5) == frozenset()
        assert accepted_model_kwargs(BackendType.OTHER) is None

    def test_unchanged_backend_keeps_everything(self):
        kept, dropped = model_kwargs_for_backend_change(
            {"anything": 1}, "onnx", "m.onnx", "onnx", "m.onnx")
        assert kept == {"anything": 1} and dropped == []

    def test_auto_resolves_from_location(self):
        kept, dropped = model_kwargs_for_backend_change(
            {"channels_first": False}, "auto", "m.onnx", "onnx", "m.onnx")
        assert dropped == []

    def test_changed_backend_drops_rejected_keys(self):
        kept, dropped = model_kwargs_for_backend_change(
            {"channels_first": False, "device": "cpu"}, "onnx", "m.onnx", "pytorch", "m.pt")
        assert kept == {"device": "cpu"}
        assert dropped == ["channels_first"]

    def test_unknown_new_backend_keeps_everything(self):
        kept, dropped = model_kwargs_for_backend_change(
            {"channels_first": False}, "onnx", "m.onnx", "auto", "m.xyz")
        assert kept == {"channels_first": False} and dropped == []


class TestInstallHelpers:
    def test_read_id2label(self, model_dir):
        assert DetectionImageClassifier.read_id2label(str(model_dir))[0] == "person"

    def test_read_id2label_missing_config(self, tmp_path):
        assert DetectionImageClassifier.read_id2label(str(tmp_path)) == {}

    def test_single_non_label_category_is_background(self):
        assert DetectionImageClassifier.split_background_category(
            ["person", "no person"], _ID2LABEL.values()) == ("no person", [])

    def test_all_labels_means_no_background(self):
        assert DetectionImageClassifier.split_background_category(
            ["person", "dog"], _ID2LABEL.values()) == (None, [])

    def test_several_non_labels_are_reported(self):
        assert DetectionImageClassifier.split_background_category(
            ["persn", "no person"], _ID2LABEL.values()) == (None, ["persn", "no person"])


class TestWrapperIntegration:
    def _wrapper(self, model_dir, **model_kwargs):
        model_kwargs.setdefault("background_category", "no person")
        return ImageClassifierWrapper(ImageClassifierModelConfig(
            model_name="detector",
            model_location=str(model_dir / "model.safetensors"),
            model_categories=["person", "no person"],
            backend="detection",
            model_kwargs={"device": "cpu", **model_kwargs},
        ))

    def test_classifies_person_above_half(self, detections, model_dir, image_path):
        found, _state = detections
        found.append((0.85, 0, (0, 0, 100, 50)))
        wrapper = self._wrapper(model_dir)
        assert wrapper.can_run is True
        assert wrapper.classify_image(image_path) == "person"
        assert wrapper.predict_image(image_path) == pytest.approx({"person": 0.85, "no person": 0.15})

    def test_low_confidence_detection_classifies_as_background(self, detections, model_dir, image_path):
        found, _state = detections
        found.append((0.35, 0, (0, 0, 100, 50)))
        assert self._wrapper(model_dir).classify_image(image_path) == "no person"

    def test_invalid_category_leaves_wrapper_unrunnable(self, detections, model_dir):
        wrapper = ImageClassifierWrapper(ImageClassifierModelConfig(
            model_name="detector", model_location=str(model_dir / "model.safetensors"),
            model_categories=["persn", "no person"], backend="detection",
            model_kwargs={"device": "cpu", "background_category": "no person"},
        ))
        assert wrapper.can_run is False

    def test_detection_settings_change_the_prediction_cache_signature(self, detections, model_dir):
        a = self._wrapper(model_dir, score_threshold=0.3)._persisted_prediction_key()
        b = self._wrapper(model_dir, score_threshold=0.5)._persisted_prediction_key()
        assert a[1] != b[1]

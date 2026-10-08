"""
Unit tests for ImageClassifierManager's runtime-wrapper cache
(image/image_classifier_manager.py): is_loaded() only reports runnable
wrappers, a failed wrapper stays cached without being rebuilt on every
lookup, and metadata changes evict wrappers whose config changed.

ImageClassifierWrapper is replaced with a recording stand-in so no model
files or ML runtimes are needed.
"""
from __future__ import annotations

import pytest

import image.image_classifier_manager as manager_module
from image.image_classifier_manager import ImageClassifierManager


class _FakeWrapper:
    """Records each construction; can_run is decided per model_location."""

    built: list = []
    failing_locations: set = set()

    def __init__(self, model_config):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.can_run = model_config.model_location not in type(self).failing_locations
        type(self).built.append(model_config)


@pytest.fixture
def fake_wrapper(monkeypatch):
    wrapper_cls = type("Fake", (_FakeWrapper,), {"built": [], "failing_locations": set()})
    monkeypatch.setattr(manager_module, "ImageClassifierWrapper", wrapper_cls)
    return wrapper_cls


def _details(name="m", location="/models/good.tflite", **extra):
    return {"model_name": name, "model_location": location, "model_categories": ["a", "b"], **extra}


@pytest.fixture
def manager(fake_wrapper):
    mgr = ImageClassifierManager()
    mgr.set_classifier_metadata([_details()])
    return mgr


class TestIsLoaded:
    def test_false_before_first_lookup(self, manager):
        assert manager.is_loaded("m") is False

    def test_true_for_runnable_wrapper(self, manager):
        manager.get_classifier("m")
        assert manager.is_loaded("m") is True

    def test_false_for_wrapper_that_failed_to_initialize(self, manager, fake_wrapper):
        fake_wrapper.failing_locations.add("/models/good.tflite")
        classifier = manager.get_classifier("m")
        assert classifier.can_run is False
        assert manager.is_loaded("m") is False


class TestFailedWrapperCaching:
    def test_failed_wrapper_is_not_rebuilt_on_repeat_lookup(self, manager, fake_wrapper):
        fake_wrapper.failing_locations.add("/models/good.tflite")
        first = manager.get_classifier("m")
        second = manager.get_classifier("m")
        assert first is second
        assert len(fake_wrapper.built) == 1


class TestSetClassifierMetadataEviction:
    def test_unchanged_config_keeps_cached_wrapper(self, manager, fake_wrapper):
        first = manager.get_classifier("m")
        manager.set_classifier_metadata([_details()])
        assert manager.get_classifier("m") is first
        assert len(fake_wrapper.built) == 1

    def test_changed_config_evicts_and_reloads(self, manager, fake_wrapper):
        fake_wrapper.failing_locations.add("/models/.gitattributes")
        manager.set_classifier_metadata([_details(location="/models/.gitattributes")])
        assert manager.get_classifier("m").can_run is False

        manager.set_classifier_metadata([_details(location="/models/good.tflite")])
        assert "m" not in manager.classifiers
        reloaded = manager.get_classifier("m")
        assert reloaded.can_run is True
        assert reloaded.model_config.model_location == "/models/good.tflite"
        assert manager.is_loaded("m") is True

    def test_changed_config_evicts_previously_working_wrapper(self, manager, fake_wrapper):
        manager.get_classifier("m")
        manager.set_classifier_metadata([_details(backend="tflite")])
        assert "m" not in manager.classifiers

    def test_removed_entry_is_evicted(self, manager):
        manager.get_classifier("m")
        manager.set_classifier_metadata([_details(name="other")])
        assert "m" not in manager.classifiers

    def test_other_models_keep_their_cached_wrappers(self, fake_wrapper):
        mgr = ImageClassifierManager()
        mgr.set_classifier_metadata([_details(), _details(name="n", location="/models/n.onnx")])
        kept = mgr.get_classifier("n")
        mgr.get_classifier("m")
        mgr.set_classifier_metadata([_details(backend="tflite"), _details(name="n", location="/models/n.onnx")])
        assert "m" not in mgr.classifiers
        assert mgr.get_classifier("n") is kept


class TestAddClassifierMetadataEviction:
    def test_unchanged_config_keeps_cached_wrapper(self, manager):
        first = manager.get_classifier("m")
        manager.add_classifier_metadata(_details())
        assert manager.get_classifier("m") is first

    def test_changed_config_evicts(self, manager):
        manager.get_classifier("m")
        manager.add_classifier_metadata(_details(location="/models/other.tflite"))
        assert "m" not in manager.classifiers

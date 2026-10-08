"""
Unit tests for the TFLite classifier backend (image/image_classifier.py):
BackendType.TFLITE parsing, extension-based auto-detection in
ImageClassifierWrapper.load_classifier(), and TFLiteImageClassifier's
dtype-driven preprocessing and output handling.

No real .tflite file or TFLite runtime is needed: TFLiteImageClassifier is
exercised against a stub interpreter patched in through
_import_interpreter_class, mirroring how test_image_classifier_onnx_backend.py
avoids the onnxruntime dependency.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from image.classifier_utils import sigmoid as _sigmoid
from image.image_classifier import BackendType, ImageClassifierWrapper, TFLiteImageClassifier
from image.image_classifier_model_config import ImageClassifierModelConfig


class _StubInterpreter:
    """Minimal stand-in for tf.lite.Interpreter with configurable tensor details."""

    input_spec: dict = {}
    output_spec: dict = {}
    output_value: np.ndarray = np.zeros((1, 2), dtype=np.float32)
    load_error: Exception | None = None

    def __init__(self, model_path=None, num_threads=None):
        if type(self).load_error is not None:
            raise type(self).load_error
        self.model_path = model_path
        self.num_threads = num_threads
        self._input = dict(type(self).input_spec)
        self.resized_to = None
        self.last_input = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [dict(self._input)]

    def get_output_details(self):
        return [dict(type(self).output_spec)]

    def resize_tensor_input(self, index, shape):
        self.resized_to = list(shape)
        self._input["shape"] = np.array(shape)
        self._input["shape_signature"] = np.array(shape)

    def set_tensor(self, index, value):
        self.last_input = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        return np.array(type(self).output_value)


def _input_spec(shape, dtype, quantization=(0.0, 0), signature=None):
    return {
        "index": 0,
        "shape": np.array(shape),
        "shape_signature": np.array(signature if signature is not None else shape),
        "dtype": dtype,
        "quantization": quantization,
    }


def _output_spec(shape, dtype, quantization=(0.0, 0)):
    return {"index": 1, "shape": np.array(shape), "dtype": dtype, "quantization": quantization}


@pytest.fixture
def stub_runtime(monkeypatch):
    """Install a fresh _StubInterpreter subclass as the only available runtime."""
    stub = type("Stub", (_StubInterpreter,), {
        "input_spec": _input_spec([1, 4, 6, 3], np.uint8),
        "output_spec": _output_spec([1, 2], np.float32),
        "output_value": np.array([[0.25, 0.75]], dtype=np.float32),
        "load_error": None,
    })
    monkeypatch.setattr(
        TFLiteImageClassifier, "_import_interpreter_class",
        classmethod(lambda cls: (stub, "stub")),
    )
    return stub


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (10, 10), color=(255, 0, 128)).save(path)
    return str(path)


class TestImportInterpreterClass:
    """Mimics tensorflow's layout: the "<pkg>.lite" submodule is an empty package and
    Interpreter is only reachable as an attribute of the top-level module."""

    @pytest.fixture
    def fake_tf(self, monkeypatch):
        import sys
        import types

        class FakeInterpreter:
            pass

        top = types.ModuleType("fake_tf_pkg")
        empty_lite = types.ModuleType("fake_tf_pkg.lite")
        monkeypatch.setitem(sys.modules, "fake_tf_pkg", top)
        monkeypatch.setitem(sys.modules, "fake_tf_pkg.lite", empty_lite)
        top.lite = types.SimpleNamespace(Interpreter=FakeInterpreter)
        return FakeInterpreter

    def test_resolves_attribute_path_on_top_level_module(self, fake_tf, monkeypatch):
        monkeypatch.setattr(TFLiteImageClassifier, "_INTERPRETER_SOURCES", (
            ("weidr_missing_runtime_xyz.interpreter", "Interpreter"),
            ("fake_tf_pkg", "lite.Interpreter"),
        ))
        interpreter_cls, source = TFLiteImageClassifier._import_interpreter_class()
        assert interpreter_cls is fake_tf
        assert source == "fake_tf_pkg.lite.Interpreter"

    def test_importing_the_lite_submodule_does_not_find_interpreter(self, fake_tf, monkeypatch):
        monkeypatch.setattr(TFLiteImageClassifier, "_INTERPRETER_SOURCES", (
            ("fake_tf_pkg.lite", "Interpreter"),
        ))
        assert TFLiteImageClassifier._import_interpreter_class() == (None, None)

    def test_tensorflow_source_uses_attribute_path(self):
        assert ("tensorflow", "lite.Interpreter") in TFLiteImageClassifier._INTERPRETER_SOURCES


class TestBackendTypeParseTflite:
    @pytest.mark.parametrize("value", ["tflite", " TFLite ", "litert", "tf_lite"])
    def test_parses_aliases(self, value):
        assert BackendType.parse(value) == BackendType.TFLITE

    def test_passthrough_enum_value(self):
        assert BackendType.parse(BackendType.TFLITE) == BackendType.TFLITE


class TestTfliteAutoDetection:
    def _make_wrapper(self, tmp_path, filename="model.tflite", backend="auto"):
        model_path = tmp_path / filename
        model_path.write_bytes(b"not a real tflite model, just needs to exist")
        config = ImageClassifierModelConfig(
            model_name="test-tflite",
            model_location=str(model_path),
            model_categories=["negative", "positive"],
            backend=backend,
        )
        return ImageClassifierWrapper(config)

    def test_tflite_extension_is_auto_detected(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        assert wrapper.backend == BackendType.TFLITE

    def test_explicit_tflite_backend_is_respected(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path, filename="model.weird_ext", backend="tflite")
        assert wrapper.backend == BackendType.TFLITE

    def test_invalid_tflite_file_fails_gracefully_without_raising(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        assert wrapper.can_run is False

    def test_missing_runtime_fails_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            TFLiteImageClassifier, "_import_interpreter_class",
            classmethod(lambda cls: (None, None)),
        )
        wrapper = self._make_wrapper(tmp_path)
        assert wrapper.can_run is False

    def test_wrapper_classifies_with_stub_runtime(self, tmp_path, stub_runtime, image_path):
        wrapper = self._make_wrapper(tmp_path)
        assert wrapper.can_run is True
        assert wrapper.classify_image(image_path) == "positive"


class TestTfliteLoading:
    def test_input_shape_inferred_from_nhwc(self, stub_runtime):
        clf = TFLiteImageClassifier("m.tflite")
        assert clf.is_loaded
        assert clf.input_shape == (6, 4)

    def test_conflicting_override_on_fixed_shape_is_ignored(self, stub_runtime):
        clf = TFLiteImageClassifier("m.tflite", input_shape=(32, 32))
        assert clf.input_shape == (6, 4)

    def test_dynamic_shape_is_resized_to_override(self, stub_runtime):
        stub_runtime.input_spec = _input_spec([1, 1, 1, 3], np.float32, signature=[-1, -1, -1, 3])
        clf = TFLiteImageClassifier("m.tflite", input_shape=(8, 5))
        assert clf.input_shape == (8, 5)
        assert clf.interpreter.resized_to == [1, 5, 8, 3]

    def test_dynamic_shape_without_override_uses_default(self, stub_runtime):
        stub_runtime.input_spec = _input_spec([1, 1, 1, 3], np.float32, signature=[-1, -1, -1, 3])
        clf = TFLiteImageClassifier("m.tflite")
        assert clf.input_shape == (224, 224)

    def test_load_error_leaves_classifier_unloaded(self, stub_runtime):
        stub_runtime.load_error = RuntimeError("Encountered unresolved custom op: ethos-u.")
        clf = TFLiteImageClassifier("m.tflite")
        assert clf.is_loaded is False


class TestTflitePreprocessing:
    def test_uint8_input_feeds_raw_pixels(self, stub_runtime, image_path):
        clf = TFLiteImageClassifier("m.tflite")
        arr = clf.preprocess_image(image_path)
        assert arr.dtype == np.uint8
        assert arr.shape == (1, 4, 6, 3)
        assert tuple(arr[0, 0, 0]) == (255, 0, 128)

    def test_int8_input_is_quantized_from_unit_range(self, stub_runtime, image_path):
        stub_runtime.input_spec = _input_spec([1, 4, 6, 3], np.int8, quantization=(1.0 / 255.0, -128))
        clf = TFLiteImageClassifier("m.tflite")
        arr = clf.preprocess_image(image_path)
        assert arr.dtype == np.int8
        assert tuple(arr[0, 0, 0]) == (127, -128, 0)

    def test_int8_input_with_unit_scale_uses_raw_pixel_range(self, stub_runtime, image_path):
        stub_runtime.input_spec = _input_spec([1, 4, 6, 3], np.int8, quantization=(1.0, -128))
        clf = TFLiteImageClassifier("m.tflite")
        arr = clf.preprocess_image(image_path)
        assert tuple(arr[0, 0, 0]) == (127, -128, 0)

    def test_float_input_uses_rescale_and_normalization(self, stub_runtime, image_path):
        stub_runtime.input_spec = _input_spec([1, 4, 6, 3], np.float32)
        clf = TFLiteImageClassifier("m.tflite", normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1])
        arr = clf.preprocess_image(image_path)
        assert arr.dtype == np.float32
        assert np.allclose(arr[0, 0, 0], [1.0, 0.0, 128 / 255.0])

    def test_float_input_without_rescale_keeps_raw_values(self, stub_runtime, image_path):
        stub_runtime.input_spec = _input_spec([1, 4, 6, 3], np.float32)
        clf = TFLiteImageClassifier("m.tflite", rescale=False)
        arr = clf.preprocess_image(image_path)
        assert np.allclose(arr[0, 0, 0], [255.0, 0.0, 128.0])


class TestTflitePredict:
    def _predict(self, clf):
        return clf.predict(np.zeros((1, 4, 6, 3), dtype=np.uint8))

    def test_probabilities_pass_through(self, stub_runtime):
        clf = TFLiteImageClassifier("m.tflite")
        assert np.allclose(self._predict(clf), [[0.25, 0.75]])

    def test_logits_get_softmax(self, stub_runtime):
        stub_runtime.output_value = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)
        clf = TFLiteImageClassifier("m.tflite")
        out = self._predict(clf)
        assert np.isclose(out.sum(), 1.0)
        assert np.argmax(out[0]) == 0

    def test_quantized_output_is_dequantized(self, stub_runtime):
        stub_runtime.output_spec = _output_spec([1, 2], np.uint8, quantization=(1.0 / 255.0, 0))
        stub_runtime.output_value = np.array([[51, 204]], dtype=np.uint8)
        clf = TFLiteImageClassifier("m.tflite")
        assert np.allclose(self._predict(clf), [[0.2, 0.8]])

    def test_single_probability_passes_through(self, stub_runtime):
        stub_runtime.output_spec = _output_spec([1, 1], np.float32)
        stub_runtime.output_value = np.array([[0.9]], dtype=np.float32)
        clf = TFLiteImageClassifier("m.tflite")
        assert np.allclose(self._predict(clf), [[0.9]])

    def test_single_logit_gets_sigmoid(self, stub_runtime):
        stub_runtime.output_spec = _output_spec([1, 1], np.float32)
        stub_runtime.output_value = np.array([[3.0]], dtype=np.float32)
        clf = TFLiteImageClassifier("m.tflite")
        p = float(_sigmoid(np.array(3.0)))
        assert np.allclose(self._predict(clf), [[p]])

    def test_wrapper_expands_single_output_to_binary(self, tmp_path, stub_runtime, image_path):
        stub_runtime.output_spec = _output_spec([1, 1], np.float32)
        stub_runtime.output_value = np.array([[0.9]], dtype=np.float32)
        model_path = tmp_path / "model.tflite"
        model_path.write_bytes(b"stub")
        wrapper = ImageClassifierWrapper(ImageClassifierModelConfig(
            model_name="binary", model_location=str(model_path),
            model_categories=["non-person", "person"],
        ))
        scores = wrapper.predict_image(image_path)
        assert scores == pytest.approx({"non-person": 0.1, "person": 0.9})
        assert wrapper.classify_image(image_path) == "person"


class TestSigmoid:
    def test_midpoint(self):
        assert np.isclose(_sigmoid(np.array(0.0)), 0.5)

    def test_is_numerically_stable_for_large_magnitudes(self):
        out = _sigmoid(np.array([-1000.0, 1000.0]))
        assert not np.any(np.isnan(out))
        assert np.allclose(out, [0.0, 1.0])

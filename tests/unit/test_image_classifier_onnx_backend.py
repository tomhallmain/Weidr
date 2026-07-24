"""
Unit tests for the ONNX Runtime classifier backend (image/image_classifier.py):
BackendType.ONNX parsing, ONNXImageClassifier's pure helper logic (input-shape
inference, softmax fallback), and extension-based auto-detection in
ImageClassifierWrapper.load_classifier().

Tests that would require actually loading a real .onnx file (needing both the
onnxruntime package and a valid model on disk) are avoided in favor of testing
the routing/detection logic and pure numpy helpers directly, mirroring how
test_image_classifier_architecture_loading.py avoids the torch/tensorflow
dependencies where possible.
"""
from __future__ import annotations

import numpy as np
import pytest

from image.image_classifier import BackendType, ONNXImageClassifier, _softmax
from image.image_classifier import ImageClassifierWrapper
from image.image_classifier_model_config import ImageClassifierModelConfig


class TestBackendTypeParseOnnx:
    def test_parses_lowercase(self):
        assert BackendType.parse("onnx") == BackendType.ONNX

    def test_parses_case_insensitive_and_whitespace(self):
        assert BackendType.parse(" ONNX ") == BackendType.ONNX

    def test_passthrough_enum_value(self):
        assert BackendType.parse(BackendType.ONNX) == BackendType.ONNX

    def test_auto_still_returns_none(self):
        assert BackendType.parse("auto") is None

    def test_unrecognized_string_is_other(self):
        assert BackendType.parse("not-a-real-backend") == BackendType.OTHER


class TestInferInputShapeFromDims:
    def test_nchw_static_dims(self):
        shape = ONNXImageClassifier._infer_input_shape_from_dims(
            [1, 3, 224, 224], channels_first=True
        )
        assert shape == (224, 224)

    def test_nhwc_static_dims(self):
        shape = ONNXImageClassifier._infer_input_shape_from_dims(
            [1, 224, 224, 3], channels_first=False
        )
        assert shape == (224, 224)

    def test_non_square_nchw(self):
        shape = ONNXImageClassifier._infer_input_shape_from_dims(
            [1, 3, 256, 320], channels_first=True
        )
        assert shape == (320, 256)

    def test_dynamic_batch_dim_is_still_resolvable(self):
        # Dynamic axes are often represented as a string (symbolic dim name)
        # by onnxruntime -- only the batch dim should ever be dynamic here.
        shape = ONNXImageClassifier._infer_input_shape_from_dims(
            ["batch", 3, 224, 224], channels_first=True
        )
        assert shape == (224, 224)

    def test_dynamic_spatial_dims_cannot_be_resolved(self):
        shape = ONNXImageClassifier._infer_input_shape_from_dims(
            [1, 3, "height", "width"], channels_first=True
        )
        assert shape is None

    def test_non_4d_shape_returns_none(self):
        assert ONNXImageClassifier._infer_input_shape_from_dims([1, 1000], channels_first=True) is None


class TestSoftmax:
    def test_output_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = _softmax(x, axis=1)
        assert np.isclose(result.sum(), 1.0)

    def test_is_numerically_stable_for_large_values(self):
        x = np.array([[1000.0, 1001.0, 1002.0]])
        result = _softmax(x, axis=1)
        assert not np.any(np.isnan(result))
        assert np.isclose(result.sum(), 1.0)

    def test_highest_logit_gets_highest_probability(self):
        x = np.array([[0.1, 5.0, 0.2]])
        result = _softmax(x, axis=1)
        assert np.argmax(result[0]) == 1


class TestOnnxAutoDetection:
    """Extension-based backend auto-detection, exercised through the full
    ImageClassifierWrapper config path -- covers the new routing regardless
    of whether onnxruntime is actually installed or the file is a valid
    model (both cases should fail gracefully via can_run=False, never raise)."""

    def _make_wrapper(self, tmp_path, filename="model.onnx", backend="auto"):
        model_path = tmp_path / filename
        model_path.write_bytes(b"not a real onnx model, just needs to exist")
        config = ImageClassifierModelConfig(
            model_name="test-onnx",
            model_location=str(model_path),
            model_categories=["a", "b"],
            backend=backend,
        )
        return ImageClassifierWrapper(config)

    def test_onnx_extension_is_auto_detected(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        assert wrapper.backend == BackendType.ONNX

    def test_explicit_onnx_backend_is_respected(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path, filename="model.weird_ext", backend="onnx")
        assert wrapper.backend == BackendType.ONNX

    def test_invalid_onnx_file_fails_gracefully_without_raising(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path)
        assert wrapper.can_run is False

    def test_unrecognized_extension_with_auto_backend_still_fails_cleanly(self, tmp_path):
        wrapper = self._make_wrapper(tmp_path, filename="model.xyz")
        assert wrapper.backend is None
        assert wrapper.can_run is False

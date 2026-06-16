"""
Unit tests for BaseCompareEmbedding.compute_embedding_for_path: combining
multiple sampled-frame embeddings (video/GIF/PDF) into a single mean-pooled,
re-normalized vector instead of embedding only the first frame.

All filesystem and FrameCache operations are mocked so no real media is read.
"""
from unittest.mock import patch

import numpy as np
import pytest

from compare.base_compare_embedding import BaseCompareEmbedding

PATH = "/base/video.mp4"
STILL_FRAME = "/cache/video_first.jpg"


def _patch_dynamic(is_dynamic: bool):
    return patch("compare.base_compare_embedding.is_classifier_dynamic_media_path", return_value=is_dynamic)


def _patch_still_frame(path=STILL_FRAME):
    return patch("compare.base_compare_embedding.FrameCache.get_image_path", return_value=path)


def _patch_sample_paths(paths: list):
    return patch.object(BaseCompareEmbedding, "_get_dynamic_media_sample_paths", return_value=paths)


class TestStaticImageSkipsSampling:
    def test_static_image_embeds_single_resolved_frame(self):
        embeddings_func = lambda p: {"path": p}  # noqa: E731 - simple fake
        with _patch_dynamic(False), _patch_still_frame():
            result = BaseCompareEmbedding.compute_embedding_for_path("/base/photo.jpg", embeddings_func)
        assert result == {"path": STILL_FRAME}


class TestSampleDynamicMediaFlag:
    def test_false_flag_skips_sampling_even_for_dynamic_media(self):
        """V-JEPA 2 (and similar natively video-aware models) pass sample_dynamic_media=False."""
        calls = []

        def embeddings_func(p):
            calls.append(p)
            return [1.0, 0.0]

        with _patch_dynamic(True), _patch_still_frame(), _patch_sample_paths(["/a.jpg", "/b.jpg"]):
            result = BaseCompareEmbedding.compute_embedding_for_path(
                PATH, embeddings_func, sample_dynamic_media=False
            )
        assert calls == [STILL_FRAME]
        assert result == [1.0, 0.0]


class TestDynamicMediaSingleSample:
    def test_single_sample_path_returns_unaveraged_embedding(self):
        embeddings_func = lambda p: {"path": p}  # noqa: E731
        with _patch_dynamic(True), _patch_sample_paths(["/only.jpg"]):
            result = BaseCompareEmbedding.compute_embedding_for_path(PATH, embeddings_func)
        assert result == {"path": "/only.jpg"}


class TestDynamicMediaAveraging:
    def test_multiple_samples_are_mean_pooled_and_renormalized(self):
        sample_paths = ["/s0.jpg", "/s1.jpg", "/s2.jpg"]
        vectors = {
            "/s0.jpg": [1.0, 0.0],
            "/s1.jpg": [0.0, 1.0],
            "/s2.jpg": [1.0, 0.0],
        }
        embeddings_func = lambda p: vectors[p]  # noqa: E731

        with _patch_dynamic(True), _patch_sample_paths(sample_paths):
            result = BaseCompareEmbedding.compute_embedding_for_path(PATH, embeddings_func)

        expected_mean = np.mean(np.array(list(vectors.values())), axis=0)
        expected = (expected_mean / np.linalg.norm(expected_mean)).tolist()
        assert result == pytest.approx(expected)
        # Sanity: result is a unit vector, valid for cosine similarity via dot product.
        assert np.linalg.norm(result) == pytest.approx(1.0)

    def test_per_sample_exceptions_are_skipped_not_fatal(self):
        sample_paths = ["/good0.jpg", "/bad.jpg", "/good1.jpg"]

        def embeddings_func(p):
            if p == "/bad.jpg":
                raise ValueError("corrupt frame")
            return [1.0, 0.0] if p == "/good0.jpg" else [0.0, 1.0]

        with _patch_dynamic(True), _patch_sample_paths(sample_paths):
            result = BaseCompareEmbedding.compute_embedding_for_path(PATH, embeddings_func)

        expected_mean = np.array([0.5, 0.5])
        expected = (expected_mean / np.linalg.norm(expected_mean)).tolist()
        assert result == pytest.approx(expected)

    def test_none_embeddings_are_filtered_like_face_detection_misses(self):
        """e.g. image_embeddings_face returns None for a frame with no detected face."""
        sample_paths = ["/face.jpg", "/noface.jpg"]
        vectors = {"/face.jpg": [0.6, 0.8], "/noface.jpg": None}
        embeddings_func = lambda p: vectors[p]  # noqa: E731

        with _patch_dynamic(True), _patch_sample_paths(sample_paths):
            result = BaseCompareEmbedding.compute_embedding_for_path(PATH, embeddings_func)

        # Only one valid embedding survives -> returned as-is (no averaging needed).
        assert result == [0.6, 0.8]

    def test_all_samples_fail_falls_back_to_single_still_frame(self):
        sample_paths = ["/bad0.jpg", "/bad1.jpg"]

        def embeddings_func(p):
            if p in sample_paths:
                raise OSError("cannot read")
            return {"path": p}

        with _patch_dynamic(True), _patch_sample_paths(sample_paths), _patch_still_frame():
            result = BaseCompareEmbedding.compute_embedding_for_path(PATH, embeddings_func)

        assert result == {"path": STILL_FRAME}

    def test_all_samples_none_falls_back_to_single_still_frame(self):
        sample_paths = ["/noface0.jpg", "/noface1.jpg"]

        def embeddings_func(p):
            if p in sample_paths:
                return None
            return {"path": p}

        with _patch_dynamic(True), _patch_sample_paths(sample_paths), _patch_still_frame():
            result = BaseCompareEmbedding.compute_embedding_for_path(PATH, embeddings_func)

        assert result == {"path": STILL_FRAME}


class TestGetDynamicMediaSamplePaths:
    def test_uses_compare_specific_config_caps(self):
        captured = {}

        def fake_stream(path, sample_ratio=None, max_samples=None):
            captured["sample_ratio"] = sample_ratio
            captured["max_samples"] = max_samples
            return 2, iter(["/a.jpg", "/b.jpg"])

        with patch("compare.base_compare_embedding.FrameCache.stream_frame_samples", side_effect=fake_stream), \
             patch("compare.base_compare_embedding.config") as mock_config:
            mock_config.compare_embedding_dynamic_media_sample_ratio = 0.25
            mock_config.compare_embedding_dynamic_media_max_samples = 3
            result = BaseCompareEmbedding._get_dynamic_media_sample_paths(PATH)

        assert result == ["/a.jpg", "/b.jpg"]
        assert captured == {"sample_ratio": 0.25, "max_samples": 3}

    def test_sampling_failure_falls_back_to_single_still_frame(self):
        with patch(
            "compare.base_compare_embedding.FrameCache.stream_frame_samples",
            side_effect=RuntimeError("boom"),
        ), _patch_still_frame():
            result = BaseCompareEmbedding._get_dynamic_media_sample_paths(PATH)

        assert result == [STILL_FRAME]

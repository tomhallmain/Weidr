"""FrameCache pseudostatic detection when a video's sampling is already cached.

A sampling run cached by a caller that did not ask for detection (another
prevalidation, a trigger-slot seek) must give the same verdict as a fresh run
with detection on: a single decoded frame is pseudostatic and a single-frame
stream, two or more frames are compared visually."""

import cv2
import numpy as np
import pytest

import image.frame_cache as frame_cache_module
from image.frame_cache import FrameCache, _make_sample_cache_key, _pseudostatic_verdict


@pytest.fixture
def fc(monkeypatch):
    for attr in ("sampled_cache", "media_stats_cache"):
        monkeypatch.setattr(FrameCache, attr, {})
    monkeypatch.setattr(frame_cache_module.config, "enable_videos", True)
    return FrameCache


def _jpeg(path, value):
    cv2.imwrite(str(path), np.full((32, 32, 3), value, dtype=np.uint8))
    return str(path)


def _cache_sampling(fc, video_path, frames, ratio=0.1):
    fc.sampled_cache[_make_sample_cache_key(video_path, ratio)] = list(frames)


class TestPseudostaticVerdict:
    def test_single_real_frame(self, tmp_path):
        frame = _jpeg(tmp_path / "f0.jpg", 10)
        assert _pseudostatic_verdict([frame]) == (True, True)

    def test_no_real_frame(self, tmp_path):
        assert _pseudostatic_verdict([str(tmp_path / "clip.mp4")]) == (None, False)

    def test_identical_frames(self, tmp_path):
        frames = [_jpeg(tmp_path / f"f{i}.jpg", 10) for i in range(3)]
        assert _pseudostatic_verdict(frames) == (True, False)

    def test_differing_frames(self, tmp_path):
        frames = [_jpeg(tmp_path / "f0.jpg", 10), _jpeg(tmp_path / "f1.jpg", 200)]
        assert _pseudostatic_verdict(frames) == (False, False)


class TestCachedSampling:
    def test_single_frame_cached_without_detection_is_pseudostatic(self, fc, tmp_path):
        video = str(tmp_path / "clip.mp4")
        _cache_sampling(fc, video, [_jpeg(tmp_path / "f0.jpg", 10)])

        planned, frames = fc.stream_frame_samples(video, sample_ratio=0.1, detect_pseudostatic=True)
        list(frames)

        assert planned == 1
        assert fc.is_pseudostatic_dynamic_media(video)
        assert fc.is_single_frame_stream(video)

    def test_identical_frames_cached_are_pseudostatic_not_single_frame(self, fc, tmp_path):
        video = str(tmp_path / "clip.mp4")
        _cache_sampling(fc, video, [_jpeg(tmp_path / f"f{i}.jpg", 10) for i in range(3)])

        list(fc.stream_frame_samples(video, sample_ratio=0.1, detect_pseudostatic=True)[1])

        assert fc.is_pseudostatic_dynamic_media(video)
        assert not fc.is_single_frame_stream(video)

    def test_differing_frames_cached_are_not_pseudostatic(self, fc, tmp_path):
        video = str(tmp_path / "clip.mp4")
        _cache_sampling(fc, video, [_jpeg(tmp_path / "f0.jpg", 10), _jpeg(tmp_path / "f1.jpg", 200)])

        list(fc.stream_frame_samples(video, sample_ratio=0.1, detect_pseudostatic=True)[1])

        assert not fc.is_pseudostatic_dynamic_media(video)

    def test_without_detection_nothing_is_recorded(self, fc, tmp_path):
        video = str(tmp_path / "clip.mp4")
        _cache_sampling(fc, video, [_jpeg(tmp_path / "f0.jpg", 10)])

        list(fc.stream_frame_samples(video, sample_ratio=0.1, detect_pseudostatic=False)[1])

        assert not fc.is_pseudostatic_dynamic_media(video)
        assert video not in fc.media_stats_cache

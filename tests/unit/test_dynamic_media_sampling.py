"""compare/dynamic_media_sampling.evaluate_dynamic_media: the frame-sampling
loop shared by classifier actions and compare filters."""

from unittest.mock import patch

from compare.dynamic_media_sampling import evaluate_dynamic_media
from image.frame_cache import FrameCache


class _Frames:
    """Iterator over frame paths that records how far it was consumed and
    whether it was closed."""

    def __init__(self, paths):
        self._it = iter(paths)
        self.yielded = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self._it)
        self.yielded += 1
        return value

    def close(self):
        self.closed = True


def _run(frames, matches, positive_ratio=0.5, planned=None):
    it = _Frames(frames)
    planned = len(frames) if planned is None else planned
    matcher = lambda p: (matches.get(p, False), "cat" if matches.get(p) else None)
    with patch.object(FrameCache, "stream_frame_samples", return_value=(planned, it)):
        result = evaluate_dynamic_media("/v.mp4", matcher, sample_ratio=0.1, positive_ratio=positive_ratio)
    return result, it


def test_no_planned_samples_returns_none():
    with patch.object(FrameCache, "stream_frame_samples", return_value=(0, iter([]))):
        assert evaluate_dynamic_media("/v.mp4", lambda p: (True, None), 0.1, 0.1) is None


def test_threshold_met_stops_early():
    frames = ["f1", "f2", "f3", "f4"]
    result, it = _run(frames, {"f1": True, "f2": True}, positive_ratio=0.5)
    assert result.threshold_met is True
    assert result.required_positive_count == 2
    assert it.yielded == 2
    assert result.matched_category == "cat"
    assert it.closed


def test_early_failure_when_threshold_unreachable():
    frames = ["f1", "f2", "f3", "f4"]
    result, it = _run(frames, {}, positive_ratio=0.75)
    assert result.threshold_met is False
    assert it.yielded == 2  # after 2 misses, 2 remaining < 3 required
    assert result.reached_last_sample is False


def test_all_samples_consumed_without_match():
    result, it = _run(["f1", "f2"], {}, positive_ratio=0.0)
    assert result.threshold_met is False
    assert result.reached_last_sample is True
    assert result.processed_samples == 2


def test_raising_matcher_counts_as_processed_non_match():
    def matcher(p):
        if p == "f1":
            raise RuntimeError("decode failed")
        return True, None

    it = _Frames(["f1", "f2"])
    with patch.object(FrameCache, "stream_frame_samples", return_value=(2, it)):
        result = evaluate_dynamic_media("/v.mp4", matcher, 0.1, 0.5)
    assert result.threshold_met is True
    assert result.processed_samples == 2
    assert result.positive_count == 1

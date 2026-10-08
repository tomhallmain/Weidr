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


# ---------------------------------------------------------------------------
# count_sample_matches / match_media / normalize_ratio
# ---------------------------------------------------------------------------

from compare.dynamic_media_sampling import (  # noqa: E402
    FrameSampling,
    count_sample_matches,
    match_media,
    normalize_ratio,
)


def test_count_records_first_match_with_slot_offset():
    frames = _Frames(["f3", "f4", "f5"])
    result = count_sample_matches(
        frames, 3, 2, lambda p: (p in ("f4", "f5"), None), slot_offset=3)
    assert result.threshold_met is True
    assert (result.first_match_index, result.first_match_path) == (4, "f4")
    assert result.last_processed_index == 5
    assert frames.closed


def test_count_without_match_leaves_first_match_unset():
    result = count_sample_matches(iter(["a", "b"]), 2, 1, lambda p: (False, None))
    assert result.first_match_index is None
    assert result.threshold_met is False
    assert result.processed_samples == 2


class TestMatchMedia:
    def test_dynamic_media_is_sampled(self, monkeypatch):
        import utils.media_utils as mu
        monkeypatch.setattr(mu, "is_classifier_dynamic_media_path", lambda p: True)
        it = _Frames(["f1", "f2"])
        with patch.object(FrameCache, "stream_frame_samples", return_value=(2, it)) as stream:
            result = match_media("/v.mp4", lambda p: (p == "f2", "cat"), FrameSampling(0.3, 0.5))
        assert result.matched is True
        assert result.matched_category == "cat"
        assert result.sampling.positive_count == 1
        assert stream.call_args.kwargs["sample_ratio"] == 0.3

    def test_still_media_uses_the_frame_cache_image(self, monkeypatch):
        import utils.media_utils as mu
        monkeypatch.setattr(mu, "is_classifier_dynamic_media_path", lambda p: False)
        seen = []
        with patch.object(FrameCache, "get_image_path", return_value="/render.png"):
            result = match_media("/drawing.svg", lambda p: (seen.append(p) or True, None), FrameSampling())
        assert seen == ["/render.png"]
        assert result.matched is True
        assert result.sampling is None

    def test_no_planned_samples_falls_back_to_one_image(self, monkeypatch):
        import utils.media_utils as mu
        monkeypatch.setattr(mu, "is_classifier_dynamic_media_path", lambda p: True)
        with patch.object(FrameCache, "stream_frame_samples", return_value=(0, iter([]))), \
                patch.object(FrameCache, "get_image_path", return_value="/first.png"):
            result = match_media("/v.mp4", lambda p: (p == "/first.png", None), FrameSampling())
        assert result.matched is True
        assert result.sampling is None


def test_normalize_ratio():
    assert normalize_ratio(0.4) == 0.4
    assert normalize_ratio(5) == 1.0
    assert normalize_ratio(-1) == 0.0
    assert normalize_ratio("bad", 0.2) == 0.2
    assert normalize_ratio(None) == 0.1

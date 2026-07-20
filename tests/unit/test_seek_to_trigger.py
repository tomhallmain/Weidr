"""
Unit tests for the Seek-to-Trigger feature.

Covers:
  - ClassifierAction.find_first_trigger_slot() algorithm (threshold, start_slot, cycling)
  - TriggerDetail population via _evaluate_image_path_match _detail_out
  - FrameCache.slot_index_to_seek_position() conversion
  - _format_trigger_detail() display helper
  - SeekToTriggerTab._restore_last_action_from_cache() cross-session persistence

All tests mock FrameCache.stream_frame_samples and the action's
_evaluate_image_path_match so no real media files or models are needed.
"""

from unittest.mock import MagicMock, patch

import pytest

from compare.classifier_action import ClassifierAction, Prevalidation, TriggerDetail, TriggerFrameResult
from compare.classifier_actions_manager import ClassifierActionsManager
from image.frame_cache import FrameCache, MediaStats, SeekPosition
from ui.compare.seek_to_trigger_tab_qt import (
    _format_trigger_detail,
    _LAST_ACTION_CACHE_KEY,
    SeekToTriggerTab,
)
from utils.constants import ClassifierActionType
from utils.translations import _


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(**kwargs) -> ClassifierAction:
    """Minimal ClassifierAction wired with no real classifiers."""
    defaults = dict(
        name="test",
        action=ClassifierActionType.NOTIFY,
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_blacklist=False,
        use_prototype=False,
        dynamic_content_sample_ratio=0.1,
        dynamic_content_positive_ratio=0.2,
    )
    defaults.update(kwargs)
    return ClassifierAction(**defaults)


def _mock_stream(planned_slots, frame_paths):
    """Return a (planned_slots, iter) pair for mocking stream_frame_samples."""
    return planned_slots, iter(frame_paths)


# ---------------------------------------------------------------------------
# find_first_trigger_slot — non-dynamic media
# ---------------------------------------------------------------------------

class TestFindFirstTriggerSlotNonDynamic:
    def test_returns_none_for_static_image(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"x")
        ca = _action()
        result = ca.find_first_trigger_slot(str(img))
        assert result is None

    def test_returns_none_for_txt_file(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_bytes(b"hello")
        ca = _action()
        result = ca.find_first_trigger_slot(str(f))
        assert result is None


# ---------------------------------------------------------------------------
# find_first_trigger_slot — dynamic media (mocked stream)
# ---------------------------------------------------------------------------

class TestFindFirstTriggerSlotAlgorithm:
    """
    Tests use a fake .mp4 path so is_classifier_dynamic_media_path returns True,
    and mock both stream_frame_samples and _evaluate_image_path_match.
    """

    FAKE_VIDEO = "/fake/video.mp4"

    def _run(self, planned_slots, match_pattern, positive_ratio=0.2, sample_ratio=0.1,
             start_slot=0):
        """
        match_pattern: list of bools, one per frame yielded by the stream.
        Returns find_first_trigger_slot() result.
        _eval accepts _detail_out so the second (detail-capture) call does not
        raise TypeError and detail stays None for these algorithm-only tests.
        """
        frame_paths = [f"/tmp/frame_{i}.jpg" for i in range(len(match_pattern))]
        ca = _action(
            dynamic_content_positive_ratio=positive_ratio,
            dynamic_content_sample_ratio=sample_ratio,
        )

        path_to_match = {f"/tmp/frame_{i}.jpg": match_pattern[i]
                         for i in range(len(match_pattern))}

        def _eval(path, _detail_out=None):
            return path_to_match.get(path, False), None

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(planned_slots, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match", side_effect=_eval):
                    return ca.find_first_trigger_slot(self.FAKE_VIDEO, start_slot=start_slot)

    # -- Threshold met -------------------------------------------------------

    def test_single_match_meets_threshold(self):
        # 5 planned slots, positive_ratio=0.2 → required=ceil(5*0.2)=1
        result = self._run(planned_slots=5, match_pattern=[True, False, False, False, False])
        assert isinstance(result, TriggerFrameResult)
        assert result.slot_index == 0
        assert result.total_planned_slots == 5

    def test_seeks_to_first_positive_not_threshold_crossing_frame(self):
        # 10 slots, ratio=0.3 → required=ceil(10*0.3)=3
        # First positive at slot 1, threshold met at slot 5
        pattern = [False, True, False, False, False, True, False, False, True, False]
        result = self._run(planned_slots=10, match_pattern=pattern, positive_ratio=0.3)
        assert result is not None
        # Threshold is met at slot 5 (3rd positive), but seek target is first positive (slot 1)
        assert result.slot_index == 1
        assert result.total_planned_slots == 10

    def test_first_and_threshold_crossing_frame_are_same_when_required_equals_one(self):
        # required=1 → first match IS the threshold crossing
        result = self._run(planned_slots=5, match_pattern=[False, True, False, False, False])
        assert result is not None
        assert result.slot_index == 1

    def test_all_frames_match_returns_slot_zero(self):
        result = self._run(planned_slots=5, match_pattern=[True, True, True, True, True])
        assert result is not None
        assert result.slot_index == 0

    # -- Threshold not met ---------------------------------------------------

    def test_no_matches_returns_none(self):
        result = self._run(planned_slots=5, match_pattern=[False, False, False, False, False])
        assert result is None

    def test_insufficient_matches_returns_none(self):
        # 10 slots, ratio=0.3 → required=3; only 2 positives
        pattern = [True, False, True, False, False, False, False, False, False, False]
        result = self._run(planned_slots=10, match_pattern=pattern, positive_ratio=0.3)
        assert result is None

    # -- Early-abort ---------------------------------------------------------

    def test_early_abort_stops_iteration_when_threshold_unreachable(self):
        """Iterator should be consumed only as far as needed."""
        consumed = []
        planned_slots = 10
        frame_paths = [f"/tmp/frame_{i}.jpg" for i in range(10)]
        ca = _action(dynamic_content_positive_ratio=0.5)

        def _eval(path, _detail_out=None):
            consumed.append(path)
            return False, None

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(planned_slots, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match", side_effect=_eval):
                    result = ca.find_first_trigger_slot("/fake/video.mp4")

        assert result is None
        assert len(consumed) < 10

    # -- Exception handling --------------------------------------------------

    def test_exception_in_evaluate_is_treated_as_no_match(self):
        # ratio=0.2, 5 slots → required=1. First frame raises, second matches.
        frame_paths = [f"/tmp/frame_{i}.jpg" for i in range(5)]
        ca = _action(dynamic_content_positive_ratio=0.2)
        call_count = [0]

        def _eval(path, _detail_out=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("decode error")
            return True, None

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(5, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match", side_effect=_eval):
                    result = ca.find_first_trigger_slot("/fake/video.mp4")

        assert result is not None
        assert result.slot_index == 1  # first *successful* positive match

    # -- Edge: zero planned slots --------------------------------------------

    def test_zero_planned_slots_returns_none(self):
        ca = _action()
        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples", return_value=(0, iter([]))):
                result = ca.find_first_trigger_slot("/fake/video.mp4")
        assert result is None

    # -- Prevalidation subclass ----------------------------------------------

    def test_works_on_prevalidation_subclass(self):
        frame_paths = ["/tmp/f0.jpg", "/tmp/f1.jpg"]
        pv = Prevalidation(
            name="pv",
            action=ClassifierActionType.HIDE,
            use_embedding=False,
            dynamic_content_positive_ratio=0.5,
        )
        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(2, frame_paths)):
                with patch.object(pv, "_evaluate_image_path_match",
                                  side_effect=lambda p, _detail_out=None: (True, None)):
                    result = pv.find_first_trigger_slot("/fake/video.mp4")
        assert result is not None
        assert result.slot_index == 0


# ---------------------------------------------------------------------------
# find_first_trigger_slot — start_slot / "next trigger" cycling
# ---------------------------------------------------------------------------

class TestFindFirstTriggerSlotStartSlot:
    """Tests for the start_slot parameter that enables next-trigger cycling."""

    FAKE_VIDEO = "/fake/video.mp4"

    def _run_with_start(self, planned_slots, match_pattern, start_slot, positive_ratio=0.2):
        frame_paths = [f"/tmp/frame_{i}.jpg" for i in range(len(match_pattern))]
        ca = _action(dynamic_content_positive_ratio=positive_ratio)

        # Path-based matching so the mock is correct regardless of call order.
        # When start_slot > 0, find_first_trigger_slot materialises all frames
        # without calling _eval, then scans the sub-range — a sequential iterator
        # would be misaligned. Path lookup is always correct, and also handles the
        # second (detail-capture) call to _evaluate_image_path_match.
        path_to_match = {f"/tmp/frame_{i}.jpg": match_pattern[i]
                         for i in range(len(match_pattern))}

        def _eval(path, _detail_out=None):
            return path_to_match.get(path, False), None

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(planned_slots, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match", side_effect=_eval):
                    return ca.find_first_trigger_slot(self.FAKE_VIDEO, start_slot=start_slot)

    def test_start_slot_zero_is_same_as_default(self):
        # [True at slot 0] → slot_index=0
        result = self._run_with_start(5, [True, False, False, False, False], start_slot=0)
        assert result is not None
        assert result.slot_index == 0

    def test_start_slot_skips_earlier_matches(self):
        # Matches at slots 0, 3. With start_slot=1, slot 0 is skipped → finds slot 3.
        # ratio=0.2, 5 planned, sub-range from slot 1 = 4 frames, required=ceil(4*0.2)=1
        pattern = [True, False, False, True, False]
        result = self._run_with_start(5, pattern, start_slot=1)
        assert result is not None
        assert result.slot_index == 3

    def test_start_slot_at_boundary_returns_none(self):
        # start_slot = planned_slots → nothing to scan
        result = self._run_with_start(5, [True, True, True, True, True], start_slot=5)
        assert result is None

    def test_start_slot_beyond_boundary_returns_none(self):
        result = self._run_with_start(5, [True, True, True, True, True], start_slot=99)
        assert result is None

    def test_start_slot_no_matches_after_start_returns_none(self):
        # Match only at slot 0, start_slot=1 → sub-range has no matches
        result = self._run_with_start(5, [True, False, False, False, False], start_slot=1)
        assert result is None

    def test_start_slot_absolute_index_in_result(self):
        # 8 frames, match at slots 2 and 5. start_slot=3 → sub-range [3..7], match at slot 5.
        # Match pattern relative to full stream: F F T F F T F F
        # With start_slot=3 we pass [False, False, True, False, False] to scanner
        pattern = [False, False, True, False, False, True, False, False]
        result = self._run_with_start(8, pattern, start_slot=3)
        assert result is not None
        assert result.slot_index == 5  # absolute index, not local

    def test_start_slot_total_planned_slots_reflects_materialized_length(self):
        # After materialisation, total_planned_slots is len(all_frames).
        pattern = [False, False, True, False, False]
        result = self._run_with_start(5, pattern, start_slot=1)
        assert result is not None
        assert result.total_planned_slots == 5  # whole materialised list


# ---------------------------------------------------------------------------
# find_first_trigger_slot — TriggerDetail population
# ---------------------------------------------------------------------------

class TestFindFirstTriggerSlotSampleRatio:
    """Verify that providing sample_ratio overrides the action's ratio and lifts the cap."""

    FAKE_VIDEO = "/fake/video.mp4"

    def test_sample_ratio_none_uses_action_ratio_without_max_samples(self):
        """Default call (no sample_ratio) must NOT pass max_samples to stream_frame_samples."""
        frame_paths = ["/tmp/f0.jpg"]
        ca = _action(dynamic_content_positive_ratio=0.2, dynamic_content_sample_ratio=0.15)

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(
                FrameCache, "stream_frame_samples",
                return_value=_mock_stream(1, frame_paths),
            ) as mock_stream:
                with patch.object(ca, "_evaluate_image_path_match",
                                  side_effect=lambda p, _detail_out=None: (False, None)):
                    ca.find_first_trigger_slot(self.FAKE_VIDEO)

        mock_stream.assert_called_once()
        kwargs = mock_stream.call_args.kwargs
        assert kwargs.get("sample_ratio") == 0.15
        assert "max_samples" not in kwargs  # normal path — no cap override

    def test_sample_ratio_override_passes_max_samples_to_stream(self):
        """Providing sample_ratio must call stream_frame_samples with max_samples set."""
        frame_paths = ["/tmp/f0.jpg"]
        ca = _action(dynamic_content_positive_ratio=0.2, dynamic_content_sample_ratio=0.15)

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(
                FrameCache, "stream_frame_samples",
                return_value=_mock_stream(1, frame_paths),
            ) as mock_stream:
                with patch.object(ca, "_evaluate_image_path_match",
                                  side_effect=lambda p, _detail_out=None: (False, None)):
                    ca.find_first_trigger_slot(self.FAKE_VIDEO, sample_ratio=0.5)

        mock_stream.assert_called_once()
        kwargs = mock_stream.call_args.kwargs
        assert kwargs.get("sample_ratio") == 0.5, "should use the provided ratio"
        assert kwargs.get("max_samples") is not None, "should override the frame cap"
        assert kwargs["max_samples"] > 1000, "cap override must be very large"

    def test_sample_ratio_override_result_is_correct(self):
        """With sample_ratio=0.5, the algorithm still returns the right trigger frame."""
        frame_paths = [f"/tmp/frame_{i}.jpg" for i in range(5)]
        ca = _action(dynamic_content_positive_ratio=0.2)
        path_to_match = {p: (i == 2) for i, p in enumerate(frame_paths)}

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(5, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match",
                                  side_effect=lambda p, _detail_out=None: (path_to_match.get(p, False), None)):
                    result = ca.find_first_trigger_slot(self.FAKE_VIDEO, sample_ratio=0.5)

        assert result is not None
        assert result.slot_index == 2


class TestFindFirstTriggerSlotDetail:
    """
    Verify that detail is populated when _evaluate_image_path_match fills _detail_out,
    and remains None when the mock does not fill it.
    """

    FAKE_VIDEO = "/fake/video.mp4"

    def test_detail_none_when_mock_does_not_populate(self):
        frame_paths = ["/tmp/f0.jpg"]
        ca = _action()

        def _eval(path, _detail_out=None):
            return True, None  # does NOT set _detail_out[0]

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(1, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match", side_effect=_eval):
                    result = ca.find_first_trigger_slot(self.FAKE_VIDEO)

        assert result is not None
        assert result.detail is None

    def test_detail_populated_when_mock_sets_detail_out(self):
        frame_paths = ["/tmp/f0.jpg"]
        ca = _action()
        expected_detail = TriggerDetail(
            trigger_type="image_classifier",
            category="portrait",
            top_predictions=[("portrait", 0.87), ("person", 0.09)],
        )

        def _eval(path, _detail_out=None):
            if _detail_out is not None:
                _detail_out[0] = expected_detail
            return True, "portrait"

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(1, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match", side_effect=_eval):
                    result = ca.find_first_trigger_slot(self.FAKE_VIDEO)

        assert result is not None
        assert result.detail is expected_detail
        assert result.detail.trigger_type == "image_classifier"
        assert result.detail.category == "portrait"
        assert result.detail.top_predictions[0] == ("portrait", 0.87)

    def test_detail_capture_exception_does_not_abort_result(self):
        """If the detail re-evaluation raises, the TriggerFrameResult is still returned."""
        frame_paths = ["/tmp/f0.jpg"]
        ca = _action()
        call_count = [0]

        def _eval(path, _detail_out=None):
            call_count[0] += 1
            if _detail_out is not None:
                raise RuntimeError("classifier crashed")
            return True, None

        with patch("compare.classifier_action.is_classifier_dynamic_media_path", return_value=True):
            with patch.object(FrameCache, "stream_frame_samples",
                              return_value=_mock_stream(1, frame_paths)):
                with patch.object(ca, "_evaluate_image_path_match", side_effect=_eval):
                    result = ca.find_first_trigger_slot(self.FAKE_VIDEO)

        assert result is not None
        assert result.slot_index == 0
        assert result.detail is None  # detail call failed; result still returned


# ---------------------------------------------------------------------------
# TriggerDetail dataclass
# ---------------------------------------------------------------------------

class TestTriggerDetail:
    def test_minimal_fields(self):
        d = TriggerDetail(trigger_type="embedding")
        assert d.trigger_type == "embedding"
        assert d.category is None
        assert d.top_predictions is None

    def test_full_fields(self):
        preds = [("cat_a", 0.9), ("cat_b", 0.1)]
        d = TriggerDetail(trigger_type="image_classifier", category="cat_a", top_predictions=preds)
        assert d.trigger_type == "image_classifier"
        assert d.category == "cat_a"
        assert d.top_predictions == preds

    def test_equality(self):
        a = TriggerDetail("prototype")
        b = TriggerDetail("prototype")
        assert a == b

    def test_inequality(self):
        a = TriggerDetail("embedding")
        b = TriggerDetail("prompt")
        assert a != b


# ---------------------------------------------------------------------------
# TriggerFrameResult dataclass
# ---------------------------------------------------------------------------

class TestTriggerFrameResult:
    def test_fields_without_detail(self):
        r = TriggerFrameResult(slot_index=3, total_planned_slots=10, frame_path="/tmp/f.jpg")
        assert r.slot_index == 3
        assert r.total_planned_slots == 10
        assert r.frame_path == "/tmp/f.jpg"
        assert r.detail is None

    def test_fields_with_detail(self):
        d = TriggerDetail(trigger_type="prompt")
        r = TriggerFrameResult(slot_index=1, total_planned_slots=5, frame_path="/tmp/f.jpg", detail=d)
        assert r.detail is d
        assert r.detail.trigger_type == "prompt"

    def test_equality(self):
        a = TriggerFrameResult(1, 10, "/tmp/a.jpg")
        b = TriggerFrameResult(1, 10, "/tmp/a.jpg")
        assert a == b

    def test_inequality(self):
        a = TriggerFrameResult(1, 10, "/tmp/a.jpg")
        b = TriggerFrameResult(2, 10, "/tmp/a.jpg")
        assert a != b

    def test_detail_not_compared_for_equality(self):
        # detail is excluded from __eq__ only if dataclass default behaviour applies.
        # Since we did NOT set eq=False, equality compares all fields including detail.
        a = TriggerFrameResult(1, 10, "/tmp/a.jpg", detail=TriggerDetail("embedding"))
        b = TriggerFrameResult(1, 10, "/tmp/a.jpg", detail=None)
        assert a != b


# ---------------------------------------------------------------------------
# SeekPosition dataclass
# ---------------------------------------------------------------------------

class TestSeekPosition:
    def test_fields(self):
        s = SeekPosition(kind="ms", value=42000)
        assert s.kind == "ms"
        assert s.value == 42000

    def test_pdf_kind(self):
        s = SeekPosition(kind="page", value=3)
        assert s.kind == "page"
        assert s.value == 3


# ---------------------------------------------------------------------------
# FrameCache.slot_index_to_seek_position
# ---------------------------------------------------------------------------

class TestSlotIndexToSeekPosition:
    FAKE_PATH = "/fake/video.mp4"

    def _inject_stats(self, stats: MediaStats) -> None:
        FrameCache.media_stats_cache[self.FAKE_PATH] = stats

    def teardown_method(self, _method) -> None:
        FrameCache.media_stats_cache.pop(self.FAKE_PATH, None)

    # -- Normal video --------------------------------------------------------

    def test_video_basic(self):
        self._inject_stats(MediaStats(
            media_type="video",
            total_items=300,
            fps=30.0,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 0, 10)
        # step = 300//10 = 30, actual_index=0, ms=0/30*1000=0
        assert pos is not None
        assert pos.kind == "ms"
        assert pos.value == 0

    def test_video_mid_slot(self):
        self._inject_stats(MediaStats(
            media_type="video",
            total_items=300,
            fps=30.0,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 3, 10)
        # step=30, actual_index=90, ms=90/30*1000=3000
        assert pos is not None
        assert pos.kind == "ms"
        assert pos.value == 3000

    def test_video_last_slot(self):
        self._inject_stats(MediaStats(
            media_type="video",
            total_items=600,
            fps=25.0,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 9, 10)
        # step=600//10=60, actual_index=540, ms=540/25*1000=21600
        assert pos is not None
        assert pos.kind == "ms"
        assert pos.value == 21600

    # -- GIF (same code path as video) ----------------------------------------

    def test_gif_uses_ms(self):
        self._inject_stats(MediaStats(
            media_type="gif",
            total_items=50,
            fps=10.0,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 2, 5)
        # step=50//5=10, actual_index=20, ms=20/10*1000=2000
        assert pos is not None
        assert pos.kind == "ms"
        assert pos.value == 2000

    # -- PDF ----------------------------------------------------------------

    def test_pdf_returns_page(self):
        self._inject_stats(MediaStats(
            media_type="pdf",
            total_items=20,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 2, 5)
        # step=20//5=4, actual_index=8
        assert pos is not None
        assert pos.kind == "page"
        assert pos.value == 8

    def test_pdf_first_page(self):
        self._inject_stats(MediaStats(
            media_type="pdf",
            total_items=10,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 0, 5)
        assert pos is not None
        assert pos.kind == "page"
        assert pos.value == 0

    # -- Missing stats -------------------------------------------------------

    def test_no_stats_returns_none(self):
        pos = FrameCache.slot_index_to_seek_position("/nonexistent/video.mp4", 1, 10)
        assert pos is None

    def test_zero_total_planned_slots_returns_none(self):
        self._inject_stats(MediaStats(media_type="video", total_items=300, fps=30.0))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 0, 0)
        assert pos is None

    def test_none_total_items_returns_none(self):
        self._inject_stats(MediaStats(media_type="video", total_items=None, fps=30.0))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 1, 10)
        assert pos is None

    # -- No FPS (duration fallback) -----------------------------------------

    def test_no_fps_falls_back_to_duration(self):
        self._inject_stats(MediaStats(
            media_type="video",
            total_items=100,
            fps=None,
            duration_seconds=60.0,
        ))
        # 5 slots, slot_index=2 → frac=2/(5-1)=0.5, ms=0.5*60*1000=30000
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 2, 5)
        assert pos is not None
        assert pos.kind == "ms"
        assert pos.value == 30000

    def test_no_fps_no_duration_returns_none(self):
        self._inject_stats(MediaStats(
            media_type="video",
            total_items=100,
            fps=None,
            duration_seconds=None,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 1, 10)
        assert pos is None

    def test_zero_fps_no_duration_returns_none(self):
        self._inject_stats(MediaStats(
            media_type="video",
            total_items=100,
            fps=0.0,
            duration_seconds=None,
        ))
        pos = FrameCache.slot_index_to_seek_position(self.FAKE_PATH, 1, 10)
        assert pos is None


# ---------------------------------------------------------------------------
# _format_trigger_detail
# ---------------------------------------------------------------------------

class TestFormatTriggerDetail:
    def test_none_returns_empty(self):
        assert _format_trigger_detail(None) == ""

    def test_embedding(self):
        d = TriggerDetail(trigger_type="embedding")
        result = _format_trigger_detail(d)
        assert result == _("Trigger: text embedding")

    def test_prompt(self):
        d = TriggerDetail(trigger_type="prompt")
        result = _format_trigger_detail(d)
        assert result == _("Trigger: prompt match")

    def test_prototype(self):
        d = TriggerDetail(trigger_type="prototype")
        result = _format_trigger_detail(d)
        assert result == _("Trigger: prototype")

    def test_filename(self):
        d = TriggerDetail(trigger_type="filename")
        result = _format_trigger_detail(d)
        assert result == _("Trigger: filename match")

    def test_image_classifier_with_category_and_predictions(self):
        preds = [("portrait", 0.87), ("person", 0.09), ("scene", 0.04)]
        d = TriggerDetail(trigger_type="image_classifier", category="portrait", top_predictions=preds)
        result = _format_trigger_detail(d)
        assert result.startswith(_("Trigger: image classifier"))
        assert "portrait" in result
        assert "87%" in result

    def test_image_classifier_low_score_predictions_omitted(self):
        preds = [("cat_a", 0.99), ("cat_b", 0.005)]  # cat_b below 1% threshold
        d = TriggerDetail(trigger_type="image_classifier", category="cat_a", top_predictions=preds)
        result = _format_trigger_detail(d)
        assert "cat_a" in result
        assert "cat_b" not in result

    def test_image_classifier_no_category_no_crash(self):
        d = TriggerDetail(trigger_type="image_classifier")
        result = _format_trigger_detail(d)
        assert result == _("Trigger: image classifier")

    def test_image_classifier_no_predictions_no_crash(self):
        d = TriggerDetail(trigger_type="image_classifier", category="portrait")
        result = _format_trigger_detail(d)
        assert "portrait" in result

    def test_unknown_trigger_type_shows_type(self):
        d = TriggerDetail(trigger_type="some_future_type")
        result = _format_trigger_detail(d)
        assert "some_future_type" in result

    def test_top_5_predictions_capped(self):
        preds = [(f"cat_{i}", (10 - i) / 10.0) for i in range(10)]
        d = TriggerDetail(trigger_type="image_classifier", category="cat_0", top_predictions=preds)
        result = _format_trigger_detail(d)
        # Only top 5 should appear; cat_5 through cat_9 should not
        for i in range(5):
            assert f"cat_{i}" in result
        for i in range(5, 10):
            assert f"cat_{i}" not in result


# ---------------------------------------------------------------------------
# Cross-session persistence via app_info_cache
# ---------------------------------------------------------------------------

class TestLastActionPersistence:
    """
    Verify that _restore_last_action_from_cache() reads the persisted action
    name from app_info_cache and resolves it to a live ClassifierAction, and
    that run_last_seek_to_trigger() uses the restored action when _last_action
    is None (i.e. fresh session).
    """

    def setup_method(self, _method):
        SeekToTriggerTab._last_action = None
        SeekToTriggerTab._last_trigger_slot.clear()

    def teardown_method(self, _method):
        SeekToTriggerTab._last_action = None
        SeekToTriggerTab._last_trigger_slot.clear()

    # -- _restore_last_action_from_cache -------------------------------------

    def test_restore_finds_classifier_action_by_name(self):
        """Persisted name resolves to the matching ClassifierAction object."""
        from tests.helpers import isolated_app_info_cache

        ca = _action(name="my_seek_action")
        old = ClassifierActionsManager.classifier_actions[:]
        ClassifierActionsManager.classifier_actions = [ca]
        try:
            isolated_app_info_cache().set_meta(_LAST_ACTION_CACHE_KEY, "my_seek_action")
            result = SeekToTriggerTab._restore_last_action_from_cache()
            assert result is ca
        finally:
            ClassifierActionsManager.classifier_actions = old

    def test_restore_finds_prevalidation_by_name(self):
        """Persisted name also searches prevalidations, not just classifier actions."""
        from tests.helpers import isolated_app_info_cache
        from compare.classifier_action import Prevalidation

        pv = Prevalidation(name="my_preval", action=ClassifierActionType.HIDE, use_embedding=False)
        old_pv = ClassifierActionsManager.prevalidations[:]
        ClassifierActionsManager.prevalidations = [pv]
        try:
            isolated_app_info_cache().set_meta(_LAST_ACTION_CACHE_KEY, "my_preval")
            result = SeekToTriggerTab._restore_last_action_from_cache()
            assert result is pv
        finally:
            ClassifierActionsManager.prevalidations = old_pv

    def test_restore_returns_none_when_name_absent_from_managers(self):
        """Returns None gracefully when the cached name no longer matches any action."""
        from tests.helpers import isolated_app_info_cache

        old = ClassifierActionsManager.classifier_actions[:]
        ClassifierActionsManager.classifier_actions = []
        try:
            isolated_app_info_cache().set_meta(_LAST_ACTION_CACHE_KEY, "deleted_action")
            result = SeekToTriggerTab._restore_last_action_from_cache()
            assert result is None
        finally:
            ClassifierActionsManager.classifier_actions = old

    def test_restore_returns_none_when_cache_empty(self):
        """Returns None when no name has ever been saved."""
        result = SeekToTriggerTab._restore_last_action_from_cache()
        assert result is None

    # -- run_last_seek_to_trigger with cache restore -------------------------

    def test_headless_run_restores_action_from_cache_on_fresh_session(self):
        """With _last_action=None, run_last_seek_to_trigger restores from cache
        and warms _last_action for subsequent in-session calls."""
        from tests.helpers import isolated_app_info_cache

        ca = _action(name="cached_action")
        old = ClassifierActionsManager.classifier_actions[:]
        ClassifierActionsManager.classifier_actions = [ca]
        try:
            isolated_app_info_cache().set_meta(_LAST_ACTION_CACHE_KEY, "cached_action")

            toasts = []

            class _FakeActions:
                def get_active_media_filepath(self):
                    return None  # no media → early exit after action is restored
                def toast(self, msg, **kw):
                    toasts.append(msg)

            SeekToTriggerTab.run_last_seek_to_trigger(_FakeActions())

            # The class-level cache should now be warmed
            assert SeekToTriggerTab._last_action is ca
            # And the toast should be about missing media, not missing action
            assert toasts, "Expected at least one toast"
            assert not any("Seek to Trigger tab" in t for t in toasts), (
                f"Got 'no previous action' toast instead of 'no media' toast: {toasts}"
            )
        finally:
            ClassifierActionsManager.classifier_actions = old

    def test_headless_run_shows_no_action_toast_when_cache_empty(self):
        """Without any persisted name, the user gets a clear prompt to use the tab."""
        toasts = []

        class _FakeActions:
            def get_active_media_filepath(self):
                return None
            def toast(self, msg, **kw):
                toasts.append(msg)

        SeekToTriggerTab.run_last_seek_to_trigger(_FakeActions())

        assert toasts, "Expected a toast"
        expected = _("No previous seek-to-trigger action — use the Seek to Trigger tab first.")
        assert any(t == expected for t in toasts), (
            f"Expected {expected!r} message, got: {toasts}"
        )

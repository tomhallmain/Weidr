"""
Integration tests for the Seek-to-Trigger feature.

Two layers of coverage:

1. MediaFrame layer — verify that video_seek_ms() on an animated GIF drives the
   MediaControlsOverlay seek slider to the expected position.  No classifiers
   involved; this is a pure media-seek assertion.

2. SeekToTriggerTab layer — verify the full pipeline:
     button click → SeekToTriggerWorker → find_first_trigger_slot (mocked)
     → slot_index_to_seek_position (stats injected) → seek_media(ms)
     → video_seek_ms(ms) → overlay slider changes.

Media: animated GIF (no VLC / ffmpeg needed, works on offscreen Qt platform).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from compare.classifier_action import ClassifierAction, TriggerDetail, TriggerFrameResult
from compare.classifier_actions_manager import ClassifierActionsManager
from image.frame_cache import FrameCache, MediaStats
from ui.app_window.media_controls_overlay import SLIDER_MAX
from utils.constants import ClassifierActionType
from utils.translations import _


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_gif_loaded(media_frame, path, qtbot) -> None:
    qtbot.waitUntil(
        lambda: (
            media_frame.media_displayed
            and media_frame.path == path
            and media_frame._gif_movie is not None
            and media_frame._gif_is_animated
        ),
        timeout=8000,
    )


def _reset_gif_to_frame_zero(media_frame) -> None:
    """Pause the movie and jump back to the beginning so tests start from a known state."""
    movie = media_frame._gif_movie
    movie.setPaused(True)
    movie.jumpToFrame(0)
    media_frame._update_gif_overlay_progress()
    QApplication.processEvents()


def _action(**kwargs) -> ClassifierAction:
    defaults = dict(
        name="test_action",
        action=ClassifierActionType.NOTIFY,
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_blacklist=False,
        use_prototype=False,
    )
    defaults.update(kwargs)
    return ClassifierAction(**defaults)


class _StubAppActions:
    """Minimal app_actions replacement for SeekToTriggerTab tests."""

    def __init__(self, media_path_fn, on_seek=None):
        self.get_active_media_filepath = media_path_fn
        self._on_seek = on_seek or (lambda ms: None)
        self.toast = lambda msg, **kw: None

    def seek_media(self, ms, pause_after=False):
        self._on_seek(ms)

    def warn(self, msg, **kw):
        pass


@contextmanager
def _patched_actions(new_list):
    """Temporarily replace ClassifierActionsManager.classifier_actions."""
    old = ClassifierActionsManager.classifier_actions[:]
    ClassifierActionsManager.classifier_actions = list(new_list)
    try:
        yield
    finally:
        ClassifierActionsManager.classifier_actions = old


# ---------------------------------------------------------------------------
# Layer 1: MediaFrame seek slider
# ---------------------------------------------------------------------------

class TestGifSeekSlider:
    """video_seek_ms() must move the overlay slider to the expected position."""

    def test_slider_advances_after_seek(self, media_frame, show_media_files, qtbot):
        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        media_frame.resize(400, 300)
        media_frame.show_media(gif_path)
        _wait_gif_loaded(media_frame, gif_path, qtbot)

        total_ms = media_frame._gif_total_duration_ms
        assert total_ms > 0, "GIF must have a positive duration"

        _reset_gif_to_frame_zero(media_frame)
        initial_slider = media_frame._controls_overlay._seek_slider.value()

        # Seek to 75% through the GIF
        target_ms = int(total_ms * 0.75)
        media_frame.video_seek_ms(target_ms)
        QApplication.processEvents()

        new_slider = media_frame._controls_overlay._seek_slider.value()

        assert new_slider > initial_slider, (
            f"Seek slider did not advance: was {initial_slider}, still {new_slider} "
            f"after seeking to {target_ms}ms / {total_ms}ms"
        )

    def test_slider_returns_to_zero_after_seek_to_start(
        self, media_frame, show_media_files, qtbot
    ):
        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        media_frame.resize(400, 300)
        media_frame.show_media(gif_path)
        _wait_gif_loaded(media_frame, gif_path, qtbot)

        total_ms = media_frame._gif_total_duration_ms

        # Seek near the end first
        media_frame._gif_movie.setPaused(True)
        media_frame.video_seek_ms(total_ms - 1)
        QApplication.processEvents()
        assert media_frame._controls_overlay._seek_slider.value() > 0, (
            "Slider should be non-zero after seeking near the end"
        )

        # Seek back to frame 0
        media_frame.video_seek_ms(0)
        QApplication.processEvents()

        slider = media_frame._controls_overlay._seek_slider.value()
        assert slider == 0, f"Slider should be 0 after seeking to start, got {slider}"

    def test_slider_fraction_matches_requested_time(
        self, media_frame, show_media_files, qtbot
    ):
        """
        Seek to exactly 50% through the GIF.  The slider fraction should be ≥ 40%
        and ≤ 60% (allowing for frame-quantisation of the 2-frame fixture).
        """
        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        media_frame.resize(400, 300)
        media_frame.show_media(gif_path)
        _wait_gif_loaded(media_frame, gif_path, qtbot)

        total_ms = media_frame._gif_total_duration_ms
        _reset_gif_to_frame_zero(media_frame)

        target_ms = total_ms // 2
        media_frame.video_seek_ms(target_ms)
        QApplication.processEvents()

        slider = media_frame._controls_overlay._seek_slider.value()
        # Slider is in [0, SLIDER_MAX=1000]; halfway ≈ 500 ± frame quantisation
        lower = int(SLIDER_MAX * 0.35)
        upper = int(SLIDER_MAX * 0.75)
        assert lower <= slider <= upper, (
            f"Slider {slider}/{SLIDER_MAX} is outside expected range "
            f"[{lower}, {upper}] after seeking to {target_ms}ms / {total_ms}ms"
        )


# ---------------------------------------------------------------------------
# Layer 2: SeekToTriggerTab end-to-end
# ---------------------------------------------------------------------------

class TestSeekToTriggerTabEndToEnd:
    """
    Full pipeline: tab → worker → find_first_trigger_slot (mocked) →
    slot_index_to_seek_position (stats injected) → seek_media(ms) →
    video_seek_ms(ms) → overlay slider moves.
    """

    def test_pipeline_calls_seek_media_with_correct_ms(
        self, media_frame, show_media_files, qtbot
    ):
        """
        The seek_media call must receive the millisecond value derived from the
        TriggerFrameResult via slot_index_to_seek_position.
        """
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        # Load GIF so the slider starts at 0
        media_frame.resize(400, 300)
        media_frame.show_media(gif_path)
        _wait_gif_loaded(media_frame, gif_path, qtbot)
        total_ms = media_frame._gif_total_duration_ms
        assert total_ms > 0
        _reset_gif_to_frame_zero(media_frame)
        initial_slider = media_frame._controls_overlay._seek_slider.value()

        # Inject FrameCache stats with controlled fps/total_items so
        # slot_index_to_seek_position computes a deterministic ms value.
        #   TOTAL_ITEMS=10, FPS=10.0, PLANNED_SLOTS=5
        #   slot_index=1 → step=10//5=2, actual=2, ms=int(2/10.0*1000)=200
        TOTAL_ITEMS = 10
        FPS = 10.0
        PLANNED_SLOTS = 5
        SLOT_INDEX = 1
        expected_ms = int((SLOT_INDEX * (TOTAL_ITEMS // PLANNED_SLOTS)) / FPS * 1000)
        assert expected_ms > 0, "expected_ms must be non-zero for this test to be meaningful"

        FrameCache.media_stats_cache[gif_path] = MediaStats(
            media_type="gif",
            total_items=TOTAL_ITEMS,
            fps=FPS,
            duration_seconds=total_ms / 1000.0,
        )

        seek_calls = []

        def _relay_seek(ms):
            seek_calls.append(ms)
            media_frame.video_seek_ms(ms)

        stub_actions = _StubAppActions(
            media_path_fn=lambda: gif_path,
            on_seek=_relay_seek,
        )

        ca = _action()
        trigger_result = TriggerFrameResult(
            slot_index=SLOT_INDEX,
            total_planned_slots=PLANNED_SLOTS,
            frame_path="/tmp/dummy_frame.jpg",
        )

        try:
            with _patched_actions([ca]):
                tab = SeekToTriggerTab(None, stub_actions)
                qtbot.addWidget(tab)
                tab.show()
                qtbot.waitExposed(tab)

                with patch.object(ca, "find_first_trigger_slot", return_value=trigger_result):
                    tab._seek_to_trigger(ca)
                    qtbot.waitUntil(
                        lambda: tab._worker is None or not tab._worker.isRunning(),
                        timeout=15000,
                    )
                    QApplication.processEvents()

            # seek_media must have been called exactly once with the expected ms
            assert len(seek_calls) == 1, (
                f"Expected exactly one seek_media call, got {len(seek_calls)}: {seek_calls}"
            )
            assert seek_calls[0] == expected_ms, (
                f"Expected seek_media({expected_ms}ms), got seek_media({seek_calls[0]}ms)"
            )

            # And the slider must have moved from its start-of-test position
            QApplication.processEvents()
            final_slider = media_frame._controls_overlay._seek_slider.value()
            assert final_slider > initial_slider, (
                f"Overlay slider did not advance: was {initial_slider}, "
                f"still {final_slider} after seek to {expected_ms}ms"
            )

        finally:
            FrameCache.media_stats_cache.pop(gif_path, None)

    def test_not_found_does_not_seek(self, show_media_files, qtbot):
        """When find_first_trigger_slot returns None, seek_media must not be called."""
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        seek_calls = []
        stub_actions = _StubAppActions(
            media_path_fn=lambda: gif_path,
            on_seek=lambda ms: seek_calls.append(ms),
        )

        ca = _action()

        with _patched_actions([ca]):
            tab = SeekToTriggerTab(None, stub_actions)
            qtbot.addWidget(tab)
            tab.show()
            qtbot.waitExposed(tab)

            with patch.object(ca, "find_first_trigger_slot", return_value=None):
                tab._seek_to_trigger(ca)
                qtbot.waitUntil(
                    lambda: tab._worker is None or not tab._worker.isRunning(),
                    timeout=15000,
                )
                QApplication.processEvents()

        assert seek_calls == [], (
            f"seek_media should not be called when no trigger found, got: {seek_calls}"
        )

    def test_next_trigger_cycling_advances_start_slot(self, show_media_files, qtbot):
        """
        Second click for the same action+path must pass start_slot = last_slot + 1
        to find_first_trigger_slot.  We verify by capturing call kwargs.
        """
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        stub_actions = _StubAppActions(media_path_fn=lambda: gif_path)

        # First result at slot 3 of 10
        first_result = TriggerFrameResult(
            slot_index=3, total_planned_slots=10, frame_path="/tmp/f.jpg"
        )
        # Second result at slot 7 of 10
        second_result = TriggerFrameResult(
            slot_index=7, total_planned_slots=10, frame_path="/tmp/f.jpg"
        )

        call_args = []

        def _find(media_path, start_slot=0, sample_ratio=None):
            call_args.append(start_slot)
            return first_result if len(call_args) == 1 else second_result

        ca = _action()

        with _patched_actions([ca]):
            tab = SeekToTriggerTab(None, stub_actions)
            qtbot.addWidget(tab)
            tab.show()
            qtbot.waitExposed(tab)

            with patch.object(ca, "find_first_trigger_slot", side_effect=_find):
                # First click
                tab._seek_to_trigger(ca)
                qtbot.waitUntil(
                    lambda: tab._worker is None or not tab._worker.isRunning(),
                    timeout=15000,
                )
                QApplication.processEvents()

                # Second click
                tab._seek_to_trigger(ca)
                qtbot.waitUntil(
                    lambda: tab._worker is None or not tab._worker.isRunning(),
                    timeout=15000,
                )
                QApplication.processEvents()

        assert len(call_args) >= 2, "find_first_trigger_slot should have been called at least twice"
        assert call_args[0] == 0, f"First click should use start_slot=0, got {call_args[0]}"
        assert call_args[1] == 4, (
            f"Second click should use start_slot=4 (last_slot+1=3+1), got {call_args[1]}"
        )

    def test_detail_line_appears_in_status_label(self, show_media_files, qtbot):
        """
        When TriggerFrameResult carries a TriggerDetail, the status label must
        contain a detail line mentioning the trigger type.
        """
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        stub_actions = _StubAppActions(media_path_fn=lambda: gif_path)

        detail = TriggerDetail(
            trigger_type="image_classifier",
            category="portrait",
            top_predictions=[("portrait", 0.92), ("person", 0.07)],
        )
        trigger_result = TriggerFrameResult(
            slot_index=0, total_planned_slots=5, frame_path="/tmp/f.jpg", detail=detail
        )

        FrameCache.media_stats_cache[gif_path] = MediaStats(
            media_type="gif", total_items=5, fps=10.0, duration_seconds=0.5
        )

        ca = _action()
        try:
            with _patched_actions([ca]):
                tab = SeekToTriggerTab(None, stub_actions)
                qtbot.addWidget(tab)
                tab.show()
                qtbot.waitExposed(tab)

                with patch.object(ca, "find_first_trigger_slot", return_value=trigger_result):
                    tab._seek_to_trigger(ca)
                    qtbot.waitUntil(
                        lambda: tab._worker is None or not tab._worker.isRunning(),
                        timeout=15000,
                    )
                    QApplication.processEvents()

            detail = tab._detail_lbl.text()
            assert detail.startswith(_("Trigger: image classifier")), (
                f"Detail label missing 'image classifier' trigger prefix: {detail!r}"
            )
            assert "portrait" in detail, (
                f"Detail label missing category 'portrait': {detail!r}"
            )
        finally:
            FrameCache.media_stats_cache.pop(gif_path, None)

    def test_density_slider_controls_sample_ratio(self, show_media_files, qtbot):
        """When checkbox is ON, sample_ratio must match the slider value."""
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        stub_actions = _StubAppActions(media_path_fn=lambda: gif_path)
        observed_ratios = []

        def _find(media_path, start_slot=0, sample_ratio=None):
            observed_ratios.append(sample_ratio)
            return None  # not found; we only care about the ratio

        ca = _action()

        with _patched_actions([ca]):
            tab = SeekToTriggerTab(None, stub_actions)
            qtbot.addWidget(tab)
            tab.show()
            qtbot.waitExposed(tab)

            # Enable the density override and set slider to 60 %
            tab._density_check.setChecked(True)
            tab._density_slider.setValue(60)
            QApplication.processEvents()

            with patch.object(ca, "find_first_trigger_slot", side_effect=_find):
                tab._seek_to_trigger(ca)
                qtbot.waitUntil(
                    lambda: tab._worker is None or not tab._worker.isRunning(),
                    timeout=15000,
                )
                QApplication.processEvents()

        assert observed_ratios, "find_first_trigger_slot was never called"
        assert abs(observed_ratios[0] - 0.60) < 0.01, (
            f"Expected sample_ratio≈0.60, got {observed_ratios[0]}"
        )

    def test_checkbox_off_passes_none_sample_ratio(self, show_media_files, qtbot):
        """When the density checkbox is unchecked (default), sample_ratio=None is passed."""
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        stub_actions = _StubAppActions(media_path_fn=lambda: gif_path)
        observed_ratios = []

        def _find(media_path, start_slot=0, sample_ratio=None):
            observed_ratios.append(sample_ratio)
            return None

        ca = _action()

        with _patched_actions([ca]):
            tab = SeekToTriggerTab(None, stub_actions)
            qtbot.addWidget(tab)
            tab.show()
            qtbot.waitExposed(tab)

            # Checkbox is unchecked by default — do NOT enable it
            assert not tab._density_check.isChecked(), "Checkbox should default to OFF"

            with patch.object(ca, "find_first_trigger_slot", side_effect=_find):
                tab._seek_to_trigger(ca)
                qtbot.waitUntil(
                    lambda: tab._worker is None or not tab._worker.isRunning(),
                    timeout=15000,
                )
                QApplication.processEvents()

        assert observed_ratios, "find_first_trigger_slot was never called"
        assert observed_ratios[0] is None, (
            f"Expected sample_ratio=None when checkbox is off, got {observed_ratios[0]}"
        )

    def test_density_warning_label_visibility(self, show_media_files, qtbot):
        """Warning label appears only when checkbox is ON and slider is at/above warn threshold."""
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        stub_actions = _StubAppActions(media_path_fn=lambda: gif_path)

        with _patched_actions([]):
            tab = SeekToTriggerTab(None, stub_actions)
            qtbot.addWidget(tab)
            tab.show()
            qtbot.waitExposed(tab)

            # Checkbox OFF — warning must stay hidden regardless of slider value
            tab._density_slider.setValue(75)
            QApplication.processEvents()
            assert not tab._density_warn_lbl.isVisible(), (
                "Warning should be hidden when checkbox is OFF, even at 75%"
            )

            # Enable the checkbox — warning should now appear at 75%
            tab._density_check.setChecked(True)
            QApplication.processEvents()
            assert tab._density_warn_lbl.isVisible(), (
                "Warning should appear when checkbox is ON and slider is at 75%"
            )

            # Drop below threshold — warning should hide again
            tab._density_slider.setValue(30)
            QApplication.processEvents()
            assert not tab._density_warn_lbl.isVisible(), (
                "Warning should be hidden at 30% even with checkbox ON"
            )

            # Back to high value — warning reappears
            tab._density_slider.setValue(100)
            QApplication.processEvents()
            assert tab._density_warn_lbl.isVisible(), (
                "Warning should be visible at 100% with checkbox ON"
            )

            # Uncheck — warning must disappear
            tab._density_check.setChecked(False)
            QApplication.processEvents()
            assert not tab._density_warn_lbl.isVisible(), (
                "Warning should hide when checkbox is unchecked"
            )

    def test_copy_path_button_writes_to_clipboard(self, show_media_files, qtbot):
        """Copy path button must write the current media path to the clipboard."""
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab

        gif_path = show_media_files.get("gif")
        if not gif_path:
            pytest.skip("Animated GIF not generated in this environment")

        stub_actions = _StubAppActions(media_path_fn=lambda: gif_path)

        with _patched_actions([]):
            tab = SeekToTriggerTab(None, stub_actions)
            qtbot.addWidget(tab)
            tab.show()
            qtbot.waitExposed(tab)

            tab._copy_path()
            QApplication.processEvents()

        clipboard_text = QApplication.clipboard().text()
        assert clipboard_text == gif_path, (
            f"Clipboard should contain {gif_path!r}, got {clipboard_text!r}"
        )

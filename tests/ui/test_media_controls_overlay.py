"""
Behaviour of MediaControlsOverlay.

Rendering is not asserted here -- it needs a real display. What is pinned is
the state machine MediaFrame drives: what the transport reports, when the bar
is allowed to appear, and which controls apply to the media in hand.

Visibility is checked with isHidden() rather than isVisible(): the overlay is
a top-level Tool window whose parent is never shown in these tests, and
isVisible() is False for an explicitly shown widget under an unmapped parent.
"""

import pytest
from PySide6.QtWidgets import QWidget

from ui.app_window.media_controls_overlay import (
    MediaControlsOverlay,
    MUTE_ICON,
    PAUSE_ICON,
    PLAY_ICON,
    SLIDER_MAX,
    UNMUTE_ICON,
    VOLUME_MAX,
    _fmt_time,
)
from utils.translations import _


@pytest.fixture
def overlay(qtbot):
    """An overlay on an unshown parent.

    The parent is kept alive for the test: it owns the overlay on the Qt side,
    so letting it go would take the widget under test with it.
    """
    parent = QWidget()
    qtbot.addWidget(parent)
    widget = MediaControlsOverlay(parent)
    yield widget
    widget.dismiss()
    widget.close()


class TestFormatTime:
    def test_under_an_hour_is_minutes_and_seconds(self):
        assert _fmt_time(0) == "0:00"
        assert _fmt_time(5_000) == "0:05"
        assert _fmt_time(65_000) == "1:05"
        assert _fmt_time(600_000) == "10:00"

    def test_an_hour_or_more_gains_an_hours_field(self):
        assert _fmt_time(3_600_000) == "1:00:00"
        assert _fmt_time(3_725_000) == "1:02:05"

    def test_a_negative_position_clamps_to_zero(self):
        """VLC can report a negative time before the first frame decodes."""
        assert _fmt_time(-1) == "0:00"


class TestProgress:
    def test_labels_and_slider_follow_the_transport(self, overlay):
        overlay.update_progress(30_000, 120_000)

        assert overlay._elapsed_label.text() == "0:30"
        assert overlay._total_label.text() == "2:00"
        assert overlay._seek_slider.value() == SLIDER_MAX // 4

    def test_a_zero_duration_leaves_the_slider_at_the_start(self, overlay):
        overlay.update_progress(0, 0)
        assert overlay._seek_slider.value() == 0

    def test_a_zero_duration_reports_no_track(self, overlay):
        """_has_track gates show_overlay(); a duration-less tick must not arm it."""
        overlay.on_track_changed()
        overlay.update_progress(0, 0)
        assert overlay._has_track is False

    def test_dragging_is_not_overwritten_by_a_progress_tick(self, overlay):
        """A tick landing mid-drag would yank the handle out from under the
        cursor."""
        overlay.update_progress(0, 120_000)
        overlay._on_slider_pressed()
        overlay._seek_slider.setValue(900)

        overlay.update_progress(1_000, 120_000)

        assert overlay._seek_slider.value() == 900


class TestPlayPauseState:
    def test_the_button_shows_pause_while_playing(self, overlay):
        overlay.on_track_changed()
        assert overlay._play_pause_btn.text() == PAUSE_ICON

    def test_the_button_shows_play_when_paused(self, overlay):
        overlay.on_track_changed()
        overlay.set_paused(True)
        assert overlay._play_pause_btn.text() == PLAY_ICON

    def test_the_button_shows_play_with_no_media(self, overlay):
        overlay.on_playback_stopped()
        assert overlay._play_pause_btn.text() == PLAY_ICON

    def test_clicking_asks_rather_than_deciding(self, overlay):
        """MediaFrame owns the transport; the overlay only reports intent."""
        received = []
        overlay.play_pause_requested.connect(lambda: received.append(True))

        overlay._on_play_pause()

        assert received == [True]
        assert overlay._is_paused is False


class TestSeeking:
    def test_releasing_the_handle_requests_the_matching_position(self, overlay):
        received = []
        overlay.seek_requested.connect(received.append)
        overlay.update_progress(0, 120_000)

        overlay._seek_slider.setValue(SLIDER_MAX // 2)
        overlay._on_slider_released()

        assert received == [60_000]

    def test_no_seek_is_requested_without_a_known_duration(self, overlay):
        received = []
        overlay.seek_requested.connect(received.append)
        overlay.update_progress(0, 0)

        overlay._seek_slider.setValue(SLIDER_MAX // 2)
        overlay._on_slider_released()

        assert received == []

    def test_dragging_while_paused_seeks_live(self, overlay):
        """Scrubbing a paused video should move the visible frame."""
        received = []
        overlay.seek_requested.connect(received.append)
        overlay.update_progress(0, 100_000)
        overlay.set_paused(True)

        overlay._on_slider_moved(SLIDER_MAX // 10)

        assert received == [10_000]

    def test_dragging_while_playing_only_previews(self, overlay):
        received = []
        overlay.seek_requested.connect(received.append)
        overlay.update_progress(0, 100_000)
        overlay.set_paused(False)

        overlay._on_slider_moved(SLIDER_MAX // 10)

        assert received == []
        assert overlay._elapsed_label.text() == "0:10"

    def test_a_drag_holds_the_bar_open(self, overlay):
        """Auto-hiding out from under an in-progress scrub would drop the drag."""
        overlay.on_track_changed()
        overlay._restart_autohide()

        overlay._on_slider_pressed()
        assert overlay._autohide_timer.isActive() is False

        overlay._on_slider_released()
        assert overlay._autohide_timer.isActive() is True

    def test_the_bar_does_not_fade_mid_drag(self, overlay):
        """A fade started by anything else must not run while the handle is held."""
        overlay.on_track_changed()
        overlay.show_overlay()
        overlay._on_slider_pressed()

        overlay._fade_out()

        assert overlay.isHidden() is False


class TestVolume:
    def test_setting_the_state_does_not_echo_back_a_change(self, overlay):
        """MediaFrame pushes its own volume in; treating that as user input
        would loop."""
        received = []
        overlay.volume_changed.connect(received.append)

        overlay.set_volume_state(40, muted=False)

        assert overlay._volume_slider.value() == 40
        assert received == []

    def test_a_volume_beyond_the_range_is_bounded(self, overlay):
        overlay.set_volume_state(500, muted=False)
        assert overlay._volume_slider.value() == VOLUME_MAX
        overlay.set_volume_state(-5, muted=False)
        assert overlay._volume_slider.value() == 0

    def test_the_mute_icon_reflects_the_state(self, overlay):
        overlay.set_volume_state(50, muted=True)
        assert overlay._mute_btn.text() == UNMUTE_ICON
        assert overlay._mute_btn.toolTip() == _("Unmute")

        overlay.set_volume_state(50, muted=False)
        assert overlay._mute_btn.text() == MUTE_ICON
        assert overlay._mute_btn.toolTip() == _("Mute")

    def test_clicking_mute_asks_rather_than_deciding(self, overlay):
        received = []
        overlay.mute_toggled.connect(lambda: received.append(True))

        overlay._on_mute_toggle()

        assert received == [True]
        assert overlay._is_muted is False


class TestControlsForTheMediaInHand:
    def test_audio_controls_can_be_hidden(self, overlay):
        """Media with no audio track -- an animated image -- has nothing for
        the volume controls to act on."""
        overlay.set_audio_controls_visible(False)

        assert overlay._mute_btn.isEnabled() is False
        assert overlay._mute_btn.isHidden() is True
        assert overlay._volume_slider.isEnabled() is False
        assert overlay._volume_slider.isHidden() is True

    def test_audio_controls_come_back(self, overlay):
        overlay.set_audio_controls_visible(False)
        overlay.set_audio_controls_visible(True)

        assert overlay._mute_btn.isEnabled() is True
        assert overlay._mute_btn.isHidden() is False
        assert overlay._volume_slider.isEnabled() is True
        assert overlay._volume_slider.isHidden() is False

    def test_a_file_without_a_seek_index_disables_seeking(self, overlay):
        """A slider that still moved would silently do nothing -- a Cues-less
        Matroska container plays but cannot seek."""
        overlay.set_no_seek_index(True)

        assert overlay._seek_slider.isEnabled() is False
        assert overlay._no_seek_label.isHidden() is False
        assert overlay._seek_slider.toolTip() == _(
            "Seeking unavailable — container has no seek index"
        )

    def test_seeking_is_restored_for_a_normal_file(self, overlay):
        overlay.set_no_seek_index(True)
        overlay.set_no_seek_index(False)

        assert overlay._seek_slider.isEnabled() is True
        assert overlay._no_seek_label.isHidden() is True
        assert overlay._seek_slider.toolTip() == ""


class TestVisibility:
    def test_it_stays_hidden_with_no_media(self, overlay):
        """Nothing is playing, so there is no transport to offer."""
        overlay.on_playback_stopped()

        overlay.show_overlay()

        assert overlay.isHidden() is True

    def test_it_appears_once_media_is_playing(self, overlay):
        overlay.on_track_changed()

        overlay.show_overlay()

        assert overlay.isHidden() is False

    def test_showing_arms_the_autohide(self, overlay):
        overlay.on_track_changed()

        overlay.show_overlay()

        assert overlay._autohide_timer.isActive() is True

    def test_dismiss_hides_without_waiting_for_the_fade(self, overlay):
        overlay.on_track_changed()
        overlay.show_overlay()

        overlay.dismiss()

        assert overlay.isHidden() is True
        assert overlay.windowOpacity() == 0.0
        assert overlay._autohide_timer.isActive() is False

    def test_stopping_playback_dismisses_it(self, overlay):
        overlay.on_track_changed()
        overlay.show_overlay()

        overlay.on_playback_stopped()

        assert overlay.isHidden() is True

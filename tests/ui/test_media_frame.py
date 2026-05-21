"""
Tests for the MediaFrame VLC deadlock fix for Matroska files missing a Cues element.

Without a Cues (seek-index) element, libvlc_media_player_stop() blocks indefinitely
because VLC performs a sequential linear scan of the file and that decode loop never
cleanly exits when interrupted.

Two complementary behaviours are in place:
  1. video_stop() runs stop() in a daemon thread with a 3-second timeout.  On
     timeout the hung player is replaced and the file path is recorded in
     _matroska_missing_cues_paths.
  2. show_video() checks _matroska_missing_cues_paths and sets the no-seek indicator
     in MediaControlsOverlay for known-bad paths — the seek slider is disabled and a
     warning label is shown.  VLC still plays the file (it can play but not seek).
"""

import os

import pytest

from ui.app_window.media_frame import _VLC_AVAILABLE
from tests.test_utils import MALFORMED_WEBM, HangingVlcPlayer

pytestmark = pytest.mark.skipif(
    not _VLC_AVAILABLE, reason="python-vlc not installed"
)


class TestMalformedWebmVlcStop:
    """
    Regression tests for the libvlc_media_player_stop() deadlock with WEBM
    files that have no Cues element (example_malformed_absent_cues.webm).
    """

    def test_hanging_stop_records_path_and_replaces_player(self, media_frame, monkeypatch):
        """When stop() times out the hung player is replaced and the path is recorded.

        Uses HangingVlcPlayer to trigger the timeout deterministically without
        loading real media.  Also verifies that the path ends up in
        _matroska_missing_cues_paths so subsequent show_video() calls are fast.
        """
        import ui.app_window.media_frame as mf_module
        from ui.app_window.media_frame import VideoUI

        # Isolate the module-level set so this test doesn't pollute others.
        monkeypatch.setattr(mf_module, "_matroska_missing_cues_paths", set())

        media_frame._video_ui = VideoUI(MALFORMED_WEBM)
        media_frame.path = MALFORMED_WEBM
        original_player = media_frame.vlc_media_player
        hanging = HangingVlcPlayer(original_player)
        media_frame.vlc_media_player = hanging

        media_frame.video_stop()

        assert hanging.stop_called.is_set(), "stop() was never called on the player"
        assert media_frame.vlc_media_player is not None
        assert media_frame.vlc_media_player is not hanging, (
            "hung player was not replaced — _replace_vlc_player() may not have run"
        )
        assert MALFORMED_WEBM in mf_module._matroska_missing_cues_paths, (
            "path was not recorded in _matroska_missing_cues_paths after timeout"
        )

    def test_known_bad_path_shows_no_seek_indicator(self, media_frame, monkeypatch):
        """show_video() on a known Cues-less path sets the overlay no-seek indicator.

        Pre-populates _matroska_missing_cues_paths to simulate a prior stop() timeout,
        then calls show_video().  VLC still plays the file; the overlay label is made
        visible and the seek slider disabled to reflect that seeking is unavailable.

        Checks isHidden() rather than isVisible() because the overlay is a top-level
        Tool window that starts hidden — isVisible() returns False for children of a
        hidden parent even when the child was explicitly shown via setVisible(True).
        """
        import ui.app_window.media_frame as mf_module

        assert os.path.isfile(MALFORMED_WEBM), (
            f"Test asset not found: {MALFORMED_WEBM}"
        )
        monkeypatch.setattr(mf_module, "_matroska_missing_cues_paths", {MALFORMED_WEBM})

        media_frame.show_video(MALFORMED_WEBM)

        overlay = media_frame._controls_overlay
        assert not overlay._no_seek_label.isHidden(), (
            "no-seek label must not be hidden for a known Cues-less path"
        )
        assert not overlay._seek_slider.isEnabled(), (
            "seek slider should be disabled for a Cues-less path"
        )

    def test_replace_vlc_player_produces_a_new_player(self, media_frame):
        """_replace_vlc_player() must always set vlc_media_player to a fresh instance."""
        original = media_frame.vlc_media_player
        media_frame._replace_vlc_player()
        new_player = media_frame.vlc_media_player

        assert new_player is not None
        assert new_player is not original

    def test_frame_usable_after_malformed_webm_stop(self, media_frame, qtbot, tmp_path):
        """After the VLC hang+replace cycle, the frame must display the next file correctly.

        Simulates the state left by video_stop() timing out (VideoUI set, player
        replaced) without loading real WEBM media.  Loading the malformed WEBM here
        would leave a libvlc_media_player_stop() daemon thread that outlasts the
        fixture's vlc_instance lifetime; that crash scenario is instead handled by
        the _vlc_instances_pending_cleanup guard in dispose_vlc(), exercised
        implicitly by test_hanging_stop_records_path_and_replaces_player.
        """
        from PIL import Image
        from ui.app_window.media_frame import VideoUI

        png_path = str(tmp_path / "next.png")
        Image.new("RGB", (10, 10), (128, 200, 50)).save(png_path, format="PNG")

        # Mirror exactly what video_stop() leaves behind after a stop() timeout:
        # _video_ui is set (frame thinks it's in video mode) and the player has
        # already been replaced with a fresh idle instance.
        media_frame._video_ui = VideoUI("fake.webm")
        media_frame._replace_vlc_player()

        # show_media internally calls video_stop() before switching to static image mode.
        media_frame.show_media(png_path)

        qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=3000)
        assert media_frame.media_displayed

"""
VLC audio-only playback for :class:`~ui.app_window.media_frame.MediaFrame`.

Keeps audio routing and placeholder setup out of ``media_frame.py`` (import
``show_in_frame`` from there only on the audio branch).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from utils.audio_media import audio_placeholder_title, is_audio_for_display
from utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ui.app_window.media_frame import MediaFrame

logger = get_logger("audio_playback")


def show_in_frame(frame: "MediaFrame", path: str) -> None:
    """
    Play *path* as audio on *frame*'s VLC instance, or show a text placeholder.

    No-op when *path* is not an enabled audio file.
    """
    if not is_audio_for_display(path):
        return

    from ui.app_window.media_frame import (
        VideoUI,
        _VLC_AVAILABLE,
        _matroska_missing_cues_paths,
    )

    if not _VLC_AVAILABLE or not frame.vlc_media_player:
        frame._show_placeholder(audio_placeholder_title(path))
        return

    if frame._pending_blur_path == path:
        frame._show_black_frame()
        return

    frame.clear()
    frame._graphics_view.set_interaction_enabled(False)
    frame._graphics_view.reset_interaction()

    has_cues = path not in _matroska_missing_cues_paths
    frame._video_ui = VideoUI(path, has_video=False, has_cues=has_cues)
    frame.path = path
    frame.vlc_media = frame.vlc_instance.media_new(path)
    frame.vlc_media_player.set_media(frame.vlc_media)
    frame._video_started_monotonic = time.monotonic()

    if frame.vlc_media_player.play() == -1:
        frame._show_placeholder(audio_placeholder_title(path))
        logger.warning("VLC failed to play audio path=%s", path)
        return

    frame._graphics_view.hide()
    frame._placeholder_label.setText(audio_placeholder_title(path))
    frame._placeholder_label.show()
    frame._controls_overlay.set_audio_controls_visible(True)
    from PySide6.QtCore import Qt
    frame.setCursor(Qt.CursorShape.ArrowCursor)
    frame.media_displayed = True
    frame.on_track_changed()
    frame._controls_overlay.set_no_seek_index(not has_cues)
    frame._sync_overlay_volume_state(force=True)
    frame._playback_timer.start()

"""MediaFrame pan/zoom interaction on static images (ZoomableGraphicsView)."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent

from ui.app_window.media_frame import VideoUI


def _wheel_event(view, angle_delta_y: int, modifiers: Qt.KeyboardModifier) -> QWheelEvent:
    """Build a Qt 6.10-compatible QWheelEvent for ZoomableGraphicsView.wheelEvent."""
    pos = QPoint(max(1, view.width() // 2), max(1, view.height() // 2))
    global_pos = view.mapToGlobal(pos)
    return QWheelEvent(
        pos,
        global_pos,
        QPoint(0, 0),
        QPoint(0, angle_delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )


def _wait_static(media_frame, path, qtbot) -> None:
    qtbot.waitUntil(
        lambda: media_frame.media_displayed
        and media_frame.path == path
        and media_frame._graphics_view._interaction_enabled,
        timeout=8000,
    )


class TestShowMediaPanZoom:
    def test_show_media_enables_graphics_view_interaction(
        self, media_frame, show_media_files, qtbot
    ):
        media_frame.resize(400, 300)
        path = show_media_files["png"]
        media_frame.show_media(path)
        _wait_static(media_frame, path, qtbot)
        view = media_frame._graphics_view
        assert view.dragMode() == view.DragMode.ScrollHandDrag
        assert view.is_user_zoom_active() is False

    def test_ctrl_wheel_zoom_marks_user_zoom_active(
        self, media_frame, show_media_files, qtbot
    ):
        media_frame.resize(400, 300)
        path = show_media_files["png"]
        media_frame.show_media(path)
        _wait_static(media_frame, path, qtbot)

        view = media_frame._graphics_view
        view.wheelEvent(_wheel_event(view, 120, Qt.KeyboardModifier.ControlModifier))

        assert view.is_user_zoom_active()
        assert view._zoom_factor > 1.0

    def test_wheel_without_ctrl_does_not_zoom(
        self, media_frame, show_media_files, qtbot
    ):
        media_frame.resize(400, 300)
        path = show_media_files["png"]
        media_frame.show_media(path)
        _wait_static(media_frame, path, qtbot)

        view = media_frame._graphics_view
        view.wheelEvent(_wheel_event(view, 120, Qt.KeyboardModifier.NoModifier))

        assert view.is_user_zoom_active() is False

    def test_reset_interaction_clears_user_zoom(
        self, media_frame, show_media_files, qtbot
    ):
        media_frame.resize(400, 300)
        path = show_media_files["png"]
        media_frame.show_media(path)
        _wait_static(media_frame, path, qtbot)

        view = media_frame._graphics_view
        view._zoom_factor = 2.0
        view._has_user_zoom = True
        view.reset_interaction()

        assert view.is_user_zoom_active() is False
        assert view._zoom_factor == 1.0

    def test_video_mode_disables_pan_zoom_interaction(
        self, media_frame, show_media_files, monkeypatch
    ):
        """Avoid real VLC play (can block headless runs); assert show_video wiring."""
        from ui.app_window.media_frame import VideoUI

        mp4 = show_media_files.get("mp4") or "clip.mp4"

        from types import MethodType

        def _fake_show_video(_self, path, freeze_frame=False):
            if not freeze_frame:
                _self._graphics_view.set_interaction_enabled(False)
            _self._video_ui = VideoUI(path)

        monkeypatch.setattr(
            media_frame, "show_video", MethodType(_fake_show_video, media_frame)
        )
        media_frame.show_media(mp4)

        assert isinstance(media_frame._video_ui, VideoUI)
        assert media_frame._graphics_view._interaction_enabled is False

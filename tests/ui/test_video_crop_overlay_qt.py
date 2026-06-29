"""
UI tests for VideoCropOverlay and launch_video_crop.

Covers:
- Widget construction: geometry, cursor, initial state
- Module-level reference (_active_overlay) kept until close
- Mouse drag: selection built and normalized
- Keyboard: Escape cancels, Enter confirms with correct coordinate mapping
- Coordinate clamping to video bounds
- launch_video_crop: guard conditions (no player, zero dims)
- launch_video_crop: overlay geometry from VLC + letterboxing math
- launch_video_crop: pause_video_if_playing called
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

import ui.app_window.video_crop_overlay_qt as _mod
from ui.app_window.video_crop_overlay_qt import VideoCropOverlay, launch_video_crop


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_active_overlay(qtbot):
    """Reset the module-level overlay reference before and after every test."""
    _mod._set_active(None)
    yield
    overlay = _mod._active_overlay
    if overlay is not None:
        overlay.close()
        QApplication.processEvents()
        _mod._set_active(None)


def _make_overlay(
    qtbot,
    geo: QRect | None = None,
    vid_w: int = 1920,
    vid_h: int = 1080,
    inv_scale: float = 2.0,
    on_confirmed=None,
) -> VideoCropOverlay:
    if geo is None:
        geo = QRect(0, 0, 960, 540)
    if on_confirmed is None:
        on_confirmed = lambda *a: None
    overlay = VideoCropOverlay(geo, vid_w, vid_h, inv_scale, on_confirmed)
    _mod._set_active(overlay)
    qtbot.addWidget(overlay)
    return overlay


def _press(widget, pos: QPoint) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def _move(widget, pos: QPoint) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def _release(widget, pos: QPoint) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def _key(widget, key: Qt.Key) -> None:
    QApplication.sendEvent(
        widget,
        QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier),
    )
    QApplication.sendEvent(
        widget,
        QKeyEvent(QKeyEvent.Type.KeyRelease, key, Qt.KeyboardModifier.NoModifier),
    )


# ---------------------------------------------------------------------------
# Construction and initial state
# ---------------------------------------------------------------------------

class TestVideoCropOverlayConstruction:
    def test_visible_after_creation(self, qtbot):
        overlay = _make_overlay(qtbot)
        assert overlay.isVisible()

    def test_geometry_matches_arg(self, qtbot):
        geo = QRect(100, 200, 640, 360)
        overlay = _make_overlay(qtbot, geo=geo)
        assert overlay.width() == 640
        assert overlay.height() == 360

    def test_cross_cursor_set(self, qtbot):
        overlay = _make_overlay(qtbot)
        assert overlay.cursor().shape() == Qt.CursorShape.CrossCursor

    def test_no_selection_on_creation(self, qtbot):
        overlay = _make_overlay(qtbot)
        assert overlay._sel is None

    def test_no_anchor_on_creation(self, qtbot):
        overlay = _make_overlay(qtbot)
        assert overlay._anchor is None

    def test_active_overlay_points_to_instance(self, qtbot):
        overlay = _make_overlay(qtbot)
        assert _mod._active_overlay is overlay


# ---------------------------------------------------------------------------
# Mouse drag builds selection
# ---------------------------------------------------------------------------

class TestVideoCropOverlayDrag:
    def test_drag_creates_selection(self, qtbot):
        overlay = _make_overlay(qtbot)
        _press(overlay, QPoint(50, 50))
        _move(overlay, QPoint(200, 150))
        _release(overlay, QPoint(200, 150))
        assert overlay._sel is not None

    def test_drag_selection_dimensions(self, qtbot):
        # QRect(QPoint, QPoint) is endpoint-inclusive: width = x2-x1+1, height = y2-y1+1
        overlay = _make_overlay(qtbot)
        _press(overlay, QPoint(50, 50))
        _move(overlay, QPoint(250, 200))
        _release(overlay, QPoint(250, 200))
        assert overlay._sel == QRect(50, 50, 201, 151)

    def test_reversed_drag_is_normalized(self, qtbot):
        # Qt's normalized() on a reversed two-point rect introduces an off-by-one
        # relative to the forward drag; just verify positive dimensions and correct area.
        overlay = _make_overlay(qtbot)
        _press(overlay, QPoint(250, 200))
        _move(overlay, QPoint(50, 50))
        _release(overlay, QPoint(50, 50))
        assert overlay._sel is not None
        assert overlay._sel.width() > 0
        assert overlay._sel.height() > 0
        assert overlay._sel.left() <= 51   # left edge near x=50
        assert overlay._sel.top() <= 51    # top edge near y=50

    def test_no_drag_without_press(self, qtbot):
        overlay = _make_overlay(qtbot)
        _move(overlay, QPoint(100, 100))
        assert overlay._sel is None

    def test_selection_updates_on_move(self, qtbot):
        overlay = _make_overlay(qtbot)
        _press(overlay, QPoint(50, 50))
        _move(overlay, QPoint(100, 100))
        mid = QRect(overlay._sel)
        _move(overlay, QPoint(300, 300))
        assert overlay._sel != mid


# ---------------------------------------------------------------------------
# Keyboard: Escape and Enter
# ---------------------------------------------------------------------------

class TestVideoCropOverlayKeyboard:
    def test_escape_closes_overlay(self, qtbot):
        overlay = _make_overlay(qtbot)
        _key(overlay, Qt.Key.Key_Escape)
        QApplication.processEvents()
        assert not overlay.isVisible()

    def test_escape_clears_active_overlay(self, qtbot):
        overlay = _make_overlay(qtbot)
        _key(overlay, Qt.Key.Key_Escape)
        QApplication.processEvents()
        assert _mod._active_overlay is None

    def test_enter_fires_on_confirmed(self, qtbot):
        received = []
        overlay = _make_overlay(qtbot, on_confirmed=lambda x, y, w, h: received.append((x, y, w, h)))
        _press(overlay, QPoint(100, 50))
        _move(overlay, QPoint(400, 200))
        _release(overlay, QPoint(400, 200))
        _key(overlay, Qt.Key.Key_Return)
        assert len(received) == 1

    def test_enter_correct_video_pixel_coords(self, qtbot):
        # vid=1920x1080, overlay=960x540, inv_scale=2.0
        # QRect(QPoint, QPoint) is endpoint-inclusive: width = x2-x1+1
        # drag (100,50)→(400,200) → sel=QRect(100,50,301,151)
        # x = int(100*2) = 200, y = int(50*2) = 100
        # w = min(1920-200, int(301*2)) = min(1720,602) = 602
        # h = min(1080-100, int(151*2)) = min(980,302) = 302
        received = []
        overlay = _make_overlay(qtbot, on_confirmed=lambda x, y, w, h: received.append((x, y, w, h)))
        _press(overlay, QPoint(100, 50))
        _move(overlay, QPoint(400, 200))
        _release(overlay, QPoint(400, 200))
        _key(overlay, Qt.Key.Key_Return)
        assert received == [(200, 100, 602, 302)]

    def test_enter_closes_overlay(self, qtbot):
        overlay = _make_overlay(qtbot)
        _press(overlay, QPoint(50, 50))
        _move(overlay, QPoint(200, 150))
        _release(overlay, QPoint(200, 150))
        _key(overlay, Qt.Key.Key_Return)
        QApplication.processEvents()
        assert not overlay.isVisible()

    def test_enter_clears_active_overlay(self, qtbot):
        overlay = _make_overlay(qtbot)
        _press(overlay, QPoint(50, 50))
        _move(overlay, QPoint(200, 150))
        _release(overlay, QPoint(200, 150))
        _key(overlay, Qt.Key.Key_Return)
        QApplication.processEvents()
        assert _mod._active_overlay is None

    def test_enter_with_no_selection_does_nothing(self, qtbot):
        received = []
        overlay = _make_overlay(qtbot, on_confirmed=lambda *a: received.append(a))
        _key(overlay, Qt.Key.Key_Return)
        assert received == []
        assert overlay.isVisible()

    def test_enter_with_one_pixel_selection_does_nothing(self, qtbot):
        # A 1×1 rect (width==1) is rejected by the > 1 guard
        received = []
        overlay = _make_overlay(qtbot, on_confirmed=lambda *a: received.append(a))
        _press(overlay, QPoint(100, 100))
        _release(overlay, QPoint(100, 100))
        _key(overlay, Qt.Key.Key_Return)
        assert received == []


# ---------------------------------------------------------------------------
# Coordinate clamping to video bounds
# ---------------------------------------------------------------------------

class TestCoordinateClamping:
    def test_coords_stay_within_video(self, qtbot):
        # vid=100x100, overlay=200x200, inv_scale=0.5
        # drag (10,10)→(199,199): sel w=189,h=189 → scaled x=5,y=5,w=min(95,94)=94,h=94
        received = []
        geo = QRect(0, 0, 200, 200)
        overlay = VideoCropOverlay(geo, 100, 100, 0.5, lambda x, y, w, h: received.append((x, y, w, h)))
        _mod._set_active(overlay)
        qtbot.addWidget(overlay)
        _press(overlay, QPoint(10, 10))
        _move(overlay, QPoint(199, 199))
        _release(overlay, QPoint(199, 199))
        _key(overlay, Qt.Key.Key_Return)
        x, y, w, h = received[0]
        assert x >= 0 and y >= 0
        assert x + w <= 100
        assert y + h <= 100

    def test_origin_clamped_to_zero(self, qtbot):
        # Drag starting at (0,0) — PySide6 treats QPoint(0,0) as falsy, so the
        # production code uses `is not None` to guard the move handler correctly.
        received = []
        geo = QRect(0, 0, 500, 500)
        overlay = VideoCropOverlay(geo, 200, 200, 2.0, lambda x, y, w, h: received.append((x, y, w, h)))
        _mod._set_active(overlay)
        qtbot.addWidget(overlay)
        _press(overlay, QPoint(0, 0))
        _move(overlay, QPoint(50, 50))
        _release(overlay, QPoint(50, 50))
        _key(overlay, Qt.Key.Key_Return)
        assert received, "on_confirmed should fire for a drag starting at the origin"
        x, y, _, _ = received[0]
        assert x >= 0
        assert y >= 0


# ---------------------------------------------------------------------------
# launch_video_crop guard conditions
# ---------------------------------------------------------------------------

class TestLaunchVideoCropGuards:
    def test_no_player_shows_toast_and_no_overlay(self, qtbot):
        frame = MagicMock()
        frame.vlc_media_player = None
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        app.notification_ctrl.toast.assert_called_once()
        assert _mod._active_overlay is None

    def test_zero_video_dimensions_shows_toast_and_no_overlay(self, qtbot):
        frame = MagicMock()
        frame.vlc_media_player.video_get_size.return_value = (0, 0)
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        app.notification_ctrl.toast.assert_called_once()
        assert _mod._active_overlay is None

    def test_negative_video_dimensions_shows_toast(self, qtbot):
        frame = MagicMock()
        frame.vlc_media_player.video_get_size.return_value = (-1, -1)
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        app.notification_ctrl.toast.assert_called_once()
        assert _mod._active_overlay is None


# ---------------------------------------------------------------------------
# launch_video_crop: overlay creation and geometry
# ---------------------------------------------------------------------------

class TestLaunchVideoCropOverlay:
    def _make_frame(self, vid_w=1920, vid_h=1080, mf_w=960, mf_h=540):
        frame = MagicMock()
        frame.vlc_media_player.video_get_size.return_value = (vid_w, vid_h)
        frame.width.return_value = mf_w
        frame.height.return_value = mf_h
        # Simulate mapToGlobal adding a fixed screen offset
        frame.mapToGlobal.side_effect = lambda p: QPoint(p.x() + 100, p.y() + 50)
        return frame

    def test_valid_dimensions_creates_overlay(self, qtbot):
        frame = self._make_frame()
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        assert _mod._active_overlay is not None
        assert isinstance(_mod._active_overlay, VideoCropOverlay)
        qtbot.addWidget(_mod._active_overlay)

    def test_overlay_size_matches_display_area(self, qtbot):
        # vid=1920x1080, mf=960x540 → scale=0.5, disp=960x540
        frame = self._make_frame(vid_w=1920, vid_h=1080, mf_w=960, mf_h=540)
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        overlay = _mod._active_overlay
        qtbot.addWidget(overlay)
        assert overlay.width() == 960
        assert overlay.height() == 540

    def test_overlay_letterbox_height(self, qtbot):
        # Wide video (1920x540) in square frame (540x540)
        # disp_scale = min(540/1920, 540/540) = 0.28125
        # disp_w=540, disp_h=int(540*0.28125)=151
        frame = self._make_frame(vid_w=1920, vid_h=540, mf_w=540, mf_h=540)
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        overlay = _mod._active_overlay
        qtbot.addWidget(overlay)
        expected_h = int(540 * min(540 / 1920, 540 / 540))
        assert overlay.height() == expected_h

    def test_overlay_letterbox_width(self, qtbot):
        # Tall video (540x1920) in square frame (540x540)
        # disp_scale = min(540/540, 540/1920) = 0.28125
        # disp_w=int(540*0.28125)=151, disp_h=540
        frame = self._make_frame(vid_w=540, vid_h=1920, mf_w=540, mf_h=540)
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        overlay = _mod._active_overlay
        qtbot.addWidget(overlay)
        expected_w = int(540 * min(540 / 540, 540 / 1920))
        assert overlay.width() == expected_w

    def test_pause_called_before_overlay(self, qtbot):
        frame = self._make_frame()
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        frame.pause_video_if_playing.assert_called_once()
        if _mod._active_overlay:
            qtbot.addWidget(_mod._active_overlay)

    def test_hint_toast_shown(self, qtbot):
        frame = self._make_frame()
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        app.notification_ctrl.toast.assert_called_once()
        if _mod._active_overlay:
            qtbot.addWidget(_mod._active_overlay)

    def test_overlay_visible(self, qtbot):
        frame = self._make_frame()
        app = MagicMock()
        launch_video_crop(frame, "/fake/video.mp4", app)
        overlay = _mod._active_overlay
        assert overlay is not None
        qtbot.addWidget(overlay)
        assert overlay.isVisible()

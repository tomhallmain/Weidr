"""
Unit tests for ZoomableGraphicsView freeform polygon-selection mode.

Covers:
- Entering / exiting polygon mode (cursor, drag mode, state)
- Points added on click; preview overlay follows the cursor
- Closing by clicking back near the first point
- Keyboard: Enter closes (3+ points), Escape cancels, Backspace undoes a point
- Wheel events suppressed during polygon mode
- Mutual exclusivity with crop mode is not asserted here (callers are expected
  not to start both at once; see window_launcher.py's shared teardown).
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QGraphicsScene

from lib.zoomable_graphics_view_qt import ZoomableGraphicsView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_view(qtbot) -> ZoomableGraphicsView:
    view = ZoomableGraphicsView()
    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 800, 600)
    view.setScene(scene)
    view.resize(800, 600)
    qtbot.addWidget(view)
    view.show()
    return view


def _click(view: ZoomableGraphicsView, pos: QPoint) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view.viewport(), event)


def _move(view: ZoomableGraphicsView, pos: QPoint) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view.viewport(), event)


# ---------------------------------------------------------------------------
# Polygon mode enter / exit
# ---------------------------------------------------------------------------

class TestPolygonModeActivation:
    def test_start_polygon_mode_sets_cross_cursor(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        assert view.cursor().shape() == Qt.CursorShape.CrossCursor

    def test_start_polygon_mode_disables_drag(self, qtbot):
        view = _make_view(qtbot)
        view.set_interaction_enabled(True)
        view.start_polygon_mode()
        assert view.dragMode() == ZoomableGraphicsView.DragMode.NoDrag

    def test_end_polygon_mode_restores_drag_mode(self, qtbot):
        view = _make_view(qtbot)
        view.set_interaction_enabled(True)
        prior = view.dragMode()
        view.start_polygon_mode()
        view.end_polygon_mode()
        assert view.dragMode() == prior

    def test_end_polygon_mode_clears_points(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(10, 10))
        _click(view, QPoint(100, 10))
        _click(view, QPoint(100, 100))
        view.end_polygon_mode()
        assert view._polygon_points == []

    def test_end_polygon_mode_removes_overlay_item(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(10, 10))
        _click(view, QPoint(100, 10))
        scene_items_before = len(view.scene().items())
        view.end_polygon_mode()
        assert len(view.scene().items()) < scene_items_before

    def test_double_start_is_idempotent(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        view.start_polygon_mode()
        assert view._polygon_mode is True

    def test_double_end_is_safe(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        view.end_polygon_mode()
        view.end_polygon_mode()  # should not raise


# ---------------------------------------------------------------------------
# Points built from clicks
# ---------------------------------------------------------------------------

class TestPolygonPointsFromClicks:
    def test_click_adds_a_point(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        assert len(view._polygon_points) == 1

    def test_multiple_clicks_accumulate_points(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        _click(view, QPoint(150, 150))
        assert len(view._polygon_points) == 3

    def test_move_updates_preview_without_adding_point(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        _move(view, QPoint(300, 300))
        assert len(view._polygon_points) == 1

    def test_overlay_item_created_after_first_click(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        assert view._polygon_path_item is None
        _click(view, QPoint(50, 50))
        assert view._polygon_path_item is not None

    def test_no_points_before_polygon_mode(self, qtbot):
        view = _make_view(qtbot)
        assert view._polygon_points == []


# ---------------------------------------------------------------------------
# Closing: click back near the start point
# ---------------------------------------------------------------------------

class TestPolygonCloseByClick:
    def test_click_near_start_with_3_points_emits_confirmed(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        _click(view, QPoint(150, 150))
        _click(view, QPoint(52, 52))  # within close-radius of (50, 50)
        assert len(confirmed) == 1
        assert len(confirmed[0]) == 3

    def test_click_near_start_does_not_add_a_4th_point(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        _click(view, QPoint(150, 150))
        _click(view, QPoint(52, 52))
        # Mode exits on confirm being handled by the caller, but the point list
        # itself must reflect exactly the 3 placed points, not a spurious 4th.
        assert len(view._polygon_points) == 3

    def test_click_far_from_start_adds_a_4th_point_instead_of_closing(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        _click(view, QPoint(150, 150))
        _click(view, QPoint(400, 400))  # far from (50, 50)
        assert len(confirmed) == 0
        assert len(view._polygon_points) == 4

    def test_click_near_start_with_fewer_than_3_points_does_not_close(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        _click(view, QPoint(52, 52))  # would be "near start" but only 2 points total
        assert len(confirmed) == 0
        assert len(view._polygon_points) == 2


# ---------------------------------------------------------------------------
# Keyboard: Enter closes, Escape cancels, Backspace undoes
# ---------------------------------------------------------------------------

class TestPolygonKeyboard:
    def test_enter_with_3_points_emits_confirmed(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        _click(view, QPoint(150, 150))
        qtbot.keyPress(view, Qt.Key.Key_Return)
        assert len(confirmed) == 1
        assert len(confirmed[0]) == 3

    def test_enter_with_fewer_than_3_points_does_not_emit(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        qtbot.keyPress(view, Qt.Key.Key_Return)
        assert len(confirmed) == 0

    def test_escape_exits_polygon_mode(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        qtbot.keyPress(view, Qt.Key.Key_Escape)
        assert not view._polygon_mode

    def test_escape_emits_crop_cancelled(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        cancelled = []
        view.crop_cancelled.connect(lambda: cancelled.append(True))
        _click(view, QPoint(50, 50))
        qtbot.keyPress(view, Qt.Key.Key_Escape)
        assert len(cancelled) == 1

    def test_backspace_removes_last_point(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        qtbot.keyPress(view, Qt.Key.Key_Backspace)
        assert len(view._polygon_points) == 1

    def test_backspace_to_zero_points_removes_overlay(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        qtbot.keyPress(view, Qt.Key.Key_Backspace)
        assert view._polygon_points == []
        assert view._polygon_path_item is None

    def test_backspace_with_no_points_is_safe(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        qtbot.keyPress(view, Qt.Key.Key_Backspace)  # should not raise
        assert view._polygon_points == []


# ---------------------------------------------------------------------------
# Wheel suppressed during polygon mode
# ---------------------------------------------------------------------------

class TestWheelDuringPolygon:
    def test_wheel_does_not_zoom_during_polygon_mode(self, qtbot):
        view = _make_view(qtbot)
        view.set_interaction_enabled(True)
        view.start_polygon_mode()
        zoom_before = view._zoom_factor
        wheel = QWheelEvent(
            QPointF(400, 300),
            QPointF(400, 300),
            QPoint(0, 120),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(view.viewport(), wheel)
        assert view._zoom_factor == zoom_before

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
    """Hover move with no button held (preview only, never adds a sweep point)."""
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view.viewport(), event)


def _drag_move(view: ZoomableGraphicsView, pos: QPoint) -> None:
    """Mouse-move with the left button held (simulates an active drag/sweep)."""
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view.viewport(), event)


def _release(view: ZoomableGraphicsView, pos: QPoint) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.LeftButton,
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


# ---------------------------------------------------------------------------
# Start-point marker: visible affordance + highlight when cursor is in range
# ---------------------------------------------------------------------------

class TestPolygonStartMarker:
    def test_no_marker_before_first_click(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        assert view._polygon_start_marker_item is None

    def test_marker_created_after_first_click(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        assert view._polygon_start_marker_item is not None

    def test_marker_removed_on_end_polygon_mode(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        view.end_polygon_mode()
        assert view._polygon_start_marker_item is None

    def test_marker_position_tracks_first_point_only(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        pos_after_first = view._polygon_start_marker_item.pos()
        _click(view, QPoint(150, 50))  # placing a 2nd point must not move the marker
        pos_after_second = view._polygon_start_marker_item.pos()
        assert pos_after_first == pos_after_second

    def test_marker_highlights_when_cursor_near_start(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        _click(view, QPoint(150, 150))
        far_color = view._polygon_start_marker_item.brush().color()
        _move(view, QPoint(52, 52))  # hover within the closing tolerance
        near_color = view._polygon_start_marker_item.brush().color()
        assert near_color != far_color

    def test_marker_reverts_when_cursor_moves_away_again(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        _click(view, QPoint(150, 50))
        _click(view, QPoint(150, 150))
        far_color = view._polygon_start_marker_item.brush().color()
        _move(view, QPoint(52, 52))
        _move(view, QPoint(300, 300))
        assert view._polygon_start_marker_item.brush().color() == far_color

    def test_marker_not_highlighted_with_fewer_than_3_points(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(50, 50))
        far_color = view._polygon_start_marker_item.brush().color()
        _move(view, QPoint(52, 52))  # "near" the only point, but too few points to close
        assert view._polygon_start_marker_item.brush().color() == far_color


# ---------------------------------------------------------------------------
# Continuous "sweep" drawing: click-and-drag adds points along the path
# ---------------------------------------------------------------------------

class TestPolygonSweep:
    def test_drag_with_button_held_adds_multiple_points(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(10, 10))
        _drag_move(view, QPoint(30, 10))
        _drag_move(view, QPoint(60, 10))
        _drag_move(view, QPoint(90, 10))
        assert len(view._polygon_points) == 4

    def test_sweep_respects_minimum_step_distance(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(10, 10))
        # Sub-threshold moves (< _polygon_sweep_min_step_px) must not add points.
        _drag_move(view, QPoint(11, 10))
        _drag_move(view, QPoint(12, 10))
        _drag_move(view, QPoint(13, 10))
        assert len(view._polygon_points) == 1

    def test_move_without_button_held_does_not_add_points(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        _click(view, QPoint(10, 10))
        _move(view, QPoint(200, 200))  # hover only, no button held
        assert len(view._polygon_points) == 1

    def test_release_does_not_close_or_confirm_polygon(self, qtbot):
        """Sweeping is just an efficient way to place points; closing always
        requires an explicit click-near-start or Enter, never automatic."""
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(10, 10))
        _drag_move(view, QPoint(60, 10))
        _drag_move(view, QPoint(60, 60))
        _drag_move(view, QPoint(10, 60))
        _release(view, QPoint(10, 60))
        assert confirmed == []
        assert view._polygon_mode is True

    def test_points_after_sweep_can_still_be_closed_by_click(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        _drag_move(view, QPoint(150, 50))
        _drag_move(view, QPoint(150, 150))
        _release(view, QPoint(150, 150))
        _click(view, QPoint(52, 52))  # back near the very first point
        assert len(confirmed) == 1

    def test_release_near_start_closes_sweep(self, qtbot):
        """Regression: sweeping the cursor back to the origin and releasing
        there must auto-close, the same as a plain click-near-start already
        does. Previously the sweep itself never checked proximity to the start
        (it only accumulates points on move), and release did nothing at all,
        so a sweep that ended near the origin had no way to close."""
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        _drag_move(view, QPoint(150, 50))
        _drag_move(view, QPoint(150, 150))
        _drag_move(view, QPoint(52, 52))  # sweep back near the start point
        _release(view, QPoint(52, 52))
        # The view only notifies via the signal; ending the mode is the
        # caller's job (window_launcher.py's _on_confirmed calls
        # end_polygon_mode() when it receives this signal), matching how
        # crop_confirmed already works for the rectangle-selection mode.
        assert len(confirmed) == 1

    def test_release_near_start_with_fewer_than_3_points_does_not_close(self, qtbot):
        view = _make_view(qtbot)
        view.start_polygon_mode()
        confirmed = []
        view.polygon_confirmed.connect(confirmed.append)
        _click(view, QPoint(50, 50))
        # 7px: past the 6px sweep-add threshold (so a 2nd point is placed) and
        # still within the 10px close radius -- but only 2 points total, which
        # is too few to have a meaningful interior to close.
        _drag_move(view, QPoint(57, 50))
        _release(view, QPoint(57, 50))
        assert len(view._polygon_points) == 2
        assert confirmed == []
        assert view._polygon_mode is True

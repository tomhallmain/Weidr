"""
Unit tests for ZoomableGraphicsView crop-selection mode.

Covers:
- Entering / exiting crop mode (cursor, drag mode, state)
- Overlay cleared on end_crop_mode
- Crop rect built correctly from simulated mouse events
- Keyboard: Escape cancels, Enter confirms
- Wheel events suppressed during crop
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
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


def _press(view: ZoomableGraphicsView, pos: QPoint) -> None:
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
# Crop mode enter / exit
# ---------------------------------------------------------------------------

class TestCropModeActivation:
    def test_start_crop_mode_sets_cross_cursor(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        assert view.cursor().shape() == Qt.CursorShape.CrossCursor

    def test_start_crop_mode_disables_drag(self, qtbot):
        view = _make_view(qtbot)
        view.set_interaction_enabled(True)
        view.start_crop_mode()
        assert view.dragMode() == ZoomableGraphicsView.DragMode.NoDrag

    def test_end_crop_mode_restores_drag_mode(self, qtbot):
        view = _make_view(qtbot)
        view.set_interaction_enabled(True)
        prior = view.dragMode()
        view.start_crop_mode()
        view.end_crop_mode()
        assert view.dragMode() == prior

    def test_end_crop_mode_clears_rect(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        _press(view, QPoint(10, 10))
        _move(view, QPoint(100, 100))
        _release(view, QPoint(100, 100))
        view.end_crop_mode()
        assert view.get_crop_rect_scene() is None

    def test_end_crop_mode_removes_overlay_item(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        _press(view, QPoint(10, 10))
        _move(view, QPoint(100, 100))
        _release(view, QPoint(100, 100))
        scene_items_before = len(view.scene().items())
        view.end_crop_mode()
        assert len(view.scene().items()) < scene_items_before

    def test_double_start_is_idempotent(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        view.start_crop_mode()
        assert view._crop_mode is True

    def test_double_end_is_safe(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        view.end_crop_mode()
        view.end_crop_mode()  # should not raise


# ---------------------------------------------------------------------------
# Rect built from simulated drag
# ---------------------------------------------------------------------------

class TestCropRectFromDrag:
    def test_crop_rect_returned_after_drag(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        _press(view, QPoint(50, 50))
        _move(view, QPoint(200, 150))
        _release(view, QPoint(200, 150))
        rect = view.get_crop_rect_scene()
        assert rect is not None
        assert rect.width() > 0
        assert rect.height() > 0

    def test_rect_is_normalized_for_reversed_drag(self, qtbot):
        """Dragging right-to-left should still yield a positive-size rect."""
        view = _make_view(qtbot)
        view.start_crop_mode()
        _press(view, QPoint(200, 200))
        _move(view, QPoint(50, 50))
        _release(view, QPoint(50, 50))
        rect = view.get_crop_rect_scene()
        assert rect is not None
        assert rect.width() > 0
        assert rect.height() > 0

    def test_no_rect_before_crop_mode(self, qtbot):
        view = _make_view(qtbot)
        assert view.get_crop_rect_scene() is None

    def test_signal_emitted_on_release(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        emitted = []
        view.crop_selection_ready.connect(emitted.append)
        _press(view, QPoint(10, 10))
        _move(view, QPoint(100, 100))
        _release(view, QPoint(100, 100))
        assert len(emitted) == 1
        assert isinstance(emitted[0], QRectF)


# ---------------------------------------------------------------------------
# Keyboard: Escape and Enter
# ---------------------------------------------------------------------------

class TestCropKeyboard:
    def test_escape_exits_crop_mode(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        qtbot.keyPress(view, Qt.Key.Key_Escape)
        assert not view._crop_mode

    def test_escape_clears_rect(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        _press(view, QPoint(10, 10))
        _move(view, QPoint(100, 100))
        _release(view, QPoint(100, 100))
        qtbot.keyPress(view, Qt.Key.Key_Escape)
        assert view.get_crop_rect_scene() is None

    def test_enter_emits_crop_confirmed(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        confirmed = []
        view.crop_confirmed.connect(confirmed.append)
        _press(view, QPoint(10, 10))
        _move(view, QPoint(200, 150))
        _release(view, QPoint(200, 150))
        qtbot.keyPress(view, Qt.Key.Key_Return)
        assert len(confirmed) == 1

    def test_enter_with_no_selection_does_not_emit(self, qtbot):
        view = _make_view(qtbot)
        view.start_crop_mode()
        confirmed = []
        view.crop_confirmed.connect(confirmed.append)
        qtbot.keyPress(view, Qt.Key.Key_Return)
        assert len(confirmed) == 0


# ---------------------------------------------------------------------------
# Wheel suppressed during crop
# ---------------------------------------------------------------------------

class TestWheelDuringCrop:
    def test_wheel_does_not_zoom_during_crop(self, qtbot):
        view = _make_view(qtbot)
        view.set_interaction_enabled(True)
        view.start_crop_mode()
        zoom_before = view._zoom_factor
        # Simulate a wheel event on the viewport
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

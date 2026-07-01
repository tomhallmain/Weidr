"""
QGraphicsView with Ctrl+wheel zoom, drag-pan, and crop-selection mode for
static image display.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsView,
)


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView wrapper that provides modular wheel-zoom, drag-pan, and two
    mutually-exclusive interactive selection modes: a rubber-band rectangle
    ("crop" mode) and a click-to-add-points freeform polygon ("polygon" mode)."""

    # Emitted when the user releases the mouse after drawing a rectangle selection.
    crop_selection_ready = Signal(QRectF)
    # Emitted when the user presses Enter/Return to confirm a rectangle selection.
    crop_confirmed = Signal(QRectF)
    # Emitted when the user presses Escape to cancel a rectangle or polygon
    # selection without confirming.
    crop_cancelled = Signal()
    # Emitted when the user closes a freeform polygon selection (click back near
    # the start point, or Enter once at least 3 points are placed). Payload is
    # a list of QPointF in scene coordinates.
    polygon_confirmed = Signal(list)

    def __init__(
        self,
        parent=None,
        *,
        zoom_step: float = 1.15,
        min_zoom: float = 0.05,
        max_zoom: float = 20.0,
        zoom_modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.ControlModifier,
    ):
        super().__init__(parent)
        self._zoom_step = float(zoom_step)
        self._min_zoom = float(min_zoom)
        self._max_zoom = float(max_zoom)
        self._zoom_modifier = zoom_modifier
        self._zoom_factor = 1.0
        self._has_user_zoom = False
        self._interaction_enabled = False

        # Crop-selection state
        self._crop_mode = False
        self._crop_anchor = None          # QPointF — scene-coord anchor on press
        self._crop_rect_item: QGraphicsRectItem | None = None
        self._crop_rect_scene: QRectF | None = None
        self._pre_crop_drag_mode = QGraphicsView.DragMode.NoDrag

        # Freeform polygon-selection state
        self._polygon_mode = False
        self._polygon_points: list[QPointF] = []  # scene-coord points placed so far
        self._polygon_path_item: QGraphicsPathItem | None = None
        self._polygon_start_marker_item: QGraphicsEllipseItem | None = None
        self._polygon_close_radius_px = 10  # view-pixel radius for "click near start to close"
        self._polygon_sweep_min_step_px = 6  # min view-pixel spacing between sweep-added points

        # Freeform polygon mode needs hover-move events (preview line + start-marker
        # highlight) even when no button is held between clicks -- Qt widgets don't
        # deliver those without tracking enabled.
        self.setMouseTracking(True)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    # ------------------------------------------------------------------
    # Pan / zoom interaction
    # ------------------------------------------------------------------

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)
        if not self._crop_mode and not self._polygon_mode:
            if self._interaction_enabled:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            else:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def reset_interaction(self, *, reset_transform: bool = True) -> None:
        if reset_transform:
            self.resetTransform()
        self._zoom_factor = 1.0
        self._has_user_zoom = False

    def fit_item(
        self,
        item: QGraphicsPixmapItem | None,
        aspect_mode=Qt.AspectRatioMode.KeepAspectRatio,
    ) -> None:
        if item is None:
            return
        pix = item.pixmap()
        if pix.isNull():
            return
        self.resetTransform()
        self.fitInView(item, aspect_mode)
        self._zoom_factor = 1.0
        self._has_user_zoom = False

    def is_user_zoom_active(self) -> bool:
        return self._has_user_zoom

    # ------------------------------------------------------------------
    # Crop-selection mode
    # ------------------------------------------------------------------

    def start_crop_mode(self) -> None:
        """Enter crop-selection mode. Disables pan/zoom until cancelled or confirmed."""
        if self._crop_mode:
            return
        self._crop_mode = True
        self._pre_crop_drag_mode = self.dragMode()
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()

    def end_crop_mode(self) -> None:
        """Exit crop-selection mode, clear the overlay, and restore prior drag mode."""
        if not self._crop_mode:
            return
        self._crop_mode = False
        self._crop_anchor = None
        self._crop_rect_scene = None
        self._remove_crop_overlay()
        self.setDragMode(self._pre_crop_drag_mode)
        self.unsetCursor()

    def get_crop_rect_scene(self) -> QRectF | None:
        """Return the current selection rect in scene coordinates, or None."""
        return self._crop_rect_scene

    def _update_crop_overlay(self) -> None:
        scene = self.scene()
        if scene is None or self._crop_rect_scene is None:
            return
        if self._crop_rect_item is None:
            pen = QPen(QColor(255, 255, 255), 1)
            pen.setCosmetic(True)  # 1 px regardless of zoom level
            brush = QBrush(QColor(100, 150, 255, 70))
            item = QGraphicsRectItem()
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(10)
            scene.addItem(item)
            self._crop_rect_item = item
        try:
            self._crop_rect_item.setRect(self._crop_rect_scene)
        except RuntimeError:
            # C++ object was deleted externally (e.g. scene.clear()) — recreate next move
            self._crop_rect_item = None

    def _remove_crop_overlay(self) -> None:
        if self._crop_rect_item is not None:
            scene = self.scene()
            if scene is not None:
                try:
                    scene.removeItem(self._crop_rect_item)
                except RuntimeError:
                    pass
            self._crop_rect_item = None

    # ------------------------------------------------------------------
    # Freeform polygon-selection mode
    # ------------------------------------------------------------------

    def start_polygon_mode(self) -> None:
        """Enter freeform polygon-selection mode: click to add points (or click
        and drag to sweep a continuous run of points), then close and confirm
        by clicking near the highlighted start marker, releasing a sweep near
        it, or pressing Enter with 3+ points placed. Backspace undoes the last
        point, Escape cancels. Disables pan/zoom meanwhile."""
        if self._polygon_mode:
            return
        self._polygon_mode = True
        self._polygon_points = []
        self._pre_crop_drag_mode = self.dragMode()
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()

    def end_polygon_mode(self) -> None:
        """Exit polygon-selection mode, clear the overlay, and restore prior drag mode."""
        if not self._polygon_mode:
            return
        self._polygon_mode = False
        self._polygon_points = []
        self._remove_polygon_overlay()
        self.setDragMode(self._pre_crop_drag_mode)
        self.unsetCursor()

    def _is_near_polygon_start(self, scene_point: QPointF) -> bool:
        """True if *scene_point* is within the closing tolerance of the first
        placed point, measured in view (screen) pixels so it stays a constant
        target size regardless of zoom level. Requires 3+ points already placed
        (a 1- or 2-point "polygon" has no meaningful interior to close)."""
        if len(self._polygon_points) < 3:
            return False
        pos_view = self.mapFromScene(scene_point)
        first_view = self.mapFromScene(self._polygon_points[0])
        dx = pos_view.x() - first_view.x()
        dy = pos_view.y() - first_view.y()
        return (dx * dx + dy * dy) ** 0.5 <= self._polygon_close_radius_px

    def _update_polygon_overlay(self, preview_point: QPointF | None = None) -> None:
        scene = self.scene()
        if scene is None or not self._polygon_points:
            return
        path = QPainterPath(self._polygon_points[0])
        for pt in self._polygon_points[1:]:
            path.lineTo(pt)
        if preview_point is not None:
            path.lineTo(preview_point)
        # Close the path visually (line back to the first point) so the fill
        # brush previews the eventual polygon area, not just an open outline.
        path.closeSubpath()
        if self._polygon_path_item is None:
            pen = QPen(QColor(255, 255, 255), 1)
            pen.setCosmetic(True)  # 1 px regardless of zoom level
            brush = QBrush(QColor(100, 150, 255, 70))
            item = QGraphicsPathItem()
            item.setPen(pen)
            item.setBrush(brush)
            item.setZValue(10)
            scene.addItem(item)
            self._polygon_path_item = item
        try:
            self._polygon_path_item.setPath(path)
        except RuntimeError:
            # C++ object was deleted externally (e.g. scene.clear()) — recreate next move
            self._polygon_path_item = None

        # Start-point marker: a small, zoom-constant-size dot anchored on the
        # first point, so it's always obvious where to click to close the loop.
        # It highlights (turns green) once the cursor is within closing range.
        if self._polygon_start_marker_item is None:
            r = self._polygon_close_radius_px
            marker = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            marker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            marker.setZValue(11)
            scene.addItem(marker)
            self._polygon_start_marker_item = marker
        try:
            near_start = preview_point is not None and self._is_near_polygon_start(preview_point)
            if near_start:
                self._polygon_start_marker_item.setPen(QPen(QColor(80, 230, 120), 2))
                self._polygon_start_marker_item.setBrush(QBrush(QColor(80, 230, 120, 160)))
            else:
                self._polygon_start_marker_item.setPen(QPen(QColor(255, 210, 60), 2))
                self._polygon_start_marker_item.setBrush(QBrush(QColor(255, 210, 60, 120)))
            self._polygon_start_marker_item.setPos(self._polygon_points[0])
        except RuntimeError:
            self._polygon_start_marker_item = None

    def _remove_polygon_overlay(self) -> None:
        scene = self.scene()
        for attr in ("_polygon_path_item", "_polygon_start_marker_item"):
            item = getattr(self, attr)
            if item is not None:
                if scene is not None:
                    try:
                        scene.removeItem(item)
                    except RuntimeError:
                        pass
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if self._polygon_mode and event.button() == Qt.MouseButton.LeftButton:
            scene_pt = self.mapToScene(event.position().toPoint())
            if self._is_near_polygon_start(scene_pt):
                self.polygon_confirmed.emit(list(self._polygon_points))
                event.accept()
                return
            self._polygon_points.append(scene_pt)
            self._update_polygon_overlay()
            event.accept()
            return
        if self._crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self._crop_anchor = self.mapToScene(event.position().toPoint())
            self._crop_rect_scene = QRectF(self._crop_anchor, self._crop_anchor)
            self._update_crop_overlay()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._polygon_mode and self._polygon_points:
            current = self.mapToScene(event.position().toPoint())
            if event.buttons() & Qt.MouseButton.LeftButton:
                # Continuous "sweep" drawing: while the button is held, keep
                # adding points as the cursor moves far enough (in view pixels)
                # from the last one, so a click-and-drag traces a smooth
                # freehand outline instead of requiring a separate click per
                # vertex. Points are only added here, never closed here --
                # closing on a release near the start point is handled in
                # mouseReleaseEvent, so releasing elsewhere just pauses the sweep.
                last_view = self.mapFromScene(self._polygon_points[-1])
                current_view = self.mapFromScene(current)
                dx = current_view.x() - last_view.x()
                dy = current_view.y() - last_view.y()
                if (dx * dx + dy * dy) ** 0.5 >= self._polygon_sweep_min_step_px:
                    self._polygon_points.append(current)
            self._update_polygon_overlay(preview_point=current)
            event.accept()
            return
        if self._crop_mode and self._crop_anchor is not None:
            current = self.mapToScene(event.position().toPoint())
            self._crop_rect_scene = QRectF(self._crop_anchor, current).normalized()
            self._update_crop_overlay()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._polygon_mode:
            # Points are placed on press/drag, not release. But a sweep's
            # natural closing gesture *is* a release -- unlike a plain click,
            # which closes on press if it lands near the start, a sweep never
            # checks proximity mid-drag (it just keeps adding points), so
            # releasing back near the start point must also close here or a
            # sweep could never auto-close at all.
            if event.button() == Qt.MouseButton.LeftButton:
                scene_pt = self.mapToScene(event.position().toPoint())
                if self._is_near_polygon_start(scene_pt):
                    self.polygon_confirmed.emit(list(self._polygon_points))
                    event.accept()
                    return
            # Otherwise swallow the release so it doesn't fall through to
            # pan/selection handling; releasing away from the start just
            # pauses the sweep, it never auto-closes.
            event.accept()
            return
        if self._crop_mode and event.button() == Qt.MouseButton.LeftButton:
            if self._crop_rect_scene and not self._crop_rect_scene.isEmpty():
                self.crop_selection_ready.emit(self._crop_rect_scene)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._polygon_mode:
            if event.key() == Qt.Key.Key_Escape:
                self.end_polygon_mode()
                self.crop_cancelled.emit()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if len(self._polygon_points) >= 3:
                    self.polygon_confirmed.emit(list(self._polygon_points))
                event.accept()
                return
            if event.key() == Qt.Key.Key_Backspace:
                if self._polygon_points:
                    self._polygon_points.pop()
                    if self._polygon_points:
                        self._update_polygon_overlay()
                    else:
                        self._remove_polygon_overlay()
                event.accept()
                return
        if self._crop_mode:
            if event.key() == Qt.Key.Key_Escape:
                self.end_crop_mode()
                self.crop_cancelled.emit()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self._crop_rect_scene and not self._crop_rect_scene.isEmpty():
                    self.crop_confirmed.emit(self._crop_rect_scene)
                event.accept()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        if self._crop_mode or self._polygon_mode:
            # Swallow wheel events during selection to keep the coordinate system stable.
            event.accept()
            return
        if not self._interaction_enabled:
            super().wheelEvent(event)
            return
        if not (event.modifiers() & self._zoom_modifier):
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return

        direction = self._zoom_step if delta > 0 else (1.0 / self._zoom_step)
        new_zoom = self._zoom_factor * direction
        new_zoom = max(self._min_zoom, min(new_zoom, self._max_zoom))
        if new_zoom == self._zoom_factor:
            event.accept()
            return
        factor = new_zoom / self._zoom_factor

        old_transform_anchor = self.transformationAnchor()
        old_resize_anchor = self.resizeAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(old_transform_anchor)
        self.setResizeAnchor(old_resize_anchor)

        self._zoom_factor = new_zoom
        self._has_user_zoom = abs(self._zoom_factor - 1.0) > 1e-6
        event.accept()

"""
QGraphicsView with Ctrl+wheel zoom and drag-pan for static image display.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsView


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView wrapper that provides modular wheel-zoom and drag-pan."""

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
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)
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

    def wheelEvent(self, event):
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

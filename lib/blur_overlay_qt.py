"""
BlurOverlay — a frameless, translucent top-level widget that covers a target
QWidget to provide a visual obscure/blur effect.

Uses the same Tool-window technique as MediaControlsOverlay so it renders
above native/OpenGL surfaces (e.g. VLC video), where QGraphicsBlurEffect
cannot reach.  Mouse events pass through to the covered widget.

Usage (manual geometry sync — mirrors the MediaControlsOverlay pattern)::

    overlay = BlurOverlay(parent=frame)
    overlay.sync_geometry(frame)
    overlay.show_blur()
    ...
    overlay.hide_blur()

Usage (automatic tracking)::

    overlay = BlurOverlay(parent=frame)
    overlay.attach(frame)   # installs event filters; call once
    overlay.show_blur()
    ...
    overlay.hide_blur()
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QWidget


class BlurOverlay(QWidget):
    """
    Semi-opaque top-level overlay that obscures a target widget's content.

    Renders above VLC's DirectX/OpenGL surface by using Qt.Tool +
    WA_TranslucentBackground (same approach as MediaControlsOverlay).
    WA_TransparentForMouseEvents forwards all clicks to the frame below.
    """

    DEFAULT_OPACITY: int = 200  # 0-255; tunable per instance via set_opacity()

    def __init__(self, parent: QWidget, opacity: int = DEFAULT_OPACITY) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._opacity: int = max(0, min(255, int(opacity)))
        self._label: str = ""
        self._blurred_bg: QPixmap | None = None

        self._tracked_frame: QWidget | None = None
        self._tracked_top: QWidget | None = None

    # ------------------------------------------------------------------
    # Geometry tracking
    # ------------------------------------------------------------------

    def attach(self, frame: QWidget) -> None:
        """
        Begin auto-tracking *frame*'s position and size.

        Installs event filters on both the frame and its top-level window so
        that the overlay repositions itself on any Move, Resize, or
        WindowStateChange event — no manual sync calls needed after this.
        """
        if self._tracked_frame is not None:
            self._tracked_frame.removeEventFilter(self)
        if self._tracked_top is not None:
            self._tracked_top.removeEventFilter(self)

        self._tracked_frame = frame
        frame.installEventFilter(self)

        top = frame.window()
        if top and top is not frame:
            self._tracked_top = top
            top.installEventFilter(self)
        else:
            self._tracked_top = None

        self.sync_geometry(frame)

    def sync_geometry(self, frame: QWidget) -> None:
        """Reposition and resize the overlay to cover *frame* exactly."""
        top_left = frame.mapToGlobal(QPoint(0, 0))
        self.setGeometry(QRect(top_left, frame.size()))

    def eventFilter(self, watched: QWidget, event: QEvent) -> bool:
        if event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
            QEvent.Type.WindowStateChange,
        ):
            if self._tracked_frame is not None:
                self.sync_geometry(self._tracked_frame)
        return False  # never consume the event

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def set_opacity(self, opacity: int) -> None:
        """Set fill opacity (0 = fully transparent, 255 = fully opaque)."""
        self._opacity = max(0, min(255, int(opacity)))
        self.update()

    def set_label(self, text: str) -> None:
        """Optional centred text drawn over the overlay (e.g. 'Blurred')."""
        self._label = text
        self.update()

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def show_blur(self, snapshot: QPixmap | None = None) -> None:
        """Show the overlay above the tracked frame.

        *snapshot* should be a grab of the media frame taken just before this
        call (e.g. via ``frame.grab()``).  It is downscaled then upscaled to
        approximate a blur.  If None or null, a plain dark tint is used.
        """
        if snapshot is not None and not snapshot.isNull():
            self._blur_snapshot(snapshot)
        else:
            self._blurred_bg = None
        self.show()
        self.raise_()

    def update_blur(self, snapshot: QPixmap | None = None) -> None:
        """Replace the blurred snapshot while the overlay stays visible.

        Call this instead of ``show_blur`` when the overlay is already shown
        and only the underlying content has changed (e.g. navigating from one
        blurred image to another).
        """
        if snapshot is not None and not snapshot.isNull():
            self._blur_snapshot(snapshot)
        else:
            self._blurred_bg = None
        self.update()

    def hide_blur(self) -> None:
        """Hide the overlay, revealing the media underneath."""
        self._blurred_bg = None
        self.hide()

    def is_blur_active(self) -> bool:
        return self.isVisible()

    # ------------------------------------------------------------------
    # Blur helpers
    # ------------------------------------------------------------------

    # Downscale factor: larger = more blur, less residual detail.
    _BLUR_FACTOR: int = 20

    def _blur_snapshot(self, source: QPixmap) -> None:
        """
        Downscale *source* then upscale it back to approximate a Gaussian blur.

        Qt's SmoothTransformation uses bilinear interpolation on the upscale
        pass, which spreads each low-resolution pixel across many output pixels
        — producing a convincing soft blur without a convolution kernel.
        """
        f = self._BLUR_FACTOR
        small_w = max(1, source.width() // f)
        small_h = max(1, source.height() // f)
        small = source.scaled(
            small_w, small_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._blurred_bg = small.scaled(
            source.width(), source.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        if self._blurred_bg is not None:
            # Use the rect-based overload so Qt handles DPI scaling correctly
            # regardless of the pixmap's devicePixelRatio.
            painter.drawPixmap(self.rect(), self._blurred_bg, self._blurred_bg.rect())
            # Frosted-glass tint over the blurred snapshot
            painter.fillRect(self.rect(), QColor(200, 200, 200, 60))
        else:
            painter.fillRect(self.rect(), QColor(0, 0, 0, self._opacity))
        if self._label:
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._label
            )

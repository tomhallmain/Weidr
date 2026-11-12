"""
VideoCropOverlay — frameless topmost widget for crop selection over a paused VLC video.

VLC renders via Direct3D into a native child HWND, so any overlay placed inside
the Qt widget hierarchy is invisible beneath VLC's surface.  This module provides
a separate OS-level window (WindowStaysOnTopHint) that sits unconditionally above
VLC, plus a ``launch_video_crop`` helper that handles pausing, dimension discovery,
geometry calculation, and background crop execution.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ui.app_window.vlc_picture_geometry import fit_rect, picture_from, read_video_track_shape
from ui.app_window.vlc_video_embed import place_tool_window
from utils.logging_setup import get_logger
from utils.translations import _

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow
    from ui.app_window.media_frame import MediaFrame

logger = get_logger("video_crop_overlay")

# Module-level reference so CPython doesn't GC the overlay while it is open.
# PySide6 top-level windows have no Qt parent to hold them alive, so without
# this the Python wrapper is destroyed immediately after launch_video_crop()
# returns (refcount → 0), closing the window before the event loop runs.
_active_overlay: VideoCropOverlay | None = None


def _set_active(overlay: VideoCropOverlay | None) -> None:
    global _active_overlay
    _active_overlay = overlay


class VideoCropOverlay(QWidget):
    """Frameless topmost overlay for rubber-band crop selection over a paused VLC video.

    Positioned to cover exactly the video content area within the media frame
    (accounting for letterboxing).  The user drags to select a region; Enter
    confirms and fires ``on_confirmed(x, y, w, h)`` in video pixels, Escape closes.
    ``inv_scale`` maps overlay x to video pixels; ``inv_scale_y`` maps y and
    defaults to the same value. They differ for a non-square sample aspect ratio.
    """

    def __init__(
        self,
        geo: QRect,
        vid_w: int,
        vid_h: int,
        inv_scale: float,
        on_confirmed: Callable[[int, int, int, int], None],
        inv_scale_y: float | None = None,
    ) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._vid_w = vid_w
        self._vid_h = vid_h
        self._inv_scale = inv_scale
        self._inv_scale_y = inv_scale if inv_scale_y is None else inv_scale_y
        self._on_confirmed = on_confirmed
        self._anchor: QPoint | None = None
        self._sel: QRect | None = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(geo)
        self.show()
        self.activateWindow()
        self.setFocus()

    def closeEvent(self, event) -> None:  # noqa: N802
        _set_active(None)
        super().closeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 80))
        if self._sel and not self._sel.isEmpty():
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(self._sel, QColor(0, 0, 0, 255))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(QPen(QColor(255, 80, 80), 2))
            p.drawRect(self._sel)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._anchor = event.position().toPoint()
            self._sel = QRect(self._anchor, self._anchor)
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._anchor is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self._sel = QRect(self._anchor, event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            sel = self._sel
            if sel and sel.width() > 1 and sel.height() > 1:
                inv_x = self._inv_scale
                inv_y = self._inv_scale_y
                x = max(0, int(sel.x() * inv_x))
                y = max(0, int(sel.y() * inv_y))
                w = min(self._vid_w - x, max(1, int(sel.width() * inv_x)))
                h = min(self._vid_h - y, max(1, int(sel.height() * inv_y)))
                self.close()
                self._on_confirmed(x, y, w, h)
            return
        super().keyPressEvent(event)


def _launch_video_rect_selector(
    media_frame: MediaFrame,
    media_path: str,
    app: AppWindow,
    *,
    apply_fn: Callable[[str, int, int, int, int], str],
    no_output_msg: str,
    success_msg: str,
    toast_msg: str,
) -> None:
    """Shared setup for a video rect-selection action: pause VLC, compute the video
    display rect, open a VideoCropOverlay above it, and run
    ``apply_fn(media_path, x, y, w, h)`` on a background thread once confirmed.
    """
    media_frame.pause_video_if_playing()

    player = media_frame.vlc_media_player
    if not player:
        app.notification_ctrl.toast(_("VLC player unavailable"))
        return
    vid_w, vid_h = player.video_get_size(0)
    if vid_w <= 0 or vid_h <= 0:
        app.notification_ctrl.toast(_("Could not determine video dimensions"))
        return

    # The picture is fitted and centred in the media frame with its displayed
    # shape (SAR and orientation applied); the selection maps back to the
    # auto-rotated pixel grid the ffmpeg filters work in.
    picture = picture_from(int(vid_w), int(vid_h), read_video_track_shape(player))
    mf_w, mf_h = media_frame.width(), media_frame.height()
    disp_x, disp_y, disp_w, disp_h = fit_rect(
        mf_w, mf_h, picture.display_w, picture.display_h
    )
    origin = media_frame.mapToGlobal(QPoint(disp_x, disp_y))
    geo = QRect(origin, QSize(disp_w, disp_h))
    inv_scale_x = picture.frame_w / disp_w   # overlay pixels → video pixels
    inv_scale_y = picture.frame_h / disp_h

    def _do_action(x: int, y: int, w: int, h: int) -> None:
        def _run() -> None:
            try:
                new_path = apply_fn(media_path, x, y, w, h)
            except RuntimeError as exc:
                _msg = str(exc)
                QTimer.singleShot(0, app, lambda: app.app_actions.warn(_msg))
                return
            if new_path and os.path.exists(new_path):
                _p = new_path
                def _on_done() -> None:
                    app.app_actions.refresh()
                    app.app_actions.success(success_msg)
                    from ui.image.media_details import MediaDetails
                    MediaDetails.open_temp_media_canvas(
                        master=app, media_path=_p,
                        app_actions=app.app_actions,
                    )
                QTimer.singleShot(0, app, _on_done)
            else:
                QTimer.singleShot(0, app, lambda: app.app_actions.warn(no_output_msg))
        from utils.running_tasks_registry import start_thread
        start_thread(_run, use_asyncio=False)

    logger.debug(
        "launch_video_rect_selector: vid=%dx%d frame=%dx%d disp=%dx%d inv_scale=%.4f,%.4f geo=%s",
        vid_w, vid_h, picture.frame_w, picture.frame_h, disp_w, disp_h,
        inv_scale_x, inv_scale_y, geo,
    )
    overlay = VideoCropOverlay(
        geo, picture.frame_w, picture.frame_h, inv_scale_x, _do_action,
        inv_scale_y=inv_scale_y,
    )
    # setGeometry alone can clamp the window onto the first display; this
    # repeats the placement in physical pixels on Windows.
    place_tool_window(overlay, media_frame, QPoint(disp_x, disp_y), disp_w, disp_h)
    _set_active(overlay)
    app.notification_ctrl.toast(toast_msg)


def launch_video_crop(media_frame: MediaFrame, media_path: str, app: AppWindow) -> None:
    """Pause VLC, compute the video display rect, and open a VideoCropOverlay above it for cropping."""
    from image.video_ops import VideoOps
    _launch_video_rect_selector(
        media_frame, media_path, app,
        apply_fn=VideoOps.crop_video,
        no_output_msg=_("Video crop produced no output"),
        success_msg=_("Video cropped"),
        toast_msg=_("Drag to select crop, Enter to confirm, Escape to cancel"),
    )


def launch_video_box(media_frame: MediaFrame, media_path: str, app: AppWindow) -> None:
    """Pause VLC, compute the video display rect, and open a VideoCropOverlay above it for
    painting a random color box, reusing the same overlay as :func:`launch_video_crop`."""
    from image.video_ops import VideoOps
    _launch_video_rect_selector(
        media_frame, media_path, app,
        apply_fn=VideoOps.draw_box_on_video,
        no_output_msg=_("Video box draw produced no output"),
        success_msg=_("Video box added"),
        toast_msg=_("Drag to select box location, Enter to confirm, Escape to cancel"),
    )


def launch_video_background_box(media_frame: MediaFrame, media_path: str, app: AppWindow) -> None:
    """Pause VLC, compute the video display rect, and open a VideoCropOverlay above it for
    painting a random color fill over everything outside the selection (the
    "background"), reusing the same overlay as :func:`launch_video_crop`."""
    from image.video_ops import VideoOps
    _launch_video_rect_selector(
        media_frame, media_path, app,
        apply_fn=VideoOps.draw_background_box_on_video,
        no_output_msg=_("Video background box draw produced no output"),
        success_msg=_("Video background box added"),
        toast_msg=_("Drag to select area to keep, Enter to confirm, Escape to cancel"),
    )

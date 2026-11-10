"""
Windows video surface for libvlc.

The main window is frameless and translucent. VLC's Direct3D swap chain is a
child of the HWND it was given at playback start. Moving the app from a
1080p monitor onto a 4K HDMI TV (different size and Windows scale, often a
different refresh rate) does not move that surface: Qt paints images on the
TV, and the video stays on the first display.

Qt's own setGeometry clamps a tool window into the screen it was created on,
so a position meant for the TV (to the right of, or above, the primary) lands
at the top of display 1. This module places an opaque window with the physical
pixel rectangle Windows already uses for the media frame, then rebuilds
playback so the swap chain is created on that display.
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget

from utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ui.app_window.media_frame import MediaFrame

logger = get_logger("vlc_video_embed")

# Long enough for Windows to finish the DPI change that a 1080p-to-4K move
# triggers, short enough that the picture is not left on the previous monitor.
_REBIND_DELAY_MS = 150


def _same_screen(left, right) -> bool:
    """True when both values refer to the same QScreen."""
    if left is right:
        return True
    if left is None or right is None:
        return False
    try:
        return left.name() == right.name() and left.geometry() == right.geometry()
    except Exception:
        return False


def _on_windows_platform() -> bool:
    """True only for the real Windows QPA, not the offscreen test platform."""
    if platform.system() != "Windows":
        return False
    app = QApplication.instance()
    return app is not None and app.platformName() == "windows"


def _local_physical_rect(frame, local_pos: QPoint, width: int, height: int):
    """Physical screen rectangle of a point inside ``frame``, or None.

    ``mapToGlobal`` is in Qt's logical coordinates. The 1080p monitor and the
    4K TV do not share a scale factor, and a tool window created on the first
    screen gets clamped back onto it. The top-level HWND is already where the
    user dragged it; ClientToScreen reports that position in real pixels, and
    the offset inside the window is scaled by the window's current device
    pixel ratio.
    """
    if not _on_windows_platform():
        return None
    top = frame.window()
    hwnd = int(top.winId() or 0)
    if not hwnd:
        return None
    try:
        import ctypes

        class _Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        user32 = ctypes.windll.user32
        user32.ClientToScreen.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Point)]
        user32.ClientToScreen.restype = ctypes.c_bool
        origin = _Point(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        ratio = float(top.devicePixelRatio() or 1.0)
        offset = frame.mapTo(top, local_pos)
        return (
            int(round(origin.x + offset.x() * ratio)),
            int(round(origin.y + offset.y() * ratio)),
            max(1, int(round(width * ratio))),
            max(1, int(round(height * ratio))),
        )
    except Exception as exc:
        logger.warning("Could not read the media frame position: %s", exc)
        return None


def place_tool_window(widget, frame, local_pos: QPoint, width: int, height: int) -> None:
    """Move a tool window over a rectangle of the media frame.

    Logical setGeometry stays so tests on the offscreen platform can read it
    back. On Windows the native move runs afterwards, in physical pixels, so a
    clamp onto the first display does not win. The native window is not
    recreated here: libvlc is already drawing into its HWND.
    """
    top_left = frame.mapToGlobal(local_pos)
    widget.setGeometry(top_left.x(), top_left.y(), width, height)
    physical = _local_physical_rect(frame, local_pos, width, height)
    hwnd = int(widget.winId() or 0) if physical is not None else 0
    if physical is not None and hwnd:
        _move_hwnd(hwnd, *physical)


def _move_hwnd(hwnd: int, x: int, y: int, width: int, height: int) -> bool:
    """Place a window in physical pixels, including positions off the primary."""
    if not _on_windows_platform() or not hwnd:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos.restype = ctypes.c_bool
        swp_noactivate = 0x0010
        return bool(user32.SetWindowPos(hwnd, None, x, y, width, height, swp_noactivate))
    except Exception as exc:
        logger.warning("Could not move the video window: %s", exc)
        return False


def _ensure_ws_clip_children(hwnd: int) -> None:
    """libvlc 3.0.21+ requires WS_CLIPCHILDREN on the HWND passed to set_hwnd.

    Without it the parent paints over the video child. VLC no longer sets the
    bit itself, and Qt does not set it on native widgets.
    """
    if platform.system() != "Windows" or not hwnd:
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        long_ptr = ctypes.c_ssize_t
        get_long = user32.GetWindowLongPtrW
        set_long = user32.SetWindowLongPtrW
        get_long.restype = long_ptr
        get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
        set_long.restype = long_ptr
        set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, long_ptr]
        style = int(get_long(hwnd, -16) or 0)
        clip_children = 0x02000000
        if style & clip_children:
            return
        set_long(hwnd, -16, style | clip_children)
    except Exception as exc:
        logger.warning("Could not set WS_CLIPCHILDREN on the video window: %s", exc)


class VlcVideoHost(QWidget):
    """Opaque window libvlc draws into. Not translucent, so DWM can move it."""

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setObjectName("vlcVideoHost")
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAutoFillBackground(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(0, 0, 0))
        self.setPalette(palette)
        self.hide()

    def mousePressEvent(self, event):
        owner = self.parentWidget()
        if owner is not None:
            window = owner.window()
            window.activateWindow()
            owner.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)


class VlcVideoEmbed:
    """Positions the video window over a media frame and rebuilds it per display."""

    def __init__(self, frame: MediaFrame):
        self._frame = frame
        self.host: VlcVideoHost | None = None
        self.bound_hwnd = 0
        self._retired: list[VlcVideoHost] = []
        self._screen = None
        self._rebind_in_progress = False
        self._screen_bound = False
        self._timer = QTimer(frame)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_REBIND_DELAY_MS)
        self._timer.timeout.connect(self._rebind)

    def stop(self) -> None:
        self._timer.stop()

    def is_host(self, widget) -> bool:
        return widget is not None and widget is self.host

    def playback_handle(self):
        """HWND of the host once it exists, else None. The frame winId is the fallback."""
        host = self.host
        if host is None or platform.system() != "Windows":
            return None
        wid = host.winId()
        return int(wid) if wid else None

    def prepare_hwnd(self) -> int:
        """Native window libvlc should draw into. 0 when there is none yet.

        On Windows this is the opaque host. Elsewhere it is the frame itself,
        which already follows the window on those platforms.
        """
        frame = self._frame
        if platform.system() != "Windows":
            wid = frame.winId()
            return int(wid) if wid else 0
        host = self._ensure_host()
        self._place(host, show=True)
        wid = int(host.winId()) if host.winId() else 0
        _ensure_ws_clip_children(wid)
        screen = self._screen_of_frame()
        if screen is not None:
            self._screen = screen
        return wid

    def sync_geometry(self) -> None:
        host = self.host
        if host is None or not host.isVisible():
            return
        self._place(host, show=False)

    def hide_host(self) -> None:
        if self.host is not None:
            self.host.hide()

    def restore_if_bound(self) -> None:
        """Show the host again after the frame was hidden during playback."""
        if self.bound_hwnd and self.host is not None:
            self._place(self.host, show=True)

    def discard_retired(self) -> None:
        hung = self._frame._hung_stop_thread
        if hung is not None and hung.is_alive():
            return
        hosts = self._retired
        self._retired = []
        for host in hosts:
            host.hide()
            host.deleteLater()

    def attach_screen_tracking(self) -> None:
        if self._screen_bound:
            return
        top = self._frame.window()
        handle = top.windowHandle() if top is not None else None
        if handle is None:
            return
        handle.screenChanged.connect(self._on_screen_changed)
        self._screen_bound = True

    def note_screen(self) -> None:
        """Schedule a swap-chain rebuild when the top-level window changes displays."""
        if self._rebind_in_progress:
            return
        screen = self._screen_of_frame()
        if screen is None:
            return
        if self._screen is None:
            self._screen = screen
            return
        if _same_screen(screen, self._screen):
            return
        self._screen = screen
        if self._playback_is_embedded():
            self._timer.start(_REBIND_DELAY_MS)

    def _on_screen_changed(self, _screen) -> None:
        self.sync_geometry()
        self._frame._position_overlay()
        self.note_screen()

    def _screen_of_frame(self):
        top = self._frame.window()
        handle = top.windowHandle() if top is not None else None
        if handle is None:
            return None
        return handle.screen()

    def _playback_is_embedded(self) -> bool:
        frame = self._frame
        video = frame._video_ui
        return (
            platform.system() == "Windows"
            and frame.vlc_media_player is not None
            and video is not None
            and bool(getattr(video, "has_video", False))
            and bool(self.bound_hwnd)
        )

    def _ensure_host(self) -> VlcVideoHost:
        if self.host is None:
            self.host = VlcVideoHost(self._frame)
        return self.host

    def _place(self, host: VlcVideoHost, *, show: bool) -> None:
        frame = self._frame
        if show:
            host.show()
        place_tool_window(
            host, frame, QPoint(0, 0), max(frame.width(), 1), max(frame.height(), 1)
        )
        if not host.isVisible():
            return
        host.raise_()
        overlay = frame._controls_overlay
        if overlay.isVisible():
            overlay.raise_()

    def _retire_host(self, *, destroy: bool) -> None:
        host = self.host
        self.host = None
        self.bound_hwnd = 0
        if host is None:
            return
        host.hide()
        if destroy:
            host.deleteLater()
        else:
            # stop() did not return; libvlc may still be presenting here.
            self._retired.append(host)

    def _rebind(self) -> None:
        """Open a new swap chain on the display the window is on now.

        libvlc reads the HWND when the vout opens. Setting the same HWND again
        is a no-op, and HWND 0 makes libvlc open its own top-level window.
        """
        frame = self._frame
        if self._rebind_in_progress or not self._playback_is_embedded():
            return
        player = frame.vlc_media_player
        if player is None or frame.vlc_instance is None or not frame.path:
            return
        self._rebind_in_progress = True
        frame._playback_timer.stop()
        try:
            position_ms = max(int(player.get_time() or 0), 0)
        except Exception:
            position_ms = 0
        try:
            rate = float(player.get_rate() or 0)
        except Exception:
            rate = 0
        if rate <= 0:
            rate = frame._last_known_rate
        try:
            paused = not bool(player.is_playing())
        except Exception:
            paused = False
        self.sync_geometry()
        logger.info("Rebuilding video output for the display the window was moved to")
        stopped_clean = frame._stop_vlc_player(player)
        self._retire_host(destroy=stopped_clean)
        player = frame.vlc_media_player
        if player is None:
            self._rebind_in_progress = False
            return
        frame.ensure_video_frame()
        old_media = frame.vlc_media
        try:
            media = frame.vlc_instance.media_new(frame.path)
        except Exception as exc:
            logger.warning("Could not reopen video after a display change: %s", exc)
            self._rebind_in_progress = False
            frame._playback_timer.start()
            return
        frame.vlc_media = media
        player.set_media(media)
        if stopped_clean and old_media is not None and old_media is not media:
            try:
                old_media.release()
            except Exception:
                pass
        if player.play() == -1:
            logger.warning("Failed to resume video after a display change")
        else:
            if position_ms > 0:
                try:
                    player.set_time(position_ms)
                except Exception:
                    pass
            frame.set_playback_rate(rate)
            if paused:
                try:
                    player.set_pause(1)
                except Exception:
                    pass
                frame.set_playback_paused(True)
            else:
                frame.set_playback_paused(False)
        frame.on_track_changed()
        frame._playback_timer.start()
        self.sync_geometry()
        self._rebind_in_progress = False

"""
Native surface libvlc draws into, sized to the picture.

The host covers only the rectangle the picture occupies, so the media
frame's own (translucent) background shows beside a portrait video instead
of libvlc's black bars. It stays hidden until libvlc has a video output.

On Windows the host is a separate opaque tool window. The main window is
frameless and translucent, and VLC's Direct3D swap chain is a child of the
HWND it was given at playback start: moving the app from a 1080p monitor
onto a 4K TV with a different Windows scale does not move that surface, so
the video stays on the first display. Qt's own setGeometry clamps a tool
window into the screen it was created on, so the host is placed with the
physical pixel rectangle Windows uses for the media frame, and playback is
rebuilt so the swap chain is created on the new display.

On other platforms the host is a native child widget of the media frame. It
moves with the window, so no placement or rebuild is needed.
"""

from __future__ import annotations

import platform
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QContextMenuEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from ui.app_window.vlc_picture_geometry import (
    VideoTrackShape,
    VlcPicture,
    fit_rect,
    read_picture,
    read_video_track_shape,
)
from utils.logging_setup import get_logger
from utils.translations import _

if TYPE_CHECKING:
    from ui.app_window.media_frame import MediaFrame

logger = get_logger("vlc_video_embed")

# Long enough for Windows to finish the DPI change that a 1080p-to-4K move
# triggers, short enough that the picture is not left on the previous monitor.
_REBIND_DELAY_MS = 150

# The host is opaque black until libvlc presents a picture, so it stays
# hidden until VLC has a video output. Polled faster than the 250 ms playback
# tick so the picture is not held back. The timeout shows the host anyway in
# case has_vout() never reports one for a stream libvlc is still drawing.
_REVEAL_POLL_MS = 30
_REVEAL_TIMEOUT_MS = 3000


_wayland_warning_logged = False


def warn_if_wayland() -> bool:
    """Log once when Qt runs on Wayland. True when the warning was logged.

    libvlc 3 embeds video on Linux only into an X11 window (set_xwindow), and
    Wayland does not let an app position its own top-level windows, which the
    controls bar and crop overlay rely on. Under XWayland (QT_QPA_PLATFORM=xcb)
    both work.
    """
    global _wayland_warning_logged
    if _wayland_warning_logged or platform.system() != "Linux":
        return False
    app = QApplication.instance()
    if app is None or not str(app.platformName()).startswith("wayland"):
        return False
    _wayland_warning_logged = True
    logger.warning(
        _(
            "Qt is running on Wayland: videos cannot be embedded in the window and "
            "the playback controls cannot be placed over the media. Start the app "
            "with QT_QPA_PLATFORM=xcb to use XWayland."
        )
    )
    return True


def _refresh_macos_gl_view(host: QWidget, rect: QRect) -> None:
    """Make libvlc's macOS video view rebuild its drawable after the host is shown.

    libvlc draws through an NSOpenGLView placed inside the host. Unhiding the
    host does not change that view's frame, so its GL context can keep the
    drawable it got while hidden and show nothing until the next resize. A
    one-pixel resize and back makes it reshape.
    """
    host.setGeometry(rect.adjusted(0, 0, 1, 0))
    host.setGeometry(rect)


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
        logger.warning(_("Could not read the media frame position: {0}").format(exc))
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
        # NOZORDER: a geometry sync must not lift the video over dialogs.
        # NOACTIVATE: the move must not take focus from the window under the cursor.
        swp_nozorder = 0x0004
        swp_noactivate = 0x0010
        return bool(
            user32.SetWindowPos(
                hwnd, None, x, y, width, height, swp_nozorder | swp_noactivate
            )
        )
    except Exception as exc:
        logger.warning(_("Could not move the video window: {0}").format(exc))
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
        logger.warning(
            _("Could not set WS_CLIPCHILDREN on the video window: {0}").format(exc)
        )


def _host_is_tool_window() -> bool:
    """Windows needs a separate window to follow the app across displays."""
    return platform.system() == "Windows"


class VlcVideoHost(QWidget):
    """Opaque native surface libvlc draws into.

    As a tool window (Windows) it is not translucent, so DWM can move it. As a
    child widget, mouse events it ignores propagate to the media frame.
    """

    def __init__(self, parent=None, *, tool_window: bool = True):
        flags = Qt.WindowType.Widget
        if tool_window:
            flags = (
                Qt.WindowType.Tool
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
        super().__init__(parent, flags)
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

    def _forward_to_owner(self, event) -> None:
        """Resend input to the media frame, mapped into its coordinates.

        A tool window is a top-level widget, so Qt does not propagate events
        it ignores to its parent: without this, right-click (context menu),
        middle-click (delete) and the wheel (navigate) do nothing over the
        video. Sent to the frame, an ignored event propagates up to AppWindow.
        A child host gets that propagation from Qt and is not forwarded.
        """
        owner = self.parentWidget()
        if owner is None or not self.isWindow():
            event.ignore()
            return
        if isinstance(event, QContextMenuEvent):
            global_pos = event.globalPos()
            resent = QContextMenuEvent(
                event.reason(), owner.mapFromGlobal(global_pos), global_pos, event.modifiers()
            )
        elif isinstance(event, QWheelEvent):
            global_pos = event.globalPosition()
            resent = QWheelEvent(
                owner.mapFromGlobal(global_pos),
                global_pos,
                event.pixelDelta(),
                event.angleDelta(),
                event.buttons(),
                event.modifiers(),
                event.phase(),
                event.inverted(),
            )
        else:
            global_pos = event.globalPosition()
            resent = QMouseEvent(
                event.type(),
                owner.mapFromGlobal(global_pos),
                global_pos,
                event.button(),
                event.buttons(),
                event.modifiers(),
            )
        QApplication.sendEvent(owner, resent)
        event.setAccepted(resent.isAccepted())

    def mousePressEvent(self, event):
        owner = self.parentWidget()
        if owner is not None:
            window = owner.window()
            window.activateWindow()
            owner.setFocus(Qt.FocusReason.MouseFocusReason)
        self._forward_to_owner(event)

    def mouseReleaseEvent(self, event):
        self._forward_to_owner(event)

    def mouseDoubleClickEvent(self, event):
        self._forward_to_owner(event)

    def wheelEvent(self, event):
        self._forward_to_owner(event)

    def contextMenuEvent(self, event):
        self._forward_to_owner(event)


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
        self._picture: VlcPicture | None = None
        self._track_shape: VideoTrackShape | None = None
        self._placed_key: tuple[QRect, float] | None = None
        self._timer = QTimer(frame)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_REBIND_DELAY_MS)
        self._timer.timeout.connect(self._rebind)
        self._awaiting_picture = False
        self._reveal_deadline = 0.0
        self._reveal_timer = QTimer(frame)
        self._reveal_timer.setInterval(_REVEAL_POLL_MS)
        self._reveal_timer.timeout.connect(self._reveal_when_ready)

    def stop(self) -> None:
        self._timer.stop()
        self._reveal_timer.stop()
        self._awaiting_picture = False

    def is_host(self, widget) -> bool:
        return widget is not None and widget is self.host

    def playback_handle(self):
        """Native handle of the host once it exists, else None. The frame winId is the fallback."""
        host = self.host
        if host is None:
            return None
        wid = host.winId()
        return int(wid) if wid else None

    def prepare_hwnd(self) -> int:
        """Native window libvlc should draw into. 0 when there is none yet.

        This is the host: a tool window on Windows, a native child widget of
        the frame elsewhere.
        """
        host = self._ensure_host()
        # A hidden host is positioned now and shown by _reveal_when_ready.
        self._place(host, show=False)
        if not host.isVisible():
            self._start_reveal()
        wid = int(host.winId()) if host.winId() else 0
        _ensure_ws_clip_children(wid)
        screen = self._screen_of_frame()
        if screen is not None:
            self._screen = screen
        return wid

    def note_new_media(self) -> None:
        """Drop the previous picture shape so the next file is measured on its own."""
        self._picture = None
        self._track_shape = None
        self._placed_key = None

    def sync_geometry(self) -> None:
        host = self.host
        if host is None or not host.isVisible():
            return
        if self._placement_key(self._content_rect()) == self._placed_key:
            return
        self._place(host, show=False)

    def hide_host(self) -> None:
        """Hide the host. A pending reveal resumes in restore_if_bound."""
        self._placed_key = None
        self._reveal_timer.stop()
        if self.host is not None:
            self.host.hide()

    def restore_if_bound(self) -> None:
        """Show the host again after the frame was hidden during playback."""
        if not self.bound_hwnd or self.host is None:
            return
        if self._awaiting_picture:
            self._place(self.host, show=False)
            self._start_reveal()
            return
        self._place(self.host, show=True)

    def _start_reveal(self) -> None:
        self._awaiting_picture = True
        self._reveal_deadline = time.monotonic() + _REVEAL_TIMEOUT_MS / 1000.0
        self._reveal_timer.start()

    def _has_video_output(self) -> bool:
        player = getattr(self._frame, "vlc_media_player", None)
        if player is None:
            return False
        try:
            has_vout = int(player.has_vout() or 0) > 0
        except Exception:
            has_vout = False
        return has_vout and self._video_picture() is not None

    def _reveal_when_ready(self) -> None:
        """Show the host once libvlc has a video output to present into it."""
        host = self.host
        if not self._awaiting_picture or host is None or not self.bound_hwnd:
            self._awaiting_picture = False
            self._reveal_timer.stop()
            return
        if not self._frame.isVisible():
            # showEvent -> restore_if_bound restarts the poll.
            self._reveal_timer.stop()
            return
        if not self._has_video_output() and time.monotonic() < self._reveal_deadline:
            return
        self._awaiting_picture = False
        self._reveal_timer.stop()
        self._place(host, show=True)

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
            self.host = VlcVideoHost(self._frame, tool_window=_host_is_tool_window())
        return self.host

    def _video_picture(self) -> VlcPicture | None:
        """Displayed picture shape, remembered once VLC has reported a size.

        The decoded size is re-read on every call because a stream can change
        resolution. The track's SAR and orientation are read once per media.
        """
        player = getattr(self._frame, "vlc_media_player", None)
        if player is None:
            return self._picture
        picture = read_picture(player, self._track_shape)
        if picture is not None and self._track_shape is None:
            # A reported size means the decoder is running and the track is listed.
            self._track_shape = read_video_track_shape(player)
            if self._track_shape is not None:
                picture = read_picture(player, self._track_shape)
        if picture is not None:
            self._picture = picture
        return self._picture

    def _content_rect(self) -> QRect:
        """Frame-local rectangle the picture occupies, aspect preserved and centred.

        The host covers only this rectangle. The rest of the media frame stays
        the frame's own background, which is what shows beside a portrait video.
        """
        frame = self._frame
        full_w, full_h = max(frame.width(), 1), max(frame.height(), 1)
        picture = self._video_picture()
        if picture is None:
            return QRect(0, 0, full_w, full_h)
        return QRect(*fit_rect(full_w, full_h, picture.display_w, picture.display_h))

    def _placement_key(self, rect: QRect) -> tuple[QRect, float]:
        """Logical global rectangle plus scale: the physical rectangle can change
        with the scale alone, e.g. when the window lands on a display of another
        Windows scale at the same logical position."""
        frame = self._frame
        global_rect = QRect(frame.mapToGlobal(rect.topLeft()), rect.size())
        return global_rect, float(frame.devicePixelRatio() or 1.0)

    def _apply_geometry(self, host: VlcVideoHost, rect: QRect) -> None:
        if host.isWindow():
            place_tool_window(host, self._frame, rect.topLeft(), rect.width(), rect.height())
        else:
            host.setGeometry(rect)

    def _place(self, host: VlcVideoHost, *, show: bool) -> None:
        frame = self._frame
        rect = self._content_rect()
        # Positioned while still hidden, so it does not appear at its last
        # rectangle first.
        self._apply_geometry(host, rect)
        if show and host.isHidden():
            host.show()
            if host.isWindow():
                # Showing can apply Qt's clamp onto the first display again;
                # the native move after it wins.
                self._apply_geometry(host, rect)
            elif platform.system() == "Darwin":
                _refresh_macos_gl_view(host, rect)
        self._placed_key = self._placement_key(rect)
        if not host.isVisible() or not show:
            return
        # Only the first show changes stacking. Later syncs keep dialogs on top.
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
        logger.info(_("Rebuilding video output for the display the window was moved to"))
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
            logger.warning(_("Could not reopen video after a display change: {0}").format(exc))
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
            logger.warning(_("Failed to resume video after a display change"))
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

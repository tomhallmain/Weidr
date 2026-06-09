"""
NotificationController -- toast display, title notifications, alerts, and label state.

Uses signals internally so it is safe to call from any thread.

Title-bar notification scheduling uses two QTimers (main-thread owned):
  _expiry_timer   — single-shot, fires when the next notification expires and
                    restores/updates the window title on the main thread.
  _cleanup_timer  — 60-second repeating, calls notification_manager.cleanup_all_expired()
                    to reclaim memory from old Notification objects.

This replaces the old threading.Timer approach, which caused a race where the
timer callback's queued Signal.emit() (background thread) could arrive in the
event queue *after* a direct main-thread emit, silently reverting newly-set
notification titles. See docs/notification_manager_timing_bug.md for full details.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.config import config
from utils.constants import ActionType
from utils.logging_setup import get_logger
from utils.notification_manager import notification_manager
from utils.translations import _

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow
logger = get_logger("notification_controller")


def _safe_close_widget(widget: QWidget | None) -> None:
    """Close *widget* if the underlying Qt object still exists."""
    if widget is None:
        return
    try:
        import shiboken6

        if not shiboken6.isValid(widget):
            return
    except Exception:
        pass
    try:
        widget.close()
    except RuntimeError:
        pass


class _NotificationSignals(QObject):
    """Signals for cross-thread toast delivery."""
    toast_requested = Signal(str, int, str)       # message, seconds, bg_color


class NotificationController:
    """
    Owns toast display, title-bar notifications, message-box alerts,
    and the sidebar state / label updates.
    """

    # Interval for periodic cleanup of expired Notification objects (ms).
    _CLEANUP_INTERVAL_MS = 60_000

    def __init__(self, app_window: AppWindow):
        self._app = app_window
        self._signals = _NotificationSignals()
        self._signals.toast_requested.connect(self._do_toast)
        self._status_title_override_active = False
        self._loading_spinner = None  # set by SidebarPanel after construction

        # Single-shot QTimer: fires when the soonest active notification for this
        # window expires. Always runs on the main thread — no Signal queuing race.
        self._expiry_timer = QTimer(self._signals)
        self._expiry_timer.setSingleShot(True)
        self._expiry_timer.timeout.connect(self._on_notification_expiry)

        # Repeating QTimer: 60-second housekeeping pass to evict old Notification
        # objects from the global list (replaces the threading.Timer cleanup loop).
        self._cleanup_timer = QTimer(self._signals)
        self._cleanup_timer.setInterval(self._CLEANUP_INTERVAL_MS)
        self._cleanup_timer.timeout.connect(self._on_cleanup_timer)
        self._cleanup_timer.start()

    # ------------------------------------------------------------------
    # Toast
    # ------------------------------------------------------------------
    def toast(
        self,
        message: str,
        time_in_seconds: int = config.toasts_persist_seconds,
        bg_color: Optional[str] = None,
    ) -> None:
        """
        Show a transient toast notification. Thread-safe: if called from
        a background thread the signal is queued to the main thread.
        """
        logger.info("Toast: " + message.replace("\n", " "))
        if not config.show_toasts:
            return
        color = bg_color or AppStyle.BG_COLOR
        self._signals.toast_requested.emit(message, time_in_seconds, color)

    def _do_toast(self, message: str, time_in_seconds: int, bg_color: str) -> None:
        """
        Main-thread implementation of toast display.

        Creates a frameless overlay widget at the top-right of the parent
        window, which auto-destructs after *time_in_seconds*.
        """
        parent = self._app

        # Calculate position: top-right of parent window
        width = 300
        height = 100
        parent_geo = parent.geometry()
        x = parent_geo.x() + parent_geo.width() - width
        y = parent_geo.y()

        previously_active = QApplication.activeWindow()

        # Create frameless overlay
        toast_widget = QWidget(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        # Top-level widget with no parent: close() alone only hides; without this,
        # every toast stays in memory (same class of leak as modal dialogs).
        toast_widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        # Explicitly avoid activation/focus stealing when showing toast.
        toast_widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        toast_widget.setFixedSize(width, height)
        toast_widget.move(x, y)
        toast_widget.setStyleSheet(
            f"background-color: {bg_color}; border: 1px solid {AppStyle.FG_COLOR};"
        )

        layout = QVBoxLayout(toast_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        label = QLabel(message.strip())
        label.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 10pt; border: none;")
        label.setWordWrap(True)
        layout.addWidget(label)

        toast_widget.show()

        # Defensive restoration for a narrow case: if a secondary window/dialog
        # owned by this app was active (e.g. FileActionsWindow), restore it.
        # Avoid broad re-activation of arbitrary windows.
        should_restore_owned_secondary = (
            previously_active is not None
            and previously_active is not toast_widget
            and previously_active is not self._app
            and previously_active.isVisible()
            and (
                previously_active.parent() is self._app
                or self._app.isAncestorOf(previously_active)
            )
        )
        if should_restore_owned_secondary:
            try:
                previously_active.activateWindow()
            except Exception:
                pass

        # Auto-destruct after the specified time (WA_DeleteOnClose makes close() delete)
        QTimer.singleShot(
            time_in_seconds * 1000,
            lambda tw=toast_widget: _safe_close_widget(tw),
        )

    # ------------------------------------------------------------------
    # Title notifications
    # ------------------------------------------------------------------
    def title_notify(
        self,
        message: str,
        base_message: str = "",
        time_in_seconds: int = 0,
        action_type: ActionType = ActionType.SYSTEM,
        is_manual: bool = True,
    ) -> None:
        """Temporarily modify the window title to show a notification message.

        Always called on the main thread (wrapped with ts() in app_actions), so
        the immediate setWindowTitle() call is direct — no signal queuing needed.
        The expiry timer is also a QTimer (main-thread), so restoration is also
        direct, eliminating the threading.Timer race described in the module docstring.
        """
        if not config.show_toasts:
            return
        if time_in_seconds == 0:
            time_in_seconds = config.title_notify_persist_seconds

        # Snapshot the bare title before adding the notification so that
        # get_display_title() can restore to it when the notification expires.
        notification_manager.set_current_title(
            self._app.get_title_from_base_dir(), window_id=self._app.window_id
        )
        notification_manager.add_notification(
            message, base_message, time_in_seconds, action_type, is_manual,
            window_id=self._app.window_id,
        )

        # In fullscreen mode the title bar is not visible, so show a toast
        # instead of updating the (invisible) title.  message is available
        # here directly, so no need to look it back up from the data layer.
        if self._app.app_actions.is_fullscreen():
            self.toast(message, time_in_seconds)
            return

        # Update the title immediately on the main thread, then arm the expiry
        # timer so the title is restored when the notification duration lapses.
        self._apply_current_title()
        self._reschedule_expiry()

    def _apply_current_title(self) -> None:
        """Set the window title to the current notification-aware display title.

        Always called on the main thread.  Fullscreen callers should use
        toast() directly instead (handled in title_notify before reaching here).
        """
        self._app.setWindowTitle(
            notification_manager.get_display_title(self._app.window_id)
        )

    def _reschedule_expiry(self) -> None:
        """(Re)arm _expiry_timer to fire when the soonest notification expires.

        If no active notifications remain, the timer is stopped so it doesn't
        fire spuriously. Always called on the main thread.
        """
        delay = notification_manager.get_next_expiry_delay(self._app.window_id)
        if delay is None:
            self._expiry_timer.stop()
            return
        # setInterval + start() restarts if already running, so this is
        # safe to call repeatedly without explicit stop/start sequencing.
        self._expiry_timer.setInterval(int(delay * 1000))
        self._expiry_timer.start()

    def _on_notification_expiry(self) -> None:
        """QTimer slot: called on the main thread when the soonest notification expires.

        Cleans up the expired entry, refreshes the title (which may still have
        other active notifications to show), and re-arms the timer for the next
        expiry if one exists.
        """
        notification_manager.cleanup_expired(self._app.window_id)
        self._apply_current_title()
        self._reschedule_expiry()

    def _on_cleanup_timer(self) -> None:
        """QTimer slot: 60-second housekeeping pass across all windows.

        Removes Notification objects that have expired from every window, not just
        the current one, preventing unbounded list growth over long sessions.
        """
        notification_manager.cleanup_all_expired()

    def set_status_title(self, message: Optional[str]) -> None:
        """
        Set/clear a non-accumulating status title override.

        This bypasses notification_manager grouping for transient progress states
        where we want a direct, latest-only title update.
        """
        if message and message.strip() != "":
            base_title = self._app.get_title_from_base_dir()
            self._app.setWindowTitle(f"{base_title} - {message}")
            self._status_title_override_active = True
            return
        if self._status_title_override_active:
            self._app.setWindowTitle(self._app.get_title_from_base_dir())
            self._status_title_override_active = False

    # ------------------------------------------------------------------
    # Alerts / errors
    # ------------------------------------------------------------------
    def alert(
        self,
        title: str,
        message: str,
        kind: str = "info",
        severity: str = "normal",
        master: Optional[QWidget] = None,
        buttons: Optional[list[tuple[str, str]]] = None,
    ) -> "bool | str":
        """
        Show a modal message box.

        Returns True for OK/Yes, False otherwise for the default two-button
        modes.  When *buttons* is provided (a list of ``(label, role)`` pairs)
        the return value is the label of the clicked button, or ``False`` if a
        reject-role button was clicked — so callers can use ``if not result:``
        uniformly regardless of button mode.

        Pported from App.alert.
        """
        logger.warning(f'Alert - Title: "{title}" Message: {message}')
        parent = master or self._app

        # For dangerous operations with high severity, use a custom styled dialog
        if severity == "high" and kind == "askokcancel":
            from lib.custom_dialogs_qt import show_high_severity_dialog
            return show_high_severity_dialog(parent, title, message, buttons=buttons)

        return qt_alert(parent, title, message, kind=kind)

    # ------------------------------------------------------------------
    # Loading spinner
    # ------------------------------------------------------------------
    def set_loading_spinner(self, spinner) -> None:
        """Wire up the sidebar spinner badge (called by SidebarPanel after init)."""
        self._loading_spinner = spinner

    def start_loading_spinner(self, force: bool = False) -> None:
        """Show the spinner.

        When *force* is False (default) the spinner is suppressed if a title-bar
        notification is already active, to avoid adding noise.  Pass
        ``force=True`` when the caller needs the spinner despite an active
        title notification — e.g. prevalidation loops, or base-directory /
        incremental loading where the indicator should stay visible.
        """
        if self._loading_spinner is None:
            return
        if not force and notification_manager.has_active_notifications(self._app.window_id):
            return
        self._loading_spinner.start()

    def stop_loading_spinner(self) -> None:
        """Always hide the spinner regardless of notification state."""
        if self._loading_spinner is not None:
            self._loading_spinner.stop()

    def handle_error(self, error_text: str, title: Optional[str] = None, kind: str = "error") -> None:
        """Display an error dialog."""
        traceback.print_exc()
        title = title or _("Error")
        self.alert(title, error_text, kind=kind)

    # ------------------------------------------------------------------
    # Sidebar label state
    # ------------------------------------------------------------------
    def set_label_state(
        self, text: Optional[str] = None, group_number: Optional[int] = None,
        size: int = -1, suffix: str = ""
    ) -> None:
        """Update the sidebar state label with the current file position info."""
        if text is not None:
            self._app.sidebar_panel.update_state_label(text)
            return

        if size > -1:
            if group_number is None:
                self._app.sidebar_panel.update_state_label("")
            else:
                args = (
                    group_number + 1,
                    len(self._app.compare_manager.file_groups),
                    size,
                )
                label_text = _("GROUP_DETAILS").format(*args) + suffix
                self._app.sidebar_panel.update_state_label(label_text)
            return

        # Default: set based on file count
        fb = self._app.file_browser
        file_count = fb.count() if fb else 0
        if file_count == 0:
            label_text = _("No media files found")
        elif file_count == 1:
            label_text = _("1 media file found")
        else:
            label_text = _("{0} media files found").format(file_count)

        # Check inclusion pattern
        inclusion_text = self._app.sidebar_panel.file_filter_entry.text().strip()
        if inclusion_text != "":
            label_text += "\n" + _("(filtered)")

        self._app.sidebar_panel.update_state_label(label_text)

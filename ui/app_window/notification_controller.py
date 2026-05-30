"""
NotificationController -- toast display, title notifications, alerts, and label state.

Extracted from: toast, title_notify, alert, handle_error, _set_label_state.
Uses signals internally so it is safe to call from any thread.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QVBoxLayout, QWidget

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
    """Signals for cross-thread toast / title-notify delivery."""
    toast_requested = Signal(str, int, str)       # message, seconds, bg_color
    title_notify_requested = Signal(str, str, int)  # message, base_message, seconds


class NotificationController:
    """
    Owns toast display, title-bar notifications, message-box alerts,
    and the sidebar state / label updates.
    """

    def __init__(self, app_window: AppWindow):
        self._app = app_window
        self._signals = _NotificationSignals()
        self._signals.toast_requested.connect(self._do_toast)
        self._signals.title_notify_requested.connect(self._do_title_notify)
        self._status_title_override_active = False
        self._loading_spinner = None  # set by SidebarPanel after construction

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
        """
        Temporarily modify the window title to show a notification message.
        Thread-safe via signals.
        """
        if not config.show_toasts:
            return
        if time_in_seconds == 0:
            time_in_seconds = config.title_notify_persist_seconds

        notification_manager.set_current_title(
            self._app.get_title_from_base_dir(), window_id=self._app.window_id
        )
        notification_manager.add_notification(
            message, base_message, time_in_seconds, action_type, is_manual,
            window_id=self._app.window_id,
        )

    def _do_title_notify(self, message: str, base_message: str, time_in_seconds: int) -> None:
        """Main-thread implementation of title notification."""
        self._app.setWindowTitle(message)
        QTimer.singleShot(
            time_in_seconds * 1000,
            lambda: self._app.setWindowTitle(
                base_message or self._app.get_title_from_base_dir()
            ),
        )

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
        inclusion_text = self._app.sidebar_panel.inclusion_pattern.text().strip()
        if inclusion_text != "":
            label_text += "\n" + _("(filtered)")

        self._app.sidebar_panel.update_state_label(label_text)

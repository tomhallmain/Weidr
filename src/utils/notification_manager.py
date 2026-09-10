"""
Notification data layer — stores, groups, and expires title-bar notifications.

All scheduling and Qt title-update calls are handled by NotificationController,
which uses QTimer (main-thread owned) to avoid threading.Timer race conditions.
This split eliminates background threads, the shared self._timer race between
cleanup and expiry paths, and unguarded cross-thread reads.
"""

import time
from threading import Lock
from typing import List, Optional

from utils.config import config
from utils.constants import ActionType
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("notification_manager")


def debug_log(msg: str) -> None:
    """Debug logging function"""
    if config.debug:
        logger.debug(f"[NotificationManager] {msg}")


class Notification:
    """A notification to be displayed in the title of a window."""

    ETC_MESSAGE = _(" etc.")

    def __init__(
        self,
        message: str,
        base_message: Optional[str] = None,
        duration: float = 5.0,
        action_type: ActionType = ActionType.SYSTEM,
        is_manual: bool = True,
        window_id: int = 0,
    ):
        self.message = message.replace("\n", " ") # TODO remove this when all translations have newline removed
        self.base_message = base_message
        self.duration = duration
        self.action_type = action_type
        self.is_manual = is_manual
        self.created_at = time.time()
        self.expires_at = self.created_at + duration
        self.count = 1  # Track number of similar notifications
        self.auto_count = 0 if is_manual else 1  # Track number of auto notifications
        self.manual_count = 1 if is_manual else 0  # Track number of manual notifications
        self.window_id = window_id  # Track which window generated this notification
        self.shown = False  # Track whether this notification has been displayed

    def get_display_message(self) -> str:
        """Get the formatted message for display."""
        prefix = self.action_type.get_translation()

        # Determine the auto/manual prefix
        if self.auto_count > 0 and self.manual_count > 0:
            prefix = f"[Auto+Manual] {prefix}"
        elif self.auto_count > 0:
            prefix = f"[Auto] {prefix}"

        if self.count > 1:
            if self.base_message:
                return f"{prefix} ({self.count}): {self.base_message}"
            return f"{prefix} ({self.count})"
        return f"{prefix}: {self.message}"


class NotificationManager:
    """
    Pure data layer: stores, groups, and expires notifications.

    Does NOT own any timers or make any Qt calls. NotificationController
    drives all scheduling via QTimer and calls these methods on the main thread,
    eliminating the threading.Timer race that caused stale title restores to
    overwrite newly-set notification titles.
    """

    def __init__(self) -> None:
        debug_log("Initializing NotificationManager")
        self._notifications: List[Notification] = []
        self._lock = Lock()
        self._current_titles: dict[int, str] = {}  # Dictionary of current titles keyed by window ID
        self._base_group_window = 3.0  # Base time window in seconds to group similar notifications
        self._current_group_window = self._base_group_window  # NOTE: Maybe make this a map too
        self._max_group_window = 10.0  # Maximum time window in seconds
        self._window_expansion_rate = 1.5  # How much to expand the window when notifications arrive
        self._window_contraction_rate = 0.8  # How much to contract the window when no notifications arrive
        self._last_notification_time = 0.0

    # ------------------------------------------------------------------
    # Backward-compat stubs (called from app_window.py / window_manager.py)
    # ------------------------------------------------------------------
    def set_app_actions(self, app_actions, window_id: int = 0) -> None:
        """No-op — title updates are now driven by NotificationController via QTimer."""
        debug_log(f"set_app_actions called for window {window_id} (no-op in new design)")

    def cleanup_threads(self) -> None:
        """No-op — no background threads remain; QTimer in NotificationController handles scheduling."""
        logger.info("cleanup_threads called (no-op: no background threads)")

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------
    def unregister_window(self, window_id: int) -> None:
        """Remove a closed window from all tracking dicts."""
        debug_log(f"Unregistering window {window_id}")
        self._current_titles.pop(window_id, None)
        with self._lock:
            self._notifications = [
                n for n in self._notifications if n.window_id != window_id
            ]

    # ------------------------------------------------------------------
    # Notification data API
    # ------------------------------------------------------------------
    def add_notification(
        self,
        message: str,
        base_message: Optional[str] = "",
        duration: float = 5.0,
        action_type: ActionType = ActionType.SYSTEM,
        is_manual: bool = True,
        window_id: int = 0,
    ) -> bool:
        """Add a new notification to the queue, or group it with an existing one.

        Returns True when the caller should refresh the window title (always on
        success). All scheduling and title-update calls are the caller's
        responsibility — NotificationController handles those via QTimer.
        """
        debug_log(f"Adding notification for window {window_id}: {message}")
        current_time = time.time()

        with self._lock:
            debug_log("Acquired lock for adding notification")
            # Adjust the group window based on notification frequency
            if current_time - self._last_notification_time < self._current_group_window:
                # Expand window if notifications are coming in quickly
                self._current_group_window = min(
                    self._current_group_window * self._window_expansion_rate,
                    self._max_group_window,
                )
            else:
                # Contract window if notifications are sparse
                self._current_group_window = max(
                    self._current_group_window * self._window_contraction_rate,
                    self._base_group_window,
                )
            self._last_notification_time = current_time

            # Look for a notification of the same type within the time window
            for notification in self._notifications:
                if (
                    notification.action_type == action_type
                    and notification.base_message == base_message
                    and notification.window_id == window_id
                    and current_time - notification.created_at < self._current_group_window
                ):
                    # Update the existing notification
                    debug_log("Updating existing notification")
                    notification.message = None  # Clear message when bundling
                    notification.created_at = current_time
                    notification.expires_at = current_time + duration
                    notification.count += 1
                    if is_manual:
                        notification.manual_count += 1
                    else:
                        notification.auto_count += 1
                    debug_log(f"Updated notification expires at: {notification.expires_at}")
                    return True

            # If no similar notification found, create a new one
            debug_log("Creating new notification")
            notification = Notification(message, base_message, duration, action_type, is_manual, window_id)
            self._notifications.append(notification)
            debug_log(f"New notification expires at: {notification.expires_at}")
            return True

    def cleanup_expired(self, window_id: int) -> None:
        """Remove expired notifications for *window_id*.

        Called from NotificationController's QTimer expiry slot so that stale
        entries don't accumulate between housekeeping runs.
        """
        current_time = time.time()
        with self._lock:
            self._notifications = [
                n for n in self._notifications
                if not (n.window_id == window_id and n.expires_at <= current_time)
            ]

    def cleanup_all_expired(self) -> None:
        """Remove all expired notifications across every window.

        Called from NotificationController's 60-second housekeeping QTimer,
        replacing the old threading.Timer-based _cleanup_old_notifications loop.
        """
        debug_log("Starting cleanup of all expired notifications")
        current_time = time.time()
        with self._lock:
            self._notifications = [n for n in self._notifications if n.expires_at > current_time]
        debug_log("Cleanup completed")

    def get_next_expiry_delay(self, window_id: int) -> Optional[float]:
        """Return seconds until the next notification for *window_id* expires.

        Returns None if there are no active notifications for that window.
        NotificationController uses this to set the interval on its QTimer so
        the expiry fires as precisely as possible without polling.
        """
        current_time = time.time()
        with self._lock:
            relevant = [
                n for n in self._notifications
                if n.window_id == window_id and n.expires_at > current_time
            ]
        if not relevant:
            return None
        return max(0.05, min(n.expires_at for n in relevant) - current_time)

    def has_active_notifications(self, window_id: int = 0) -> bool:
        """Return True if there are unexpired notifications for *window_id*."""
        current_time = time.time()
        with self._lock:
            return any(
                n.window_id == window_id and n.expires_at > current_time
                for n in self._notifications
            )

    def set_current_title(self, title: str, window_id: int = 0) -> None:
        """Set the current (base) window title used when no notifications are active."""
        debug_log(f"Setting current title for window {window_id}: {title}")
        with self._lock:
            self._current_titles[window_id] = title

    def get_display_title(self, window_id: int = 0) -> str:
        """Get the title that should be displayed, including any active notifications."""
        debug_log(f"Getting display title for window {window_id}")
        with self._lock:
            window_notifications = [
                n for n in self._notifications if n.window_id == window_id
            ]
            base_title = self._current_titles.get(window_id, "")

        # Filter out expired notifications
        current_time = time.time()
        window_notifications = [n for n in window_notifications if n.expires_at > current_time]

        if not window_notifications:
            debug_log(f"No active notifications for window {window_id}, returning base title")
            return base_title

        # Sort notifications by creation time (newest first)
        sorted_notifications = sorted(
            window_notifications, key=lambda n: n.created_at, reverse=True
        )

        # Start with the base title; reserve space for " - " and potential "etc."
        remaining_length = 250 - len(base_title) - 3

        # Add notifications until we hit the length limit
        notification_parts: list[str] = []
        for notification in sorted_notifications:
            display_msg = notification.get_display_message()
            if len(display_msg) + 3 <= remaining_length:  # +3 for " - "
                notification_parts.append(display_msg)
                remaining_length -= len(display_msg) + 3
            else:
                break

        if not notification_parts:
            debug_log("No notifications fit in title length limit")
            return base_title

        # Combine notifications
        combined = " - ".join(notification_parts)

        # If we couldn't show all notifications, add "etc."
        if len(sorted_notifications) > len(notification_parts):
            if remaining_length >= 5:  # If we have space for " etc."
                combined += Notification.ETC_MESSAGE

        debug_log(f"Returning combined title with {len(notification_parts)} notifications")
        return f"{base_title} - {combined}"


# Global singleton — data only, no threads.
notification_manager = NotificationManager()

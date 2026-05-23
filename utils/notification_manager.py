import time
from threading import Lock, Timer
from typing import List, Optional

from utils.config import config
from utils.constants import ActionType
from utils.logging_setup import get_logger
from utils.translations import I18N

_ = I18N._

logger = get_logger("notification_manager")


def debug_log(msg: str):
    """Debug logging function"""
    if config.debug:
        logger.debug(f"[NotificationManager] {msg}")


class Notification:
    """A notification to be displayed in the title of a window."""

    ETC_MESSAGE = _(" etc.")

    def __init__(self, message: str, base_message: Optional[str] = None, duration: float = 5.0,
                 action_type: ActionType = ActionType.SYSTEM, is_manual: bool = True, window_id: int = 0):
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
    def __init__(self):
        debug_log("Initializing NotificationManager")
        self._notifications: List[Notification] = []
        self._lock = Lock()
        self._timer: Optional[Timer] = None
        self._current_titles = {}  # Dictionary of current titles keyed by window ID
        self._base_group_window = 3.0  # Base time window in seconds to group similar notifications
        self._current_group_window = self._base_group_window # NOTE: Maybe make this a map too
        self._max_group_window = 10.0  # Maximum time window in seconds
        self._window_expansion_rate = 1.5  # How much to expand the window when notifications arrive
        self._window_contraction_rate = 0.8  # How much to contract the window when no notifications arrive
        self._last_notification_time = 0.0
        self._cleanup_interval = 60.0  # Clean up notifications every 60 seconds
        self._app_actions = {}  # Dictionary of app_actions keyed by window ID
        self._schedule_cleanup()

    def set_app_actions(self, app_actions, window_id=0):
        """Set the app_actions for a specific window."""
        debug_log(f"Setting app_actions for window {window_id}")
        self._app_actions[window_id] = app_actions

    def unregister_window(self, window_id: int) -> None:
        """Remove a closed window from all tracking dicts."""
        debug_log(f"Unregistering window {window_id}")
        self._app_actions.pop(window_id, None)
        self._current_titles.pop(window_id, None)

    def cleanup_threads(self):
        """Clean up all threads."""
        logger.info("Cleaning up all threads")
        if self._timer:
            self._timer.cancel()
        self._timer = None

    def _schedule_cleanup(self) -> None:
        """Schedule periodic cleanup of old notifications."""
        debug_log("Scheduling cleanup")
        with self._lock:
            if self._timer:
                debug_log("Cancelling existing timer")
                self._timer.cancel()
            debug_log("Creating new cleanup timer")
            self._timer = Timer(self._cleanup_interval, self._cleanup_old_notifications)
            self._timer.start()

    def _cleanup_old_notifications(self) -> None:
        """Remove notifications that are older than the maximum group window."""
        debug_log("Starting cleanup of old notifications")
        current_time = time.time()
        with self._lock:
            self._notifications = [n for n in self._notifications if current_time - n.created_at < self._max_group_window]
        debug_log("Cleanup completed")
        # Schedule next cleanup outside the lock to avoid deadlock
        self._schedule_cleanup()

    def _schedule_update(self) -> None:
        """Schedule the next title update for cleanup only."""
        debug_log("Starting schedule update")
        # Cancel existing timer outside the lock to avoid deadlock
        if self._timer:
            debug_log("Cancelling existing timer")
            self._timer.cancel()
            self._timer = None

        with self._lock:
            if not self._notifications:
                debug_log("No notifications to schedule")
                return

            # Find the next notification that will expire
            current_time = time.time()
            next_expiry = min(n.expires_at for n in self._notifications)
            delay = max(0.1, next_expiry - current_time)
            debug_log(f"Scheduling cleanup in {delay:.2f} seconds")

            def timer_callback():
                debug_log("Timer callback executing")
                self._update_title()

            self._timer = Timer(delay, timer_callback)
            self._timer.daemon = True
            self._timer.start()
            debug_log("Timer started")
        debug_log("Schedule update completed")

    def _update_title(self) -> None:
        """Update the title based on current notifications."""
        debug_log("Starting title update")
        current_titles = {}
        
        with self._lock:
            debug_log("Acquired lock for title update")
            # Filter out expired notifications
            current_time = time.time()
            debug_log(f"Current time: {current_time}")
            debug_log(f"Number of notifications before filtering: {len(self._notifications)}")
            
            # Log each notification's expiration time
            for n in self._notifications:
                debug_log(f"Notification expires at: {n.expires_at} (in {n.expires_at - current_time:.2f} seconds)")
            
            self._notifications = [n for n in self._notifications if n.expires_at > current_time]
            debug_log(f"Number of notifications after filtering: {len(self._notifications)}")
            
            # Get the title for each window while we still have the lock
            for window_id in self._app_actions.keys():
                current_titles[window_id] = self.get_display_title(window_id)
            
            # Check if we need to schedule another update
            needs_update = bool(self._notifications)
        
        # Update the title using all callbacks outside the lock
        if current_titles:
            # Create a copy of items to safely iterate while potentially modifying the dict
            for window_id, app_actions in list(self._app_actions.items()):
                if window_id in current_titles:
                    debug_log(f"Calling title update callback for window {window_id}")
                    try:
                        app_actions.title(current_titles[window_id])
                    except Exception as e:
                        debug_log(f"Failed to update title for window {window_id}: {e}")
                        # Qt window/widget callbacks can fail after disposal with
                        # runtime/reference errors (e.g. wrapped C/C++ object deleted).
                        error_text = str(e).lower()
                        is_disposed_window_error = (
                            isinstance(e, (RuntimeError, ReferenceError))
                            and (
                                "deleted" in error_text
                                or "invalid" in error_text
                                or "c/c++ object" in error_text
                            )
                        )
                        if is_disposed_window_error:
                            self._app_actions.pop(window_id, None)
                            self._current_titles.pop(window_id, None)
                        else:
                            logger.error(f"Failed to update title for window {window_id}: {e}")
        
        # Schedule next update outside the lock if needed
        if needs_update:
            debug_log("Scheduling next update")
            self._schedule_update()
            
        debug_log("Title update completed")

    def add_notification(self, message: str, base_message: Optional[str] = "", duration: float = 5.0,
                         action_type: ActionType = ActionType.SYSTEM, is_manual: bool = True, window_id: int = 0) -> None:
        """Add a new notification to the queue."""
        debug_log(f"Adding notification for window {window_id}: {message}")
        current_time = time.time()
        should_update = False
        current_title = None
        
        with self._lock:
            debug_log("Acquired lock for adding notification")
            # Adjust the group window based on notification frequency
            if current_time - self._last_notification_time < self._current_group_window:
                # Expand window if notifications are coming in quickly
                self._current_group_window = min(self._current_group_window * self._window_expansion_rate, self._max_group_window)
            else:
                # Contract window if notifications are sparse
                self._current_group_window = max(self._current_group_window * self._window_contraction_rate, self._base_group_window)
            
            self._last_notification_time = current_time
            
            # Look for a notification of the same type within the time window
            for notification in self._notifications:
                if (notification.action_type == action_type and
                    notification.base_message == base_message and
                    notification.window_id == window_id and
                    current_time - notification.created_at < self._current_group_window):
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
                    should_update = True
                    debug_log(f"Updated notification expires at: {notification.expires_at}")
                    break

            if not should_update:
                # If no similar notification found, create a new one
                debug_log("Creating new notification")
                notification = Notification(message, base_message, duration, action_type, is_manual, window_id)
                self._notifications.append(notification)
                debug_log(f"New notification expires at: {notification.expires_at}")
                should_update = True

        if should_update:
            # Get the title while we still have the lock
            current_title = self.get_display_title(window_id)
            # Schedule cleanup
            self._schedule_update()

        # Update title outside the lock
        if should_update and current_title is not None and window_id in self._app_actions:
            app_actions = self._app_actions[window_id]
            if app_actions.is_fullscreen():
                debug_log(f"Calling toast callback for window {window_id}")
                app_actions.toast(message, duration)
            else:
                debug_log(f"Calling title update callback for window {window_id}")
                app_actions.title(current_title)
        debug_log("Notification addition completed")

    def has_active_notifications(self, window_id: int = 0) -> bool:
        """Return True if there are unexpired notifications for *window_id*."""
        current_time = time.time()
        with self._lock:
            return any(
                n.window_id == window_id and n.expires_at > current_time
                for n in self._notifications
            )

    def set_current_title(self, title: str, window_id: int = 0) -> None:
        """Set the current window title."""
        debug_log(f"Setting current title for window {window_id}: {title}")
        with self._lock:
            self._current_titles[window_id] = title

    def get_display_title(self, window_id: int = 0) -> str:
        """Get the title that should be displayed, including any active notifications."""
        debug_log(f"Getting display title for window {window_id}")
        # Note: This method assumes it's being called while the lock is already held
        window_notifications = [n for n in self._notifications if n.window_id == window_id]
        if not window_notifications:
            debug_log(f"No notifications for window {window_id}, returning base title")
            return self._current_titles.get(window_id, "")

        # Filter out expired notifications
        current_time = time.time()

        if config.debug:
            debug_log(f"Current time in get_display_title: {current_time}")
            debug_log(f"Number of notifications before filtering: {len(window_notifications)}")
            
            # Log each notification's expiration time
            for n in window_notifications:
                debug_log(f"Notification expires at: {n.expires_at} (in {n.expires_at - current_time:.2f} seconds)")
        
        window_notifications = [n for n in window_notifications if n.expires_at > current_time]
        debug_log(f"Number of notifications after filtering: {len(window_notifications)}")
        
        if not window_notifications:
            debug_log(f"No active notifications after filtering for window {window_id}")
            return self._current_titles.get(window_id, "")

        # Sort notifications by creation time (newest first)
        sorted_notifications = sorted(window_notifications, key=lambda n: n.created_at, reverse=True)
        
        # Start with the base title
        title = self._current_titles.get(window_id, "")
        remaining_length = 250 - len(title) - 3  # Reserve space for " - " and potential "etc."
        
        # Add notifications until we hit the length limit
        notification_parts = []
        for notification in sorted_notifications:
            display_msg = notification.get_display_message()
            if len(display_msg) + 3 <= remaining_length:  # +3 for " - "
                notification_parts.append(display_msg)
                remaining_length -= len(display_msg) + 3
            else:
                break
        
        if not notification_parts:
            debug_log("No notifications fit in title length limit")
            return title
            
        # Combine notifications
        combined = " - ".join(notification_parts)
        
        # If we couldn't show all notifications, add "etc."
        if len(sorted_notifications) > len(notification_parts):
            if remaining_length >= 5:  # If we have space for " etc."
                combined += Notification.ETC_MESSAGE
        
        debug_log(f"Returning combined title with {len(notification_parts)} notifications")
        return f"{title} - {combined}"

# Global instance
notification_manager = NotificationManager() 
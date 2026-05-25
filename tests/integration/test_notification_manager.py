"""
Integration tests for NotificationManager.

Tests notification grouping, display title formatting, and window-ID
scoping without requiring a Qt runtime or real app_actions.

Covers notification bundling, title formatting, expiry, and window scoping
without a Qt runtime or the global ``notification_manager`` singleton.
"""

import time
import pytest

from utils.notification_manager import Notification, NotificationManager
from utils.constants import ActionType


@pytest.fixture
def manager():
    nm = NotificationManager()
    nm.cleanup_threads()  # cancel the background cleanup timer immediately
    yield nm
    nm.cleanup_threads()


class TestNotification:
    def test_manual_counts(self):
        n = Notification("msg", action_type=ActionType.MOVE_FILE, is_manual=True)
        assert n.manual_count == 1
        assert n.auto_count == 0

    def test_auto_counts(self):
        n = Notification("msg", action_type=ActionType.MOVE_FILE, is_manual=False)
        assert n.manual_count == 0
        assert n.auto_count == 1

    def test_display_message_single(self):
        n = Notification("test.jpg", action_type=ActionType.MOVE_FILE, is_manual=True)
        msg = n.get_display_message()
        assert "test.jpg" in msg
        assert n.count == 1

    def test_display_message_auto_prefix(self):
        n = Notification("test.jpg", action_type=ActionType.MOVE_FILE, is_manual=False)
        msg = n.get_display_message()
        assert "[Auto]" in msg

    def test_display_message_no_auto_prefix_for_manual(self):
        n = Notification("test.jpg", action_type=ActionType.MOVE_FILE, is_manual=True)
        assert "[Auto]" not in n.get_display_message()

    def test_display_message_bundled_count(self):
        n = Notification("test.jpg", base_message="files", action_type=ActionType.MOVE_FILE)
        n.count = 5
        msg = n.get_display_message()
        assert "(5)" in msg


class TestNotificationManagerTitles:
    def test_empty_manager_returns_base_title(self, manager):
        manager.set_current_title("Weidr - /my/dir", window_id=0)
        assert manager.get_display_title(0) == "Weidr - /my/dir"

    def test_active_notification_appended_to_title(self, manager):
        manager.set_current_title("Weidr", window_id=0)
        manager.add_notification("Moved file.jpg", action_type=ActionType.MOVE_FILE, duration=10.0)
        title = manager.get_display_title(0)
        assert title.startswith("Weidr")
        assert "Moved" in title or ActionType.MOVE_FILE.get_translation() in title

    def test_expired_notification_not_in_title(self, manager):
        manager.set_current_title("Weidr", window_id=0)
        manager.add_notification(
            "Moved file.jpg", action_type=ActionType.MOVE_FILE, duration=0.01
        )
        time.sleep(0.05)
        title = manager.get_display_title(0)
        assert title == "Weidr"

    def test_notifications_scoped_by_window_id(self, manager):
        manager.set_current_title("Window0", window_id=0)
        manager.set_current_title("Window1", window_id=1)
        manager.add_notification(
            "event", action_type=ActionType.SYSTEM, duration=10.0, window_id=0
        )
        assert "event" in manager.get_display_title(0) or ActionType.SYSTEM.get_translation() in manager.get_display_title(0)
        assert manager.get_display_title(1) == "Window1"

    def test_no_title_set_returns_empty(self, manager):
        assert manager.get_display_title(99) == ""


class TestNotificationBundling:
    def test_same_type_within_window_bundles(self, manager):
        manager._current_group_window = 10.0  # force a wide grouping window
        manager.add_notification("a.jpg", base_message="", action_type=ActionType.MOVE_FILE, duration=10.0)
        manager.add_notification("b.jpg", base_message="", action_type=ActionType.MOVE_FILE, duration=10.0)
        notifications = [n for n in manager._notifications if n.action_type == ActionType.MOVE_FILE]
        assert len(notifications) == 1
        assert notifications[0].count == 2

    def test_different_types_not_bundled(self, manager):
        manager._current_group_window = 10.0
        manager.add_notification("a.jpg", action_type=ActionType.MOVE_FILE, duration=10.0)
        manager.add_notification("b.jpg", action_type=ActionType.COPY_FILE, duration=10.0)
        assert len(manager._notifications) == 2

    def test_mixed_auto_manual_increments_both_counts(self, manager):
        manager._current_group_window = 10.0
        manager.add_notification("a.jpg", base_message="", action_type=ActionType.MOVE_FILE, is_manual=False, duration=10.0)
        manager.add_notification("b.jpg", base_message="", action_type=ActionType.MOVE_FILE, is_manual=True, duration=10.0)
        n = manager._notifications[0]
        assert n.auto_count == 1
        assert n.manual_count == 1


class TestBundlingScenarios:
    """End-to-end bundling and title behaviour with realistic message strings."""

    def test_single_move_notification_in_title(self, manager):
        manager.set_current_title("Weidr - C:/test/images", window_id=0)
        manager.add_notification(
            "Moved file: test1.jpg to /new/location",
            action_type=ActionType.MOVE_FILE,
            is_manual=True,
            duration=10.0,
        )
        title = manager.get_display_title(0)
        assert title.startswith("Weidr")
        assert "test1.jpg" in title

    def test_rapid_same_type_bundles_under_default_group_window(self, manager):
        for i in range(3):
            manager.add_notification(
                f"Moved file: test{i + 2}.jpg to /new/location",
                action_type=ActionType.MOVE_FILE,
                is_manual=True,
                duration=10.0,
            )
            time.sleep(0.05)
        bundled = [n for n in manager._notifications if n.action_type == ActionType.MOVE_FILE]
        assert len(bundled) == 1
        assert bundled[0].count == 3

    def test_remove_file_auto_uses_auto_prefix(self, manager):
        manager.set_current_title("Weidr", window_id=0)
        manager.add_notification(
            "Removed file: test6.jpg",
            action_type=ActionType.REMOVE_FILE,
            is_manual=False,
            duration=10.0,
        )
        assert manager._notifications[0].auto_count == 1
        assert "[Auto]" in manager._notifications[0].get_display_message()

    def test_custom_duration_active_then_cleared_from_title(self, manager):
        manager.set_current_title("Weidr", window_id=0)
        manager.add_notification(
            "This notification will stay for 2 seconds",
            duration=2.0,
            action_type=ActionType.SYSTEM,
        )
        assert manager.has_active_notifications(0)
        assert "2 seconds" in manager.get_display_title(0)
        time.sleep(2.1)
        assert not manager.has_active_notifications(0)
        assert manager.get_display_title(0) == "Weidr"

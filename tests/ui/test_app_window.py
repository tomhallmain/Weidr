"""
UI integration tests for the main AppWindow.

Covers construction, sidebar, set_base_dir, and keyboard navigation.
See also: test_search_controller, test_slideshow, test_masonry_view,
test_window_launcher.
"""

import io
import os
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.app_window.app_window import AppWindow
from utils.constants import Mode, SortBy
from utils.config import config


# ---------------------------------------------------------------------------
# Construction & initial state
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_window_creates_without_error(self, window):
        assert window is not None

    def test_initial_mode_is_browse(self, window):
        assert window.mode == Mode.BROWSE

    def test_initial_base_dir_is_none(self, window):
        assert window.base_dir is None

    def test_title_contains_app_name(self, window):
        assert "Weidr" in window.windowTitle()

    def test_splitter_has_two_children(self, window):
        assert window.splitter.count() == 2


# ---------------------------------------------------------------------------
# Sidebar widget presence & initial values
# ---------------------------------------------------------------------------

class TestSidebarWidgets:
    def test_set_dir_button_exists(self, window):
        assert window.sidebar_panel.set_base_dir_btn is not None

    def test_set_dir_entry_placeholder(self, window):
        placeholder = window.sidebar_panel.set_base_dir_box.placeholderText()
        assert placeholder  # non-empty

    def test_mode_label_shows_browse(self, window):
        assert Mode.BROWSE.get_text() in window.sidebar_panel.label_mode.text()

    def test_sort_by_combo_populated(self, window):
        combo = window.sidebar_panel.sort_by_choice
        assert combo.count() == len(SortBy.members())

    def test_recursive_check_reflects_config(self, window):
        assert window.sidebar_panel.recursive_check.isChecked() == config.browse_recursive

    def test_fill_canvas_check_reflects_config(self, window):
        assert window.sidebar_panel.fill_canvas_check.isChecked() == config.fill_canvas

    def test_search_return_closest_check_reflects_config(self, window):
        cb = window.sidebar_panel.search_return_closest_check
        assert cb.isChecked() == config.search_only_return_closest

    def test_search_image_entry_exists(self, window):
        assert window.sidebar_panel.search_media_path_box is not None

    def test_search_text_entry_exists(self, window):
        assert window.sidebar_panel.search_text_box is not None


# ---------------------------------------------------------------------------
# set_base_dir
# ---------------------------------------------------------------------------

class TestSetBaseDir:
    def test_set_base_dir_updates_base_dir(self, window_with_dir):
        win, media_dir = window_with_dir
        assert win.base_dir == media_dir

    def test_set_base_dir_populates_file_browser(self, window_with_dir):
        win, _ = window_with_dir
        assert win.file_browser.count() == 3

    def test_set_base_dir_invalid_path_ignored(self, window):
        window.set_base_dir("/this/path/does/not/exist")
        assert window.base_dir is None

    def test_set_base_dir_entry_box_accepts_text(self, window, media_dir, qtbot):
        """Typing into the entry and pressing Return triggers set_base_dir."""
        box = window.sidebar_panel.set_base_dir_box
        box.setText(media_dir)
        qtbot.keyClick(box, Qt.Key.Key_Return)
        qtbot.waitUntil(lambda: window.base_dir == media_dir, timeout=2000)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class TestNavigation:
    def test_first_file_shown_after_set_dir(self, window_with_dir):
        win, _ = window_with_dir
        assert win.media_path is not None
        assert win.media_path.endswith(".png")

    def test_right_arrow_advances_to_next_file(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        first = win.media_path
        qtbot.keyClick(win, Qt.Key.Key_Right)
        qtbot.waitUntil(lambda: win.media_path != first, timeout=2000)
        assert win.media_path != first

    def test_left_arrow_goes_back(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        # Move forward first
        qtbot.keyClick(win, Qt.Key.Key_Right)
        qtbot.waitUntil(
            lambda: win.media_path is not None and win.media_path.endswith("img02.png"),
            timeout=2000,
        )
        second = win.media_path
        qtbot.keyClick(win, Qt.Key.Key_Left)
        qtbot.waitUntil(lambda: win.media_path != second, timeout=2000)
        assert win.media_path != second

    def test_home_key_goes_to_first_file(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        # Advance a step first
        qtbot.keyClick(win, Qt.Key.Key_Right)
        qtbot.waitUntil(lambda: win.media_path is not None and "img02" in win.media_path, timeout=2000)
        qtbot.keyClick(win, Qt.Key.Key_Home)
        qtbot.waitUntil(lambda: win.media_path is not None and "img01" in win.media_path, timeout=2000)

    def test_end_key_goes_to_last_file(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        qtbot.keyClick(win, Qt.Key.Key_End)
        qtbot.waitUntil(lambda: win.media_path is not None and "img03" in win.media_path, timeout=2000)

    def test_navigation_wraps_forward(self, window_with_dir, qtbot):
        """Right-arrow past the last file should wrap to the first."""
        win, _ = window_with_dir
        qtbot.keyClick(win, Qt.Key.Key_End)
        qtbot.waitUntil(lambda: win.media_path is not None and "img03" in win.media_path, timeout=2000)
        qtbot.keyClick(win, Qt.Key.Key_Right)
        qtbot.waitUntil(lambda: win.media_path is not None and "img01" in win.media_path, timeout=2000)


# ---------------------------------------------------------------------------
# _check_large_directory_before_load
# ---------------------------------------------------------------------------

class TestCheckLargeDirectoryBeforeLoad:
    """Guard against the sort-cache corruption bug.

    The fix ensures file_browser.directory is restored to the original directory
    BEFORE the confirmation dialog runs.  If the periodic store_info_cache() timer
    fires while the modal is open, it must target the original directory — not the
    new large directory whose cache entry is about to be read.
    """

    def _patch_for_large_dir(self, monkeypatch, window, alert_side_effect):
        """Patch _gather_files and is_slow_total_files so the check always
        reaches the confirmation dialog, and replace alert() with *alert_side_effect*."""
        monkeypatch.setattr(
            window.file_browser, "_gather_files", lambda files=None: None
        )
        monkeypatch.setattr(
            window.file_browser, "is_slow_total_files", lambda threshold=5000: True
        )
        monkeypatch.setattr(window.notification_ctrl, "alert", alert_side_effect)

    def test_directory_restored_before_alert_fires(self, window, tmp_path, monkeypatch):
        """file_browser.directory must equal the original dir when the modal opens."""
        old_dir = str(tmp_path / "old")
        new_dir = str(tmp_path / "new")
        os.makedirs(old_dir)
        os.makedirs(new_dir)
        window.file_browser.directory = old_dir

        dir_during_alert = []

        def _capture(*args, **kwargs):
            dir_during_alert.append(window.file_browser.directory)
            return True

        self._patch_for_large_dir(monkeypatch, window, _capture)
        result = window._check_large_directory_before_load(new_dir)

        assert result is False  # user confirmed
        assert dir_during_alert == [old_dir], (
            "file_browser.directory was not restored before the alert; "
            "a periodic cache-store during the dialog would corrupt the new dir's sort"
        )

    def test_sort_cache_not_corrupted_by_timer_during_alert(
        self, window, tmp_path, monkeypatch, isolated_singletons
    ):
        """Simulates the periodic timer firing during the dialog.

        Before the fix, file_browser.directory pointed at new_dir during the modal,
        so a store_info_cache() call would overwrite new_dir's cached sort with the
        current (old) sort.  After the fix the write targets old_dir only.
        """
        old_dir = str(tmp_path / "old")
        new_dir = str(tmp_path / "new")
        os.makedirs(old_dir)
        os.makedirs(new_dir)

        cache = isolated_singletons
        cache.set(new_dir, "sort_by", SortBy.MODIFY_TIME.get_text())

        window.file_browser.directory = old_dir
        window.file_browser.sort_by = SortBy.NAME

        def _timer_fires(*args, **kwargs):
            # Reproduce what store_info_cache() does for the sort key.
            cache.set(
                window.file_browser.directory,
                "sort_by",
                window.file_browser.get_sort_by().get_text(),
            )
            return True

        self._patch_for_large_dir(monkeypatch, window, _timer_fires)
        window._check_large_directory_before_load(new_dir)

        assert cache.get(new_dir, "sort_by") == SortBy.MODIFY_TIME.get_text(), (
            "new_dir's cached sort was overwritten during the alert dialog"
        )

    def test_new_dir_marked_confirmed_after_user_accepts(self, window, tmp_path, monkeypatch):
        old_dir = str(tmp_path / "old")
        new_dir = str(tmp_path / "new")
        os.makedirs(old_dir)
        os.makedirs(new_dir)
        window.file_browser.directory = old_dir

        from files.file_browser import FileBrowser
        self._patch_for_large_dir(monkeypatch, window, lambda *a, **kw: True)
        window._check_large_directory_before_load(new_dir)

        assert new_dir in FileBrowser.have_confirmed_directories

    def test_returns_true_when_user_cancels(self, window, tmp_path, monkeypatch):
        old_dir = str(tmp_path / "old")
        new_dir = str(tmp_path / "new")
        os.makedirs(old_dir)
        os.makedirs(new_dir)
        window.file_browser.directory = old_dir

        self._patch_for_large_dir(monkeypatch, window, lambda *a, **kw: False)
        result = window._check_large_directory_before_load(new_dir)

        assert result is True

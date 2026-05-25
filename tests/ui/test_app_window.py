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

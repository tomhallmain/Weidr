"""
Integration test for the Random Purge feature.

Verifies the full end-to-end flow:
  - Group compare runs and produces multi-file similarity groups
  - random_purge_groups() confirms via dialog, deletes all but one file per group
  - Exactly one file per group survives on disk
  - Group state is fully cleared after the operation
"""

import os

import pytest
from PySide6.QtWidgets import QApplication

from tests.ui.app_window_fixtures import _teardown_app_window
from ui.app_window.app_window import AppWindow
from utils.config import config
from utils.constants import CompareMode


@pytest.fixture
def window_with_compare_dir(qtbot, compare_colors_dir):
    """AppWindow loaded with the solid-color compare directory."""
    win = AppWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win.set_base_dir(compare_colors_dir["dir"])
    qtbot.waitUntil(lambda: win.base_dir == compare_colors_dir["dir"], timeout=3000)
    yield win, compare_colors_dir
    _teardown_app_window(win)


def _run_group_compare(win, qtbot, immediate_compare_debounce):
    immediate_compare_debounce(win.search_ctrl)
    win.sidebar_panel.search_media_path_box.clear()
    win.sidebar_panel.search_text_box.clear()
    win.search_ctrl.set_search()
    qtbot.waitUntil(lambda: not win.search_ctrl.is_compare_running(), timeout=20000)


class TestRandomPurge:

    def test_random_purge_keeps_one_file_per_group_and_clears_state(
        self,
        window_with_compare_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        """
        After random_purge_groups():
          - Exactly one file per original group survives on disk
          - file_groups and files_matched are empty
        """
        win, colors = window_with_compare_dir

        monkeypatch.setattr(config, "delete_instantly", True)
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)

        # Return True for any alert (suppresses "no match" dialogs and confirms purge).
        monkeypatch.setitem(win.app_actions._actions, "_alert", lambda *a, **k: True)

        _run_group_compare(win, qtbot, immediate_compare_debounce)

        file_groups = win.compare_manager.file_groups
        assert file_groups, "group compare produced no groups"
        assert any(len(g) > 1 for g in file_groups.values()), (
            "at least one group must have more than one file for purge to be meaningful"
        )

        group_count = len(file_groups)
        grouped_paths = [
            path for group in file_groups.values() for path in group.keys()
        ]

        win.compare_manager.random_purge_groups()
        QApplication.processEvents()

        # State must be fully cleared.
        assert win.compare_manager.file_groups == {}, (
            "file_groups should be empty after random purge"
        )
        assert win.compare_manager.files_matched == [], (
            "files_matched should be empty after random purge"
        )

        # One file per group must survive; all others must be gone.
        survivors = [p for p in grouped_paths if os.path.exists(p)]
        assert len(survivors) == group_count, (
            f"Expected exactly {group_count} surviving file(s) (one per group), "
            f"got {len(survivors)}"
        )

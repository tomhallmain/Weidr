"""
Integration tests for the Random Purge feature.

Covers two layers:

1. End-to-end (AppWindow): group compare runs, random_purge_groups() confirms,
   deletes all but one file per group, and clears group state.

2. Quality filter (CompareWrapper direct): when a group contains only one image
   that meets the minimum dimension threshold, that image is always kept and the
   undersized ones are deleted.
"""

import os

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication
from unittest.mock import MagicMock

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


class TestRandomPurgeQualityFilter:
    """CompareWrapper directly: the dimension gate must favour the qualifying image."""

    def test_only_image_above_min_dimension_is_kept(self, tmp_path):
        """
        Group has three 48×48 images (below the 120px threshold) and one 200×200
        image (above it).  random_purge_groups() must always keep the large image
        regardless of random ordering.
        """
        from compare.compare_wrapper import CompareWrapper

        # Create one large (qualifying) and three small (disqualified) images.
        large = str(tmp_path / "large.png")
        smalls = [str(tmp_path / f"small_{i}.png") for i in range(3)]
        Image.new("RGB", (200, 200), (100, 100, 100)).save(large)
        for path in smalls:
            Image.new("RGB", (48, 48), (100, 100, 100)).save(path)

        all_paths = [large] + smalls

        deleted = []

        app_actions = MagicMock()
        app_actions.alert.return_value = True
        app_actions.delete.side_effect = lambda path, **kw: (
            deleted.append(path), os.remove(path)
        )

        wrapper = CompareWrapper(master=None, compare_mode=None, app_actions=app_actions)
        wrapper.file_groups = {0: {p: 1.0 for p in all_paths}}
        wrapper.group_indexes = [0]

        wrapper.random_purge_groups()

        survivors = [p for p in all_paths if os.path.exists(p)]
        assert survivors == [large], (
            f"Expected only the large image to survive, got: {survivors}"
        )

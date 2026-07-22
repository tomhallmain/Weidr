"""
UI test for RecentDirectoryWindow's "open a new window and immediately run a
search there" feature -- Ctrl+A (WindowLauncher.open_secondary_compare_window
with the current media, then RecentDirectoryWindow._on_directory_selected()
once the user picks a target directory).

This is the simpler, pre-existing sibling of FileActionsWindow's "Search in
New Window" (docs/file-actions-search-in-new-window.md): a real base_dir
instead of a file list, but the exact same do_search=True /
WindowManager.add_secondary_window() path -- and the exact same bug.
AppWindow.__init__'s do_search path called SearchController.set_search() with
no arguments, which reads the sidebar's search_media_path_box; nothing
populates that box for a brand-new window, so it always ran a GROUP compare
instead of a search regardless of media_path. No test existed for this
feature at all, which is presumably why the regression went unnoticed.
"""
from __future__ import annotations

import os

import pytest

from files.recent_directories import RecentDirectories
from tests.ui.app_window_fixtures import _teardown_app_window, make_png
from ui.app_window.window_manager import WindowManager
from ui.files.recent_directory_window_qt import RecentDirectoryWindow
from utils.constants import Mode


@pytest.fixture(autouse=True)
def _clear_recent_directories():
    RecentDirectories.directories.clear()
    RecentDirectories.directory_history.clear()
    RecentDirectories.last_set_directory = None
    RecentDirectories.last_comparison_directory = None
    yield
    RecentDirectories.directories.clear()
    RecentDirectories.directory_history.clear()
    RecentDirectories.last_set_directory = None
    RecentDirectories.last_comparison_directory = None


class TestRecentDirectoryWindowRunCompareMedia:
    def test_selecting_a_directory_opens_new_window_and_runs_search_not_group_compare(
        self, window_with_dir, qtbot, bypass_password, tmp_path, monkeypatch
    ):
        win, media_dir = window_with_dir
        query_media = os.path.join(media_dir, "img01.png")
        assert os.path.isfile(query_media)

        target_dir = str(tmp_path / "target_dir")
        os.makedirs(target_dir, exist_ok=True)
        make_png(os.path.join(target_dir, "candidate.png"))

        run_calls = []
        monkeypatch.setattr(
            "compare.compare_manager.CompareManager.run",
            lambda self, args: run_calls.append(args),
        )

        # Mirrors WindowLauncher.open_secondary_compare_window(run_compare_media=...)
        # (Ctrl+A) constructing the picker with the query media already set.
        picker = RecentDirectoryWindow(
            win, True, win.app_actions, base_dir=win.get_base_dir(),
            run_compare_media=query_media,
        )
        qtbot.addWidget(picker)

        ids_before = set(WindowManager._secondary_toplevels.keys())
        # Simulates the user picking target_dir from the list.
        picker._on_directory_selected(target_dir)
        new_ids = set(WindowManager._secondary_toplevels.keys()) - ids_before

        try:
            assert len(new_ids) == 1, "selecting a directory must open exactly one new window"
            new_window = WindowManager._secondary_toplevels[new_ids.pop()]
            try:
                qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
                assert run_calls[0].mode == Mode.SEARCH, (
                    "must run a SEARCH using query_media, not a GROUP comparison "
                    "over target_dir"
                )
                assert run_calls[0].search_media_path == query_media
            finally:
                _teardown_app_window(new_window)
        finally:
            picker.close()
            picker.deleteLater()

    def test_selecting_a_directory_with_no_run_compare_media_just_opens_window(
        self, window_with_dir, qtbot, tmp_path
    ):
        """Sanity check: run_compare_media="" (Ctrl+W, no media) must not try
        to run any search -- new_window(base_dir=_dir) only."""
        win, _media_dir = window_with_dir
        target_dir = str(tmp_path / "target_dir2")
        os.makedirs(target_dir, exist_ok=True)
        make_png(os.path.join(target_dir, "candidate.png"))

        picker = RecentDirectoryWindow(
            win, True, win.app_actions, base_dir=win.get_base_dir(), run_compare_media="",
        )
        qtbot.addWidget(picker)

        ids_before = set(WindowManager._secondary_toplevels.keys())
        picker._on_directory_selected(target_dir)
        new_ids = set(WindowManager._secondary_toplevels.keys()) - ids_before

        try:
            assert len(new_ids) == 1
            new_window = WindowManager._secondary_toplevels[new_ids.pop()]
            try:
                qtbot.wait(100)
                assert new_window.mode == Mode.BROWSE
            finally:
                _teardown_app_window(new_window)
        finally:
            picker.close()
            picker.deleteLater()

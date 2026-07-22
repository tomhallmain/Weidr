"""
UI tests for FileActionsWindow's "Search in New Window" feature
(docs/file-actions-search-in-new-window.md).

TestOpenSearchInNewWindow covers FileActionsWindow._open_search_in_new_window()'s
own logic (file-list construction from action history, active-media
resolution) with WindowManager.add_secondary_window mocked out, so no second
real window is spawned. base_dir is always None here -- deliberately: the
file list spans wherever its files actually live, not one directory, and a
derived directory would incorrectly scope base_dir-keyed prevalidations to
whatever directory happened to be most common (see design doc section 5).
The compare engine's own internal base_dir gets a designated substitute
elsewhere -- see TestFileListCompareArgs in
tests/ui/test_search_controller_advanced.py.

TestAddSecondaryWindowFileList and TestDoSearchMediaPathBug cover two real
bugs found when this feature was manually tested end to end (not just
mocked): (1) opening a new window with do_search=True never worked for ANY
caller -- AppWindow.__init__ just called set_search() with no arguments,
which reads the sidebar's search_media_path_box, and nothing populates that
box for a brand-new window, so it always ran a GROUP compare instead of a
search. This affected the pre-existing recent_directory_window_qt.py flow
too, not just this feature. (2) A window with no base_dir had no real
FileBrowser content, so normal browsing (arrow keys, etc.) didn't work while
or after a file-list compare ran -- fixed by activating FileBrowser's
pre-PySide6-port "explicit file list" mode (use_file_paths_json), previously
only a global config toggle with no per-window activation path or UI hook.
"""
from __future__ import annotations

import os
from datetime import datetime

import pytest

from files.file_action import FileAction
from tests.ui.app_window_fixtures import _teardown_app_window
from ui.app_window.window_manager import WindowManager
from ui.files.file_actions_window_qt import FileActionsWindow
from utils.config import config
from utils.constants import Mode
from utils.utils import Utils


@pytest.fixture
def file_actions_window(window_with_dir, qtbot, monkeypatch):
    """Open FileActionsWindow on a real AppWindow, history cleared per test."""
    win, media_dir = window_with_dir
    FileAction.action_history.clear()

    monkeypatch.setattr(
        "ui.image.media_details.MediaDetails.open_temp_media_canvas",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "files.marked_files.MarkedFiles.move_marks_to_dir_static",
        lambda **_kwargs: None,
    )

    win.window_launcher.open_file_actions_window()
    qtbot.waitUntil(
        lambda: FileActionsWindow._instance is not None
        and FileActionsWindow._instance.isVisible(),
        timeout=5000,
    )
    actions_win = FileActionsWindow._instance
    qtbot.addWidget(actions_win)
    yield actions_win, win, media_dir
    FileAction.action_history.clear()
    try:
        actions_win.close()
        actions_win.deleteLater()
    finally:
        FileActionsWindow._instance = None


class TestOpenSearchInNewWindow:
    def test_builds_file_list_from_existing_destination_files_only(
        self, file_actions_window, monkeypatch
    ):
        actions_win, win, media_dir = file_actions_window
        existing1 = os.path.join(media_dir, "img01.png")
        existing2 = os.path.join(media_dir, "img02.png")
        missing = os.path.join(media_dir, "does_not_exist.png")
        assert os.path.isfile(existing1) and os.path.isfile(existing2)
        assert not os.path.isfile(missing)

        FileAction.action_history.extend([
            FileAction(Utils.move_file, os.path.dirname(existing1),
                       original_marks=["/src/a.png"], new_files=[existing1]),
            FileAction(Utils.copy_file, os.path.dirname(existing2),
                       original_marks=["/src/b.png"], new_files=[existing2]),
            # Duplicate destination (e.g. moved then copied elsewhere and back)
            FileAction(Utils.move_file, os.path.dirname(existing1),
                       original_marks=["/src/c.png"], new_files=[existing1]),
            # Destination file no longer exists on disk
            FileAction(Utils.move_file, os.path.dirname(missing),
                       original_marks=["/src/d.png"], new_files=[missing]),
        ])
        # Delete actions have no new_files -- naturally excluded
        FileAction.add_delete_action("/src/e.png")

        query_path = os.path.join(media_dir, "img03.png")
        monkeypatch.setattr(win.app_actions, "get_active_media_filepath", lambda: query_path)

        captured = {}
        monkeypatch.setattr(
            WindowManager,
            "add_secondary_window",
            classmethod(lambda cls, **kwargs: captured.update(kwargs)),
        )

        actions_win._open_search_in_new_window()

        assert captured["file_list"] == [existing1, existing2]
        assert captured["media_path"] == query_path
        assert captured["do_search"] is True
        # Deliberately no base_dir -- see the module docstring.
        assert captured["base_dir"] is None

    def test_no_active_media_toasts_and_does_not_open_window(
        self, file_actions_window, monkeypatch
    ):
        actions_win, win, media_dir = file_actions_window
        FileAction.action_history.append(
            FileAction(Utils.move_file, media_dir, new_files=[os.path.join(media_dir, "img01.png")])
        )
        monkeypatch.setattr(win.app_actions, "get_active_media_filepath", lambda: None)
        opened = []
        monkeypatch.setattr(
            WindowManager, "add_secondary_window", classmethod(lambda cls, **kwargs: opened.append(kwargs))
        )
        toasts = []
        monkeypatch.setattr(win.app_actions, "toast", lambda msg, **k: toasts.append(msg))

        actions_win._open_search_in_new_window()

        assert opened == []
        assert len(toasts) == 1

    def test_no_existing_destination_files_toasts_and_does_not_open_window(
        self, file_actions_window, monkeypatch
    ):
        actions_win, win, media_dir = file_actions_window
        FileAction.action_history.append(
            FileAction(Utils.move_file, media_dir, new_files=[os.path.join(media_dir, "gone.png")])
        )
        monkeypatch.setattr(
            win.app_actions, "get_active_media_filepath",
            lambda: os.path.join(media_dir, "img01.png"),
        )
        opened = []
        monkeypatch.setattr(
            WindowManager, "add_secondary_window", classmethod(lambda cls, **kwargs: opened.append(kwargs))
        )
        toasts = []
        monkeypatch.setattr(win.app_actions, "toast", lambda msg, **k: toasts.append(msg))

        actions_win._open_search_in_new_window()

        assert opened == []
        assert len(toasts) == 1


class TestAddSecondaryWindowFileList:
    """WindowManager.add_secondary_window's file_list plumbing: both the
    reuse-an-existing-window and open-a-new-window branches must thread
    file_list through to SearchController.set_search() when do_search is set,
    and a new window must get real FileBrowser content (via
    FileBrowser.enable_explicit_file_list())."""

    def test_reuse_existing_window_passes_file_list_to_set_search(
        self, window_with_dir, monkeypatch
    ):
        win, media_dir = window_with_dir
        monkeypatch.setattr(config, "always_open_new_windows", False)
        captured = {}

        def _fake_set_search(event=None, file_list=None):
            captured["file_list"] = file_list

        monkeypatch.setattr(win.search_ctrl, "set_search", _fake_set_search)

        file_list = [
            os.path.join(media_dir, "img01.png"),
            os.path.join(media_dir, "img02.png"),
        ]
        WindowManager.add_secondary_window(
            base_dir=media_dir,
            media_path=file_list[0],
            do_search=True,
            file_list=file_list,
        )

        assert captured.get("file_list") == file_list

    def test_none_base_dir_never_reuses_another_none_base_dir_window(
        self, window_with_dir, qtbot
    ):
        """Two independent file-list searches (base_dir=None) must never be
        merged into the same window -- "no directory" isn't a real,
        distinguishing location to match on."""
        win, media_dir = window_with_dir
        file_list_1 = [os.path.join(media_dir, "img01.png")]
        file_list_2 = [os.path.join(media_dir, "img02.png")]

        ids_before = set(WindowManager._secondary_toplevels.keys())
        WindowManager.add_secondary_window(base_dir=None, do_search=False, file_list=file_list_1)
        ids_after_first = set(WindowManager._secondary_toplevels.keys())
        WindowManager.add_secondary_window(base_dir=None, do_search=False, file_list=file_list_2)
        ids_after_second = set(WindowManager._secondary_toplevels.keys())

        new_after_first = ids_after_first - ids_before
        new_after_second = ids_after_second - ids_after_first
        try:
            assert len(new_after_first) == 1
            assert len(new_after_second) == 1  # a second, distinct window -- not reused
        finally:
            qtbot.wait(50)
            for wid in new_after_first | new_after_second:
                _teardown_app_window(WindowManager._secondary_toplevels[wid])

    def test_new_window_activates_explicit_file_list_browsing(self, window_with_dir, qtbot):
        win, media_dir = window_with_dir
        file_list = [os.path.join(media_dir, "img01.png"), os.path.join(media_dir, "img02.png")]

        ids_before = set(WindowManager._secondary_toplevels.keys())
        WindowManager.add_secondary_window(base_dir=None, do_search=False, file_list=file_list)
        new_ids = set(WindowManager._secondary_toplevels.keys()) - ids_before
        assert len(new_ids) == 1
        new_window = WindowManager._secondary_toplevels[new_ids.pop()]
        try:
            qtbot.waitUntil(lambda: new_window.file_browser.use_file_paths_json, timeout=2000)
            assert set(new_window.file_browser.get_files()) == set(file_list)
            # Browsing works with no real directory -- base_dir stays unset,
            # never surfaced to prevalidation's directory-profile matching.
            assert new_window.get_base_dir() is None
        finally:
            _teardown_app_window(new_window)


class TestDoSearchMediaPathBug:
    """The bug this feature's manual test caught: AppWindow.__init__'s
    do_search path called set_search() with no arguments, which reads the
    sidebar's search_media_path_box -- nothing populates that box for a
    brand-new window, so args.not_searching() was always True and it ran a
    GROUP compare instead of a search, regardless of media_path. This
    predates the file-list feature: recent_directory_window_qt.py already
    called add_secondary_window(do_search=True, media_path=...) with a real
    base_dir via the same broken path."""

    def test_new_window_do_search_runs_search_not_group_compare(
        self, window_with_dir, qtbot, bypass_password, tmp_path, monkeypatch
    ):
        from tests.ui.app_window_fixtures import make_png

        win, media_dir = window_with_dir
        other_dir = str(tmp_path / "other_dir")
        os.makedirs(other_dir, exist_ok=True)
        make_png(os.path.join(other_dir, "other.png"))
        query_media = os.path.join(media_dir, "img01.png")

        run_calls = []
        monkeypatch.setattr(
            "compare.compare_manager.CompareManager.run",
            lambda self, args: run_calls.append(args),
        )

        ids_before = set(WindowManager._secondary_toplevels.keys())
        WindowManager.add_secondary_window(
            base_dir=other_dir, media_path=query_media, do_search=True,
        )
        new_ids = set(WindowManager._secondary_toplevels.keys()) - ids_before
        assert len(new_ids) == 1
        new_window = WindowManager._secondary_toplevels[new_ids.pop()]
        try:
            qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=3000)
            assert run_calls[0].mode == Mode.SEARCH
            assert run_calls[0].search_media_path == query_media
        finally:
            _teardown_app_window(new_window)

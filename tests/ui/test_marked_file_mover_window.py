"""UI smoke tests for MarkedFileMover (move-marks window)."""

import os

import pytest
from PySide6.QtWidgets import QApplication, QPushButton, QScrollArea

from files.marked_files import MarkedFiles
from ui.files.marked_file_mover_qt import MarkedFileMover
from utils.translations import I18N

_tr = I18N._


def _close_mark_mover() -> None:
    win = MarkedFileMover._current_window
    if win is not None:
        try:
            win.close_windows()
            win.close()
            win.deleteLater()
        except RuntimeError:
            pass
    MarkedFileMover._current_window = None
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture
def marked_files_state():
    """Snapshot MarkedFiles class state for isolation."""
    saved = {
        "file_marks": MarkedFiles.file_marks[:],
        "mark_target_dirs": MarkedFiles.mark_target_dirs[:],
        "last_set_target_dir": MarkedFiles.last_set_target_dir,
    }
    yield saved
    MarkedFiles.file_marks = saved["file_marks"][:]
    MarkedFiles.mark_target_dirs = saved["mark_target_dirs"][:]
    MarkedFiles.last_set_target_dir = saved["last_set_target_dir"]
    _close_mark_mover()


@pytest.fixture(autouse=True)
def _mark_mover_cleanup():
    yield
    _close_mark_mover()


@pytest.fixture
def mark_mover_window(window_with_dir, qtbot, bypass_password, marked_files_state):
    """Open GUI MarkedFileMover with one marked file."""
    app_win, media_dir = window_with_dir
    media_path = app_win.file_browser.get_files()[0]
    MarkedFiles.file_marks = [media_path]
    MarkedFiles.mark_target_dirs = [media_dir]
    MarkedFiles.last_set_target_dir = media_dir

    app_win.file_marks_ctrl.open_move_marks_window(open_gui=True)
    qtbot.waitUntil(
        lambda: MarkedFileMover._current_window is not None
        and MarkedFileMover._current_window.isVisible(),
        timeout=5000,
    )
    mover = MarkedFileMover._current_window
    qtbot.addWidget(mover)
    qtbot.waitExposed(mover, timeout=3000)

    yield mover, app_win, media_path, media_dir


class TestMarkedFileMoverWindowOpen:
    def test_open_move_marks_window_shows_gui_controls(
        self, mark_mover_window
    ):
        mover, _app_win, media_path, media_dir = mark_mover_window

        assert mover._is_gui
        assert mover._current_media == media_path
        assert mover._scroll is not None
        assert media_dir in mover._filtered_target_dirs

        buttons = {btn.text() for btn in mover.findChildren(QPushButton)}
        assert _tr("MOVE") in buttons
        assert _tr("COPY") in buttons
        assert _tr("DELETE") in buttons
        assert "1" in mover.windowTitle()

    def test_second_open_focuses_existing_window(
        self, mark_mover_window, qtbot
    ):
        mover, app_win, _media_path, _media_dir = mark_mover_window
        first_id = id(mover)

        app_win.file_marks_ctrl.open_move_marks_window(open_gui=True)
        qtbot.waitUntil(
            lambda: MarkedFileMover._current_window is not None,
            timeout=3000,
        )

        assert id(MarkedFileMover._current_window) == first_id

    def test_show_window_builds_scroll_area_for_targets(
        self, window_with_dir, qtbot, bypass_password, marked_files_state
    ):
        app_win, media_dir = window_with_dir
        parent_target = os.path.dirname(media_dir)
        MarkedFiles.file_marks = [os.path.join(media_dir, "img01.png")]
        MarkedFiles.mark_target_dirs = [media_dir, parent_target]

        MarkedFileMover.show_window(
            app_win,
            True,
            True,
            MarkedFiles.file_marks[0],
            app_win.mode,
            app_win.app_actions,
            base_dir=media_dir,
        )
        qtbot.waitUntil(
            lambda: MarkedFileMover._current_window is not None,
            timeout=5000,
        )
        mover = MarkedFileMover._current_window
        scrolls = mover.findChildren(QScrollArea)
        assert scrolls
        assert len(mover._filtered_target_dirs) >= 1

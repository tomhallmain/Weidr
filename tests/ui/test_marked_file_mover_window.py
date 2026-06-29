"""UI smoke tests for MarkedFileMover (move-marks window)."""

import os

import pytest
from PySide6.QtWidgets import QPushButton, QScrollArea

from files.marked_files import MarkedFiles
from ui.files.marked_file_mover_qt import MarkedFileMover
from utils.translations import _
from utils.utils import Utils

_tr = _


def _close_mark_mover() -> None:
    win = MarkedFileMover._current_window
    MarkedFileMover._current_window = None
    if win is not None:
        try:
            win.close()
        except RuntimeError:
            pass


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


class TestMarkedFileMoverProgress:
    """Tests for the progress dialog used during large mark-move operations."""

    THRESHOLD = MarkedFileMover.LARGE_FILE_OP_SHOW_PROGRESS_THRESHOLD  # 100

    @pytest.fixture(autouse=True)
    def _restore_cancelled_flag(self):
        old = MarkedFiles.is_cancelled_action
        yield
        MarkedFiles.is_cancelled_action = old

    def test_below_threshold_returns_none_tuple(self, qtbot):
        """No dialog or callback for operations below the show-progress threshold."""
        progress, callback = MarkedFileMover.build_marks_progress(
            None, self.THRESHOLD - 1, Utils.move_file
        )
        assert progress is None
        assert callback is None

    def test_at_threshold_creates_dialog_and_callback(self, qtbot):
        """At exactly the threshold a dialog and callback are returned."""
        progress, callback = MarkedFileMover.build_marks_progress(
            None, self.THRESHOLD, Utils.move_file
        )
        qtbot.addWidget(progress)
        assert progress is not None
        assert callback is not None

    def test_move_label_mentions_moving(self, qtbot):
        """Progress dialog for a move operation says 'Moving'."""
        progress, _ = MarkedFileMover.build_marks_progress(None, 200, Utils.move_file)
        qtbot.addWidget(progress)
        assert "Moving" in progress.labelText()

    def test_copy_label_mentions_copying(self, qtbot):
        """Progress dialog for a copy operation says 'Copying'."""
        progress, _ = MarkedFileMover.build_marks_progress(None, 200, Utils.copy_file)
        qtbot.addWidget(progress)
        assert "Copying" in progress.labelText()

    def test_generic_label_when_move_func_is_none(self, qtbot):
        """Progress dialog with no move_func falls back to 'Processing'."""
        progress, _ = MarkedFileMover.build_marks_progress(None, 200, None)
        qtbot.addWidget(progress)
        assert "Processing" in progress.labelText()

    def test_callback_sets_cancelled_flag_when_dialog_wasCanceled(
        self, qtbot, monkeypatch
    ):
        """Callback sets is_cancelled_action=True when the dialog reports cancellation."""
        MarkedFiles.is_cancelled_action = False
        progress, callback = MarkedFileMover.build_marks_progress(
            None, 200, Utils.move_file
        )
        qtbot.addWidget(progress)
        monkeypatch.setattr(progress, "wasCanceled", lambda: True)
        callback(50, 200)
        assert MarkedFiles.is_cancelled_action is True

    def test_callback_does_not_set_cancelled_flag_when_dialog_not_cancelled(
        self, qtbot, monkeypatch
    ):
        """Callback leaves is_cancelled_action untouched when the dialog is not cancelled."""
        MarkedFiles.is_cancelled_action = False
        progress, callback = MarkedFileMover.build_marks_progress(
            None, 200, Utils.move_file
        )
        qtbot.addWidget(progress)
        monkeypatch.setattr(progress, "wasCanceled", lambda: False)
        callback(50, 200)
        assert MarkedFiles.is_cancelled_action is False

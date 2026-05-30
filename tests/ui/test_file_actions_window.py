"""UI tests for FileActionsWindow (history list, filters, delete rows)."""

from datetime import datetime

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from files.file_action import FileAction
from ui.files.file_actions_window_qt import FileActionsWindow
from utils.constants import FileActionKind
from utils.translations import _
from utils.utils import Utils

_tr = _


def _label_texts(widget) -> list[str]:
    return [lbl.text() for lbl in widget.findChildren(QLabel) if lbl.text()]


def _seed_history() -> None:
    FileAction.action_history.clear()
    FileAction.action_history.extend(
        [
            FileAction(
                Utils.move_file,
                "/browse/out",
                original_marks=["/browse/in/moved.png"],
                new_files=["/browse/out/moved.png"],
                auto=False,
                timestamp=datetime(2025, 6, 1, 10, 0, 0),
            ),
            FileAction(
                Utils.copy_file,
                "/browse/copy",
                original_marks=["/browse/in/copied.png"],
                new_files=["/browse/copy/copied.png"],
                auto=False,
                timestamp=datetime(2025, 6, 1, 11, 0, 0),
            ),
        ]
    )
    FileAction.add_delete_action("/browse/in/removed.jpg", auto=False)


@pytest.fixture
def file_actions_window(window_with_dir, qtbot, monkeypatch):
    """Open FileActionsWindow with history seeded and heavy callbacks stubbed."""
    win, _ = window_with_dir
    _seed_history()

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
    yield actions_win, win
    try:
        actions_win.close()
        actions_win.deleteLater()
    finally:
        FileActionsWindow._instance = None
        app = QApplication.instance()
        if app is not None:
            app.processEvents()


class TestFileActionsWindowOpen:
    def test_open_file_actions_window_shows_history_entries(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        qtbot.waitUntil(
            lambda: any("moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )
        texts = _label_texts(actions_win)
        assert any("moved.png" in t for t in texts)
        assert any("copied.png" in t for t in texts)
        assert any("removed.jpg" in t for t in texts)
        assert any(t == _tr("Delete") for t in texts)
        assert any(t == _tr("Move") for t in texts)
        assert any(t == _tr("Copy") for t in texts)

class TestFileActionsWindowFilters:
    def test_delete_type_filter_shows_only_delete_rows(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        combo = actions_win._type_combo
        delete_index = next(
            i for i in range(combo.count()) if combo.itemData(i) == FileActionKind.DELETE
        )
        combo.setCurrentIndex(delete_index)

        qtbot.waitUntil(
            lambda: any("removed.jpg" in t for t in _label_texts(actions_win))
            and not any("moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )

    def test_clear_history_empties_the_list(self, file_actions_window, qtbot):
        actions_win, _app_win = file_actions_window
        qtbot.waitUntil(
            lambda: any("moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )

        actions_win._clear_action_history()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        assert len(FileAction.action_history) == 0
        qtbot.waitUntil(
            lambda: not any("moved.png" in t for t in _label_texts(actions_win)),
            timeout=2000,
        )

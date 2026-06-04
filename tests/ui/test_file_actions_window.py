"""UI tests for FileActionsWindow (history list, filters, delete rows)."""

from datetime import datetime

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

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
            FileAction(
                Utils.move_file,
                "/browse/auto_out",
                original_marks=["/browse/in/auto_moved.png"],
                new_files=["/browse/auto_out/auto_moved.png"],
                auto=True,
                timestamp=datetime(2025, 6, 1, 12, 0, 0),
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


def _set_combo_by_data(combo: QComboBox, target_data) -> None:
    """Select the combo item whose userData equals target_data."""
    for i in range(combo.count()):
        if combo.itemData(i) == target_data:
            combo.setCurrentIndex(i)
            return
    raise ValueError(f"No combo item with data {target_data!r}")


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

    def test_initiator_combo_has_any_user_auto_options(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        combo = actions_win._initiator_combo
        labels = [combo.itemText(i) for i in range(combo.count())]
        assert _tr("Any") in labels
        assert _tr("User") in labels
        assert _tr("Auto") in labels
        assert combo.count() == 3


class TestFileActionsWindowFilters:
    def test_delete_type_filter_shows_only_delete_rows(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        _set_combo_by_data(actions_win._type_combo, FileActionKind.DELETE)

        qtbot.waitUntil(
            lambda: any("removed.jpg" in t for t in _label_texts(actions_win))
            and not any("moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )

    def test_move_type_filter_shows_only_move_rows(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        _set_combo_by_data(actions_win._type_combo, FileActionKind.MOVE)

        qtbot.waitUntil(
            lambda: any("moved.png" in t for t in _label_texts(actions_win))
            and not any("copied.png" in t for t in _label_texts(actions_win))
            and not any("removed.jpg" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )

    def test_copy_type_filter_shows_only_copy_rows(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        _set_combo_by_data(actions_win._type_combo, FileActionKind.COPY)

        qtbot.waitUntil(
            lambda: any("copied.png" in t for t in _label_texts(actions_win))
            and not any("moved.png" in t for t in _label_texts(actions_win))
            and not any("removed.jpg" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )

    def test_user_initiator_filter_shows_only_user_rows(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        qtbot.waitUntil(
            lambda: any("moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )
        _set_combo_by_data(actions_win._initiator_combo, False)

        qtbot.waitUntil(
            lambda: any("moved.png" in t for t in _label_texts(actions_win))
            and not any("auto_moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )
        texts = _label_texts(actions_win)
        assert any("copied.png" in t for t in texts)
        assert any("removed.jpg" in t for t in texts)

    def test_auto_initiator_filter_shows_only_auto_rows(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        qtbot.waitUntil(
            lambda: any("moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )
        _set_combo_by_data(actions_win._initiator_combo, True)

        qtbot.waitUntil(
            lambda: any("auto_moved.png" in t for t in _label_texts(actions_win))
            and not any("copied.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )
        texts = _label_texts(actions_win)
        assert not any("moved.png" == t for t in texts)
        assert not any("removed.jpg" in t for t in texts)

    def test_any_initiator_filter_restores_all_rows(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        _set_combo_by_data(actions_win._initiator_combo, True)
        qtbot.waitUntil(
            lambda: not any("copied.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )
        _set_combo_by_data(actions_win._initiator_combo, None)

        qtbot.waitUntil(
            lambda: any("copied.png" in t for t in _label_texts(actions_win))
            and any("auto_moved.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )

    def test_initiator_filter_label_shown_for_user_filter(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        _set_combo_by_data(actions_win._initiator_combo, False)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        assert actions_win._filter_label.isVisible()
        assert _tr("user only") in actions_win._filter_label.text()

    def test_initiator_filter_label_shown_for_auto_filter(
        self, file_actions_window, qtbot
    ):
        actions_win, _app_win = file_actions_window
        _set_combo_by_data(actions_win._initiator_combo, True)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        assert actions_win._filter_label.isVisible()
        assert _tr("auto only") in actions_win._filter_label.text()

    def test_clear_history_resets_initiator_filter(self, file_actions_window, qtbot):
        actions_win, _app_win = file_actions_window
        _set_combo_by_data(actions_win._initiator_combo, True)
        qtbot.waitUntil(
            lambda: not any("copied.png" in t for t in _label_texts(actions_win)),
            timeout=3000,
        )

        actions_win._clear_action_history()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        assert actions_win._initiator_filter is None
        assert actions_win._initiator_combo.currentIndex() == 0

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

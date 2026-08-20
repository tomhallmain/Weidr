"""
UI tests for file-handling interceptor rules.

Two layers:
  TestFileInterceptorRulesWindow — the editor dialog opened via WindowLauncher
                                   (empty state, listing, reorder, remove).
  TestInterceptorRulesDuringTransfer — the rules actually intercepting a real
                                   MarkedFiles move against files on disk.
"""

import os

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from files.file_action import FileAction
from files.file_interceptor_rule import (
    FileInterceptorRule,
    InterceptorAppliesTo,
    InterceptorBehavior,
    InterceptorTransformOp,
)
from files.file_interceptor_rules_manager import FileInterceptorRulesManager
from files.marked_files import MarkedFiles
from tests.helpers import isolated_app_info_cache
from ui.files.file_interceptor_rules_window_qt import FileInterceptorRulesWindow
from utils.translations import _
from utils.utils import Utils

# Tests unpack `win, _ = window_with_dir`, which would shadow the translation
# function for the rest of the test body.
_tr = _


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_rules_window() -> FileInterceptorRulesWindow | None:
    """The live singleton, not a topLevelWidgets() scan.

    A window closed by a previous test can linger in topLevelWidgets() until
    its deleteLater() is reaped, and scanning would return that stale, empty
    window instead of the one this test just opened.
    """
    return FileInterceptorRulesWindow._instance


def _close_all_rules_windows() -> None:
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        if isinstance(widget, FileInterceptorRulesWindow):
            try:
                widget.close()
                widget.deleteLater()
            except RuntimeError:
                pass
    FileInterceptorRulesWindow._instance = None
    app.processEvents()


def _label_texts(widget) -> list[str]:
    if widget is None:
        return []
    try:
        return [lbl.text() for lbl in widget.findChildren(QLabel) if lbl.text()]
    except RuntimeError:
        return []


def _buttons_with_text(widget, text: str) -> list[QPushButton]:
    try:
        return [b for b in widget.findChildren(QPushButton) if b.text() == text]
    except RuntimeError:
        return []


@pytest.fixture(autouse=True)
def _rules_window_cleanup():
    yield
    _close_all_rules_windows()


@pytest.fixture(autouse=True)
def isolated_rules():
    """Snapshot the process-wide rule list so tests cannot leak into each other."""
    saved = FileInterceptorRulesManager.rules[:]
    FileInterceptorRulesManager.rules = []
    yield
    FileInterceptorRulesManager.rules = saved


@pytest.fixture
def isolated_marks():
    """Snapshot MarkedFiles/FileAction class state touched by a transfer."""
    saved = {
        "file_marks": MarkedFiles.file_marks[:],
        "previous_marks": MarkedFiles.previous_marks[:],
        "last_set_target_dir": MarkedFiles.last_set_target_dir,
        "delete_lock": MarkedFiles.delete_lock,
        "action_history": FileAction.action_history[:],
    }
    yield
    MarkedFiles.file_marks = saved["file_marks"]
    MarkedFiles.previous_marks = saved["previous_marks"]
    MarkedFiles.last_set_target_dir = saved["last_set_target_dir"]
    MarkedFiles.delete_lock = saved["delete_lock"]
    FileAction.action_history = saved["action_history"]


def _open_rules_window(win, qtbot) -> FileInterceptorRulesWindow:
    win.window_launcher.open_file_interceptor_rules_window()
    qtbot.waitUntil(lambda: _find_rules_window() is not None, timeout=5000)
    rules_win = _find_rules_window()
    qtbot.addWidget(rules_win)
    qtbot.waitExposed(rules_win, timeout=3000)
    return rules_win


# ---------------------------------------------------------------------------
# Editor dialog
# ---------------------------------------------------------------------------

class TestFileInterceptorRulesWindow:
    def test_open_shows_empty_state_when_no_rules(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        rules_win = _open_rules_window(win, qtbot)

        assert _tr("No interceptor rules configured.") in _label_texts(rules_win)

    def test_open_lists_configured_rule(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(
                name="needs rename",
                match_filename_patterns=["IMG_"],
                behavior=InterceptorBehavior.BLOCK,
            )
        ]
        rules_win = _open_rules_window(win, qtbot)

        texts = " ".join(_label_texts(rules_win))
        assert "needs rename" in texts
        # The summary line renders the rule's own conditions.
        assert "IMG_" in texts

    def test_move_down_reorders_and_persists(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(name="first"),
            FileInterceptorRule(name="second"),
        ]
        rules_win = _open_rules_window(win, qtbot)

        down_buttons = _buttons_with_text(rules_win, "↓")
        # Only the last row's down button is disabled, so the first is clickable.
        assert down_buttons and down_buttons[0].isEnabled()
        down_buttons[0].click()
        qtbot.waitUntil(
            lambda: FileInterceptorRulesManager.rules[0].name == "second", timeout=3000
        )

        assert [r.name for r in FileInterceptorRulesManager.rules] == ["second", "first"]
        stored = isolated_app_info_cache().get_meta(
            FileInterceptorRulesManager.RULES_KEY, default_val=[]
        )
        assert [r["name"] for r in stored] == ["second", "first"]

    def test_first_row_cannot_move_up_and_last_cannot_move_down(
        self, window_with_dir, qtbot
    ):
        win, _ = window_with_dir
        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(name="first"),
            FileInterceptorRule(name="second"),
        ]
        rules_win = _open_rules_window(win, qtbot)

        up_buttons = _buttons_with_text(rules_win, "↑")
        down_buttons = _buttons_with_text(rules_win, "↓")
        assert not up_buttons[0].isEnabled()
        assert not down_buttons[-1].isEnabled()

    def test_remove_button_deletes_rule(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(name="doomed"),
            FileInterceptorRule(name="survivor"),
        ]
        rules_win = _open_rules_window(win, qtbot)

        remove_buttons = _buttons_with_text(rules_win, "×")
        assert len(remove_buttons) == 2
        remove_buttons[0].click()
        qtbot.waitUntil(
            lambda: len(FileInterceptorRulesManager.rules) == 1, timeout=3000
        )

        assert [r.name for r in FileInterceptorRulesManager.rules] == ["survivor"]

    def test_reopening_reuses_the_singleton(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        first = _open_rules_window(win, qtbot)
        win.window_launcher.open_file_interceptor_rules_window()
        QApplication.instance().processEvents()

        assert _find_rules_window() is first


# ---------------------------------------------------------------------------
# Interception during a real transfer
# ---------------------------------------------------------------------------

class TestInterceptorRulesDuringTransfer:
    def _capture_toasts(self, win, monkeypatch) -> list:
        """Record toast text instead of building real toast widgets."""
        toasts: list = []
        monkeypatch.setitem(
            win.app_actions._actions,
            "toast",
            lambda message, *a, **kw: toasts.append(message),
        )
        return toasts

    def test_block_rule_prevents_move_and_leaves_file_in_place(
        self, window_with_dir, qtbot, monkeypatch, isolated_marks, tmp_path
    ):
        win, media_dir = window_with_dir
        target = tmp_path / "target"
        target.mkdir()
        media_path = win.file_browser.get_files()[0]
        toasts = self._capture_toasts(win, monkeypatch)

        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(
                name="rename first",
                block_message="Rename via RefacDir first",
                behavior=InterceptorBehavior.BLOCK,
            )
        ]
        MarkedFiles.file_marks = [media_path]

        MarkedFiles.move_marks_to_dir_static(
            win.app_actions, target_dir=str(target), move_func=Utils.move_file
        )

        assert os.path.isfile(media_path), "blocked file must stay where it was"
        assert list(target.iterdir()) == []
        assert any("Rename via RefacDir first" in t for t in toasts)

    def test_blocked_file_stays_marked(
        self, window_with_dir, qtbot, monkeypatch, isolated_marks, tmp_path
    ):
        win, media_dir = window_with_dir
        target = tmp_path / "target"
        target.mkdir()
        media_path = win.file_browser.get_files()[0]
        self._capture_toasts(win, monkeypatch)
        monkeypatch.setattr(
            "files.marked_files.config.clear_marks_with_errors_after_move", False
        )

        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(name="blocker", behavior=InterceptorBehavior.BLOCK)
        ]
        MarkedFiles.file_marks = [media_path]

        MarkedFiles.move_marks_to_dir_static(
            win.app_actions, target_dir=str(target), move_func=Utils.move_file
        )

        assert MarkedFiles.file_marks == [media_path]

    def test_copy_only_rule_does_not_block_a_move(
        self, window_with_dir, qtbot, monkeypatch, isolated_marks, tmp_path
    ):
        win, media_dir = window_with_dir
        target = tmp_path / "target"
        target.mkdir()
        media_path = win.file_browser.get_files()[0]
        self._capture_toasts(win, monkeypatch)

        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(
                name="copy blocker",
                behavior=InterceptorBehavior.BLOCK,
                applies_to=InterceptorAppliesTo.COPY_ONLY,
            )
        ]
        MarkedFiles.file_marks = [media_path]

        MarkedFiles.move_marks_to_dir_static(
            win.app_actions, target_dir=str(target), move_func=Utils.move_file
        )

        assert os.path.isfile(str(target / os.path.basename(media_path)))

    def test_transform_rule_moves_the_converted_file(
        self, window_with_dir, qtbot, monkeypatch, isolated_marks, tmp_path
    ):
        win, media_dir = window_with_dir
        target = tmp_path / "target"
        target.mkdir()
        media_path = win.file_browser.get_files()[0]
        assert media_path.lower().endswith(".png")
        self._capture_toasts(win, monkeypatch)

        deleted: list = []
        monkeypatch.setitem(
            win.app_actions._actions,
            "delete",
            lambda path, *a, **kw: deleted.append(path),
        )

        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(
                name="jpg on the way out",
                behavior=InterceptorBehavior.TRANSFORM,
                transform_op=InterceptorTransformOp.CONVERT_TO_JPG,
                delete_original_after_transform=True,
            )
        ]
        MarkedFiles.file_marks = [media_path]

        MarkedFiles.move_marks_to_dir_static(
            win.app_actions, target_dir=str(target), move_func=Utils.move_file
        )

        moved = [p.name for p in target.iterdir()]
        assert moved == [os.path.splitext(os.path.basename(media_path))[0] + ".jpg"]
        # The untransformed original is handed to the app's delete action.
        assert deleted == [media_path]

    def test_transform_rule_on_copy_never_deletes_the_original(
        self, window_with_dir, qtbot, monkeypatch, isolated_marks, tmp_path
    ):
        win, media_dir = window_with_dir
        target = tmp_path / "target"
        target.mkdir()
        media_path = win.file_browser.get_files()[0]
        self._capture_toasts(win, monkeypatch)

        deleted: list = []
        monkeypatch.setitem(
            win.app_actions._actions,
            "delete",
            lambda path, *a, **kw: deleted.append(path),
        )

        FileInterceptorRulesManager.rules = [
            FileInterceptorRule(
                name="jpg on the way out",
                behavior=InterceptorBehavior.TRANSFORM,
                transform_op=InterceptorTransformOp.CONVERT_TO_JPG,
                delete_original_after_transform=True,
            )
        ]
        MarkedFiles.file_marks = [media_path]

        MarkedFiles.move_marks_to_dir_static(
            win.app_actions, target_dir=str(target), move_func=Utils.copy_file
        )

        assert deleted == []
        assert os.path.isfile(media_path)

    def test_no_rules_leaves_transfer_untouched(
        self, window_with_dir, qtbot, monkeypatch, isolated_marks, tmp_path
    ):
        win, media_dir = window_with_dir
        target = tmp_path / "target"
        target.mkdir()
        media_path = win.file_browser.get_files()[0]
        self._capture_toasts(win, monkeypatch)
        MarkedFiles.file_marks = [media_path]

        MarkedFiles.move_marks_to_dir_static(
            win.app_actions, target_dir=str(target), move_func=Utils.move_file
        )

        assert [p.name for p in target.iterdir()] == [os.path.basename(media_path)]
        assert not os.path.isfile(media_path)

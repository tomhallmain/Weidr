"""
UI tests for the auto-sorted media confirmation gate (Ctrl+Shift+K).

Covers the keybinding wiring, the category gate itself (skip / decline /
accept), and the small editor dialog that configures the category set.
"""

import os
from datetime import datetime

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QLineEdit

from files.auto_sort_confirmation import AutoSortConfirmation
from files.file_action import FileAction
from lib.aware_entry_qt import AwareEntry
from ui.files.auto_sort_confirmation_window_qt import AutoSortConfirmationWindow
from ui.image.auto_sort_open_gate import open_last_auto_sorted_media
from utils.utils import Utils


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_confirm_categories():
    saved = set(AutoSortConfirmation.confirm_categories)
    AutoSortConfirmation.confirm_categories = set()
    yield
    AutoSortConfirmation.confirm_categories = saved


@pytest.fixture(autouse=True)
def isolated_action_history():
    saved = FileAction.action_history[:]
    FileAction.action_history.clear()
    yield
    FileAction.action_history.clear()
    FileAction.action_history.extend(saved)


@pytest.fixture
def sorted_media(tmp_path):
    """A file recorded as auto-moved into a 'landscapes' category directory."""
    from tests.ui.app_window_fixtures import make_png

    category_dir = tmp_path / "sorted" / "landscapes"
    category_dir.mkdir(parents=True)
    media_path = str(category_dir / "photo.png")
    make_png(media_path, (10, 120, 60))

    FileAction.update_history(
        FileAction(
            Utils.move_file,
            str(category_dir),
            original_marks=[str(tmp_path / "photo.png")],
            new_files=[media_path],
            auto=True,
            timestamp=datetime(2026, 1, 1, 9, 0, 0),
        )
    )
    return media_path


@pytest.fixture
def canvas_calls(monkeypatch):
    """Record open_temp_media_canvas calls instead of building a real canvas."""
    from ui.image.media_details import MediaDetails

    calls: list = []
    monkeypatch.setattr(
        MediaDetails,
        "open_temp_media_canvas",
        staticmethod(lambda *a, **kw: calls.append((a, kw))),
    )
    return calls


def _set_alert_result(monkeypatch, result: bool) -> list:
    """Force the confirmation dialog's answer; returns a list recording calls."""
    import ui.app_window.notification_controller as _nc

    calls: list = []
    monkeypatch.setattr(
        _nc, "qt_alert", lambda *a, **kw: calls.append((a, kw)) or result
    )
    return calls


def _find_shortcut(win, key_str: str):
    target = QKeySequence(key_str)
    for shortcut in win.key_binding_mgr._shortcuts:
        if shortcut.key().matches(target) == QKeySequence.SequenceMatch.ExactMatch:
            return shortcut
    return None


# ---------------------------------------------------------------------------
# Keybinding wiring
# ---------------------------------------------------------------------------

class TestCtrlShiftKWiring:
    def test_shortcut_is_registered(self, window):
        assert _find_shortcut(window, "Ctrl+Shift+K") is not None

    def test_shortcut_opens_last_auto_sorted_media(
        self, window_with_dir, sorted_media, canvas_calls, monkeypatch
    ):
        win, _ = window_with_dir
        # The binding is guarded against text-entry focus; keep it unsuppressed.
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", False)
        _find_shortcut(win, "Ctrl+Shift+K").activated.emit()

        assert len(canvas_calls) == 1
        assert sorted_media in canvas_calls[0][0]

    def test_shortcut_respects_the_category_gate(
        self, window_with_dir, sorted_media, canvas_calls, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", False)
        AutoSortConfirmation.set_confirm_required("landscapes", True)
        _set_alert_result(monkeypatch, False)

        _find_shortcut(win, "Ctrl+Shift+K").activated.emit()

        assert canvas_calls == []


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------

class TestAutoSortOpenGate:
    def test_ungated_category_opens_without_a_dialog(
        self, window_with_dir, sorted_media, canvas_calls, monkeypatch
    ):
        win, _ = window_with_dir
        alerts = _set_alert_result(monkeypatch, True)

        open_last_auto_sorted_media(win, win.app_actions)

        assert len(canvas_calls) == 1
        assert alerts == []

    def test_gated_category_declined_does_not_open(
        self, window_with_dir, sorted_media, canvas_calls, monkeypatch
    ):
        win, _ = window_with_dir
        AutoSortConfirmation.set_confirm_required("landscapes", True)
        alerts = _set_alert_result(monkeypatch, False)

        open_last_auto_sorted_media(win, win.app_actions)

        assert canvas_calls == []
        assert len(alerts) == 1

    def test_gated_category_accepted_opens(
        self, window_with_dir, sorted_media, canvas_calls, monkeypatch
    ):
        win, _ = window_with_dir
        AutoSortConfirmation.set_confirm_required("landscapes", True)
        alerts = _set_alert_result(monkeypatch, True)

        open_last_auto_sorted_media(win, win.app_actions)

        assert len(canvas_calls) == 1
        assert sorted_media in canvas_calls[0][0]
        assert len(alerts) == 1

    def test_gate_matches_category_case_insensitively(
        self, window_with_dir, sorted_media, canvas_calls, monkeypatch
    ):
        win, _ = window_with_dir
        AutoSortConfirmation.set_confirm_required("LANDSCAPES", True)
        _set_alert_result(monkeypatch, False)

        open_last_auto_sorted_media(win, win.app_actions)

        assert canvas_calls == []

    def test_declining_leaves_the_file_alone(
        self, window_with_dir, sorted_media, canvas_calls, monkeypatch
    ):
        """Declining is a plain no-op -- nothing is moved back or unmarked."""
        win, _ = window_with_dir
        AutoSortConfirmation.set_confirm_required("landscapes", True)
        _set_alert_result(monkeypatch, False)
        history_before = FileAction.action_history[:]

        open_last_auto_sorted_media(win, win.app_actions)

        assert os.path.isfile(sorted_media)
        assert FileAction.action_history == history_before

    def test_no_auto_action_in_history_is_a_noop(
        self, window_with_dir, canvas_calls, monkeypatch
    ):
        win, _ = window_with_dir
        # Clear after the window fixture: building an AppWindow loads the
        # cached action history, which the autouse fixture ran too early to see.
        FileAction.action_history.clear()
        alerts = _set_alert_result(monkeypatch, True)

        open_last_auto_sorted_media(win, win.app_actions)

        assert canvas_calls == []
        assert alerts == []


# ---------------------------------------------------------------------------
# Category editor dialog
# ---------------------------------------------------------------------------

def _find_confirm_window() -> AutoSortConfirmationWindow | None:
    """The live singleton — see the note in test_file_interceptor_rules_window."""
    return AutoSortConfirmationWindow._instance


@pytest.fixture(autouse=True)
def _confirm_window_cleanup():
    yield
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        if isinstance(widget, AutoSortConfirmationWindow):
            try:
                widget.close()
                widget.deleteLater()
            except RuntimeError:
                pass
    AutoSortConfirmationWindow._instance = None
    app.processEvents()


class TestAutoSortConfirmationWindow:
    def _open(self, win, qtbot) -> AutoSortConfirmationWindow:
        win.window_launcher.open_auto_sort_confirmation_window()
        qtbot.waitUntil(lambda: _find_confirm_window() is not None, timeout=5000)
        confirm_win = _find_confirm_window()
        qtbot.addWidget(confirm_win)
        qtbot.waitExposed(confirm_win, timeout=3000)
        return confirm_win

    def test_field_prefills_with_configured_categories(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, _ = window_with_dir
        AutoSortConfirmation.set_categories(["portraits", "landscapes"])
        confirm_win = self._open(win, qtbot)

        field = confirm_win.findChild(QLineEdit)
        assert field is not None
        assert field.text() == "landscapes, portraits"

    def test_saving_replaces_the_category_set(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, _ = window_with_dir
        AutoSortConfirmation.set_categories(["stale"])
        monkeypatch.setitem(
            win.app_actions._actions, "toast", lambda *a, **kw: None
        )
        confirm_win = self._open(win, qtbot)

        confirm_win.findChild(QLineEdit).setText(" Landscapes , portraits ")
        confirm_win._save()

        assert AutoSortConfirmation.get_categories() == ["landscapes", "portraits"]
        assert not AutoSortConfirmation.is_confirm_required("stale")

    def test_saved_categories_gate_the_open(
        self, window_with_dir, qtbot, sorted_media, canvas_calls, monkeypatch
    ):
        """End-to-end: what the dialog saves is what the gate later enforces."""
        win, _ = window_with_dir
        monkeypatch.setitem(
            win.app_actions._actions, "toast", lambda *a, **kw: None
        )
        confirm_win = self._open(win, qtbot)
        confirm_win.findChild(QLineEdit).setText("landscapes")
        confirm_win._save()

        _set_alert_result(monkeypatch, False)
        open_last_auto_sorted_media(win, win.app_actions)

        assert canvas_calls == []

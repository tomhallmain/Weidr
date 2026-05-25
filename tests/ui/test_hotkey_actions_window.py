"""
Tests for HotkeyActionsWindow keyboard behaviour.

The grid labels rows as Shift-T / Shift-0..9 because those are the **runtime**
mark hotkeys on the main window (KeyBindingManager: Shift+digit runs marks,
plain digit also runs marks). This dialog is different: its QShortcuts are
Shift+T and bare digits 0-9 to **configure** targets (SET_HOTKEY_ACTIONS).

Keyboard tests send bare digit keys (no Shift) so they hit this window's
shortcuts, not the parent's Shift+digit bindings.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from files.file_action import FileAction
from ui.files.hotkey_actions_window_qt import HotkeyActionsWindow
from utils.translations import I18N
from utils.utils import Utils

_tr = I18N._


def _focus_hotkey_window(hw: HotkeyActionsWindow) -> None:
    hw.raise_()
    hw.activateWindow()
    hw.setFocus()


@pytest.fixture(autouse=True)
def _hotkey_window_cleanup():
    yield
    win = HotkeyActionsWindow._instance
    if win is not None:
        try:
            win.close()
            win.deleteLater()
        except RuntimeError:
            pass
    HotkeyActionsWindow._instance = None
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture
def hotkey_win(window_with_dir, qtbot, bypass_password):
    """Open a standalone HotkeyActionsWindow with stub callbacks."""
    win, _ = window_with_dir
    permanent_calls = []
    hotkey_calls = []
    marks_calls = []

    def _track_marks(number, shift_pressed=False):
        marks_calls.append((number, shift_pressed))

    # Parent AppWindow also binds 0-9 and Shift+0-9; track if events leak there.
    win.file_marks_ctrl.run_hotkey_marks_action = _track_marks

    hw = HotkeyActionsWindow(
        master=win,
        app_actions=win.app_actions,
        set_permanent_action_callback=lambda: permanent_calls.append(True),
        set_hotkey_action_callback=lambda hotkey_override=None: hotkey_calls.append(
            hotkey_override
        ),
    )
    qtbot.addWidget(hw)
    hw.show()
    qtbot.waitExposed(hw)
    _focus_hotkey_window(hw)

    yield hw, permanent_calls, hotkey_calls, marks_calls

    if hw.isVisible():
        hw.close()
    HotkeyActionsWindow._instance = None


class TestHotkeyActionsWindowConstruction:
    def test_window_is_visible_after_show(self, hotkey_win, qtbot):
        hw, *_ = hotkey_win
        assert hw.isVisible()

    def test_window_instance_registered(self, hotkey_win):
        hw, *_ = hotkey_win
        assert HotkeyActionsWindow._instance is hw

    def test_header_labels_present(self, hotkey_win):
        hw, *_ = hotkey_win
        texts = [lbl.text() for lbl in hw.findChildren(QLabel) if lbl.text()]
        assert any(_tr("Key name") in t for t in texts)
        assert any(_tr("Target directory") in t for t in texts)

    def test_set_buttons_present_for_each_hotkey(self, hotkey_win):
        """One Set button per hotkey row: T + 0-9 = 11 rows."""
        hw, *_ = hotkey_win
        set_btns = [
            b for b in hw.findChildren(QPushButton) if b.text() == _tr("Set")
        ]
        assert len(set_btns) == 11

    def test_key_name_column_shows_runtime_shift_prefix(self, hotkey_win):
        """Display names match main-window mark chords (Shift-T, Shift-1, ...)."""
        hw, *_ = hotkey_win
        texts = [lbl.text() for lbl in hw.findChildren(QLabel) if lbl.text()]
        shift_labels = [t for t in texts if t.startswith("Shift-")]
        assert len(shift_labels) == 11


class TestHotkeyActionsWindowKeyboard:
    def test_escape_closes_window(self, hotkey_win, qtbot):
        hw, *_ = hotkey_win
        _focus_hotkey_window(hw)
        qtbot.keyClick(hw, Qt.Key.Key_Escape)
        qtbot.waitUntil(lambda: not hw.isVisible(), timeout=2000)
        assert HotkeyActionsWindow._instance is None

    def test_escape_clears_instance_reference(self, hotkey_win, qtbot):
        hw, *_ = hotkey_win
        _focus_hotkey_window(hw)
        qtbot.keyClick(hw, Qt.Key.Key_Escape)
        qtbot.waitUntil(lambda: HotkeyActionsWindow._instance is None, timeout=2000)

    def test_bare_digit_key_triggers_set_hotkey_callback(
        self, hotkey_win, qtbot
    ):
        hw, _, hotkey_calls, marks_calls = hotkey_win
        _focus_hotkey_window(hw)
        qtbot.keyClick(hw, Qt.Key.Key_3)
        assert hotkey_calls == [3], (
            f"Configure shortcut is bare '3', not Shift+3; calls={hotkey_calls}, "
            f"marks={marks_calls}"
        )

    def test_bare_digit_0_triggers_set_hotkey_callback(self, hotkey_win, qtbot):
        hw, _, hotkey_calls, marks_calls = hotkey_win
        _focus_hotkey_window(hw)
        qtbot.keyClick(hw, Qt.Key.Key_0)
        assert hotkey_calls == [0]
        assert marks_calls == []

    def test_shift_digit_does_not_trigger_set_callback(self, hotkey_win, qtbot):
        """Shift+digit is not wired on this dialog (only bare digits configure)."""
        hw, permanent_calls, hotkey_calls, _ = hotkey_win
        _focus_hotkey_window(hw)
        qtbot.keyClick(hw, Qt.Key.Key_3, Qt.KeyboardModifier.ShiftModifier)
        assert hotkey_calls == []
        assert permanent_calls == []

    def test_all_bare_digits_trigger_set_hotkey_override(
        self, hotkey_win, qtbot
    ):
        hw, _, hotkey_calls, marks_calls = hotkey_win
        _focus_hotkey_window(hw)
        for d in range(10):
            key = getattr(Qt.Key, f"Key_{d}")
            qtbot.keyClick(hw, key)
        assert sorted(hotkey_calls) == list(range(10))
        assert marks_calls == []

    def test_shift_t_triggers_permanent_set_callback(
        self, hotkey_win, qtbot
    ):
        hw, permanent_calls, hotkey_calls, _ = hotkey_win
        _focus_hotkey_window(hw)
        qtbot.keyClick(hw, Qt.Key.Key_T, Qt.KeyboardModifier.ShiftModifier)
        assert permanent_calls == [True]
        assert hotkey_calls == []


class TestHotkeyActionsWindowClose:
    def test_close_clears_instance(self, hotkey_win, qtbot):
        hw, *_ = hotkey_win
        hw.close()
        qtbot.waitUntil(lambda: HotkeyActionsWindow._instance is None, timeout=2000)

    def test_reads_file_action_hotkey_state(self, window_with_dir, qtbot, bypass_password):
        """Rows resolve targets from FileAction, not FileActionsWindow."""
        win, media_dir = window_with_dir
        FileAction.hotkey_actions[1] = FileAction(Utils.move_file, media_dir)
        try:
            hw = HotkeyActionsWindow(
                master=win,
                app_actions=win.app_actions,
                set_permanent_action_callback=lambda: None,
                set_hotkey_action_callback=lambda **_kwargs: None,
            )
            qtbot.addWidget(hw)
            texts = [lbl.text() for lbl in hw.findChildren(QLabel) if lbl.text()]
            assert any("Shift-1" in t for t in texts)
            assert any(media_dir in t for t in texts)
        finally:
            FileAction.hotkey_actions.pop(1, None)

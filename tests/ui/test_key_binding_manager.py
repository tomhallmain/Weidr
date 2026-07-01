"""
Tests for KeyBindingManager and the keybinding layer of AppWindow.

Separated from test_app_window.py (which covers basic Left/Right/Home/End
navigation) to isolate the guard mechanism, shortcut registration, and
non-navigation shortcut dispatch.

Organisation:
  TestGuardWrapper          — pure unit: _guarded() wrapper logic
  TestShortcutRegistration  — all expected key sequences are present and unique
  TestGuardIntegration      — guard suppresses / fires based on AwareEntry focus
  TestViewShortcuts         — F11, Ctrl+H, Ctrl+S, Ctrl+Shift+G (observable state)
  TestNavigationExtended    — PgUp / PgDown (not in test_app_window.py)
  TestDigitShortcuts        — 0–9 and Shift+0–9 hotkey marks shortcuts
  TestFileMarkShortcuts     — Shift+M (mark toggle) via observable side effects
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

from files.marked_files import MarkedFiles
from lib.aware_entry_qt import AwareEntry
from ui.app_window.key_binding_manager import KeyBindingManager
from utils.constants import ViewMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_shortcut(km: KeyBindingManager, key_str: str) -> bool:
    """True if *km* has a shortcut registered for *key_str*."""
    target = QKeySequence(key_str)
    for s in km._shortcuts:
        if s.key().matches(target) == QKeySequence.SequenceMatch.ExactMatch:
            return True
    return False


# ---------------------------------------------------------------------------
# TestGuardWrapper — pure unit, no Qt window needed
# ---------------------------------------------------------------------------

class TestGuardWrapper:
    def test_fires_when_no_entry_has_focus(self, monkeypatch):
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", False)
        calls = []
        wrapped = KeyBindingManager._guarded(lambda: calls.append(True))
        wrapped()
        assert len(calls) == 1

    def test_suppressed_when_entry_has_focus(self, monkeypatch):
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", True)
        calls = []
        wrapped = KeyBindingManager._guarded(lambda: calls.append(True))
        wrapped()
        assert len(calls) == 0

    def test_flag_checked_at_call_time_not_wrap_time(self, monkeypatch):
        """Guard reads an_entry_has_focus at invocation, not at wrapping."""
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", False)
        calls = []
        wrapped = KeyBindingManager._guarded(lambda: calls.append(True))
        # Change the flag after wrapping but before calling
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", True)
        wrapped()
        assert len(calls) == 0

    def test_multiple_calls_respect_current_flag(self, monkeypatch):
        calls = []
        wrapped = KeyBindingManager._guarded(lambda: calls.append(True))

        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", False)
        wrapped()
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", True)
        wrapped()
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", False)
        wrapped()

        assert len(calls) == 2  # only the two calls when focus was False


# ---------------------------------------------------------------------------
# TestShortcutRegistration — key sequences present and unique
# ---------------------------------------------------------------------------

class TestShortcutRegistration:
    def test_large_number_of_shortcuts_registered(self, window):
        # _bind_all registers well over 50 shortcuts (navigation + file ops +
        # view + marks + digits + window management).
        assert len(window.key_binding_mgr._shortcuts) >= 50

    def test_no_duplicate_key_sequences(self, window):
        km = window.key_binding_mgr
        all_keys = [s.key().toString() for s in km._shortcuts]
        dupes = [k for k in all_keys if all_keys.count(k) > 1]
        assert not dupes, f"Duplicate shortcuts registered: {set(dupes)}"

    def test_navigation_keys_all_present(self, window):
        km = window.key_binding_mgr
        for key in ["Left", "Right", "Home", "End", "PgUp", "PgDown"]:
            assert _has_shortcut(km, key), f"{key!r} not registered"

    def test_shift_nav_keys_present(self, window):
        km = window.key_binding_mgr
        assert _has_shortcut(km, "Shift+Left")
        assert _has_shortcut(km, "Shift+Right")
        assert _has_shortcut(km, "Shift+Backspace")

    def test_ctrl_keys_present(self, window):
        km = window.key_binding_mgr
        for key in ["Ctrl+H", "Ctrl+S", "Ctrl+Q", "Ctrl+B", "Ctrl+Tab",
                    "Ctrl+C", "Ctrl+Z", "Ctrl+R"]:
            assert _has_shortcut(km, key), f"{key!r} not registered"

    def test_shift_letter_keys_present(self, window):
        km = window.key_binding_mgr
        for key in ["Shift+M", "Shift+V", "Shift+Delete", "Shift+D",
                    "Shift+R", "Shift+S", "Shift+A", "Shift+H"]:
            assert _has_shortcut(km, key), f"{key!r} not registered"

    def test_all_digit_shortcuts_present(self, window):
        km = window.key_binding_mgr
        for d in range(10):
            assert _has_shortcut(km, str(d)), f"digit {d} not registered"

    def test_all_shift_digit_shortcuts_present(self, window):
        km = window.key_binding_mgr
        for d in range(10):
            assert _has_shortcut(km, f"Shift+{d}"), f"Shift+{d} not registered"

    def test_function_keys_present(self, window):
        km = window.key_binding_mgr
        assert _has_shortcut(km, "F11")
        assert _has_shortcut(km, "F1")

    def test_ctrl_shift_combos_present(self, window):
        km = window.key_binding_mgr
        for key in ["Ctrl+Shift+G", "Ctrl+Shift+N", "Ctrl+Shift+C",
                    "Ctrl+Shift+D", "Ctrl+Return"]:
            assert _has_shortcut(km, key), f"{key!r} not registered"

    def test_interactive_crop_and_box_shortcuts_present(self, window):
        km = window.key_binding_mgr
        assert _has_shortcut(km, "Ctrl+Shift+P"), "Interactive Crop shortcut not registered"
        assert _has_shortcut(km, "Ctrl+Shift+B"), "Interactive Box shortcut not registered"


# ---------------------------------------------------------------------------
# TestGuardIntegration — full Qt path: keyClick + AwareEntry focus state
# ---------------------------------------------------------------------------

class TestGuardIntegration:
    """Verify that guarded shortcuts are suppressed when an entry has focus.

    Shift+M (add_or_remove_mark) is used as the test vehicle because its
    side effect — MarkedFiles.file_marks grows by one — is directly observable
    without requiring disk I/O or dialogs.
    """

    def test_guarded_shortcut_fires_when_no_entry_has_focus(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", False)
        assert MarkedFiles.file_marks == []
        qtbot.keyClick(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        assert len(MarkedFiles.file_marks) == 1

    def test_guarded_shortcut_suppressed_when_entry_has_focus(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", True)
        qtbot.keyClick(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        assert MarkedFiles.file_marks == [], (
            "Shift+M should be suppressed while an AwareEntry has focus"
        )

    def test_unguarded_shortcut_fires_even_with_entry_focus(
        self, window, qtbot, monkeypatch
    ):
        """F11 is registered with guarded=False, so it fires unconditionally."""
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", True)
        assert not window.fullscreen
        qtbot.keyClick(window, Qt.Key.Key_F11)
        assert window.fullscreen


# ---------------------------------------------------------------------------
# TestViewShortcuts — observable state changes for view-layer shortcuts
# ---------------------------------------------------------------------------

class TestViewShortcuts:
    def test_f11_toggles_fullscreen(self, window, qtbot):
        assert not window.fullscreen
        qtbot.keyClick(window, Qt.Key.Key_F11)
        assert window.fullscreen
        qtbot.keyClick(window, Qt.Key.Key_F11)
        assert not window.fullscreen

    def test_ctrl_h_toggles_sidebar_visibility(self, window, qtbot):
        assert window.sidebar_panel.isVisible()
        qtbot.keyClick(window, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier)
        assert not window.sidebar_panel.isVisible()
        qtbot.keyClick(window, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier)
        assert window.sidebar_panel.isVisible()

    def test_ctrl_s_starts_slideshow(self, window, qtbot):
        assert not window.slideshow_config.slideshow_running
        qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        assert window.slideshow_config.slideshow_running
        # Second press cycles to show_new_media mode (not slideshow_running)
        qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        assert not window.slideshow_config.slideshow_running

    def test_ctrl_shift_g_toggles_masonry_view(self, window, qtbot):
        assert window.view_mode == ViewMode.FULL
        qtbot.keyClick(
            window, Qt.Key.Key_G,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        )
        assert window.view_mode == ViewMode.MASONRY
        qtbot.keyClick(
            window, Qt.Key.Key_G,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        )
        assert window.view_mode == ViewMode.FULL

    def test_shift_f_also_toggles_fullscreen(self, window, qtbot):
        """Shift+F is an alias for F11 (guarded)."""
        assert not window.fullscreen
        qtbot.keyClick(window, Qt.Key.Key_F, Qt.KeyboardModifier.ShiftModifier)
        assert window.fullscreen


# ---------------------------------------------------------------------------
# TestNavigationExtended — PgUp / PgDown (not in test_app_window.py)
# ---------------------------------------------------------------------------

class TestNavigationExtended:
    def test_pgdown_changes_current_file(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        first = win.media_path
        assert first is not None
        qtbot.keyClick(win, Qt.Key.Key_PageDown)
        qtbot.waitUntil(lambda: win.media_path != first, timeout=2000)
        assert win.media_path != first

    def test_pgup_changes_current_file(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        # Move forward first so there's somewhere to go back to.
        first = win.media_path
        qtbot.keyClick(win, Qt.Key.Key_Right)
        qtbot.waitUntil(lambda: win.media_path != first, timeout=2000)
        second = win.media_path
        qtbot.keyClick(win, Qt.Key.Key_PageUp)
        qtbot.waitUntil(lambda: win.media_path != second, timeout=2000)
        assert win.media_path != second

    def test_pgdown_pgup_round_trip(self, window_with_dir, qtbot):
        """PgDown then PgUp returns to the starting file (3-file set)."""
        win, _ = window_with_dir
        origin = win.media_path
        qtbot.keyClick(win, Qt.Key.Key_PageDown)
        qtbot.waitUntil(lambda: win.media_path != origin, timeout=2000)
        mid = win.media_path
        qtbot.keyClick(win, Qt.Key.Key_PageUp)
        qtbot.waitUntil(lambda: win.media_path != mid, timeout=2000)
        assert win.media_path == origin


# ---------------------------------------------------------------------------
# TestDigitShortcuts — 0–9 dispatch via lambda-based closures
# ---------------------------------------------------------------------------

class TestDigitShortcuts:
    """
    Digit shortcuts are bound via lambda closures:

        lambda _n=i: app.file_marks_ctrl.run_hotkey_marks_action(
            number=_n, shift_pressed=False)

    The lambda performs a fresh attribute lookup on file_marks_ctrl at call
    time, so monkeypatching the instance attribute after window creation works.
    """

    def _patch_marks_action(self, win, monkeypatch):
        calls = []
        monkeypatch.setattr(
            win.file_marks_ctrl,
            "run_hotkey_marks_action",
            lambda number, shift_pressed=False: calls.append((number, shift_pressed)),
        )
        return calls

    def test_digit_1_calls_run_hotkey_marks_action(
        self, window, qtbot, monkeypatch
    ):
        calls = self._patch_marks_action(window, monkeypatch)
        qtbot.keyClick(window, Qt.Key.Key_1)
        assert any(n == 1 and not s for n, s in calls), (
            f"Expected run_hotkey_marks_action(1, False); calls: {calls}"
        )

    def test_digit_0_calls_run_hotkey_marks_action(
        self, window, qtbot, monkeypatch
    ):
        calls = self._patch_marks_action(window, monkeypatch)
        qtbot.keyClick(window, Qt.Key.Key_0)
        assert any(n == 0 and not s for n, s in calls), (
            f"Expected run_hotkey_marks_action(0, False); calls: {calls}"
        )

    def test_shift_digit_1_passes_shift_pressed_true(
        self, window, qtbot, monkeypatch
    ):
        calls = self._patch_marks_action(window, monkeypatch)
        # Shift+<digit> is registered with guarded=False, so fires regardless
        # of AwareEntry focus.
        qtbot.keyClick(window, Qt.Key.Key_1, Qt.KeyboardModifier.ShiftModifier)
        assert any(n == 1 and s for n, s in calls), (
            f"Expected run_hotkey_marks_action(1, True); calls: {calls}"
        )

    def test_each_digit_passes_correct_number(self, window, qtbot, monkeypatch):
        calls = self._patch_marks_action(window, monkeypatch)
        for d in range(10):
            key = getattr(Qt.Key, f"Key_{d}")
            qtbot.keyClick(window, key)
        numbers_called = [n for n, _ in calls]
        for d in range(10):
            assert d in numbers_called, f"Digit {d} not seen in calls: {calls}"

    def test_digit_shortcuts_suppressed_when_entry_has_focus(
        self, window, qtbot, monkeypatch
    ):
        """Digit shortcuts are guarded — suppressed when an entry has focus."""
        calls = self._patch_marks_action(window, monkeypatch)
        monkeypatch.setattr(AwareEntry, "an_entry_has_focus", True)
        qtbot.keyClick(window, Qt.Key.Key_5)
        assert not calls, "Digit 5 should be suppressed while an entry has focus"


# ---------------------------------------------------------------------------
# TestFileMarkShortcuts — Shift+M and Ctrl+C observable side effects
# ---------------------------------------------------------------------------

class TestFileMarkShortcuts:
    def test_shift_m_adds_mark_for_current_file(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        assert MarkedFiles.file_marks == []
        qtbot.keyClick(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        assert len(MarkedFiles.file_marks) == 1
        assert MarkedFiles.file_marks[0] == win.media_path

    def test_shift_m_removes_mark_on_second_press(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        qtbot.keyClick(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        assert len(MarkedFiles.file_marks) == 1
        qtbot.keyClick(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        assert MarkedFiles.file_marks == []

    def test_ctrl_c_is_registered_as_shortcut(self, window):
        """Ctrl+C (copy_marks_list) is present in the registered shortcuts."""
        assert _has_shortcut(window.key_binding_mgr, "Ctrl+C")

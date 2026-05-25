"""
Integration tests for the full CompareWrapper pipeline via the AppWindow UI.

Tests exercise:
  - Opening CompareSettingsWindow, adjusting threshold, clicking Apply
  - Running COLOR_MATCHING search (find images similar to a query image)
  - Running COLOR_MATCHING group compare (cluster all images by similarity)
  - Navigating group results with show_next_group()

COLOR_MATCHING is used throughout: it requires no ML models and produces
deterministic results with solid-color PNG test images.  The synthetic image
set (from compare_colors_dir fixture) contains 5 red, 5 blue, and 5 green
images with LAB ΔE76 << 15 within each family and >> 60 across families, so
the default threshold of 15 cleanly separates all three groups.
"""

import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from tests.ui.app_window_fixtures import _teardown_app_window
from ui.app_window.app_window import AppWindow
from utils.constants import CompareMode
from utils.translations import I18N

_tr = I18N._


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def window_with_compare_dir(qtbot, compare_colors_dir):
    """AppWindow loaded with the solid-color compare directory."""
    win = AppWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win.set_base_dir(compare_colors_dir["dir"])
    qtbot.waitUntil(lambda: win.base_dir == compare_colors_dir["dir"], timeout=3000)
    yield win, compare_colors_dir
    _teardown_app_window(win)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_compare_settings(win):
    """Open CompareSettingsWindow for win's compare_manager and return it."""
    from ui.compare.compare_settings_window_qt import CompareSettingsWindow
    CompareSettingsWindow.open(parent=win, compare_manager=win.compare_manager)
    return CompareSettingsWindow._open_windows[win.compare_manager]


def _click_apply(settings_win, qtbot):
    """Find and click the Apply button in the settings window."""
    apply_text = _tr("Apply")
    for btn in settings_win.findChildren(QPushButton):
        if btn.text() == apply_text:
            qtbot.mouseClick(btn, Qt.MouseButton.LeftButton)
            return
    raise AssertionError(f"Apply button ({apply_text!r}) not found in CompareSettingsWindow")


def _suppress_app_alerts(monkeypatch, win) -> None:
    """Prevent compare-engine alert calls from opening a blocking modal dialog.

    _build_app_actions stores ts(notification_ctrl.alert) in _actions["_alert"]
    at AppWindow construction time.  The ts() wrapper uses BlockingQueuedConnection,
    so if the real alert shows a dialog the background worker blocks indefinitely.
    Patching the dict entry directly bypasses both the ts wrapper and the dialog.
    """
    monkeypatch.setitem(win.app_actions._actions, "_alert", lambda *a, **k: None)


def _run_group_compare(win, qtbot, immediate_compare_debounce):
    """Clear sidebar search fields and trigger a group compare; wait for completion."""
    immediate_compare_debounce(win.search_ctrl)
    win.sidebar_panel.search_media_path_box.clear()
    win.sidebar_panel.search_text_box.clear()
    win.search_ctrl.set_search()
    qtbot.waitUntil(lambda: not win.search_ctrl.is_compare_running(), timeout=20000)


# ---------------------------------------------------------------------------
# Search pipeline
# ---------------------------------------------------------------------------

class TestCompareSearchPipeline:
    """Search mode: query with a red image and verify result ordering."""

    def test_search_returns_same_color_family_first(
        self,
        window_with_compare_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        """After a COLOR_MATCHING search on a red image, the top-N results are
        all red siblings — not blue, green, or outlier images."""
        win, colors = window_with_compare_dir
        immediate_compare_debounce(win.search_ctrl)

        # --- Configure mode and threshold via CompareSettingsWindow ---
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        settings_win = _open_compare_settings(win)
        settings_win._threshold_combo.setCurrentText("15")
        _click_apply(settings_win, qtbot)

        # Prevent any "No Match Found" dialog from blocking the worker thread.
        _suppress_app_alerts(monkeypatch, win)

        # --- Run search with the first red image as the query ---
        search_img = colors["red"][0]
        red_siblings = set(colors["red"][1:])   # 4 other red images

        win.search_ctrl._set_media_run_search(search_img)
        qtbot.waitUntil(lambda: not win.search_ctrl.is_compare_running(), timeout=20000)

        # --- Results: top-N slots must all be red siblings ---
        files_matched = win.compare_manager.files_matched
        assert files_matched, "expected non-empty search results"

        # Utils.get_valid_file joins with "/" (forward slash) on all platforms,
        # so the search_media_path may not string-equal the backslash path from
        # the directory scan.  This causes the search image itself to appear as
        # a self-match at position 0.  Normalise paths before comparing.
        norm_search = os.path.normpath(search_img)
        non_self = [f for f in files_matched if os.path.normpath(f) != norm_search]
        top_n = set(non_self[: len(red_siblings)])
        assert red_siblings == top_n, (
            f"Expected first {len(red_siblings)} non-self results to be red siblings;\n"
            f"  got : {top_n}\n"
            f"  want: {red_siblings}"
        )

    def test_settings_window_threshold_is_applied_to_compare_manager(
        self,
        window_with_compare_dir,
        qtbot,
    ):
        """Changing the threshold combo in CompareSettingsWindow and clicking Apply
        updates the compare_manager's threshold to the chosen value."""
        win, _ = window_with_compare_dir

        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        settings_win = _open_compare_settings(win)

        # Pick a non-default threshold value that is present in the combo.
        settings_win._threshold_combo.setCurrentText("20")
        _click_apply(settings_win, qtbot)

        assert win.compare_manager.get_threshold() == 20

    def test_no_search_results_when_threshold_is_zero(
        self,
        window_with_compare_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        """With threshold=0 no images match (ΔE76 is never strictly < 0), so
        files_matched is empty when search_only_return_closest=True."""
        win, colors = window_with_compare_dir
        immediate_compare_debounce(win.search_ctrl)

        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        settings_win = _open_compare_settings(win)
        settings_win._threshold_combo.setCurrentText("0")
        _click_apply(settings_win, qtbot)

        # compare_colors.py holds a module-level reference to the original
        # config singleton (not the per-test one from isolated_singletons).
        # Patch search_only_return_closest on that same object so the module
        # only returns files that genuinely pass the ΔE76 < threshold gate.
        import compare.compare_colors as _cc
        monkeypatch.setattr(_cc.config, "search_only_return_closest", True)

        # The "No Match Found" alert is dispatched via app_actions._actions["_alert"]
        # — a BlockingQueuedConnection wrapper captured at AppWindow construction.
        # Patching notification_ctrl.alert is too late; patch the dict entry directly.
        _suppress_app_alerts(monkeypatch, win)

        search_img = colors["red"][0]
        win.search_ctrl._set_media_run_search(search_img)
        qtbot.waitUntil(lambda: not win.search_ctrl.is_compare_running(), timeout=20000)

        assert win.compare_manager.files_matched == [], (
            "expected no matches with threshold=0"
        )


# ---------------------------------------------------------------------------
# Group pipeline
# ---------------------------------------------------------------------------

class TestCompareGroupPipeline:
    """Group mode: all images cluster into per-color-family groups."""

    def test_group_compare_produces_at_least_three_groups(
        self,
        window_with_compare_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        """Running a group compare on the 15 color images (+ 3 outliers) yields
        at least 3 groups — one per color family."""
        win, colors = window_with_compare_dir

        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        settings_win = _open_compare_settings(win)
        settings_win._threshold_combo.setCurrentText("15")
        _click_apply(settings_win, qtbot)

        _suppress_app_alerts(monkeypatch, win)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        file_groups = win.compare_manager.file_groups
        assert len(file_groups) >= 3, (
            f"expected ≥3 groups (one per color family), got {len(file_groups)}"
        )

    def test_each_group_is_a_pure_color_family(
        self,
        window_with_compare_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        """No group should mix files from different color families.
        Every group must be a subset of exactly one family (red, blue, or green)."""
        win, colors = window_with_compare_dir

        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)

        _suppress_app_alerts(monkeypatch, win)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        red_set = set(colors["red"])
        blue_set = set(colors["blue"])
        green_set = set(colors["green"])

        for group_idx, group in win.compare_manager.file_groups.items():
            group_files = set(group.keys())
            is_pure = (
                group_files.issubset(red_set)
                or group_files.issubset(blue_set)
                or group_files.issubset(green_set)
            )
            assert is_pure, (
                f"Group {group_idx} mixes color families: {group_files}"
            )

    def test_show_next_group_advances_to_different_group(
        self,
        window_with_compare_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        """Calling show_next_group() changes files_matched to a different set of files."""
        win, colors = window_with_compare_dir

        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)

        _suppress_app_alerts(monkeypatch, win)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        assert win.compare_manager.file_groups, "group compare produced no results"
        assert len(win.compare_manager.file_groups) >= 2, (
            "need at least 2 groups to test navigation"
        )

        first_group = list(win.compare_manager.files_matched)

        # Passing file_browser takes an early-return branch that navigates the
        # file browser cursor rather than advancing the group index.  Omit it
        # to get the standard group-index increment + set_current_group path.
        win.compare_manager.show_next_group()

        second_group = list(win.compare_manager.files_matched)
        assert second_group, "files_matched should be non-empty after show_next_group"
        assert first_group != second_group, (
            "show_next_group() did not advance to a different group"
        )

    def test_all_color_families_represented_in_groups(
        self,
        window_with_compare_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        """After group compare, the union of all grouped files contains at least
        one member from each of the three color families."""
        win, colors = window_with_compare_dir

        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)

        _suppress_app_alerts(monkeypatch, win)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        all_grouped = set()
        for group in win.compare_manager.file_groups.values():
            all_grouped.update(group.keys())

        red_set = set(colors["red"])
        blue_set = set(colors["blue"])
        green_set = set(colors["green"])

        assert all_grouped & red_set, "no red images found in any group"
        assert all_grouped & blue_set, "no blue images found in any group"
        assert all_grouped & green_set, "no green images found in any group"

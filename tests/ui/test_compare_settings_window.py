"""UI smoke tests for CompareSettingsWindow (via WindowLauncher)."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton

from ui.compare.compare_settings_window_qt import CompareSettingsWindow
from utils.constants import Sort
from utils.translations import _

_tr = _


def _close_compare_settings_windows() -> None:
    for win in list(CompareSettingsWindow._open_windows.values()):
        try:
            win.close()
            win.deleteLater()
        except RuntimeError:
            pass
    CompareSettingsWindow._open_windows.clear()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture(autouse=True)
def _compare_settings_cleanup():
    yield
    _close_compare_settings_windows()


@pytest.fixture
def compare_settings_window(window_with_dir, qtbot):
    """Open CompareSettingsWindow for the app compare manager."""
    app_win, _ = window_with_dir
    cm = app_win.compare_manager

    app_win.window_launcher.open_compare_settings_window()
    qtbot.waitUntil(lambda: cm in CompareSettingsWindow._open_windows, timeout=5000)
    settings_win = CompareSettingsWindow._open_windows[cm]
    qtbot.addWidget(settings_win)
    qtbot.waitExposed(settings_win, timeout=3000)

    yield settings_win, app_win

    _close_compare_settings_windows()


class TestCompareSettingsWindowOpen:
    def test_open_via_launcher_shows_settings_ui(
        self, compare_settings_window
    ):
        settings_win, app_win = compare_settings_window

        assert settings_win.isVisible()
        assert settings_win._compare_manager is app_win.compare_manager
        assert settings_win._filter_panel is not None
        assert settings_win._add_instance_btn is not None

        titles = [lbl.text() for lbl in settings_win.findChildren(QLabel)]
        assert _tr("Compare Settings") in titles

    def test_second_open_focuses_existing_window(
        self, compare_settings_window, qtbot
    ):
        settings_win, app_win = compare_settings_window
        first_id = id(settings_win)

        app_win.window_launcher.open_compare_settings_window()
        qtbot.waitUntil(
            lambda: app_win.compare_manager in CompareSettingsWindow._open_windows,
            timeout=3000,
        )
        second = CompareSettingsWindow._open_windows[app_win.compare_manager]

        assert id(second) == first_id
        assert second.isVisible()

    def test_add_instance_button_present(self, compare_settings_window):
        settings_win, _ = compare_settings_window
        buttons = settings_win.findChildren(QPushButton)
        assert any(
            _tr("+ Add Instance") in btn.text() or "Add Instance" in btn.text()
            for btn in buttons
        )

    def test_group_sort_combo_is_present(self, compare_settings_window):
        settings_win, _ = compare_settings_window
        assert settings_win._group_sort_combo is not None
        assert settings_win._group_sort_combo.count() == len(Sort.non_random_options())

    def test_group_sort_combo_apply_updates_config(
        self, compare_settings_window, qtbot, monkeypatch
    ):
        """Selecting ASC in the group sort combo and clicking Apply persists
        Sort.ASC to config.compare_group_sort."""
        import utils.config as cfg_module
        import ui.compare.compare_settings_window_qt as csw_module

        settings_win, _ = compare_settings_window

        # compare_settings_window_qt bound `config` at import time, before the
        # per-test isolated singleton was created.  Route its module-level name
        # to the isolated instance so _on_apply writes to the right object.
        monkeypatch.setattr(csw_module, "config", cfg_module.config)

        combo = settings_win._group_sort_combo
        asc_index = next(
            i for i in range(combo.count()) if combo.itemData(i) == Sort.ASC
        )
        combo.setCurrentIndex(asc_index)

        apply_text = _tr("Apply")
        apply_btn = next(
            btn for btn in settings_win.findChildren(QPushButton)
            if btn.text() == apply_text
        )
        qtbot.mouseClick(apply_btn, Qt.MouseButton.LeftButton)

        assert cfg_module.config.compare_group_sort == Sort.ASC

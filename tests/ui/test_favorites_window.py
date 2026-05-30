"""UI smoke tests for FavoritesWindow (via WindowLauncher)."""

import os

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from ui.files.favorites_window_qt import FavoritesWindow
from utils.app_info_cache import app_info_cache
from utils.translations import _


def _label_texts(widget) -> list[str]:
    if widget is None:
        return []
    try:
        return [lbl.text() for lbl in widget.findChildren(QLabel) if lbl.text()]
    except RuntimeError:
        # Window may be torn down while the suite processes pending Qt events.
        return []


def _find_favorites_window() -> FavoritesWindow | None:
    app = QApplication.instance()
    if app is None:
        return None
    for widget in app.topLevelWidgets():
        if isinstance(widget, FavoritesWindow):
            return widget
    return None


def _close_all_favorites_windows() -> None:
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        if isinstance(widget, FavoritesWindow):
            widget.close()
            widget.deleteLater()
    app.processEvents()


@pytest.fixture(autouse=True)
def _favorites_window_cleanup():
    yield
    _close_all_favorites_windows()


class TestFavoritesWindow:
    def test_open_shows_empty_state_when_no_favorites(
        self, window_with_dir, qtbot
    ):
        win, _ = window_with_dir
        win.window_launcher.open_favorites_window()

        qtbot.waitUntil(lambda: _find_favorites_window() is not None, timeout=5000)
        fav_win = _find_favorites_window()
        qtbot.addWidget(fav_win)
        qtbot.waitExposed(fav_win, timeout=3000)

        texts = " ".join(_label_texts(fav_win)).lower()
        assert "favorite" in texts or "favourit" in texts

    def test_open_lists_seeded_favorite(
        self, window_with_dir, qtbot
    ):
        win, media_dir = window_with_dir
        norm_dir = os.path.normpath(os.path.abspath(media_dir))
        media_path = os.path.normpath(win.file_browser.get_files()[0])
        app_info_cache.set(norm_dir, "favorites", [media_path])

        win.window_launcher.open_favorites_window()
        qtbot.waitUntil(lambda: _find_favorites_window() is not None, timeout=5000)

        def _seeded_row_visible() -> bool:
            fav = _find_favorites_window()
            if fav is None:
                return False
            return any("img01.png" in t for t in _label_texts(fav))

        qtbot.waitUntil(_seeded_row_visible, timeout=5000)
        fav_win = _find_favorites_window()
        assert fav_win is not None
        qtbot.addWidget(fav_win)
        assert FavoritesWindow.has_any_favorites

    def test_add_favorite_persists_in_cache(self, window_with_dir):
        win, media_dir = window_with_dir
        media_path = os.path.join(media_dir, "img02.png")
        toasts = []

        FavoritesWindow.add_favorite(
            media_dir, media_path, toast_callback=lambda msg: toasts.append(msg)
        )

        assert media_path in FavoritesWindow.get_favorites(media_dir)
        assert toasts

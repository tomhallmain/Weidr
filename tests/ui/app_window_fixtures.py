"""Shared helpers and pytest fixtures for AppWindow UI tests."""

import os

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from ui.app_window.app_window import AppWindow


def _teardown_app_window(win: AppWindow) -> None:
    """Stop timers, unregister, and drop the QWidget."""
    win.on_closing()
    # hide() rather than close() to avoid re-entering on_closing() via closeEvent
    # and to prevent QApplication.quit() from being called during test teardown.
    win.hide()
    win.deleteLater()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def make_png(path: str, color: tuple = (255, 0, 0)) -> None:
    """Write a minimal 10×10 RGB PNG to *path*."""
    img = Image.new("RGB", (10, 10), color)
    img.save(path, format="PNG")


@pytest.fixture
def media_dir(tmp_path):
    """Temp directory containing three small PNG files, alphabetically named."""
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)], start=1):
        make_png(str(tmp_path / f"img{i:02d}.png"), color)
    return str(tmp_path)


@pytest.fixture
def window(qtbot):
    """A fresh AppWindow with no initial directory."""
    win = AppWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    yield win
    _teardown_app_window(win)


@pytest.fixture
def window_with_dir(qtbot, media_dir):
    """AppWindow with *media_dir* pre-loaded."""
    win = AppWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win.set_base_dir(media_dir)
    qtbot.waitUntil(lambda: win.base_dir == media_dir, timeout=2000)
    yield win, media_dir
    _teardown_app_window(win)

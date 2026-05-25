"""Masonry view: tile activation and PgUp/PgDn pagination."""

import os

import pytest
from PySide6.QtCore import Qt

from tests.ui.app_window_fixtures import make_png
from utils.constants import ViewMode


def _enter_masonry(win, qtbot) -> None:
    win.toggle_masonry_view()
    qtbot.waitUntil(lambda: win.view_mode == ViewMode.MASONRY, timeout=2000)
    qtbot.waitUntil(lambda: len(win.masonry_browser._tiles) > 0, timeout=3000)


@pytest.fixture
def small_page_size(monkeypatch):
    """Use two tiles per page so three-file fixtures span multiple pages."""
    monkeypatch.setattr("ui.app_window.masonry_browser.PAGE_SIZE", 2)


class TestMasonryTileActivation:
    def test_tile_click_returns_to_full_view_and_loads_file(
        self, window_with_dir, qtbot
    ):
        win, media_dir = window_with_dir
        target = os.path.join(media_dir, "img03.png")
        _enter_masonry(win, qtbot)

        tile = next(t for t in win.masonry_browser._tiles if t.filepath == target)
        qtbot.mouseClick(tile, Qt.MouseButton.LeftButton)

        qtbot.waitUntil(lambda: win.view_mode == ViewMode.FULL, timeout=2000)
        assert win._media_stack.currentIndex() == 0
        assert win.media_path is not None
        assert os.path.normcase(win.media_path) == os.path.normcase(target)

    def test_tile_activated_signal_invokes_handler(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, media_dir = window_with_dir
        target = os.path.join(media_dir, "img02.png")
        calls = []
        monkeypatch.setattr(
            win,
            "_on_masonry_tile_activated",
            lambda path: calls.append(path),
        )
        _enter_masonry(win, qtbot)

        win.masonry_browser.tile_activated.emit(target)

        assert calls == [target]


def _fire_pgdown_shortcut(win) -> None:
    """Activate the same handler bound to PgDown in KeyBindingManager."""
    win.media_navigator.page_down()


def _fire_pgup_shortcut(win) -> None:
    """Activate the same handler bound to PgUp in KeyBindingManager."""
    win.media_navigator.page_up()


class TestMasonryPagination:
    def test_pgdn_advances_page_in_masonry_mode(
        self, window_with_dir, qtbot, small_page_size
    ):
        win, _ = window_with_dir
        _enter_masonry(win, qtbot)
        qtbot.waitUntil(
            lambda: "Page 1 / 2" in win.masonry_browser._page_label.text(),
            timeout=2000,
        )

        _fire_pgdown_shortcut(win)

        qtbot.waitUntil(
            lambda: "Page 2 / 2" in win.masonry_browser._page_label.text(),
            timeout=2000,
        )
        assert len(win.masonry_browser._tiles) == 1
        assert win.masonry_browser._tiles[0].filepath.endswith("img03.png")

    def test_pgup_returns_to_previous_page(
        self, window_with_dir, qtbot, small_page_size
    ):
        win, _ = window_with_dir
        _enter_masonry(win, qtbot)
        _fire_pgdown_shortcut(win)
        qtbot.waitUntil(
            lambda: "Page 2 / 2" in win.masonry_browser._page_label.text(),
            timeout=2000,
        )

        _fire_pgup_shortcut(win)

        qtbot.waitUntil(
            lambda: "Page 1 / 2" in win.masonry_browser._page_label.text(),
            timeout=2000,
        )
        assert len(win.masonry_browser._tiles) == 2

    def test_pgup_on_first_page_is_no_op(
        self, window_with_dir, qtbot, small_page_size
    ):
        win, _ = window_with_dir
        _enter_masonry(win, qtbot)
        label_before = win.masonry_browser._page_label.text()

        _fire_pgup_shortcut(win)

        assert win.masonry_browser._page_label.text() == label_before
        assert win.masonry_browser._page == 0

    def test_many_files_populate_starts_on_current_file_page(
        self, qtbot, tmp_path, small_page_size
    ):
        from ui.app_window.app_window import AppWindow

        for i in range(1, 6):
            make_png(str(tmp_path / f"img{i:02d}.png"))
        media_dir = str(tmp_path)

        win = AppWindow()
        qtbot.addWidget(win)
        win.show()
        qtbot.waitExposed(win)
        try:
            win.set_base_dir(media_dir)
            qtbot.waitUntil(lambda: win.base_dir == media_dir, timeout=2000)
            # Move to last file so masonry opens on its page (PAGE_SIZE=2 → page 2).
            win.file_browser.go_to_file(os.path.join(media_dir, "img05.png"))
            win.media_navigator.create_media(os.path.join(media_dir, "img05.png"))

            _enter_masonry(win, qtbot)
            qtbot.waitUntil(
                lambda: "Page 3 / 3" in win.masonry_browser._page_label.text(),
                timeout=2000,
            )
            assert any(t.filepath.endswith("img05.png") for t in win.masonry_browser._tiles)
        finally:
            from tests.ui.app_window_fixtures import _teardown_app_window

            _teardown_app_window(win)

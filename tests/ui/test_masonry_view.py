"""UI tests for MasonryBrowser toggle and pagination."""

from utils.constants import ViewMode

class TestMasonryView:
    def test_toggle_masonry_shows_grid_stack(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.toggle_masonry_view()
        qtbot.waitUntil(lambda: win.view_mode == ViewMode.MASONRY, timeout=2000)
        assert win._media_stack.currentIndex() == 1
        qtbot.waitUntil(
            lambda: win.masonry_browser._page_bar.isVisible()
            and "3" in win.masonry_browser._page_label.text(),
            timeout=3000,
        )

        win.toggle_masonry_view()
        qtbot.waitUntil(lambda: win.view_mode == ViewMode.FULL, timeout=2000)
        assert win._media_stack.currentIndex() == 0

    def test_masonry_page_label_reflects_file_count(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.toggle_masonry_view()
        qtbot.waitUntil(lambda: win.masonry_browser._page_bar.isVisible(), timeout=2000)
        label = win.masonry_browser._page_label.text()
        assert "3" in label

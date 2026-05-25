"""UI tests for SearchController wiring on AppWindow."""

import os

class TestSearchController:
    def test_search_ctrl_attached(self, window):
        assert window.search_ctrl is not None
        assert window.search_ctrl._fb is window.file_browser

    def test_is_compare_running_false_initially(self, window):
        assert window.search_ctrl.is_compare_running() is False

    def test_get_search_media_path_resolves_basename(self, window_with_dir):
        win, _ = window_with_dir
        win.sidebar_panel.search_media_path_box.setText("img02.png")
        resolved = win.search_ctrl.get_search_media_path()
        assert resolved is not None
        assert os.path.normcase(resolved) == os.path.normcase(
            win.file_browser.get_files()[1]
        )

    def test_get_search_media_path_empty_when_cleared(self, window_with_dir):
        win, _ = window_with_dir
        win.sidebar_panel.search_media_path_box.clear()
        assert win.search_ctrl.get_search_media_path() is None

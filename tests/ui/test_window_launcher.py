"""UI tests for WindowLauncher secondary windows."""

class TestWindowLauncher:
    def test_window_launcher_attached(self, window):
        assert window.window_launcher is not None
        assert window.window_launcher._app is window

    def test_open_go_to_file_window(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.window_launcher.open_go_to_file_window()
        qtbot.waitUntil(
            lambda: win.window_launcher._go_to_file_window is not None
            and win.window_launcher._go_to_file_window.isVisible(),
            timeout=3000,
        )
        first = win.window_launcher._go_to_file_window
        win.window_launcher.open_go_to_file_window()
        assert win.window_launcher._go_to_file_window is first

    def test_open_recent_directory_window(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.window_launcher.open_recent_directory_window()
        qtbot.waitExposed(win, timeout=2000)

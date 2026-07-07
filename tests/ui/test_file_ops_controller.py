"""FileOpsController delete/hide/copy hooks — temp dirs only, no real deletes."""

import os

from PySide6.QtWidgets import QApplication

from files.marked_files import MarkedFiles
from utils.constants import Mode
from utils.config import config


class TestFileOpsController:
    def test_delete_media_browse_invokes_remove_path_without_disk_delete(
        self, window_with_dir, bypass_password, monkeypatch
    ):
        win, media_dir = window_with_dir
        monkeypatch.setattr(config, "delete_instantly", True)
        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append(path),
        )
        monkeypatch.setattr(win, "release_media_canvas", lambda: None)

        win.set_mode(Mode.BROWSE)
        target = win.file_browser.current_file()
        assert target is not None
        assert os.path.isfile(target)

        win.file_ops_ctrl.delete_media()

        assert removed == [target]
        assert os.path.isfile(target)
        MarkedFiles.delete_lock = False

    def test_delete_media_skipped_while_compare_running(
        self, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(config, "delete_instantly", True)
        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append(path),
        )
        monkeypatch.setattr(win, "is_compare_running", lambda: True)

        win.set_mode(Mode.BROWSE)
        target = win.file_browser.current_file()
        win.file_ops_ctrl.delete_media()

        assert removed == []
        assert target is not None and os.path.isfile(target)

    def test_handle_delete_does_not_touch_disk_when_remove_mocked(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(config, "delete_instantly", True)
        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append(path),
        )

        path = win.file_browser.current_file()
        win.file_ops_ctrl.handle_delete(path, toast=False)

        assert removed == [path]
        assert os.path.isfile(path)
        MarkedFiles.delete_lock = False

    def test_hide_current_media_adds_to_hidden_list(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(win.media_navigator, "show_next_media", lambda: None)
        win.compare_manager.hidden_media.clear()

        path = win.file_browser.current_file()
        win.file_ops_ctrl.hide_current_media(media_path=path)

        assert path in win.compare_manager.hidden_media

    def test_copy_media_path_puts_full_path_on_clipboard(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(config, "escape_backslash_filepaths", False)
        path = win.file_browser.current_file()
        win.file_ops_ctrl.copy_media_path(path)
        clipboard_text = QApplication.clipboard().text()
        assert os.path.normcase(clipboard_text) == os.path.normcase(path)


class TestFileCheckLargeDirectorySkip:
    """checking_files=False skips the periodic file check in large directories,
    unless the new-media slideshow is active — it needs the check to keep
    watching for newly added files regardless of directory size."""

    def test_skipped_when_checking_files_false_and_slideshow_inactive(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        win.set_mode(Mode.BROWSE)
        win.file_browser.checking_files = False
        win.slideshow_config.show_new_media = False
        refreshed = []
        monkeypatch.setattr(win, "refresh", lambda **kw: refreshed.append(kw))

        win.file_ops_ctrl._on_file_check()

        assert refreshed == []

    def test_not_skipped_when_new_media_slideshow_active(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        win.set_mode(Mode.BROWSE)
        win.file_browser.checking_files = False
        win.slideshow_config.show_new_media = True
        refreshed = []
        monkeypatch.setattr(win, "refresh", lambda **kw: refreshed.append(kw))

        win.file_ops_ctrl._on_file_check()

        assert len(refreshed) == 1
        assert refreshed[0]["show_new_media"] is True

"""FileOpsController delete_current_base_dir (isolated temp dirs, mocked removal)."""

import os

import pytest
from PIL import Image

from files.marked_files import MarkedFiles
from files.recent_directories import RecentDirectories
from utils.config import config
from utils.translations import _

_tr = _


@pytest.fixture
def deletable_dirs(tmp_path):
    """Directory to delete plus a valid replacement from RecentDirectories."""
    base = tmp_path / "gallery"
    replacement = tmp_path / "other"
    base.mkdir()
    replacement.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(base / "shot.png", format="PNG")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(replacement / "keep.png", format="PNG")
    RecentDirectories.directories = [str(replacement), str(base)]
    return str(base), str(replacement)


class TestFileOpsDeleteDirectory:
    def test_delete_current_base_dir_switches_and_removes_dir(
        self, window_with_dir, deletable_dirs, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        base_dir, replacement_dir = deletable_dirs
        monkeypatch.setattr(config, "delete_instantly", True)

        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append((path, kwargs.get("is_directory"))),
        )
        marks_cleared = []
        monkeypatch.setattr(
            MarkedFiles,
            "remove_marks_for_base_dir",
            lambda bd, _actions: marks_cleared.append(bd),
        )
        def _confirm_delete(*_args, **kwargs):
            buttons = kwargs.get("buttons") or []
            return buttons[0][0] if buttons else True

        monkeypatch.setattr(win.notification_ctrl, "alert", _confirm_delete)
        monkeypatch.setattr(
            "files.recent_directories.RecentDirectories.remove_directory",
            lambda _d: None,
        )
        monkeypatch.setattr(
            "utils.app_info_cache.app_info_cache.clear_directory_cache",
            lambda _d: None,
        )
        monkeypatch.setattr(
            "utils.app_info_cache.app_info_cache.store",
            lambda: None,
        )

        win.set_base_dir(base_dir)
        assert win.base_dir == base_dir

        win.file_ops_ctrl.delete_current_base_dir()

        assert marks_cleared == [base_dir]
        assert any(p == base_dir and is_dir for p, is_dir in removed)
        assert win.base_dir == replacement_dir
        assert os.path.isdir(base_dir)

    def test_delete_current_base_dir_no_op_for_invalid_path(
        self, window, bypass_password, monkeypatch
    ):
        alerts = []
        monkeypatch.setattr(
            window.notification_ctrl,
            "alert",
            lambda *args, **kwargs: alerts.append(kwargs.get("kind")),
        )
        monkeypatch.setattr(window, "get_base_dir", lambda: None)
        window.file_ops_ctrl.delete_current_base_dir()
        assert alerts == ["warning"]


class TestDeleteDirectoryMoveFilesBranch:
    def test_delete_current_base_dir_move_files_moves_contents_then_deletes(
        self, window_with_dir, deletable_dirs, bypass_password, monkeypatch, tmp_path
    ):
        win, _ = window_with_dir
        base_dir, replacement_dir = deletable_dirs
        target_dir = tmp_path / "archive"
        target_dir.mkdir()
        monkeypatch.setattr(config, "delete_instantly", True)

        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append((path, kwargs.get("is_directory"))),
        )
        monkeypatch.setattr(
            MarkedFiles,
            "remove_marks_for_base_dir",
            lambda _bd, _actions: None,
        )
        monkeypatch.setattr(
            "files.recent_directories.RecentDirectories.remove_directory",
            lambda _d: None,
        )
        monkeypatch.setattr(
            "utils.app_info_cache.app_info_cache.clear_directory_cache",
            lambda _d: None,
        )
        monkeypatch.setattr(
            "utils.app_info_cache.app_info_cache.store",
            lambda: None,
        )
        monkeypatch.setattr(
            "lib.fast_directory_picker_qt.get_existing_directory",
            lambda *_a, **_k: str(target_dir),
        )

        def _choose_move_files(*_args, **kwargs):
            for label, _role in kwargs.get("buttons") or []:
                if label == _tr("Move Files"):
                    return label
            return False

        toasts = []
        monkeypatch.setattr(win.notification_ctrl, "alert", _choose_move_files)
        monkeypatch.setattr(
            win.notification_ctrl,
            "toast",
            lambda msg, **kwargs: toasts.append(msg),
        )
        monkeypatch.setattr(win.notification_ctrl, "title_notify", lambda *_a, **_k: None)

        win.set_base_dir(base_dir)
        win.file_ops_ctrl.delete_current_base_dir()

        assert (target_dir / "shot.png").is_file()
        assert win.base_dir == replacement_dir
        assert any(p == base_dir and is_dir for p, is_dir in removed)
        assert toasts

    def test_delete_current_base_dir_move_files_cancelled_at_picker(
        self, window_with_dir, deletable_dirs, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        base_dir, replacement_dir = deletable_dirs

        def _choose_move_files(*_args, **kwargs):
            for label, _role in kwargs.get("buttons") or []:
                if label == _tr("Move Files"):
                    return label
            return False

        monkeypatch.setattr(win.notification_ctrl, "alert", _choose_move_files)
        monkeypatch.setattr(
            "lib.fast_directory_picker_qt.get_existing_directory",
            lambda *_a, **_k: None,
        )
        remove_called = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: remove_called.append(path),
        )

        win.set_base_dir(base_dir)
        win.file_ops_ctrl.delete_current_base_dir()

        assert win.base_dir == base_dir
        assert os.path.isfile(os.path.join(base_dir, "shot.png"))
        assert remove_called == []

"""FileOpsController delete in search/compare modes and group removal hooks."""

import os
from types import SimpleNamespace

from utils.constants import Mode
from utils.config import config
from utils.translations import I18N

_ = I18N._


class TestFileOpsSearchDelete:
    def test_delete_media_search_mode_toggled_search_media(
        self, window_with_dir, bypass_password, monkeypatch
    ):
        win, media_dir = window_with_dir
        monkeypatch.setattr(config, "delete_instantly", True)
        target = os.path.join(media_dir, "img02.png")
        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append(path),
        )
        monkeypatch.setattr(win, "release_media_canvas", lambda: None)
        monkeypatch.setattr(
            win.media_navigator,
            "get_active_media_filepath",
            lambda: target,
        )
        monkeypatch.setattr(
            win.compare_manager,
            "has_compare",
            lambda: True,
        )
        group_removed = []

        def fake_compare():
            return SimpleNamespace(
                remove_from_groups=lambda files: group_removed.extend(files)
            )

        monkeypatch.setattr(win.compare_manager, "compare", fake_compare)
        updates = []
        monkeypatch.setattr(
            win.compare_manager,
            "_update_groups_for_removed_file",
            lambda *args, **kwargs: updates.append((args, kwargs)),
        )

        win.set_mode(Mode.SEARCH)
        win.is_toggled_view_matches = False
        win.compare_manager.search_media_path = target

        win.file_ops_ctrl.delete_media()

        assert removed == [target]
        assert os.path.isfile(target)
        assert group_removed == [target]
        assert len(updates) == 1
        assert win.compare_manager.search_media_path is None

    def test_delete_media_search_mode_warns_when_no_matches(
        self, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        warnings = []
        monkeypatch.setattr(
            win.app_actions,
            "warn",
            lambda msg: warnings.append(msg),
        )
        monkeypatch.setattr(
            win.media_navigator,
            "is_toggled_search_media",
            lambda: False,
        )
        from compare.compare_manager import CompareManager

        monkeypatch.setattr(
            CompareManager,
            "files_matched",
            property(lambda self: []),
        )

        win.set_mode(Mode.SEARCH)
        win.is_toggled_view_matches = True
        win.file_ops_ctrl.delete_media()

        assert warnings


class TestFileOpsRemoveFromGroups:
    def test_handle_remove_files_from_groups_calls_compare_update(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        filepath = win.file_browser.get_files()[0]
        updates = []
        monkeypatch.setattr(
            win.compare_manager,
            "_get_file_group_map",
            lambda _mode: {filepath: (1, 2)},
        )
        monkeypatch.setattr(
            win.compare_manager,
            "_update_groups_for_removed_file",
            lambda *args, **kwargs: updates.append((args, kwargs)),
        )
        monkeypatch.setattr(
            win.compare_manager,
            "current_match",
            lambda: filepath,
        )

        win.set_mode(Mode.SEARCH)
        win.file_ops_ctrl.handle_remove_files_from_groups([filepath])

        assert len(updates) == 1
        args, kwargs = updates[0]
        assert args[0] == Mode.SEARCH
        assert args[1] == 1
        assert args[2] == 2
        assert kwargs.get("set_group") is False

    def test_handle_remove_files_clears_search_media_path(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        filepath = win.file_browser.get_files()[0]
        win.compare_manager.search_media_path = filepath
        monkeypatch.setattr(
            win.compare_manager,
            "_get_file_group_map",
            lambda _mode: {filepath: (0, 0)},
        )
        monkeypatch.setattr(
            win.compare_manager,
            "_update_groups_for_removed_file",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(win.compare_manager, "current_match", lambda: filepath)

        win.file_ops_ctrl.handle_remove_files_from_groups([filepath])

        assert win.compare_manager.search_media_path is None

"""Unit tests for FileAction history persistence and delete entries."""

from datetime import datetime

import pytest

from files.file_action import FileAction, _delete_file_sentinel
from utils.constants import FileActionKind
from utils.utils import Utils


@pytest.fixture(autouse=True)
def _clear_file_action_history():
    saved = FileAction.action_history[:]
    FileAction.action_history.clear()
    yield
    FileAction.action_history.clear()
    FileAction.action_history.extend(saved)


def _move_entry(target: str = "/data/out") -> FileAction:
    return FileAction(
        Utils.move_file,
        target,
        original_marks=["/data/in/photo.jpg"],
        new_files=[f"{target}/photo.jpg"],
        auto=False,
        timestamp=datetime(2025, 6, 1, 12, 0, 0),
    )


def _copy_entry(target: str = "/data/copy_dest") -> FileAction:
    return FileAction(
        Utils.copy_file,
        target,
        original_marks=["/data/in/doc.png"],
        new_files=[f"{target}/doc.png"],
        auto=False,
        timestamp=datetime(2025, 6, 1, 13, 0, 0),
    )


def _delete_entry(source_dir: str = "/data/in") -> FileAction:
    deleted = f"{source_dir}/gone.jpg"
    FileAction.add_delete_action(deleted, auto=False)
    return FileAction.action_history[0]


class TestFileActionSerialization:
    def test_move_roundtrip_to_dict_and_back(self):
        entry = _move_entry()
        restored = FileAction.from_dict(entry.to_dict())

        assert restored.action == Utils.move_file
        assert restored.target == entry.target
        assert restored.original_marks == entry.original_marks
        assert restored.new_files == entry.new_files
        assert restored.auto is False
        assert restored.timestamp == entry.timestamp
        assert restored.action_kind() == FileActionKind.MOVE

    def test_copy_roundtrip_to_dict_and_back(self):
        entry = _copy_entry()
        restored = FileAction.from_dict(entry.to_dict())

        assert restored.action == Utils.copy_file
        assert restored.action_kind() == FileActionKind.COPY
        assert restored.new_files == entry.new_files

    def test_delete_roundtrip_to_dict_and_back(self):
        entry = _delete_entry()
        blob = entry.to_dict()

        assert blob["action"] == "delete_file"
        restored = FileAction.from_dict(blob)

        assert restored.is_delete_action()
        assert restored.action is _delete_file_sentinel
        assert restored.original_marks == entry.original_marks
        assert restored.action_kind() == FileActionKind.DELETE
        assert restored.relevant_files == entry.original_marks

    def test_store_and_load_actions_via_app_info_cache(self):
        import files.file_action as fa_module
        from tests.helpers import isolated_app_info_cache

        cache = isolated_app_info_cache()
        fa_module.app_info_cache = cache

        FileAction.action_history.clear()
        FileAction.action_history.extend([_move_entry(), _copy_entry()])
        FileAction.add_delete_action("/data/in/gone.jpg", auto=False)
        FileAction.store_actions()

        FileAction.action_history.clear()
        FileAction.load_actions()

        assert len(FileAction.action_history) == 3
        kinds = {a.action_kind() for a in FileAction.action_history}
        assert kinds == {FileActionKind.MOVE, FileActionKind.COPY, FileActionKind.DELETE}

        stored = cache.get_meta("file_actions", default_val=[])
        assert len(stored) == 3
        assert any(item["action"] == "delete_file" for item in stored)


class TestFileActionHistoryQueries:
    def test_get_history_action_skips_deletes_by_default(self):
        FileAction.action_history.extend([_delete_entry(), _move_entry()])

        action = FileAction.get_history_action(start_index=0, include_deletes=False)

        assert action is not None
        assert action.action_kind() == FileActionKind.MOVE

    def test_get_history_action_can_include_deletes(self):
        deleted = _delete_entry()
        FileAction.action_history.append(_move_entry())

        action = FileAction.get_history_action(start_index=0, include_deletes=True)

        assert action == deleted
        assert action.is_delete_action()

    def test_get_action_statistics_counts_deletes(self):
        FileAction.action_history.append(_move_entry())
        FileAction.add_delete_action("/data/in/gone.jpg", auto=False)

        stats = FileAction.get_action_statistics(today_only=False, kind=FileActionKind.DELETE)

        assert len(stats) == 1
        target_dir = next(iter(stats))
        assert stats[target_dir]["deleted"] == 1
        assert stats[target_dir]["moved"] == 0

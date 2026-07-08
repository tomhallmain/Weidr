"""
Tests for the removal-undo snapshot on CompareWrapper.

Moving a file out of the compare base directory removes it from the group
state and the stored checkpoint. The move can be undone (Ctrl+Z /
undo_move_marks, or the file actions window), but until this feature the
compare result no longer knew about the file and its group changes were
irreversible. capture_removal_undo_snapshot() saves one version of the
pre-removal state (move flows only — deletes are excluded by design since they
cannot be undone from the app), and maybe_restore_removal_undo_snapshot()
restores it once every snapshot file exists on disk again.
"""
from __future__ import annotations

import os
import pickle
from types import SimpleNamespace

from compare.compare_result import CompareResult
from compare.compare_wrapper import CompareWrapper
from utils.constants import CompareMode, Mode


def _touch(directory, name) -> str:
    path = str(directory / name)
    with open(path, "wb") as f:
        f.write(b"x")
    return path


def _stub_app_actions():
    """No-op app actions for the paths _update_groups_for_removed_file hits
    when the last group is removed (alert / set_mode / label / next media)."""
    return SimpleNamespace(
        alert=lambda *a, **k: None,
        set_mode=lambda *a, **k: None,
        show_next_media=lambda *a, **k: None,
        toast=lambda *a, **k: None,
        _set_label_state=lambda *a, **k: None,
        create_media=lambda *a, **k: None,
        release_media_canvas=lambda *a, **k: None,
    )


def _stored_compare_result(tmp_path, files, files_grouped, file_groups, store=True):
    compare_result = CompareResult(
        base_dir=str(tmp_path), files=files, mode=CompareMode.CLIP_EMBEDDING
    )
    compare_result.files_grouped = files_grouped
    compare_result.file_groups = file_groups
    compare_result.is_complete = True
    if store:
        compare_result.store()
    return compare_result


def _group_mode_wrapper(tmp_path, filepaths, store_result=True):
    """Wrapper + fake compare with one group holding *filepaths* (group mode)."""
    group = {f: 0.1 * (i + 1) for i, f in enumerate(filepaths)}
    files_grouped = {i: (0, 0.1 * (i + 1)) for i in range(len(filepaths))}
    compare_result = _stored_compare_result(
        tmp_path, list(filepaths), dict(files_grouped), {0: dict(group)}, store=store_result
    )
    wrapper = CompareWrapper(
        master=None, compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=_stub_app_actions()
    )
    wrapper.file_groups = {0: dict(group)}
    wrapper.files_grouped = dict(files_grouped)
    wrapper.group_indexes = [0]
    wrapper.files_matched = sorted(group, key=lambda f: group[f])
    readded_files: list = []
    wrapper._compare = SimpleNamespace(
        compare_result=compare_result,
        base_dir=str(tmp_path),
        COMPARE_MODE=CompareMode.CLIP_EMBEDDING,
        is_run_search=False,
        readd_files=readded_files.extend,
    )
    return wrapper, readded_files


def _move_out_and_update(wrapper, tmp_path, filepath, match_index, out_dir):
    """Simulate the move-out flow: relocate the file, then run the same group
    updates AppWindow.refresh(removed_files=...) applies for a move."""
    os.makedirs(out_dir, exist_ok=True)
    os.rename(filepath, os.path.join(out_dir, os.path.basename(filepath)))
    wrapper._update_groups_for_removed_file(Mode.GROUP, 0, match_index, set_group=True)
    wrapper._sync_result_after_deletion(filepath)


def _move_back(tmp_path, filepath, out_dir):
    os.rename(os.path.join(out_dir, os.path.basename(filepath)), filepath)


class TestRemovalUndoSnapshotRoundtrip:
    def test_capture_and_restore_roundtrip_group_mode(self, tmp_path):
        a = _touch(tmp_path, "a.jpg")
        b = _touch(tmp_path, "b.jpg")
        c = _touch(tmp_path, "c.jpg")
        wrapper, readded_files = _group_mode_wrapper(tmp_path, [a, b, c])
        original_file_groups = {0: dict(wrapper.file_groups[0])}
        original_files_grouped = dict(wrapper.files_grouped)
        out_dir = str(tmp_path / "elsewhere")

        wrapper.capture_removal_undo_snapshot([a], Mode.GROUP)
        _move_out_and_update(wrapper, tmp_path, a, 0, out_dir)
        assert a not in wrapper.file_groups[0]
        assert a not in wrapper.files_matched

        # Not undone yet: the file is still gone, nothing must be restored.
        assert wrapper.maybe_restore_removal_undo_snapshot() is None
        assert a not in wrapper.file_groups[0]

        _move_back(tmp_path, a, out_dir)
        snapshot = wrapper.maybe_restore_removal_undo_snapshot()

        assert snapshot is not None
        assert snapshot.removed_files == [a]
        assert snapshot.app_mode == Mode.GROUP
        assert wrapper.file_groups == original_file_groups
        assert wrapper.files_grouped == original_files_grouped
        assert wrapper.files_matched == [a, b, c]
        assert readded_files == [a]
        # The snapshot is one-shot: consumed by a successful restore.
        assert wrapper._removal_undo_snapshot is None

    def test_restore_rewrites_stored_checkpoint(self, tmp_path):
        a = _touch(tmp_path, "a.jpg")
        b = _touch(tmp_path, "b.jpg")
        c = _touch(tmp_path, "c.jpg")
        wrapper, _ = _group_mode_wrapper(tmp_path, [a, b, c])
        out_dir = str(tmp_path / "elsewhere")
        cache_path = CompareResult.cache_path(str(tmp_path), CompareMode.CLIP_EMBEDDING)

        wrapper.capture_removal_undo_snapshot([a], Mode.GROUP)
        _move_out_and_update(wrapper, tmp_path, a, 0, out_dir)
        with open(cache_path, "rb") as f:
            after_removal = pickle.load(f)
        assert a not in after_removal.file_groups[0]
        assert a not in after_removal._dir_files_hash

        _move_back(tmp_path, a, out_dir)
        assert wrapper.maybe_restore_removal_undo_snapshot() is not None

        with open(cache_path, "rb") as f:
            restored = pickle.load(f)
        assert a in restored.file_groups[0]
        assert restored._dir_files_hash == [a, b, c]

    def test_no_checkpoint_written_when_none_existed(self, tmp_path):
        a = _touch(tmp_path, "a.jpg")
        b = _touch(tmp_path, "b.jpg")
        c = _touch(tmp_path, "c.jpg")
        wrapper, _ = _group_mode_wrapper(tmp_path, [a, b, c], store_result=False)
        out_dir = str(tmp_path / "elsewhere")
        cache_path = CompareResult.cache_path(str(tmp_path), CompareMode.CLIP_EMBEDDING)
        assert not os.path.exists(cache_path)

        wrapper.capture_removal_undo_snapshot([a], Mode.GROUP)
        _move_out_and_update(wrapper, tmp_path, a, 0, out_dir)
        _move_back(tmp_path, a, out_dir)

        assert wrapper.maybe_restore_removal_undo_snapshot() is not None
        assert not os.path.exists(cache_path)

    def test_restore_waits_until_all_files_return(self, tmp_path):
        a = _touch(tmp_path, "a.jpg")
        b = _touch(tmp_path, "b.jpg")
        c = _touch(tmp_path, "c.jpg")
        wrapper, readded_files = _group_mode_wrapper(tmp_path, [a, b, c])
        out_dir = str(tmp_path / "elsewhere")

        wrapper.capture_removal_undo_snapshot([a, b], Mode.GROUP)
        _move_out_and_update(wrapper, tmp_path, a, 0, out_dir)
        _move_out_and_update(wrapper, tmp_path, b, 0, out_dir)

        _move_back(tmp_path, a, out_dir)
        assert wrapper.maybe_restore_removal_undo_snapshot() is None
        assert readded_files == []

        _move_back(tmp_path, b, out_dir)
        snapshot = wrapper.maybe_restore_removal_undo_snapshot()
        assert snapshot is not None
        assert sorted(snapshot.removed_files) == [a, b]
        assert sorted(readded_files) == [a, b]
        assert wrapper.files_matched == [a, b, c]


class TestRemovalUndoSnapshotGuards:
    def test_irrelevant_removal_does_not_clobber_snapshot(self, tmp_path):
        """The refresh from an undo move (files outside the compare dir) must
        not replace a pending snapshot with a meaningless one."""
        a = _touch(tmp_path, "a.jpg")
        b = _touch(tmp_path, "b.jpg")
        c = _touch(tmp_path, "c.jpg")
        wrapper, _ = _group_mode_wrapper(tmp_path, [a, b, c])

        wrapper.capture_removal_undo_snapshot([a], Mode.GROUP)
        wrapper.capture_removal_undo_snapshot(
            [str(tmp_path / "elsewhere" / "unrelated.jpg")], Mode.GROUP
        )

        assert wrapper._removal_undo_snapshot is not None
        assert wrapper._removal_undo_snapshot.removed_files == [a]

    def test_no_snapshot_without_compare(self, tmp_path):
        wrapper = CompareWrapper(
            master=None, compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=None
        )
        wrapper.capture_removal_undo_snapshot(["/somewhere/a.jpg"], Mode.GROUP)
        assert wrapper._removal_undo_snapshot is None
        assert wrapper.maybe_restore_removal_undo_snapshot() is None

    def test_clear_compare_invalidates_snapshot(self, tmp_path):
        a = _touch(tmp_path, "a.jpg")
        b = _touch(tmp_path, "b.jpg")
        c = _touch(tmp_path, "c.jpg")
        wrapper, _ = _group_mode_wrapper(tmp_path, [a, b, c])

        wrapper.capture_removal_undo_snapshot([a], Mode.GROUP)
        assert wrapper._removal_undo_snapshot is not None
        wrapper.clear_compare()
        assert wrapper._removal_undo_snapshot is None

    def test_base_dir_change_drops_snapshot(self, tmp_path):
        a = _touch(tmp_path, "a.jpg")
        b = _touch(tmp_path, "b.jpg")
        c = _touch(tmp_path, "c.jpg")
        wrapper, readded_files = _group_mode_wrapper(tmp_path, [a, b, c])

        wrapper.capture_removal_undo_snapshot([a], Mode.GROUP)
        wrapper._compare.base_dir = str(tmp_path / "other_dir")

        assert wrapper.maybe_restore_removal_undo_snapshot() is None
        assert wrapper._removal_undo_snapshot is None
        assert readded_files == []

    def test_search_mode_relevance_via_files_matched(self, tmp_path):
        """In search mode file_groups can be empty; files_matched membership
        must be enough for a removal to qualify for a snapshot."""
        a = _touch(tmp_path, "a.jpg")
        wrapper = CompareWrapper(
            master=None, compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=None
        )
        wrapper.files_matched = [a]
        wrapper.files_grouped = {a: 0.9}
        wrapper._compare = SimpleNamespace(
            compare_result=None,
            base_dir=str(tmp_path),
            COMPARE_MODE=CompareMode.CLIP_EMBEDDING,
            is_run_search=True,
            readd_files=lambda fps: None,
        )

        wrapper.capture_removal_undo_snapshot([a], Mode.SEARCH)
        assert wrapper._removal_undo_snapshot is not None
        assert wrapper._removal_undo_snapshot.removed_files == [a]

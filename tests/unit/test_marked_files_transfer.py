"""Unit tests for marks-transfer session locking."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from files.file_action import FileAction
from files.marked_files import MarkedFiles
from utils.config import config
from utils.translations import _
from utils.utils import Utils


def test_guard_mark_mutation_allows_when_idle():
    MarkedFiles.is_performing_action = False
    assert MarkedFiles.guard_mark_mutation() is True


def test_guard_mark_mutation_blocks_during_transfer():
    MarkedFiles.is_performing_action = True
    try:
        assert MarkedFiles.guard_mark_mutation() is False
        assert MarkedFiles.add_mark_if_not_present("/tmp/extra.jpg") is False
    finally:
        MarkedFiles.is_performing_action = False


def test_toggle_mark_adds_when_absent():
    MarkedFiles.file_marks = []
    assert MarkedFiles.toggle_mark("/dir/a.jpg") is True
    assert MarkedFiles.file_marks == ["/dir/a.jpg"]


def test_toggle_mark_removes_when_present():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/b.jpg"]
    assert MarkedFiles.toggle_mark("/dir/a.jpg") is False
    assert MarkedFiles.file_marks == ["/dir/b.jpg"]


def test_toggle_mark_resets_cursor_when_it_falls_out_of_range():
    MarkedFiles.file_marks = ["/dir/a.jpg"]
    MarkedFiles.mark_cursor = 0
    MarkedFiles.toggle_mark("/dir/a.jpg")
    assert MarkedFiles.file_marks == []
    assert MarkedFiles.mark_cursor == -1


def test_toggle_mark_blocks_during_transfer():
    MarkedFiles.file_marks = []
    MarkedFiles.is_performing_action = True
    try:
        with pytest.raises(Exception):
            MarkedFiles.toggle_mark("/dir/a.jpg")
        assert MarkedFiles.file_marks == []
    finally:
        MarkedFiles.is_performing_action = False


def test_toggle_mark_blocks_while_delete_lock_is_set():
    MarkedFiles.file_marks = []
    MarkedFiles.delete_lock = True
    try:
        with pytest.raises(Exception):
            MarkedFiles.toggle_mark("/dir/a.jpg")
        assert MarkedFiles.file_marks == []
    finally:
        MarkedFiles.delete_lock = False


def test_advance_mark_cursor_moves_forward():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/b.jpg"]
    MarkedFiles.mark_cursor = -1
    marked_file, wrapped = MarkedFiles.advance_mark_cursor()
    assert marked_file == "/dir/a.jpg"
    assert wrapped is False


def test_advance_mark_cursor_moves_backward():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/b.jpg"]
    MarkedFiles.mark_cursor = 1
    marked_file, wrapped = MarkedFiles.advance_mark_cursor(backward=True)
    assert marked_file == "/dir/a.jpg"
    assert wrapped is False


def test_advance_mark_cursor_backward_from_the_sentinel_lands_on_the_last_mark():
    MarkedFiles.file_marks = ["/dir/a.jpg"]
    MarkedFiles.mark_cursor = -1
    marked_file, wrapped = MarkedFiles.advance_mark_cursor(backward=True)
    assert marked_file == "/dir/a.jpg"
    assert MarkedFiles.mark_cursor == 0
    assert wrapped is False


def test_advance_mark_cursor_backward_cycles_without_running_off_the_start():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/b.jpg"]
    MarkedFiles.mark_cursor = -1
    visited = [MarkedFiles.advance_mark_cursor(backward=True)[0] for _ in range(5)]
    assert visited == ["/dir/b.jpg", "/dir/a.jpg", "/dir/b.jpg", "/dir/a.jpg", "/dir/b.jpg"]


def test_advance_mark_cursor_wraps_past_the_end():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/b.jpg"]
    MarkedFiles.mark_cursor = 1
    marked_file, wrapped = MarkedFiles.advance_mark_cursor()
    assert marked_file == "/dir/a.jpg"
    assert wrapped is True


def test_advance_mark_cursor_raises_with_no_marks():
    MarkedFiles.file_marks = []
    with pytest.raises(Exception):
        MarkedFiles.advance_mark_cursor()


def test_clear_file_marks_empties_the_list_and_resets_the_cursor():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/b.jpg"]
    MarkedFiles.mark_cursor = 1
    assert MarkedFiles.clear_file_marks(MagicMock()) is True
    assert MarkedFiles.file_marks == []
    assert MarkedFiles.mark_cursor == -1


def test_clear_file_marks_blocks_during_transfer():
    MarkedFiles.file_marks = ["/dir/a.jpg"]
    MarkedFiles.is_performing_action = True
    try:
        assert MarkedFiles.clear_file_marks(MagicMock()) is False
        assert MarkedFiles.file_marks == ["/dir/a.jpg"]
    finally:
        MarkedFiles.is_performing_action = False


def test_add_series_adds_only_the_unmarked_ones():
    MarkedFiles.file_marks = ["/dir/a.jpg"]
    added = MarkedFiles.add_series(["/dir/a.jpg", "/dir/b.jpg", "/dir/c.jpg"])
    assert added == 2
    assert MarkedFiles.file_marks == ["/dir/a.jpg", "/dir/b.jpg", "/dir/c.jpg"]


def test_add_series_with_no_new_files_adds_nothing():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/b.jpg"]
    added = MarkedFiles.add_series(["/dir/a.jpg", "/dir/b.jpg"])
    assert added == 0
    assert MarkedFiles.file_marks == ["/dir/a.jpg", "/dir/b.jpg"]


def test_add_series_blocks_during_transfer():
    MarkedFiles.file_marks = []
    MarkedFiles.is_performing_action = True
    try:
        with pytest.raises(Exception):
            MarkedFiles.add_series(["/dir/a.jpg"])
        assert MarkedFiles.file_marks == []
    finally:
        MarkedFiles.is_performing_action = False


def test_apply_file_marks_clears_successful_and_keeps_failed():
    MarkedFiles.file_marks = ["/dir/a.jpg", "/dir/c.jpg"]
    MarkedFiles._apply_file_marks_after_transfer(
        files_to_move=["/dir/a.jpg", "/dir/c.jpg"],
        exceptions={"/dir/c.jpg": ("error", "/target/c.jpg")},
        invalid_files=[],
        single_image=False,
    )
    assert MarkedFiles.file_marks == ["/dir/c.jpg"]


def test_is_transfer_running_aliases_is_performing_action():
    MarkedFiles.is_performing_action = True
    try:
        assert MarkedFiles.is_transfer_running() is True
    finally:
        MarkedFiles.is_performing_action = False


def _patch_downstream(monkeypatch, result, calls=None):
    def _lookup(media_path, base_dir, app_actions, force_refresh=False):
        if calls is not None:
            calls.append((media_path, base_dir, force_refresh))
        return result
    monkeypatch.setattr("files.related_image.get_downstream_related_images", _lookup)


def test_set_marks_from_downstream_replaces_the_marks(monkeypatch):
    MarkedFiles.file_marks = ["/dir/old.jpg"]
    calls = []
    _patch_downstream(monkeypatch, ["/other/a_1.png", "/other/a_2.png"], calls)

    found, marks_set = MarkedFiles.set_marks_from_downstream_related_images(
        "/dir/a.png", "/other", MagicMock()
    )

    assert marks_set is True
    assert found == ["/other/a_1.png", "/other/a_2.png"]
    assert MarkedFiles.file_marks == found
    assert calls == [("/dir/a.png", "/other", True)]


def test_set_marks_from_downstream_with_nothing_found_keeps_the_marks(monkeypatch):
    MarkedFiles.file_marks = ["/dir/old.jpg"]
    _patch_downstream(monkeypatch, None)

    found, marks_set = MarkedFiles.set_marks_from_downstream_related_images(
        "/dir/a.png", "/other", MagicMock()
    )

    assert (found, marks_set) == (None, False)
    assert MarkedFiles.file_marks == ["/dir/old.jpg"]


def test_set_marks_from_downstream_blocks_during_transfer(monkeypatch):
    MarkedFiles.file_marks = ["/dir/old.jpg"]
    _patch_downstream(monkeypatch, ["/other/a_1.png"])
    MarkedFiles.is_performing_action = True
    try:
        found, marks_set = MarkedFiles.set_marks_from_downstream_related_images(
            "/dir/a.png", "/other", MagicMock()
        )
    finally:
        MarkedFiles.is_performing_action = False

    assert (found, marks_set) == (["/other/a_1.png"], False)
    assert MarkedFiles.file_marks == ["/dir/old.jpg"]


# =======================================================================
# Large-operation cancellation behaviour
# (pre-flight for the partial-stop / keep-progress feature)
# =======================================================================


@pytest.fixture()
def _isolated_transfer_state():
    """Save and restore MarkedFiles + FileAction class-level state."""
    saved = {
        "file_marks": MarkedFiles.file_marks[:],
        "previous_marks": MarkedFiles.previous_marks[:],
        "is_performing_action": MarkedFiles.is_performing_action,
        "is_cancelled_action": MarkedFiles.is_cancelled_action,
        "is_shutdown_requested": MarkedFiles.is_shutdown_requested,
        "last_set_target_dir": MarkedFiles.last_set_target_dir,
        "delete_lock": MarkedFiles.delete_lock,
        "action_history": FileAction.action_history[:],
    }
    yield
    MarkedFiles.file_marks = saved["file_marks"]
    MarkedFiles.previous_marks = saved["previous_marks"]
    MarkedFiles.is_performing_action = saved["is_performing_action"]
    MarkedFiles.is_cancelled_action = saved["is_cancelled_action"]
    MarkedFiles.is_shutdown_requested = saved["is_shutdown_requested"]
    MarkedFiles.last_set_target_dir = saved["last_set_target_dir"]
    MarkedFiles.delete_lock = saved["delete_lock"]
    FileAction.action_history = saved["action_history"]


def _make_text_files(directory: Path, count: int) -> list:
    directory.mkdir(exist_ok=True)
    paths = []
    for i in range(count):
        p = directory / f"file{i:03d}.txt"
        p.write_bytes(b"data")
        paths.append(str(p))
    return paths


def _mock_app_actions(base_dir: str) -> MagicMock:
    aa = MagicMock()
    aa.is_compare_running.return_value = False
    aa.get_base_dir.return_value = base_dir
    return aa


def test_progress_callback_stops_loop_after_one_file(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """is_cancelled_action set via progress callback stops the loop after the first file."""
    source = tmp_path / "src"
    target = tmp_path / "dst"
    files = _make_text_files(source, count=5)
    target.mkdir()

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()
    monkeypatch.setattr(MarkedFiles, "undo_move_marks", lambda *a, **kw: None)

    def _cancel_after_first(done: int, total: int) -> None:
        if done >= 1:
            MarkedFiles.is_cancelled_action = True

    result = MarkedFiles.move_marks_to_dir_static(
        _mock_app_actions(str(source)),
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_after_first,
    )

    assert result == (False, False)
    assert len(list(target.iterdir())) == 1


def test_previous_marks_reflects_files_moved_before_cancel(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """previous_marks contains exactly the files moved before cancellation."""
    source = tmp_path / "src"
    target = tmp_path / "dst"
    files = _make_text_files(source, count=4)
    target.mkdir()

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()
    monkeypatch.setattr(MarkedFiles, "undo_move_marks", lambda *a, **kw: None)

    def _cancel_after_two(done: int, total: int) -> None:
        if done >= 2:
            MarkedFiles.is_cancelled_action = True

    MarkedFiles.move_marks_to_dir_static(
        _mock_app_actions(str(source)),
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_after_two,
    )

    assert len(MarkedFiles.previous_marks) == 2
    assert all(p in files for p in MarkedFiles.previous_marks)


def test_revert_choice_calls_undo_move_marks(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """When the user chooses 'revert', undo_move_marks is invoked exactly once."""
    source = tmp_path / "src"
    target = tmp_path / "dst"
    files = _make_text_files(source, count=3)
    target.mkdir()

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()

    undo_calls = []
    monkeypatch.setattr(
        MarkedFiles, "undo_move_marks", lambda *a, **kw: undo_calls.append(True)
    )

    aa = _mock_app_actions(str(source))
    aa.alert.return_value = False  # User picks "Cancel" (revert)

    def _cancel_after_first(done: int, total: int) -> None:
        if done >= 1:
            MarkedFiles.is_cancelled_action = True

    MarkedFiles.move_marks_to_dir_static(
        aa,
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_after_first,
    )

    assert len(undo_calls) == 1


def test_cancellation_mid_transfer_at_scale(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """
    200 × 1 MB sparse files cancelled halfway through mirrors a real video-library move.

    Verifies the loop stops at the cancel point, previous_marks tracks the moved
    files accurately, and undo_move_marks is invoked exactly once.
    """
    FILE_COUNT = 200
    CANCEL_AFTER = 100
    FILE_SIZE = 1024 * 1024  # 1 MB nominal; sparse write avoids real disk I/O

    source = tmp_path / "src"
    target = tmp_path / "dst"
    source.mkdir()
    target.mkdir()

    files = []
    for i in range(FILE_COUNT):
        p = source / f"video_{i:04d}.mp4"
        with open(str(p), "wb") as fh:
            fh.seek(FILE_SIZE - 1)
            fh.write(b"\0")
        files.append(str(p))

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()

    undo_calls = []
    monkeypatch.setattr(
        MarkedFiles, "undo_move_marks", lambda *a, **kw: undo_calls.append(True)
    )

    aa = _mock_app_actions(str(source))
    aa.alert.return_value = False  # User picks "Cancel" (revert)

    def _cancel_at_midpoint(done: int, total: int) -> None:
        if done >= CANCEL_AFTER:
            MarkedFiles.is_cancelled_action = True

    result = MarkedFiles.move_marks_to_dir_static(
        aa,
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_at_midpoint,
    )

    assert result == (False, False)
    assert len(list(target.iterdir())) == CANCEL_AFTER
    assert len(MarkedFiles.previous_marks) == CANCEL_AFTER
    assert len(undo_calls) == 1


def test_keep_choice_leaves_unmoved_files_in_marks(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """Keep-progress path: moved files leave marks; remaining unprocessed files stay marked."""
    source = tmp_path / "src"
    target = tmp_path / "dst"
    files = _make_text_files(source, count=4)
    target.mkdir()

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()

    aa = _mock_app_actions(str(source))
    aa.alert.return_value = True  # User picks "OK" (keep)

    def _cancel_after_two(done: int, total: int) -> None:
        if done >= 2:
            MarkedFiles.is_cancelled_action = True

    MarkedFiles.move_marks_to_dir_static(
        aa,
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_after_two,
    )

    # 2 files moved to target, 2 remain in source as marks
    assert len(list(target.iterdir())) == 2
    assert len(MarkedFiles.file_marks) == 2
    assert all(os.path.exists(f) for f in MarkedFiles.file_marks)
    # Moved files must NOT appear in the remaining marks
    assert not any(f in MarkedFiles.file_marks for f in MarkedFiles.previous_marks)


def test_keep_choice_records_partial_action_in_history(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """Keep path records only the moved files in action history so Ctrl+Z undoes the right set."""
    source = tmp_path / "src"
    target = tmp_path / "dst"
    files = _make_text_files(source, count=5)
    target.mkdir()

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()
    FileAction.action_history.clear()

    aa = _mock_app_actions(str(source))
    aa.alert.return_value = True  # Keep

    def _cancel_after_three(done: int, total: int) -> None:
        if done >= 3:
            MarkedFiles.is_cancelled_action = True

    MarkedFiles.move_marks_to_dir_static(
        aa,
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_after_three,
    )

    assert len(FileAction.action_history) == 1
    recorded = FileAction.action_history[0]
    assert len(recorded.new_files) == 3
    assert recorded.target == str(target)


def test_shutdown_during_transfer_auto_keeps_without_dialog(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """
    When is_shutdown_requested is True the keep path is taken automatically,
    without calling app_actions.alert, so shutdown is never blocked by a dialog.
    """
    source = tmp_path / "src"
    target = tmp_path / "dst"
    files = _make_text_files(source, count=4)
    target.mkdir()

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()
    MarkedFiles.is_shutdown_requested = True
    FileAction.action_history.clear()

    undo_calls = []
    monkeypatch.setattr(
        MarkedFiles, "undo_move_marks", lambda *a, **kw: undo_calls.append(True)
    )

    aa = _mock_app_actions(str(source))

    def _cancel_after_two(done: int, total: int) -> None:
        if done >= 2:
            MarkedFiles.is_cancelled_action = True

    MarkedFiles.move_marks_to_dir_static(
        aa,
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_after_two,
    )

    # No dialog shown and no undo — shutdown auto-kept.
    aa.alert.assert_not_called()
    assert len(undo_calls) == 0
    # 2 files physically in target, 2 remaining stay marked for a follow-up run.
    assert len(list(target.iterdir())) == 2
    assert len(MarkedFiles.file_marks) == 2
    # The partial action must be committed to history so that Ctrl+Z after restart
    # can undo exactly the 2 files that were moved, and delete_lock must be cleared
    # so that undo path is not blocked.
    assert len(FileAction.action_history) == 1
    assert len(FileAction.action_history[0].new_files) == 2
    assert FileAction.action_history[0].target == str(target)
    assert MarkedFiles.delete_lock is False


def test_no_undo_when_all_file_ops_fail_before_cancel(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """undo_move_marks is NOT called when no files were successfully moved before cancel."""
    source = tmp_path / "src"
    target = tmp_path / "dst"
    files = _make_text_files(source, count=3)
    target.mkdir()

    MarkedFiles.file_marks = files[:]
    MarkedFiles.previous_marks.clear()

    undo_calls = []
    monkeypatch.setattr(
        MarkedFiles, "undo_move_marks", lambda *a, **kw: undo_calls.append(True)
    )

    def _failing_move(src, dst, **_kw):
        raise Exception("simulated failure")

    def _cancel_after_first(done: int, total: int) -> None:
        if done >= 1:
            MarkedFiles.is_cancelled_action = True

    MarkedFiles.move_marks_to_dir_static(
        _mock_app_actions(str(source)),
        target_dir=str(target),
        move_func=_failing_move,
        progress_callback=_cancel_after_first,
    )

    assert len(undo_calls) == 0
    assert len(list(target.iterdir())) == 0


# =======================================================================
# Name collisions with files already in the target
# =======================================================================


@pytest.fixture()
def _collision_config(monkeypatch, _isolated_transfer_state):
    """A move onto an existing name fails instead of overwriting, deletes
    are immediate (no send2trash dependency), failed files stay marked, and
    there is no earlier action for the copy-then-move check to find."""
    monkeypatch.setattr(config, "move_marks_overwrite_existing_file", False)
    monkeypatch.setattr(config, "clear_marks_with_errors_after_move", False)
    monkeypatch.setattr(config, "delete_instantly", True)
    monkeypatch.setattr(config, "trash_folder", None)
    monkeypatch.setattr(MarkedFiles, "gimp_opened_in_last_action", False)
    FileAction.action_history.clear()


def _move_marks(source, target, app_actions=None):
    return MarkedFiles.move_marks_to_dir_static(
        app_actions or _mock_app_actions(str(source)),
        target_dir=str(target),
        move_func=Utils.move_file,
    )


def test_identical_existing_file_deletes_the_source_and_unmarks_it(tmp_path, _collision_config):
    source, target = tmp_path / "src", tmp_path / "dst"
    [src_file] = _make_text_files(source, count=1)
    target.mkdir()
    (target / "file000.txt").write_bytes(b"data")
    MarkedFiles.file_marks = [src_file]

    assert _move_marks(source, target) == (True, True)

    assert not os.path.exists(src_file)
    assert (target / "file000.txt").read_bytes() == b"data"
    # Deleted from the source, so not re-marked as a failed transfer.
    assert MarkedFiles.file_marks == []


def test_different_existing_file_keeps_the_source_marked(tmp_path, _collision_config):
    source, target = tmp_path / "src", tmp_path / "dst"
    [src_file] = _make_text_files(source, count=1)
    target.mkdir()
    (target / "file000.txt").write_bytes(b"other")
    MarkedFiles.file_marks = [src_file]

    assert _move_marks(source, target) == (True, True)

    assert os.path.exists(src_file)
    assert (target / "file000.txt").read_bytes() == b"other"
    assert MarkedFiles.file_marks == [src_file]


@pytest.mark.parametrize("approved", [True, False])
def test_several_identical_existing_files_ask_once_for_the_batch(tmp_path, _collision_config, approved):
    source, target = tmp_path / "src", tmp_path / "dst"
    files = _make_text_files(source, count=2)
    target.mkdir()
    for path in files:
        (target / os.path.basename(path)).write_bytes(b"data")
    MarkedFiles.file_marks = files[:]
    aa = _mock_app_actions(str(source))
    aa.alert.return_value = approved

    _move_marks(source, target, aa)

    assert aa.alert.call_count == 1
    assert aa.alert.call_args.args[0] == _("Delete Sources for Existing Files")
    assert [os.path.exists(p) for p in files] == [not approved] * 2
    assert MarkedFiles.file_marks == ([] if approved else files)


def test_same_pixels_with_different_metadata_replaces_the_target(tmp_path, _collision_config):
    source, target = tmp_path / "src", tmp_path / "dst"
    source.mkdir()
    target.mkdir()
    src_file = source / "photo.png"
    info = PngInfo()
    info.add_text("note", "source copy")
    Image.new("RGB", (4, 4), (10, 20, 30)).save(src_file, format="PNG", pnginfo=info)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(target / "photo.png", format="PNG")
    source_bytes = src_file.read_bytes()
    MarkedFiles.file_marks = [str(src_file)]

    # Handling this entry removes it from the exceptions being iterated.
    assert _move_marks(source, target) == (True, True)

    assert not src_file.exists()
    assert (target / "photo.png").read_bytes() == source_bytes
    assert MarkedFiles.file_marks == []

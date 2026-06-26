"""Unit tests for marks-transfer session locking."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from files.file_action import FileAction
from files.marked_files import MarkedFiles
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
        "last_set_target_dir": MarkedFiles.last_set_target_dir,
        "delete_lock": MarkedFiles.delete_lock,
        "action_history": FileAction.action_history[:],
    }
    yield
    MarkedFiles.file_marks = saved["file_marks"]
    MarkedFiles.previous_marks = saved["previous_marks"]
    MarkedFiles.is_performing_action = saved["is_performing_action"]
    MarkedFiles.is_cancelled_action = saved["is_cancelled_action"]
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


def test_undo_called_when_cancelled_with_files_moved(
    tmp_path, monkeypatch, _isolated_transfer_state
):
    """undo_move_marks is invoked exactly once when a cancelled transfer has moved some files."""
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

    def _cancel_after_first(done: int, total: int) -> None:
        if done >= 1:
            MarkedFiles.is_cancelled_action = True

    MarkedFiles.move_marks_to_dir_static(
        _mock_app_actions(str(source)),
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

    def _cancel_at_midpoint(done: int, total: int) -> None:
        if done >= CANCEL_AFTER:
            MarkedFiles.is_cancelled_action = True

    result = MarkedFiles.move_marks_to_dir_static(
        _mock_app_actions(str(source)),
        target_dir=str(target),
        move_func=Utils.move_file,
        progress_callback=_cancel_at_midpoint,
    )

    assert result == (False, False)
    assert len(list(target.iterdir())) == CANCEL_AFTER
    assert len(MarkedFiles.previous_marks) == CANCEL_AFTER
    assert len(undo_calls) == 1


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

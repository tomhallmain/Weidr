"""Undo of a marks transfer (MarkedFiles.undo_move_marks).

Files go back to the directory each came from -- not the directory the window
happens to show -- and undo acts on the transfer that filled previous_marks
(MarkedFiles.previous_action), not on FileAction.action_history[0], which
deletes, image ops and automated moves also write to.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from files.file_action import FileAction
from files.marked_files import MarkedFiles
from utils.config import config
from utils.translations import _
from utils.utils import Utils


@pytest.fixture(autouse=True)
def _isolated_marks_state(monkeypatch):
    saved = {
        "file_marks": MarkedFiles.file_marks[:],
        "previous_marks": MarkedFiles.previous_marks[:],
        "previous_action": MarkedFiles.previous_action,
        "is_performing_action": MarkedFiles.is_performing_action,
        "is_cancelled_action": MarkedFiles.is_cancelled_action,
        "last_set_target_dir": MarkedFiles.last_set_target_dir,
        "delete_lock": MarkedFiles.delete_lock,
        "action_history": FileAction.action_history[:],
    }
    monkeypatch.setattr(config, "move_marks_overwrite_existing_file", False)
    MarkedFiles.is_performing_action = False
    MarkedFiles.delete_lock = False
    yield
    MarkedFiles.file_marks = saved["file_marks"]
    MarkedFiles.previous_marks = saved["previous_marks"]
    MarkedFiles.previous_action = saved["previous_action"]
    MarkedFiles.is_performing_action = saved["is_performing_action"]
    MarkedFiles.is_cancelled_action = saved["is_cancelled_action"]
    MarkedFiles.last_set_target_dir = saved["last_set_target_dir"]
    MarkedFiles.delete_lock = saved["delete_lock"]
    FileAction.action_history = saved["action_history"]


def _files(directory: Path, *names) -> list:
    directory.mkdir(exist_ok=True)
    paths = []
    for name in names:
        p = directory / name
        p.write_bytes(name.encode())
        paths.append(str(p))
    return paths


def _app_actions(base_dir) -> MagicMock:
    aa = MagicMock()
    aa.is_compare_running.return_value = False
    aa.get_base_dir.return_value = str(base_dir)
    return aa


def _transfer(aa, files, target, move_func=Utils.move_file):
    MarkedFiles.file_marks = list(files)
    MarkedFiles.move_marks_to_dir_static(aa, target_dir=str(target), move_func=move_func)


def test_files_go_back_to_their_origin_not_the_window_directory(tmp_path):
    source, target, elsewhere = tmp_path / "src", tmp_path / "dst", tmp_path / "elsewhere"
    files = _files(source, "a.txt", "b.txt")
    target.mkdir()
    elsewhere.mkdir()
    aa = _app_actions(source)
    _transfer(aa, files, target)
    aa.get_base_dir.return_value = str(elsewhere)  # the window moved on

    MarkedFiles.undo_move_marks(aa)

    assert all(os.path.isfile(f) for f in files)
    assert os.listdir(elsewhere) == []
    assert os.listdir(target) == []
    aa.toast.assert_called_with(
        _("Moved back {0} files from {1} to {2}").format(2, str(target), str(source)))


def test_files_from_several_directories_each_go_back(tmp_path):
    first = _files(tmp_path / "one", "a.txt")
    second = _files(tmp_path / "two", "b.txt")
    target = tmp_path / "dst"
    target.mkdir()
    aa = _app_actions(tmp_path / "one")
    _transfer(aa, first + second, target)

    MarkedFiles.undo_move_marks(aa)

    assert os.path.isfile(first[0]) and os.path.isfile(second[0])
    aa.toast.assert_called_with(
        _("Moved back {0} files from {1} to {2} directories").format(2, str(target), 2))


def test_a_later_image_op_in_the_history_does_not_redirect_the_undo(tmp_path):
    """action_history[0] used to decide what to undo: an image op there turned
    the undo into a removal of the moved files."""
    source, target = tmp_path / "src", tmp_path / "dst"
    files = _files(source, "a.txt", "b.txt")
    target.mkdir()
    aa = _app_actions(source)
    _transfer(aa, files, target)
    generated = _files(target, "a_rotated.txt")[0]

    def rotate_image(*_args, **_kwargs):
        return None

    FileAction.update_history(FileAction(rotate_image, str(target), [files[0]], [generated]))

    MarkedFiles.undo_move_marks(aa)

    assert all(os.path.isfile(f) for f in files)
    assert os.path.isfile(generated)


def test_undoing_a_copy_removes_the_copies(tmp_path):
    source, target = tmp_path / "src", tmp_path / "dst"
    files = _files(source, "a.txt")
    target.mkdir()
    aa = _app_actions(source)
    _transfer(aa, files, target, move_func=Utils.copy_file)

    MarkedFiles.undo_move_marks(aa)

    assert os.path.isfile(files[0])
    assert os.listdir(target) == []


def test_a_failed_pair_is_kept_for_another_try(tmp_path):
    source, target = tmp_path / "src", tmp_path / "dst"
    files = _files(source, "a.txt", "b.txt")
    target.mkdir()
    aa = _app_actions(source)
    _transfer(aa, files, target)
    _files(source, "a.txt")  # a new file now blocks a.txt's way back

    with pytest.raises(Exception):
        MarkedFiles.undo_move_marks(aa)

    assert os.path.isfile(files[1])
    assert MarkedFiles.previous_marks == [files[0]]
    assert MarkedFiles.previous_action.new_files == [str(target / "a.txt")]
    # FileAction.__eq__ compares action and target only, so check identity.
    assert all(a is not MarkedFiles.previous_action for a in FileAction.action_history)


def test_nothing_to_undo_changes_nothing(tmp_path):
    MarkedFiles.previous_marks.clear()
    MarkedFiles.previous_action = None
    aa = _app_actions(tmp_path)

    MarkedFiles.undo_move_marks(aa)

    aa.toast.assert_called_once_with(_("No marked files move or copy to undo."))
    aa.refresh.assert_not_called()


def test_delete_lock_is_reported(tmp_path):
    source, target = tmp_path / "src", tmp_path / "dst"
    files = _files(source, "a.txt")
    target.mkdir()
    aa = _app_actions(source)
    _transfer(aa, files, target)
    MarkedFiles.delete_lock = True

    MarkedFiles.undo_move_marks(aa)

    aa.warn.assert_called_once_with(_("Cannot undo the last marked files move after a file was deleted."))
    assert not os.path.exists(files[0])


def test_a_chosen_destination_takes_every_file(tmp_path):
    source, target, chosen = tmp_path / "src", tmp_path / "dst", tmp_path / "chosen"
    files = _files(source, "a.txt", "b.txt")
    target.mkdir()
    chosen.mkdir()
    aa = _app_actions(source)
    _transfer(aa, files, target)
    suggested = []

    def choose(suggested_dir):
        suggested.append(suggested_dir)
        return str(chosen)

    MarkedFiles.undo_move_marks(aa, get_destination_dir_callback=choose)

    assert suggested == [str(source)]
    assert sorted(os.listdir(chosen)) == ["a.txt", "b.txt"]

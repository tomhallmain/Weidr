"""Undoing a marks move from a real AppWindow (Ctrl+Z and Ctrl+X handlers).

Moved files go back to the directories they came from, whatever directory
the window shows by then, and the undo acts on the marks transfer, not on
whatever FileAction was recorded last.
"""

import os

import ui.files.marked_file_mover_qt as mover_module
from files.file_action import FileAction
from files.marked_files import MarkedFiles
from tests.ui.app_window_fixtures import make_png
from ui.files.marked_file_mover_qt import MarkedFileMover
from utils.utils import Utils


def _images(directory):
    # media_dir is the test's tmp_path, which also holds the isolated cache,
    # configs and logs directories.
    return sorted(os.path.join(directory, n) for n in os.listdir(directory) if n.endswith(".png"))


def _move_marks(win, paths, target_dir):
    for p in paths:
        MarkedFiles.add_mark_if_not_present(p, win.app_actions)
    MarkedFiles.move_marks_to_dir_static(
        win.app_actions, target_dir=target_dir, move_func=Utils.move_file,
        current_media=win.app_actions.get_active_media_filepath(),
    )


def _switch_directory(win, qtbot, directory):
    win.set_base_dir(directory)
    qtbot.waitUntil(lambda: win.base_dir == directory, timeout=3000)


def test_ctrl_z_returns_files_to_their_origin_after_the_window_moved_on(window_with_dir, tmp_path_factory, qtbot):
    win, media_dir = window_with_dir
    moved = _images(media_dir)[:2]
    target = tmp_path_factory.mktemp("target")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    make_png(str(elsewhere / "other.png"))

    _move_marks(win, moved, str(target))
    assert sorted(os.listdir(target)) == sorted(os.path.basename(p) for p in moved)
    _switch_directory(win, qtbot, str(elsewhere))

    win.file_marks_ctrl.revert_last_marks_change()

    assert all(os.path.isfile(p) for p in moved)
    assert os.listdir(target) == []
    assert os.listdir(elsewhere) == ["other.png"]


def test_ctrl_z_after_an_image_op_still_moves_the_marked_files_back(window_with_dir, tmp_path_factory):
    """The image op is the newest FileAction; undo must not treat it as the
    transfer (which removed the moved files)."""
    win, media_dir = window_with_dir
    moved = _images(media_dir)[:2]
    target = tmp_path_factory.mktemp("target")
    _move_marks(win, moved, str(target))
    generated = str(target / "generated.png")
    make_png(generated)

    def rotate_image(*_args, **_kwargs):
        return None

    FileAction.update_history(FileAction(rotate_image, str(target), [moved[0]], [generated]))

    win.file_marks_ctrl.revert_last_marks_change()

    assert all(os.path.isfile(p) for p in moved)
    assert os.path.isfile(generated)


def test_ctrl_x_moves_files_to_the_chosen_directory(window_with_dir, tmp_path_factory, monkeypatch):
    win, media_dir = window_with_dir
    moved = _images(media_dir)[:2]
    target = tmp_path_factory.mktemp("target")
    chosen = tmp_path_factory.mktemp("chosen")
    _move_marks(win, moved, str(target))
    offered = []

    def pick(_parent, _title, start_dir, **_kwargs):
        offered.append(start_dir)
        return str(chosen)

    monkeypatch.setattr(mover_module, "get_existing_directory", pick)

    MarkedFileMover.undo_move_marks(win.app_actions, ask_destination=True)

    assert offered == [media_dir]
    assert sorted(os.listdir(chosen)) == sorted(os.path.basename(p) for p in moved)


def test_ctrl_z_with_nothing_to_undo_leaves_files_alone(window_with_dir):
    win, media_dir = window_with_dir
    before = _images(media_dir)

    win.file_marks_ctrl.revert_last_marks_change()

    assert _images(media_dir) == before

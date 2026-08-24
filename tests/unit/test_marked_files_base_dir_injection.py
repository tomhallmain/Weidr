"""The move-undo path resolves the current base directory through a seam.

When a partial transfer is rolled back, the files go back into the directory
being browsed. That directory used to come straight off app_actions; it can now
be supplied directly, with app_actions as the fallback so existing callers are
unaffected.

The distinction that matters here: get_base_dir_callback is NOT this value. It
prompts the user to pick a directory and is only consulted by undo_move_marks
when no directory is supplied at all, so the two must not be conflated.
"""

import inspect

from files.marked_files import MarkedFiles


def _impl_params():
    return inspect.signature(MarkedFiles._move_marks_to_dir_impl).parameters


def _static_params():
    return inspect.signature(MarkedFiles.move_marks_to_dir_static).parameters


class TestSeamExists:
    def test_impl_accepts_the_injection(self):
        assert "get_current_base_dir" in _impl_params()

    def test_static_entry_point_accepts_the_injection(self):
        assert "get_current_base_dir" in _static_params()

    def test_injection_is_optional_on_both(self):
        # Existing callers pass neither, so both must default.
        assert _impl_params()["get_current_base_dir"].default is None
        assert _static_params()["get_current_base_dir"].default is None

    def test_prompting_callback_is_still_a_separate_parameter(self):
        # Conflating the two would replace a silent directory read with a
        # directory-picker dialog.
        params = _impl_params()
        assert "get_base_dir_callback" in params
        assert "get_current_base_dir" in params


class TestUndoStillAcceptsBothSources:
    """undo_move_marks consults the prompting callback only when handed no
    directory. That is what keeps the new seam from changing behaviour: the
    rollback path always supplies one.

    The rollback path itself is not exercised here -- reaching it needs a
    populated FileAction history, a valid target directory and an in-flight
    transfer, which is integration-level setup.
    """

    def test_undo_keeps_the_directory_argument(self):
        params = inspect.signature(MarkedFiles.undo_move_marks).parameters
        assert "base_dir" in params

    def test_undo_keeps_the_prompting_callback_optional(self):
        params = inspect.signature(MarkedFiles.undo_move_marks).parameters
        assert params["get_base_dir_callback"].default is None

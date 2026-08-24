"""End-to-end proof that a compare runs with no Qt at all.

This is the check the headless compare path exists for: assemble the ports --
the Qt-free AppActions, the responsiveness stand-in, the threaded runner -- and
drive a real comparison with no QApplication, no window and no display.
Running a compare is one of several things a headless caller needs to do, not
the only one -- `tests/unit/test_headless_file_marking.py` covers opening a
directory and marking/moving/deleting files, none of which involves a compare.

COLOR_MATCHING is used because it needs no ML model and is deterministic on the
solid-colour fixture: five red, five blue and five green images whose LAB
distances separate cleanly at the default threshold.

Lives in tests/compare/ rather than tests/ui/ precisely because that package
creates no QApplication -- running here is itself part of the assertion.
"""

import threading

import pytest

from compare.compare_args import CompareArgs
from compare.compare_manager import CompareManager
from utils.background_runner import ThreadedTaskRunner
from utils.config import config
from utils.constants import CompareMode, Mode
from utils.headless_app_actions import build_headless_app_actions
from utils.ui_responsiveness import NullResponsiveness


def _manager(base_dir):
    """A CompareManager wired for a caller that has no window."""
    app_actions = build_headless_app_actions({"get_base_dir": lambda: base_dir})
    manager = CompareManager(
        master=None,
        app_actions=app_actions,
        get_base_dir=lambda: base_dir,
        responsiveness=NullResponsiveness(),
    )
    manager.set_primary_mode(CompareMode.COLOR_MATCHING)
    return manager, app_actions


def _group_args(base_dir, app_actions=None):
    # Mode.GROUP matters: with Mode.SEARCH the engine asks for confirmation
    # before switching to a group run, and the headless port declines, so the
    # run would return having done nothing.
    #
    # app_actions is set on the args as well, mirroring what the window does --
    # the engine reaches for it when a stored checkpoint turns out to be
    # unusable.
    return CompareArgs(
        base_dir=base_dir,
        mode=Mode.GROUP,
        compare_mode=CompareMode.COLOR_MATCHING,
        recursive=False,
        store_checkpoints=False,
        app_actions=app_actions,
        # CompareArgs defaults this to embedding_similarity_threshold (0.9),
        # but CompareColors uses the same field as a LAB colour distance where
        # the meaningful value is 15. The window sets it from the sidebar; a
        # caller building args directly has to match it to the mode or nothing
        # groups at all.
        compare_threshold=config.color_diff_threshold,
    )


class TestHeadlessGroupCompare:
    def test_no_qapplication_exists(self):
        # Guards the premise of every other test here. If some import pulled a
        # QApplication into being, this file would stop proving anything.
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is not None:
            pytest.skip(
                "a QApplication already exists -- another package's tests ran "
                "first in this process, so this file's premise cannot be checked here"
            )

    def test_run_groups_the_colour_families(self, compare_colors_dir):
        base_dir = compare_colors_dir["dir"]
        manager, actions = _manager(base_dir)

        manager.run(_group_args(base_dir, actions))

        wrapper = manager._primary_wrapper()
        assert wrapper.file_groups, "no groups were produced"
        # Same bar the GUI-driven test for this fixture uses: the three colour
        # families must each form at least one group.
        assert len(wrapper.file_groups) >= 3, (
            f"expected at least three groups, got {len(wrapper.file_groups)}"
        )

    def test_every_grouped_file_comes_from_the_directory(self, compare_colors_dir):
        base_dir = compare_colors_dir["dir"]
        manager, actions = _manager(base_dir)
        manager.run(_group_args(base_dir, actions))

        grouped = {
            path
            for group in manager._primary_wrapper().file_groups.values()
            for path in group
        }
        expected = set(
            compare_colors_dir["red"]
            + compare_colors_dir["blue"]
            + compare_colors_dir["green"]
            + compare_colors_dir["outliers"]
        )
        assert grouped <= expected
        assert grouped, "nothing was grouped"

    def test_families_are_not_mixed(self, compare_colors_dir):
        base_dir = compare_colors_dir["dir"]
        manager, actions = _manager(base_dir)
        manager.run(_group_args(base_dir, actions))

        families = {}
        for name in ("red", "blue", "green"):
            families.update({p: name for p in compare_colors_dir[name]})

        for group in manager._primary_wrapper().file_groups.values():
            names = {families[p] for p in group if p in families}
            assert len(names) <= 1, f"group mixes colour families: {names}"


class TestHeadlessBackgroundRun:
    """The same run driven through the Qt-free execution strategy."""

    def test_runs_to_completion_on_the_threaded_runner(self, compare_colors_dir):
        base_dir = compare_colors_dir["dir"]
        manager, actions = _manager(base_dir)
        runner = ThreadedTaskRunner()
        errors = []
        finished = threading.Event()

        runner.start(
            manager.run,
            [_group_args(base_dir, actions)],
            on_error=errors.append,
            on_finished=finished.set,
        )

        assert finished.wait(120), "headless compare did not finish"
        assert errors == [], f"compare reported an error: {errors}"
        assert manager._primary_wrapper().file_groups


class TestPortsAreActuallyExercised:
    """The run must go through the stand-ins, not merely avoid crashing."""

    def test_presentation_calls_are_absorbed(self, compare_colors_dir):
        # The engine reports progress and sets labels throughout. Capturing one
        # of those shows the port was on the path, not bypassed.
        base_dir = compare_colors_dir["dir"]
        seen = []
        app_actions = build_headless_app_actions({
            "get_base_dir": lambda: base_dir,
            "_set_label_state": lambda *a, **k: seen.append(a),
        })
        manager = CompareManager(
            master=None,
            app_actions=app_actions,
            get_base_dir=lambda: base_dir,
            responsiveness=NullResponsiveness(),
        )
        manager.set_primary_mode(CompareMode.COLOR_MATCHING)
        manager.run(_group_args(base_dir, app_actions))
        assert seen, "the engine never reported through the presentation port"

    def test_a_window_repaint_request_is_harmless_without_a_window(self, compare_colors_dir):
        # run_group ends by displaying a match, which asks the owning window to
        # repaint. With master=None there is no window; this must not raise.
        base_dir = compare_colors_dir["dir"]
        manager, actions = _manager(base_dir)
        manager.run(_group_args(base_dir, actions))
        manager._primary_wrapper()._request_repaint()

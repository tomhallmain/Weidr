"""Tests for the Qt-free AppActions implementation.

Lives in tests/unit/ deliberately: only tests/ui/conftest.py creates a
QApplication, so everything here runs with no Qt application object at all.
The subprocess test at the end goes further and proves PySide6 is never
imported, which no in-process assertion can show once another test in the same
session has already imported it.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from utils.app_actions import AppActions
from utils.headless_app_actions import (
    DOMAIN_ACTIONS,
    MESSAGE_ACTIONS,
    NEUTRAL_RETURN_ACTIONS,
    NOOP_ACTIONS,
    HeadlessActionUnavailable,
    NullRelatedImagesSignals,
    build_headless_app_actions,
    missing_domain_actions,
)


class TestContractPartition:
    """The four groups must together satisfy the contract AppActions enforces."""

    def test_partition_covers_every_required_action(self):
        covered = (
            set(MESSAGE_ACTIONS)
            | set(NOOP_ACTIONS)
            | set(NEUTRAL_RETURN_ACTIONS)
            | set(DOMAIN_ACTIONS)
        )
        assert AppActions.REQUIRED_ACTIONS - covered == set()

    def test_groups_are_disjoint(self):
        groups = {
            "message": set(MESSAGE_ACTIONS),
            "noop": set(NOOP_ACTIONS),
            "neutral": set(NEUTRAL_RETURN_ACTIONS),
            "domain": set(DOMAIN_ACTIONS),
        }
        names = list(groups)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert groups[a] & groups[b] == set(), f"{a}/{b} overlap"

    def test_builds_without_arguments(self):
        # AppActions.__init__ raises on a missing required action, so a
        # successful build is itself the contract check.
        assert isinstance(build_headless_app_actions(), AppActions)


class TestDisplayActions:
    def test_noop_action_returns_none(self):
        actions = build_headless_app_actions()
        assert actions.refresh() is None
        assert actions.start_loading_spinner() is None

    def test_message_action_accepts_the_call(self):
        actions = build_headless_app_actions()
        assert actions.toast("hello") is None

    def test_neutral_returns(self):
        actions = build_headless_app_actions()
        assert actions.get_window() is None
        assert actions.get_open_windows() == []
        assert actions.is_fullscreen() is False
        assert actions.is_media_muted() is False

    def test_open_windows_list_is_not_shared_between_calls(self):
        actions = build_headless_app_actions()
        first = actions.get_open_windows()
        first.append("leaked")
        assert actions.get_open_windows() == []

    def test_alert_declines_rather_than_consenting(self):
        # The Qt contract is that callers may write `if not result:` for every
        # button mode. False is the answer that declines.
        actions = build_headless_app_actions()
        assert actions.alert("Title", "Body") is False
        assert actions.alert("Title", "Body", kind="askyesno") is False

    def test_warn_and_success_route_through_toast(self):
        seen = []
        actions = build_headless_app_actions({"toast": lambda msg, *a, **k: seen.append(msg)})
        actions.warn("careful")
        actions.success("done")
        assert seen == ["careful", "done"]


class TestDomainActions:
    def test_unsupplied_domain_action_raises_when_called(self):
        actions = build_headless_app_actions()
        with pytest.raises(HeadlessActionUnavailable) as exc:
            actions.delete("/some/file.jpg")
        assert "delete" in str(exc.value)

    def test_unsupplied_domain_action_is_safe_to_reference(self):
        # _build_callbacks() binds hide_current_media without calling it, so
        # merely reaching the attribute must not raise.
        actions = build_headless_app_actions()
        assert callable(actions.hide_current_media)

    def test_supplied_domain_action_is_used(self):
        actions = build_headless_app_actions({"get_base_dir": lambda: "/data"})
        assert actions.get_base_dir() == "/data"

    def test_restore_compare_state_is_a_display_action(self):
        # It announces that moved-out files came back and the UI reacts; the
        # domain half is reachable directly on CompareManager, so headless it
        # is a no-op rather than something the caller must supply.
        assert "restore_compare_state_for_undone_move" in NOOP_ACTIONS
        assert "restore_compare_state_for_undone_move" not in DOMAIN_ACTIONS
        actions = build_headless_app_actions()
        assert actions.restore_compare_state_for_undone_move() is None

    @pytest.mark.parametrize(
        "action, why",
        [
            ("copy_media_path", "writes the clipboard and toasts"),
            ("restart_slideshow_timer_after_interaction", "resets a display timer"),
            ("request_media_blur", "blurs the displayed media"),
        ],
    )
    def test_display_only_actions_need_no_caller_wiring(self, action, why):
        # Each was first taken for a domain call because a Qt-free package
        # invokes it. What it does decides the side it belongs on, not who
        # calls it: none of these outlive a window.
        assert action in NOOP_ACTIONS, why
        assert action not in DOMAIN_ACTIONS, why
        assert getattr(build_headless_app_actions(), action)() is None

    def test_show_next_media_does_not_stand_in_for_the_cursor(self):
        # Advancing the browser cursor is FileBrowser.next_file(); this action
        # is that plus rendering. A headless caller advances the cursor
        # directly, so the port answers False rather than pretending to move.
        assert "show_next_media" in NEUTRAL_RETURN_ACTIONS
        assert "show_next_media" not in DOMAIN_ACTIONS
        assert build_headless_app_actions().show_next_media() is False

    @pytest.mark.parametrize("action", ["go_to_file", "go_to_file_by_index"])
    def test_navigation_reports_that_it_did_not_navigate(self, action):
        # These search for a file and display it; without a screen there is
        # nothing to do. They return bool, so the stub must answer False --
        # "did not navigate" -- rather than a bare None.
        assert action in NEUTRAL_RETURN_ACTIONS
        assert action not in DOMAIN_ACTIONS
        assert getattr(build_headless_app_actions(), action)("anything") is False

    def test_missing_domain_actions_reports_the_gap(self):
        actions = build_headless_app_actions({"get_base_dir": lambda: "/data"})
        missing = missing_domain_actions(actions)
        assert "get_base_dir" not in missing
        assert "delete" in missing

    def test_fully_supplied_has_no_gap(self):
        supplied = {name: (lambda *a, **k: None) for name in DOMAIN_ACTIONS}
        actions = build_headless_app_actions(supplied)
        assert missing_domain_actions(actions) == []

    def test_unknown_action_name_is_rejected(self):
        with pytest.raises(ValueError) as exc:
            build_headless_app_actions({"not_a_real_action": lambda: None})
        assert "not_a_real_action" in str(exc.value)


class TestRelatedImagesBridge:
    def test_signals_are_the_qt_free_stand_in(self):
        actions = build_headless_app_actions()
        assert isinstance(actions.related_images_signals(), NullRelatedImagesSignals)

    def test_notify_does_not_raise(self):
        actions = build_headless_app_actions()
        assert actions.notify_related_images_result("done", "moved", {"n": 1}) is None


_NO_QT_PROBE = textwrap.dedent(
    """
    import sys

    class _BlockPySide6:
        def find_spec(self, name, path=None, target=None):
            if name == "PySide6" or name.startswith("PySide6."):
                raise AssertionError("PySide6 was imported: " + name)
            return None

    sys.meta_path.insert(0, _BlockPySide6())

    from utils.headless_app_actions import build_headless_app_actions

    actions = build_headless_app_actions({"get_base_dir": lambda: "/data"})
    actions.toast("hello")
    actions.refresh()
    assert actions.alert("t", "m") is False
    assert actions.get_base_dir() == "/data"
    actions.notify_related_images_result("done")
    print("NO_QT_OK")
    """
)


def test_usable_without_importing_pyside6(tmp_path):
    """Runs in a fresh interpreter that raises if anything imports PySide6."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("QT_QPA_PLATFORM", None)

    result = subprocess.run(
        [sys.executable, "-c", _NO_QT_PROBE],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"probe failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "NO_QT_OK" in result.stdout

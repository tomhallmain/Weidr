"""
Tests for direct-display prevalidation support (docs/skip-handling-direct-media-display.md):

- dry_run on ClassifierAction: matching returns the would-be action without
  executing run_action, and _dispatch_action executes nothing under dry_run.
- ClassifierActionsManager.advise_media: dry-run advisory for external-context
  files — no base-dir gating, no caching, overrides respected.
- CompareWrapper.show_boundary_match: Home/End across compare matches with a
  config-gated skip loop; End targets the last group/match.
"""
from __future__ import annotations

from types import SimpleNamespace

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction
from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_pipeline_runner import _dispatch_action
from compare.compare_wrapper import CompareWrapper
from compare.lookahead import Lookahead
from files.file_action import FileAction, _delete_file_sentinel
from utils.config import config
from utils.constants import ClassifierActionType, CompareMode
from utils.translations import _
from utils.utils import Utils


# ---------------------------------------------------------------------------
# dry_run on ClassifierAction
# ---------------------------------------------------------------------------

def _matching_action(monkeypatch, action_type) -> tuple[ClassifierAction, list]:
    action = ClassifierAction(name="TestAction", action=action_type)
    executed = []
    monkeypatch.setattr(
        action.__class__, "_evaluate_image_path_match",
        lambda self, image_path, lookahead_eval_cache=None: (True, None),
    )
    monkeypatch.setattr(
        action.__class__, "run_action",
        lambda self, image_path, callbacks, base_directory=None, resolved_category=None: (
            executed.append(image_path) or self.action
        ),
    )
    return action, executed


class TestClassifierActionDryRun:
    def test_dry_run_returns_action_without_executing(self, monkeypatch):
        action, executed = _matching_action(monkeypatch, ClassifierActionType.MOVE)

        result = action.run_on_image_path("/x/a.jpg", ActionCallbacks(), dry_run=True)

        assert result == ClassifierActionType.MOVE
        assert executed == []

    def test_wet_run_still_executes(self, monkeypatch):
        action, executed = _matching_action(monkeypatch, ClassifierActionType.MOVE)

        result = action.run_on_image_path("/x/a.jpg", ActionCallbacks(), dry_run=False)

        assert result == ClassifierActionType.MOVE
        assert executed == ["/x/a.jpg"]

    def test_dry_run_through_run_on_media_path(self, monkeypatch):
        action, executed = _matching_action(monkeypatch, ClassifierActionType.HIDE)

        result = action.run_on_media_path("/x/a.jpg", ActionCallbacks(), dry_run=True)

        assert result == ClassifierActionType.HIDE
        assert executed == []

    def test_no_match_returns_none_in_dry_run(self, monkeypatch):
        action, executed = _matching_action(monkeypatch, ClassifierActionType.MOVE)
        monkeypatch.setattr(
            action.__class__, "_evaluate_image_path_match",
            lambda self, image_path, lookahead_eval_cache=None: (False, None),
        )

        assert action.run_on_image_path("/x/a.jpg", ActionCallbacks(), dry_run=True) is None
        assert executed == []


class TestDispatchActionDryRun:
    def test_dry_run_invokes_no_callbacks(self):
        calls = []
        callbacks = ActionCallbacks(
            hide_callback=lambda *a, **k: calls.append("hide"),
            notify_callback=lambda *a, **k: calls.append("notify"),
        )

        _dispatch_action(
            ClassifierActionType.HIDE, None, "TestPipeline", "/x/a.jpg",
            callbacks, None, dry_run=True,
        )

        assert calls == []

    def test_wet_run_invokes_callbacks(self):
        calls = []
        callbacks = ActionCallbacks(
            hide_callback=lambda *a, **k: calls.append("hide"),
            notify_callback=lambda *a, **k: calls.append("notify"),
        )

        _dispatch_action(
            ClassifierActionType.HIDE, None, "TestPipeline", "/x/a.jpg",
            callbacks, None, dry_run=False,
        )

        assert "hide" in calls


# ---------------------------------------------------------------------------
# advise_media
# ---------------------------------------------------------------------------

def _stub_prevalidation(result_action, received: dict, profile=None,
                        is_move=False, action_modifier=""):
    def run_on_media_path(media_path, callbacks, base_directory=None, dry_run=False):
        received.update(
            media_path=media_path, base_directory=base_directory, dry_run=dry_run
        )
        return result_action

    return SimpleNamespace(
        name="StubPreval",
        is_active=True,
        can_run=True,
        profile=profile,
        action_modifier=action_modifier,
        is_move_action=lambda: is_move,
        media_type_allowed=lambda p: True,
        run_on_media_path=run_on_media_path,
    )


class TestAdviseMedia:
    def _setup_manager(self, monkeypatch, prevalidations):
        monkeypatch.setattr(ClassifierActionsManager, "_prevalidations_initialized", True)
        monkeypatch.setattr(ClassifierActionsManager, "prevalidations", prevalidations)
        monkeypatch.setattr(ClassifierActionsManager, "user_prevalidation_overrides", set())
        monkeypatch.setattr(ClassifierActionsManager, "prevalidated_cache", {})
        monkeypatch.setattr(Lookahead, "lookaheads", [])
        from compare.classifier_pipeline import ClassifierPipelines
        monkeypatch.setattr(
            ClassifierPipelines, "get_prevalidation_pipelines", staticmethod(lambda: [])
        )

    def test_returns_would_be_action_and_name_via_dry_run(self, monkeypatch):
        received: dict = {}
        self._setup_manager(
            monkeypatch, [_stub_prevalidation(ClassifierActionType.MOVE, received)]
        )

        action, name = ClassifierActionsManager.advise_media("/elsewhere/a.jpg")

        assert action == ClassifierActionType.MOVE
        assert name == "StubPreval"
        assert received["dry_run"] is True
        assert received["base_directory"] is None

    def test_no_cache_writes(self, monkeypatch):
        received: dict = {}
        self._setup_manager(
            monkeypatch, [_stub_prevalidation(ClassifierActionType.MOVE, received)]
        )

        ClassifierActionsManager.advise_media("/elsewhere/a.jpg")

        assert ClassifierActionsManager.prevalidated_cache == {}

    def test_overrides_respected(self, monkeypatch):
        received: dict = {}
        self._setup_manager(
            monkeypatch, [_stub_prevalidation(ClassifierActionType.MOVE, received)]
        )
        monkeypatch.setattr(
            ClassifierActionsManager, "user_prevalidation_overrides", {"/elsewhere/a.jpg"}
        )

        assert ClassifierActionsManager.advise_media("/elsewhere/a.jpg") == (None, None)
        assert received == {}

    def test_no_match_returns_none(self, monkeypatch):
        received: dict = {}
        self._setup_manager(monkeypatch, [_stub_prevalidation(None, received)])

        assert ClassifierActionsManager.advise_media("/elsewhere/a.jpg") == (None, None)
        assert received["dry_run"] is True

    def test_origin_base_dir_in_excluded_dirs_suppresses_advisory(self, monkeypatch):
        """A request from a window browsing a known move/copy target must not
        advise at all — browsing there runs no prevalidations either."""
        received: dict = {}
        self._setup_manager(
            monkeypatch, [_stub_prevalidation(ClassifierActionType.MOVE, received)]
        )
        monkeypatch.setattr(
            ClassifierActionsManager, "directories_to_exclude", ["/origin/sorted"]
        )

        result = ClassifierActionsManager.advise_media(
            "/elsewhere/a.jpg", base_dir="/origin/sorted"
        )

        assert result == (None, None)
        assert received == {}

    def test_origin_profile_gating_respected(self, monkeypatch):
        """A profile-gated prevalidation only advises for requests originating
        from a window whose base_dir is in the profile's directories."""
        received: dict = {}
        profile = SimpleNamespace(directories=["/profiled/dir"])
        self._setup_manager(
            monkeypatch,
            [_stub_prevalidation(ClassifierActionType.MOVE, received, profile=profile)],
        )

        assert ClassifierActionsManager.advise_media(
            "/elsewhere/a.jpg", base_dir="/unrelated/dir"
        ) == (None, None)
        assert received == {}

        assert ClassifierActionsManager.advise_media(
            "/elsewhere/a.jpg", base_dir="/profiled/dir"
        ) == (ClassifierActionType.MOVE, "StubPreval")

    def test_origin_move_target_exclusion_respected(self, monkeypatch):
        """A MOVE prevalidation targeting the originating base_dir is skipped,
        matching prevalidate_media's gating."""
        received: dict = {}
        self._setup_manager(
            monkeypatch,
            [_stub_prevalidation(ClassifierActionType.MOVE, received,
                                 is_move=True, action_modifier="/origin/dir")],
        )

        assert ClassifierActionsManager.advise_media(
            "/elsewhere/a.jpg", base_dir="/origin/dir"
        ) == (None, None)
        assert received == {}

        assert ClassifierActionsManager.advise_media(
            "/elsewhere/a.jpg", base_dir="/another/dir"
        ) == (ClassifierActionType.MOVE, "StubPreval")


# ---------------------------------------------------------------------------
# show_boundary_match (compare-mode Home/End)
# ---------------------------------------------------------------------------

def _boundary_wrapper():
    created, toasts = [], []
    app_actions = SimpleNamespace(
        toast=lambda msg, *a, **k: toasts.append(msg),
        create_media=lambda path: created.append(path),
        _set_label_state=lambda *a, **k: None,
    )
    wrapper = CompareWrapper(
        master=SimpleNamespace(update=lambda: None),
        compare_mode=CompareMode.CLIP_EMBEDDING,
        app_actions=app_actions,
    )
    wrapper.file_groups = {
        0: {"/d/a.jpg": 0.1, "/d/b.jpg": 0.2},
        1: {"/d/c.jpg": 0.1, "/d/d.jpg": 0.2},
    }
    wrapper.group_indexes = [0, 1]
    wrapper._compare = SimpleNamespace(
        compare_result=SimpleNamespace(has_meaningful_supergroups=lambda: False)
    )
    return wrapper, created, toasts


class TestShowBoundaryMatch:
    def test_home_shows_first_match_of_first_group(self, monkeypatch):
        wrapper, created, _ = _boundary_wrapper()
        monkeypatch.setattr(config, "prevalidate_on_direct_media_display", True)
        monkeypatch.setattr(CompareWrapper, "skip_media", lambda self, p: False)

        wrapper.show_boundary_match(last_file=False)

        assert wrapper.current_group_index == 0
        assert created == ["/d/a.jpg"]

    def test_end_shows_last_match_of_last_group(self, monkeypatch):
        wrapper, created, _ = _boundary_wrapper()
        monkeypatch.setattr(config, "prevalidate_on_direct_media_display", True)
        monkeypatch.setattr(CompareWrapper, "skip_media", lambda self, p: False)

        wrapper.show_boundary_match(last_file=True)

        assert wrapper.current_group_index == 1
        assert wrapper.match_index == len(wrapper.files_matched) - 1
        assert created == ["/d/d.jpg"]

    def test_skip_loop_advances_past_skipped_media(self, monkeypatch):
        wrapper, created, _ = _boundary_wrapper()
        monkeypatch.setattr(config, "prevalidate_on_direct_media_display", True)
        monkeypatch.setattr(
            CompareWrapper, "skip_media", lambda self, p: p == "/d/d.jpg"
        )

        wrapper.show_boundary_match(last_file=True)

        assert created == ["/d/c.jpg"]

    def test_all_skipped_toasts_and_falls_back_to_start(self, monkeypatch):
        wrapper, created, toasts = _boundary_wrapper()
        monkeypatch.setattr(config, "prevalidate_on_direct_media_display", True)
        monkeypatch.setattr(CompareWrapper, "skip_media", lambda self, p: True)

        wrapper.show_boundary_match(last_file=False)

        assert created == ["/d/a.jpg"]
        assert any(_("All media in the group are skipped") in t for t in toasts)

    def test_config_off_skips_no_prevalidation(self, monkeypatch):
        wrapper, created, _ = _boundary_wrapper()
        monkeypatch.setattr(config, "prevalidate_on_direct_media_display", False)
        skip_calls = []
        monkeypatch.setattr(
            CompareWrapper, "skip_media",
            lambda self, p: skip_calls.append(p) or True,
        )

        wrapper.show_boundary_match(last_file=False)

        assert skip_calls == []
        assert created == ["/d/a.jpg"]

    def test_no_groups_toasts(self, monkeypatch):
        wrapper, created, toasts = _boundary_wrapper()
        wrapper.file_groups = {}
        wrapper.group_indexes = []

        wrapper.show_boundary_match(last_file=False)

        assert created == []
        assert len(toasts) == 1


# ---------------------------------------------------------------------------
# Recent-file-action exemptions (Appendix A.2 / A.3 / A.4)
# ---------------------------------------------------------------------------

def _move_action(new_files, auto=False):
    return FileAction(Utils.move_file, "/target", original_marks=[], new_files=new_files, auto=auto)


class TestWasRecentlyActioned:
    def test_filename_match_regardless_of_directory(self, monkeypatch):
        monkeypatch.setattr(FileAction, "action_history", [_move_action(["/t/a.jpg"])])

        assert FileAction.was_recently_actioned("/completely/other/a.jpg") is True
        assert FileAction.was_recently_actioned("/t/b.jpg") is False

    def test_multi_file_actions_only_check_first_file(self, monkeypatch):
        monkeypatch.setattr(
            FileAction, "action_history",
            [_move_action(["/t/a.jpg", "/t/b.jpg", "/t/c.jpg"])],
        )

        assert FileAction.was_recently_actioned("/x/a.jpg") is True
        assert FileAction.was_recently_actioned("/x/b.jpg") is False

    def test_delete_actions_not_considered(self, monkeypatch):
        delete_action = FileAction(
            _delete_file_sentinel, None, original_marks=["/t/a.jpg"], new_files=["/t/a.jpg"]
        )
        monkeypatch.setattr(FileAction, "action_history", [delete_action])

        assert FileAction.was_recently_actioned("/x/a.jpg") is False

    def test_copy_actions_are_considered(self, monkeypatch):
        copy_action = FileAction(Utils.copy_file, "/target", new_files=["/t/a.jpg"])
        monkeypatch.setattr(FileAction, "action_history", [copy_action])

        assert FileAction.was_recently_actioned("/x/a.jpg") is True

    def test_auto_filter(self, monkeypatch):
        monkeypatch.setattr(
            FileAction, "action_history", [_move_action(["/t/a.jpg"], auto=True)]
        )

        assert FileAction.was_recently_actioned("/x/a.jpg", auto=True) is True
        assert FileAction.was_recently_actioned("/x/a.jpg", auto=False) is False
        assert FileAction.was_recently_actioned("/x/a.jpg", auto=None) is True

    def test_lookback_bound(self, monkeypatch):
        fillers = [_move_action([f"/t/f{i}.jpg"]) for i in range(100)]
        monkeypatch.setattr(
            FileAction, "action_history", fillers + [_move_action(["/t/a.jpg"])]
        )

        assert FileAction.was_recently_actioned("/x/a.jpg", max_actions=100) is False
        assert FileAction.was_recently_actioned("/x/a.jpg", max_actions=101) is True


class TestIsExemptFromDirectDisplayCheck:
    def test_recent_action_exempts(self, monkeypatch):
        monkeypatch.setattr(FileAction, "action_history", [_move_action(["/t/a.jpg"])])
        monkeypatch.setattr(ClassifierActionsManager, "directories_to_exclude", [])

        assert ClassifierActionsManager.is_exempt_from_direct_display_check("/x/a.jpg") is True

    def test_known_move_target_directory_exempts(self, monkeypatch):
        monkeypatch.setattr(FileAction, "action_history", [])
        monkeypatch.setattr(
            ClassifierActionsManager, "directories_to_exclude", ["/sorted/output"]
        )

        assert ClassifierActionsManager.is_exempt_from_direct_display_check(
            "/sorted/output/a.jpg"
        ) is True

    def test_unrelated_file_not_exempt(self, monkeypatch):
        monkeypatch.setattr(FileAction, "action_history", [_move_action(["/t/a.jpg"])])
        monkeypatch.setattr(
            ClassifierActionsManager, "directories_to_exclude", ["/sorted/output"]
        )

        assert ClassifierActionsManager.is_exempt_from_direct_display_check(
            "/elsewhere/b.jpg"
        ) is False

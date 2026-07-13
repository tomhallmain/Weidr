"""
Unit tests for ClassifierActionsTab.run_classifier_action's gather-vs-skip
branching: prototype-only actions must not trigger directory gathering (the
vectorized batch path resolves its own files from directory_paths), every
other validation type must gather media_paths first via
ClassifierActionsManager.gather_sorted_media_paths and pass them through.
"""
from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction
from ui.compare.classifier_actions_tab_qt import ClassifierActionsTab
from utils.constants import ClassifierActionType

_NOOP = lambda *a, **kw: None


def _action(**kwargs) -> ClassifierAction:
    defaults = dict(
        name="test",
        action=ClassifierActionType.NOTIFY,
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_prototype=False,
    )
    defaults.update(kwargs)
    return ClassifierAction(**defaults)


class TestRunClassifierActionGathering:
    def test_prototype_only_skips_gathering(self, monkeypatch):
        gather_calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_actions_tab_qt.ClassifierActionsManager.gather_sorted_media_paths",
            lambda dirs: gather_calls.append(dirs) or [],
        )
        run_calls = []
        ca = _action(use_prototype=True)
        monkeypatch.setattr(ca, "run", lambda *a, **kw: run_calls.append(kw))

        ClassifierActionsTab.run_classifier_action(
            ca, ["/some/dir"], ActionCallbacks(notify_callback=_NOOP)
        )

        assert gather_calls == []
        assert run_calls[0]["media_paths"] is None

    def test_image_classifier_action_gathers_and_passes_media_paths(self, monkeypatch):
        gather_calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_actions_tab_qt.ClassifierActionsManager.gather_sorted_media_paths",
            lambda dirs: gather_calls.append(dirs) or [("/some/dir/a.jpg", "/some/dir")],
        )
        run_calls = []
        ca = _action(use_image_classifier=True)
        monkeypatch.setattr(ca, "run", lambda *a, **kw: run_calls.append(kw))

        ClassifierActionsTab.run_classifier_action(
            ca, ["/some/dir"], ActionCallbacks(notify_callback=_NOOP)
        )

        assert gather_calls == [["/some/dir"]]
        assert run_calls[0]["media_paths"] == [("/some/dir/a.jpg", "/some/dir")]

    def test_prototype_plus_classifier_gathers_too(self, monkeypatch):
        """Prototype enabled alongside another type is not prototype-only,
        so gathering must still happen."""
        gather_calls = []
        monkeypatch.setattr(
            "ui.compare.classifier_actions_tab_qt.ClassifierActionsManager.gather_sorted_media_paths",
            lambda dirs: gather_calls.append(dirs) or [],
        )
        ca = _action(use_prototype=True, use_image_classifier=True)
        monkeypatch.setattr(ca, "run", lambda *a, **kw: None)

        ClassifierActionsTab.run_classifier_action(
            ca, ["/some/dir"], ActionCallbacks(notify_callback=_NOOP)
        )

        assert gather_calls == [["/some/dir"]]

    def test_on_complete_is_forwarded_to_run(self, monkeypatch):
        monkeypatch.setattr(
            "ui.compare.classifier_actions_tab_qt.ClassifierActionsManager.gather_sorted_media_paths",
            lambda dirs: [],
        )
        run_calls = []
        ca = _action(use_image_classifier=True)
        monkeypatch.setattr(ca, "run", lambda *a, **kw: run_calls.append(kw))
        sentinel = lambda stats: None

        ClassifierActionsTab.run_classifier_action(
            ca, ["/some/dir"], ActionCallbacks(notify_callback=_NOOP), on_complete=sentinel
        )

        assert run_calls[0]["on_complete"] is sentinel


class TestNotifyClassifierActionComplete:
    """_notify_classifier_action_complete only touches self._app_actions, so it
    can be tested on a bare instance without constructing the full Qt widget."""

    def _tab_with_fake_app_actions(self):
        tab = ClassifierActionsTab.__new__(ClassifierActionsTab)
        calls = []
        tab._app_actions = type(
            "FakeAppActions", (), {"success": lambda self, msg, time_in_seconds=None: calls.append((msg, time_in_seconds))}
        )()
        return tab, calls

    def test_formats_summary_message(self):
        tab, calls = self._tab_with_fake_app_actions()
        tab._notify_classifier_action_complete({
            "action_name": "Rotate",
            "files_checked": 10,
            "outcomes": 3,
            "moves": 3,
            "copies": 0,
            "deletes": 0,
            "errors": 1,
        })

        assert len(calls) == 1
        message, time_in_seconds = calls[0]
        assert "Rotate" in message
        assert "10" in message
        assert "3" in message
        assert "1" in message
        assert time_in_seconds == 10

    def test_missing_keys_default_to_zero(self):
        tab, calls = self._tab_with_fake_app_actions()
        tab._notify_classifier_action_complete({})
        assert len(calls) == 1

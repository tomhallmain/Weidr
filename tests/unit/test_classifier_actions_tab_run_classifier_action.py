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

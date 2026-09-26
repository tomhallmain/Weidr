"""
UI tests for ClassifierActionsTab — focused on profile-dropdown persistence.

Run with:
    pytest tests/ui/test_classifier_actions_tab_qt.py -v
"""

from __future__ import annotations

import pytest

from files.directory_profile import DirectoryProfile
from ui.compare.classifier_actions_tab_qt import ClassifierActionsTab, _PROFILE_CACHE_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeActions:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _add_profiles(names):
    for name in names:
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name=name, directories=[])
        )


def _make_tab(qtbot, actions=None) -> ClassifierActionsTab:
    tab = ClassifierActionsTab(None, actions or _FakeActions())
    qtbot.addWidget(tab)
    return tab


# ---------------------------------------------------------------------------
# Profile dropdown persistence
# ---------------------------------------------------------------------------

class TestProfileDropdownPersistence:
    def test_profile_combo_populated_from_profiles(self, qtbot, isolated_singletons):
        _add_profiles(["Alpha", "Beta"])
        tab = _make_tab(qtbot)
        items = [tab._profile_combo.itemText(i) for i in range(tab._profile_combo.count())]
        assert "Alpha" in items
        assert "Beta" in items

    def test_initial_selection_restored_from_cache(self, qtbot, isolated_singletons):
        _add_profiles(["X", "Y", "Z"])
        isolated_singletons.set_meta(_PROFILE_CACHE_KEY, "Y")
        tab = _make_tab(qtbot)
        assert tab._profile_combo.currentText() == "Y"

    def test_missing_cached_profile_falls_back_to_first(self, qtbot, isolated_singletons):
        _add_profiles(["P1", "P2"])
        isolated_singletons.set_meta(_PROFILE_CACHE_KEY, "DoesNotExist")
        tab = _make_tab(qtbot)
        assert tab._profile_combo.currentText() == "P1"

    def test_selection_change_updates_cache(self, qtbot, isolated_singletons):
        _add_profiles(["A", "B", "C"])
        tab = _make_tab(qtbot)
        tab._profile_combo.setCurrentText("C")
        assert isolated_singletons.get_meta(_PROFILE_CACHE_KEY) == "C"

    def test_empty_cache_does_not_change_selection(self, qtbot, isolated_singletons):
        _add_profiles(["First", "Second"])
        tab = _make_tab(qtbot)
        assert tab._profile_combo.currentIndex() == 0

    def test_no_profiles_combo_disabled(self, qtbot, isolated_singletons):
        tab = _make_tab(qtbot)
        assert not tab._profile_combo.isEnabled()


# ---------------------------------------------------------------------------
# Pipelines section: activation and runs refuse invalid pipelines
# ---------------------------------------------------------------------------

def _relative_move_pipeline(output_root: str = ""):
    from compare.classifier_pipeline import (
        AlwaysCondition, ClassifierPipeline, NodeOutcome, OutcomeType, PipelineNode,
    )
    from utils.constants import ClassifierActionType
    p = ClassifierPipeline(name="RelativeMove", is_active=False, output_root=output_root)
    p.nodes = [PipelineNode(
        name="move", condition=AlwaysCondition(),
        on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.MOVE,
                             action_modifier="sorted"),
    )]
    return p


@pytest.fixture
def alerts(monkeypatch):
    """Captures qt_alert in both tab modules; set alerts.answer for confirmations."""
    import ui.compare.classifier_actions_tab_qt as actions_mod
    import ui.compare.classifier_pipelines_tab_qt as pipelines_mod

    class _Alerts(list):
        answer = True

    shown = _Alerts()

    def fake(_parent, _title, msg, kind="info"):
        shown.append((msg, kind))
        return shown.answer

    monkeypatch.setattr(actions_mod, "qt_alert", fake)
    monkeypatch.setattr(pipelines_mod, "qt_alert", fake)
    return shown


@pytest.fixture
def batch_runs(monkeypatch):
    from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
    runs = []
    monkeypatch.setattr(
        ClassifierPipelinesTab, "start_batch_run",
        staticmethod(lambda parent, app_actions, pipeline, directories, profile_name:
                     runs.append((pipeline, directories, profile_name))),
    )
    return runs


class TestPipelineActivation:
    def test_invalid_pipeline_is_not_activated(self, qtbot, isolated_singletons, alerts):
        from PySide6.QtWidgets import QCheckBox
        tab = _make_tab(qtbot)
        p = _relative_move_pipeline()
        checkbox = QCheckBox()
        qtbot.addWidget(checkbox)
        checkbox.setChecked(True)
        tab._set_pipeline_active(p, True, checkbox)
        assert p.is_active is False
        assert checkbox.isChecked() is False
        assert [kind for _msg, kind in alerts] == ["error"]

    def test_valid_pipeline_is_activated(self, qtbot, isolated_singletons, alerts, tmp_path):
        tab = _make_tab(qtbot)
        p = _relative_move_pipeline(str(tmp_path))
        tab._set_pipeline_active(p, True)
        assert p.is_active is True
        assert alerts == []

    def test_invalid_pipeline_can_be_deactivated(self, qtbot, isolated_singletons, alerts):
        tab = _make_tab(qtbot)
        p = _relative_move_pipeline()
        p.is_active = True
        tab._set_pipeline_active(p, False)
        assert p.is_active is False
        assert alerts == []


class TestRunSinglePipeline:
    def _tab_with_profile(self, qtbot):
        DirectoryProfile.directory_profiles.append(DirectoryProfile(name="P", directories=["/d1", "/d2"]))
        tab = _make_tab(qtbot)
        tab._profile_combo.setCurrentText("P")
        return tab

    def test_invalid_pipeline_is_refused_before_confirmation(
            self, qtbot, isolated_singletons, alerts, batch_runs):
        from compare import classifier_pipeline_batch as pipeline_batch
        tab = self._tab_with_profile(qtbot)
        p = _relative_move_pipeline()
        tab._run_single_pipeline(p)
        assert alerts == [(str(pipeline_batch.PipelineValidationError(p.name, p.validate())), "error")]
        assert batch_runs == []

    def test_confirmed_run_goes_through_the_batch_run(
            self, qtbot, isolated_singletons, alerts, batch_runs, tmp_path):
        tab = self._tab_with_profile(qtbot)
        p = _relative_move_pipeline(str(tmp_path))
        tab._run_single_pipeline(p)
        assert batch_runs == [(p, ["/d1", "/d2"], "P")]

    def test_cancelled_run_does_not_start(self, qtbot, isolated_singletons, alerts, batch_runs, tmp_path):
        alerts.answer = False
        tab = self._tab_with_profile(qtbot)
        tab._run_single_pipeline(_relative_move_pipeline(str(tmp_path)))
        assert batch_runs == []

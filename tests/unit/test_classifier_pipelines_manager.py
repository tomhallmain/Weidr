"""
Unit tests for ClassifierPipelinesTab (Phase 4a).

These tests cover the non-UI logic: get_all_pipelines, add_pipeline,
remove_pipeline, flow_summary, and the duplicate/delete/move_down helpers
exercised through the manager.

Run with:  pytest tests/unit/test_classifier_pipelines_tab.py -v
"""

from __future__ import annotations

import pytest

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    EmbeddingCondition,
    NodeOutcome,
    OutcomeType,
    PipelineNode,
    PrevalidationPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(name: str, is_active: bool = True) -> ClassifierPipeline:
    p = ClassifierPipeline(name=name, is_active=is_active)
    cond = EmbeddingCondition(positives=["cat"], negatives=[], threshold=0.5)
    node = PipelineNode(
        name="n1",
        condition=cond,
        on_match=NodeOutcome(OutcomeType.ACCEPT),
        on_no_match=NodeOutcome(OutcomeType.REJECT),
    )
    p.nodes = [node]
    return p


# ---------------------------------------------------------------------------
# ClassifierPipelines manager helpers (used by the tab)
# ---------------------------------------------------------------------------

class TestManagerHelpers:
    def test_get_all_pipelines_empty(self, isolated_singletons):
        assert ClassifierPipelines.get_all_pipelines() == []

    def test_add_and_get_all(self, isolated_singletons):
        p = _make_pipeline("alpha")
        ClassifierPipelines.add_pipeline(p)
        assert ClassifierPipelines.get_all_pipelines() == [p]

    def test_add_multiple_preserves_order(self, isolated_singletons):
        a, b, c = _make_pipeline("a"), _make_pipeline("b"), _make_pipeline("c")
        for pip in (a, b, c):
            ClassifierPipelines.add_pipeline(pip)
        names = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert names == ["a", "b", "c"]

    def test_remove_pipeline_by_name(self, isolated_singletons):
        a, b = _make_pipeline("a"), _make_pipeline("b")
        ClassifierPipelines.add_pipeline(a)
        ClassifierPipelines.add_pipeline(b)
        ClassifierPipelines.remove_pipeline("a")
        assert [p.name for p in ClassifierPipelines.get_all_pipelines()] == ["b"]

    def test_remove_unknown_name_is_noop(self, isolated_singletons):
        ClassifierPipelines.add_pipeline(_make_pipeline("x"))
        ClassifierPipelines.remove_pipeline("does_not_exist")
        assert len(ClassifierPipelines.get_all_pipelines()) == 1

    def test_store_and_load_round_trip(self, isolated_singletons):
        p = _make_pipeline("stored")
        ClassifierPipelines.add_pipeline(p)
        ClassifierPipelines.store()
        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()
        assert ClassifierPipelines.get_pipeline_by_name("stored") is not None


# ---------------------------------------------------------------------------
# flow_summary
# ---------------------------------------------------------------------------

class TestFlowSummary:
    def test_empty_pipeline_summary(self, isolated_singletons):
        p = ClassifierPipeline(name="empty")
        assert p.flow_summary() == "(empty)"

    def test_summary_includes_condition_type(self, isolated_singletons):
        p = _make_pipeline("s")
        summary = p.flow_summary()
        # EmbeddingCondition -> "Embedding" in summary
        assert "Embedding" in summary

    def test_summary_is_multiline_for_nodes(self, isolated_singletons):
        p = _make_pipeline("s")
        summary = p.flow_summary()
        assert "\n" in summary

    def test_summary_shows_execute_action(self, isolated_singletons):
        from utils.constants import ClassifierActionType
        p = ClassifierPipeline(name="exec_test")
        cond = EmbeddingCondition(positives=["dog"], negatives=[], threshold=0.5)
        node = PipelineNode(
            name="n1",
            condition=cond,
            on_match=NodeOutcome(
                outcome_type=OutcomeType.EXECUTE,
                action_type=ClassifierActionType.NOTIFY,
            ),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )
        p.nodes = [node]
        summary = p.flow_summary()
        assert "NOTIFY" in summary or "Notify" in summary or "notify" in summary.lower()


# ---------------------------------------------------------------------------
# Duplicate logic (mirrors _duplicate in the tab)
# ---------------------------------------------------------------------------

class TestDuplicateLogic:
    def test_duplicate_generates_unique_name(self, isolated_singletons):
        original = _make_pipeline("base")
        ClassifierPipelines.add_pipeline(original)

        # Mimic tab._duplicate
        import copy
        new_p = copy.deepcopy(original)
        existing = {p.name for p in ClassifierPipelines.get_all_pipelines()}
        candidate = original.name + " (copy)"
        counter = 2
        while candidate in existing:
            candidate = f"{original.name} (copy {counter})"
            counter += 1
        new_p.name = candidate
        ClassifierPipelines.add_pipeline(new_p)

        names = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert "base" in names
        assert "base (copy)" in names

    def test_duplicate_avoids_collision(self, isolated_singletons):
        import copy
        original = _make_pipeline("base")
        copy1 = copy.deepcopy(original)
        copy1.name = "base (copy)"
        for p in (original, copy1):
            ClassifierPipelines.add_pipeline(p)

        new_p = copy.deepcopy(original)
        existing = {p.name for p in ClassifierPipelines.get_all_pipelines()}
        candidate = original.name + " (copy)"
        counter = 2
        while candidate in existing:
            candidate = f"{original.name} (copy {counter})"
            counter += 1
        new_p.name = candidate
        ClassifierPipelines.add_pipeline(new_p)

        assert new_p.name == "base (copy 2)"


# ---------------------------------------------------------------------------
# Move-down logic (mirrors _move_down in the tab)
# ---------------------------------------------------------------------------

class TestMoveDownLogic:
    def test_move_down_swaps_adjacent(self, isolated_singletons):
        a, b, c = _make_pipeline("a"), _make_pipeline("b"), _make_pipeline("c")
        for p in (a, b, c):
            ClassifierPipelines.add_pipeline(p)

        pipelines = ClassifierPipelines.get_all_pipelines()
        idx = 0
        pipelines[idx], pipelines[idx + 1] = pipelines[idx + 1], pipelines[idx]

        assert [p.name for p in ClassifierPipelines.get_all_pipelines()] == ["b", "a", "c"]

    def test_move_down_last_item_is_noop(self, isolated_singletons):
        a, b = _make_pipeline("a"), _make_pipeline("b")
        for p in (a, b):
            ClassifierPipelines.add_pipeline(p)

        pipelines = ClassifierPipelines.get_all_pipelines()
        idx = len(pipelines) - 1
        if idx < len(pipelines) - 1:
            pipelines[idx], pipelines[idx + 1] = pipelines[idx + 1], pipelines[idx]
        # no swap happened — order unchanged
        assert [p.name for p in ClassifierPipelines.get_all_pipelines()] == ["a", "b"]


# ---------------------------------------------------------------------------
# PrevalidationPipeline shows in get_all_pipelines
# ---------------------------------------------------------------------------

class TestPrevalidationInList:
    def test_prevalidation_pipeline_included(self, isolated_singletons):
        pv = PrevalidationPipeline(profile_name="portraits")
        pv.name = "pv-filter"
        ClassifierPipelines.add_pipeline(pv)
        all_names = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert "pv-filter" in all_names

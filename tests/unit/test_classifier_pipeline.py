"""
Unit tests for compare.classifier_pipeline (Phase 1 — core model).

All tests operate on plain Python objects; no real classifiers, cache files,
or user data are touched.  The isolated_singletons + reset_app_globals
autouse fixtures from the root conftest.py keep app_info_cache and config
pointed at per-test temp directories.
"""

import pytest

from compare.classifier_pipeline import (
    # Conditions
    EmbeddingCondition,
    ClassifierRankCondition,
    PrototypeCondition,
    PromptCondition,
    LookaheadCondition,
    NodeResultCondition,
    CompositeCondition,
    _condition_from_dict,
    # Outcome
    NodeOutcome,
    OutcomeType,
    # Node
    PipelineNode,
    # Pipeline
    ClassifierPipeline,
    PrevalidationPipeline,
    ClassifierPipelines,
)
from utils.constants import ClassifierActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(name, condition, on_match=None, on_no_match=None):
    return PipelineNode(
        name=name,
        condition=condition,
        on_match=on_match or NodeOutcome.continue_(),
        on_no_match=on_no_match or NodeOutcome.accept(),
    )


def _simple_pipeline(*nodes, name="test"):
    return ClassifierPipeline(name=name, nodes=list(nodes))


# ---------------------------------------------------------------------------
# Condition round-trip serialization
# ---------------------------------------------------------------------------

class TestConditionSerialization:
    def test_embedding_roundtrip(self):
        c = EmbeddingCondition(positives=["a", "b"], negatives=["c"], threshold=0.5)
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, EmbeddingCondition)
        assert c2.positives == ["a", "b"]
        assert c2.negatives == ["c"]
        assert c2.threshold == 0.5

    def test_classifier_rank_roundtrip(self):
        c = ClassifierRankCondition(
            classifier_name="my_model",
            categories=["cat_a", "cat_b"],
            min_rank=2,
            max_rank=3,
            min_confidence=0.15,
        )
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, ClassifierRankCondition)
        assert c2.classifier_name == "my_model"
        assert c2.categories == ["cat_a", "cat_b"]
        assert c2.min_rank == 2
        assert c2.max_rank == 3
        assert c2.min_confidence == 0.15

    def test_prototype_roundtrip(self):
        c = PrototypeCondition(
            prototype_directory="/pos",
            negative_prototype_directory="/neg",
            threshold=0.3,
            negative_lambda=0.7,
        )
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, PrototypeCondition)
        assert c2.prototype_directory == "/pos"
        assert c2.negative_lambda == 0.7

    def test_prompt_roundtrip(self):
        c = PromptCondition(use_blacklist=True)
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, PromptCondition)
        assert c2.use_blacklist is True

    def test_lookahead_roundtrip(self):
        c = LookaheadCondition(lookahead_name="my_lookahead")
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, LookaheadCondition)
        assert c2.lookahead_name == "my_lookahead"

    def test_node_result_roundtrip(self):
        c = NodeResultCondition(node_name="clip_check", expected_result=False)
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, NodeResultCondition)
        assert c2.node_name == "clip_check"
        assert c2.expected_result is False

    def test_composite_roundtrip(self):
        c = CompositeCondition(
            operator="AND",
            sub_conditions=[
                NodeResultCondition("a", True),
                NodeResultCondition("b", False),
            ],
        )
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, CompositeCondition)
        assert c2.operator == "AND"
        assert len(c2.sub_conditions) == 2
        assert isinstance(c2.sub_conditions[0], NodeResultCondition)
        assert c2.sub_conditions[1].expected_result is False

    def test_nested_composite_roundtrip(self):
        inner = CompositeCondition("OR", [
            EmbeddingCondition(["x"]),
            PromptCondition(),
        ])
        outer = CompositeCondition("NOT", [inner])
        c2 = _condition_from_dict(outer.to_dict())
        assert c2.operator == "NOT"
        assert isinstance(c2.sub_conditions[0], CompositeCondition)
        assert c2.sub_conditions[0].operator == "OR"

    def test_unknown_condition_type_raises(self):
        with pytest.raises(ValueError, match="Unknown condition_type"):
            _condition_from_dict({"condition_type": "does_not_exist"})


# ---------------------------------------------------------------------------
# NodeOutcome round-trip
# ---------------------------------------------------------------------------

class TestNodeOutcome:
    def test_continue_roundtrip(self):
        o = NodeOutcome(OutcomeType.CONTINUE)
        o2 = NodeOutcome.from_dict(o.to_dict())
        assert o2.outcome_type == OutcomeType.CONTINUE
        assert o2.target_node is None

    def test_execute_roundtrip(self):
        o = NodeOutcome(
            OutcomeType.EXECUTE,
            action_type=ClassifierActionType.MOVE,
            action_modifier="/some/dir",
        )
        o2 = NodeOutcome.from_dict(o.to_dict())
        assert o2.outcome_type == OutcomeType.EXECUTE
        assert o2.action_type == ClassifierActionType.MOVE
        assert o2.action_modifier == "/some/dir"

    def test_goto_roundtrip(self):
        o = NodeOutcome(OutcomeType.GOTO, target_node="step_two")
        o2 = NodeOutcome.from_dict(o.to_dict())
        assert o2.outcome_type == OutcomeType.GOTO
        assert o2.target_node == "step_two"

    def test_string_outcome_type_accepted(self):
        o = NodeOutcome("ACCEPT")
        assert o.outcome_type == OutcomeType.ACCEPT

    def test_string_action_type_accepted(self):
        o = NodeOutcome(OutcomeType.EXECUTE, action_type="NOTIFY")
        assert o.action_type == ClassifierActionType.NOTIFY

    def test_summary_execute(self):
        o = NodeOutcome(OutcomeType.EXECUTE,
                        action_type=ClassifierActionType.MOVE,
                        action_modifier="/foo")
        assert "MOVE" in o.summary()
        assert "/foo" in o.summary()

    def test_summary_goto(self):
        o = NodeOutcome(OutcomeType.GOTO, target_node="node_b")
        assert "node_b" in o.summary()


# ---------------------------------------------------------------------------
# PipelineNode round-trip
# ---------------------------------------------------------------------------

class TestPipelineNode:
    def test_roundtrip(self):
        node = _make_node(
            "check",
            EmbeddingCondition(["pos"], [], 0.4),
            on_match=NodeOutcome(OutcomeType.EXECUTE,
                                 action_type=ClassifierActionType.NOTIFY),
        )
        node2 = PipelineNode.from_dict(node.to_dict())
        assert node2.name == "check"
        assert isinstance(node2.condition, EmbeddingCondition)
        assert node2.condition.threshold == 0.4
        assert node2.on_match.outcome_type == OutcomeType.EXECUTE
        assert node2.on_match.action_type == ClassifierActionType.NOTIFY
        assert node2.on_no_match.outcome_type == OutcomeType.ACCEPT


# ---------------------------------------------------------------------------
# ClassifierPipeline round-trip
# ---------------------------------------------------------------------------

class TestClassifierPipelineSerialization:
    def test_empty_pipeline_roundtrip(self):
        p = ClassifierPipeline(name="empty", description="desc")
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.name == "empty"
        assert p2.description == "desc"
        assert p2.nodes == []
        assert p2.is_active is True

    def test_full_pipeline_roundtrip(self):
        p = ClassifierPipeline(
            name="full",
            nodes=[
                _make_node("n1", EmbeddingCondition(["a"])),
                _make_node("n2", ClassifierRankCondition("mdl", ["cat"], 1, 2)),
            ],
            default_action=ClassifierActionType.NOTIFY,
            default_reject_action=ClassifierActionType.HIDE,
            is_active=False,
        )
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.name == "full"
        assert len(p2.nodes) == 2
        assert p2.nodes[1].name == "n2"
        assert p2.default_action == ClassifierActionType.NOTIFY
        assert p2.default_reject_action == ClassifierActionType.HIDE
        assert p2.is_active is False

    def test_none_actions_roundtrip(self):
        p = ClassifierPipeline(name="no_action")
        d = p.to_dict()
        assert d["default_action"] is None
        p2 = ClassifierPipeline.from_dict(d)
        assert p2.default_action is None


# ---------------------------------------------------------------------------
# PrevalidationPipeline
# ---------------------------------------------------------------------------

class TestPrevalidationPipeline:
    def test_roundtrip(self):
        p = PrevalidationPipeline(
            name="prevalidation test",
            profile_name="my_profile",
            nodes=[_make_node("n1", EmbeddingCondition(["x"]))],
        )
        d = p.to_dict()
        assert d["pipeline_class"] == "prevalidation"
        assert d["profile_name"] == "my_profile"

        p2 = PrevalidationPipeline.from_dict(d)
        assert p2.profile_name == "my_profile"
        assert len(p2.nodes) == 1

    def test_none_profile(self):
        p = PrevalidationPipeline(name="no profile")
        assert p.profile_name is None
        p2 = PrevalidationPipeline.from_dict(p.to_dict())
        assert p2.profile_name is None


# ---------------------------------------------------------------------------
# ClassifierPipelines storage
# ---------------------------------------------------------------------------

class TestClassifierPipelinesStorage:
    def test_load_empty(self):
        ClassifierPipelines.load()
        assert ClassifierPipelines.pipelines == []

    def test_store_and_reload(self):
        ClassifierPipelines.pipelines = [
            ClassifierPipeline(name="stored_pipeline",
                               nodes=[_make_node("n1", EmbeddingCondition(["x"]))]),
            PrevalidationPipeline(name="stored_prevalidation", profile_name="prof"),
        ]
        ClassifierPipelines.store()
        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()

        assert len(ClassifierPipelines.pipelines) == 2
        assert ClassifierPipelines.pipelines[0].name == "stored_pipeline"
        assert isinstance(ClassifierPipelines.pipelines[1], PrevalidationPipeline)
        assert ClassifierPipelines.pipelines[1].profile_name == "prof"

    def test_get_pipeline_by_name(self):
        ClassifierPipelines.pipelines = [
            ClassifierPipeline(name="alpha"),
            ClassifierPipeline(name="beta"),
        ]
        assert ClassifierPipelines.get_pipeline_by_name("beta").name == "beta"
        assert ClassifierPipelines.get_pipeline_by_name("gamma") is None

    def test_get_active_for_profile(self):
        ClassifierPipelines.pipelines = [
            PrevalidationPipeline(name="active_match", profile_name="prof_a", is_active=True),
            PrevalidationPipeline(name="inactive_match", profile_name="prof_a", is_active=False),
            PrevalidationPipeline(name="wrong_profile", profile_name="prof_b", is_active=True),
            ClassifierPipeline(name="plain", is_active=True),  # not a PrevalidationPipeline
        ]
        results = ClassifierPipelines.get_active_pipelines_for_profile("prof_a")
        assert len(results) == 1
        assert results[0].name == "active_match"

    def test_corrupt_entry_skipped(self, monkeypatch):
        """A bad cache entry should be skipped, not crash the load."""
        from utils import app_info_cache as aic
        # A node with an unknown condition_type causes _condition_from_dict to raise
        # ValueError, which load() catches and skips.
        bad_node = {
            "name": "n",
            "condition": {"condition_type": "DOES_NOT_EXIST"},
            "on_match": {},
            "on_no_match": {},
        }
        raw = [
            {"name": "good", "nodes": [], "is_active": True},
            {"name": "bad", "nodes": [bad_node], "is_active": True},
        ]
        monkeypatch.setattr(
            aic.app_info_cache, "get_meta",
            lambda key, default_val=None: raw if key == "classifier_pipelines" else default_val,
        )
        ClassifierPipelines.load()
        assert len(ClassifierPipelines.pipelines) == 1
        assert ClassifierPipelines.pipelines[0].name == "good"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_pipeline(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition(["x"]),
                       on_match=NodeOutcome.continue_(),
                       on_no_match=NodeOutcome.accept()),
            _make_node("n2", NodeResultCondition("n1", True),
                       on_match=NodeOutcome(OutcomeType.EXECUTE,
                                            action_type=ClassifierActionType.NOTIFY),
                       on_no_match=NodeOutcome.accept()),
        )
        assert p.validate() == []

    def test_empty_name(self):
        p = ClassifierPipeline(name="  ")
        errors = p.validate()
        assert any("name" in e.lower() for e in errors)

    def test_duplicate_node_names(self):
        p = _simple_pipeline(
            _make_node("same", EmbeddingCondition()),
            _make_node("same", EmbeddingCondition()),
        )
        errors = p.validate()
        assert any("Duplicate" in e for e in errors)

    def test_goto_nonexistent_target(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.GOTO, target_node="ghost")),
        )
        errors = p.validate()
        assert any("ghost" in e for e in errors)

    def test_goto_backward_reference_rejected(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition()),
            _make_node("n2", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.GOTO, target_node="n1")),
        )
        errors = p.validate()
        assert any("cycle" in e.lower() for e in errors)

    def test_goto_forward_reference_ok(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.GOTO, target_node="n2")),
            _make_node("n2", EmbeddingCondition()),
        )
        assert p.validate() == []

    def test_node_result_references_later_node(self):
        p = _simple_pipeline(
            _make_node("n1", NodeResultCondition("n2", True)),
            _make_node("n2", EmbeddingCondition()),
        )
        errors = p.validate()
        assert any("n2" in e and "prior" in e for e in errors)

    def test_node_result_references_nonexistent(self):
        p = _simple_pipeline(
            _make_node("n1", NodeResultCondition("ghost", True)),
        )
        errors = p.validate()
        assert any("ghost" in e for e in errors)

    def test_execute_without_action_type(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=None)),
        )
        errors = p.validate()
        assert any("action_type" in e for e in errors)

    def test_composite_not_requires_one_subcondition(self):
        p = _simple_pipeline(
            _make_node("n1", CompositeCondition("NOT", [
                EmbeddingCondition(), EmbeddingCondition()
            ])),
        )
        errors = p.validate()
        assert any("NOT" in e for e in errors)

    def test_composite_xor_requires_two_subconditions(self):
        p = _simple_pipeline(
            _make_node("n1", CompositeCondition("XOR", [EmbeddingCondition()])),
        )
        errors = p.validate()
        assert any("XOR" in e for e in errors)

    def test_composite_and_requires_at_least_two(self):
        p = _simple_pipeline(
            _make_node("n1", CompositeCondition("AND", [EmbeddingCondition()])),
        )
        errors = p.validate()
        assert any("AND" in e for e in errors)

    def test_classifier_rank_bad_ranks(self):
        p = _simple_pipeline(
            _make_node("n1", ClassifierRankCondition("mdl", ["x"], min_rank=3, max_rank=1)),
        )
        errors = p.validate()
        assert any("max_rank" in e for e in errors)

    def test_classifier_rank_zero_min_rank(self):
        p = _simple_pipeline(
            _make_node("n1", ClassifierRankCondition("mdl", ["x"], min_rank=0, max_rank=1)),
        )
        errors = p.validate()
        assert any("min_rank" in e for e in errors)

    def test_empty_node_name(self):
        p = _simple_pipeline(_make_node("", EmbeddingCondition()))
        errors = p.validate()
        assert any("empty" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Flow preview
# ---------------------------------------------------------------------------

class TestFlowPreview:
    def test_empty_pipeline(self):
        p = ClassifierPipeline(name="empty")
        assert "(no nodes)" in p.flow_preview()

    def test_preview_contains_node_names(self):
        p = _simple_pipeline(
            _make_node("clip_check", EmbeddingCondition(["nsfw"])),
            _make_node("model_check", ClassifierRankCondition("m", ["x"], 1, 1)),
        )
        preview = p.flow_preview()
        assert "clip_check" in preview
        assert "model_check" in preview

    def test_preview_contains_outcome_symbols(self):
        p = _simple_pipeline(_make_node("n1", EmbeddingCondition()))
        preview = p.flow_preview()
        assert "✓" in preview
        assert "✗" in preview

    def test_preview_shows_execute_action(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.EXECUTE,
                                            action_type=ClassifierActionType.MOVE,
                                            action_modifier="/out")),
        )
        preview = p.flow_preview()
        assert "MOVE" in preview
        assert "/out" in preview

    def test_preview_shows_default_action(self):
        p = ClassifierPipeline(
            name="p",
            nodes=[_make_node("n1", EmbeddingCondition())],
            default_action=ClassifierActionType.NOTIFY,
        )
        preview = p.flow_preview()
        assert "NOTIFY" in preview


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

class TestSummaries:
    def test_condition_summaries_dont_crash(self):
        conditions = [
            EmbeddingCondition(["x"], ["y"], 0.3),
            ClassifierRankCondition("m", ["c"], 2, 3, 0.1),
            PrototypeCondition("/a", "/b", 0.25),
            PromptCondition(True),
            LookaheadCondition("lk"),
            NodeResultCondition("prev", False),
            CompositeCondition("OR", [EmbeddingCondition(), PromptCondition()]),
        ]
        for c in conditions:
            assert isinstance(c.summary(), str)
            assert len(c.summary()) > 0

    def test_pipeline_node_condition_summary(self):
        node = _make_node("n", EmbeddingCondition(["test"]))
        assert "Embedding" in node.condition_summary()

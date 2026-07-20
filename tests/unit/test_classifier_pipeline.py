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
    MediaTypeCondition,
    PrototypeCondition,
    PromptCondition,
    FilenameContainsCondition,
    LookaheadCondition,
    NodeResultCondition,
    CompositeCondition,
    BaseStemMatchCondition,
    UnknownSuffixCondition,
    RelatedImageCondition,
    GroupCondition,
    GroupChildResultCondition,
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
from utils.constants import ClassifierActionType, CompareMediaType


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

    def test_always_roundtrip(self):
        from compare.classifier_pipeline import AlwaysCondition
        c = AlwaysCondition()
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, AlwaysCondition)
        assert c2.condition_type == "always"

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

    def test_base_stem_match_roundtrip(self):
        c = BaseStemMatchCondition(require_match=False)
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, BaseStemMatchCondition)
        assert c2.require_match is False

    def test_base_stem_match_defaults(self):
        c = BaseStemMatchCondition()
        assert c.require_match is True
        assert c.suffix_filter == []
        assert c.search_directory == ""
        assert c.max_stem_group_size == 0
        c2 = _condition_from_dict(c.to_dict())
        assert c2.require_match is True
        assert c2.suffix_filter == []
        assert c2.search_directory == ""
        assert c2.max_stem_group_size == 0

    def test_base_stem_match_missing_key_defaults_on_deserialize(self):
        c2 = _condition_from_dict({"condition_type": "base_stem_match"})
        assert isinstance(c2, BaseStemMatchCondition)
        assert c2.require_match is True
        assert c2.suffix_filter == []
        assert c2.search_directory == ""
        assert c2.max_stem_group_size == 0

    def test_base_stem_match_suffix_filter_roundtrip(self):
        c = BaseStemMatchCondition(require_match=True, suffix_filter=["_A", "_ani", "_animal"])
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, BaseStemMatchCondition)
        assert c2.suffix_filter == ["_A", "_ani", "_animal"]
        assert c2.require_match is True

    def test_base_stem_match_suffix_filter_backward_compat_string(self):
        """Old configs serialised suffix_filter as a plain string; must still deserialise."""
        c2 = _condition_from_dict({"condition_type": "base_stem_match", "suffix_filter": "_A"})
        assert c2.suffix_filter == ["_A"]

    def test_base_stem_match_suffix_filter_backward_compat_empty_string(self):
        c2 = _condition_from_dict({"condition_type": "base_stem_match", "suffix_filter": ""})
        assert c2.suffix_filter == []

    def test_base_stem_match_search_directory_roundtrip(self):
        c = BaseStemMatchCondition(search_directory="/some/target/A")
        c2 = _condition_from_dict(c.to_dict())
        assert c2.search_directory == "/some/target/A"

    def test_base_stem_match_summary_found(self):
        assert "found" in BaseStemMatchCondition(require_match=True).summary()

    def test_base_stem_match_summary_not_found(self):
        assert "not found" in BaseStemMatchCondition(require_match=False).summary()

    def test_base_stem_match_summary_includes_suffix_when_set(self):
        s = BaseStemMatchCondition(suffix_filter=["_A", "_ani"]).summary()
        assert "_A" in s
        assert "_ani" in s

    def test_base_stem_match_summary_no_suffix_label_when_empty(self):
        s = BaseStemMatchCondition(suffix_filter=[]).summary()
        assert "suffix" not in s.lower()

    def test_base_stem_match_summary_shows_dir_when_no_suffix(self):
        s = BaseStemMatchCondition(search_directory="/target/A").summary()
        assert "/target/A" in s

    def test_base_stem_match_max_stem_group_size_roundtrip(self):
        c = BaseStemMatchCondition(max_stem_group_size=2)
        c2 = _condition_from_dict(c.to_dict())
        assert c2.max_stem_group_size == 2

    def test_base_stem_match_summary_shows_max_when_set(self):
        s = BaseStemMatchCondition(max_stem_group_size=2).summary()
        assert "max=2" in s

    def test_base_stem_match_summary_no_max_when_zero(self):
        s = BaseStemMatchCondition(max_stem_group_size=0).summary()
        assert "max" not in s

    def test_related_image_roundtrip(self):
        c = RelatedImageCondition(edit_suffix="_edit", search_directory="/custom", count_threshold=3)
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, RelatedImageCondition)
        assert c2.edit_suffix == "_edit"
        assert c2.search_directory == "/custom"
        assert c2.count_threshold == 3

    def test_related_image_defaults(self):
        c = RelatedImageCondition()
        d = c.to_dict()
        c2 = _condition_from_dict(d)
        assert c2.edit_suffix == ""
        assert c2.search_directory == ""
        assert c2.count_threshold == 1
        assert c2.use_configured_search_directories is True

    def test_related_image_use_configured_search_directories_roundtrip(self):
        c = RelatedImageCondition(edit_suffix="_e", use_configured_search_directories=False)
        c2 = _condition_from_dict(c.to_dict())
        assert c2.use_configured_search_directories is False

    def test_related_image_missing_keys_default_on_deserialize(self):
        c2 = _condition_from_dict({"condition_type": "related_image", "edit_suffix": "_v2"})
        assert isinstance(c2, RelatedImageCondition)
        assert c2.edit_suffix == "_v2"
        assert c2.search_directory == ""
        assert c2.count_threshold == 1
        # Old configs without the key should default to True (backward compat).
        assert c2.use_configured_search_directories is True

    def test_group_roundtrip(self):
        child = PipelineNode(
            name="c1",
            condition=FilenameContainsCondition(["draft"], case_sensitive=True),
        )
        c = GroupCondition(operator="AND", nodes=[child])
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, GroupCondition)
        assert c2.operator == "AND"
        assert len(c2.nodes) == 1
        assert c2.nodes[0].name == "c1"
        assert isinstance(c2.nodes[0].condition, FilenameContainsCondition)
        assert c2.nodes[0].condition.patterns == ["draft"]
        assert c2.nodes[0].condition.case_sensitive is True

    def test_group_defaults(self):
        c = GroupCondition()
        assert c.operator == "OR"
        assert c.nodes == []

    def test_group_child_result_roundtrip(self):
        c = GroupChildResultCondition(
            group_node_name="grp",
            child_node_name="child_a",
            expected_result=False,
        )
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, GroupChildResultCondition)
        assert c2.group_node_name == "grp"
        assert c2.child_node_name == "child_a"
        assert c2.expected_result is False

    def test_group_child_result_defaults(self):
        c = GroupChildResultCondition()
        assert c.group_node_name == ""
        assert c.child_node_name == ""
        assert c.expected_result is True

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

    def test_display_summary_uses_translations(self):
        o = NodeOutcome(
            OutcomeType.EXECUTE_AND_CONTINUE,
            action_type=ClassifierActionType.GENERATE,
            action_modifier="_apple",
        )
        text = o.display_summary()
        assert OutcomeType.EXECUTE_AND_CONTINUE.display() in text
        assert ClassifierActionType.GENERATE.get_translation() in text
        assert "_apple" in text
        assert "EXECUTE_AND_CONTINUE" not in text
        assert "GENERATE" not in text


class TestOutcomeTypeDisplay:
    def test_display_values_are_translated_strings(self):
        displays = OutcomeType.display_values()
        assert len(displays) == len(OutcomeType)
        assert "EXECUTE_AND_CONTINUE" not in displays

    def test_get_by_value_name_or_display(self):
        ot = OutcomeType.EXECUTE_AND_CONTINUE
        assert OutcomeType.get(ot.value) is ot
        assert OutcomeType.get(ot.name) is ot
        assert OutcomeType.get(ot.display()) is ot


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

    def test_generation_type_roundtrip(self):
        from utils.constants import ImageGenerationType
        p = ClassifierPipeline(name="gen_type", generation_type=ImageGenerationType.CONTROL_NET)
        d = p.to_dict()
        assert d["generation_type"] == "control_net"
        p2 = ClassifierPipeline.from_dict(d)
        assert p2.generation_type == ImageGenerationType.CONTROL_NET

    def test_generation_type_defaults_to_none(self):
        p = ClassifierPipeline(name="no_gen_type")
        assert p.generation_type is None
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.generation_type is None

    def test_generation_type_backward_compat(self):
        # Dicts without the key (older saves) should deserialise without error.
        d = {"name": "old", "nodes": []}
        p = ClassifierPipeline.from_dict(d)
        assert p.generation_type is None


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
        ClassifierPipelines.store()
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
        ClassifierPipelines._rebuild_type_cache()
        results = ClassifierPipelines.get_active_pipelines_for_profile("prof_a")
        assert len(results) == 1
        assert results[0].name == "active_match"

    def test_corrupt_entry_skipped(self, monkeypatch):
        """A bad cache entry should be skipped, not crash the load."""
        from tests.helpers import isolated_app_info_cache
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
            isolated_app_info_cache(), "get_meta",
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
        from utils.translations import _
        p = _simple_pipeline(
            _make_node("same", EmbeddingCondition()),
            _make_node("same", EmbeddingCondition()),
        )
        errors = p.validate()
        assert any(e == _("Duplicate node name: {0}.").format("same") for e in errors)

    def test_goto_nonexistent_target(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.GOTO, target_node="ghost")),
        )
        errors = p.validate()
        assert any("ghost" in e for e in errors)

    def test_goto_backward_reference_rejected(self):
        from utils.translations import _
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition()),
            _make_node("n2", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.GOTO, target_node="n1")),
        )
        errors = p.validate()
        expected = _("Node {0}: GOTO target {1} must be a later node (cycle prevention).").format("n2", "n1")
        assert any(e == expected for e in errors)

    def test_goto_forward_reference_ok(self):
        p = _simple_pipeline(
            _make_node("n1", EmbeddingCondition(),
                       on_match=NodeOutcome(OutcomeType.GOTO, target_node="n2")),
            _make_node("n2", EmbeddingCondition()),
        )
        assert p.validate() == []

    def test_node_result_references_later_node(self):
        from utils.translations import _
        p = _simple_pipeline(
            _make_node("n1", NodeResultCondition("n2", True)),
            _make_node("n2", EmbeddingCondition()),
        )
        errors = p.validate()
        expected = _("Node {0}: NodeResultCondition references {1} which is not a prior node.").format("n1", "n2")
        assert any(e == expected for e in errors)

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
        from utils.translations import _
        p = _simple_pipeline(_make_node("", EmbeddingCondition()))
        errors = p.validate()
        assert any(e == _("A node has an empty name.") for e in errors)

    def test_related_image_condition_no_edit_suffix_fails(self):
        p = _simple_pipeline(_make_node("n1", RelatedImageCondition(edit_suffix="")))
        errors = p.validate()
        assert any("edit_suffix" in e for e in errors)

    def test_related_image_condition_with_edit_suffix_no_error(self):
        p = _simple_pipeline(_make_node("n1", RelatedImageCondition(edit_suffix="_edit")))
        errors = [e for e in p.validate() if "edit_suffix" in e]
        assert errors == []

    def test_related_image_condition_nonexistent_search_directory_fails(self):
        p = _simple_pipeline(
            _make_node("n1", RelatedImageCondition(
                edit_suffix="_edit",
                search_directory="/definitely/does/not/exist",
            ))
        )
        errors = p.validate()
        assert any("search_directory" in e for e in errors)

    def test_related_image_condition_valid_search_directory_no_error(self, tmp_path):
        p = _simple_pipeline(
            _make_node("n1", RelatedImageCondition(
                edit_suffix="_edit",
                search_directory=str(tmp_path),
            ))
        )
        errors = [e for e in p.validate() if "search_directory" in e]
        assert errors == []

    def test_related_image_condition_empty_search_directory_no_error(self):
        """Empty search_directory falls back to base_directory at runtime; not a validation error."""
        p = _simple_pipeline(
            _make_node("n1", RelatedImageCondition(edit_suffix="_edit", search_directory=""))
        )
        errors = [e for e in p.validate() if "search_directory" in e]
        assert errors == []

    def test_base_stem_match_nonexistent_search_directory_fails(self):
        p = _simple_pipeline(
            _make_node("n1", BaseStemMatchCondition(search_directory="/definitely/does/not/exist"))
        )
        errors = p.validate()
        assert any("search_directory" in e for e in errors)

    def test_base_stem_match_valid_search_directory_no_error(self, tmp_path):
        p = _simple_pipeline(
            _make_node("n1", BaseStemMatchCondition(search_directory=str(tmp_path)))
        )
        errors = [e for e in p.validate() if "search_directory" in e]
        assert errors == []

    def test_base_stem_match_empty_search_directory_no_error(self):
        p = _simple_pipeline(_make_node("n1", BaseStemMatchCondition(search_directory="")))
        errors = [e for e in p.validate() if "search_directory" in e]
        assert errors == []

    # -- UnknownSuffixCondition --

    def test_unknown_suffix_defaults(self):
        c = UnknownSuffixCondition()
        assert c.expected_suffixes == []
        assert c.search_directory == ""
        assert c.classifier_name == ""
        assert c.inference_threshold == 0.85
        assert c.use_base_directory is False

    def test_unknown_suffix_roundtrip(self):
        c = UnknownSuffixCondition(
            expected_suffixes=["_a", "_b"],
            search_directory="/some/dir",
            classifier_name="my_model",
            inference_threshold=0.9,
            use_base_directory=False,
        )
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, UnknownSuffixCondition)
        assert c2.expected_suffixes == ["_a", "_b"]
        assert c2.search_directory == "/some/dir"
        assert c2.classifier_name == "my_model"
        assert c2.inference_threshold == 0.9
        assert c2.use_base_directory is False

    def test_unknown_suffix_use_base_directory_roundtrip(self):
        c = UnknownSuffixCondition(expected_suffixes=["_a"], use_base_directory=True)
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, UnknownSuffixCondition)
        assert c2.use_base_directory is True

    def test_unknown_suffix_missing_keys_defaults(self):
        c2 = _condition_from_dict({"condition_type": "unknown_suffix"})
        assert isinstance(c2, UnknownSuffixCondition)
        assert c2.expected_suffixes == []
        assert c2.inference_threshold == 0.85
        assert c2.use_base_directory is False

    def test_unknown_suffix_summary_shows_suffixes(self):
        s = UnknownSuffixCondition(expected_suffixes=["_a", "_b"]).summary()
        assert "_a" in s and "_b" in s

    def test_unknown_suffix_summary_shows_classifier(self):
        s = UnknownSuffixCondition(classifier_name="mdl").summary()
        assert "mdl" in s

    def test_unknown_suffix_summary_shows_base_dir_flag(self):
        s = UnknownSuffixCondition(use_base_directory=True).summary()
        assert "base" in s

    def test_unknown_suffix_summary_shows_explicit_dir(self):
        s = UnknownSuffixCondition(search_directory="/foo/bar").summary()
        assert "/foo/bar" in s


    def test_unknown_suffix_nonexistent_search_directory_fails(self):
        p = _simple_pipeline(
            _make_node("n1", UnknownSuffixCondition(search_directory="/not/real"))
        )
        errors = p.validate()
        assert any("search_directory" in e for e in errors)

    def test_unknown_suffix_valid_search_directory_no_error(self, tmp_path):
        p = _simple_pipeline(
            _make_node("n1", UnknownSuffixCondition(search_directory=str(tmp_path)))
        )
        errors = [e for e in p.validate() if "search_directory" in e]
        assert errors == []

    def test_related_image_generate_outcome_mismatched_modifier_fails(self):
        p = _simple_pipeline(
            _make_node(
                "n1",
                RelatedImageCondition(edit_suffix="_edit"),
                on_match=NodeOutcome(
                    OutcomeType.EXECUTE,
                    action_type=ClassifierActionType.GENERATE,
                    action_modifier="_different",
                ),
            )
        )
        errors = p.validate()
        assert any("action_modifier" in e for e in errors)

    def test_related_image_generate_outcome_matching_modifier_no_error(self):
        p = _simple_pipeline(
            _make_node(
                "n1",
                RelatedImageCondition(edit_suffix="_edit"),
                on_match=NodeOutcome(
                    OutcomeType.EXECUTE,
                    action_type=ClassifierActionType.GENERATE,
                    action_modifier="_edit",
                ),
            )
        )
        errors = [e for e in p.validate() if "action_modifier" in e]
        assert errors == []

    def test_related_image_non_generate_outcome_modifier_not_checked(self):
        """action_modifier mismatch is only flagged for GENERATE outcomes."""
        p = _simple_pipeline(
            _make_node(
                "n1",
                RelatedImageCondition(edit_suffix="_edit"),
                on_match=NodeOutcome(
                    OutcomeType.EXECUTE,
                    action_type=ClassifierActionType.MOVE,
                    action_modifier="_different",
                ),
            )
        )
        errors = [e for e in p.validate() if "action_modifier" in e]
        assert errors == []

    def test_group_no_children_fails(self):
        from utils.translations import _
        p = _simple_pipeline(_make_node("n1", GroupCondition(operator="OR", nodes=[])))
        errors = p.validate()
        expected = _("Node {0}: GroupCondition has no child nodes.").format("n1")
        assert any(e == expected for e in errors)

    def test_group_bad_operator_fails(self):
        child = PipelineNode("c1", FilenameContainsCondition(["x"]))
        p = _simple_pipeline(_make_node("n1", GroupCondition(operator="XOR", nodes=[child])))
        errors = p.validate()
        assert any("operator" in e.lower() for e in errors)

    def test_group_duplicate_child_names_fails(self):
        from utils.translations import _
        p = _simple_pipeline(_make_node("n1", GroupCondition(operator="OR", nodes=[
            PipelineNode("dup", FilenameContainsCondition(["a"])),
            PipelineNode("dup", FilenameContainsCondition(["b"])),
        ])))
        errors = p.validate()
        expected = _("Node {0}: GroupCondition duplicate child name {1}.").format("n1", "dup")
        assert any(e == expected for e in errors)

    def test_group_valid_passes(self):
        p = _simple_pipeline(_make_node("n1", GroupCondition(operator="OR", nodes=[
            PipelineNode("c1", FilenameContainsCondition(["draft"])),
            PipelineNode("c2", EmbeddingCondition(["portrait"])),
        ])))
        errors = p.validate()
        assert errors == []

    def test_group_child_result_no_group_name_fails(self):
        p = _simple_pipeline(_make_node("n1", GroupChildResultCondition(
            group_node_name="", child_node_name="c1"
        )))
        errors = p.validate()
        assert any("group_node_name" in e for e in errors)

    def test_group_child_result_no_child_name_fails(self):
        p = _simple_pipeline(_make_node("n1", GroupChildResultCondition(
            group_node_name="grp", child_node_name=""
        )))
        errors = p.validate()
        assert any("child_node_name" in e for e in errors)

    def test_group_child_result_references_non_prior_group_fails(self):
        # "grp" doesn't exist as a prior node at all
        p = _simple_pipeline(_make_node("n1", GroupChildResultCondition(
            group_node_name="grp", child_node_name="c1"
        )))
        errors = p.validate()
        assert any("grp" in e for e in errors)

    def test_group_child_result_valid_when_prior_group_exists(self):
        group_node = _make_node("grp", GroupCondition(operator="OR", nodes=[
            PipelineNode("c1", FilenameContainsCondition(["x"])),
        ]))
        check_node = _make_node("check", GroupChildResultCondition(
            group_node_name="grp", child_node_name="c1", expected_result=True
        ))
        p = _simple_pipeline(group_node, check_node)
        assert p.validate() == []


# ---------------------------------------------------------------------------
# Demo pipeline
# ---------------------------------------------------------------------------

class TestDemoPipeline:
    def test_node_count(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        assert len(demo.nodes) == 9

    def test_is_inactive(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        assert demo.is_active is False

    def test_all_condition_types_present(self):
        # ClassifierRankCondition and LookaheadCondition are intentionally excluded from
        # the demo because their validators require runtime-registered resources
        # (classifiers and lookaheads) that don't exist outside a configured session.
        demo = ClassifierPipelines.build_demo_pipeline()
        top_level_types = {
            getattr(n.condition, "condition_type", None) for n in demo.nodes
        }
        expected = {
            "media_type", "group", "group_child_result", "embedding",
            "composite", "prototype", "node_result", "related_image",
        }
        assert expected <= top_level_types, (
            f"Missing condition types: {expected - top_level_types}"
        )

    def test_group_node_has_filename_contains_children(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        group_nodes = [n for n in demo.nodes
                       if getattr(n.condition, "condition_type", None) == "group"]
        assert group_nodes, "Expected at least one GroupCondition node"
        child_types = {
            getattr(c.condition, "condition_type", None)
            for n in group_nodes for c in n.condition.nodes
        }
        assert "filename_contains" in child_types

    def test_composite_contains_embedding_and_prompt(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        composite_nodes = [n for n in demo.nodes
                           if getattr(n.condition, "condition_type", None) == "composite"]
        assert composite_nodes
        sub_types = {
            getattr(s, "condition_type", None)
            for n in composite_nodes for s in n.condition.sub_conditions
        }
        assert "embedding" in sub_types
        assert "prompt" in sub_types

    def test_all_outcome_types_present(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        outcome_types = set()
        for node in demo.nodes:
            outcome_types.add(node.on_match.outcome_type)
            outcome_types.add(node.on_no_match.outcome_type)
        assert OutcomeType.CONTINUE in outcome_types
        assert OutcomeType.GOTO in outcome_types
        assert OutcomeType.EXECUTE in outcome_types
        assert OutcomeType.ACCEPT in outcome_types
        assert OutcomeType.REJECT in outcome_types

    def test_validates_without_errors(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        errors = demo.validate()
        assert errors == [], f"Demo pipeline has validation errors: {errors}"

    def test_roundtrip(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        demo2 = ClassifierPipeline.from_dict(demo.to_dict())
        assert demo2.name == demo.name
        assert len(demo2.nodes) == len(demo.nodes)
        for orig, restored in zip(demo.nodes, demo2.nodes):
            assert orig.name == restored.name
            assert orig.condition.condition_type == restored.condition.condition_type

    def test_flow_summary_contains_all_node_names(self):
        demo = ClassifierPipelines.build_demo_pipeline()
        summary = demo.flow_summary()
        for node in demo.nodes:
            assert node.name in summary


# ---------------------------------------------------------------------------
# Flow preview
# ---------------------------------------------------------------------------

class TestFlowPreview:
    def test_empty_pipeline(self):
        from utils.translations import _
        p = ClassifierPipeline(name="empty")
        assert _("(no nodes)") in p.flow_preview()

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
        assert ClassifierActionType.MOVE.get_translation() in preview
        assert "/out" in preview

    def test_preview_shows_default_action(self):
        p = ClassifierPipeline(
            name="p",
            nodes=[_make_node("n1", EmbeddingCondition())],
            default_action=ClassifierActionType.NOTIFY,
        )
        preview = p.flow_preview()
        assert ClassifierActionType.NOTIFY.get_translation() in preview


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

class TestSummaries:
    def test_condition_summaries_dont_crash(self):
        conditions = [
            EmbeddingCondition(["x"], ["y"], 0.3),
            ClassifierRankCondition("m", ["c"], 2, 3, 0.1),
            PrototypeCondition("/a", "/b", 0.25),
            PromptCondition(use_blacklist=True),
            LookaheadCondition("lk"),
            NodeResultCondition("prev", False),
            CompositeCondition("OR", [EmbeddingCondition(), PromptCondition()]),
            MediaTypeCondition([CompareMediaType.IMAGE, CompareMediaType.GIF]),
            RelatedImageCondition(edit_suffix="_edit", count_threshold=2),
            GroupCondition(operator="AND", nodes=[
                PipelineNode("c1", FilenameContainsCondition(["x"])),
                PipelineNode("c2", EmbeddingCondition(["y"])),
            ]),
            GroupChildResultCondition(
                group_node_name="grp", child_node_name="c1", expected_result=True
            ),
        ]
        for c in conditions:
            assert isinstance(c.summary(), str)
            assert len(c.summary()) > 0

    def test_group_summary_contains_operator_and_child_names(self):
        c = GroupCondition(operator="OR", nodes=[
            PipelineNode("alpha", FilenameContainsCondition(["x"])),
            PipelineNode("beta", EmbeddingCondition(["y"])),
        ])
        s = c.summary()
        assert "OR" in s
        assert "alpha" in s
        assert "beta" in s

    def test_group_child_result_summary_contains_names(self):
        c = GroupChildResultCondition(
            group_node_name="my_group", child_node_name="child_x", expected_result=False
        )
        s = c.summary()
        assert "my_group" in s
        assert "child_x" in s
        assert "False" in s

    def test_pipeline_node_condition_summary(self):
        from utils.translations import _
        node = _make_node("n", EmbeddingCondition(["test"]))
        assert node.condition_summary().startswith(_("Embedding"))

    def test_condition_display_summaries_dont_crash(self):
        from utils.constants import CompareMediaType
        conditions = [
            EmbeddingCondition(["x"], ["y"], 0.3),
            ClassifierRankCondition("m", ["c"], 2, 3, 0.1),
            PrototypeCondition("/a", "/b", 0.25),
            PromptCondition(use_blacklist=True),
            FilenameContainsCondition(["pat"], case_sensitive=True),
            LookaheadCondition("lk"),
            NodeResultCondition("prev", False),
            CompositeCondition("OR", [EmbeddingCondition(), PromptCondition()]),
            MediaTypeCondition([CompareMediaType.IMAGE, CompareMediaType.GIF]),
            BaseStemMatchCondition(require_match=True, suffix_filter=["_apple"]),
            UnknownSuffixCondition(expected_suffixes=["_apple"], use_base_directory=True),
            RelatedImageCondition(edit_suffix="_edit", count_threshold=2),
            GroupCondition(operator="AND", nodes=[
                PipelineNode("c1", FilenameContainsCondition(["x"])),
                PipelineNode("c2", EmbeddingCondition(["y"])),
            ]),
            GroupChildResultCondition(
                group_node_name="grp", child_node_name="c1", expected_result=True
            ),
        ]
        for c in conditions:
            result = c.display_summary()
            assert isinstance(result, str) and len(result) > 0

    def test_display_summary_translates_pipeline_categories_token(self):
        from utils.translations import _
        c = ClassifierRankCondition(classifier_name="m", inherit_categories=True)
        assert _("(pipeline categories)") in c.display_summary()
        assert "(pipeline categories)" in c.summary()  # summary() keeps English literal

    def test_display_summary_blacklist(self):
        from utils.translations import _
        c = PromptCondition(use_blacklist=True)
        assert c.display_summary() == _("Blacklist")

    def test_display_summary_case_sensitivity(self):
        from utils.translations import _
        cs = FilenameContainsCondition(["x"], case_sensitive=True)
        ci = FilenameContainsCondition(["x"], case_sensitive=False)
        assert _("case-sensitive") in cs.display_summary()
        assert _("case-insensitive") in ci.display_summary()

    def test_display_summary_composite_recurses(self):
        c = CompositeCondition("AND", [EmbeddingCondition(["cat"]), PromptCondition()])
        s = c.display_summary()
        assert "AND" in s
        # sub-conditions' display_summary() output should appear
        assert "Embedding" in s or "cat" in s

    def test_display_summary_base_stem_match_translates_mode(self):
        from utils.translations import _
        found = BaseStemMatchCondition(require_match=True)
        not_found = BaseStemMatchCondition(require_match=False)
        assert _("found") in found.display_summary()
        assert _("not found") in not_found.display_summary()

    def test_condition_summary_uses_display_summary(self):
        from utils.translations import _
        node = _make_node("n", EmbeddingCondition(["test"]))
        assert _("Embedding") in node.condition_summary()

    def test_condition_summary_no_condition(self):
        from utils.translations import _
        node = PipelineNode(name="n")
        node.condition = None
        assert node.condition_summary() == _("(no condition)")


# ---------------------------------------------------------------------------
# MediaTypeCondition — model, serialization, validation
# ---------------------------------------------------------------------------

class TestMediaTypeCondition:
    def test_enum_instances_stored(self):
        c = MediaTypeCondition([CompareMediaType.IMAGE, CompareMediaType.VIDEO])
        assert c.media_types == [CompareMediaType.IMAGE, CompareMediaType.VIDEO]

    def test_string_values_coerced(self):
        c = MediaTypeCondition(["image", "pdf", "gif"])
        assert c.media_types == [CompareMediaType.IMAGE, CompareMediaType.PDF, CompareMediaType.GIF]

    def test_mixed_coercion(self):
        c = MediaTypeCondition([CompareMediaType.SVG, "audio"])
        assert c.media_types == [CompareMediaType.SVG, CompareMediaType.AUDIO]

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            MediaTypeCondition(["not_a_type"])

    def test_roundtrip(self):
        c = MediaTypeCondition([CompareMediaType.IMAGE, CompareMediaType.VIDEO])
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, MediaTypeCondition)
        assert c2.media_types == [CompareMediaType.IMAGE, CompareMediaType.VIDEO]

    def test_serializes_to_string_values(self):
        c = MediaTypeCondition([CompareMediaType.PDF, CompareMediaType.GIF])
        d = c.to_dict()
        assert d["condition_type"] == "media_type"
        assert d["media_types"] == ["pdf", "gif"]

    def test_condition_from_dict_dispatch(self):
        c = _condition_from_dict({"condition_type": "media_type", "media_types": ["image"]})
        assert isinstance(c, MediaTypeCondition)
        assert c.media_types == [CompareMediaType.IMAGE]

    def test_empty_list_from_dict(self):
        c = _condition_from_dict({"condition_type": "media_type", "media_types": []})
        assert isinstance(c, MediaTypeCondition)
        assert c.media_types == []

    def test_summary_contains_type_values(self):
        c = MediaTypeCondition([CompareMediaType.IMAGE, CompareMediaType.VIDEO])
        s = c.summary()
        assert "image" in s
        assert "video" in s

    def test_validation_rejects_empty_media_types(self):
        from utils.translations import _
        p = _simple_pipeline(_make_node("n1", MediaTypeCondition([])))
        errors = p.validate()
        expected = _("Node {0}: MediaTypeCondition has no media_types.").format("n1")
        assert any(e == expected for e in errors)

    def test_validation_accepts_non_empty(self):
        p = _simple_pipeline(
            _make_node("n1", MediaTypeCondition([CompareMediaType.IMAGE]),
                       on_match=NodeOutcome(OutcomeType.EXECUTE,
                                            action_type=ClassifierActionType.NOTIFY),
                       on_no_match=NodeOutcome.accept()),
        )
        assert p.validate() == []

    def test_usable_as_composite_sub_condition(self):
        c = CompositeCondition("AND", [
            MediaTypeCondition([CompareMediaType.VIDEO]),
            EmbeddingCondition(["action"]),
        ])
        c2 = _condition_from_dict(c.to_dict())
        assert isinstance(c2, CompositeCondition)
        assert isinstance(c2.sub_conditions[0], MediaTypeCondition)
        assert c2.sub_conditions[0].media_types == [CompareMediaType.VIDEO]


# ---------------------------------------------------------------------------
# ClassifierPipeline.applies_to_media_types — model, serialization
# ---------------------------------------------------------------------------

class TestPipelineAppliesToMediaTypes:
    def test_default_is_none(self):
        p = ClassifierPipeline(name="p")
        assert p.applies_to_media_types is None

    def test_enum_instances_stored(self):
        p = ClassifierPipeline(name="p",
                               applies_to_media_types=[CompareMediaType.IMAGE, CompareMediaType.PDF])
        assert p.applies_to_media_types == [CompareMediaType.IMAGE, CompareMediaType.PDF]

    def test_string_values_coerced(self):
        p = ClassifierPipeline(name="p", applies_to_media_types=["video", "gif"])
        assert p.applies_to_media_types == [CompareMediaType.VIDEO, CompareMediaType.GIF]

    def test_empty_list_collapses_to_none(self):
        p = ClassifierPipeline(name="p", applies_to_media_types=[])
        assert p.applies_to_media_types is None

    def test_none_serializes_to_none(self):
        d = ClassifierPipeline(name="p").to_dict()
        assert d["applies_to_media_types"] is None

    def test_list_serializes_to_string_values(self):
        p = ClassifierPipeline(name="p",
                               applies_to_media_types=[CompareMediaType.IMAGE, CompareMediaType.VIDEO])
        assert p.to_dict()["applies_to_media_types"] == ["image", "video"]

    def test_none_round_trips(self):
        p = ClassifierPipeline(name="p")
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.applies_to_media_types is None

    def test_list_round_trips(self):
        p = ClassifierPipeline(name="p",
                               applies_to_media_types=[CompareMediaType.GIF, CompareMediaType.PDF])
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.applies_to_media_types == [CompareMediaType.GIF, CompareMediaType.PDF]

    def test_backward_compat_missing_key(self):
        d = ClassifierPipeline(name="p").to_dict()
        d.pop("applies_to_media_types")
        p2 = ClassifierPipeline.from_dict(d)
        assert p2.applies_to_media_types is None

    def test_media_type_allowed_none_permits_all(self):
        p = ClassifierPipeline(name="p", applies_to_media_types=None)
        from unittest.mock import patch
        for mt in CompareMediaType:
            with patch("utils.media_utils.get_media_type_for_path", return_value=mt):
                assert p.media_type_allowed("/any/path") is True

    def test_media_type_allowed_restricts(self):
        p = ClassifierPipeline(name="p",
                               applies_to_media_types=[CompareMediaType.IMAGE])
        from unittest.mock import patch
        with patch("utils.media_utils.get_media_type_for_path",
                   return_value=CompareMediaType.IMAGE):
            assert p.media_type_allowed("/img/photo.jpg") is True
        with patch("utils.media_utils.get_media_type_for_path",
                   return_value=CompareMediaType.VIDEO):
            assert p.media_type_allowed("/vid/clip.mp4") is False

    # -- categories field --

    def test_category_map_defaults_empty(self):
        p = ClassifierPipeline(name="p")
        assert p.category_map == {}

    def test_category_map_roundtrip(self):
        m = {"Apple": "_apple", "Banana": "_banana", "Cherry": "_cherry"}
        p = ClassifierPipeline(name="p", category_map=m)
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.category_map == m

    def test_category_map_omitted_from_dict_when_empty(self):
        p = ClassifierPipeline(name="p", category_map={})
        d = p.to_dict()
        assert "category_map" not in d

    def test_category_map_present_in_dict_when_set(self):
        p = ClassifierPipeline(name="p", category_map={"A": "_a", "B": "_b"})
        assert p.to_dict()["category_map"] == {"A": "_a", "B": "_b"}

    def test_category_map_missing_key_defaults_to_empty(self):
        d = ClassifierPipeline(name="p").to_dict()
        d.pop("category_map", None)
        p2 = ClassifierPipeline.from_dict(d)
        assert p2.category_map == {}

    def test_category_map_backward_compat_old_categories_list(self):
        # Old serialised format used "categories": ["_a", "_b"] with no "category_map".
        # from_dict should convert it to an identity map.
        d = {"name": "p", "nodes": [], "categories": ["_apple", "_banana"]}
        p = ClassifierPipeline.from_dict(d)
        assert p.category_map == {"_apple": "_apple", "_banana": "_banana"}

    def test_category_map_new_format_takes_precedence_over_old(self):
        # If both keys are present (shouldn't happen in practice), category_map wins.
        d = {
            "name": "p", "nodes": [],
            "category_map": {"Apple": "_apple"},
            "categories": ["_old"],
        }
        p = ClassifierPipeline.from_dict(d)
        assert p.category_map == {"Apple": "_apple"}

    def test_prevalidation_pipeline_inherits_field(self):
        p = PrevalidationPipeline(name="pv", profile_name="prof",
                                  applies_to_media_types=[CompareMediaType.PDF])
        assert p.applies_to_media_types == [CompareMediaType.PDF]

    def test_prevalidation_pipeline_round_trips(self):
        p = PrevalidationPipeline(name="pv", profile_name="prof",
                                  applies_to_media_types=[CompareMediaType.IMAGE])
        p2 = PrevalidationPipeline.from_dict(p.to_dict())
        assert p2.applies_to_media_types == [CompareMediaType.IMAGE]
        assert p2.profile_name == "prof"

    def test_prevalidation_pipeline_category_map_round_trips(self):
        m = {"Dog": "_dog", "Cat": "_cat"}
        p = PrevalidationPipeline(name="pv", profile_name=None, category_map=m)
        p2 = PrevalidationPipeline.from_dict(p.to_dict())
        assert p2.category_map == m

    def test_prevalidation_backward_compat_old_categories_list(self):
        d = {
            "name": "pv", "nodes": [], "pipeline_class": "prevalidation",
            "profile_name": None, "categories": ["_x", "_y"],
        }
        p = PrevalidationPipeline.from_dict(d)
        assert p.category_map == {"_x": "_x", "_y": "_y"}


# ---------------------------------------------------------------------------
# TestCategoryMapWarnings
# ---------------------------------------------------------------------------

class TestCategoryMapWarnings:
    """validate_warnings() detects suffix mismatches; does not affect validate()."""

    def _pipeline_with_map(self, nodes, category_map):
        return ClassifierPipeline(name="p", nodes=nodes, category_map=category_map)

    def test_no_warnings_when_map_empty(self):
        node = _make_node("n", BaseStemMatchCondition(suffix_filter=["_apple"]))
        p = self._pipeline_with_map([node], {})
        assert p.validate_warnings() == []

    def test_no_warnings_when_all_suffixes_match(self):
        node = _make_node("n", BaseStemMatchCondition(suffix_filter=["_apple", "_banana"]))
        p = self._pipeline_with_map(
            [node], {"Apple": "_apple", "Banana": "_banana"}
        )
        assert p.validate_warnings() == []

    def test_warning_for_base_stem_match_unknown_suffix(self):
        node = _make_node("n", BaseStemMatchCondition(suffix_filter=["_apple", "_unknown"]))
        p = self._pipeline_with_map([node], {"Apple": "_apple"})
        warnings = p.validate_warnings()
        assert len(warnings) == 1
        assert "_unknown" in warnings[0]
        assert "n" in warnings[0]

    def test_warning_for_unknown_suffix_condition(self):
        node = _make_node("n", UnknownSuffixCondition(expected_suffixes=["_apple", "_nope"]))
        p = self._pipeline_with_map([node], {"Apple": "_apple"})
        warnings = p.validate_warnings()
        assert len(warnings) == 1
        assert "_nope" in warnings[0]

    def test_warning_recurses_into_composite(self):
        inner = UnknownSuffixCondition(expected_suffixes=["_apple", "_missing"])
        node = _make_node("guard", CompositeCondition(operator="NOT", sub_conditions=[inner]))
        p = self._pipeline_with_map([node], {"Apple": "_apple"})
        warnings = p.validate_warnings()
        assert len(warnings) == 1
        assert "_missing" in warnings[0]

    def test_warning_recurses_into_group(self):
        child = PipelineNode(
            name="child",
            condition=BaseStemMatchCondition(suffix_filter=["_gone"]),
        )
        outer = _make_node("grp", GroupCondition(operator="OR", nodes=[child]))
        p = self._pipeline_with_map([outer], {"A": "_apple"})
        warnings = p.validate_warnings()
        assert len(warnings) == 1
        assert "_gone" in warnings[0]
        assert "grp/child" in warnings[0]

    def test_multiple_mismatches_reported_separately(self):
        node = _make_node("n", BaseStemMatchCondition(suffix_filter=["_a", "_b", "_c"]))
        p = self._pipeline_with_map([node], {})
        # Empty map → no warnings (map must be non-empty to trigger checks)
        assert p.validate_warnings() == []

    def test_multiple_mismatches_with_non_empty_map(self):
        node = _make_node("n", BaseStemMatchCondition(suffix_filter=["_a", "_b", "_c"]))
        p = self._pipeline_with_map([node], {"A": "_a"})
        warnings = p.validate_warnings()
        assert len(warnings) == 2
        assert any("_b" in w for w in warnings)
        assert any("_c" in w for w in warnings)

    def test_warnings_do_not_block_validate(self):
        # A node with a mismatched suffix should still pass validate()
        node = _make_node("n", BaseStemMatchCondition(suffix_filter=["_unknown"]))
        p = self._pipeline_with_map([node], {"Apple": "_apple"})
        assert p.validate() == []
        assert len(p.validate_warnings()) == 1

    def test_build_category_fill_pipeline_has_no_warnings(self):
        # The demo pipeline's suffix_filter / expected_suffixes should all be
        # values present in its own category_map.
        p = ClassifierPipelines.build_category_fill_pipeline()
        assert p.category_map == {"Apple": "_apple", "Banana": "_banana", "Cherry": "_cherry"}
        assert p.validate_warnings() == []

    def test_pipeline_categories_used_by_runner_are_suffix_values(self):
        # Smoke-check that list(category_map.values()) gives the suffixes.
        p = ClassifierPipelines.build_category_fill_pipeline()
        suffixes = list(p.category_map.values())
        assert set(suffixes) == {"_apple", "_banana", "_cherry"}


# ---------------------------------------------------------------------------
# TestClassifierRankInheritCategories
# ---------------------------------------------------------------------------

class TestClassifierRankInheritCategories:
    """ClassifierRankCondition.inherit_categories serialization and warning behavior."""

    def test_default_is_false(self):
        c = ClassifierRankCondition(classifier_name="m")
        assert c.inherit_categories is False

    def test_roundtrip_false(self):
        c = ClassifierRankCondition(classifier_name="m", inherit_categories=False)
        c2 = _condition_from_dict(c.to_dict())
        assert c2.inherit_categories is False

    def test_roundtrip_true(self):
        c = ClassifierRankCondition(classifier_name="m", inherit_categories=True)
        c2 = _condition_from_dict(c.to_dict())
        assert c2.inherit_categories is True

    def test_backward_compat_missing_key_defaults_false(self):
        d = {"condition_type": "classifier_rank", "classifier_name": "m",
             "categories": ["_a"], "min_rank": 1, "max_rank": 1, "min_confidence": 0.0}
        c = _condition_from_dict(d)
        assert c.inherit_categories is False

    def test_summary_shows_pipeline_categories_when_inherit_and_no_explicit(self):
        c = ClassifierRankCondition(classifier_name="m", inherit_categories=True)
        assert "(pipeline categories)" in c.summary()

    def test_summary_shows_explicit_categories_when_inherit_false(self):
        c = ClassifierRankCondition(classifier_name="m", categories=["_a"], inherit_categories=False)
        assert "_a" in c.summary()
        assert "pipeline" not in c.summary()

    def test_summary_shows_explicit_when_both_inherit_and_categories_set(self):
        # inherit_categories=True is only used by runner when condition.categories is empty.
        # If both are set the explicit list is used; summary reflects that.
        c = ClassifierRankCondition(classifier_name="m", categories=["_a"], inherit_categories=True)
        assert "(pipeline categories)" not in c.summary()

    def test_warning_when_inherit_true_and_no_category_map(self):
        node = _make_node("n", ClassifierRankCondition(
            classifier_name="m", inherit_categories=True
        ))
        p = ClassifierPipeline(name="p", nodes=[node], category_map={})
        warnings = p.validate_warnings()
        assert len(warnings) == 1
        assert "inherit_categories" in warnings[0]
        assert "n" in warnings[0]

    def test_no_warning_when_inherit_true_and_category_map_set(self):
        node = _make_node("n", ClassifierRankCondition(
            classifier_name="m", inherit_categories=True
        ))
        p = ClassifierPipeline(name="p", nodes=[node],
                               category_map={"Apple": "_apple"})
        assert p.validate_warnings() == []

    def test_no_warning_when_inherit_false_regardless_of_map(self):
        node = _make_node("n", ClassifierRankCondition(
            classifier_name="m", inherit_categories=False
        ))
        p = ClassifierPipeline(name="p", nodes=[node], category_map={})
        assert p.validate_warnings() == []

    def test_inherit_does_not_block_validate(self):
        # inherit_categories is not a hard error; validate() ignores it.
        node = _make_node("n", ClassifierRankCondition(
            classifier_name="my_model", inherit_categories=True
        ))
        p = ClassifierPipeline(name="p", nodes=[node], category_map={})
        # validate() may emit a "classifier not registered" error in a live
        # environment, but it should NOT emit anything about inherit_categories.
        errors = p.validate()
        assert not any("inherit" in e for e in errors)


# ---------------------------------------------------------------------------
# TestSeedCategory
# ---------------------------------------------------------------------------

class TestSeedCategory:
    """Tests for the seed_category pipeline-level field."""

    _CAT_MAP = {"Apple": "_apple", "Banana": "_banana"}

    # -- defaults / serialization --

    def test_defaults_to_empty_string(self):
        p = ClassifierPipeline(name="p")
        assert p.seed_category == ""

    def test_roundtrip_with_value(self):
        p = ClassifierPipeline(name="p", category_map=self._CAT_MAP, seed_category="Apple")
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.seed_category == "Apple"

    def test_omitted_from_dict_when_empty(self):
        p = ClassifierPipeline(name="p", category_map=self._CAT_MAP)
        assert "seed_category" not in p.to_dict()

    def test_present_in_dict_when_set(self):
        p = ClassifierPipeline(name="p", category_map=self._CAT_MAP, seed_category="Banana")
        assert p.to_dict()["seed_category"] == "Banana"

    def test_missing_key_in_dict_defaults_to_empty(self):
        d = {"name": "p", "nodes": [], "category_map": self._CAT_MAP}
        p = ClassifierPipeline.from_dict(d)
        assert p.seed_category == ""

    # -- validate() --

    def test_validate_passes_when_key_in_category_map(self):
        p = ClassifierPipeline(name="p", category_map=self._CAT_MAP, seed_category="Apple")
        errors = p.validate()
        assert not any("seed_category" in e for e in errors)

    def test_validate_errors_when_category_map_empty(self):
        from utils.translations import _
        p = ClassifierPipeline(name="p", category_map={}, seed_category="Apple")
        errors = p.validate()
        expected = _("seed_category '{0}' is set but category_map is empty.").format("Apple")
        assert any(e == expected for e in errors)

    def test_validate_errors_when_key_not_in_category_map(self):
        p = ClassifierPipeline(name="p", category_map=self._CAT_MAP, seed_category="Cherry")
        errors = p.validate()
        assert any("seed_category" in e and "Cherry" in e for e in errors)

    def test_validate_no_error_when_seed_category_empty(self):
        # Empty seed_category is valid regardless of category_map state.
        p = ClassifierPipeline(name="p", category_map={})
        assert p.validate() == []


class TestRunSortBy:
    """Tests for the run_sort_by pipeline-level field (batch-run file order)."""

    def test_defaults_to_none(self):
        p = ClassifierPipeline(name="p")
        assert p.run_sort_by is None

    def test_roundtrip_with_value(self):
        from utils.constants import SortBy
        p = ClassifierPipeline(name="p", run_sort_by=SortBy.NAME)
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.run_sort_by == SortBy.NAME

    def test_omitted_from_dict_when_unset(self):
        p = ClassifierPipeline(name="p")
        assert "run_sort_by" not in p.to_dict()

    def test_present_in_dict_when_set_as_get_text_string(self):
        from utils.constants import SortBy
        p = ClassifierPipeline(name="p", run_sort_by=SortBy.NAME)
        assert p.to_dict()["run_sort_by"] == SortBy.NAME.get_text()

    def test_missing_key_in_dict_defaults_to_none(self):
        d = {"name": "p", "nodes": []}
        p = ClassifierPipeline.from_dict(d)
        assert p.run_sort_by is None

    def test_validate_passes_when_unset(self):
        p = ClassifierPipeline(name="p")
        assert p.validate() == []

    def test_validate_passes_for_valid_sort(self):
        from utils.constants import SortBy
        p = ClassifierPipeline(name="p", run_sort_by=SortBy.NAME)
        assert p.validate() == []


class TestSortFilesForRun:
    """Tests for ClassifierPipeline.sort_files_for_run."""

    def test_no_sort_returns_files_unchanged(self):
        p = ClassifierPipeline(name="p")
        files = ["/tmp/b.png", "/tmp/a.png"]
        result = p.sort_files_for_run(files)
        assert result is files  # unset run_sort_by -> identity, no wrapping/sorting at all

    def test_empty_files_list_returns_empty(self):
        from utils.constants import SortBy
        p = ClassifierPipeline(name="p", run_sort_by=SortBy.NAME)
        assert p.sort_files_for_run([]) == []

    def test_name_sort_orders_alphabetically(self):
        import os
        from utils.constants import SortBy
        p = ClassifierPipeline(name="p", run_sort_by=SortBy.NAME)
        files = ["/tmp/b.png", "/tmp/a.png", "/tmp/c.png"]
        result = p.sort_files_for_run(files)
        assert [os.path.basename(f) for f in result] == ["a.png", "b.png", "c.png"]


class TestCategoryMapNumericSuffixValidation:
    def test_numeric_suffix_rejected(self):
        from utils.translations import _
        p = ClassifierPipeline(name="p", category_map={"Size": "_1280"})
        errors = p.validate()
        expected = _("category_map entry '{0}': suffix '{1}' is purely numeric and cannot be used as a suffix.").format(
            "Size", "_1280"
        )
        assert any(e == expected for e in errors)

    def test_numeric_suffix_without_separator_rejected(self):
        from utils.translations import _
        p = ClassifierPipeline(name="p", category_map={"Index": "001"})
        errors = p.validate()
        expected = _("category_map entry '{0}': suffix '{1}' is purely numeric and cannot be used as a suffix.").format(
            "Index", "001"
        )
        assert any(e == expected for e in errors)

    def test_alpha_suffix_accepted(self):
        p = ClassifierPipeline(name="p", category_map={"Apple": "_apple"})
        assert p.validate() == []

    def test_alpha_suffix_with_numeric_variant_accepted(self):
        p = ClassifierPipeline(name="p", category_map={"Apple": "_apple_1"})
        assert p.validate() == []

    def test_multiple_entries_one_numeric_reports_that_entry(self):
        from utils.translations import _
        p = ClassifierPipeline(name="p", category_map={"Apple": "_apple", "Bad": "_999"})
        errors = p.validate()
        expected = _("category_map entry '{0}': suffix '{1}' is purely numeric and cannot be used as a suffix.").format(
            "Bad", "_999"
        )
        assert any(e == expected for e in errors)
        assert not any("Apple" in e for e in errors)


# ---------------------------------------------------------------------------
# move_to_working_dir field
# ---------------------------------------------------------------------------

class TestMoveToWorkingDir:
    def test_default_true(self):
        p = ClassifierPipeline(name="p")
        assert p.move_to_working_dir is True

    def test_explicit_false_stored(self):
        p = ClassifierPipeline(name="p", move_to_working_dir=False)
        assert p.move_to_working_dir is False

    def test_to_dict_includes_field(self):
        p = ClassifierPipeline(name="p", move_to_working_dir=False)
        assert p.to_dict()["move_to_working_dir"] is False

    def test_from_dict_round_trip_false(self):
        p = ClassifierPipeline(name="p", move_to_working_dir=False)
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.move_to_working_dir is False

    def test_from_dict_round_trip_true(self):
        p = ClassifierPipeline(name="p", move_to_working_dir=True)
        p2 = ClassifierPipeline.from_dict(p.to_dict())
        assert p2.move_to_working_dir is True

    def test_backward_compat_missing_key_defaults_true(self):
        d = {"name": "p", "nodes": []}
        p = ClassifierPipeline.from_dict(d)
        assert p.move_to_working_dir is True

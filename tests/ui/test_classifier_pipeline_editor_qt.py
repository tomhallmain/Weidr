"""
UI tests for classifier_pipeline_editor_qt.py (Phase 4b + 4c).

Coverage:
  - _StringListEditor: add, remove, set/get items
  - _EmbeddingPanel: load / get_condition round-trip
  - _ClassifierRankPanel: load / get_condition round-trip
  - _PrototypePanel: load / get_condition round-trip
  - _PromptPanel: load / get_condition, blacklist toggle
  - _LookaheadPanel: load / get_condition
  - _NodeResultPanel: load / get_condition, set_prior_nodes
  - _CompositePanel: add/remove sub-conditions, load / get_condition
  - _OutcomeEditorWidget: all five OutcomeTypes
  - ClassifierPipelineEditorDialog: construction, node CRUD, save logic

Run with:
    pytest tests/ui/test_classifier_pipeline_editor_qt.py -v
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    ClassifierRankCondition,
    CompositeCondition,
    EmbeddingCondition,
    LookaheadCondition,
    NodeOutcome,
    NodeResultCondition,
    OutcomeType,
    PipelineNode,
    PrevalidationPipeline,
    PromptCondition,
    PrototypeCondition,
)
from ui.compare.classifier_pipeline_editor_qt import (
    ClassifierPipelineEditorDialog,
    _BaseStemMatchPanel,
    _ClassifierRankPanel,
    _CompositePanel,
    _EmbeddingPanel,
    _LookaheadPanel,
    _NodeResultPanel,
    _OutcomeEditorWidget,
    _PromptPanel,
    _PrototypePanel,
    _RelatedImagePanel,
    _StringListEditor,
    _SubCondRow,
)
from utils.constants import ClassifierActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeAppActions:
    """Minimal stand-in for app_actions; editor only uses it for run-on-current."""
    current_media_path = None


def _make_node(name: str = "n1") -> PipelineNode:
    return PipelineNode(
        name=name,
        condition=EmbeddingCondition(positives=["cat"], negatives=[], threshold=0.3),
        on_match=NodeOutcome(OutcomeType.ACCEPT),
        on_no_match=NodeOutcome(OutcomeType.REJECT),
    )


def _make_pipeline(name: str = "test_pipe") -> ClassifierPipeline:
    p = ClassifierPipeline(name=name)
    p.nodes = [_make_node("n1"), _make_node("n2")]
    return p


def _open_dialog(qtbot, pipeline=None) -> ClassifierPipelineEditorDialog:
    dlg = ClassifierPipelineEditorDialog(None, _FakeAppActions(), lambda: None, pipeline)
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# _StringListEditor
# ---------------------------------------------------------------------------

class TestStringListEditor:
    def test_initial_empty(self, qtbot):
        w = _StringListEditor()
        qtbot.addWidget(w)
        assert w.get_items() == []

    def test_set_and_get_items(self, qtbot):
        w = _StringListEditor()
        qtbot.addWidget(w)
        w.set_items(["alpha", "beta", "gamma"])
        assert w.get_items() == ["alpha", "beta", "gamma"]

    def test_add_via_entry(self, qtbot):
        w = _StringListEditor()
        qtbot.addWidget(w)
        w._entry.setText("hello")
        w._add_item()
        assert "hello" in w.get_items()

    def test_add_strips_and_clears_entry(self, qtbot):
        w = _StringListEditor()
        qtbot.addWidget(w)
        w._entry.setText("  world  ")
        w._add_item()
        assert w.get_items() == ["world"]
        assert w._entry.text() == ""

    def test_add_empty_string_is_ignored(self, qtbot):
        w = _StringListEditor()
        qtbot.addWidget(w)
        w._entry.setText("   ")
        w._add_item()
        assert w.get_items() == []

    def test_remove_selected(self, qtbot):
        w = _StringListEditor()
        qtbot.addWidget(w)
        w.set_items(["a", "b", "c"])
        w._list.setCurrentRow(1)
        w._remove_selected()
        assert w.get_items() == ["a", "c"]

    def test_set_items_replaces_previous(self, qtbot):
        w = _StringListEditor()
        qtbot.addWidget(w)
        w.set_items(["old"])
        w.set_items(["new1", "new2"])
        assert w.get_items() == ["new1", "new2"]


# ---------------------------------------------------------------------------
# _EmbeddingPanel
# ---------------------------------------------------------------------------

class TestEmbeddingPanel:
    def test_default_load_none(self, qtbot):
        p = _EmbeddingPanel()
        qtbot.addWidget(p)
        p.load(None)
        c = p.get_condition()
        assert isinstance(c, EmbeddingCondition)
        assert c.positives == []
        assert c.negatives == []
        assert abs(c.threshold - 0.23) < 1e-6

    def test_round_trip(self, qtbot):
        p = _EmbeddingPanel()
        qtbot.addWidget(p)
        original = EmbeddingCondition(positives=["dog", "puppy"], negatives=["cat"], threshold=0.45)
        p.load(original)
        result = p.get_condition()
        assert result.positives == ["dog", "puppy"]
        assert result.negatives == ["cat"]
        assert abs(result.threshold - 0.45) < 1e-6

    def test_condition_type(self, qtbot):
        p = _EmbeddingPanel()
        qtbot.addWidget(p)
        p.load(None)
        assert p.get_condition().condition_type == "embedding"

    def test_load_wrong_type_uses_defaults(self, qtbot):
        p = _EmbeddingPanel()
        qtbot.addWidget(p)
        p.load(PromptCondition(use_blacklist=True))
        c = p.get_condition()
        assert c.positives == []


# ---------------------------------------------------------------------------
# _ClassifierRankPanel
# ---------------------------------------------------------------------------

class TestClassifierRankPanel:
    def test_default_load_none(self, qtbot):
        p = _ClassifierRankPanel()
        qtbot.addWidget(p)
        p.load(None)
        c = p.get_condition()
        assert isinstance(c, ClassifierRankCondition)
        assert c.categories == []
        assert c.min_rank == 1
        assert c.max_rank == 1
        assert c.min_confidence == 0.0

    def test_round_trip(self, qtbot):
        p = _ClassifierRankPanel()
        qtbot.addWidget(p)
        # Set categories manually via the list editor
        p._categories.set_items(["explicit", "suggestive"])
        p._min_rank.setValue(2)
        p._max_rank.setValue(3)
        p._min_confidence.setValue(0.15)
        c = p.get_condition()
        assert c.categories == ["explicit", "suggestive"]
        assert c.min_rank == 2
        assert c.max_rank == 3
        assert abs(c.min_confidence - 0.15) < 1e-6

    def test_condition_type(self, qtbot):
        p = _ClassifierRankPanel()
        qtbot.addWidget(p)
        assert p.get_condition().condition_type == "classifier_rank"

    def test_load_condition_sets_categories(self, qtbot):
        p = _ClassifierRankPanel()
        qtbot.addWidget(p)
        cond = ClassifierRankCondition(
            classifier_name="my_model",
            categories=["safe"],
            min_rank=1,
            max_rank=2,
            min_confidence=0.5,
        )
        p.load(cond)
        c = p.get_condition()
        assert c.categories == ["safe"]
        assert c.min_rank == 1
        assert c.max_rank == 2
        assert abs(c.min_confidence - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# _PrototypePanel
# ---------------------------------------------------------------------------

class TestPrototypePanel:
    def test_default_load_none(self, qtbot):
        p = _PrototypePanel()
        qtbot.addWidget(p)
        p.load(None)
        c = p.get_condition()
        assert isinstance(c, PrototypeCondition)
        assert c.prototype_directory == ""
        assert c.negative_prototype_directory == ""

    def test_round_trip(self, qtbot):
        p = _PrototypePanel()
        qtbot.addWidget(p)
        original = PrototypeCondition(
            prototype_directory="/pos",
            negative_prototype_directory="/neg",
            threshold=0.6,
            negative_lambda=0.3,
        )
        p.load(original)
        result = p.get_condition()
        assert result.prototype_directory == "/pos"
        assert result.negative_prototype_directory == "/neg"
        assert abs(result.threshold - 0.6) < 1e-6
        assert abs(result.negative_lambda - 0.3) < 1e-6

    def test_condition_type(self, qtbot):
        p = _PrototypePanel()
        qtbot.addWidget(p)
        assert p.get_condition().condition_type == "prototype"


# ---------------------------------------------------------------------------
# _PromptPanel
# ---------------------------------------------------------------------------

class TestPromptPanel:
    def test_default_load_none(self, qtbot):
        p = _PromptPanel()
        qtbot.addWidget(p)
        p.load(None)
        c = p.get_condition()
        assert isinstance(c, PromptCondition)
        assert c.use_blacklist is False
        assert c.prompts == []

    def test_round_trip_with_prompts(self, qtbot):
        p = _PromptPanel()
        qtbot.addWidget(p)
        original = PromptCondition(prompts=["gore", "violence"], use_blacklist=False)
        p.load(original)
        result = p.get_condition()
        assert result.prompts == ["gore", "violence"]
        assert result.use_blacklist is False

    def test_blacklist_mode(self, qtbot):
        p = _PromptPanel()
        qtbot.addWidget(p)
        original = PromptCondition(use_blacklist=True)
        p.load(original)
        result = p.get_condition()
        assert result.use_blacklist is True

    def test_prompts_disabled_when_blacklist_checked(self, qtbot):
        p = _PromptPanel()
        qtbot.addWidget(p)
        p._blacklist_cb.setChecked(True)
        assert not p._prompts.isEnabled()

    def test_prompts_enabled_when_blacklist_unchecked(self, qtbot):
        p = _PromptPanel()
        qtbot.addWidget(p)
        p._blacklist_cb.setChecked(False)
        assert p._prompts.isEnabled()

    def test_condition_type(self, qtbot):
        p = _PromptPanel()
        qtbot.addWidget(p)
        assert p.get_condition().condition_type == "prompt"


# ---------------------------------------------------------------------------
# _LookaheadPanel
# ---------------------------------------------------------------------------

class TestLookaheadPanel:
    def test_condition_type(self, qtbot):
        p = _LookaheadPanel()
        qtbot.addWidget(p)
        assert p.get_condition().condition_type == "lookahead"

    def test_load_missing_name_is_safe(self, qtbot):
        p = _LookaheadPanel()
        qtbot.addWidget(p)
        # no lookaheads registered → combo shows placeholder
        cond = LookaheadCondition(lookahead_name="nonexistent")
        p.load(cond)
        # Should not crash; get_condition returns whatever is current
        result = p.get_condition()
        assert isinstance(result, LookaheadCondition)

    def test_load_none_is_safe(self, qtbot):
        p = _LookaheadPanel()
        qtbot.addWidget(p)
        p.load(None)
        result = p.get_condition()
        assert isinstance(result, LookaheadCondition)


# ---------------------------------------------------------------------------
# _NodeResultPanel
# ---------------------------------------------------------------------------

class TestNodeResultPanel:
    def test_default_state(self, qtbot):
        p = _NodeResultPanel()
        qtbot.addWidget(p)
        p.load(None)
        c = p.get_condition()
        assert isinstance(c, NodeResultCondition)
        assert c.expected_result is True

    def test_set_prior_nodes_populates_combo(self, qtbot):
        p = _NodeResultPanel()
        qtbot.addWidget(p)
        p.set_prior_nodes(["clip_check", "model_check"])
        assert p._node_combo.count() == 2
        assert p._node_combo.itemText(0) == "clip_check"

    def test_round_trip(self, qtbot):
        p = _NodeResultPanel()
        qtbot.addWidget(p)
        p.set_prior_nodes(["n1", "n2"])
        original = NodeResultCondition(node_name="n2", expected_result=False)
        p.load(original)
        result = p.get_condition()
        assert result.node_name == "n2"
        assert result.expected_result is False

    def test_empty_prior_nodes_shows_placeholder(self, qtbot):
        p = _NodeResultPanel()
        qtbot.addWidget(p)
        p.set_prior_nodes([])
        assert p._node_combo.count() == 1
        assert "no prior" in p._node_combo.itemText(0).lower()

    def test_condition_type(self, qtbot):
        p = _NodeResultPanel()
        qtbot.addWidget(p)
        assert p.get_condition().condition_type == "node_result"


# ---------------------------------------------------------------------------
# _CompositePanel
# ---------------------------------------------------------------------------

class TestCompositePanel:
    def test_default_empty(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        p.load(None)
        c = p.get_condition()
        assert isinstance(c, CompositeCondition)
        assert c.sub_conditions == []
        assert c.operator == "AND"

    def test_add_sub_condition(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        p._add_sub()
        assert len(p._row_data) == 1

    def test_add_two_sub_conditions(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        p._add_sub()
        p._add_sub()
        c = p.get_condition()
        assert len(c.sub_conditions) == 2

    def test_remove_last(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        p._add_sub()
        p._add_sub()
        p._remove_last()
        assert len(p._row_data) == 1

    def test_remove_last_when_empty_is_safe(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        p._remove_last()  # must not raise
        assert len(p._row_data) == 0

    def test_operator_round_trip(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        original = CompositeCondition(
            operator="OR",
            sub_conditions=[
                EmbeddingCondition(positives=["a"], negatives=[], threshold=0.3),
                PromptCondition(use_blacklist=True),
            ],
        )
        p.load(original)
        c = p.get_condition()
        assert c.operator == "OR"
        assert len(c.sub_conditions) == 2

    def test_load_clears_previous_rows(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        p._add_sub()
        p._add_sub()
        p._add_sub()
        # Reload with only 1 sub-condition
        p.load(CompositeCondition(
            operator="NOT",
            sub_conditions=[EmbeddingCondition(positives=["x"], negatives=[], threshold=0.2)],
        ))
        assert len(p._row_data) == 1

    def test_set_prior_nodes_forwarded(self, qtbot):
        p = _CompositePanel()
        qtbot.addWidget(p)
        p._add_sub()
        p.set_prior_nodes(["prev1", "prev2"])
        _, sub_row = p._row_data[0]
        assert sub_row._prior_nodes == ["prev1", "prev2"]


# ---------------------------------------------------------------------------
# _RelatedImagePanel
# ---------------------------------------------------------------------------

class TestRelatedImagePanel:
    def test_load_and_get_roundtrip(self, qtbot):
        from compare.classifier_pipeline import RelatedImageCondition
        panel = _RelatedImagePanel()
        qtbot.addWidget(panel)
        cond = RelatedImageCondition(edit_suffix="_edit", search_directory="/custom", count_threshold=3)
        panel.load(cond)
        result = panel.get_condition()
        assert isinstance(result, RelatedImageCondition)
        assert result.edit_suffix == "_edit"
        assert result.search_directory == "/custom"
        assert result.count_threshold == 3

    def test_defaults_on_fresh_panel(self, qtbot):
        from compare.classifier_pipeline import RelatedImageCondition
        panel = _RelatedImagePanel()
        qtbot.addWidget(panel)
        result = panel.get_condition()
        assert result.edit_suffix == ""
        assert result.search_directory == ""
        assert result.count_threshold == 1

    def test_load_wrong_type_resets_to_defaults(self, qtbot):
        from compare.classifier_pipeline import RelatedImageCondition, EmbeddingCondition
        panel = _RelatedImagePanel()
        qtbot.addWidget(panel)
        panel.load(EmbeddingCondition())
        result = panel.get_condition()
        assert result.edit_suffix == ""
        assert result.count_threshold == 1

    def test_condition_type_attribute(self):
        assert _RelatedImagePanel.condition_type == "related_image"


# ---------------------------------------------------------------------------
# _BaseStemMatchPanel
# ---------------------------------------------------------------------------

class TestBaseStemMatchPanel:
    def test_load_and_get_roundtrip_require_true(self, qtbot):
        from compare.classifier_pipeline import BaseStemMatchCondition
        panel = _BaseStemMatchPanel()
        qtbot.addWidget(panel)
        panel.load(BaseStemMatchCondition(require_match=True))
        result = panel.get_condition()
        assert isinstance(result, BaseStemMatchCondition)
        assert result.require_match is True

    def test_load_and_get_roundtrip_require_false(self, qtbot):
        from compare.classifier_pipeline import BaseStemMatchCondition
        panel = _BaseStemMatchPanel()
        qtbot.addWidget(panel)
        panel.load(BaseStemMatchCondition(require_match=False))
        result = panel.get_condition()
        assert result.require_match is False

    def test_defaults_on_fresh_panel(self, qtbot):
        from compare.classifier_pipeline import BaseStemMatchCondition
        panel = _BaseStemMatchPanel()
        qtbot.addWidget(panel)
        result = panel.get_condition()
        assert result.require_match is True

    def test_load_wrong_type_resets_to_default(self, qtbot):
        from compare.classifier_pipeline import BaseStemMatchCondition, EmbeddingCondition
        panel = _BaseStemMatchPanel()
        qtbot.addWidget(panel)
        panel.load(EmbeddingCondition())
        result = panel.get_condition()
        assert result.require_match is True

    def test_condition_type_attribute(self):
        assert _BaseStemMatchPanel.condition_type == "base_stem_match"


# ---------------------------------------------------------------------------
# _OutcomeEditorWidget
# ---------------------------------------------------------------------------

class TestOutcomeEditorWidget:
    def test_default_outcome_is_continue(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        outcome = w.get_outcome()
        assert outcome.outcome_type == OutcomeType.CONTINUE

    def test_load_continue(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.load(NodeOutcome(OutcomeType.CONTINUE), [])
        assert w.get_outcome().outcome_type == OutcomeType.CONTINUE

    def test_load_accept(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.load(NodeOutcome(OutcomeType.ACCEPT), [])
        assert w.get_outcome().outcome_type == OutcomeType.ACCEPT

    def test_load_reject(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.load(NodeOutcome(OutcomeType.REJECT), [])
        assert w.get_outcome().outcome_type == OutcomeType.REJECT

    def test_load_goto(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.load(
            NodeOutcome(OutcomeType.GOTO, target_node="node_3"),
            ["node_2", "node_3", "node_4"],
        )
        out = w.get_outcome()
        assert out.outcome_type == OutcomeType.GOTO
        assert out.target_node == "node_3"

    def test_load_execute(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.load(
            NodeOutcome(
                OutcomeType.EXECUTE,
                action_type=ClassifierActionType.NOTIFY,
            ),
            [],
        )
        out = w.get_outcome()
        assert out.outcome_type == OutcomeType.EXECUTE
        assert out.action_type == ClassifierActionType.NOTIFY

    def test_execute_with_modifier(self, qtbot):
        w = _OutcomeEditorWidget("On no-match:")
        qtbot.addWidget(w)
        w.load(
            NodeOutcome(
                OutcomeType.EXECUTE,
                action_type=ClassifierActionType.MOVE,
                action_modifier="/target/dir",
            ),
            [],
        )
        out = w.get_outcome()
        assert out.action_type == ClassifierActionType.MOVE
        assert out.action_modifier == "/target/dir"

    def test_set_later_nodes_updates_goto_combo(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.set_later_nodes(["a", "b", "c"])
        assert w._goto_combo.count() == 3

    def test_set_later_nodes_empty_shows_placeholder(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.set_later_nodes([])
        assert "no later" in w._goto_combo.itemText(0).lower()

    def test_goto_dependent_widget_visible_only_for_goto(self, qtbot):
        w = _OutcomeEditorWidget("On match:")
        qtbot.addWidget(w)
        w.load(NodeOutcome(OutcomeType.CONTINUE), [])
        assert w._goto_combo.isHidden()
        w.load(NodeOutcome(OutcomeType.GOTO, target_node="n"), ["n"])
        assert not w._goto_combo.isHidden()


# ---------------------------------------------------------------------------
# ClassifierPipelineEditorDialog — construction
# ---------------------------------------------------------------------------

class TestEditorDialogConstruction:
    def test_new_pipeline_dialog_constructs(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        assert dlg is not None
        assert dlg._pipeline is not None
        assert dlg._is_edit is False

    def test_edit_pipeline_dialog_constructs(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline("existing")
        dlg = _open_dialog(qtbot, pipeline)
        assert dlg._is_edit is True
        assert dlg._original_name == "existing"

    def test_pipeline_name_pre_populated(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline("my_pipe")
        dlg = _open_dialog(qtbot, pipeline)
        assert dlg._name_edit.text() == "my_pipe"

    def test_active_checkbox_reflects_pipeline(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        pipeline.is_active = False
        dlg = _open_dialog(qtbot, pipeline)
        assert not dlg._active_cb.isChecked()

    def test_prevalidation_type_sets_combo(self, qtbot, isolated_singletons):
        pv = PrevalidationPipeline(profile_name="portraits")
        pv.name = "pv_test"
        dlg = _open_dialog(qtbot, pv)
        assert dlg._type_combo.currentIndex() == 1

    def test_general_type_is_default(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        assert dlg._type_combo.currentIndex() == 0

    def test_profile_picker_hidden_for_general(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._type_combo.setCurrentIndex(0)
        assert dlg._profile_combo.isHidden()

    def test_profile_picker_visible_for_prevalidation(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._type_combo.setCurrentIndex(1)
        assert not dlg._profile_combo.isHidden()

    def test_node_list_populated_on_open(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        assert dlg._node_list.count() == 2

    def test_first_node_selected_on_open(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        assert dlg._node_list.currentRow() == 0


# ---------------------------------------------------------------------------
# ClassifierPipelineEditorDialog — node operations
# ---------------------------------------------------------------------------

class TestEditorNodeOperations:
    def test_add_node_increases_count(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        initial = len(dlg._pipeline.nodes)
        dlg._add_node()
        assert len(dlg._pipeline.nodes) == initial + 1

    def test_add_node_selects_new_node(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        dlg._add_node()
        assert dlg._node_list.currentRow() == len(dlg._pipeline.nodes) - 1

    def test_remove_node_decreases_count(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        initial = len(dlg._pipeline.nodes)
        dlg._node_list.setCurrentRow(0)
        dlg._remove_node()
        assert len(dlg._pipeline.nodes) == initial - 1

    def test_remove_only_node_disables_editor(self, qtbot, isolated_singletons):
        pipeline = ClassifierPipeline(name="single")
        pipeline.nodes = [_make_node("only")]
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        dlg._remove_node()
        assert not dlg._node_editor_widget.isEnabled()

    def test_move_node_up_swaps_order(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        pipeline.nodes[0].name = "first"
        pipeline.nodes[1].name = "second"
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(1)
        dlg._move_node_up()
        assert dlg._pipeline.nodes[0].name == "second"
        assert dlg._pipeline.nodes[1].name == "first"

    def test_move_node_down_swaps_order(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        pipeline.nodes[0].name = "first"
        pipeline.nodes[1].name = "second"
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        dlg._move_node_down()
        assert dlg._pipeline.nodes[0].name == "second"
        assert dlg._pipeline.nodes[1].name == "first"

    def test_move_up_from_top_is_noop(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        original_order = [n.name for n in dlg._pipeline.nodes]
        dlg._node_list.setCurrentRow(0)
        dlg._move_node_up()
        assert [n.name for n in dlg._pipeline.nodes] == original_order

    def test_move_down_from_bottom_is_noop(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        original_order = [n.name for n in dlg._pipeline.nodes]
        dlg._node_list.setCurrentRow(len(pipeline.nodes) - 1)
        dlg._move_node_down()
        assert [n.name for n in dlg._pipeline.nodes] == original_order

    def test_node_name_edit_updates_model(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        dlg._node_name_edit.setText("renamed_node")
        assert dlg._pipeline.nodes[0].name == "renamed_node"

    def test_node_name_edit_updates_list_label(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        dlg._node_name_edit.setText("shiny_new")
        assert "shiny_new" in dlg._node_list.item(0).text()


# ---------------------------------------------------------------------------
# ClassifierPipelineEditorDialog — flush and condition loading
# ---------------------------------------------------------------------------

class TestEditorFlush:
    def test_flush_writes_name_to_node(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        dlg._node_name_edit.setText("flushed_name")
        dlg._flush_node_to_model()
        assert dlg._pipeline.nodes[0].name == "flushed_name"

    def test_flush_when_no_node_selected_is_safe(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._current_node_idx = None
        dlg._flush_node_to_model()  # must not raise

    def test_condition_type_change_switches_stack(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        # Switch to "Classifier Rank" (index 1)
        dlg._condition_type_combo.setCurrentIndex(1)
        assert dlg._condition_stack.currentIndex() == 1

    def test_condition_type_change_resets_panel(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        # Switch to Prototype (index 2)
        dlg._condition_type_combo.setCurrentIndex(2)
        cond = dlg._classifier_rank_panel.get_condition()
        # After reset, categories should be empty
        assert cond.categories == []

    def test_load_node_populates_outcome_editors(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        pipeline.nodes[0].on_match = NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.NOTIFY)
        pipeline.nodes[0].on_no_match = NodeOutcome(OutcomeType.REJECT)
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        assert dlg._on_match_editor._type_combo.currentText() == OutcomeType.EXECUTE.value
        assert dlg._on_no_match_editor._type_combo.currentText() == OutcomeType.REJECT.value


# ---------------------------------------------------------------------------
# ClassifierPipelineEditorDialog — save logic
# ---------------------------------------------------------------------------

class TestEditorSave:
    @pytest.fixture(autouse=True)
    def _patch_alert(self, monkeypatch):
        """Prevent qt_alert → QMessageBox.exec() from blocking the test runner."""
        monkeypatch.setattr(
            "ui.compare.classifier_pipeline_editor_qt.qt_alert",
            lambda *a, **kw: None,
        )

    def test_save_new_pipeline_adds_to_manager(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("brand_new")
        dlg._save()
        assert ClassifierPipelines.get_pipeline_by_name("brand_new") is not None

    def test_save_stores_to_cache(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("cached_pipe")
        dlg._save()
        # Reload and check
        saved = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert "cached_pipe" in saved

    def test_save_empty_name_rejected(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("")
        before = len(ClassifierPipelines.get_all_pipelines())
        dlg._save()
        assert len(ClassifierPipelines.get_all_pipelines()) == before

    def test_save_duplicate_name_rejected_for_new(self, qtbot, isolated_singletons):
        existing = _make_pipeline("dupe")
        ClassifierPipelines.add_pipeline(existing)
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("dupe")
        before = len(ClassifierPipelines.get_all_pipelines())
        dlg._save()
        assert len(ClassifierPipelines.get_all_pipelines()) == before

    def test_save_edit_allows_same_name(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline("keep_name")
        ClassifierPipelines.add_pipeline(pipeline)
        dlg = _open_dialog(qtbot, pipeline)
        # Name unchanged — should save without conflict error
        dlg._name_edit.setText("keep_name")
        dlg._save()
        assert ClassifierPipelines.get_pipeline_by_name("keep_name") is not None

    def test_save_edit_updates_pipeline_in_place(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline("edit_me")
        ClassifierPipelines.add_pipeline(pipeline)
        dlg = _open_dialog(qtbot, pipeline)
        dlg._name_edit.setText("edit_me")
        dlg._desc_edit.setPlainText("Updated description")
        dlg._save()
        result = ClassifierPipelines.get_pipeline_by_name("edit_me")
        assert result is not None
        assert result.description == "Updated description"

    def test_save_edit_rename_succeeds(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline("old_name")
        ClassifierPipelines.add_pipeline(pipeline)
        dlg = _open_dialog(qtbot, pipeline)
        dlg._name_edit.setText("new_name")
        dlg._save()
        assert ClassifierPipelines.get_pipeline_by_name("old_name") is None
        assert ClassifierPipelines.get_pipeline_by_name("new_name") is not None

    def test_save_creates_prevalidation_pipeline(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("pv_save_test")
        dlg._type_combo.setCurrentIndex(1)  # Prevalidation
        dlg._save()
        result = ClassifierPipelines.get_pipeline_by_name("pv_save_test")
        assert isinstance(result, PrevalidationPipeline)

    def test_save_general_type_is_base_class(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("gen_save_test")
        dlg._type_combo.setCurrentIndex(0)  # General
        dlg._save()
        result = ClassifierPipelines.get_pipeline_by_name("gen_save_test")
        assert type(result) is ClassifierPipeline

    def test_save_active_flag_persisted(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("active_test")
        dlg._active_cb.setChecked(False)
        dlg._save()
        result = ClassifierPipelines.get_pipeline_by_name("active_test")
        assert result.is_active is False

    def test_save_default_action_persisted(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("action_test")
        dlg._default_action_combo.setCurrentText(ClassifierActionType.NOTIFY.value)
        dlg._save()
        result = ClassifierPipelines.get_pipeline_by_name("action_test")
        assert result.default_action == ClassifierActionType.NOTIFY

    def test_save_none_default_action(self, qtbot, isolated_singletons):
        from utils.translations import _
        dlg = _open_dialog(qtbot)
        dlg._name_edit.setText("no_action_test")
        dlg._default_action_combo.setCurrentText(_("(none)"))
        dlg._save()
        result = ClassifierPipelines.get_pipeline_by_name("no_action_test")
        assert result.default_action is None


# ---------------------------------------------------------------------------
# ClassifierPipelineEditorDialog — flow preview
# ---------------------------------------------------------------------------

def _scene_text(dlg) -> str:
    """Collect all text from QGraphicsTextItems in the flow preview scene."""
    from PySide6.QtWidgets import QGraphicsTextItem
    return "\n".join(
        item.toPlainText()
        for item in dlg._flow_scene.items()
        if isinstance(item, QGraphicsTextItem)
    )


class TestEditorFlowPreview:
    def test_flow_preview_not_empty_for_pipeline_with_nodes(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        assert len(_scene_text(dlg)) > 0

    def test_flow_preview_updates_on_name_change(self, qtbot, isolated_singletons):
        pipeline = _make_pipeline()
        dlg = _open_dialog(qtbot, pipeline)
        dlg._node_list.setCurrentRow(0)
        dlg._node_name_edit.setText("unique_changed_name_xyz")
        assert "unique_changed_name_xyz" in _scene_text(dlg)

    def test_flow_preview_empty_pipeline_shows_something(self, qtbot, isolated_singletons):
        dlg = _open_dialog(qtbot)
        # Empty pipeline shows a placeholder text item in the scene
        assert len(dlg._flow_scene.items()) > 0

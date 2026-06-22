"""
Unit tests for compare.classifier_pipeline_runner (Phase 2 — execution engine).

All external ML dependencies (CLIP, classifiers, prototypes, prompt extractor)
are mocked so no real models are loaded.  File-system operations in action
dispatch are also mocked.
"""

import pytest
import numpy as np

from unittest.mock import patch

from compare.action_callbacks import ActionCallbacks
from files.related_image import clear_base_stem_dir_cache
from utils.config import config

from compare.classifier_pipeline import (
    BaseStemMatchCondition,
    ClassifierPipeline,
    ClassifierRankCondition,
    CompositeCondition,
    EmbeddingCondition,
    FilenameContainsCondition,
    GroupCondition,
    GroupChildResultCondition,
    LookaheadCondition,
    MediaTypeCondition,
    NodeOutcome,
    NodeResultCondition,
    OutcomeType,
    PipelineNode,
    PromptCondition,
    PrototypeCondition,
    RelatedImageCondition,
    UnknownSuffixCondition,
)
from compare.pipeline_run_report import PipelineRunReport, PipelineRunStats
from compare.classifier_pipeline_runner import (
    _evaluate_condition,
    _eval_base_stem_match,
    _eval_unknown_suffix,
    _eval_classifier_rank,
    _eval_composite,
    _eval_filename_contains,
    _eval_group,
    _eval_lookahead,
    _eval_media_type,
    _eval_prompt,
    _eval_prototype,
    _eval_related_image,
    _resolve_stem_group,
    _seed_exists_in_dirs,
    run_pipeline,
)
from utils.constants import ClassifierActionType, CompareMediaType


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _node(name, condition, on_match=None, on_no_match=None):
    return PipelineNode(
        name=name,
        condition=condition,
        on_match=on_match or NodeOutcome.continue_(),
        on_no_match=on_no_match or NodeOutcome.accept(),
    )


def _execute(action_type, modifier=""):
    return NodeOutcome(OutcomeType.EXECUTE, action_type=action_type, action_modifier=modifier)


def _execute_and_continue(action_type, modifier=""):
    return NodeOutcome(OutcomeType.EXECUTE_AND_CONTINUE, action_type=action_type, action_modifier=modifier)


def _pipeline(*nodes, name="test", default_action=None):
    return ClassifierPipeline(name=name, nodes=list(nodes), default_action=default_action)


def _callbacks():
    """Return mock callback objects that record calls."""
    calls = {"hide": [], "notify": [], "mark": [], "blur": []}
    return (
        calls,
        lambda p: calls["hide"].append(p),
        lambda *a, **kw: calls["notify"].append((a, kw)),
        lambda p: calls["mark"].append(p),
        lambda p: calls["blur"].append(p),
    )


IMAGE = "/fake/image.jpg"


# ---------------------------------------------------------------------------
# EmbeddingCondition
# ---------------------------------------------------------------------------

class TestEmbeddingCondition:
    def test_match(self, monkeypatch):
        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare",
                            staticmethod(lambda *a, **kw: True))
        result, score = _evaluate_condition(
            EmbeddingCondition(["x"], [], 0.3), IMAGE, {}, {}
        )
        assert result is True
        assert score is None

    def test_no_match(self, monkeypatch):
        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare",
                            staticmethod(lambda *a, **kw: False))
        result, _ = _evaluate_condition(
            EmbeddingCondition(["x"], [], 0.3), IMAGE, {}, {}
        )
        assert result is False


# ---------------------------------------------------------------------------
# ClassifierRankCondition
# ---------------------------------------------------------------------------

class TestClassifierRankCondition:
    def _mock_classifier(self, monkeypatch, predictions: dict):
        """predictions = {category: score}"""
        import image.image_classifier_manager as mgr_mod

        class FakeClassifier:
            def predict_image(self, path):
                return predictions

            def predict_image_ranked(self, path):
                return sorted(predictions.items(), key=lambda kv: kv[1], reverse=True)

        class FakeManager:
            def get_classifier(self, name):
                return FakeClassifier()

        monkeypatch.setattr(mgr_mod, "image_classifier_manager", FakeManager())

    def test_top1_match(self, monkeypatch):
        self._mock_classifier(monkeypatch, {"safe": 0.1, "explicit": 0.9})
        result, score = _eval_classifier_rank(
            ClassifierRankCondition("m", ["explicit"], min_rank=1, max_rank=1),
            IMAGE,
        )
        assert result is True
        assert score == pytest.approx(0.9)

    def test_top1_no_match(self, monkeypatch):
        self._mock_classifier(monkeypatch, {"safe": 0.9, "explicit": 0.1})
        result, _ = _eval_classifier_rank(
            ClassifierRankCondition("m", ["explicit"], min_rank=1, max_rank=1),
            IMAGE,
        )
        assert result is False

    def test_second_rank_match(self, monkeypatch):
        self._mock_classifier(monkeypatch, {"safe": 0.6, "suggestive": 0.3, "explicit": 0.1})
        result, score = _eval_classifier_rank(
            ClassifierRankCondition("m", ["suggestive"], min_rank=2, max_rank=2),
            IMAGE,
        )
        assert result is True
        assert score == pytest.approx(0.3)

    def test_rank_range_match(self, monkeypatch):
        self._mock_classifier(monkeypatch, {"safe": 0.5, "suggestive": 0.3, "explicit": 0.2})
        result, _ = _eval_classifier_rank(
            ClassifierRankCondition("m", ["explicit"], min_rank=2, max_rank=3),
            IMAGE,
        )
        assert result is True

    def test_min_confidence_not_met(self, monkeypatch):
        self._mock_classifier(monkeypatch, {"safe": 0.8, "explicit": 0.05})
        result, _ = _eval_classifier_rank(
            ClassifierRankCondition("m", ["explicit"], min_rank=2, max_rank=2,
                                   min_confidence=0.1),
            IMAGE,
        )
        assert result is False

    def test_missing_classifier(self, monkeypatch):
        import image.image_classifier_manager as mgr_mod

        class FakeManager:
            def get_classifier(self, name):
                return None

        monkeypatch.setattr(mgr_mod, "image_classifier_manager", FakeManager())
        result, _ = _eval_classifier_rank(
            ClassifierRankCondition("missing_model", ["x"], 1, 1),
            IMAGE,
        )
        assert result is False


# ---------------------------------------------------------------------------
# PrototypeCondition
# ---------------------------------------------------------------------------

class TestPrototypeCondition:
    def _mock_prototype(self, monkeypatch, pos_sim=0.8, neg_sim=0.2):
        import compare.embedding_prototype as ep_mod

        fake_proto = np.ones(512)

        def fake_calculate(directory, **kw):
            return fake_proto

        def fake_compare(image_path, proto, session_cache_key=None, negative_prototype=0):
            return neg_sim if negative_prototype == 1 else pos_sim

        monkeypatch.setattr(ep_mod.EmbeddingPrototype, "calculate_prototype_from_directory",
                            staticmethod(fake_calculate))
        monkeypatch.setattr(ep_mod.EmbeddingPrototype, "compare_with_prototype",
                            staticmethod(fake_compare))

        # Also clear the module-level prototype cache so each test starts fresh
        import compare.classifier_pipeline_runner as runner
        runner._pos_prototype_cache.clear()
        runner._neg_prototype_cache.clear()

    def test_above_threshold(self, monkeypatch):
        self._mock_prototype(monkeypatch, pos_sim=0.9)
        result, score = _eval_prototype(
            PrototypeCondition("/pos", threshold=0.5), IMAGE
        )
        assert result is True
        assert score == pytest.approx(0.9)

    def test_below_threshold(self, monkeypatch):
        self._mock_prototype(monkeypatch, pos_sim=0.2)
        result, _ = _eval_prototype(
            PrototypeCondition("/pos", threshold=0.5), IMAGE
        )
        assert result is False

    def test_negative_prototype_reduces_score(self, monkeypatch):
        # pos_sim=0.8, neg_sim=0.5, lambda=0.5 → final=0.8-0.25=0.55
        self._mock_prototype(monkeypatch, pos_sim=0.8, neg_sim=0.5)
        result, score = _eval_prototype(
            PrototypeCondition("/pos", "/neg", threshold=0.6, negative_lambda=0.5),
            IMAGE,
        )
        assert score == pytest.approx(0.55)
        assert result is False  # 0.55 < 0.6

    def test_empty_directory_returns_false(self, monkeypatch):
        import compare.classifier_pipeline_runner as runner
        runner._pos_prototype_cache.clear()
        result, _ = _eval_prototype(PrototypeCondition(""), IMAGE)
        assert result is False

    def test_prototype_cached_on_second_call(self, monkeypatch):
        import compare.classifier_pipeline_runner as runner
        runner._pos_prototype_cache.clear()
        call_count = [0]

        import compare.embedding_prototype as ep_mod

        def fake_calculate(directory, **kw):
            call_count[0] += 1
            return np.ones(512)

        monkeypatch.setattr(ep_mod.EmbeddingPrototype, "calculate_prototype_from_directory",
                            staticmethod(fake_calculate))
        monkeypatch.setattr(ep_mod.EmbeddingPrototype, "compare_with_prototype",
                            staticmethod(lambda *a, **kw: 0.9))

        _eval_prototype(PrototypeCondition("/pos", threshold=0.5), IMAGE)
        _eval_prototype(PrototypeCondition("/pos", threshold=0.5), IMAGE)
        assert call_count[0] == 1  # second call used cache


# ---------------------------------------------------------------------------
# PromptCondition
# ---------------------------------------------------------------------------

class TestPromptCondition:
    def _mock_extractor(self, monkeypatch, positive="a photo of a dog", negative=""):
        import image.image_data_extractor as ext_mod
        monkeypatch.setattr(
            ext_mod.image_data_extractor,
            "extract_prompts_all_strategies",
            lambda path: (positive, negative),
        )

    def test_prompt_match(self, monkeypatch):
        self._mock_extractor(monkeypatch, positive="sunset beach photo")
        result, _ = _eval_prompt(PromptCondition(prompts=["sunset"]), IMAGE)
        assert result is True

    def test_prompt_no_match(self, monkeypatch):
        self._mock_extractor(monkeypatch, positive="mountain landscape")
        result, _ = _eval_prompt(PromptCondition(prompts=["sunset"]), IMAGE)
        assert result is False

    def test_no_prompts_list_returns_false(self, monkeypatch):
        self._mock_extractor(monkeypatch, positive="anything")
        result, _ = _eval_prompt(PromptCondition(prompts=[]), IMAGE)
        assert result is False

    def test_none_positive_prompt_returns_false(self, monkeypatch):
        import image.image_data_extractor as ext_mod
        monkeypatch.setattr(
            ext_mod.image_data_extractor,
            "extract_prompts_all_strategies",
            lambda path: (None, None),
        )
        result, _ = _eval_prompt(PromptCondition(prompts=["x"]), IMAGE)
        assert result is False

    def test_case_insensitive(self, monkeypatch):
        self._mock_extractor(monkeypatch, positive="Beautiful Sunset")
        result, _ = _eval_prompt(PromptCondition(prompts=["sunset"]), IMAGE)
        assert result is True


# ---------------------------------------------------------------------------
# FilenameContainsCondition
# ---------------------------------------------------------------------------

class TestFilenameContainsCondition:
    def test_match_case_insensitive(self):
        cond = FilenameContainsCondition(["draft"], case_sensitive=False)
        result, score = _eval_filename_contains(cond, "/some/path/My_Draft_v2.jpg")
        assert result is True
        assert score == "draft"

    def test_no_match(self):
        cond = FilenameContainsCondition(["draft"], case_sensitive=False)
        result, _ = _eval_filename_contains(cond, "/some/path/final_image.jpg")
        assert result is False

    def test_case_sensitive_match(self):
        cond = FilenameContainsCondition(["Draft"], case_sensitive=True)
        result, score = _eval_filename_contains(cond, "/path/My_Draft_v2.jpg")
        assert result is True
        assert score == "Draft"

    def test_case_sensitive_no_match_due_to_case(self):
        cond = FilenameContainsCondition(["draft"], case_sensitive=True)
        result, _ = _eval_filename_contains(cond, "/path/My_Draft_v2.jpg")
        assert result is False

    def test_empty_patterns_returns_false(self):
        cond = FilenameContainsCondition([], case_sensitive=False)
        result, _ = _eval_filename_contains(cond, "/path/anything.jpg")
        assert result is False

    def test_multiple_patterns_first_match_returned(self):
        cond = FilenameContainsCondition(["alpha", "beta", "gamma"], case_sensitive=False)
        result, score = _eval_filename_contains(cond, "/path/beta_image.jpg")
        assert result is True
        assert score == "beta"

    def test_only_filename_checked_not_directory(self):
        # "draft" appears in the directory path but not in the filename itself
        cond = FilenameContainsCondition(["draft"], case_sensitive=False)
        result, _ = _eval_filename_contains(cond, "/draft/images/final.jpg")
        assert result is False

    def test_via_evaluate_condition_dispatch(self):
        cond = FilenameContainsCondition(["_hq"], case_sensitive=False)
        result, score = _evaluate_condition(cond, "/photos/sunset_hq.jpg", {}, {})
        assert result is True
        assert score == "_hq"

    def test_via_run_pipeline_executes_action(self):
        cond = FilenameContainsCondition(["_reject"], case_sensitive=False)
        p = _pipeline(
            _node("n1", cond, on_match=_execute(ClassifierActionType.HIDE),
                  on_no_match=NodeOutcome.accept())
        )
        calls, hide, notify, mark, blur = _callbacks()
        result = run_pipeline(p, "/media/photo_reject_001.jpg",
                              ActionCallbacks(hide_callback=hide, notify_callback=notify))
        assert result == ClassifierActionType.HIDE
        assert "/media/photo_reject_001.jpg" in calls["hide"]

    def test_via_run_pipeline_no_match_accepts(self):
        cond = FilenameContainsCondition(["_reject"], case_sensitive=False)
        p = _pipeline(
            _node("n1", cond, on_match=_execute(ClassifierActionType.HIDE),
                  on_no_match=NodeOutcome.accept())
        )
        result = run_pipeline(p, "/media/photo_keep_001.jpg", ActionCallbacks())
        assert result is None

    def test_serialization_round_trip(self):
        original = FilenameContainsCondition(["_wip", "_draft"], case_sensitive=True)
        from compare.classifier_pipeline import _condition_from_dict
        restored = _condition_from_dict(original.to_dict())
        assert isinstance(restored, FilenameContainsCondition)
        assert restored.patterns == ["_wip", "_draft"]
        assert restored.case_sensitive is True

    def test_summary_contains_patterns(self):
        cond = FilenameContainsCondition(["foo", "bar"])
        assert "foo" in cond.summary()
        assert "bar" in cond.summary()


# ---------------------------------------------------------------------------
# LookaheadCondition
# ---------------------------------------------------------------------------

class TestLookaheadCondition:
    def _setup(self, monkeypatch, clip_result: bool, is_prevalidation=False,
               prevalidation_positives=None):
        from compare.lookahead import Lookahead

        lk = Lookahead(
            name="test_lk",
            name_or_text="nsfw" if not is_prevalidation else "my_prevalidation",
            threshold=0.3,
            is_prevalidation_name=is_prevalidation,
        )
        monkeypatch.setattr(Lookahead, "lookaheads", [lk])

        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare",
                            staticmethod(lambda *a, **kw: clip_result))

        if is_prevalidation and prevalidation_positives is not None:
            import compare.classifier_actions_manager as cam_mod

            class FakePrevalidation:
                positives = prevalidation_positives
                negatives = []

            monkeypatch.setattr(
                cam_mod.ClassifierActionsManager,
                "get_prevalidation_by_name",
                staticmethod(lambda name: FakePrevalidation()),
            )

    def test_text_lookahead_match(self, monkeypatch):
        self._setup(monkeypatch, clip_result=True)
        result, _ = _eval_lookahead(LookaheadCondition("test_lk"), IMAGE)
        assert result is True

    def test_text_lookahead_no_match(self, monkeypatch):
        self._setup(monkeypatch, clip_result=False)
        result, _ = _eval_lookahead(LookaheadCondition("test_lk"), IMAGE)
        assert result is False

    def test_prevalidation_lookahead(self, monkeypatch):
        self._setup(monkeypatch, clip_result=True, is_prevalidation=True,
                    prevalidation_positives=["nsfw"])
        result, _ = _eval_lookahead(LookaheadCondition("test_lk"), IMAGE)
        assert result is True

    def test_missing_lookahead_returns_false(self, monkeypatch):
        from compare.lookahead import Lookahead
        monkeypatch.setattr(Lookahead, "lookaheads", [])
        result, _ = _eval_lookahead(LookaheadCondition("nonexistent"), IMAGE)
        assert result is False


# ---------------------------------------------------------------------------
# NodeResultCondition
# ---------------------------------------------------------------------------

class TestNodeResultCondition:
    def test_expects_true_and_got_true(self):
        result, score = _evaluate_condition(
            NodeResultCondition("prev", True), IMAGE,
            {"prev": True}, {},
        )
        assert result is True
        assert score == pytest.approx(1.0)

    def test_expects_false_and_got_false(self):
        result, _ = _evaluate_condition(
            NodeResultCondition("prev", False), IMAGE,
            {"prev": False}, {},
        )
        assert result is True

    def test_mismatch(self):
        result, _ = _evaluate_condition(
            NodeResultCondition("prev", True), IMAGE,
            {"prev": False}, {},
        )
        assert result is False

    def test_missing_prior_node_returns_false(self):
        result, _ = _evaluate_condition(
            NodeResultCondition("ghost", True), IMAGE, {}, {}
        )
        assert result is False


# ---------------------------------------------------------------------------
# CompositeCondition
# ---------------------------------------------------------------------------

class TestCompositeCondition:
    def _stub(self, value: bool):
        """A NodeResultCondition that always resolves to `value` from pre-seeded results."""
        name = "t" if value else "f"
        return NodeResultCondition(name, True)

    def _results(self):
        return {"t": True, "f": False}

    def test_and_all_true(self):
        c = CompositeCondition("AND", [self._stub(True), self._stub(True)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is True

    def test_and_one_false(self):
        c = CompositeCondition("AND", [self._stub(True), self._stub(False)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is False

    def test_or_one_true(self):
        c = CompositeCondition("OR", [self._stub(False), self._stub(True)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is True

    def test_or_all_false(self):
        c = CompositeCondition("OR", [self._stub(False), self._stub(False)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is False

    def test_not_true_gives_false(self):
        c = CompositeCondition("NOT", [self._stub(True)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is False

    def test_not_false_gives_true(self):
        c = CompositeCondition("NOT", [self._stub(False)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is True

    def test_xor_different(self):
        c = CompositeCondition("XOR", [self._stub(True), self._stub(False)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is True

    def test_xor_same(self):
        c = CompositeCondition("XOR", [self._stub(True), self._stub(True)])
        result, _ = _eval_composite(c, IMAGE, self._results(), {})
        assert result is False


# ---------------------------------------------------------------------------
# run_pipeline — control flow
# ---------------------------------------------------------------------------

class TestRunPipelineControlFlow:
    def _patch_embedding(self, monkeypatch, result: bool):
        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare",
                            staticmethod(lambda *a, **kw: result))

    def test_inactive_pipeline_returns_none(self, monkeypatch):
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])))
        p.is_active = False
        assert run_pipeline(p, IMAGE, ActionCallbacks()) is None

    def test_empty_pipeline_returns_none(self):
        p = ClassifierPipeline(name="empty")
        assert run_pipeline(p, IMAGE, ActionCallbacks()) is None

    def test_execute_on_match(self, monkeypatch):
        self._patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=_execute(ClassifierActionType.NOTIFY))
        )
        calls, hide, notify, mark, blur = _callbacks()
        result = run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=notify))
        assert result == ClassifierActionType.NOTIFY
        assert len(calls["notify"]) == 1

    def test_accept_on_no_match(self, monkeypatch):
        self._patch_embedding(monkeypatch, False)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_no_match=NodeOutcome(OutcomeType.ACCEPT))
        )
        assert run_pipeline(p, IMAGE, ActionCallbacks()) is None

    def test_continue_advances_to_next_node(self, monkeypatch):
        # n1 matches → CONTINUE; n2 matches → EXECUTE NOTIFY
        self._patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=NodeOutcome.continue_(),
                  on_no_match=NodeOutcome.accept()),
            _node("n2", EmbeddingCondition(["y"]),
                  on_match=_execute(ClassifierActionType.NOTIFY),
                  on_no_match=NodeOutcome.accept()),
        )
        result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result == ClassifierActionType.NOTIFY

    def test_goto_skips_node(self, monkeypatch):
        # n1 matches → GOTO n3; n2 should be skipped; n3 → EXECUTE
        results_map = {"n1_cond": True, "n3_cond": True}
        call_count = {"n2": 0}

        import compare.compare_embeddings_clip as clip_mod

        def side_effect(image_path, positives, negatives, threshold):
            return results_map.get(positives[0] if positives else "", False)

        monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare",
                            staticmethod(side_effect))

        p = _pipeline(
            _node("n1", EmbeddingCondition(["n1_cond"]),
                  on_match=NodeOutcome(OutcomeType.GOTO, target_node="n3"),
                  on_no_match=NodeOutcome.accept()),
            _node("n2", EmbeddingCondition(["n2_cond"]),
                  on_match=_execute(ClassifierActionType.HIDE),
                  on_no_match=NodeOutcome.accept()),
            _node("n3", EmbeddingCondition(["n3_cond"]),
                  on_match=_execute(ClassifierActionType.NOTIFY),
                  on_no_match=NodeOutcome.accept()),
        )
        result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result == ClassifierActionType.NOTIFY

    def test_reject_fires_default_reject_action(self, monkeypatch):
        self._patch_embedding(monkeypatch, True)
        calls, hide, notify, mark, blur = _callbacks()
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=NodeOutcome(OutcomeType.REJECT))
        )
        p.default_reject_action = ClassifierActionType.NOTIFY
        result = run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=notify))
        assert result == ClassifierActionType.NOTIFY
        assert len(calls["notify"]) == 1

    def test_default_action_when_exhausted(self, monkeypatch):
        self._patch_embedding(monkeypatch, False)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_no_match=NodeOutcome.continue_()),
            default_action=ClassifierActionType.NOTIFY,
        )
        calls, _, notify, _, _ = _callbacks()
        result = run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=notify))
        assert result == ClassifierActionType.NOTIFY

    def test_node_exception_treated_as_no_match(self, monkeypatch):
        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(
            clip_mod.CompareEmbeddingClip, "multi_text_compare",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=_execute(ClassifierActionType.NOTIFY),
                  on_no_match=NodeOutcome.accept()),
        )
        assert run_pipeline(p, IMAGE, ActionCallbacks()) is None

    def test_node_results_available_to_later_nodes(self, monkeypatch):
        """The CLIP-vs-model disagreement pattern from the design doc."""
        # n1 (embedding): match → CONTINUE; n2 (node_result n1=True): match → EXECUTE
        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare",
                            staticmethod(lambda *a, **kw: True))

        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=NodeOutcome.continue_(),
                  on_no_match=NodeOutcome.accept()),
            _node("n2", NodeResultCondition("n1", True),
                  on_match=_execute(ClassifierActionType.ADD_MARK),
                  on_no_match=NodeOutcome.accept()),
        )
        calls, _, notify, mark, _ = _callbacks()
        result = run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=notify, add_mark_callback=mark))
        assert result == ClassifierActionType.ADD_MARK
        assert IMAGE in calls["mark"]

    def test_execute_and_continue_fires_action_and_advances(self, monkeypatch):
        """EXECUTE_AND_CONTINUE dispatches the action but does not halt — the next node still runs."""
        self._patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=_execute_and_continue(ClassifierActionType.NOTIFY),
                  on_no_match=NodeOutcome.accept()),
            _node("n2", EmbeddingCondition(["y"]),
                  on_match=_execute(ClassifierActionType.ADD_MARK),
                  on_no_match=NodeOutcome.accept()),
        )
        calls, _, notify, mark, _ = _callbacks()
        result = run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=notify, add_mark_callback=mark))
        assert len(calls["notify"]) == 2     # n1 NOTIFY + n2 ADD_MARK both notify
        assert IMAGE in calls["mark"]        # n2 also ran
        assert result == ClassifierActionType.ADD_MARK

    def test_execute_halts_pipeline_before_later_nodes(self, monkeypatch):
        """EXECUTE (non-continuing) halts after the first node — n2 must not run."""
        self._patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=_execute(ClassifierActionType.NOTIFY),
                  on_no_match=NodeOutcome.accept()),
            _node("n2", EmbeddingCondition(["y"]),
                  on_match=_execute(ClassifierActionType.ADD_MARK),
                  on_no_match=NodeOutcome.accept()),
        )
        calls, _, notify, mark, _ = _callbacks()
        result = run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=notify, add_mark_callback=mark))
        assert result == ClassifierActionType.NOTIFY
        assert IMAGE not in calls["mark"]    # n2 never ran

    def test_execute_and_continue_all_categories_fire(self):
        """Category-fill pattern: every generate node with EXECUTE_AND_CONTINUE fires for a seed image."""
        generated = []

        def _cat_node(name, suffix):
            return PipelineNode(
                name=name,
                condition=FilenameContainsCondition(["seed"], case_sensitive=False),
                on_match=_execute_and_continue(ClassifierActionType.GENERATE, suffix),
                on_no_match=NodeOutcome.continue_(),
            )

        p = _pipeline(
            _cat_node("Generate apple",  "_apple"),
            _cat_node("Generate banana", "_banana"),
            _cat_node("Generate cherry", "_cherry"),
        )
        callbacks = ActionCallbacks(
            generate_callback=lambda path, suffix=None: generated.append((path, suffix))
        )
        run_pipeline(p, "/fake/seed_image.jpg", callbacks)

        assert len(generated) == 3
        assert generated[0][1] == "_apple"
        assert generated[1][1] == "_banana"
        assert generated[2][1] == "_cherry"


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

class TestActionDispatch:
    def _patch_embedding(self, monkeypatch, value=True):
        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare",
                            staticmethod(lambda *a, **kw: value))

    def _run(self, monkeypatch, action_type, modifier="", **extra_callbacks):
        self._patch_embedding(monkeypatch)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=_execute(action_type, modifier))
        )
        calls, hide, notify, mark, blur = _callbacks()
        run_pipeline(
            p, IMAGE,
            ActionCallbacks(
                hide_callback=extra_callbacks.get("hide", hide),
                notify_callback=extra_callbacks.get("notify", notify),
                add_mark_callback=extra_callbacks.get("mark", mark),
                blur_callback=extra_callbacks.get("blur", blur),
            ),
        )
        return calls

    def test_notify_action(self, monkeypatch):
        calls = self._run(monkeypatch, ClassifierActionType.NOTIFY)
        assert len(calls["notify"]) == 1

    def test_hide_action(self, monkeypatch):
        calls = self._run(monkeypatch, ClassifierActionType.HIDE)
        assert IMAGE in calls["hide"]
        assert len(calls["notify"]) == 1

    def test_add_mark_action(self, monkeypatch):
        calls = self._run(monkeypatch, ClassifierActionType.ADD_MARK)
        assert IMAGE in calls["mark"]

    def test_blur_action(self, monkeypatch):
        calls = self._run(monkeypatch, ClassifierActionType.BLUR)
        assert IMAGE in calls["blur"]

    def test_skip_action(self, monkeypatch):
        calls = self._run(monkeypatch, ClassifierActionType.SKIP)
        assert len(calls["notify"]) == 1  # skip notifies

    def test_move_action(self, monkeypatch, tmp_path):
        self._patch_embedding(monkeypatch)
        target = str(tmp_path / "dest")

        # Mock FileAction and Utils so no real move happens
        import files.file_action as fa_mod
        import utils.utils as uu_mod
        moved = []
        monkeypatch.setattr(fa_mod.FileAction, "add_file_action",
                            staticmethod(lambda fn, src, tgt, **kw: moved.append((src, tgt))))
        monkeypatch.setattr(uu_mod.Utils, "get_relative_dirpath",
                            staticmethod(lambda p, levels=2: p))

        p = _pipeline(_node("n1", EmbeddingCondition(["x"]),
                            on_match=_execute(ClassifierActionType.MOVE, target)))
        calls, _, notify, _, _ = _callbacks()
        result = run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=notify))
        assert result == ClassifierActionType.MOVE
        assert len(moved) == 1
        assert moved[0][1] == target

    def test_delete_action(self, monkeypatch, tmp_path):
        # Create a real temporary file to delete
        img = tmp_path / "image.jpg"
        img.write_bytes(b"fake")
        self._patch_embedding(monkeypatch)

        import utils.utils as uu_mod
        import threading
        monkeypatch.setattr(uu_mod.Utils, "file_operation_lock", threading.Lock())

        p = _pipeline(_node("n1", EmbeddingCondition(["x"]),
                            on_match=_execute(ClassifierActionType.DELETE)))
        calls, _, notify, _, _ = _callbacks()
        run_pipeline(p, str(img), ActionCallbacks(notify_callback=notify))
        assert not img.exists()

    def test_no_callbacks_does_not_crash(self, monkeypatch):
        self._patch_embedding(monkeypatch)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"]),
                            on_match=_execute(ClassifierActionType.NOTIFY)))
        result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result == ClassifierActionType.NOTIFY


# ---------------------------------------------------------------------------
# MediaTypeCondition — execution
# ---------------------------------------------------------------------------

def _patch_media_type(media_type: CompareMediaType):
    return patch("utils.media_utils.get_media_type_for_path", return_value=media_type)


class TestMediaTypeConditionRunner:
    def test_match_when_type_in_list(self):
        c = MediaTypeCondition([CompareMediaType.IMAGE, CompareMediaType.GIF])
        with _patch_media_type(CompareMediaType.IMAGE):
            result, score = _eval_media_type(c, "/img/photo.jpg")
        assert result is True
        assert score == "image"

    def test_no_match_when_type_not_in_list(self):
        c = MediaTypeCondition([CompareMediaType.IMAGE])
        with _patch_media_type(CompareMediaType.VIDEO):
            result, score = _eval_media_type(c, "/vid/clip.mp4")
        assert result is False
        assert score == "video"

    def test_empty_list_never_matches(self):
        c = MediaTypeCondition([])
        with _patch_media_type(CompareMediaType.IMAGE):
            result, _ = _eval_media_type(c, "/img/photo.jpg")
        assert result is False

    def test_via_evaluate_condition(self):
        c = MediaTypeCondition([CompareMediaType.PDF])
        with _patch_media_type(CompareMediaType.PDF):
            result, score = _evaluate_condition(c, "/doc/file.pdf", {}, {})
        assert result is True
        assert score == "pdf"

    def test_as_composite_sub_condition(self):
        media_cond = MediaTypeCondition([CompareMediaType.VIDEO])
        embedding_cond = EmbeddingCondition(["action"])
        composite = CompositeCondition("AND", [media_cond, embedding_cond])
        with _patch_media_type(CompareMediaType.VIDEO):
            with patch("compare.compare_embeddings_clip.CompareEmbeddingClip.multi_text_compare",
                       return_value=True):
                result, _ = _evaluate_condition(composite, "/vid/clip.mp4", {}, {})
        assert result is True

    def test_and_false_when_media_type_not_in_list(self):
        media_cond = MediaTypeCondition([CompareMediaType.VIDEO])
        embedding_cond = EmbeddingCondition(["action"])
        composite = CompositeCondition("AND", [media_cond, embedding_cond])
        # IMAGE type → media_cond fails → AND must be False regardless of embedding.
        # _eval_composite evaluates all sub-conditions eagerly, so patch both.
        with _patch_media_type(CompareMediaType.IMAGE):
            with patch("utils.media_utils.get_media_type_for_path", return_value=CompareMediaType.IMAGE):
                with patch("compare.compare_embeddings_clip.CompareEmbeddingClip.multi_text_compare",
                           return_value=True):
                    result, _ = _evaluate_condition(composite, "/img/photo.jpg", {}, {})
        assert result is False

    def test_score_is_media_type_string_value(self):
        for mt in [CompareMediaType.IMAGE, CompareMediaType.PDF, CompareMediaType.AUDIO]:
            c = MediaTypeCondition([mt])
            with patch("utils.media_utils.get_media_type_for_path", return_value=mt):
                _, score = _eval_media_type(c, "/f")
            assert score == mt.value


# ---------------------------------------------------------------------------
# BaseStemMatchCondition
# ---------------------------------------------------------------------------

class TestBaseStemMatchConditionRunner:
    # extract_filename_base_stem and find_files_by_base_stem are lazy-imported
    # from files.related_image inside _eval_base_stem_match, so patch the source
    # module. config is also lazy-imported; patch its attribute on the singleton.

    def setup_method(self):
        clear_base_stem_dir_cache()

    def teardown_method(self):
        clear_base_stem_dir_cache()

    def _cond(self, require_match=True):
        return BaseStemMatchCondition(require_match=require_match)

    def test_match_when_file_found_require_true(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_edit.jpg"])
        result, score = _eval_base_stem_match(self._cond(require_match=True), IMAGE)
        assert result is True
        assert score is None

    def test_no_match_when_file_not_found_require_true(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: [])
        result, _ = _eval_base_stem_match(self._cond(require_match=True), IMAGE)
        assert result is False

    def test_inverted_match_when_file_not_found(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: [])
        result, _ = _eval_base_stem_match(self._cond(require_match=False), IMAGE)
        assert result is True

    def test_no_base_stem_returns_false(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: None)
        result, _ = _eval_base_stem_match(self._cond(), IMAGE)
        assert result is False

    def test_empty_dirs_returns_false(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        result, _ = _eval_base_stem_match(self._cond(), IMAGE)
        assert result is False

    def test_dispatched_via_evaluate_condition(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_x.jpg"])
        result, _ = _evaluate_condition(self._cond(), IMAGE, {}, {})
        assert result is True

    def test_use_cache_true_passed_to_find(self, monkeypatch):
        captured_kwargs = []
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])

        def fake_find(dirs, stem, **kw):
            captured_kwargs.append(kw)
            return []

        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem", fake_find)
        _eval_base_stem_match(self._cond(), IMAGE)
        assert captured_kwargs[0].get("use_cache") is True

    def test_suffix_filter_matches_file_with_suffix(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_A.jpg", "/dir/stem_B.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_A"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_suffix_filter_matches_with_intermediate_tokens(self, monkeypatch):
        # The base stem and suffix need not be adjacent; anything may sit between them.
        # e.g. base="{ID}_{ts}", file="{ID}_{ts}_0_a.jpg", suffix="_a" → match.
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_0_A.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_A"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_suffix_filter_excludes_non_matching_file(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_B.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_A"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is False

    def test_suffix_filter_empty_matches_all(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_B.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=[])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_suffix_filter_case_insensitive(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_A.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_a"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_suffix_filter_inverted_no_match_means_pass(self, monkeypatch):
        # require_match=False: passes when suffix is NOT found
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_B.jpg"])
        cond = BaseStemMatchCondition(require_match=False, suffix_filter=["_A"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True  # no _A found → passes when require_match=False

    def test_suffix_filter_multi_alias_matches_any(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_animal.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_a", "_ani", "_animal"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_suffix_filter_multi_alias_no_false_positive(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_B.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_a", "_ani", "_animal"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is False

    def test_suffix_filter_trailing_digit_matches(self, monkeypatch):
        # stem_a2 should match suffix _a (trailing digit is a generation counter).
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_a2.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_a"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_suffix_filter_trailing_digit_long_suffix(self, monkeypatch):
        # stem_animal3 should match suffix _animal.
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_animal3.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_animal"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_suffix_filter_trailing_digit_no_false_positive(self, monkeypatch):
        # stem_b2 must NOT match suffix _a even after digit stripping.
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_b2.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_a"])
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is False

    def test_search_directory_overrides_config_dirs(self, monkeypatch):
        captured_dirs = []

        def fake_find(dirs, stem, **kw):
            captured_dirs.extend(dirs)
            return ["/custom/stem_A.jpg"]

        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/config_dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem", fake_find)
        cond = BaseStemMatchCondition(require_match=True, search_directory="/custom")
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True
        assert captured_dirs == ["/custom"]
        assert "/config_dir" not in captured_dirs

    # -- max_stem_group_size (overflow-detection mode) --

    def test_max_stem_group_exceeded_returns_true(self, monkeypatch):
        """More matches than the limit → overflow → True (wire on_match=REJECT)."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [f"/dir/stem_{i}.jpg" for i in range(6)])
        cond = BaseStemMatchCondition(max_stem_group_size=5)
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_max_stem_group_at_limit_returns_false(self, monkeypatch):
        """Exactly at the limit (not exceeded) → False (within bounds)."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [f"/dir/stem_{i}.jpg" for i in range(5)])
        cond = BaseStemMatchCondition(max_stem_group_size=5)
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is False

    def test_max_stem_group_zero_disables_overflow_check(self, monkeypatch):
        """max_stem_group_size=0 with no pipeline_categories → overflow disabled; normal require_match applies."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [f"/dir/stem_{i}.jpg" for i in range(100)])
        cond = BaseStemMatchCondition(require_match=True, max_stem_group_size=0)
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True  # normal: 100 matches found, require_match=True → True

    def test_pipeline_categories_infers_limit_on_overflow(self, monkeypatch):
        """max_stem_group_size=0 + pipeline_categories → effective limit = len(categories) + 1."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        # 3 categories → effective limit = 4; 5 matches → overflow
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [f"/dir/stem_{i}.jpg" for i in range(5)])
        cond = BaseStemMatchCondition()
        result, _ = _eval_base_stem_match(cond, IMAGE, pipeline_categories=["_a", "_b", "_c"])
        assert result is True

    def test_pipeline_categories_at_limit_returns_false(self, monkeypatch):
        """Exactly at the inferred limit (len(categories) + 1) → not overflowed."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        # 3 categories → effective limit = 4; 4 matches → within limit
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [f"/dir/stem_{i}.jpg" for i in range(4)])
        cond = BaseStemMatchCondition()
        result, _ = _eval_base_stem_match(cond, IMAGE, pipeline_categories=["_a", "_b", "_c"])
        assert result is False

    def test_explicit_max_stem_group_size_takes_precedence_over_categories(self, monkeypatch):
        """Explicit max_stem_group_size overrides pipeline_categories inference."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        # 3 categories would infer limit=4, but explicit limit=10; 5 matches → no overflow
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [f"/dir/stem_{i}.jpg" for i in range(5)])
        cond = BaseStemMatchCondition(max_stem_group_size=10)
        result, _ = _eval_base_stem_match(cond, IMAGE, pipeline_categories=["_a", "_b", "_c"])
        assert result is False

    def test_max_stem_group_warning_in_report(self, monkeypatch):
        """Exceeding the limit emits a WARNING entry in the report."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [f"/dir/stem_{i}.jpg" for i in range(10)])
        cond = BaseStemMatchCondition(max_stem_group_size=5)
        report = PipelineRunReport()
        _eval_base_stem_match(cond, IMAGE, node_name="uniqueness", report=report)
        warnings = [m for m in report.messages() if m.severity == "WARNING"]
        assert len(warnings) == 1
        assert "10" in warnings[0].detail

    def test_search_directory_empty_falls_back_to_config(self, monkeypatch):
        captured_dirs = []

        def fake_find(dirs, stem, **kw):
            captured_dirs.extend(dirs)
            return []

        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/config_dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem", fake_find)
        cond = BaseStemMatchCondition(require_match=True, search_directory="")
        _eval_base_stem_match(cond, IMAGE)
        assert captured_dirs == ["/config_dir"]

    def test_search_directory_set_but_config_empty_still_searches(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/custom/stem_A.jpg"])
        cond = BaseStemMatchCondition(require_match=True, search_directory="/custom")
        result, _ = _eval_base_stem_match(cond, IMAGE)
        assert result is True

    def test_report_notable_when_multiple_matches(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_A.jpg", "/dir/stem_A2.jpg"])
        report = PipelineRunReport()
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_A"])
        _eval_base_stem_match(cond, IMAGE, node_name="Generate A", report=report)
        msgs = report.messages()
        assert len(msgs) == 1
        assert msgs[0].severity == "NOTABLE"
        assert msgs[0].node == "Generate A"
        assert msgs[0].image_path == IMAGE

    def test_no_report_when_single_match(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_A.jpg"])
        report = PipelineRunReport()
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_A"])
        _eval_base_stem_match(cond, IMAGE, node_name="Generate A", report=report)
        assert not report.has_messages()

    def test_no_report_when_report_is_none(self, monkeypatch):
        """No error when report=None (the default) and multiple matches exist."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_A.jpg", "/dir/stem_A2.jpg"])
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_A"])
        result, _ = _eval_base_stem_match(cond, IMAGE)  # must not raise
        assert result is True

    def test_report_threaded_through_run_pipeline(self, monkeypatch):
        """report passed to run_pipeline reaches _eval_base_stem_match."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_A.jpg", "/dir/stem_A2.jpg"])
        report = PipelineRunReport()
        cond = BaseStemMatchCondition(require_match=True, suffix_filter=["_A"])
        p = _pipeline(_node("Gen A", cond, on_match=NodeOutcome.accept(), on_no_match=NodeOutcome.accept()))
        run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        assert report.has_messages()
        assert report.messages()[0].node == "Gen A"


# ---------------------------------------------------------------------------
# PipelineRunReport
# ---------------------------------------------------------------------------

class TestPipelineRunReport:
    def test_add_and_retrieve(self):
        r = PipelineRunReport()
        r.add("INFO", "node1", "/img.jpg", "some detail")
        msgs = r.messages()
        assert len(msgs) == 1
        assert msgs[0].severity == "INFO"
        assert msgs[0].node == "node1"
        assert msgs[0].image_path == "/img.jpg"
        assert msgs[0].detail == "some detail"
        assert msgs[0].data is None

    def test_add_with_data(self):
        r = PipelineRunReport()
        r.add("NOTABLE", "n", "/img.jpg", "d", data={"files": ["a", "b"]})
        assert r.messages()[0].data == {"files": ["a", "b"]}

    def test_has_messages_empty(self):
        assert not PipelineRunReport().has_messages()

    def test_has_messages_after_add(self):
        r = PipelineRunReport()
        r.add("WARNING", "n", "/img.jpg", "d")
        assert r.has_messages()

    def test_clear(self):
        r = PipelineRunReport()
        r.add("INFO", "n", "/img.jpg", "d")
        r.clear()
        assert not r.has_messages()
        assert r.messages() == []

    def test_messages_returns_copy(self):
        r = PipelineRunReport()
        r.add("INFO", "n", "/img.jpg", "d")
        first = r.messages()
        r.add("INFO", "n", "/img.jpg", "d2")
        assert len(first) == 1  # snapshot was not mutated

    def test_format_completion_report_minimal(self):
        r = PipelineRunReport()
        text = r.format_completion_report(
            PipelineRunStats(
                pipeline_name="test pipe",
                profile_name="my profile",
                directories=["/work"],
                files_by_directory={"/work": 3},
                files_evaluated=3,
                action_counts={"GENERATE": 1, "(no action)": 2},
            )
        )
        assert "test pipe" in text
        assert "my profile" in text
        assert "/work" in text
        assert "3" in text
        assert "Generate" in text
        assert "\n" in text

    def test_format_completion_report_includes_message_sections(self):
        r = PipelineRunReport()
        r.add("WARNING", "uniqueness", "/dir/stem.png", "Stem not unique")
        r.add(
            "NOTABLE", "Generate apple", "/dir/seed.png",
            "2 files share base stem",
            data={"matches": ["/dir/seed_apple.png", "/dir/seed_apple2.png"]},
        )
        text = r.format_completion_report(
            PipelineRunStats(pipeline_name="p", files_evaluated=1)
        )
        assert "Warnings" in text
        assert "Notable" in text
        assert "uniqueness" in text
        assert "seed_apple.png" in text
        assert "seed_apple2.png" in text

    def test_format_completion_report_groups_duplicate_messages(self):
        r = PipelineRunReport()
        shared_data = {"matches": ["/dir/stem__cher.png"], "base_stem": "stem"}
        r.add("NOTABLE", "Stem uniqueness check", "/dir/stem__appl.png",
              "3 files share base stem", data=shared_data)
        r.add("NOTABLE", "Stem uniqueness check", "/dir/stem__banan.png",
              "3 files share base stem", data=shared_data)
        text = r.format_completion_report(
            PipelineRunStats(pipeline_name="p", files_evaluated=2)
        )
        assert "stem__appl.png" in text
        assert "stem__banan.png" in text
        assert text.count("3 files share base stem") == 1  # detail appears once
        assert text.count("stem__cher.png") == 1           # match appears once
        assert "2 files" in text                           # grouped header

    def test_format_completion_report_does_not_group_distinct_messages(self):
        r = PipelineRunReport()
        r.add("NOTABLE", "node", "/dir/a.png", "same detail")
        r.add("NOTABLE", "node", "/dir/b.png", "different detail")
        text = r.format_completion_report(
            PipelineRunStats(pipeline_name="p", files_evaluated=2)
        )
        assert text.count("same detail") == 1
        assert text.count("different detail") == 1
        assert "a.png" in text
        assert "b.png" in text

    def test_format_completion_report_omits_empty_sections(self):
        r = PipelineRunReport()
        text = r.format_completion_report(
            PipelineRunStats(pipeline_name="p", files_evaluated=0)
        )
        assert "Warnings" not in text
        assert "Notable" not in text


# ---------------------------------------------------------------------------
# DebouncedGenerateQueue
# ---------------------------------------------------------------------------

class TestDebouncedGenerateQueue:
    def test_submit_and_shutdown_calls_fn(self):
        from compare.debounced_generate_queue import DebouncedGenerateQueue
        called = []
        q = DebouncedGenerateQueue(dispatch_interval=0)
        q.submit(called.append, "a")
        q.shutdown()
        assert called == ["a"]

    def test_multiple_submits_all_dispatched_in_order(self):
        from compare.debounced_generate_queue import DebouncedGenerateQueue
        called = []
        q = DebouncedGenerateQueue(dispatch_interval=0)
        for v in ["x", "y", "z"]:
            q.submit(called.append, v)
        q.shutdown()
        assert called == ["x", "y", "z"]

    def test_shutdown_after_no_submissions(self):
        from compare.debounced_generate_queue import DebouncedGenerateQueue
        q = DebouncedGenerateQueue(dispatch_interval=0)
        q.shutdown()  # must not hang or raise
        assert not q.is_alive()

    def test_no_sleep_when_interval_zero(self):
        """With dispatch_interval=0 the queue drains without sleeping."""
        import time
        from compare.debounced_generate_queue import DebouncedGenerateQueue
        called = []
        q = DebouncedGenerateQueue(dispatch_interval=0)
        for i in range(5):
            q.submit(called.append, i)
        start = time.monotonic()
        q.shutdown()
        elapsed = time.monotonic() - start
        assert called == list(range(5))
        assert elapsed < 1.0  # should complete well under 1 s with 0 interval

    def test_generate_queue_used_in_dispatch_action(self):
        """When generate_queue is provided, generate_callback is submitted rather than called directly."""
        from compare.debounced_generate_queue import DebouncedGenerateQueue
        from compare.classifier_pipeline_runner import _dispatch_action
        called_direct = []
        submitted = []

        class FakeQueue:
            def submit(self, fn, *args, **kwargs):
                submitted.append((fn, args))

        callbacks = ActionCallbacks(generate_callback=called_direct.append)
        _dispatch_action(
            ClassifierActionType.GENERATE, "_suffix", "pipe", IMAGE,
            callbacks, None, generate_queue=FakeQueue(),
        )
        assert called_direct == []
        assert len(submitted) == 1
        assert submitted[0][1][0] == IMAGE  # image_path forwarded

    def test_generate_dispatched_directly_without_queue(self):
        from compare.classifier_pipeline_runner import _dispatch_action
        called = []
        callbacks = ActionCallbacks(generate_callback=lambda path, mod: called.append(path))
        _dispatch_action(
            ClassifierActionType.GENERATE, None, "pipe", IMAGE,
            callbacks, None,
        )
        assert called == [IMAGE]

    def test_generate_queue_threaded_through_run_pipeline(self):
        """run_pipeline forwards generate_queue to _dispatch_action."""
        submitted = []

        class FakeQueue:
            def submit(self, fn, *args, **kwargs):
                submitted.append(args[0])  # image_path

        cond = FilenameContainsCondition(["image"], case_sensitive=False)  # matches IMAGE
        p = _pipeline(_node("n1", cond, on_match=_execute(ClassifierActionType.GENERATE)))
        callbacks = ActionCallbacks(generate_callback=lambda path, mod: None)
        run_pipeline(p, IMAGE, callbacks, generate_queue=FakeQueue())
        assert IMAGE in submitted


# ---------------------------------------------------------------------------
# UnknownSuffixCondition
# ---------------------------------------------------------------------------

class TestUnknownSuffixConditionRunner:
    def setup_method(self):
        clear_base_stem_dir_cache()

    def teardown_method(self):
        clear_base_stem_dir_cache()

    def _cond(self, expected_suffixes=None, classifier_name="", inference_threshold=0.85):
        return UnknownSuffixCondition(
            expected_suffixes=expected_suffixes or ["_a", "_b", "_c", "_d"],
            classifier_name=classifier_name,
            inference_threshold=inference_threshold,
        )

    def _patch_find(self, files):
        return patch("compare.classifier_pipeline_runner.find_files_by_base_stem",
                     return_value=files)

    def _patch_stem(self, stem="stem"):
        return patch("compare.classifier_pipeline_runner.extract_filename_base_stem",
                     return_value=stem)

    def test_clean_set_returns_false(self, monkeypatch):
        """All files have recognised suffixes → no unknown file → returns False."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem.jpg", "/dir/stem_a.jpg", "/dir/stem_b.jpg"])
        result, _ = _eval_unknown_suffix(self._cond(), IMAGE)
        assert result is False

    def test_seed_only_returns_false(self, monkeypatch):
        """Seed image alone (no suffix) is never flagged."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem.jpg"])
        result, _ = _eval_unknown_suffix(self._cond(), "/dir/stem.jpg")
        assert result is False

    def test_current_image_excluded_from_check(self, monkeypatch):
        """The image being evaluated is never flagged as unknown even if suffix not in expected."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: [IMAGE])
        result, _ = _eval_unknown_suffix(self._cond(), IMAGE)
        assert result is False

    def test_unknown_suffix_no_classifier_returns_true(self, monkeypatch):
        """File with unrecognised suffix and no classifier → unresolvable → True."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem_x.jpg"])
        result, _ = _eval_unknown_suffix(self._cond(classifier_name=""), IMAGE)
        assert result is True

    def test_unknown_suffix_classifier_resolves_returns_false(self, monkeypatch):
        """Classifier returns high-confidence result → file resolved → False."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem_x.jpg"])

        fake_classifier = type("C", (), {"predict_image_ranked": lambda self, p: [("animal", 0.92)]})()
        fake_manager = type("M", (), {"get_classifier": lambda self, n: fake_classifier})()

        with patch("image.image_classifier_manager.image_classifier_manager", fake_manager):
            result, _ = _eval_unknown_suffix(
                self._cond(classifier_name="animals", inference_threshold=0.85), IMAGE
            )
        assert result is False

    def test_unknown_suffix_classifier_below_threshold_returns_true(self, monkeypatch):
        """Classifier confidence below threshold → not resolved → True."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem_x.jpg"])

        fake_classifier = type("C", (), {"predict_image_ranked": lambda self, p: [("animal", 0.60)]})()
        fake_manager = type("M", (), {"get_classifier": lambda self, n: fake_classifier})()

        with patch("image.image_classifier_manager.image_classifier_manager", fake_manager):
            result, _ = _eval_unknown_suffix(
                self._cond(classifier_name="animals", inference_threshold=0.85), IMAGE
            )
        assert result is True

    def test_notable_message_emitted_for_unknown_file(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem_x.jpg"])
        report = PipelineRunReport()
        _eval_unknown_suffix(self._cond(), IMAGE, node_name="guard", report=report)
        msgs = report.messages()
        notable = [m for m in msgs if m.severity == "NOTABLE"]
        assert len(notable) == 1
        assert "stem_x.jpg" in notable[0].detail

    def test_warning_message_emitted_when_unresolvable(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem_x.jpg"])
        report = PipelineRunReport()
        _eval_unknown_suffix(self._cond(classifier_name=""), IMAGE, node_name="guard", report=report)
        warnings = [m for m in report.messages() if m.severity == "WARNING"]
        assert len(warnings) == 1

    def test_info_message_emitted_when_classifier_resolves(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem_x.jpg"])

        fake_classifier = type("C", (), {"predict_image_ranked": lambda self, p: [("animal", 0.91)]})()
        fake_manager = type("M", (), {"get_classifier": lambda self, n: fake_classifier})()

        report = PipelineRunReport()
        with patch("image.image_classifier_manager.image_classifier_manager", fake_manager):
            _eval_unknown_suffix(
                self._cond(classifier_name="animals", inference_threshold=0.85),
                IMAGE, node_name="guard", report=report,
            )
        info_msgs = [m for m in report.messages() if m.severity == "INFO"]
        assert len(info_msgs) == 1
        assert "animal" in info_msgs[0].detail

    def test_no_base_stem_returns_false(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: None)
        result, _ = _eval_unknown_suffix(self._cond(), IMAGE)
        assert result is False

    def test_empty_dirs_returns_false(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        result, _ = _eval_unknown_suffix(self._cond(), IMAGE)
        assert result is False

    def test_trailing_digit_variant_is_recognised(self, monkeypatch):
        """stem_a2 matches expected suffix _a → not flagged as unknown."""
        monkeypatch.setattr("compare.classifier_pipeline_runner.extract_filename_base_stem",
                            lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_pipeline_runner.find_files_by_base_stem",
                            lambda *a, **kw: ["/dir/stem_a2.jpg"])
        result, _ = _eval_unknown_suffix(self._cond(), IMAGE)
        assert result is False

    def test_use_base_directory_ignores_config_dirs(self, monkeypatch, tmp_path):
        """use_base_directory=True → guard scans base_directory, not config dirs."""
        # Place an unknown-suffix file in the "config" dir — guard must not see it.
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "stem_x.jpg").touch()
        # Working dir is empty.
        working_dir = tmp_path / "working"
        working_dir.mkdir()

        monkeypatch.setattr(config, "directories_to_search_for_related_images",
                            [str(config_dir)])
        cond = UnknownSuffixCondition(
            expected_suffixes=["_a", "_b", "_c", "_d"],
            use_base_directory=True,
        )
        result, _ = _eval_unknown_suffix(
            cond, str(working_dir / "stem.jpg"),
            base_directory=str(working_dir),
        )
        assert result is False  # config dir not scanned; working dir is clean

    def test_use_base_directory_detects_unknown_in_base_dir(self, monkeypatch, tmp_path):
        """use_base_directory=True → unknown file in base_directory is detected."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        (working_dir / "stem_zzz.jpg").touch()

        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        cond = UnknownSuffixCondition(
            expected_suffixes=["_a", "_b", "_c", "_d"],
            use_base_directory=True,
        )
        result, _ = _eval_unknown_suffix(
            cond, str(working_dir / "stem.jpg"),
            base_directory=str(working_dir),
        )
        assert result is True  # _zzz is unrecognised and unresolvable

    def test_use_base_directory_falls_back_to_image_dir(self, monkeypatch, tmp_path):
        """use_base_directory=True with no base_directory → uses image's own directory."""
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "stem_zzz.jpg").touch()

        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        cond = UnknownSuffixCondition(
            expected_suffixes=["_a", "_b"],
            use_base_directory=True,
        )
        result, _ = _eval_unknown_suffix(
            cond, str(img_dir / "stem.jpg"),
            base_directory=None,
        )
        assert result is True  # fell back to image's dir, found stem_zzz.jpg



# ---------------------------------------------------------------------------
# RelatedImageCondition
# ---------------------------------------------------------------------------

class TestRelatedImageConditionRunner:
    def _cond(self, edit_suffix="_edit", search_directory="", count_threshold=1,
              use_configured_search_directories=False):
        return RelatedImageCondition(
            edit_suffix=edit_suffix,
            search_directory=search_directory,
            count_threshold=count_threshold,
            use_configured_search_directories=use_configured_search_directories,
        )

    def _patch_gate(self, result):
        return patch("files.related_image.should_run_generate_action", return_value=result)

    def test_gate_true_returns_true(self):
        with self._patch_gate(True):
            result, score = _eval_related_image(self._cond(), IMAGE, "/base")
        assert result is True
        assert score is None

    def test_gate_false_returns_false(self):
        with self._patch_gate(False):
            result, _ = _eval_related_image(self._cond(), IMAGE, "/base")
        assert result is False

    def test_search_directory_on_condition_takes_priority(self):
        captured = []

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            captured.append(search_dir)
            return True

        cond = self._cond(search_directory="/from_cond")
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            _eval_related_image(cond, IMAGE, "/base_dir")

        assert captured == ["/from_cond"]

    def test_falls_back_to_base_directory(self):
        captured = []

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            captured.append(search_dir)
            return True

        cond = self._cond(search_directory="", use_configured_search_directories=False)
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            _eval_related_image(cond, IMAGE, "/base_dir")

        assert captured == ["/base_dir"]

    def test_falls_back_to_image_directory(self):
        captured = []

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            captured.append(search_dir)
            return True

        cond = self._cond(search_directory="", use_configured_search_directories=False)
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            _eval_related_image(cond, "/dir/source.jpg", None)

        assert captured == ["/dir"]

    def test_count_threshold_forwarded(self):
        captured = []

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            captured.append(count_threshold)
            return True

        cond = self._cond(count_threshold=5)
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            _eval_related_image(cond, IMAGE, "/base")

        assert captured == [5]

    def test_dispatched_via_evaluate_condition(self):
        with self._patch_gate(True):
            result, _ = _evaluate_condition(self._cond(), IMAGE, {}, {})
        assert result is True

    def test_base_directory_threaded_through_run_pipeline(self):
        """base_directory passed to run_pipeline reaches _eval_related_image."""
        captured = []

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            captured.append(search_dir)
            return False

        cond = RelatedImageCondition(
            edit_suffix="_edit",
            search_directory="",
            use_configured_search_directories=False,
        )
        p = _pipeline(
            _node("n1", cond,
                  on_match=_execute(ClassifierActionType.GENERATE),
                  on_no_match=NodeOutcome.accept()),
        )
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            run_pipeline(p, IMAGE, ActionCallbacks(), base_directory="/pipeline_base")

        assert captured == ["/pipeline_base"]

    def test_use_configured_dirs_searches_all_config_dirs(self, monkeypatch):
        captured = []

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            captured.append(search_dir)
            return True

        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/cfg1", "/cfg2"])
        cond = RelatedImageCondition(
            edit_suffix="_edit",
            search_directory="",
            use_configured_search_directories=True,
        )
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            result, _ = _eval_related_image(cond, IMAGE, "/base_dir")

        assert result is True
        assert captured == ["/cfg1", "/cfg2"]
        assert "/base_dir" not in captured

    def test_use_configured_dirs_returns_false_if_any_dir_has_file(self, monkeypatch):
        """If any configured dir already has the file, generation should be suppressed."""
        results = {"/cfg1": True, "/cfg2": False}

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            return results[search_dir]

        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/cfg1", "/cfg2"])
        cond = RelatedImageCondition(
            edit_suffix="_edit",
            use_configured_search_directories=True,
        )
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            result, _ = _eval_related_image(cond, IMAGE, None)

        assert result is False

    def test_use_configured_dirs_empty_config_returns_false(self, monkeypatch):
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        cond = RelatedImageCondition(
            edit_suffix="_edit",
            use_configured_search_directories=True,
        )
        with patch("files.related_image.should_run_generate_action", return_value=True):
            result, _ = _eval_related_image(cond, IMAGE, None)

        assert result is False


# ---------------------------------------------------------------------------
# GroupCondition / GroupChildResultCondition
# ---------------------------------------------------------------------------

class TestGroupConditionRunner:
    def _group(self, operator="OR", children=None):
        """Build a GroupCondition with FilenameContains children (no ML deps)."""
        nodes = [
            PipelineNode(name, FilenameContainsCondition([name], case_sensitive=False))
            for name in (children or [])
        ]
        return GroupCondition(operator=operator, nodes=nodes)

    def test_or_any_child_matches(self):
        cond = self._group("OR", ["alpha", "beta"])
        nr, ns = {}, {}
        result, _ = _eval_group(cond, "grp", "/path/alpha_image.jpg", nr, ns, None)
        assert result is True
        assert nr["grp/alpha"] is True
        assert nr["grp/beta"] is False

    def test_or_no_child_matches(self):
        cond = self._group("OR", ["alpha", "beta"])
        nr, ns = {}, {}
        result, _ = _eval_group(cond, "grp", "/path/gamma.jpg", nr, ns, None)
        assert result is False
        assert nr["grp/alpha"] is False
        assert nr["grp/beta"] is False

    def test_and_all_match(self):
        cond = self._group("AND", ["alpha", "beta"])
        nr, ns = {}, {}
        result, _ = _eval_group(cond, "grp", "/path/alpha_beta.jpg", nr, ns, None)
        assert result is True

    def test_and_one_fails(self):
        cond = self._group("AND", ["alpha", "beta"])
        nr, ns = {}, {}
        result, _ = _eval_group(cond, "grp", "/path/alpha_only.jpg", nr, ns, None)
        assert result is False
        assert nr["grp/alpha"] is True
        assert nr["grp/beta"] is False

    def test_all_children_evaluated_no_short_circuit(self):
        """OR does not short-circuit: all child results must be stored."""
        cond = self._group("OR", ["a", "b", "c"])
        nr, ns = {}, {}
        _eval_group(cond, "g", "/path/a_image.jpg", nr, ns, None)
        assert "g/a" in nr
        assert "g/b" in nr
        assert "g/c" in nr

    def test_child_exception_treated_as_false(self, monkeypatch):
        import compare.compare_embeddings_clip as clip_mod
        monkeypatch.setattr(
            clip_mod.CompareEmbeddingClip, "multi_text_compare",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        cond = GroupCondition(operator="OR", nodes=[
            PipelineNode("exploding", EmbeddingCondition(["x"])),
            PipelineNode("safe", FilenameContainsCondition(["safe"])),
        ])
        nr, ns = {}, {}
        result, _ = _eval_group(cond, "g", "/path/safe.jpg", nr, ns, None)
        assert nr["g/exploding"] is False
        assert nr["g/safe"] is True
        assert result is True  # OR: at least one matched

    def test_dispatches_via_evaluate_condition(self):
        cond = self._group("OR", ["draft"])
        nr, ns = {}, {}
        result, _ = _evaluate_condition(cond, "/path/draft.jpg", nr, ns, node_name="grp")
        assert result is True
        assert nr["grp/draft"] is True

    def test_group_child_result_match(self):
        cond = GroupChildResultCondition("grp", "alpha", expected_result=True)
        result, score = _evaluate_condition(cond, IMAGE, {"grp/alpha": True}, {})
        assert result is True
        assert score == pytest.approx(1.0)

    def test_group_child_result_expected_false(self):
        cond = GroupChildResultCondition("grp", "alpha", expected_result=False)
        result, _ = _evaluate_condition(cond, IMAGE, {"grp/alpha": False}, {})
        assert result is True

    def test_group_child_result_mismatch(self):
        cond = GroupChildResultCondition("grp", "alpha", expected_result=True)
        result, _ = _evaluate_condition(cond, IMAGE, {"grp/alpha": False}, {})
        assert result is False

    def test_group_child_result_missing_key(self):
        cond = GroupChildResultCondition("grp", "ghost", expected_result=True)
        result, _ = _evaluate_condition(cond, IMAGE, {}, {})
        assert result is False

    def test_run_pipeline_routes_on_child_match(self):
        """Group stores child results; downstream GroupChildResult routes correctly."""
        n_group = PipelineNode(
            name="hints",
            condition=GroupCondition(operator="OR", nodes=[
                PipelineNode("hide", FilenameContainsCondition(["_hide"])),
                PipelineNode("keep", FilenameContainsCondition(["_keep"])),
            ]),
            on_match=NodeOutcome.continue_(),
            on_no_match=NodeOutcome.continue_(),
        )
        n_check = PipelineNode(
            name="act_on_hide",
            condition=GroupChildResultCondition("hints", "hide", expected_result=True),
            on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.HIDE),
            on_no_match=NodeOutcome.accept(),
        )
        p = ClassifierPipeline(name="t", nodes=[n_group, n_check])
        calls, hide, _, _, _ = _callbacks()
        result = run_pipeline(p, "/photos/cat_hide.jpg",
                              ActionCallbacks(hide_callback=hide))
        assert result == ClassifierActionType.HIDE
        assert "/photos/cat_hide.jpg" in calls["hide"]

    def test_run_pipeline_child_no_match_accepts(self):
        """Non-matching child → GroupChildResult no_match → ACCEPT."""
        n_group = PipelineNode(
            name="hints",
            condition=GroupCondition(operator="OR", nodes=[
                PipelineNode("hide", FilenameContainsCondition(["_hide"])),
                PipelineNode("keep", FilenameContainsCondition(["_keep"])),
            ]),
            on_match=NodeOutcome.continue_(),
            on_no_match=NodeOutcome.continue_(),
        )
        n_check = PipelineNode(
            name="act_on_hide",
            condition=GroupChildResultCondition("hints", "hide", expected_result=True),
            on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.HIDE),
            on_no_match=NodeOutcome.accept(),
        )
        p = ClassifierPipeline(name="t", nodes=[n_group, n_check])
        result = run_pipeline(p, "/photos/cat_keep.jpg", ActionCallbacks())
        assert result is None  # accepted

    def test_run_pipeline_and_group_all_must_match(self):
        """AND group only matches when both children match."""
        n_group = PipelineNode(
            name="grp",
            condition=GroupCondition(operator="AND", nodes=[
                PipelineNode("a", FilenameContainsCondition(["_a_"])),
                PipelineNode("b", FilenameContainsCondition(["_b_"])),
            ]),
            on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.NOTIFY),
            on_no_match=NodeOutcome.accept(),
        )
        p = ClassifierPipeline(name="t", nodes=[n_group])
        calls, _, notify, _, _ = _callbacks()
        cb = ActionCallbacks(notify_callback=notify)

        assert run_pipeline(p, "/path/_a_.jpg", cb) is None        # only a matches
        assert run_pipeline(p, "/path/_a__b_.jpg", cb) == ClassifierActionType.NOTIFY


# ---------------------------------------------------------------------------
# run_pipeline — applies_to_media_types gate
# ---------------------------------------------------------------------------

class TestRunPipelineMediaTypeGate:
    def _matching_pipeline(self, applies_to=None):
        """A pipeline that executes NOTIFY when its embedding node matches."""
        return ClassifierPipeline(
            name="gated",
            applies_to_media_types=applies_to,
            nodes=[_node("n1", EmbeddingCondition(["x"]),
                         on_match=_execute(ClassifierActionType.NOTIFY))],
        )

    def _patch_embedding_match(self):
        return patch("compare.compare_embeddings_clip.CompareEmbeddingClip.multi_text_compare",
                     return_value=True)

    def test_none_applies_to_runs_for_all_types(self, monkeypatch):
        p = self._matching_pipeline(applies_to=None)
        with _patch_media_type(CompareMediaType.VIDEO):
            with self._patch_embedding_match():
                result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result == ClassifierActionType.NOTIFY

    def test_allowed_type_proceeds(self):
        p = self._matching_pipeline(applies_to=[CompareMediaType.IMAGE])
        with _patch_media_type(CompareMediaType.IMAGE):
            with self._patch_embedding_match():
                result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result == ClassifierActionType.NOTIFY

    def test_disallowed_type_returns_none_without_evaluating(self):
        p = self._matching_pipeline(applies_to=[CompareMediaType.IMAGE])
        evaluated = []
        with _patch_media_type(CompareMediaType.VIDEO):
            with patch.object(p.nodes[0], "condition",
                              wraps=p.nodes[0].condition) as mock_cond:
                result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result is None

    def test_disallowed_type_does_not_call_evaluate_condition(self):
        p = self._matching_pipeline(applies_to=[CompareMediaType.PDF])
        calls = []
        original = _evaluate_condition

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        with _patch_media_type(CompareMediaType.IMAGE):
            with patch("compare.classifier_pipeline_runner._evaluate_condition", side_effect=spy):
                result = run_pipeline(p, IMAGE, ActionCallbacks())

        assert result is None
        assert not calls, "_evaluate_condition must not be called for a disallowed type"

    def test_inactive_pipeline_still_returns_none(self):
        p = ClassifierPipeline(name="p", is_active=False,
                               applies_to_media_types=[CompareMediaType.IMAGE],
                               nodes=[_node("n1", EmbeddingCondition(["x"]),
                                           on_match=_execute(ClassifierActionType.NOTIFY))])
        with _patch_media_type(CompareMediaType.IMAGE):
            result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result is None


# ---------------------------------------------------------------------------
# Stem-group skip: _seed_exists_in_dirs and _resolve_stem_group (§5.13)
# ---------------------------------------------------------------------------

class TestSeedExistsInDirs:
    def setup_method(self):
        clear_base_stem_dir_cache()

    def teardown_method(self):
        clear_base_stem_dir_cache()

    def test_exact_stem_match_found(self, tmp_path):
        (tmp_path / "rose.jpg").touch()
        (tmp_path / "rose_a.jpg").touch()
        assert _seed_exists_in_dirs("rose", [str(tmp_path)]) is True

    def test_suffix_variant_not_treated_as_seed(self, tmp_path):
        (tmp_path / "rose_a.jpg").touch()
        assert _seed_exists_in_dirs("rose", [str(tmp_path)]) is False

    def test_empty_dirs_returns_false(self):
        assert _seed_exists_in_dirs("rose", []) is False

    def test_multiple_dirs_found_in_second(self, tmp_path):
        d1 = tmp_path / "a"
        d1.mkdir()
        d2 = tmp_path / "b"
        d2.mkdir()
        (d2 / "rose.jpg").touch()
        assert _seed_exists_in_dirs("rose", [str(d1), str(d2)]) is True


class TestResolveStemGroup:
    def setup_method(self):
        clear_base_stem_dir_cache()

    def teardown_method(self):
        clear_base_stem_dir_cache()

    # -- helpers --

    def _patch_seed_exists(self, dirs_with_seed, dirs_without_seed=None):
        """Patch _seed_exists_in_dirs: returns True if dirs match dirs_with_seed."""
        def fake(base_stem, dirs):
            return list(dirs) == list(dirs_with_seed)
        return patch("compare.classifier_pipeline_runner._seed_exists_in_dirs", side_effect=fake)

    # -- Type 1 valid --

    def test_type1_valid_seed_in_target(self, tmp_path, monkeypatch):
        seed = tmp_path / "rose.jpg"
        seed.touch()
        target = tmp_path / "target"
        target.mkdir()
        (target / "rose.jpg").touch()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [str(target)])
        should_eval, should_mark = _resolve_stem_group(str(seed), "rose", is_seed=True)
        assert should_eval is True
        assert should_mark is True

    # -- Malformed C --

    def test_malformed_c_seed_not_in_target(self, tmp_path, monkeypatch):
        seed = tmp_path / "rose.jpg"
        seed.touch()
        target = tmp_path / "target"
        target.mkdir()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [str(target)])
        should_eval, should_mark = _resolve_stem_group(str(seed), "rose", is_seed=True)
        assert should_eval is False
        assert should_mark is True

    # -- Type 1 derivative --

    def test_type1_derivative_seed_in_working_dir(self, tmp_path, monkeypatch):
        working = tmp_path / "work"
        working.mkdir()
        (working / "rose.jpg").touch()
        deriv = working / "rose_a.jpg"
        deriv.touch()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [str(tmp_path)])
        should_eval, should_mark = _resolve_stem_group(str(deriv), "rose", is_seed=False)
        assert should_eval is True
        assert should_mark is False  # seed will mark done

    # -- Type 2 valid --

    def test_type2_valid_metadata_and_seed_in_target(self, tmp_path, monkeypatch):
        working = tmp_path / "work"
        working.mkdir()
        target = tmp_path / "target"
        target.mkdir()
        (target / "rose.jpg").touch()
        deriv = working / "rose_a.jpg"
        deriv.touch()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [str(target)])
        # No seed in working dir; metadata resolves to the target file
        with patch("compare.classifier_pipeline_runner.get_related_image_path",
                   return_value=(str(target / "rose.jpg"), True)):
            should_eval, should_mark = _resolve_stem_group(str(deriv), "rose", is_seed=False)
        assert should_eval is True
        assert should_mark is True  # first derivative acts as source

    # -- Malformed A --

    def test_malformed_a_no_metadata(self, tmp_path, monkeypatch):
        working = tmp_path / "work"
        working.mkdir()
        target = tmp_path / "target"
        target.mkdir()
        deriv = working / "rose_a.jpg"
        deriv.touch()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [str(target)])
        with patch("compare.classifier_pipeline_runner.get_related_image_path",
                   return_value=(None, False)):
            should_eval, should_mark = _resolve_stem_group(str(deriv), "rose", is_seed=False)
        assert should_eval is False
        assert should_mark is True

    # -- Malformed B --

    def test_malformed_b_metadata_but_not_in_target(self, tmp_path, monkeypatch):
        working = tmp_path / "work"
        working.mkdir()
        target = tmp_path / "target"
        target.mkdir()
        deriv = working / "rose_a.jpg"
        deriv.touch()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [str(target)])
        with patch("compare.classifier_pipeline_runner.get_related_image_path",
                   return_value=("/some/unfiled/rose.jpg", False)):
            should_eval, should_mark = _resolve_stem_group(str(deriv), "rose", is_seed=False)
        assert should_eval is False
        assert should_mark is True

    # -- No target dirs fallback --

    def test_no_target_dirs_seed_evaluates_and_marks(self, tmp_path, monkeypatch):
        seed = tmp_path / "rose.jpg"
        seed.touch()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        should_eval, should_mark = _resolve_stem_group(str(seed), "rose", is_seed=True)
        assert should_eval is True
        assert should_mark is True

    def test_no_target_dirs_derivative_evaluates_does_not_mark(self, tmp_path, monkeypatch):
        deriv = tmp_path / "rose_a.jpg"
        deriv.touch()
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        should_eval, should_mark = _resolve_stem_group(str(deriv), "rose", is_seed=False)
        assert should_eval is True
        assert should_mark is False


class TestProcessedStemsIntegration:
    """End-to-end tests for the processed_stems skip in run_pipeline."""

    def _always_match_pipeline(self):
        return ClassifierPipeline(
            name="p",
            nodes=[_node("n1", FilenameContainsCondition(["image"]),
                          on_match=_execute(ClassifierActionType.NOTIFY))],
        )

    def test_stem_already_in_set_skips_evaluation(self):
        p = self._always_match_pipeline()
        stems = {"image"}  # IMAGE = "/fake/image.jpg", base_stem = "image"
        with patch("compare.classifier_pipeline_runner._resolve_stem_group",
                   return_value=(True, True)) as mock_resolve:
            result = run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=stems)
        assert result is None
        mock_resolve.assert_not_called()  # skipped before resolve

    def test_malformed_group_skipped_and_marked(self):
        p = self._always_match_pipeline()
        stems = set()
        with patch("compare.classifier_pipeline_runner._resolve_stem_group",
                   return_value=(False, True)):
            result = run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=stems)
        assert result is None
        assert "image" in stems  # marked done to suppress later derivatives

    def test_type1_derivative_evaluated_but_not_marked(self):
        p = ClassifierPipeline(
            name="p",
            nodes=[_node("n1", FilenameContainsCondition(["image"]),
                          on_match=NodeOutcome.accept())],
        )
        stems = set()
        with patch("compare.classifier_pipeline_runner._resolve_stem_group",
                   return_value=(True, False)):  # type-1 derivative
            run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=stems)
        assert "image" not in stems  # seed must mark it, not the derivative

    def test_seed_marked_after_execute(self):
        p = self._always_match_pipeline()
        stems = set()
        with patch("compare.classifier_pipeline_runner._resolve_stem_group",
                   return_value=(True, True)):
            run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=stems)
        assert "image" in stems

    def test_seed_marked_after_accept(self):
        p = ClassifierPipeline(
            name="p",
            nodes=[_node("n1", FilenameContainsCondition(["image"]),
                          on_match=NodeOutcome.accept())],
        )
        stems = set()
        with patch("compare.classifier_pipeline_runner._resolve_stem_group",
                   return_value=(True, True)):
            run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=stems)
        assert "image" in stems

    def test_second_call_same_stem_skipped(self):
        p = self._always_match_pipeline()
        stems = set()
        calls = []
        original_eval = _evaluate_condition

        def spy(*args, **kwargs):
            calls.append(args)
            return original_eval(*args, **kwargs)

        with patch("compare.classifier_pipeline_runner._resolve_stem_group",
                   return_value=(True, True)):
            with patch("compare.classifier_pipeline_runner._evaluate_condition",
                       side_effect=spy):
                run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=stems)
                run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=stems)

        assert len(calls) == 1  # second call skipped entirely

    def test_no_processed_stems_never_skips(self):
        p = self._always_match_pipeline()
        calls = []
        original_eval = _evaluate_condition

        def spy(*args, **kwargs):
            calls.append(args)
            return original_eval(*args, **kwargs)

        with patch("compare.classifier_pipeline_runner._evaluate_condition",
                   side_effect=spy):
            run_pipeline(p, IMAGE, ActionCallbacks())
            run_pipeline(p, IMAGE, ActionCallbacks())

        assert len(calls) == 2  # both calls run fully


# ---------------------------------------------------------------------------
# Seed-category guard
# ---------------------------------------------------------------------------

class TestSeedCategoryGuard:
    """Tests for the pipeline-level seed_category guard in run_pipeline.

    IMAGE = "/fake/image.jpg" is a seed: its file stem ("image") equals the
    base stem extracted by extract_filename_base_stem ("image").
    "/fake/image_apple.jpg" is NOT a seed: file stem "image_apple" ≠ "image".
    """

    _CAT_MAP = {"Apple": "_apple", "Banana": "_banana"}

    # A GENERATE-apple node whose condition would always match IMAGE.
    def _gen_apple_node(self, outcome_type=OutcomeType.EXECUTE, on_no_match=None):
        return PipelineNode(
            name="gen_apple",
            condition=FilenameContainsCondition(["image"]),
            on_match=NodeOutcome(
                outcome_type,
                action_type=ClassifierActionType.GENERATE,
                action_modifier="_apple",
            ),
            on_no_match=on_no_match or NodeOutcome.accept(),
        )

    def _pipeline_with_seed_category(self, *nodes):
        return ClassifierPipeline(
            name="p",
            category_map=self._CAT_MAP,
            seed_category="Apple",
            nodes=list(nodes),
        )

    def test_generate_skipped_for_seed_image(self):
        """Guard fires: GENERATE not dispatched and on_no_match (accept) applies."""
        p = self._pipeline_with_seed_category(self._gen_apple_node())
        dispatched = []
        with patch("compare.classifier_pipeline_runner._dispatch_action",
                   side_effect=lambda *a, **kw: dispatched.append(a[0])):
            result = run_pipeline(p, IMAGE, ActionCallbacks())
        assert result is None          # on_no_match = accept
        assert not dispatched

    def test_evaluate_condition_not_called_for_guarded_node(self):
        """Guard bypasses _evaluate_condition entirely — no wasted evaluation."""
        p = self._pipeline_with_seed_category(self._gen_apple_node())
        calls = []
        with patch("compare.classifier_pipeline_runner._evaluate_condition",
                   side_effect=lambda *a, **kw: calls.append(True) or (False, None)):
            run_pipeline(p, IMAGE, ActionCallbacks())
        assert not calls

    def test_guard_does_not_fire_for_non_seed_image(self):
        """Non-seed: condition evaluated and GENERATE dispatched normally.

        Uses a long-stem path so extract_filename_base_stem strips the _apple
        suffix (stem prefix must be ≥10 chars for suffix stripping to trigger).
        """
        p = self._pipeline_with_seed_category(self._gen_apple_node())
        dispatched = []
        # image_17820172251234 is 21 chars → _apple suffix is stripped →
        # base_stem="image_17820172251234", file_stem="image_17820172251234_apple" → is_seed=False
        # The prefix also contains "image" so FilenameContainsCondition(["image"]) matches.
        non_seed = "/fake/image_17820172251234_apple.jpg"
        with patch("compare.classifier_pipeline_runner._dispatch_action",
                   side_effect=lambda *a, **kw: dispatched.append(a[0])):
            run_pipeline(p, non_seed, ActionCallbacks())
        assert ClassifierActionType.GENERATE in dispatched

    def test_guard_does_not_fire_for_different_category_node(self):
        """Seed image but the node generates Banana, not Apple: condition evaluated."""
        gen_banana = PipelineNode(
            name="gen_banana",
            condition=FilenameContainsCondition(["image"]),
            on_match=NodeOutcome(
                OutcomeType.EXECUTE,
                action_type=ClassifierActionType.GENERATE,
                action_modifier="_banana",
            ),
            on_no_match=NodeOutcome.accept(),
        )
        p = self._pipeline_with_seed_category(gen_banana)
        dispatched = []
        with patch("compare.classifier_pipeline_runner._dispatch_action",
                   side_effect=lambda *a, **kw: dispatched.append(a[0])):
            run_pipeline(p, IMAGE, ActionCallbacks())
        assert ClassifierActionType.GENERATE in dispatched

    def test_guard_does_not_fire_when_seed_category_unset(self):
        """No seed_category: GENERATE dispatched normally even for a seed image."""
        p = ClassifierPipeline(
            name="p",
            category_map=self._CAT_MAP,
            nodes=[self._gen_apple_node()],
        )
        dispatched = []
        with patch("compare.classifier_pipeline_runner._dispatch_action",
                   side_effect=lambda *a, **kw: dispatched.append(a[0])):
            run_pipeline(p, IMAGE, ActionCallbacks())
        assert ClassifierActionType.GENERATE in dispatched

    def test_guard_adds_info_to_report(self):
        """Guard emits an INFO PipelineMessage when it fires."""
        from compare.pipeline_run_report import PipelineRunReport
        p = self._pipeline_with_seed_category(self._gen_apple_node())
        report = PipelineRunReport()
        with patch("compare.classifier_pipeline_runner._dispatch_action"):
            run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        infos = [m for m in report.messages()
                 if m.severity == "INFO" and m.node == "gen_apple"]
        assert infos, "Expected an INFO message for the guarded node"
        assert "Apple" in infos[0].detail

    def test_guard_fires_for_execute_and_continue_outcome(self):
        """Guard also applies when on_match is EXECUTE_AND_CONTINUE."""
        p = self._pipeline_with_seed_category(
            self._gen_apple_node(outcome_type=OutcomeType.EXECUTE_AND_CONTINUE)
        )
        dispatched = []
        with patch("compare.classifier_pipeline_runner._dispatch_action",
                   side_effect=lambda *a, **kw: dispatched.append(a[0])):
            run_pipeline(p, IMAGE, ActionCallbacks())
        assert not dispatched

    def test_only_matching_category_guarded_in_multi_node_pipeline(self):
        """Apple node is guarded; Banana node is evaluated and fires normally."""
        gen_banana = PipelineNode(
            name="gen_banana",
            condition=FilenameContainsCondition(["image"]),
            on_match=NodeOutcome(
                OutcomeType.EXECUTE,
                action_type=ClassifierActionType.GENERATE,
                action_modifier="_banana",
            ),
            on_no_match=NodeOutcome.accept(),
        )
        p = self._pipeline_with_seed_category(
            self._gen_apple_node(on_no_match=NodeOutcome.continue_()), gen_banana
        )
        dispatched = []
        with patch("compare.classifier_pipeline_runner._dispatch_action",
                   side_effect=lambda *a, **kw: dispatched.append(a[0])):
            run_pipeline(p, IMAGE, ActionCallbacks())
        # Apple guarded → on_no_match (CONTINUE) → Banana evaluated and matches
        assert dispatched == [ClassifierActionType.GENERATE]
        # Verify it was the banana node that fired, not apple
        # (apple was guarded; banana condition matched)

    def test_guard_works_without_processed_stems(self):
        """Guard is independent of the processed_stems path."""
        p = self._pipeline_with_seed_category(self._gen_apple_node())
        dispatched = []
        with patch("compare.classifier_pipeline_runner._dispatch_action",
                   side_effect=lambda *a, **kw: dispatched.append(a[0])):
            run_pipeline(p, IMAGE, ActionCallbacks(), processed_stems=None)
        assert not dispatched

"""
Unit tests for compare.classifier_pipeline_runner (Phase 2 — execution engine).

All external ML dependencies (CLIP, classifiers, prototypes, prompt extractor)
are mocked so no real models are loaded.  File-system operations in action
dispatch are also mocked.
"""

import pytest
import numpy as np

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierRankCondition,
    CompositeCondition,
    EmbeddingCondition,
    FilenameContainsCondition,
    LookaheadCondition,
    NodeOutcome,
    NodeResultCondition,
    OutcomeType,
    PipelineNode,
    PromptCondition,
    PrototypeCondition,
)
from compare.classifier_pipeline_runner import (
    _evaluate_condition,
    _eval_classifier_rank,
    _eval_composite,
    _eval_filename_contains,
    _eval_lookahead,
    _eval_prompt,
    _eval_prototype,
    run_pipeline,
)
from utils.constants import ClassifierActionType


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
        result = run_pipeline(p, "/media/photo_reject_001.jpg", hide_callback=hide,
                              notify_callback=notify)
        assert result == ClassifierActionType.HIDE
        assert "/media/photo_reject_001.jpg" in calls["hide"]

    def test_via_run_pipeline_no_match_accepts(self):
        cond = FilenameContainsCondition(["_reject"], case_sensitive=False)
        p = _pipeline(
            _node("n1", cond, on_match=_execute(ClassifierActionType.HIDE),
                  on_no_match=NodeOutcome.accept())
        )
        result = run_pipeline(p, "/media/photo_keep_001.jpg")
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
        assert run_pipeline(p, IMAGE) is None

    def test_empty_pipeline_returns_none(self):
        p = ClassifierPipeline(name="empty")
        assert run_pipeline(p, IMAGE) is None

    def test_execute_on_match(self, monkeypatch):
        self._patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=_execute(ClassifierActionType.NOTIFY))
        )
        calls, hide, notify, mark, blur = _callbacks()
        result = run_pipeline(p, IMAGE, notify_callback=notify)
        assert result == ClassifierActionType.NOTIFY
        assert len(calls["notify"]) == 1

    def test_accept_on_no_match(self, monkeypatch):
        self._patch_embedding(monkeypatch, False)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_no_match=NodeOutcome(OutcomeType.ACCEPT))
        )
        assert run_pipeline(p, IMAGE) is None

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
        result = run_pipeline(p, IMAGE)
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
        result = run_pipeline(p, IMAGE)
        assert result == ClassifierActionType.NOTIFY

    def test_reject_fires_default_reject_action(self, monkeypatch):
        self._patch_embedding(monkeypatch, True)
        calls, hide, notify, mark, blur = _callbacks()
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=NodeOutcome(OutcomeType.REJECT))
        )
        p.default_reject_action = ClassifierActionType.NOTIFY
        result = run_pipeline(p, IMAGE, notify_callback=notify)
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
        result = run_pipeline(p, IMAGE, notify_callback=notify)
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
        assert run_pipeline(p, IMAGE) is None

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
        result = run_pipeline(p, IMAGE, notify_callback=notify, add_mark_callback=mark)
        assert result == ClassifierActionType.ADD_MARK
        assert IMAGE in calls["mark"]


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
            hide_callback=extra_callbacks.get("hide", hide),
            notify_callback=extra_callbacks.get("notify", notify),
            add_mark_callback=extra_callbacks.get("mark", mark),
            blur_callback=extra_callbacks.get("blur", blur),
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
        result = run_pipeline(p, IMAGE, notify_callback=notify)
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
        run_pipeline(p, str(img), notify_callback=notify)
        assert not img.exists()

    def test_no_callbacks_does_not_crash(self, monkeypatch):
        self._patch_embedding(monkeypatch)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"]),
                            on_match=_execute(ClassifierActionType.NOTIFY)))
        result = run_pipeline(p, IMAGE)  # no callbacks passed
        assert result == ClassifierActionType.NOTIFY

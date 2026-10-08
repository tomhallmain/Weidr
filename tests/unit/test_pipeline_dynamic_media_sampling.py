"""
Dynamic media (video, GIF, PDF, ePub) in classifier pipeline conditions: image-content
conditions (classifier rank, embedding, prototype) judge such files by sampled
frames/pages with the pipeline's dynamic_content_* ratios, and negation applies to
the file-level result.

Dynamic-media detection and frame sampling are stubbed: /vid/* paths count as
dynamic media whose four frames are /vid/<name>#0..#3.
"""
from __future__ import annotations

import pytest

import compare.classifier_pipeline_runner as runner
from compare.action_callbacks import ActionCallbacks
from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierRankCondition,
    CompositeCondition,
    EmbeddingCondition,
    NodeOutcome,
    OutcomeType,
    PipelineNode,
    PrototypeCondition,
)
from compare.classifier_pipeline_runner import _evaluate_condition, run_pipeline
from compare.dynamic_media_sampling import DEFAULT_RATIO, FrameSampling
from utils.constants import ClassifierActionType

VIDEO = "/vid/a.mp4"
HALF = FrameSampling(sample_ratio=0.2, positive_ratio=0.5)  # 2 of 4 frames must match


@pytest.fixture
def frames(monkeypatch):
    """Records each stream_frame_samples call as (media_path, sample_ratio)."""
    import utils.media_utils as mu
    from image.frame_cache import FrameCache
    monkeypatch.setattr(mu, "is_classifier_dynamic_media_path", lambda p: p.startswith("/vid/"))
    calls = []

    def stream(media_path, sample_ratio=0.1, detect_pseudostatic=False, max_samples=None):
        calls.append((media_path, sample_ratio))
        paths = [f"{media_path}#{i}" for i in range(4)]
        return len(paths), iter(paths)

    monkeypatch.setattr(FrameCache, "stream_frame_samples", stream)
    return calls


@pytest.fixture
def classify(monkeypatch):
    """Install a fake image classifier: classify[path] = {category: score}; records paths."""
    import image.image_classifier_manager as mgr_mod
    predictions: dict = {}
    seen: list = []

    class FakeClassifier:
        def predict_image_ranked(self, path):
            seen.append(path)
            scores = predictions.get(path, {"person": 0.1, "none": 0.9})
            return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    class FakeManager:
        def get_classifier(self, name):
            return FakeClassifier()

    monkeypatch.setattr(mgr_mod, "image_classifier_manager", FakeManager())
    predictions["seen"] = seen
    return predictions


def _person_in(classify, *frame_indices, score=0.8):
    for i in frame_indices:
        classify[f"{VIDEO}#{i}"] = {"person": score, "none": 1.0 - score}


def _rank(negate=False, categories=("person",)):
    return ClassifierRankCondition("m", list(categories), min_rank=1, max_rank=1, negate=negate)


class TestClassifierRankSampling:
    def test_enough_matching_frames_match(self, frames, classify):
        _person_in(classify, 2, 3)
        assert _evaluate_condition(_rank(), VIDEO, {}, {}, sampling=HALF) == (True, pytest.approx(0.8))

    def test_too_few_matching_frames_do_not_match(self, frames, classify):
        _person_in(classify, 3)
        matched, _score = _evaluate_condition(_rank(), VIDEO, {}, {}, sampling=HALF)
        assert matched is False

    def test_negation_applies_to_the_file(self, frames, classify):
        # One person frame is not enough for the video to contain a person,
        # so "no person" holds for the video even though one frame has one.
        _person_in(classify, 3)
        assert _evaluate_condition(_rank(negate=True), VIDEO, {}, {}, sampling=HALF)[0] is True
        _person_in(classify, 0, 1)
        assert _evaluate_condition(_rank(negate=True), VIDEO, {}, {}, sampling=HALF)[0] is False

    def test_misconfigured_negated_condition_stays_no_match(self, frames, classify):
        cond = _rank(negate=True, categories=())
        assert _evaluate_condition(cond, VIDEO, {}, {}, sampling=HALF) == (False, None)
        assert frames == []

    def test_without_sampling_the_file_itself_is_classified(self, frames, classify):
        _evaluate_condition(_rank(), VIDEO, {}, {})
        assert classify["seen"] == [VIDEO]
        assert frames == []

    def test_still_images_are_not_sampled(self, frames, classify):
        _evaluate_condition(_rank(), "/img/photo.png", {}, {}, sampling=HALF)
        assert classify["seen"] == ["/img/photo.png"]
        assert frames == []

    def test_composite_children_are_sampled(self, frames, classify):
        _person_in(classify, 0, 1)
        not_person = CompositeCondition(operator="NOT", sub_conditions=[_rank()])
        assert _evaluate_condition(not_person, VIDEO, {}, {}, sampling=HALF)[0] is False


class TestOtherImageContentConditions:
    def test_embedding_condition_is_sampled(self, frames, monkeypatch):
        from compare.compare_embeddings_clip import CompareEmbeddingClip
        monkeypatch.setattr(CompareEmbeddingClip, "multi_text_compare",
                            classmethod(lambda cls, p, pos, neg, thr: p.endswith(("#0", "#1"))))
        monkeypatch.setattr(CompareEmbeddingClip, "cached_multi_text_score",
                            classmethod(lambda cls, p, pos, neg: 0.3 if p.endswith("#0") else 0.1))
        cond = EmbeddingCondition(positives=["a person"])
        assert _evaluate_condition(cond, VIDEO, {}, {}, sampling=HALF) == (True, pytest.approx(0.3))

    def test_prototype_condition_is_sampled(self, frames, monkeypatch):
        monkeypatch.setattr(runner, "_eval_prototype", lambda c, p: (p.endswith("#2"), 0.5))
        cond = PrototypeCondition(prototype_directory="/protos")
        assert _evaluate_condition(cond, VIDEO, {}, {}, sampling=HALF)[0] is False
        assert _evaluate_condition(
            cond, VIDEO, {}, {}, sampling=FrameSampling(0.2, 0.25))[0] is True


class TestPipelineSettings:
    def test_run_pipeline_uses_the_pipeline_ratios(self, frames, classify):
        _person_in(classify, 1)
        hidden = []
        pipeline = ClassifierPipeline(
            name="p",
            nodes=[PipelineNode(
                name="person",
                condition=_rank(),
                on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.HIDE),
                on_no_match=NodeOutcome.accept(),
            )],
            dynamic_content_sample_ratio=0.3,
            dynamic_content_positive_ratio=0.25,
        )
        result = run_pipeline(pipeline, VIDEO, ActionCallbacks(hide_callback=hidden.append))
        assert frames == [(VIDEO, 0.3)]
        assert result == ClassifierActionType.HIDE
        assert hidden == [VIDEO]

    def test_defaults_are_not_serialized(self):
        d = ClassifierPipeline(name="p").to_dict()
        assert "dynamic_content_sample_ratio" not in d
        assert "dynamic_content_positive_ratio" not in d
        restored = ClassifierPipeline.from_dict(d)
        assert restored.frame_sampling() == FrameSampling(DEFAULT_RATIO, DEFAULT_RATIO)

    def test_ratios_round_trip_and_clamp(self):
        p = ClassifierPipeline(name="p", dynamic_content_sample_ratio=0.4,
                               dynamic_content_positive_ratio=0.6)
        assert ClassifierPipeline.from_dict(p.to_dict()).frame_sampling() == FrameSampling(0.4, 0.6)
        clamped = ClassifierPipeline.from_dict({"name": "p", "dynamic_content_sample_ratio": 7,
                                                "dynamic_content_positive_ratio": "bad"})
        assert clamped.frame_sampling() == FrameSampling(1.0, DEFAULT_RATIO)

    def test_prevalidation_pipeline_round_trips_ratios(self):
        from compare.classifier_pipeline import PrevalidationPipeline
        p = PrevalidationPipeline(name="p", dynamic_content_positive_ratio=0.5)
        restored = PrevalidationPipeline.from_dict(p.to_dict())
        assert restored.dynamic_content_positive_ratio == 0.5

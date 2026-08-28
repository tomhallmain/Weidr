"""Unit tests for VarianceFromOriginalCondition.

Covers the dataclass round-trip, validation, and the evaluator's band /
invert / short-circuit behaviour. Embedding computation is stubbed: the
condition's own logic is what's under test, not the embedding models.
"""

import pytest

from compare.classifier_pipeline import ClassifierPipeline
from compare.classifier_pipeline_conditions import VarianceFromOriginalCondition
from compare.classifier_pipeline_nodes import (
    NodeOutcome,
    OutcomeType,
    PipelineNode,
    _condition_from_dict,
)
import compare.classifier_pipeline_runner as runner
from utils.constants import ClassifierActionType


class _FakeMode:
    """Stands in for a CompareMode member; only .name is consulted."""
    name = "CLIP_EMBEDDING"


def _node(condition, name="Variance"):
    return PipelineNode(
        name=name,
        condition=condition,
        on_match=NodeOutcome(
            outcome_type=OutcomeType.EXECUTE_AND_CONTINUE,
            action_type=ClassifierActionType.ADD_MARK,
        ),
        on_no_match=NodeOutcome(outcome_type=OutcomeType.CONTINUE),
    )


class TestSerialization:
    def test_round_trip_preserves_every_field(self):
        original = VarianceFromOriginalCondition(
            min_similarity=0.4, max_similarity=0.8,
            compare_mode="SIGLIP_EMBEDDING", invert=True, match_on_unresolved=True,
        )
        restored = _condition_from_dict(original.to_dict())

        assert isinstance(restored, VarianceFromOriginalCondition)
        assert restored.min_similarity == 0.4
        assert restored.max_similarity == 0.8
        assert restored.compare_mode == "SIGLIP_EMBEDDING"
        assert restored.invert is True
        assert restored.match_on_unresolved is True

    def test_condition_type_is_stable(self):
        assert VarianceFromOriginalCondition().to_dict()["condition_type"] == (
            "variance_from_original"
        )

    def test_defaults_applied_for_a_sparse_dict(self):
        restored = _condition_from_dict({"condition_type": "variance_from_original"})
        assert restored.min_similarity == 0.55
        assert restored.max_similarity == 0.95
        assert restored.compare_mode == "CLIP_EMBEDDING"
        assert restored.invert is False
        assert restored.match_on_unresolved is False


class TestValidation:
    def _errors(self, condition) -> list:
        pipeline = ClassifierPipeline(name="p", nodes=[_node(condition)])
        return pipeline.validate()

    def test_inverted_bounds_rejected(self):
        errors = self._errors(
            VarianceFromOriginalCondition(min_similarity=0.9, max_similarity=0.2)
        )
        assert any("min_similarity" in e for e in errors)

    def test_out_of_range_bounds_rejected(self):
        assert self._errors(VarianceFromOriginalCondition(min_similarity=-0.5))
        assert self._errors(VarianceFromOriginalCondition(max_similarity=1.5))

    def test_non_embedding_mode_rejected(self):
        errors = self._errors(VarianceFromOriginalCondition(compare_mode="SIZE"))
        assert any("SIZE" in e for e in errors)

    def test_valid_condition_has_no_errors(self):
        assert self._errors(VarianceFromOriginalCondition()) == []

    def test_very_wide_band_warns_but_does_not_error(self):
        condition = VarianceFromOriginalCondition(min_similarity=0.0, max_similarity=1.0)
        pipeline = ClassifierPipeline(name="p", nodes=[_node(condition)])
        assert pipeline.validate() == []
        assert any("wide" in w for w in pipeline.validate_warnings())


class TestEvaluator:
    """The evaluator is driven through stubbed seed resolution + embeddings."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        runner._seed_embedding_cache.clear()
        yield
        runner._seed_embedding_cache.clear()

    @pytest.fixture
    def stub(self, monkeypatch, tmp_path):
        """Wire a resolvable seed and a controllable similarity."""
        seed = tmp_path / "img.png"
        seed.write_bytes(b"x")
        state = {"similarity": 0.75, "calls": 0}

        monkeypatch.setattr(
            "files.related_image.get_image_edit_redo_params",
            lambda path: (str(seed), "_fix"),
        )
        monkeypatch.setattr(runner, "_resolve_embedding_mode", lambda name: _FakeMode())

        def _compute(path, mode):
            state["calls"] += 1
            return [1.0, 0.0]

        monkeypatch.setattr("compare.embedding_capture.compute_media_embedding", _compute)
        monkeypatch.setattr(runner, "_cosine_similarity", lambda a, b: state["similarity"])
        return state

    def _eval(self, condition, is_seed=False, path="/x/img_fix.png"):
        return runner._eval_variance_from_original(condition, path, is_seed)

    def test_seed_never_matches_and_resolves_nothing(self, monkeypatch):
        """Short-circuits before touching seed resolution or embeddings."""
        def _boom(*args, **kwargs):
            raise AssertionError("seed short-circuit did not fire")

        monkeypatch.setattr("files.related_image.get_image_edit_redo_params", _boom)
        assert self._eval(VarianceFromOriginalCondition(), is_seed=True) == (False, None)

    def test_inside_band_does_not_fire(self, stub):
        stub["similarity"] = 0.75
        matched, score = self._eval(VarianceFromOriginalCondition())
        assert matched is False
        assert score == 0.75

    def test_too_different_fires(self, stub):
        stub["similarity"] = 0.20
        matched, score = self._eval(VarianceFromOriginalCondition())
        assert matched is True
        assert score == 0.20

    def test_too_similar_fires(self, stub):
        stub["similarity"] = 0.99
        assert self._eval(VarianceFromOriginalCondition())[0] is True

    def test_band_bounds_are_inclusive(self, stub):
        condition = VarianceFromOriginalCondition(min_similarity=0.55, max_similarity=0.95)
        for boundary in (0.55, 0.95):
            stub["similarity"] = boundary
            assert self._eval(condition)[0] is False

    def test_invert_is_the_exact_complement(self, stub):
        plain = VarianceFromOriginalCondition()
        inverted = VarianceFromOriginalCondition(invert=True)
        for similarity in (0.20, 0.55, 0.75, 0.95, 0.99):
            stub["similarity"] = similarity
            assert self._eval(plain)[0] is not self._eval(inverted)[0]

    def test_unresolved_original_respects_the_flag(self, monkeypatch):
        monkeypatch.setattr(
            "files.related_image.get_image_edit_redo_params", lambda path: (None, None)
        )
        assert self._eval(VarianceFromOriginalCondition())[0] is False
        assert self._eval(
            VarianceFromOriginalCondition(match_on_unresolved=True)
        )[0] is True

    def test_unknown_compare_mode_is_unresolved_not_a_crash(self, monkeypatch, tmp_path):
        seed = tmp_path / "img.png"
        seed.write_bytes(b"x")
        monkeypatch.setattr(
            "files.related_image.get_image_edit_redo_params",
            lambda path: (str(seed), "_fix"),
        )
        condition = VarianceFromOriginalCondition(compare_mode="NOT_A_MODE")
        assert self._eval(condition) == (False, None)

    def test_seed_embedding_is_computed_once_per_group(self, stub):
        """Without the cache this is one embedding per derivative, not per seed."""
        condition = VarianceFromOriginalCondition()
        for suffix in ("_fix", "_alt", "_var"):
            self._eval(condition, path=f"/x/img{suffix}.png")
        # 3 derivative embeddings + 1 shared seed embedding.
        assert stub["calls"] == 4


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        assert runner._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert runner._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_magnitude_does_not_matter(self):
        assert runner._cosine_similarity([2.0, 0.0], [9.0, 0.0]) == pytest.approx(1.0)

    def test_zero_vector_is_uncomparable(self):
        assert runner._cosine_similarity([0.0, 0.0], [1.0, 0.0]) is None

    def test_shape_mismatch_is_uncomparable(self):
        assert runner._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) is None

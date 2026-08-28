"""Unit tests for CompareModels' similarity scoring.

Covers the two defects that made a text-based model search return nothing:
the lora dimension scoring a non-overlap as zero (capping any models-only
search below the threshold), and exact-only matching.
"""

import pytest

from compare.compare_models import CompareModels, _dimension_score, model_similarity


class TestDimensionScore:
    def test_unconstrained_dimension_returns_none(self):
        """None is what lets the caller leave the dimension out entirely."""
        assert _dimension_score([], ["anything"]) is None

    def test_no_candidates_scores_zero(self):
        assert _dimension_score(["a"], []) == 0.0

    def test_exact_match_scores_one(self):
        assert _dimension_score(["modelA"], ["modelA"]) == 1.0

    def test_substring_match_scores_one(self):
        assert _dimension_score(["Vision"], ["realisticVisionV51_v51VAE"]) == 1.0

    def test_match_is_case_insensitive(self):
        assert _dimension_score(["realisticvision"], ["realisticVisionV51"]) == 1.0

    def test_unrelated_term_scores_zero(self):
        assert _dimension_score(["dreamshaper"], ["realisticVision"]) == 0.0

    def test_score_is_the_fraction_of_terms_found(self):
        assert _dimension_score(["a", "zzz"], ["aaa"]) == 0.5

    def test_extra_candidate_names_do_not_dilute(self):
        """Scored on terms found, not overlap ratio: a file with many models
        is not penalised for the ones the search never asked about."""
        assert _dimension_score(["a"], ["aaa", "bbb", "ccc"]) == 1.0


class TestLorasNeverExclude:
    """A match on one dimension must not be dragged down by the other."""

    def test_exact_model_match_with_differing_loras_is_a_full_match(self):
        assert model_similarity(["mA"], ["lX"], ["mA"], ["lY"]) == 1.0

    def test_exact_lora_match_with_differing_models_is_a_full_match(self):
        assert model_similarity(["mA"], ["lX"], ["mB"], ["lX"]) == 1.0

    def test_both_matching_is_a_full_match(self):
        assert model_similarity(["mA"], ["lX"], ["mA"], ["lX"]) == 1.0

    def test_neither_matching_scores_zero(self):
        assert model_similarity(["mA"], ["lX"], ["mB"], ["lY"]) == 0.0

    def test_models_only_search_is_not_capped_by_absent_loras(self):
        """The reported bug: a text search carries no loras, so the old fixed
        0.7/0.3 blend capped even an exact match at 0.7."""
        score = model_similarity(["mA"], [], ["mA"], ["lX"])
        assert score == 1.0
        assert score >= CompareModels.THRESHOLD_MATCH

    def test_result_clears_the_modes_own_threshold(self):
        for candidate_loras in ([], ["lX"], ["lX", "lY"]):
            assert model_similarity(["mA"], [], ["mA"], candidate_loras) >= (
                CompareModels.THRESHOLD_MATCH
            )


class TestModelSimilarityEdges:
    def test_both_sides_empty_is_a_full_match(self):
        """Preserved: two files that both carry no models group together."""
        assert model_similarity([], [], [], []) == 1.0

    def test_empty_search_side_scores_zero(self):
        assert model_similarity([], [], ["mA"], ["lX"]) == 0.0

    def test_empty_candidate_side_scores_zero(self):
        assert model_similarity(["mA"], ["lX"], [], []) == 0.0

    def test_partial_term_match_scores_between(self):
        score = model_similarity(["mA", "zzz"], [], ["mA"], [])
        assert score == 0.5

    @pytest.mark.parametrize(
        "search,expected",
        [
            (["realisticVisionV51_v51VAE"], 1.0),   # exact stem
            (["realisticVision"], 1.0),             # leading substring
            (["VisionV51"], 1.0),                   # interior substring
            (["REALISTICVISION"], 1.0),             # case-insensitive
            (["dreamshaper"], 0.0),                 # unrelated
        ],
    )
    def test_text_search_against_a_real_checkpoint_stem(self, search, expected):
        assert model_similarity(
            search, [], ["realisticVisionV51_v51VAE"], ["detailTweaker"]
        ) == expected


class TestThresholdScale:
    def test_models_mode_keeps_its_own_threshold(self):
        """MODELS scores are not on the embedding scale; applying the
        embedding threshold (0.9) to them rejected every result."""
        from compare.compare_args import CompareArgs
        from compare.compare_manager import CompareManager
        from utils.config import config
        from utils.constants import CompareMode

        assert CompareModels.THRESHOLD_MATCH < config.embedding_similarity_threshold

        manager = CompareManager.__new__(CompareManager)
        manager._threshold = config.embedding_similarity_threshold
        manager._counter_limit = None
        manager._primary_mode = CompareMode.MODELS

        args = CompareArgs()
        try:
            manager.apply_settings_to_args(args)
        except AttributeError as e:
            pytest.skip(f"CompareManager needs more state than this stub provides: {e}")
        # An explicitly-set embedding threshold must not carry over to MODELS.
        assert args.threshold == CompareModels.THRESHOLD_MATCH

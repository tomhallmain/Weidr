"""
Unit tests for image/classifier_utils.py: output-to-probability conversion,
mapping scores onto configured categories (including the single-output binary
case and count mismatches), top-category selection and split-positive
assignment, plus the wrappers' use of them on mismatched model output.
"""
from __future__ import annotations

import numpy as np
import pytest

from image.classifier_utils import (
    ensure_probabilities,
    format_prediction_line,
    logits_to_probabilities,
    map_scores_to_categories,
    pick_split_positive,
    sigmoid,
    softmax,
    top_category,
)
from image.image_classifier import ImageClassifierWrapper
from image.image_classifier_model_config import ImageClassifierModelConfig


class TestLogitsToProbabilities:
    def test_multi_class_gets_softmax(self):
        out = logits_to_probabilities(np.array([[2.0, -1.0, 0.5]]))
        assert np.isclose(out.sum(), 1.0)
        assert np.argmax(out[0]) == 0

    def test_single_logit_gets_sigmoid_not_softmax(self):
        out = logits_to_probabilities(np.array([[2.0]]))
        assert np.allclose(out, sigmoid(np.array([[2.0]])))
        assert not np.isclose(out[0, 0], 1.0)


class TestEnsureProbabilities:
    def test_values_in_unit_range_pass_through(self):
        x = np.array([[0.2, 0.8]])
        assert np.allclose(ensure_probabilities(x), x)

    def test_multi_class_logits_get_softmax(self):
        out = ensure_probabilities(np.array([[3.0, -2.0]]))
        assert np.allclose(out, softmax(np.array([[3.0, -2.0]]), axis=1))

    def test_single_logit_gets_sigmoid(self):
        out = ensure_probabilities(np.array([[-4.0]]))
        assert 0.0 < out[0, 0] < 0.5

    def test_single_probability_passes_through(self):
        assert np.allclose(ensure_probabilities(np.array([[0.7]])), [[0.7]])


class TestMapScoresToCategories:
    def test_matching_counts_pair_by_position(self):
        assert map_scores_to_categories([0.1, 0.9], ["a", "b"], "m") == pytest.approx({"a": 0.1, "b": 0.9})

    def test_single_score_with_two_categories_expands_to_binary(self):
        result = map_scores_to_categories(np.array([0.8]), ["neg", "pos"], "m")
        assert result == pytest.approx({"neg": 0.2, "pos": 0.8})

    def test_fewer_outputs_map_only_overlap(self):
        assert map_scores_to_categories([0.3, 0.7], ["a", "b", "c"], "m") == pytest.approx({"a": 0.3, "b": 0.7})

    def test_more_outputs_map_only_overlap(self):
        assert map_scores_to_categories([0.1, 0.2, 0.7], ["a", "b"], "m") == pytest.approx({"a": 0.1, "b": 0.2})

    def test_single_score_with_one_category_is_not_expanded(self):
        assert map_scores_to_categories([0.4], ["only"], "m") == pytest.approx({"only": 0.4})


class TestTopCategory:
    def test_highest_score_wins(self):
        assert top_category({"a": 0.1, "b": 0.9}, ["a", "b"]) == "b"

    def test_tie_goes_to_earliest_category(self):
        assert top_category({"a": 0.5, "b": 0.5}, ["b", "a"]) == "b"

    def test_unscored_categories_are_skipped(self):
        assert top_category({"a": 0.2}, ["a", "b", "c"]) == "a"

    def test_no_scored_category_raises_value_error(self):
        with pytest.raises(ValueError):
            top_category({}, ["a"])


class TestPickSplitPositive:
    def test_group_beating_neutral_assigns_highest_member(self):
        result = pick_split_positive(
            {"safe": 0.4, "mild": 0.35, "explicit": 0.25},
            [["mild", "explicit"]], ["safe"], [], 0.05,
        )
        assert result == ("mild", ["mild", "explicit"], pytest.approx(0.6))

    def test_severity_order_takes_precedence(self):
        result = pick_split_positive(
            {"safe": 0.1, "mild": 0.6, "explicit": 0.3},
            [["mild", "explicit"]], ["safe"], ["explicit", "mild"], 0.05,
        )
        assert result[0] == "explicit"

    def test_margin_not_exceeded_returns_none(self):
        assert pick_split_positive(
            {"safe": 0.48, "mild": 0.26, "explicit": 0.26},
            [["mild", "explicit"]], ["safe"], [], 0.05,
        ) is None

    def test_single_member_groups_are_ignored(self):
        assert pick_split_positive({"a": 0.9, "b": 0.1}, [["a"]], ["b"], [], 0.05) is None


def test_format_prediction_line_orders_by_score():
    assert format_prediction_line({"a": 0.1, "b": 0.9}) == "b=0.900000, a=0.100000"


class _StubBackend:
    def __init__(self, output):
        self.output = np.array(output)
        self.is_loaded = True

    def predict_image(self, image_path):
        return self.output


def _wrapper_with_output(tmp_path, categories, output):
    """(wrapper whose backend returns *output*, image path). Config validation runs
    against a real file but the backend itself is stubbed."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"stub")
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"stub")
    wrapper = ImageClassifierWrapper(ImageClassifierModelConfig(
        model_name="m", model_location=str(model_path), model_categories=categories,
    ))
    wrapper.classifier = _StubBackend(output)
    wrapper.can_run = True
    return wrapper, str(image_path)


class TestWrapperCountMismatch:
    def test_fewer_outputs_than_categories_does_not_raise(self, tmp_path):
        wrapper, image = _wrapper_with_output(tmp_path, ["a", "b", "c"], [[0.3, 0.7]])
        assert wrapper.classify_image(image) == "b"

    def test_unscored_category_threshold_test_is_false(self, tmp_path):
        wrapper, image = _wrapper_with_output(tmp_path, ["a", "b", "c"], [[0.3, 0.7]])
        assert wrapper.test_image_for_category(image, "c", 0.0) is False

    def test_single_output_binary_model(self, tmp_path):
        wrapper, image = _wrapper_with_output(tmp_path, ["no", "yes"], [[0.2]])
        assert wrapper.predict_image(image) == pytest.approx({"no": 0.8, "yes": 0.2})
        assert wrapper.classify_image(image) == "no"

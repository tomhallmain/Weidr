"""Score handling shared by the image and audio classifier wrappers and backends:
raw model output to probabilities, probabilities to per-category scores, and
per-category scores to a single resulting category."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.logging_setup import get_logger

logger = get_logger("classifier_utils")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic function."""
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Probabilities from logits shaped [batch, N, ...]: softmax over axis 1, or a
    sigmoid when N == 1, where the single value is a binary model's positive logit
    (a softmax over one value is always 1.0)."""
    logits = np.asarray(logits)
    if logits.ndim >= 2 and logits.shape[1] == 1:
        return sigmoid(logits)
    return softmax(logits, axis=1)


def ensure_probabilities(output: np.ndarray) -> np.ndarray:
    """*output* as-is when every value is already in [0, 1]; otherwise it is treated
    as logits (see logits_to_probabilities)."""
    output = np.asarray(output)
    if np.all(output >= 0) and np.all(output <= 1):
        return output
    return logits_to_probabilities(output)


def map_scores_to_categories(
    scores: Sequence[float],
    categories: Sequence[str],
    model_name: str,
    log=logger,
) -> Dict[str, float]:
    """Pair one row of model output with *categories* by position.

    A single score with two categories is a binary model's positive-class
    probability and becomes ``[1 - p, p]``, so the negative category goes first.
    Any other count mismatch logs a warning and maps only the overlapping
    positions; categories past the end of the output get no score.
    """
    flat = np.asarray(scores).reshape(-1)
    if len(flat) == 1 and len(categories) == 2:
        p = float(flat[0])
        return {categories[0]: 1.0 - p, categories[1]: p}
    if len(flat) != len(categories):
        log.warning(f"Model {model_name!r} outputs {len(flat)} classes but expected {len(categories)}")
    return {category: float(score) for category, score in zip(categories, flat)}


def top_category(classed_predictions: Dict[str, float], categories: Iterable[str]) -> str:
    """Highest-scoring category among *categories* that received a score; ties go
    to the earliest in *categories*."""
    scored = [c for c in categories if c in classed_predictions]
    if not scored:
        raise ValueError("Model output did not score any configured category")
    return max(scored, key=lambda c: classed_predictions[c])


def derive_neutral_categories_from_positive_groups(
    model_categories: List[str],
    positive_groups: List[List[str]],
) -> List[str]:
    """Categories not present in any positive group (complement of the union of groups)."""
    positive_categories: set[str] = set()
    for group in positive_groups:
        positive_categories.update(group)
    return [cat for cat in model_categories if cat not in positive_categories]


def pick_split_positive(
    classed_predictions: Dict[str, float],
    positive_groups: List[List[str]],
    neutral_categories: List[str],
    severity_order: List[str],
    margin: float,
) -> Optional[Tuple[str, List[str], float]]:
    """Split-positive assignment: among positive groups of two or more categories,
    the one whose combined score beats the aggregate neutral score by more than
    *margin* and is highest. Returns (assigned category, group, combined score), or
    None when no group qualifies. The assigned category is the first in
    *severity_order* that belongs to the group and scored above 0, else the group's
    highest-scoring category."""
    neutral_prob = sum(classed_predictions.get(cat, 0) for cat in neutral_categories)
    best: Optional[Tuple[str, List[str], float]] = None
    best_combined_prob = 0.0

    for group_cats in positive_groups:
        if len(group_cats) <= 1:
            continue
        combined_prob = sum(classed_predictions.get(cat, 0) for cat in group_cats)
        if combined_prob > neutral_prob + margin and combined_prob > best_combined_prob:
            best_combined_prob = combined_prob
            picked: Optional[str] = None
            for severe_cat in severity_order or []:
                if severe_cat in group_cats and classed_predictions.get(severe_cat, 0) > 0:
                    picked = severe_cat
                    break
            if not picked:
                picked = max(
                    ((cat, classed_predictions.get(cat, 0)) for cat in group_cats),
                    key=lambda x: x[1],
                )[0]
            best = (picked, group_cats, combined_prob)

    if best is None or not best[0]:
        return None
    return best


def format_prediction_line(classed_predictions: Dict[str, float]) -> str:
    """One-line ``name=score`` listing, highest score first, for debug logs."""
    ordered_pairs = sorted(classed_predictions.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{name}={score:.6f}" for name, score in ordered_pairs)

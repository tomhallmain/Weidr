"""Category-set helpers shared by classifier actions and compare filters."""

from __future__ import annotations

from typing import Iterable, Optional


def model_strategy_positive_categories(positive_groups: Optional[Iterable]) -> frozenset[str]:
    """Union of a classifier model config's ``positive_groups`` -- the categories
    model-strategy classification counts as a match. Entries that are not
    non-empty lists/tuples are skipped."""
    positives: set[str] = set()
    for group in positive_groups or []:
        if isinstance(group, (list, tuple)) and group:
            positives.update(group)
    return frozenset(positives)

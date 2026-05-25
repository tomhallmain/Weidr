"""
Shared checks for iterative vs matrix embedding group compare.

Used by ``test_embedding_matrix_equivalence.py`` (pytest) and mirrors the
validation in ``tests/test_compare_embedding_matrix.py`` (manual script).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from compare.compare_embeddings_clip import CompareEmbeddingClip
from tests.analysis import (
    convert_matrix_to_roll_index_output,
    reverse_table_row_order,
)


def similarity_matrix_list(embeddings: np.ndarray) -> list[list[float]]:
    """Full pairwise dot-product matrix (embeddings are L2-normalized)."""
    return (embeddings @ embeddings.T).tolist()


def iterative_roll_similarity_table(compare: CompareEmbeddingClip) -> list[list[float]]:
    """One row per roll index i (i >= 1): dot(emb[k], emb[(k-i) % n]) for all k."""
    n = compare.compare_data.n_files_found
    rows: list[list[float]] = []
    for i in range(1, n):
        _, diff_scores = compare._compute_iterative_similarities(i)
        rows.append([float(diff_scores[k]) for k in range(n)])
    return rows


def group_count(compare: CompareEmbeddingClip) -> int:
    return sum(1 for g in compare.compare_result.file_groups.values() if len(g) >= 2)


def files_in_multi_file_groups(compare: CompareEmbeddingClip) -> int:
    return sum(len(g) for g in compare.compare_result.file_groups.values() if len(g) >= 2)


def each_group_is_single_cluster(
    compare: CompareEmbeddingClip,
    cluster_by_path: dict,
) -> bool:
    """True when no similarity group mixes files from different fixture clusters."""
    for group in compare.compare_result.file_groups.values():
        if len(group) < 2:
            continue
        clusters = {cluster_by_path[path] for path in group}
        if len(clusters) != 1:
            return False
    return True


def summarize_table_diff(
    left: list[list[float]],
    right: list[list[float]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Elementwise stats for two tables with identical shape."""
    if not left or not right:
        return {
            "rows": 0,
            "cols": 0,
            "max_abs_diff": 0.0,
            "cells_above_tolerance": 0,
            "total_cells": 0,
            "tolerance": tolerance,
            "scores_consistent": True,
        }
    rows, cols = len(left), len(left[0])
    if len(right) != rows or len(right[0]) != cols:
        raise ValueError(
            f"Table shape mismatch: left {rows}x{cols}, "
            f"right {len(right)}x{len(right[0]) if right else 0}"
        )

    max_abs = 0.0
    above = 0
    for i in range(rows):
        for j in range(cols):
            diff = abs(left[i][j] - right[i][j])
            max_abs = max(max_abs, diff)
            if diff > tolerance:
                above += 1

    return {
        "rows": rows,
        "cols": cols,
        "max_abs_diff": max_abs,
        "cells_above_tolerance": above,
        "total_cells": rows * cols,
        "tolerance": tolerance,
        "scores_consistent": above == 0,
    }


def roll_tables_agree(
    compare: CompareEmbeddingClip,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """
    Compare iterative roll-index scores to layout derived from the full matrix.

    Returns stats dict including ``scores_consistent``.
    """
    embeddings = compare._file_embeddings
    matrix_full = similarity_matrix_list(embeddings)
    roll_iter = iterative_roll_similarity_table(compare)
    roll_from_full = convert_matrix_to_roll_index_output(matrix_full)
    roll_iter_display = reverse_table_row_order(roll_iter)
    return summarize_table_diff(roll_iter_display, roll_from_full, tolerance=tolerance)

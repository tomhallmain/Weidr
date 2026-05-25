"""
Manual regression script: iterative vs matrix embedding group compare.

Run from repo root (not collected by pytest — see tests/conftest.py collect_ignore):

    python tests/test_compare_embedding_matrix.py

Compare checkpoints live under the tested directory as
``weidr_result_clip_embedding.pkl`` (see compare/compare_result.py).

After both compare paths run, this script checks that:
  - iterative and matrix roll-index score tables match (within float tolerance), and
  - optional group counts from each path are reported.

Artifacts are written to ``tests/output/``; see the printed validation summary
(values near zero in the diff file are floating-point noise, not failures).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from compare.compare_args import CompareArgs
from compare.compare_embeddings_clip import CompareEmbeddingClip
from compare.compare_result import CompareResult
from tests.analysis import (
    convert_matrix_to_roll_index_output,
    print_formatted_table,
    reverse_table_row_order,
    table_elementwise_subtraction,
)

_REPO_TESTS = Path(__file__).resolve().parent
_OUTPUT_DIR = _REPO_TESTS / "output"

# Roll-index layout agreement (iterative reversed vs matrix-derived layout).
SCORE_TABLE_TOLERANCE = 1e-5

# Skip printing full score tables to the console above this file count.
PRINT_TABLE_MAX_FILES = 40


def _similarity_matrix_list(embeddings: np.ndarray) -> list[list[float]]:
    """Full pairwise dot-product matrix (CLIP embeddings are L2-normalized)."""
    return (embeddings @ embeddings.T).tolist()


def _iterative_roll_similarity_table(compare: CompareEmbeddingClip) -> list[list[float]]:
    """
    One row per roll index i (i >= 1): dot(emb[k], emb[(k-i) % n]) for all k.
    """
    n = compare.compare_data.n_files_found
    rows: list[list[float]] = []
    for i in range(1, n):
        _, diff_scores = compare._compute_iterative_similarities(i)
        rows.append([float(diff_scores[k]) for k in range(n)])
    return rows


def _group_count(compare: CompareEmbeddingClip) -> int:
    return sum(1 for g in compare.compare_result.file_groups.values() if len(g) >= 2)


def _summarize_table_diff(
    left: list[list[float]],
    right: list[list[float]],
    *,
    tolerance: float = SCORE_TABLE_TOLERANCE,
) -> dict[str, Any]:
    """Elementwise stats for two tables with identical shape."""
    if not left or not right:
        return {
            "rows": 0,
            "cols": 0,
            "max_abs_diff": 0.0,
            "cells_above_tolerance": 0,
            "total_cells": 0,
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
    total = rows * cols
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
        "total_cells": total,
        "tolerance": tolerance,
        "scores_consistent": above == 0,
    }


def _print_validation_report(
    *,
    n_files: int,
    iter_groups: int,
    matrix_groups: int,
    embeddings_match: bool,
    score_stats: dict[str, Any],
) -> None:
    print("\n--- Validation summary ---")
    print(f"Files compared: {n_files}")
    print(f"Embeddings array identical (iter vs matrix): {embeddings_match}")
    print(f"Groups (>=2 files) — iterative: {iter_groups}, matrix: {matrix_groups}")
    if iter_groups != matrix_groups:
        print("  NOTE: Group counts differ; score tables can still match while grouping logic diverges.")
    print(
        f"Roll-index score tables: {score_stats['rows']} x {score_stats['cols']} "
        f"({score_stats['total_cells']} cells)"
    )
    print(f"  max |diff|: {score_stats['max_abs_diff']:.6g}")
    print(f"  cells with |diff| > {score_stats['tolerance']}: {score_stats['cells_above_tolerance']}")
    if score_stats["scores_consistent"]:
        print("  PASS: iterative and matrix-derived scores agree within tolerance.")
    else:
        print("  FAIL: score tables differ beyond tolerance — inspect comparison diff JSON.")
    print("---\n")


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path}")


def _verify_written_diff(path: Path, tolerance: float = SCORE_TABLE_TOLERANCE) -> dict[str, Any]:
    """Light check that saved diff JSON matches in-memory expectations."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return {"rows": 0, "cols": 0, "max_abs_diff": 0.0, "file_consistent": True}
    flat = [v for row in data for v in row]
    max_abs = max(abs(v) for v in flat)
    return {
        "rows": len(data),
        "cols": len(data[0]),
        "max_abs_diff": max_abs,
        "file_consistent": max_abs <= tolerance,
    }


def test_embeddings_matrix() -> None:
    directory_to_test = input("Enter directory to test: ").strip().strip('"')
    if not os.path.isdir(directory_to_test):
        raise FileNotFoundError(f"{directory_to_test} does not exist!")

    base_dir = os.path.abspath(directory_to_test)
    args = CompareArgs(base_dir=base_dir)
    args_iter = args.clone()
    args_matrix = args.clone()
    args_matrix.use_matrix_comparison = True

    compare_iter = CompareEmbeddingClip(args=args_iter)
    compare_matrix = CompareEmbeddingClip(args=args_matrix)

    for comp in (compare_iter, compare_matrix):
        comp.get_files()
        comp.get_data()

    n = compare_iter.compare_data.n_files_found
    if n != compare_matrix.compare_data.n_files_found:
        raise RuntimeError(
            f"File counts differ: iterative={n}, matrix={compare_matrix.compare_data.n_files_found}"
        )
    print(f"Comparing {n} files under {base_dir}")

    compare_iter.run_comparison()
    compare_matrix.run_comparison()

    iter_groups = _group_count(compare_iter)
    matrix_groups = _group_count(compare_matrix)
    print(f"Iterative path: {iter_groups} groups (>=2 files)")
    print(f"Matrix path:    {matrix_groups} groups (>=2 files)")

    pkl_path = CompareResult.cache_path(base_dir, CompareEmbeddingClip.COMPARE_MODE)
    if os.path.isfile(pkl_path):
        print(f"Compare checkpoint (if store_checkpoints used): {pkl_path}")

    embeddings_match = np.allclose(
        compare_iter._file_embeddings,
        compare_matrix._file_embeddings,
        rtol=0,
        atol=0,
    )

    embeddings = compare_iter._file_embeddings
    matrix_full = _similarity_matrix_list(embeddings)
    roll_iter = _iterative_roll_similarity_table(compare_iter)
    roll_from_full = convert_matrix_to_roll_index_output(matrix_full)

    # Legacy analysis.py compared reversed iterative rows to matrix-as-roll layout.
    roll_iter_display = reverse_table_row_order(roll_iter)
    comparison = table_elementwise_subtraction(roll_iter_display, roll_from_full)

    score_stats = _summarize_table_diff(roll_iter_display, roll_from_full)
    _print_validation_report(
        n_files=n,
        iter_groups=iter_groups,
        matrix_groups=matrix_groups,
        embeddings_match=bool(embeddings_match),
        score_stats=score_stats,
    )

    if n <= PRINT_TABLE_MAX_FILES:
        print_formatted_table(
            roll_iter_display,
            title="Iterative roll-index scores (reversed row order)",
        )
        print_formatted_table(roll_from_full, title="Full matrix converted to roll-index layout")
        print_formatted_table(comparison, title="Difference (iterative − matrix layout)")
    else:
        print(
            f"Skipping console tables ({n} files > {PRINT_TABLE_MAX_FILES}); "
            "see tests/output/*.json and validation summary above."
        )

    diff_path = _OUTPUT_DIR / "embeddings_comparison_diff.json"
    _save_json(_OUTPUT_DIR / "embeddings_iterative_roll.json", roll_iter)
    _save_json(_OUTPUT_DIR / "embeddings_matrix_full.json", matrix_full)
    _save_json(_OUTPUT_DIR / "embeddings_matrix_as_roll.json", roll_from_full)
    _save_json(diff_path, comparison)

    file_check = _verify_written_diff(diff_path)
    print(
        f"Saved diff file check: max |diff| = {file_check['max_abs_diff']:.6g} "
        f"({'OK' if file_check['file_consistent'] else 'unexpected'})"
    )


if __name__ == "__main__":
    test_embeddings_matrix()

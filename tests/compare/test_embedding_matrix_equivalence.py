"""
Regression tests: iterative vs matrix embedding **group compare** (CLIP path).

Production background
---------------------
When ``CompareArgs.use_matrix_comparison`` is true (the default), group compare
uses ``BaseCompareEmbedding._compute_matrix_similarities()`` — chunked matrix
multiply over all L2-normalized embeddings, emitting sparse upper-triangle pairs
above the similarity threshold. When false, the legacy path rolls the embedding
matrix with ``np.roll`` and scores each offset separately (``_compute_iterative_similarities``).

Both paths feed the same grouping logic (``_process_similarity_results``), but
they must agree on **which pairs are similar and with what score**, or users
would see different groups depending on a checkbox.

What this module checks
-----------------------
We run **both** paths on the same fixed library and assert:

1. **Score-table equivalence** (primary regression): pairwise dot products are
   rearranged into a *roll-index table* — one row per roll offset ``i >= 1``,
   column ``k`` holds ``dot(emb[k], emb[(k-i) % n])``. The iterative path builds
   this table directly; the matrix path builds a full ``n×n`` similarity matrix
   and converts it to roll-index layout via ``tests.analysis.convert_matrix_to_roll_index_output``
   (same helper as the manual script). Tables must match within ``SCORE_TABLE_TOLERANCE``
   (``1e-5``). Tiny diffs (~1e-15) are normal float noise; see the manual script’s
   validation summary for real-directory runs.

2. **Grouping equivalence**: multi-file group **count** matches between paths, and
   no group mixes files from different fixture clusters (clusters must not merge).

3. **Fixture geometry** (fast pre-check): synthetic embeddings are built so
   within-cluster cosine similarity ≥ 0.9 and cross-cluster / outlier pairs stay below 0.9.

What we deliberately do *not* test here
---------------------------------------
- Real CLIP inference (weights are not loaded).
- Checkpoint pickle I/O or ``tests/output/*.json`` artifacts (see manual script).
- Composite multi-mode compare or color/size modes.
- That group counts match a specific number on real photos (synthetic data only).

Fixture and isolation
---------------------
``tests.fixtures.embedding_matrix_fixtures`` creates **26** tiny PNGs and a catalog
of precomputed 512-D vectors (three tight clusters + one outlier family). CLIP is
patched at ``compare.compare_embeddings_clip.image_embeddings_clip`` so
``get_data()`` loads fixture vectors without GPU work. ``stable_chunk_ram`` pins RAM
budget so chunked similarity chunk sizes are deterministic in CI.

Manual counterpart
------------------
``tests/test_compare_embedding_matrix.py`` is the same validation against a
user-chosen directory (interactive ``input()``, writes JSON under ``tests/output/``,
excluded from pytest via ``collect_ignore``). Use that for one-off checks on large
libraries; use **this module** in the automated suite.

Test classes
------------
``TestEmbeddingMatrixEquivalence``
    End-to-end compare runs and score/group assertions.

``TestEmbeddingMatrixCatalogBuilder``
    Fixture builder reproducibility only.

Helpers live in ``tests.compare.embedding_matrix_validation`` (shared roll-table
and group checks, usable from other tests if needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from compare.compare_args import CompareArgs
from compare.compare_embeddings_clip import CompareEmbeddingClip
from tests.compare.embedding_matrix_validation import (
    each_group_is_single_cluster,
    files_in_multi_file_groups,
    group_count,
    roll_tables_agree,
)
from tests.fixtures.embedding_matrix_fixtures import (
    EMBEDDING_SIMILARITY_THRESHOLD,
    N_FILES,
    SCORE_TABLE_TOLERANCE,
    EmbeddingMatrixCatalog,
    build_embedding_matrix_catalog,
    fake_image_embeddings_clip,
)


@pytest.fixture
def stable_chunk_ram(monkeypatch):
    monkeypatch.setattr(
        "compare.base_compare.Utils.calculate_available_ram",
        lambda: 8.0,
    )


@pytest.fixture
def patched_clip_embeddings(monkeypatch, embedding_matrix_catalog):
    """Use fixture embeddings instead of loading CLIP weights."""

    def _fake(path: str) -> np.ndarray:
        return fake_image_embeddings_clip(path, embedding_matrix_catalog)

    monkeypatch.setattr(
        "compare.compare_embeddings_clip.image_embeddings_clip",
        _fake,
    )
    return embedding_matrix_catalog


def _run_both_paths(catalog: EmbeddingMatrixCatalog):
    args = CompareArgs(
        base_dir=catalog.dir,
        overwrite=True,
        store_checkpoints=False,
        compare_threshold=EMBEDDING_SIMILARITY_THRESHOLD,
    )
    args_iter = args.clone()
    args_matrix = args.clone()
    args_matrix.use_matrix_comparison = True

    compare_iter = CompareEmbeddingClip(args=args_iter)
    compare_matrix = CompareEmbeddingClip(args=args_matrix)

    for comp in (compare_iter, compare_matrix):
        comp.get_files()
        comp.get_data()

    n = compare_iter.compare_data.n_files_found
    assert n == catalog.n_files == N_FILES
    assert compare_matrix.compare_data.n_files_found == n

    compare_iter.run_comparison()
    compare_matrix.run_comparison()
    return compare_iter, compare_matrix


class TestEmbeddingMatrixEquivalence:
    def test_roll_index_score_tables_match(
        self,
        patched_clip_embeddings,
        stable_chunk_ram,
    ):
        catalog = patched_clip_embeddings
        compare_iter, compare_matrix = _run_both_paths(catalog)

        assert np.allclose(
            compare_iter._file_embeddings,
            compare_matrix._file_embeddings,
            rtol=0,
            atol=0,
        )

        stats = roll_tables_agree(compare_iter, tolerance=SCORE_TABLE_TOLERANCE)
        assert stats["total_cells"] > 0
        assert stats["scores_consistent"], (
            f"max |diff|={stats['max_abs_diff']}, "
            f"cells above {stats['tolerance']}: {stats['cells_above_tolerance']}"
        )

    def test_group_counts_match_and_some_files_group(
        self,
        patched_clip_embeddings,
        stable_chunk_ram,
    ):
        catalog = patched_clip_embeddings
        compare_iter, compare_matrix = _run_both_paths(catalog)

        iter_groups = group_count(compare_iter)
        matrix_groups = group_count(compare_matrix)
        assert iter_groups == matrix_groups
        assert iter_groups >= 3, "expected separate groups per cluster family"
        assert iter_groups <= 6, "should not fragment into one group per file"
        assert each_group_is_single_cluster(compare_iter, catalog.cluster_by_path)
        assert each_group_is_single_cluster(compare_matrix, catalog.cluster_by_path)

        grouped_files = files_in_multi_file_groups(compare_iter)
        assert grouped_files >= 18, "cluster members should be grouped"

    def test_catalog_intra_and_inter_cluster_similarity(
        self,
        embedding_matrix_catalog,
    ):
        """Sanity-check fixture geometry before running compare."""
        catalog = embedding_matrix_catalog
        paths = catalog.paths
        emb = np.stack([catalog.embeddings_by_path[p] for p in paths])
        sim = emb @ emb.T

        def _paths_for(cluster: str) -> list[int]:
            return [i for i, p in enumerate(paths) if catalog.cluster_by_path[p] == cluster]

        for cluster in ("cluster_a", "cluster_b", "cluster_c"):
            idx = _paths_for(cluster)
            for i in idx:
                for j in idx:
                    if i != j:
                        assert sim[i, j] >= EMBEDDING_SIMILARITY_THRESHOLD

        a0, b0, o0 = _paths_for("cluster_a")[0], _paths_for("cluster_b")[0], _paths_for("outlier")[0]
        assert sim[a0, b0] < EMBEDDING_SIMILARITY_THRESHOLD
        assert sim[a0, o0] < EMBEDDING_SIMILARITY_THRESHOLD
        assert sim[b0, o0] < EMBEDDING_SIMILARITY_THRESHOLD


class TestEmbeddingMatrixCatalogBuilder:
    def test_build_is_deterministic(self, tmp_path):
        d = str(tmp_path)
        c1 = build_embedding_matrix_catalog(d)
        c2 = build_embedding_matrix_catalog(d)
        assert c1.n_files == c2.n_files == N_FILES
        for path in c1.paths:
            assert np.allclose(
                c1.embeddings_by_path[path],
                c2.embeddings_by_path[path],
            )

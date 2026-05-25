"""
Unit tests for embedding pairwise similarity helpers in compare/base_compare.py.

These back ``BaseCompare.chunked_similarity_*``, used by the optional
``use_matrix_comparison`` path in ``BaseCompareEmbedding`` (default on; toggle in Compare Settings).
Normal embedding group compare uses the iterative ``np.roll`` path instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from compare.base_compare import BaseCompare


def _pair_set(pairs, *, decimals: int = 9):
    """Normalize (i, j, sim) triples for order-independent comparison."""
    return {(int(i), int(j), round(float(s), decimals)) for i, j, s in pairs}


class TestCalculateChunkSize:
    def test_returns_at_least_one_row(self):
        emb = np.zeros((100, 512), dtype=np.float32)
        assert BaseCompare.calculate_chunk_size(emb, max_mem_gb=0.001) >= 1

    def test_scales_with_memory_budget(self):
        emb = np.zeros((200, 64), dtype=np.float32)
        small = BaseCompare.calculate_chunk_size(emb, max_mem_gb=0.5)
        large = BaseCompare.calculate_chunk_size(emb, max_mem_gb=4.0)
        assert large >= small


class TestChunkedSimilarityEquivalence:
    """Loop vs vectorized implementations must find the same upper-triangle pairs."""

    @pytest.fixture(autouse=True)
    def _stable_ram_budget(self, monkeypatch):
        # Chunked helpers use half of this value; fixed RAM → deterministic chunking.
        monkeypatch.setattr(
            "compare.base_compare.Utils.calculate_available_ram",
            lambda: 8.0,
        )

    def test_random_normalized_embeddings_agree(self):
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((40, 32)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        loop = BaseCompare.chunked_similarity(emb, threshold=0.85)
        vec = BaseCompare.chunked_similarity_vectorized(emb, threshold=0.85)
        assert _pair_set(loop) == _pair_set(vec)

    def test_identical_rows_produce_unit_similarity_pairs(self):
        emb = np.ones((3, 16), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        loop = BaseCompare.chunked_similarity(emb, threshold=0.5)
        vec = BaseCompare.chunked_similarity_vectorized(emb, threshold=0.5)
        expected = _pair_set([(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)])

        assert _pair_set(loop) == expected
        assert _pair_set(vec) == expected

    def test_no_pairs_when_threshold_above_one(self):
        emb = np.eye(4, dtype=np.float32)
        assert BaseCompare.chunked_similarity(emb, threshold=1.1) == []
        assert BaseCompare.chunked_similarity_vectorized(emb, threshold=1.1) == []

    def test_upper_triangle_only_no_self_pairs(self):
        emb = np.eye(5, dtype=np.float32)
        pairs = BaseCompare.chunked_similarity_vectorized(emb, threshold=0.5)
        for i, j, _ in pairs:
            assert i < j


class TestComputeMatrixSimilarities:
    def test_returns_sparse_triples_not_ndarray(self, monkeypatch):
        from compare.base_compare_embedding import BaseCompareEmbedding
        from compare.compare_args import CompareArgs

        monkeypatch.setattr(
            "compare.base_compare.Utils.calculate_available_ram",
            lambda: 8.0,
        )

        comp = BaseCompareEmbedding.__new__(BaseCompareEmbedding)
        comp.args = CompareArgs()
        comp.embedding_similarity_threshold = 0.5
        # Identical normalized rows → cosine similarity 1.0 (not np.eye, which is orthogonal).
        emb = np.ones((4, 16), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        comp._file_embeddings = emb

        pairs = comp._compute_matrix_similarities()
        assert isinstance(pairs, list)
        assert pairs
        for i, j, score in pairs:
            assert i < j
            assert score >= 0.5

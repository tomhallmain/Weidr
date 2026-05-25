"""
Cosine similarity conventions used by embedding compare.

Production paths:
  - ``BaseCompareEmbedding._compute_embedding_diff`` — dot product on L2-normalized rows
  - ``compare.model.embedding_similarity`` — ``F.cosine_similarity`` on torch tensors
  - ``EmbeddingPrototype.compare_embeddings_with_prototype`` — dot on normalized rows
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from compare.model import embedding_similarity


class TestTorchCosineConvention:
    """Documents that F.cosine_similarity normalizes internally (temp_test_cosine_similarity)."""

    def test_unnormalized_matches_pre_normalized(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)

        raw = F.cosine_similarity(x, y, dim=0)
        x_n = x / torch.norm(x)
        y_n = y / torch.norm(y)
        pre_norm = F.cosine_similarity(x_n, y_n, dim=0)

        assert torch.allclose(raw, pre_norm, atol=1e-6)


class TestNumpyDotMatchesCosine:
    """Normalized dot product is what matrix/iterative embedding compare uses."""

    def test_dot_equals_cosine_for_unit_vectors(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        a_u = a / np.linalg.norm(a)
        b_u = b / np.linalg.norm(b)
        dot = float(np.dot(a_u, b_u))

        ta = torch.tensor(a_u)
        tb = torch.tensor(b_u)
        cos = float(F.cosine_similarity(ta, tb, dim=0))

        assert dot == pytest.approx(cos, abs=1e-5)


class TestEmbeddingSimilarityHelper:
    def test_identical_embeddings_score_one(self):
        emb = [1.0, 0.0, 0.0]
        assert embedding_similarity(emb, emb) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_embeddings_score_zero(self):
        assert embedding_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-5)

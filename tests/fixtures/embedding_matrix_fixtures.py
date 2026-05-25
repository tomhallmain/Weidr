"""
Synthetic CLIP-style embedding catalog for iterative vs matrix compare tests.

Creates 26 minimal PNGs under a temp directory and a parallel dict of fixed
512-dimensional L2-normalized embeddings. Three tight clusters should form
compare groups at threshold 0.9; outliers use an orthogonal direction and
should not match cluster members.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

EMBED_DIM = 512
SCORE_TABLE_TOLERANCE = 1e-5

# Cluster sizes (26 files total): 8 + 8 + 6 + 4 outliers
_CLUSTER_A_COUNT = 8
_CLUSTER_B_COUNT = 8
_CLUSTER_C_COUNT = 6
_OUTLIER_COUNT = 4
N_FILES = _CLUSTER_A_COUNT + _CLUSTER_B_COUNT + _CLUSTER_C_COUNT + _OUTLIER_COUNT

EMBEDDING_SIMILARITY_THRESHOLD = 0.9


@dataclass
class EmbeddingMatrixCatalog:
    """Paths and precomputed embeddings for one synthetic library."""

    dir: str
    paths: List[str] = field(default_factory=list)
    embeddings_by_path: Dict[str, np.ndarray] = field(default_factory=dict)
    cluster_by_path: Dict[str, str] = field(default_factory=dict)

    @property
    def n_files(self) -> int:
        return len(self.paths)


def _unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero vector cannot be normalized")
    return v / n


def _cluster_centroids() -> Dict[str, np.ndarray]:
    """Mutually near-orthogonal directions in 512-D."""
    a = np.zeros(EMBED_DIM, dtype=np.float32)
    a[0:32] = 1.0
    b = np.zeros(EMBED_DIM, dtype=np.float32)
    b[64:96] = 1.0
    c = np.zeros(EMBED_DIM, dtype=np.float32)
    c[128:160] = 1.0
    outlier = np.zeros(EMBED_DIM, dtype=np.float32)
    outlier[256:288] = 1.0
    return {
        "cluster_a": _unit_vector(a),
        "cluster_b": _unit_vector(b),
        "cluster_c": _unit_vector(c),
        "outlier": _unit_vector(outlier),
    }


def _member_embedding(
    centroid: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float = 0.001,
) -> np.ndarray:
    return _unit_vector(centroid + noise_scale * rng.standard_normal(EMBED_DIM).astype(np.float32))


def _write_png(directory: str, name: str) -> str:
    path = os.path.join(directory, name)
    Image.new("RGB", (16, 16), (128, 128, 128)).save(path, format="PNG")
    return path


def build_embedding_matrix_catalog(directory: str) -> EmbeddingMatrixCatalog:
    """Populate *directory* with PNG stubs and return embedding metadata."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    centroids = _cluster_centroids()
    catalog = EmbeddingMatrixCatalog(dir=directory)

    specs: List[tuple[str, str, int]] = [
        ("cluster_a", "cluster_a", _CLUSTER_A_COUNT),
        ("cluster_b", "cluster_b", _CLUSTER_B_COUNT),
        ("cluster_c", "cluster_c", _CLUSTER_C_COUNT),
        ("outlier", "outlier", _OUTLIER_COUNT),
    ]

    for cluster_key, prefix, count in specs:
        centroid = centroids[cluster_key]
        for i in range(count):
            name = f"{prefix}_{i:02d}.png"
            path = _write_png(directory, name)
            catalog.paths.append(path)
            catalog.embeddings_by_path[path] = _member_embedding(centroid, rng)
            catalog.cluster_by_path[path] = cluster_key

    catalog.paths.sort()
    assert catalog.n_files == N_FILES
    return catalog


def fake_image_embeddings_clip(image_path: str, catalog: EmbeddingMatrixCatalog) -> np.ndarray:
    """Drop-in for ``compare.model.image_embeddings_clip`` using the catalog."""
    emb = catalog.embeddings_by_path.get(image_path)
    if emb is None:
        raise KeyError(f"No fixture embedding for path: {image_path!r}")
    return np.asarray(emb, dtype=np.float32)


@pytest.fixture
def embedding_matrix_catalog(tmp_path) -> EmbeddingMatrixCatalog:
    """26 synthetic images with clustered CLIP-style embeddings."""
    return build_embedding_matrix_catalog(str(tmp_path))

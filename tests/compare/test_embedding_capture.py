"""
Unit tests for compare/embedding_capture.py -- ad-hoc single-file embedding
computation for the embedding seed library (docs/embedding-seed-library.md,
section 5.4), deliberately independent of CompareWrapper/CompareManager and
any active compare instance.

The success path inside compute_media_embedding (real model inference) is
not exercised here -- only the pure-Python early-return guards -- since
that would require the actual embedding model dependencies to be present.
"""
from __future__ import annotations

from compare.embedding_capture import compute_media_embedding, embedding_capture_modes
from utils.constants import CompareMode


class TestEmbeddingCaptureModes:
    def test_includes_embedding_architectures(self):
        modes = embedding_capture_modes()
        assert CompareMode.CLIP_EMBEDDING in modes
        assert CompareMode.SIGLIP_EMBEDDING in modes
        assert CompareMode.FACE_EMBEDDING in modes

    def test_excludes_non_embedding_modes(self):
        modes = embedding_capture_modes()
        assert CompareMode.COLOR_MATCHING not in modes
        assert CompareMode.COLOR_HISTOGRAM not in modes
        assert CompareMode.SIZE not in modes
        assert CompareMode.MODELS not in modes


class TestComputeMediaEmbedding:
    def test_non_embedding_mode_returns_none(self, tmp_path):
        media_path = tmp_path / "a.png"
        media_path.write_bytes(b"not a real png, just needs to exist")
        assert compute_media_embedding(str(media_path), CompareMode.SIZE) is None

    def test_missing_file_returns_none(self, tmp_path):
        missing = str(tmp_path / "missing.png")
        assert compute_media_embedding(missing, CompareMode.CLIP_EMBEDDING) is None

    def test_empty_path_returns_none(self):
        assert compute_media_embedding("", CompareMode.CLIP_EMBEDDING) is None

    def test_does_not_require_an_active_compare_instance(self, tmp_path):
        """No CompareWrapper/CompareManager is constructed anywhere in this
        test file -- these are plain module-level functions."""
        missing = str(tmp_path / "missing.png")
        assert compute_media_embedding(missing, CompareMode.FACE_EMBEDDING) is None

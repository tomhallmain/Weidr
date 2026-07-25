"""
Unit tests for the CLAP audio embedding compare mode (Track A).

Covers only the parts testable without real model inference (no torch/transformers
model loading, mirroring compare/model.py's other embedding functions, which have
no dedicated inference tests either -- see tests/compare/test_embedding_cosine_similarity.py):
mode registration, file-scoping, and dispatch wiring. Constructing
CompareEmbeddingAudioClap is safe without mocking since model loading is lazy
(only triggered when image_embeddings_func/text_embeddings_func are actually
called, not at __init__).
"""

from __future__ import annotations

import os

from compare.compare_args import CompareArgs
from compare.compare_embeddings_audio_clap import CompareEmbeddingAudioClap, gather_audio_files
from compare.compare_wrapper import CompareWrapper
from utils.constants import CompareMode


class TestCompareModeRegistration:
    def test_get_text(self):
        assert CompareMode.AUDIO_CLAP_EMBEDDING.get_text() == "CLAP Audio Embedding"

    def test_is_embedding(self):
        assert CompareMode.AUDIO_CLAP_EMBEDDING.is_embedding() is True

    def test_in_embedding_modes(self):
        assert CompareMode.AUDIO_CLAP_EMBEDDING in CompareMode.embedding_modes()

    def test_in_text_search_modes(self):
        assert CompareMode.AUDIO_CLAP_EMBEDDING in CompareMode.text_search_modes()

    def test_get_by_name_roundtrip(self):
        assert CompareMode.get("AUDIO_CLAP_EMBEDDING") == CompareMode.AUDIO_CLAP_EMBEDDING


class TestGatherAudioFiles:
    def test_only_audio_extensions_returned(self, tmp_path):
        (tmp_path / "a.mp3").write_bytes(b"")
        (tmp_path / "b.wav").write_bytes(b"")
        (tmp_path / "c.jpg").write_bytes(b"")
        (tmp_path / "d.png").write_bytes(b"")

        files = gather_audio_files(base_dir=str(tmp_path), recursive=False)
        names = {os.path.basename(f) for f in files}

        assert "a.mp3" in names or "b.wav" in names
        assert "c.jpg" not in names
        assert "d.png" not in names

    def test_ignores_passed_exts_and_include_flags(self, tmp_path):
        # Confirms the override -- exts/include_videos/include_gifs/include_pdfs
        # are all ignored in favor of config.audio_types, since a visual model's
        # exts list has no meaning for an audio-only compare mode.
        (tmp_path / "only.wav").write_bytes(b"")
        files = gather_audio_files(
            base_dir=str(tmp_path),
            exts=[".this_is_not_a_real_extension"],
            recursive=False,
            include_videos=True,
            include_gifs=True,
            include_pdfs=True,
        )
        names = {os.path.basename(f) for f in files}
        assert "only.wav" in names


class TestCompareEmbeddingAudioClapConstruction:
    def test_constructs_without_loading_a_model(self):
        instance = CompareEmbeddingAudioClap(CompareArgs())
        assert instance.COMPARE_MODE == CompareMode.AUDIO_CLAP_EMBEDDING
        assert instance.image_embeddings_func.__name__ == "image_embeddings_clap"
        assert instance.text_embeddings_func.__name__ == "text_embeddings_clap"

    def test_file_embeddings_shape(self):
        instance = CompareEmbeddingAudioClap(CompareArgs())
        assert instance._file_embeddings.shape == (0, 512)


class TestCompareWrapperDispatch:
    def test_new_compare_builds_audio_clap_instance(self):
        wrapper = CompareWrapper(master=None, compare_mode=CompareMode.AUDIO_CLAP_EMBEDDING, app_actions=None)
        wrapper.new_compare(CompareArgs())
        assert isinstance(wrapper._compare, CompareEmbeddingAudioClap)

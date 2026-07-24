"""
Unit tests for _architecture_label (ui/compare/embedding_seed_library_window_qt.py) --
the friendly architecture label shown on each row of the embedding seed
library window, alongside name/tags/deprecated-status.
"""
from __future__ import annotations

from compare.embedding_seed import EmbeddingSeed
from ui.compare.embedding_seed_library_window_qt import _architecture_label
from utils.constants import CompareMode
from utils.translations import _


class TestArchitectureLabel:
    def test_recognized_model_resolves_to_friendly_text(self):
        seed = EmbeddingSeed(name="a", embedding_model="CLIP_EMBEDDING")
        assert _architecture_label(seed) == CompareMode.CLIP_EMBEDDING.get_text()

    def test_different_recognized_model_resolves_correctly(self):
        seed = EmbeddingSeed(name="b", embedding_model="SIGLIP_EMBEDDING")
        assert _architecture_label(seed) == CompareMode.SIGLIP_EMBEDDING.get_text()

    def test_unrecognized_model_falls_back_to_raw_string(self):
        """CompareMode.get() raises for anything it doesn't recognize (e.g. a
        hand-edited cache entry, or an architecture removed in a later
        version) -- must not propagate, should just show the raw value.
        Not translated (returned as-is), so this needs no locale handling."""
        seed = EmbeddingSeed(name="c", embedding_model="NOT_A_REAL_ARCHITECTURE")
        assert _architecture_label(seed) == "NOT_A_REAL_ARCHITECTURE"

    def test_empty_model_falls_back_to_unknown(self):
        seed = EmbeddingSeed(name="d", embedding_model="")
        assert _architecture_label(seed) == _("Unknown")

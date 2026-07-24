"""
UI tests for EmbeddingSeedLibraryWindow._search_with_seed
(docs/embedding-seed-library.md, section 6.1): validates the seed's
embedding_model against the active compare mode and fails fast -- before
touching run_compare at all -- on a mismatch.
"""
from __future__ import annotations

import numpy as np

from compare.embedding_seed import EmbeddingSeed
from ui.compare.embedding_seed_library_window_qt import EmbeddingSeedLibraryWindow
from utils.constants import CompareMode, Mode


def _make_window(win, qtbot):
    window = EmbeddingSeedLibraryWindow(win, win.app_actions)
    qtbot.addWidget(window)
    return window


class TestSearchWithSeed:
    def test_matching_mode_runs_compare_and_increments_use(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, _ = window_with_dir
        win.compare_manager.compare_mode = CompareMode.CLIP_EMBEDDING

        seed = EmbeddingSeed(
            name="matching", embedding_model="CLIP_EMBEDDING", positive=np.array([1.0, 0.0])
        )
        EmbeddingSeed.create_seed(seed)

        run_calls = []
        warn_calls = []
        mode_calls = []
        monkeypatch.setattr(win.app_actions, "run_compare", lambda args: run_calls.append(args))
        monkeypatch.setattr(win.app_actions, "warn", lambda msg, **k: warn_calls.append(msg))
        monkeypatch.setattr(win.app_actions, "set_mode", lambda m: mode_calls.append(m))

        window = _make_window(win, qtbot)
        window._listbox.setCurrentRow(0)

        window._search_with_seed()

        assert warn_calls == []
        assert mode_calls == [Mode.SEARCH]
        assert len(run_calls) == 1
        args = run_calls[0]
        assert args.positive_seed_vectors is not None
        assert len(args.positive_seed_vectors) == 1
        assert args.positive_seed_vectors[0] is seed.positive
        assert args.negative_seed_vectors is None
        assert seed.use_count == 1

    def test_mismatched_mode_warns_and_does_not_run_compare(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, _ = window_with_dir
        win.compare_manager.compare_mode = CompareMode.SIGLIP_EMBEDDING

        seed = EmbeddingSeed(
            name="mismatched", embedding_model="CLIP_EMBEDDING", positive=np.array([1.0, 0.0])
        )
        EmbeddingSeed.create_seed(seed)

        run_calls = []
        warn_calls = []
        monkeypatch.setattr(win.app_actions, "run_compare", lambda args: run_calls.append(args))
        monkeypatch.setattr(win.app_actions, "warn", lambda msg, **k: warn_calls.append(msg))

        window = _make_window(win, qtbot)
        window._listbox.setCurrentRow(0)

        window._search_with_seed()

        assert run_calls == []
        assert len(warn_calls) == 1
        assert "CLIP" in warn_calls[0]
        assert seed.use_count == 0

    def test_negative_flag_populates_negative_seed_vectors_not_positive(
        self, window_with_dir, qtbot, monkeypatch
    ):
        win, _ = window_with_dir
        win.compare_manager.compare_mode = CompareMode.CLIP_EMBEDDING

        seed = EmbeddingSeed(
            name="neg-search", embedding_model="CLIP_EMBEDDING", positive=np.array([1.0, 0.0])
        )
        EmbeddingSeed.create_seed(seed)

        run_calls = []
        monkeypatch.setattr(win.app_actions, "run_compare", lambda args: run_calls.append(args))
        monkeypatch.setattr(win.app_actions, "warn", lambda msg, **k: None)
        monkeypatch.setattr(win.app_actions, "set_mode", lambda m: None)

        window = _make_window(win, qtbot)
        window._listbox.setCurrentRow(0)

        window._search_with_seed(negative=True)

        assert len(run_calls) == 1
        args = run_calls[0]
        assert args.positive_seed_vectors is None
        assert args.negative_seed_vectors is not None
        assert args.negative_seed_vectors[0] is seed.positive

    def test_seeds_own_negative_vector_accompanies_positive_search(
        self, window_with_dir, qtbot, monkeypatch
    ):
        """Dual-prototype seed: using it as the main (positive) search input
        also carries along its own baked-in negative companion vector."""
        win, _ = window_with_dir
        win.compare_manager.compare_mode = CompareMode.CLIP_EMBEDDING

        seed = EmbeddingSeed(
            name="dual",
            embedding_model="CLIP_EMBEDDING",
            positive=np.array([1.0, 0.0]),
            negative=np.array([0.0, 1.0]),
        )
        EmbeddingSeed.create_seed(seed)

        run_calls = []
        monkeypatch.setattr(win.app_actions, "run_compare", lambda args: run_calls.append(args))
        monkeypatch.setattr(win.app_actions, "warn", lambda msg, **k: None)
        monkeypatch.setattr(win.app_actions, "set_mode", lambda m: None)

        window = _make_window(win, qtbot)
        window._listbox.setCurrentRow(0)

        window._search_with_seed()

        args = run_calls[0]
        assert args.positive_seed_vectors[0] is seed.positive
        assert args.negative_seed_vectors[0] is seed.negative

    def test_no_seed_selected_is_a_no_op(self, window_with_dir, qtbot, monkeypatch):
        win, _ = window_with_dir
        run_calls = []
        monkeypatch.setattr(win.app_actions, "run_compare", lambda args: run_calls.append(args))

        window = _make_window(win, qtbot)
        # No seeds exist, so currentRow() is -1 -- _selected_seed() returns None.
        window._search_with_seed()

        assert run_calls == []

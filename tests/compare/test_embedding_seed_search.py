"""
Integration tests for the embedding-seed-library search consumption path
(docs/embedding-seed-library.md, section 6.1): CompareArgs.positive_seed_vectors
/ negative_seed_vectors feed straight into BaseCompareEmbedding.search_multimodal()
alongside whatever a path/text search produces -- find_similars_to_embeddings
already accepted raw vectors, so no change to the core similarity math.
"""
from __future__ import annotations

import numpy as np

from compare.compare_args import CompareArgs
from compare.compare_embeddings_clip import CompareEmbeddingClip


def _unit(vec):
    arr = np.array(vec, dtype=float)
    return arr / np.linalg.norm(arr)


def _make_compare(tmp_path, threshold=0.3):
    args = CompareArgs(base_dir=str(tmp_path), compare_threshold=threshold)
    compare = CompareEmbeddingClip(args=args)
    compare.verbose = False
    return compare


def _seed_files(compare, files_to_embeddings: dict):
    files = list(files_to_embeddings.keys())
    compare.compare_data.files_found = files
    compare._file_embeddings = np.array([files_to_embeddings[f] for f in files])


class TestSearchMultimodalWithSeedVectors:
    def test_positive_seed_vector_alone_ranks_by_similarity(self, tmp_path):
        compare = _make_compare(tmp_path)
        _seed_files(
            compare,
            {
                "close.png": _unit([0.99, 0.05]),
                "far.png": _unit([0.0, 1.0]),
            },
        )
        compare.args.positive_seed_vectors = [_unit([1.0, 0.0])]

        result = compare.search_multimodal()

        files_grouped = result[0]
        assert "close.png" in files_grouped
        assert files_grouped["close.png"] > files_grouped.get("far.png", -1)

    def test_negative_seed_vector_penalizes_similar_files(self, tmp_path):
        compare = _make_compare(tmp_path)
        _seed_files(
            compare,
            {
                "a.png": _unit([1.0, 0.0]),
                "b.png": _unit([0.0, 1.0]),
            },
        )
        # Positive seed matches both somewhat; negative seed specifically
        # targets "a.png"'s direction, so "b.png" should rank higher.
        compare.args.positive_seed_vectors = [_unit([0.7, 0.7])]
        compare.args.negative_seed_vectors = [_unit([1.0, 0.0])]

        result = compare.search_multimodal()

        files_grouped = result[0]
        assert files_grouped["b.png"] > files_grouped["a.png"]

    def test_seed_vector_composes_with_path_based_search(self, tmp_path, monkeypatch):
        """A seed vector is appended alongside whatever search_media_path
        already tokenizes -- it doesn't replace the path-based search."""
        compare = _make_compare(tmp_path)
        _seed_files(
            compare,
            {
                "match.png": _unit([0.9, 0.1]),
                "unrelated.png": _unit([0.0, 1.0]),
            },
        )
        compare.args.search_media_path = "source.png"
        compare.args.positive_seed_vectors = [_unit([0.9, 0.1])]

        captured = {}

        def _fake_tokenize_media(path, out_list, *a, **k):
            out_list.append(_unit([1.0, 0.0]))
            captured["tokenized"] = path

        monkeypatch.setattr(compare, "_tokenize_media", _fake_tokenize_media)

        result = compare.search_multimodal()

        assert captured["tokenized"] == "source.png"
        assert "match.png" in result[0]

    def test_no_positive_or_negative_anything_logs_and_returns_empty(self, tmp_path):
        compare = _make_compare(tmp_path)
        _seed_files(compare, {"a.png": _unit([1.0, 0.0])})

        assert compare.search_multimodal() == {0: {}}


class TestFindSimilarsThresholdAdjustment:
    def test_seed_vector_only_search_uses_full_threshold_not_loosened_text_threshold(
        self, tmp_path
    ):
        """A seed-vector-only search behaves like a real image match (full
        threshold), not the /3-loosened threshold reserved for freeform text
        searches with no image/vector anchor at all."""
        compare = _make_compare(tmp_path, threshold=0.3)
        _seed_files(compare, {"a.png": _unit([1.0, 0.0])})
        compare.args.positive_seed_vectors = [_unit([1.0, 0.0])]

        captured = {}
        original = compare._compute_multiembedding_diff

        def _spy(positive_embeddings=[], negative_embeddings=[], threshold=0.0):
            captured["threshold"] = threshold
            return original(positive_embeddings, negative_embeddings, threshold)

        compare._compute_multiembedding_diff = _spy
        compare.find_similars_to_embeddings([_unit([1.0, 0.0])], [])

        assert captured["threshold"] == compare.embedding_similarity_threshold

    def test_text_only_search_still_uses_loosened_threshold(self, tmp_path):
        """Unrelated regression guard: a search with no path, no seed vectors
        (i.e. text-only) should still get the original /3 loosening."""
        compare = _make_compare(tmp_path, threshold=0.3)
        _seed_files(compare, {"a.png": _unit([1.0, 0.0])})

        captured = {}
        original = compare._compute_multiembedding_diff

        def _spy(positive_embeddings=[], negative_embeddings=[], threshold=0.0):
            captured["threshold"] = threshold
            return original(positive_embeddings, negative_embeddings, threshold)

        compare._compute_multiembedding_diff = _spy
        compare.find_similars_to_embeddings([_unit([1.0, 0.0])], [])

        assert captured["threshold"] == compare.embedding_similarity_threshold / 3


class TestIsRunSearchRecognizesSeedVectors:
    """Regression coverage for a bug where a seed-only search (no
    search_media_path/search_text) left BaseCompare.is_run_search False --
    a second, independent "is this a search" computation
    (BaseCompare._has_search_inputs) that CompareArgs.not_searching()'s own
    fix didn't reach. That mismatch made CompareWrapper.run() think the
    request was a GROUP run while args.mode was already Mode.SEARCH,
    triggering an unwanted "Confirm group run" prompt and, if accepted,
    forcibly switching the app to Mode.GROUP mid-search-result (which in
    turn scrambled ordering-sensitive views like masonry that key off mode)."""

    def test_positive_seed_vectors_alone_sets_is_run_search_true(self, tmp_path):
        args = CompareArgs(base_dir=str(tmp_path))
        args.positive_seed_vectors = [_unit([1.0, 0.0])]
        compare = CompareEmbeddingClip(args=args)
        assert compare.is_run_search is True

    def test_negative_seed_vectors_alone_sets_is_run_search_true(self, tmp_path):
        args = CompareArgs(base_dir=str(tmp_path))
        args.negative_seed_vectors = [_unit([1.0, 0.0])]
        compare = CompareEmbeddingClip(args=args)
        assert compare.is_run_search is True

    def test_no_search_inputs_at_all_is_run_search_false(self, tmp_path):
        args = CompareArgs(base_dir=str(tmp_path))
        compare = CompareEmbeddingClip(args=args)
        assert compare.is_run_search is False

    def test_empty_seed_vector_lists_do_not_count_as_search_inputs(self, tmp_path):
        args = CompareArgs(base_dir=str(tmp_path))
        args.positive_seed_vectors = []
        args.negative_seed_vectors = []
        compare = CompareEmbeddingClip(args=args)
        assert compare.is_run_search is False

    def test_swapping_args_on_an_existing_instance_recomputes_is_run_search(self, tmp_path):
        """Mirrors CompareWrapper.run()'s reuse path: self._compare.args = args;
        self._compare.sync_search_state()."""
        compare = _make_compare(tmp_path)
        assert compare.is_run_search is False

        new_args = CompareArgs(base_dir=str(tmp_path))
        new_args.positive_seed_vectors = [_unit([1.0, 0.0])]
        compare.args = new_args
        compare.sync_search_state()

        assert compare.is_run_search is True

"""Regression tests for CompareModels reading model data after the cache is freed.

CompareData.save_data() sets file_data_dict to None once it has persisted the
cache -- each compare mode is expected to keep whatever it needs in its own
in-memory accumulator. CompareModels didn't, so get_data() populated the dict,
save_data() freed it, and the search/comparison paths then read it back:

    TypeError: argument of type 'NoneType' is not iterable

It only surfaced when the dict was actually written (save_data skips the whole
block unless has_new_file_data or overwrite), which is why a first search in a
directory failed while a repeat search against a warm cache worked.
"""

import pickle

import pytest

from compare.compare_args import CompareArgs
from compare.compare_models import CompareModels, _as_models_tuple


def _compare(tmp_path, **args_kwargs):
    args = CompareArgs(base_dir=str(tmp_path), **args_kwargs)
    compare = CompareModels(args=args, gather_files_func=lambda **kwargs: [])
    # Well under the 0.7 default, so an exact model match isn't riding on the
    # threshold boundary (identical models with no loras scores exactly 0.7).
    compare.set_similarity_threshold(0.5)
    return compare


class TestAsModelsTuple:
    def test_valid_pair_passes_through(self):
        assert _as_models_tuple((["m"], ["l"])) == (["m"], ["l"])

    @pytest.mark.parametrize(
        "value", [None, "legacy-string", (), (["m"],), (["m"], ["l"], ["x"]), ["m", "l"]]
    )
    def test_unexpected_shapes_normalise_to_empty(self, value):
        """Missing or legacy-shaped cache values must not unpack incorrectly."""
        assert _as_models_tuple(value) == ([], [])


class TestSearchAfterCacheFreed:
    def test_search_multimodal_survives_freed_cache(self, tmp_path):
        """The exact reported crash: file_data_dict is None by search time."""
        path = str(tmp_path / "a.png")
        compare = _compare(tmp_path, search_text="somemodel")
        compare.compare_data.files_found = [path]
        compare.compare_data.file_data_dict = None  # save_data() freed it
        compare._file_models = {path: (["somemodel"], [])}

        result = compare.search_multimodal()
        assert path in result[0]

    def test_search_recovers_models_from_disk_when_accumulator_empty(self, tmp_path):
        """No accumulator (e.g. a run that never called get_data) reloads the cache."""
        path = str(tmp_path / "a.png")
        with open(tmp_path / CompareModels.CACHE_FILENAME, "wb") as store:
            pickle.dump({path: (["somemodel"], [])}, store)

        compare = _compare(tmp_path, search_text="somemodel")
        compare.compare_data.files_found = [path]
        compare.compare_data.file_data_dict = None
        compare._file_models = {}

        result = compare.search_multimodal()
        assert path in result[0]

    def test_search_with_no_cache_on_disk_is_empty_not_an_error(self, tmp_path):
        path = str(tmp_path / "a.png")
        compare = _compare(tmp_path, search_text="somemodel")
        compare.compare_data.files_found = [path]
        compare.compare_data.file_data_dict = None
        compare._file_models = {}

        assert compare.search_multimodal()[0] == {}

    def test_find_similars_survives_freed_cache(self, tmp_path):
        search_path = str(tmp_path / "search.png")
        match = str(tmp_path / "match.png")
        other = str(tmp_path / "other.png")
        compare = _compare(tmp_path)
        compare.compare_data.files_found = [search_path, match, other]
        compare.compare_data.file_data_dict = None
        compare._file_models = {
            search_path: (["somemodel"], []),
            match: (["somemodel"], []),
            other: (["different"], []),
        }

        result = compare.find_similars_to_media(search_path, 0)
        assert match in result[0]
        assert other not in result[0]


class TestPopulateModelsFromCache:
    def test_reloads_from_disk_when_dict_was_freed(self, tmp_path):
        path = str(tmp_path / "a.png")
        with open(tmp_path / CompareModels.CACHE_FILENAME, "wb") as store:
            pickle.dump({path: (["modelA"], ["loraA"])}, store)

        compare = _compare(tmp_path)
        compare.compare_data.files_found = [path]
        compare.compare_data.file_data_dict = None

        compare._populate_models_from_cache()
        assert compare._file_models == {path: (["modelA"], ["loraA"])}

    def test_files_missing_from_cache_normalise_to_empty(self, tmp_path):
        path = str(tmp_path / "absent.png")
        compare = _compare(tmp_path)
        compare.compare_data.files_found = [path]
        compare.compare_data.file_data_dict = None

        compare._populate_models_from_cache()
        assert compare._file_models == {path: ([], [])}

    def test_ensure_does_not_refill_a_covering_accumulator(self, tmp_path):
        """A populated accumulator is authoritative -- no needless disk reload."""
        path = str(tmp_path / "a.png")
        compare = _compare(tmp_path)
        compare.compare_data.files_found = [path]
        compare.compare_data.file_data_dict = None
        compare._file_models = {path: (["from-memory"], [])}

        compare._ensure_file_models()
        assert compare._file_models == {path: (["from-memory"], [])}


class TestRemoveFromGroups:
    def test_removal_with_freed_cache_does_not_raise(self, tmp_path):
        """Same latent crash as the search paths: `in` against a None dict."""
        first = str(tmp_path / "a.png")
        second = str(tmp_path / "b.png")
        compare = _compare(tmp_path)
        compare.compare_data.files_found = [first, second]
        compare.compare_data.file_data_dict = None
        compare._file_models = {first: ([], []), second: ([], [])}

        compare.remove_from_groups([first])

        assert compare.compare_data.files_found == [second]
        assert first not in compare._file_models

    def test_removal_also_drops_the_live_cache_entry(self, tmp_path):
        first = str(tmp_path / "a.png")
        compare = _compare(tmp_path)
        compare.compare_data.files_found = [first]
        compare.compare_data.file_data_dict = {first: ([], [])}
        compare._file_models = {first: ([], [])}

        compare.remove_from_groups([first])

        assert compare.compare_data.file_data_dict == {}
        assert compare._file_models == {}

"""Unit tests for files/sortable_file.py — suffix extraction and laziness."""

import files.sortable_file as sortable_file_module
from files.sortable_file import SortableFile


class TestGetSuffix:
    def test_variant_filename_returns_trailing_suffix(self):
        sf = SortableFile("/some/dir/image_v2.png")
        assert sf.get_suffix() == "_v2"

    def test_plain_filename_returns_empty_suffix(self):
        sf = SortableFile("/some/dir/vacation.png")
        assert sf.get_suffix() == ""

    def test_long_generated_filename_returns_trailing_suffix(self):
        sf = SortableFile("/some/dir/SDWebUI_17602175357792320_0_s.png")
        assert sf.get_suffix() == "_0_s"

    def test_suffix_is_computed_lazily_and_cached(self, monkeypatch):
        calls = []
        original = sortable_file_module.extract_filename_base_stem

        def counting_extract(basename):
            calls.append(basename)
            return original(basename)

        monkeypatch.setattr(sortable_file_module, "extract_filename_base_stem", counting_extract)

        sf = SortableFile("/some/dir/image_v2.png")
        assert sf.suffix is None  # not computed at construction time
        assert sf.get_suffix() == "_v2"
        assert sf.get_suffix() == "_v2"
        assert len(calls) == 1  # computed once, then cached

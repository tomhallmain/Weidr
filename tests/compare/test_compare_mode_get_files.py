"""
BaseCompare.get_files() across compare modes: the data filter applies,
config.image_types is left unchanged, a text-only search inserts no search
path, and each mode scans the media types it can compare.
"""
from __future__ import annotations

import pytest

from compare.compare_args import CompareArgs
from compare.compare_colors import CompareColors
from compare.compare_models import CompareModels
from compare.compare_prompts import ComparePrompts
from compare.compare_prompts_exact import ComparePromptsExact
from compare.compare_size import CompareSize
from utils.config import config

ALL_CLASSES = {
    "CompareColors": CompareColors,
    "CompareSize": CompareSize,
    "CompareModels": CompareModels,
    "ComparePrompts": ComparePrompts,
    "ComparePromptsExact": ComparePromptsExact,
}

# Modes that compare generation metadata, which the extractor reads from images only.
METADATA_CLASSES = {
    "CompareModels": CompareModels,
    "ComparePrompts": ComparePrompts,
    "ComparePromptsExact": ComparePromptsExact,
}

FILES = ["/d/a.png", "/d/b.png", "/d/c.png", "/d/d.png"]


class _ActiveFilter:
    """Stand-in data filter; apply_filter is patched to read keep_files."""

    def __init__(self, keep_files):
        self.keep_files = keep_files

    def is_active(self):
        return True


def _make(compare_cls, tmp_path, gathered=None, **arg_attrs):
    calls = []

    def _gather(**kwargs):
        calls.append(kwargs)
        return list(gathered if gathered is not None else FILES)

    args = CompareArgs(base_dir=str(tmp_path))
    for name, value in arg_attrs.items():
        setattr(args, name, value)
    return compare_cls(args=args, gather_files_func=_gather), calls


@pytest.mark.parametrize("compare_cls", ALL_CLASSES.values(), ids=ALL_CLASSES.keys())
class TestGetFilesInEveryMode:
    def test_data_filter_is_applied(self, tmp_path, compare_cls, monkeypatch):
        import compare.compare_filters as compare_filters

        monkeypatch.setattr(
            compare_filters, "apply_filter",
            lambda files, f, **kwargs: [p for p in files if p in f.keep_files])
        compare, _ = _make(compare_cls, tmp_path)
        compare.args.data_filter = _ActiveFilter(keep_files={"/d/a.png", "/d/c.png"})

        compare.get_files()

        assert compare.files == ["/d/a.png", "/d/c.png"]
        assert compare.data_filter_stats == (2, 4)

    def test_config_image_types_unchanged_with_gifs(self, tmp_path, compare_cls):
        before = list(config.image_types)
        compare, _ = _make(compare_cls, tmp_path, include_gifs=True)

        compare.get_files()
        compare.get_files()

        assert config.image_types == before

    def test_text_only_search_inserts_no_search_path(self, tmp_path, compare_cls):
        args = CompareArgs(base_dir=str(tmp_path))
        args.search_text = "a red car"
        compare = compare_cls(args=args, gather_files_func=lambda **kwargs: list(FILES))

        compare.get_files()

        assert None not in compare.files
        assert compare.files == FILES


@pytest.mark.parametrize("compare_cls", METADATA_CLASSES.values(), ids=METADATA_CLASSES.keys())
def test_metadata_modes_scan_no_videos_or_documents(tmp_path, compare_cls):
    compare, calls = _make(
        compare_cls, tmp_path,
        include_videos=True, include_gifs=True, include_pdfs=True, include_epubs=True)

    compare.get_files()

    assert calls[0]["include_videos"] is False
    assert calls[0]["include_pdfs"] is False
    assert calls[0]["include_epubs"] is False
    assert calls[0]["include_gifs"] is True


def test_size_mode_scans_what_args_enable(tmp_path):
    compare, calls = _make(
        CompareSize, tmp_path,
        include_videos=True, include_gifs=False, include_pdfs=True, include_epubs=True)

    compare.get_files()

    assert calls[0]["include_videos"] is True
    assert calls[0]["include_gifs"] is False
    assert calls[0]["include_pdfs"] is True
    assert calls[0]["include_epubs"] is True

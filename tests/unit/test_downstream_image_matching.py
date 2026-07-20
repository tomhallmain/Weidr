"""Unit tests for related-image matching helpers.

Covers:
  - get_related_image_path (check_extra_directories=None path)
  - get_sources_with_downstream_in_dir  (Ctrl+Y logic)
  - get_downstream_files_for_sources    (Ctrl+Shift+Y logic)

All three matching channels are exercised:
  1. Exact metadata match  (stored related-image path == source path)
  2. Loose basename match  (same filename, different directory; basename > 10 chars)
  3. Variant suffix match  (_[A-Za-z]{1,8} stripped from dir-Y stem; extension must agree)
"""

import os

import pytest
from PIL import Image, PngImagePlugin

from files.related_image import (
    get_related_image_path,
    get_sources_with_downstream_in_dir,
    get_downstream_files_for_sources,
)
from utils.config import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(path: str, related_image: str | None = None) -> None:
    img = Image.new("RGB", (4, 4), (128, 128, 128))
    if related_image is not None:
        info = PngImagePlugin.PngInfo()
        info.add_text("related_image", related_image)
        img.save(path, pnginfo=info)
    else:
        img.save(path, format="PNG")


def _make_jpg(path: str) -> None:
    Image.new("RGB", (4, 4), (200, 200, 200)).save(path, format="JPEG")


# ---------------------------------------------------------------------------
# Autouse fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _png_file_types(monkeypatch):
    monkeypatch.setattr(config, "file_types", [".png"])


# ---------------------------------------------------------------------------
# TestGetRelatedImagePath — the basic building block
# ---------------------------------------------------------------------------

class TestGetRelatedImagePath:
    def test_returns_stored_path(self, tmp_path):
        source = str(tmp_path / "source.png")
        derived = str(tmp_path / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=source)

        related, _exact = get_related_image_path(
            derived, check_extra_directories=None
        )
        assert related == source

    def test_returns_none_when_no_metadata(self, tmp_path):
        plain = str(tmp_path / "plain.png")
        _make_png(plain)

        related, _exact = get_related_image_path(
            plain, check_extra_directories=None
        )
        assert related is None

    def test_does_not_search_extra_directories(self, tmp_path, monkeypatch):
        """check_extra_directories=None must short-circuit before any filesystem probe."""
        source = str(tmp_path / "source.png")
        derived = str(tmp_path / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=source)

        probed = []
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [str(tmp_path)])
        original_isfile = os.path.isfile
        monkeypatch.setattr(os.path, "isfile", lambda p: (probed.append(p), original_isfile(p))[1])

        get_related_image_path(derived, check_extra_directories=None)

        # os.path.isfile must never have been called with the related path
        assert source not in probed


# ---------------------------------------------------------------------------
# TestGetSourcesWithDownstreamInDir
# ---------------------------------------------------------------------------

class TestGetSourcesWithDownstreamInDir:
    def test_exact_metadata_match(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "source.png")
        unrelated = str(dir_x / "unrelated.png")
        derived = str(dir_y / "derived.png")
        _make_png(source)
        _make_png(unrelated)
        _make_png(derived, related_image=source)

        result = get_sources_with_downstream_in_dir(
            [source, unrelated], str(dir_y)
        )
        assert result == [source]

    def test_no_downstream_returns_empty(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "source.png")
        other = str(dir_y / "other.png")
        _make_png(source)
        _make_png(other)

        result = get_sources_with_downstream_in_dir([source], str(dir_y))
        assert result == []

    def test_loose_basename_match(self, tmp_path):
        """Metadata points to same filename in a third directory — loose match fires."""
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "longname_source.png")  # basename > 10 chars
        elsewhere = str(tmp_path / "longname_source.png")  # same basename, different dir
        derived = str(dir_y / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=elsewhere)

        result = get_sources_with_downstream_in_dir([source], str(dir_y))
        assert result == [source]

    def test_basename_too_short_skips_loose_match(self, tmp_path):
        """Basename ≤ 10 chars must not trigger the loose match."""
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "short.png")  # "short.png" = 9 chars
        elsewhere = str(tmp_path / "short.png")
        derived = str(dir_y / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=elsewhere)

        result = get_sources_with_downstream_in_dir([source], str(dir_y))
        assert result == []

    def test_variant_suffix_match(self, tmp_path):
        """Dir-Y file named <stem>_<suffix>.png matches dir-X file named <stem>.png."""
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "portrait.png")
        derived = str(dir_y / "portrait_edit.png")
        _make_png(source)
        _make_png(derived)  # no metadata — variant suffix is the only link

        result = get_sources_with_downstream_in_dir([source], str(dir_y))
        assert result == [source]

    def test_variant_suffix_extension_mismatch_not_matched(self, tmp_path, monkeypatch):
        """portrait.png in dir-X must not match portrait_edit.jpg in dir-Y."""
        monkeypatch.setattr(config, "file_types", [".png", ".jpg"])
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "portrait.png")
        derived = str(dir_y / "portrait_edit.jpg")
        _make_png(source)
        _make_jpg(derived)

        result = get_sources_with_downstream_in_dir([source], str(dir_y))
        assert result == []

    def test_multiple_sources_partial_match(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source_a = str(dir_x / "has_downstream.png")
        source_b = str(dir_x / "no_downstream.png")
        derived = str(dir_y / "derived.png")
        _make_png(source_a)
        _make_png(source_b)
        _make_png(derived, related_image=source_a)

        result = get_sources_with_downstream_in_dir(
            [source_a, source_b], str(dir_y)
        )
        assert result == [source_a]

    def test_empty_source_list_returns_empty(self, tmp_path):
        dir_y = tmp_path / "dir_y"
        dir_y.mkdir()
        _make_png(str(dir_y / "derived.png"))

        result = get_sources_with_downstream_in_dir([], str(dir_y))
        assert result == []

    def test_empty_dir_y_returns_empty(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "source.png")
        _make_png(source)

        result = get_sources_with_downstream_in_dir([source], str(dir_y))
        assert result == []


# ---------------------------------------------------------------------------
# TestGetDownstreamFilesForSources
# ---------------------------------------------------------------------------

class TestGetDownstreamFilesForSources:
    def test_exact_metadata_match(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "source.png")
        derived = str(dir_y / "derived.png")
        unrelated = str(dir_y / "unrelated.png")
        _make_png(source)
        _make_png(derived, related_image=source)
        _make_png(unrelated)

        result = get_downstream_files_for_sources([source], str(dir_y))
        assert result == [derived]

    def test_no_match_returns_empty(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "source.png")
        other = str(dir_y / "other.png")
        _make_png(source)
        _make_png(other)

        result = get_downstream_files_for_sources([source], str(dir_y))
        assert result == []

    def test_loose_basename_match(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "longname_source.png")
        elsewhere = str(tmp_path / "longname_source.png")
        derived = str(dir_y / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=elsewhere)

        result = get_downstream_files_for_sources([source], str(dir_y))
        assert result == [derived]

    def test_variant_suffix_match(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "portrait.png")
        derived = str(dir_y / "portrait_edit.png")
        _make_png(source)
        _make_png(derived)

        result = get_downstream_files_for_sources([source], str(dir_y))
        assert result == [derived]

    def test_variant_suffix_extension_mismatch_not_matched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "file_types", [".png", ".jpg"])
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "portrait.png")
        derived = str(dir_y / "portrait_edit.jpg")
        _make_png(source)
        _make_jpg(derived)

        result = get_downstream_files_for_sources([source], str(dir_y))
        assert result == []

    def test_multiple_derived_files_for_one_source(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "source.png")
        derived_a = str(dir_y / "derived_a.png")
        derived_b = str(dir_y / "derived_b.png")
        unrelated = str(dir_y / "unrelated.png")
        _make_png(source)
        _make_png(derived_a, related_image=source)
        _make_png(derived_b, related_image=source)
        _make_png(unrelated)

        result = get_downstream_files_for_sources([source], str(dir_y))
        assert set(result) == {derived_a, derived_b}

    def test_multiple_sources_each_matched(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source_a = str(dir_x / "source_a.png")
        source_b = str(dir_x / "source_b.png")
        derived_a = str(dir_y / "derived_a.png")
        derived_b = str(dir_y / "derived_b.png")
        _make_png(source_a)
        _make_png(source_b)
        _make_png(derived_a, related_image=source_a)
        _make_png(derived_b, related_image=source_b)

        result = get_downstream_files_for_sources(
            [source_a, source_b], str(dir_y)
        )
        assert set(result) == {derived_a, derived_b}

    def test_empty_source_list_returns_empty(self, tmp_path):
        dir_y = tmp_path / "dir_y"
        dir_y.mkdir()
        _make_png(str(dir_y / "derived.png"))

        result = get_downstream_files_for_sources([], str(dir_y))
        assert result == []

    def test_empty_dir_y_returns_empty(self, tmp_path):
        dir_x = tmp_path / "dir_x"
        dir_y = tmp_path / "dir_y"
        dir_x.mkdir(); dir_y.mkdir()

        source = str(dir_x / "source.png")
        _make_png(source)

        result = get_downstream_files_for_sources([source], str(dir_y))
        assert result == []


# ---------------------------------------------------------------------------
# Same-directory rescue of stale related-image pointers
# ---------------------------------------------------------------------------

class TestSameDirectoryPointerRescue:
    """A pointer left stale by a drive-letter change (D:\\ -> F:\\) or a
    directory move must resolve to the same-named file sitting beside the
    image that references it — when the trailing directory components of
    both locations agree — instead of reporting no exact match."""

    def _make_pair(self, tmp_path, pointer):
        sub = tmp_path / "media" / "pics"
        sub.mkdir(parents=True, exist_ok=True)
        source = str(sub / "source_image.png")
        derived = str(sub / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=pointer)
        return source, derived

    def test_stale_pointer_with_matching_tail_resolves_beside_image(self, tmp_path):
        # Same trailing components (media/pics), nonexistent prefix — the
        # drive-migration / moved-ancestor shape.
        stale = str(tmp_path / "GONE_DRIVE" / "media" / "pics" / "source_image.png")
        source, derived = self._make_pair(tmp_path, stale)
        related, found = get_related_image_path(derived, check_extra_directories=False)
        assert found is True
        assert os.path.normpath(related) == os.path.normpath(source)

    def test_media_details_text_reports_rescued_match(self, tmp_path):
        from files.related_image import get_related_image_text
        from utils.translations import _
        stale = str(tmp_path / "GONE_DRIVE" / "media" / "pics" / "source_image.png")
        source, derived = self._make_pair(tmp_path, stale)
        text = get_related_image_text(derived)
        assert _(" (Exact Match Not Found)") not in text
        assert os.path.normpath(text) == os.path.normpath(source)

    def test_tail_mismatch_is_not_rescued(self, tmp_path):
        # Same basename exists beside the image, but the pointer referenced
        # clearly different parent directories — must not be claimed.
        stale = str(tmp_path / "GONE_DRIVE" / "other" / "place" / "source_image.png")
        _source, derived = self._make_pair(tmp_path, stale)
        related, found = get_related_image_path(derived, check_extra_directories=False)
        assert found is False
        assert related == stale

    def test_missing_neighbor_is_not_rescued(self, tmp_path):
        # Tail matches but no same-named file exists beside the image.
        stale = str(tmp_path / "GONE_DRIVE" / "media" / "pics" / "never_existed.png")
        _source, derived = self._make_pair(tmp_path, stale)
        related, found = get_related_image_path(derived, check_extra_directories=False)
        assert found is False
        assert related == stale

    def test_pointer_to_own_basename_is_not_rescued(self, tmp_path):
        # A file must not be claimed as its own related image.
        stale = str(tmp_path / "GONE_DRIVE" / "media" / "pics" / "derived.png")
        _source, derived = self._make_pair(tmp_path, stale)
        related, found = get_related_image_path(derived, check_extra_directories=False)
        assert found is False
        assert related == stale

    def test_valid_pointer_unaffected(self, tmp_path):
        # Existing behavior pinned: a pointer that exists on disk resolves
        # directly, no rescue involved.
        sub = tmp_path / "media" / "pics"
        sub.mkdir(parents=True, exist_ok=True)
        source = str(sub / "source_image.png")
        derived = str(sub / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=source)
        related, found = get_related_image_path(derived, check_extra_directories=False)
        assert found is True
        assert related == source


# ---------------------------------------------------------------------------
# get_downstream_related_images — quiet flag
# ---------------------------------------------------------------------------

class TestGetDownstreamRelatedImagesQuiet:
    """quiet=True suppresses the per-directory result toasts so multi-directory
    callers (all-open-windows search) can emit one aggregate summary."""

    def _app_actions(self, toasts):
        from types import SimpleNamespace
        return SimpleNamespace(toast=lambda msg, **kw: toasts.append(msg))

    def test_quiet_suppresses_no_result_toast(self, tmp_path):
        from files.related_image import get_downstream_related_images
        toasts = []
        result = get_downstream_related_images(
            str(tmp_path / "absent_source.png"), str(tmp_path),
            self._app_actions(toasts), force_refresh=True, quiet=True,
        )
        assert result is None
        assert toasts == []

    def test_default_still_toasts(self, tmp_path):
        from files.related_image import get_downstream_related_images
        toasts = []
        result = get_downstream_related_images(
            str(tmp_path / "absent_source.png"), str(tmp_path),
            self._app_actions(toasts), force_refresh=True,
        )
        assert result is None
        assert len(toasts) == 1

    def test_quiet_suppresses_found_toast(self, tmp_path):
        from files.related_image import get_downstream_related_images
        source = str(tmp_path / "src.png")
        derived = str(tmp_path / "derived.png")
        _make_png(source)
        _make_png(derived, related_image=source)
        toasts = []
        result = get_downstream_related_images(
            source, str(tmp_path),
            self._app_actions(toasts), force_refresh=True, quiet=True,
        )
        assert result == [derived]
        assert toasts == []

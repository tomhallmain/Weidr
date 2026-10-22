"""Unit tests for related-image family grouping and the all-suffixes filter.

Covers group_related_families, family_has_all_suffixes,
get_families_with_all_suffixes and parse_suffix_list.
"""

import os

import pytest
from PIL import Image, PngImagePlugin

from files.related_image import (
    family_has_all_suffixes,
    get_families_with_all_suffixes,
    group_related_families,
    parse_suffix_list,
)
from utils.config import config


def _make_png(path: str, related_image: str | None = None) -> str:
    img = Image.new("RGB", (4, 4), (128, 128, 128))
    if related_image is not None:
        info = PngImagePlugin.PngInfo()
        info.add_text("related_image", related_image)
        img.save(path, pnginfo=info)
    else:
        img.save(path, format="PNG")
    return path


@pytest.fixture(autouse=True)
def _png_file_types(monkeypatch):
    monkeypatch.setattr(config, "file_types", [".png"])


def _paths(tmp_path, *names):
    return [_make_png(str(tmp_path / n)) for n in names]


class TestGroupRelatedFamilies:
    def test_groups_by_filename_pattern(self, tmp_path):
        paths = _paths(tmp_path, "cat.png", "cat_edit.png", "dog.png", "dog_up.png")
        families = group_related_families(paths)
        assert families == {
            "cat.png": [paths[0], paths[1]],
            "dog.png": [paths[2], paths[3]],
        }

    def test_missing_intermediate_is_not_skipped(self, tmp_path):
        """img_c_b must not attach to img when its direct parent img_c is
        absent -- img_c may be a seed of its own, unrelated to img."""
        paths = _paths(tmp_path, "img.png", "img_c_b.png")
        assert group_related_families(paths) == {
            "img.png": [paths[0]],
            "img_c_b.png": [paths[1]],
        }

    def test_chain_through_present_intermediate(self, tmp_path):
        paths = _paths(tmp_path, "img.png", "img_c.png", "img_c_b.png")
        assert group_related_families(paths) == {"img.png": paths}

    def test_groups_by_metadata(self, tmp_path):
        seed = _make_png(str(tmp_path / "seed_image.png"))
        derived = _make_png(str(tmp_path / "ComfyUI_00001_.png"), related_image=seed)
        assert group_related_families([seed, derived]) == {"seed_image.png": [seed, derived]}

    def test_external_seed_does_not_form_a_family(self, tmp_path):
        """Derivatives pointing straight outside the set are each their own seed;
        they are not grouped under the external source."""
        outside = str(tmp_path / "elsewhere" / "origin_image.png")
        a = _make_png(str(tmp_path / "origin_image_edit.png"), related_image=outside)
        b = _make_png(str(tmp_path / "origin_image_up.png"), related_image=outside)
        assert group_related_families([a, b]) == {
            "origin_image_edit.png": [a],
            "origin_image_up.png": [b],
        }

    def test_chain_stops_at_last_member_inside_the_set(self, tmp_path):
        """B points outside; B's own derivatives make B the family seed."""
        outside = str(tmp_path / "elsewhere" / "A.png")
        b = _make_png(str(tmp_path / "bravo_seed.png"), related_image=outside)
        b_edit = _make_png(str(tmp_path / "bravo_seed_edit.png"))
        b_meta = _make_png(str(tmp_path / "ComfyUI_00009_.png"), related_image=b)
        assert group_related_families([b, b_edit, b_meta]) == {
            "bravo_seed.png": [b, b_edit, b_meta],
        }

    def test_default_origin_resolution_still_crosses_boundary(self, tmp_path):
        """stop_at_boundary is opt-in: the related-image sort keeps keying
        on the external source."""
        from files.related_image import get_origin_basename
        from files.sortable_file import SortableFile
        outside = str(tmp_path / "elsewhere" / "A.png")
        b = SortableFile(_make_png(str(tmp_path / "bravo_seed.png"), related_image=outside))
        lookup = {b.basename: b}
        assert get_origin_basename(b, lookup) == "A.png"
        assert get_origin_basename(b, lookup, stop_at_boundary=True) == "bravo_seed.png"


class TestFamilyHasAllSuffixes:
    def test_seed_stem_does_not_count(self):
        # The seed's own "_edit" is part of its name, not a derivative marker.
        assert not family_has_all_suffixes(["/d/cat_edit.png"], "cat_edit.png", ["edit"])

    def test_requires_every_suffix(self):
        members = ["/d/cat.png", "/d/cat_edit.png"]
        assert family_has_all_suffixes(members, "cat.png", ["edit"])
        assert not family_has_all_suffixes(members, "cat.png", ["edit", "upscale"])

    def test_empty_suffixes_never_qualify(self):
        assert not family_has_all_suffixes(["/d/cat.png", "/d/cat_edit.png"], "cat.png", [])


class TestGetFamiliesWithAllSuffixes:
    def test_marks_whole_family_including_extras_and_repeats(self, tmp_path):
        paths = _paths(
            tmp_path,
            "cat.png", "cat_edit.png", "cat_edit_2.png", "cat_up.png", "cat_other.png",
            "dog.png", "dog_edit.png",
            "eel.png",
        )
        stats: dict = {}
        result = get_families_with_all_suffixes(paths, ["edit", "up"], stats=stats)
        assert result == paths[:5]
        assert stats == {"families": 3, "matched_families": 1}

    def test_preserves_input_order(self, tmp_path):
        paths = _paths(tmp_path, "b_up.png", "a.png", "b.png", "a_up.png")
        assert get_families_with_all_suffixes(paths, ["up"]) == paths

    def test_external_seed_derivatives_never_qualify(self, tmp_path):
        outside = str(tmp_path / "elsewhere" / "origin_image.png")
        paths = [
            _make_png(str(tmp_path / "origin_image_edit.png"), related_image=outside),
            _make_png(str(tmp_path / "origin_image_up.png"), related_image=outside),
        ]
        assert get_families_with_all_suffixes(paths, ["edit", "up"]) == []

    def test_boundary_seed_family_qualifies(self, tmp_path):
        outside = str(tmp_path / "elsewhere" / "A.png")
        paths = [
            _make_png(str(tmp_path / "bravo_seed.png"), related_image=outside),
            _make_png(str(tmp_path / "bravo_seed_edit.png")),
            _make_png(str(tmp_path / "bravo_seed_up.png")),
        ]
        assert get_families_with_all_suffixes(paths, ["edit", "up"]) == paths

    def test_metadata_linked_derivative_counts_on_full_stem(self, tmp_path):
        seed = _make_png(str(tmp_path / "seed_image.png"))
        renamed = _make_png(str(tmp_path / "render_upscale.png"), related_image=seed)
        edit = _make_png(str(tmp_path / "seed_image_edit.png"))
        paths = [seed, renamed, edit]
        assert get_families_with_all_suffixes(paths, ["edit", "upscale"]) == paths


class TestParseSuffixList:
    def test_splits_strips_and_dedupes(self):
        assert parse_suffix_list(" _edit, up  EDIT,,upscale ") == (["edit", "up", "upscale"], [])

    @pytest.mark.parametrize("bad", ["ab-c", "123", "toolongsuffix"])
    def test_reports_invalid_entries(self, bad):
        suffixes, errors = parse_suffix_list(f"edit {bad}")
        assert suffixes == ["edit"]
        assert len(errors) == 1 and errors[0].startswith(bad)

    def test_empty_input(self):
        assert parse_suffix_list("") == ([], [])

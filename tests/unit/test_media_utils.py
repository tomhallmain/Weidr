"""Unit tests for utils/media_utils.py media-type classification."""

import pytest

from utils.config import config
from utils.constants import MediaType
from utils.media_utils import (
    get_media_type_for_path,
    get_paged_document_pdf,
    is_animated_image_candidate,
    is_classifier_dynamic_media_path,
    is_paged_document_path,
    is_large_image_dims,
    scale_dims,
)


@pytest.mark.parametrize(
    "flag_name,ext,enabled_type,disabled_type",
    [
        ("enable_videos", ".mp4", MediaType.VIDEO, MediaType.UNCONFIGURED),
        ("enable_gifs", ".gif", MediaType.GIF, MediaType.UNCONFIGURED),
        ("enable_pdfs", ".pdf", MediaType.PDF, MediaType.UNCONFIGURED),
        ("enable_epubs", ".epub", MediaType.EPUB, MediaType.UNCONFIGURED),
        ("enable_epubs", ".EPUB", MediaType.EPUB, MediaType.UNCONFIGURED),
        ("enable_svgs", ".svg", MediaType.SVG, MediaType.UNCONFIGURED),
        ("enable_html", ".html", MediaType.HTML, MediaType.UNCONFIGURED),
        ("enable_audio", ".mp3", MediaType.AUDIO, MediaType.UNCONFIGURED),
    ],
)
def test_get_media_type_respects_enable_flags(
    monkeypatch, tmp_path, flag_name, ext, enabled_type, disabled_type
):
    path = str(tmp_path / f"sample{ext}")
    path_obj = tmp_path / f"sample{ext}"
    path_obj.write_bytes(b"x")

    monkeypatch.setattr(config, flag_name, True)
    assert get_media_type_for_path(path) == enabled_type

    monkeypatch.setattr(config, flag_name, False)
    assert get_media_type_for_path(path) == disabled_type


def test_get_media_type_htm_uses_html_flag(monkeypatch, tmp_path):
    path = str(tmp_path / "index.htm")
    (tmp_path / "index.htm").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(config, "enable_html", True)
    assert get_media_type_for_path(path) == MediaType.HTML

    monkeypatch.setattr(config, "enable_html", False)
    assert get_media_type_for_path(path) == MediaType.UNCONFIGURED


class TestScaleDims:
    def test_an_image_that_already_fits_is_untouched(self):
        assert scale_dims((100, 50), (200, 200)) == (100, 50)
        assert scale_dims((100, 200), (100, 200)) == (100, 200)
        assert scale_dims((50, 80), (100, 200)) == (50, 80)

    def test_a_wider_image_is_bounded_by_width(self):
        assert scale_dims((400, 200), (100, 200)) == (100, 50)
        assert scale_dims((400, 100), (200, 200)) == (200, 50)

    def test_a_taller_image_is_bounded_by_height(self):
        assert scale_dims((100, 400), (200, 200)) == (50, 200)
        assert scale_dims((200, 400), (200, 100)) == (50, 100)

    def test_maximize_grows_a_smaller_image_to_the_box(self):
        assert scale_dims((100, 50), (200, 200), maximize=True) == (200, 100)
        assert scale_dims((50, 100), (200, 200), maximize=True) == (100, 200)

    def test_without_maximize_a_smaller_image_stays_small(self):
        assert scale_dims((100, 50), (200, 200), maximize=False) == (100, 50)

    def test_a_square_image_in_a_square_box_is_unchanged(self):
        assert scale_dims((200, 200), (200, 200), maximize=True) == (200, 200)

    @pytest.mark.parametrize(
        "dims", [(100, 50), (50, 100), (16, 9), (9, 16), (1000, 3), (3, 1000)]
    )
    @pytest.mark.parametrize("box", [(200, 200), (320, 180)])
    def test_the_result_always_fits_the_box(self, dims, box):
        """Growing to meet one side without checking the other overflows: a
        100x50 image in a 200x200 box became 400x200, cropping a fill-canvas
        animation in MediaFrame._update_gif_scale_mode."""
        width, height = scale_dims(dims, box, maximize=True)
        assert width <= box[0] and height <= box[1]

        width, height = scale_dims(dims, box, maximize=False)
        assert width <= box[0] and height <= box[1]

    @pytest.mark.parametrize("dims", [(1000, 3), (3, 1000)])
    def test_an_extreme_ratio_does_not_collapse_a_side_to_zero(self, dims):
        """A zero-width QMovie scale target or QSize is not renderable."""
        width, height = scale_dims(dims, (200, 200))
        assert width >= 1 and height >= 1

    @pytest.mark.parametrize("dims", [(0, 0), (0, 100), (100, 0), (-5, 10)])
    def test_degenerate_dimensions_do_not_divide_by_zero(self, dims):
        assert scale_dims(dims, (200, 200), maximize=True) == (200, 200)
        assert scale_dims(dims, (200, 200)) == (200, 200)


def test_is_large_image_dims_uses_config_threshold(monkeypatch):
    monkeypatch.setattr(config, "large_image_dim_threshold_px", 1000)
    assert is_large_image_dims((1001, 500))
    assert not is_large_image_dims((1000, 500))


def test_get_media_type_plain_image_when_enabled(monkeypatch, tmp_path):
    path = str(tmp_path / "photo.png")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(config, "enable_pdfs", False)
    assert get_media_type_for_path(path) == MediaType.IMAGE


class TestPagedDocumentPath:
    @pytest.mark.parametrize("name", ["book.epub", "BOOK.EPUB"])
    def test_epub_follows_its_flag(self, monkeypatch, name):
        monkeypatch.setattr(config, "enable_epubs", True)
        assert is_paged_document_path(name)
        monkeypatch.setattr(config, "enable_epubs", False)
        assert not is_paged_document_path(name)

    def test_pdf_follows_its_flag_only(self, monkeypatch):
        monkeypatch.setattr(config, "enable_epubs", False)
        monkeypatch.setattr(config, "enable_pdfs", True)
        assert is_paged_document_path("doc.pdf")
        monkeypatch.setattr(config, "enable_pdfs", False)
        monkeypatch.setattr(config, "enable_epubs", True)
        assert not is_paged_document_path("doc.pdf")

    @pytest.mark.parametrize("name", ["a.png", "a.html", "a.mp4", "", None])
    def test_other_paths_are_not_paged(self, monkeypatch, name):
        monkeypatch.setattr(config, "enable_pdfs", True)
        monkeypatch.setattr(config, "enable_epubs", True)
        assert not is_paged_document_path(name)

    def test_pdf_is_its_own_paged_pdf(self):
        assert get_paged_document_pdf("/x/doc.pdf") == "/x/doc.pdf"

    def test_epub_is_classifier_dynamic_media(self, monkeypatch, tmp_path):
        path = tmp_path / "book.epub"
        path.write_bytes(b"x")
        monkeypatch.setattr(config, "enable_epubs", True)
        assert is_classifier_dynamic_media_path(str(path))
        monkeypatch.setattr(config, "enable_epubs", False)
        assert not is_classifier_dynamic_media_path(str(path))

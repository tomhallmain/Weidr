"""Unit tests for utils/media_utils.py media-type classification."""

import pytest

from utils.config import config
from utils.constants import MediaType
from utils.media_utils import get_media_type_for_path


@pytest.mark.parametrize(
    "flag_name,ext,enabled_type,disabled_type",
    [
        ("enable_videos", ".mp4", MediaType.VIDEO, MediaType.UNCONFIGURED),
        ("enable_gifs", ".gif", MediaType.GIF, MediaType.UNCONFIGURED),
        ("enable_pdfs", ".pdf", MediaType.PDF, MediaType.UNCONFIGURED),
        ("enable_svgs", ".svg", MediaType.SVG, MediaType.UNCONFIGURED),
        ("enable_html", ".html", MediaType.HTML, MediaType.UNCONFIGURED),
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


def test_get_media_type_plain_image_when_enabled(monkeypatch, tmp_path):
    path = str(tmp_path / "photo.png")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(config, "enable_pdfs", False)
    assert get_media_type_for_path(path) == MediaType.IMAGE

"""Masonry grid thumbnail loading for audio files with embedded cover art."""

import os

from PIL import Image
from PySide6.QtGui import QImage

from ui.app_window.masonry_browser import _ThumbnailLoader, THUMB_MAX_DIM


def _make_jpg(path: str) -> str:
    Image.new("RGB", (40, 30), (200, 100, 50)).save(path, format="JPEG")
    return path


def test_audio_tile_loads_extracted_cover_art(tmp_path, qtbot, monkeypatch):
    """_ThumbnailLoader._load swaps in the extracted cover JPEG and decodes it
    normally, instead of returning a null QImage."""
    audio_path = os.path.join(tmp_path, "track.mp3")
    open(audio_path, "wb").close()
    cover_path = _make_jpg(os.path.join(tmp_path, "track_cover.jpg"))

    monkeypatch.setattr("ui.app_window.masonry_browser.is_audio_for_display", lambda p: True)
    monkeypatch.setattr("ui.app_window.masonry_browser._FRAME_CACHE_AVAILABLE", True)
    monkeypatch.setattr(
        "ui.app_window.masonry_browser.VideoOps.extract_attached_pic",
        lambda media_path, cache_dir: cover_path,
    )

    loader = _ThumbnailLoader(audio_path, THUMB_MAX_DIM)
    image = loader._load()

    assert isinstance(image, QImage)
    assert not image.isNull()
    assert image.width() > 0 and image.height() > 0


def test_audio_tile_without_cover_art_returns_null_image(tmp_path, qtbot, monkeypatch):
    """No attached_pic stream -> null QImage so the caller falls back to the ♪ placeholder."""
    audio_path = os.path.join(tmp_path, "track.mp3")
    open(audio_path, "wb").close()

    monkeypatch.setattr("ui.app_window.masonry_browser.is_audio_for_display", lambda p: True)
    monkeypatch.setattr("ui.app_window.masonry_browser._FRAME_CACHE_AVAILABLE", True)
    monkeypatch.setattr(
        "ui.app_window.masonry_browser.VideoOps.extract_attached_pic",
        lambda media_path, cache_dir: None,
    )

    loader = _ThumbnailLoader(audio_path, THUMB_MAX_DIM)
    image = loader._load()

    assert isinstance(image, QImage)
    assert image.isNull()

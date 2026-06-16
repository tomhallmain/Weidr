"""MediaFrame display of embedded audio cover art (attached_pic extraction)."""

import os

from PIL import Image

from image.video_ops import VideoOps


def _make_jpg(path: str) -> str:
    Image.new("RGB", (16, 16), (10, 20, 30)).save(path, format="JPEG")
    return path


def test_audio_with_cover_art_shows_extracted_image(
    media_frame, tmp_path, qtbot, monkeypatch
):
    """When extract_attached_pic finds a cover, it is displayed instead of the
    text placeholder, even when VLC itself is unavailable."""
    audio_path = os.path.join(tmp_path, "song.mp3")
    open(audio_path, "wb").close()
    cover_path = _make_jpg(os.path.join(tmp_path, "cover.jpg"))

    monkeypatch.setattr("ui.app_window.media_frame._VLC_AVAILABLE", False)
    monkeypatch.setattr(
        VideoOps, "extract_attached_pic", staticmethod(lambda media_path, cache_dir: cover_path)
    )

    media_frame.show_media(audio_path)
    qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=3000)

    assert media_frame.path == audio_path
    assert media_frame._graphics_view.isVisible()
    assert not media_frame._placeholder_label.isVisible()


def test_audio_without_cover_art_shows_text_placeholder(
    media_frame, tmp_path, qtbot, monkeypatch
):
    """When no attached_pic stream exists, the ♪/text placeholder is shown unchanged."""
    audio_path = os.path.join(tmp_path, "song.mp3")
    open(audio_path, "wb").close()

    monkeypatch.setattr("ui.app_window.media_frame._VLC_AVAILABLE", False)
    monkeypatch.setattr(
        VideoOps, "extract_attached_pic", staticmethod(lambda media_path, cache_dir: None)
    )

    media_frame.show_media(audio_path)
    qtbot.waitUntil(
        lambda: media_frame._placeholder_label.isVisible()
        and "song.mp3" in media_frame._placeholder_label.text(),
        timeout=3000,
    )

    assert "Audio:" in media_frame._placeholder_label.text()

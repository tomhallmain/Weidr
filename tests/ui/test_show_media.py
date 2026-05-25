"""
Parametrized MediaFrame.show_media coverage for common raster and video types.

Assets are created per test in tmp_path. Optional formats (HEIC, AVIF, animated
WebP, MP4) are skipped when Pillow, Qt, ffmpeg, or VLC are unavailable.
"""

import pytest
from PySide6.QtGui import QImageReader

from ui.app_window.media_frame import VideoUI, _VLC_AVAILABLE
from tests.fixtures.show_media_assets import (
    require_extension_in_config,
    show_media_files,
)

pytest.importorskip("PIL")

STATIC_KINDS = ("png", "jpg", "jpeg", "webp_static")
ANIMATED_KINDS = ("gif", "webp_animated")
OPTIONAL_IMAGE_KINDS = (
    ("heic", ".heic"),
    ("avif", ".avif"),
)


def _wait_static_displayed(media_frame, path, qtbot) -> None:
    qtbot.waitUntil(
        lambda: media_frame.media_displayed
        and media_frame.path == path
        and media_frame._gif_movie is None
        and not isinstance(media_frame._video_ui, VideoUI),
        timeout=8000,
    )


def _wait_animated_displayed(media_frame, qtbot) -> None:
    qtbot.waitUntil(
        lambda: media_frame.media_displayed
        and media_frame._gif_movie is not None
        and media_frame._gif_movie.isValid(),
        timeout=8000,
    )


@pytest.mark.parametrize("kind", STATIC_KINDS)
def test_show_media_static_raster(media_frame, show_media_files, kind, qtbot):
    path = show_media_files[kind]
    media_frame.show_media(path)
    _wait_static_displayed(media_frame, path, qtbot)
    assert not media_frame._placeholder_label.isVisible() or (
        media_frame._placeholder_label.text() == ""
    )


@pytest.mark.parametrize("kind", ANIMATED_KINDS)
def test_show_media_animated_raster(media_frame, show_media_files, kind, qtbot):
    path = show_media_files.get(kind)
    if not path:
        pytest.skip(f"No {kind} asset generated for this environment")

    if kind == "webp_animated":
        reader = QImageReader(path)
        if not reader.supportsAnimation():
            pytest.skip("Qt QImageReader does not support animated WebP here")

    media_frame.show_media(path)
    _wait_animated_displayed(media_frame, qtbot)
    assert media_frame.path == path


@pytest.mark.parametrize("kind,ext", OPTIONAL_IMAGE_KINDS)
def test_show_media_optional_modern_still_image(
    media_frame, show_media_files, kind, ext, qtbot
):
    require_extension_in_config(ext)
    path = show_media_files.get(kind)
    if not path:
        pytest.skip(f"Pillow cannot write {ext} in this environment")

    media_frame.show_media(path)
    _wait_static_displayed(media_frame, path, qtbot)


@pytest.mark.skipif(not _VLC_AVAILABLE, reason="python-vlc not installed")
def test_show_media_mp4_starts_video_ui(media_frame, show_media_files, qtbot):
    path = show_media_files.get("mp4")
    if not path:
        pytest.skip("ffmpeg not available to synthesize a short MP4")

    media_frame.show_media(path)
    qtbot.waitUntil(
        lambda: isinstance(media_frame._video_ui, VideoUI)
        and media_frame.path == path,
        timeout=10000,
    )


@pytest.mark.skipif(_VLC_AVAILABLE, reason="only applies when VLC is missing")
def test_show_media_mp4_placeholder_without_vlc(
    media_frame, show_media_files, qtbot
):
    path = show_media_files.get("mp4")
    if not path:
        pytest.skip("ffmpeg not available to synthesize a short MP4")

    media_frame.show_media(path)
    qtbot.waitUntil(
        lambda: "Video:" in media_frame._placeholder_label.text(),
        timeout=3000,
    )


class TestShowMediaSingleFrameStream:
    """show_media single-frame-stream branch: FrameCache drives static display + freeze-frame VLC."""

    def test_cached_frame_displayed_and_freeze_video_called(
        self, media_frame, show_media_files, qtbot, monkeypatch
    ):
        """When FrameCache marks a video as a single-frame stream AND a cached JPEG is
        present, show_media displays that frame as a static image and calls
        show_video(freeze_frame=True) so VLC plays audio without overpainting."""
        from image.frame_cache import FrameCache, MediaStats

        # Use a real PNG as the "video path" so os.path.exists passes; use the
        # JPEG as the separately cached frame to verify which path is displayed.
        video_path = show_media_files["png"]
        cached_frame = show_media_files["jpg"]

        FrameCache.media_stats_cache[video_path] = MediaStats(
            media_type="video", single_frame_stream=True
        )
        # Key format is "<media_path>|<ratio>"; get_any_cached_sampled_frame
        # scans for any key whose prefix matches.
        FrameCache.sampled_cache[f"{video_path}|0.1"] = [cached_frame]

        monkeypatch.setattr("ui.app_window.media_frame._VLC_AVAILABLE", True)
        monkeypatch.setattr(media_frame, "vlc_media_player", object())
        monkeypatch.setattr(
            "ui.app_window.media_frame.is_video_for_display", lambda p: True
        )
        show_video_calls = []
        monkeypatch.setattr(
            media_frame,
            "show_video",
            lambda path, freeze_frame=False: show_video_calls.append((path, freeze_frame)),
        )

        media_frame.show_media(video_path)

        qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=5000)
        assert show_video_calls == [(video_path, True)], show_video_calls

    def test_no_cached_frame_falls_through_to_normal_video(
        self, media_frame, show_media_files, monkeypatch
    ):
        """When the stream is flagged as single-frame but no cached JPEG exists yet,
        show_media falls through to a normal show_video call (no freeze_frame)."""
        from image.frame_cache import FrameCache, MediaStats

        video_path = show_media_files["png"]

        FrameCache.media_stats_cache[video_path] = MediaStats(
            media_type="video", single_frame_stream=True
        )
        # Intentionally no sampled_cache entry for this path.

        monkeypatch.setattr("ui.app_window.media_frame._VLC_AVAILABLE", True)
        monkeypatch.setattr(media_frame, "vlc_media_player", object())
        monkeypatch.setattr(
            "ui.app_window.media_frame.is_video_for_display", lambda p: True
        )
        show_video_calls = []
        monkeypatch.setattr(
            media_frame,
            "show_video",
            lambda path, freeze_frame=False: show_video_calls.append((path, freeze_frame)),
        )

        media_frame.show_media(video_path)

        assert show_video_calls == [(video_path, False)], show_video_calls

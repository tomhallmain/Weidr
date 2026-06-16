"""UI smoke tests for MediaDetails (metadata labels, refresh)."""

import os
import types

import pytest
from PySide6.QtWidgets import QApplication

from image.video_ops import VideoOps
from ui.image.media_details import MediaDetails


def _close_media_details(app_win) -> None:
    details = app_win.app_actions.media_details_window()
    if details is not None:
        try:
            details.close_windows()
            details.close()
            details.deleteLater()
        except RuntimeError:
            pass
        app_win.app_actions.set_media_details_window(None)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture
def mock_media_metadata(monkeypatch):
    """Stable prompt/model extraction for deterministic label text."""
    monkeypatch.setattr(
        "image.image_data_extractor.image_data_extractor.get_image_prompts_and_models",
        lambda _path: (
            "test positive prompt",
            "test negative prompt",
            ["model-alpha"],
            ["lora-beta"],
            False,
        ),
    )
    monkeypatch.setattr(
        "ui.image.media_details.MediaDetails.get_related_image_text",
        lambda _self: "related: (none)",
    )


@pytest.fixture
def media_details_window(window_with_dir, qtbot, bypass_password, mock_media_metadata):
    """Open MediaDetails for the first PNG in the loaded directory."""
    app_win, media_dir = window_with_dir
    media_path = app_win.file_browser.get_files()[0]

    app_win.window_launcher.open_media_details(media_path=media_path)
    qtbot.waitUntil(
        lambda: app_win.app_actions.media_details_window() is not None,
        timeout=5000,
    )
    details = app_win.app_actions.media_details_window()
    qtbot.addWidget(details)
    qtbot.waitExposed(details, timeout=3000)

    yield details, app_win, media_path, media_dir

    _close_media_details(app_win)


@pytest.fixture(autouse=True)
def _cleanup_media_details_singleton():
    yield
    MediaDetails.temp_media_canvas = None


class TestMediaDetailsWindow:
    def test_open_populates_path_dimensions_and_prompt_labels(
        self, media_details_window
    ):
        details, _app_win, media_path, _media_dir = media_details_window

        assert details._lbl_path.text() == media_path
        assert details._lbl_dims.text() == "10x10"
        assert details._lbl_mode.text() == "RGB"
        assert "test positive prompt" in details._lbl_positive.text()
        assert "model-alpha" in details._lbl_models.text()
        assert "lora-beta" in details._lbl_loras.text()
        assert "related:" in details._lbl_related_image.text()

    def test_update_media_details_refreshes_labels(
        self, media_details_window, qtbot
    ):
        details, app_win, _first_path, media_dir = media_details_window
        second_path = os.path.join(media_dir, "img02.png")
        index_text = app_win.file_browser.get_index_details()

        details.update_media_details(second_path, index_text)
        qtbot.waitUntil(
            lambda: details._lbl_path.text() == second_path,
            timeout=2000,
        )

        assert details._lbl_dims.text() == "10x10"
        assert details._lbl_path.text() == second_path


# ---------------------------------------------------------------------------
# Audio metadata gather — no Qt window needed
# ---------------------------------------------------------------------------

def _audio_stub(path: str) -> types.SimpleNamespace:
    """Minimal stand-in for a MediaDetails instance for the gather methods."""
    stub = types.SimpleNamespace(
        _media_path=path,
        get_related_image_text=lambda: "(Not available)",
    )
    stub._gather_audio_details = MediaDetails._gather_audio_details.__get__(stub, MediaDetails)
    stub._gather_video_details = MediaDetails._gather_video_details.__get__(stub, MediaDetails)
    return stub


class TestAudioMetadataGather:
    def test_audio_file_with_tags_shows_title_artist_album(self, monkeypatch):
        """A regular audio file whose tags include title/artist/album surfaces
        that info in the positive field and reports the codec + duration."""
        probe = {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "sample_rate": "44100",
                    "disposition": {"attached_pic": 0},
                }
            ],
            "format": {
                "duration": "225.0",   # 3:45
                "bit_rate": "800000",
                "tags": {
                    "title": "My Song",
                    "artist": "Test Artist",
                    "album": "Test Album",
                },
            },
        }
        monkeypatch.setattr(VideoOps, "find_ffprobe_executable", staticmethod(lambda: "ffprobe"))
        monkeypatch.setattr(VideoOps, "ffprobe_json", staticmethod(lambda _path: probe))

        stub = _audio_stub("song.flac")
        image_mode, image_dims, positive, negative, models, loras, _related, failed = (
            stub._gather_audio_details()
        )

        assert "flac" in image_mode.lower()
        assert "3:45" in image_dims
        assert "800 kbps" in image_dims
        assert "44,100 Hz" in image_dims
        assert "My Song" in positive
        assert "Test Artist" in positive
        assert "Test Album" in positive
        assert negative == ""
        assert models == []
        assert not failed

    def test_ambiguous_container_with_attached_pic_stream_shows_audio_details(
        self, monkeypatch
    ):
        """An M4A whose only 'video' stream is an embedded cover (attached_pic=1)
        must be treated as audio, not video, even when _gather_video_details is
        the entry point (i.e. media_type was wrongly classified as VIDEO)."""
        probe = {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "44100",
                    "disposition": {"attached_pic": 0},
                },
                {
                    "codec_type": "video",
                    "codec_name": "mjpeg",
                    "width": 1280,
                    "height": 720,
                    "disposition": {"attached_pic": 1},
                },
            ],
            "format": {
                "duration": "181.0",   # 3:01
                "bit_rate": "132861",
                "tags": {
                    "title": "The Farmers Song",
                    "artist": "Some Artist",
                    "major_brand": "M4A ",
                },
            },
        }
        monkeypatch.setattr(VideoOps, "find_ffprobe_executable", staticmethod(lambda: "ffprobe"))
        monkeypatch.setattr(VideoOps, "ffprobe_json", staticmethod(lambda _path: probe))

        stub = _audio_stub("song.m4a")
        # Call via _gather_video_details to simulate the misclassified-as-VIDEO path
        image_mode, image_dims, positive, negative, models, loras, _related, failed = (
            stub._gather_video_details()
        )

        assert "aac" in image_mode.lower(), f"Expected audio codec in mode, got: {image_mode!r}"
        assert "3:01" in image_dims, f"Expected duration in dims, got: {image_dims!r}"
        assert "The Farmers Song" in positive
        assert "Some Artist" in positive
        assert negative == ""
        assert not failed

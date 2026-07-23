"""UI smoke tests for MediaDetails (metadata labels, refresh)."""

import os
import types

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QPushButton

from image.frame_cache import FrameCache
from image.video_ops import VideoOps
from ui.image.media_details import MediaDetails
from utils.constants import CompareMediaType
from utils.translations import _


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


def _open_details_for(app_win, qtbot, media_path):
    """Open MediaDetails for an arbitrary path (not tied to app_win's base dir)."""
    app_win.window_launcher.open_media_details(media_path=media_path)
    qtbot.waitUntil(
        lambda: app_win.app_actions.media_details_window() is not None,
        timeout=5000,
    )
    details = app_win.app_actions.media_details_window()
    qtbot.addWidget(details)
    qtbot.waitExposed(details, timeout=3000)
    return details


def _button_texts(details) -> set:
    return {b.text() for b in details.findChildren(QPushButton)}


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


# ---------------------------------------------------------------------------
# _editable_image_path() — frame resolution for editing actions
# ---------------------------------------------------------------------------

def _editable_path_stub(media_path: str, media_type: CompareMediaType) -> types.SimpleNamespace:
    """Minimal stand-in for a MediaDetails instance for _editable_image_path()."""
    stub = types.SimpleNamespace(
        _media_path=media_path,
        _image_path=media_path,
        media_type=media_type,
    )
    stub._editable_image_path = MediaDetails._editable_image_path.__get__(stub, MediaDetails)
    return stub


class TestEditableImagePath:
    def test_plain_image_passes_through_image_path(self, show_media_files):
        """No FrameCache remap needed -- _image_path is already the source file."""
        png_path = show_media_files["png"]
        stub = _editable_path_stub(png_path, CompareMediaType.IMAGE)

        assert stub._editable_image_path() == png_path

    def test_gif_resolves_to_a_real_single_frame_raster(self, show_media_files):
        """FrameCache.get_image_path() deliberately leaves GIF's _image_path as
        the original animated file (so metadata/tags/file-stat keep seeing the
        real file) -- but editing ops (rotate/crop/etc.) need an actual static
        frame, so _editable_image_path() must resolve one on demand rather
        than handing ImageOps the raw multi-frame .gif."""
        gif_path = show_media_files["gif"]
        stub = _editable_path_stub(gif_path, CompareMediaType.GIF)

        resolved = stub._editable_image_path()

        assert resolved != gif_path
        assert os.path.isfile(resolved)
        with Image.open(resolved) as image:
            assert getattr(image, "n_frames", 1) == 1

    def test_gif_resolution_is_cached_across_calls(self, show_media_files):
        gif_path = show_media_files["gif"]
        stub = _editable_path_stub(gif_path, CompareMediaType.GIF)

        first = stub._editable_image_path()
        second = stub._editable_image_path()

        assert first == second


# ---------------------------------------------------------------------------
# _window_title_for_media_type() — dynamic window title per media type
# ---------------------------------------------------------------------------

def _title_stub(media_type: CompareMediaType) -> types.SimpleNamespace:
    """Minimal stand-in for a MediaDetails instance for _window_title_for_media_type()."""
    stub = types.SimpleNamespace(media_type=media_type)
    stub._window_title_for_media_type = (
        MediaDetails._window_title_for_media_type.__get__(stub, MediaDetails)
    )
    return stub


class TestWindowTitleForMediaType:
    def test_video_title(self):
        assert _title_stub(CompareMediaType.VIDEO)._window_title_for_media_type() == _("Video details")

    def test_audio_title(self):
        assert _title_stub(CompareMediaType.AUDIO)._window_title_for_media_type() == _("Audio details")

    def test_image_title(self):
        assert _title_stub(CompareMediaType.IMAGE)._window_title_for_media_type() == _("Image details")

    @pytest.mark.parametrize(
        "media_type",
        [
            CompareMediaType.GIF,
            CompareMediaType.PDF,
            CompareMediaType.SVG,
            CompareMediaType.HTML,
            CompareMediaType.UNCONFIGURED,
        ],
    )
    def test_other_types_fall_back_to_generic_media_details(self, media_type):
        assert _title_stub(media_type)._window_title_for_media_type() == _("Media details")

    def test_window_title_set_on_open(self, window, qtbot, bypass_password, show_media_files):
        details = _open_details_for(window, qtbot, show_media_files["png"])
        assert details.windowTitle() == _("Image details")

    def test_window_title_updates_on_media_switch(
        self, window, qtbot, bypass_password, show_media_files
    ):
        details = _open_details_for(window, qtbot, show_media_files["png"])
        assert details.windowTitle() == _("Image details")

        details.update_media_details(show_media_files["gif"], "1/2")
        qtbot.waitUntil(lambda: details.windowTitle() == _("Media details"), timeout=2000)


# ---------------------------------------------------------------------------
# _show_temp_path — "Temp Path" row visibility, including the HTML fix
# ---------------------------------------------------------------------------

class TestShowTempPathRow:
    def test_hidden_for_plain_image(self, window, qtbot, bypass_password, show_media_files):
        details = _open_details_for(window, qtbot, show_media_files["png"])

        assert details._show_temp_path is False
        assert not details._lbl_temp_path_header.isVisible()

    def test_shown_for_html(self, window, qtbot, bypass_password, show_media_files, monkeypatch):
        """Regression test: _show_temp_path used to only check .svg/.pdf, so an
        HTML source's cached raster path never appeared in the "Temp Path" row."""
        html_path = show_media_files["html"]
        resolved_cache_path = show_media_files["png"]
        monkeypatch.setattr(
            FrameCache, "get_image_path", classmethod(lambda _cls, _path: resolved_cache_path)
        )

        details = _open_details_for(window, qtbot, html_path)

        assert details._show_temp_path is True
        assert details._lbl_temp_path_header.isVisible()
        assert details._lbl_temp_path.text() == resolved_cache_path


# ---------------------------------------------------------------------------
# Editing grid stays visible for GIF (regression guard for _editable_image_path)
# ---------------------------------------------------------------------------

class TestEditingButtonsVisibleForGif:
    def test_editing_grid_shown_for_gif(self, window, qtbot, bypass_password, show_media_files):
        """GIF is a raster-supporting type (supports_raster_image_details() is
        True), so the editing grid stays visible -- editing a GIF now resolves
        to an actual frame via _editable_image_path() instead of being hidden
        or handed the raw multi-frame file."""
        details = _open_details_for(window, qtbot, show_media_files["gif"])

        texts = _button_texts(details)

        assert _("Rotate Image Left") in texts
        assert _("Crop Image (Smart Detect)") in texts

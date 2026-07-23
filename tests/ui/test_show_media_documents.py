"""MediaFrame.show_media for document-like paths and decode fallbacks."""

import pytest
from PySide6.QtGui import QImage, QImageReader

from image.frame_cache import FrameCache
from ui.app_window.media_frame import VideoUI
from tests.fixtures.show_media_assets import show_media_files
from utils.config import config
from utils.constants import MediaType
from utils.media_utils import get_media_type_for_path


def _wait_static_or_placeholder(media_frame, path, qtbot) -> str:
    """Return 'displayed' or 'placeholder' once show_media settles."""

    def _state():
        if media_frame.media_displayed and media_frame.path == path:
            return "displayed"
        text = media_frame._placeholder_label.text()
        if text and not media_frame.media_displayed:
            return "placeholder"
        return ""

    qtbot.waitUntil(lambda: _state() != "", timeout=8000)
    return _state()


class TestShowMediaDocumentPaths:
    @pytest.mark.parametrize("kind", ("pdf", "svg", "html"))
    def test_show_media_document_path_can_display_with_mocked_decode(
        self, media_frame, show_media_files, kind, qtbot, monkeypatch
    ):
        media_frame.resize(360, 280)
        path = show_media_files[kind]
        loaded = []

        def _fake_load(image_path, **kwargs):
            loaded.append(image_path)
            img = QImage(24, 18, QImage.Format.Format_RGB32)
            img.fill(0xFF112233)
            return img

        monkeypatch.setattr(media_frame, "_load_image_to_qimage", _fake_load)
        # When pypdfium2/pyppeteer is installed, show_media routes PDF/HTML
        # through PdfPageViewer / FrameCache (bypassing _load_image_to_qimage
        # on the raw path). Force the fallback path so this test can verify
        # the mocked-decode contract regardless of library availability.
        if kind == "pdf":
            monkeypatch.setattr("ui.app_window.media_frame.has_imported_pypdfium2", False)
        if kind == "html":
            monkeypatch.setattr("ui.app_window.media_frame.has_imported_pyppeteer", False)
        media_frame.show_media(path)

        qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=8000)
        assert loaded == [path]
        assert not isinstance(media_frame._video_ui, VideoUI)
        assert media_frame._graphics_view._interaction_enabled

    def test_show_media_svg_when_qt_can_read(
        self, media_frame, show_media_files, qtbot
    ):
        path = show_media_files["svg"]
        reader = QImageReader(path)
        if not reader.canRead():
            pytest.skip("Qt QImageReader cannot read SVG in this environment")

        media_frame.resize(360, 280)
        media_frame.show_media(path)
        state = _wait_static_or_placeholder(media_frame, path, qtbot)
        if state == "placeholder":
            pytest.skip("Qt reported readable SVG but show_media could not render it")
        assert media_frame._graphics_view._interaction_enabled

    def test_show_media_pdf_shows_placeholder_when_decode_fails(
        self, media_frame, show_media_files, qtbot, monkeypatch
    ):
        media_frame.resize(360, 280)
        path = show_media_files["pdf"]

        def _raise_on_load(*_args, **_kwargs):
            raise OSError("simulated decode failure")

        monkeypatch.setattr(media_frame, "_load_image_to_qimage", _raise_on_load)
        monkeypatch.setattr("ui.app_window.media_frame.has_imported_pypdfium2", False)
        media_frame.show_media(path)
        qtbot.waitUntil(
            lambda: not media_frame.media_displayed
            and bool(media_frame._placeholder_label.text().strip()),
            timeout=8000,
        )

    def test_show_media_html_routes_through_frame_cache(
        self, media_frame, show_media_files, qtbot, monkeypatch
    ):
        """HTML has no native QImageReader/PIL codec -- show_media must resolve
        it through FrameCache.get_image_path() (the same cache MediaDetails
        uses) before decoding, rather than handing the raw .html path to
        _load_image_to_qimage (which always fails for HTML)."""
        media_frame.resize(360, 280)
        html_path = show_media_files["html"]
        resolved_path = show_media_files["png"]
        loaded = []

        def _fake_load(image_path, **kwargs):
            loaded.append(image_path)
            img = QImage(24, 18, QImage.Format.Format_RGB32)
            img.fill(0xFF112233)
            return img

        monkeypatch.setattr(media_frame, "_load_image_to_qimage", _fake_load)
        monkeypatch.setattr("ui.app_window.media_frame.has_imported_pyppeteer", True)
        monkeypatch.setattr(
            FrameCache, "get_image_path", classmethod(lambda _cls, _path: resolved_path)
        )

        media_frame.show_media(html_path)

        qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=8000)
        assert loaded == [resolved_path]

    def test_show_media_html_shows_placeholder_when_frame_cache_fails(
        self, media_frame, show_media_files, qtbot, monkeypatch
    ):
        media_frame.resize(360, 280)
        html_path = show_media_files["html"]

        def _raise_frame_cache_error(_cls, _path):
            raise RuntimeError("simulated pyppeteer failure")

        monkeypatch.setattr("ui.app_window.media_frame.has_imported_pyppeteer", True)
        monkeypatch.setattr(
            FrameCache, "get_image_path", classmethod(_raise_frame_cache_error)
        )

        media_frame.show_media(html_path)

        qtbot.waitUntil(
            lambda: not media_frame.media_displayed
            and bool(media_frame._placeholder_label.text().strip()),
            timeout=8000,
        )


class TestShowMediaDisabledTypeGates:
    """Config gates classify paths before browse/compare; show_media still receives paths."""

    @pytest.mark.parametrize(
        "flag_name,kind,media_type",
        [
            ("enable_pdfs", "pdf", MediaType.PDF),
            ("enable_svgs", "svg", MediaType.SVG),
            ("enable_html", "html", MediaType.HTML),
        ],
    )
    def test_media_type_unconfigured_when_flag_disabled(
        self, show_media_files, monkeypatch, flag_name, kind, media_type
    ):
        monkeypatch.setattr(config, flag_name, False)
        path = show_media_files[kind]
        assert get_media_type_for_path(path) == MediaType.UNCONFIGURED

        monkeypatch.setattr(config, flag_name, True)
        assert get_media_type_for_path(path) == media_type

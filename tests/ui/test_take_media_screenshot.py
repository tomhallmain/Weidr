"""
Tests for AppWindow.take_media_screenshot and _copy_cached_media_to_clipboard.

_copy_cached_media_to_clipboard: resolves a FrameCache raster for PDF/SVG/HTML
and puts it on the system clipboard.

take_media_screenshot: dispatches based on media type —
  * time-based media (video/GIF) → save frame to file (not tested here; requires VLC)
  * PDF / SVG / HTML            → copy via _copy_cached_media_to_clipboard
  * plain raster                → warn user
"""

import os

import pytest
from PySide6.QtWidgets import QApplication

from image.frame_cache import FrameCache
from utils.constants import MediaType
from utils.translations import I18N

_tr = I18N._


# ---------------------------------------------------------------------------
# _copy_cached_media_to_clipboard
# ---------------------------------------------------------------------------

class TestCopyMediaToClipboard:
    def test_cache_hit_puts_image_on_clipboard(self, window_with_dir, monkeypatch):
        """A path already in FrameCache.cache is loaded and copied to the clipboard."""
        win, _ = window_with_dir
        png_path = win.file_browser.get_files()[0]

        # Seed the cache directly — get_cached_path returns cls.cache.get(path).
        FrameCache.cache["fake.pdf"] = png_path

        ok, err = win._copy_cached_media_to_clipboard("fake.pdf")

        assert ok, err
        assert err == ""
        img = QApplication.clipboard().image()
        assert not img.isNull()

    def test_cache_miss_falls_back_to_get_image_path(self, window_with_dir, monkeypatch):
        """When get_cached_path returns None, get_image_path is tried and used."""
        win, _ = window_with_dir
        jpg_path = win.file_browser.get_files()[0]

        # cache is empty (reset_app_globals clears FrameCache before each test)
        monkeypatch.setattr(FrameCache, "get_image_path", lambda p: jpg_path)

        ok, err = win._copy_cached_media_to_clipboard("fake.pdf")

        assert ok, err

    def test_no_cached_path_returns_error_message(self, window_with_dir, monkeypatch):
        """When both cache sources yield None the method returns False + a message."""
        win, _ = window_with_dir
        monkeypatch.setattr(FrameCache, "get_image_path", lambda p: None)

        ok, err = win._copy_cached_media_to_clipboard("fake.pdf")

        assert not ok
        assert err  # non-empty error string

    def test_get_image_path_exception_returned_as_error_string(
        self, window_with_dir, monkeypatch
    ):
        """If get_image_path raises, the exception message is surfaced as the error."""
        win, _ = window_with_dir

        def _raise(p):
            raise ImportError("pypdfium2 not installed")

        monkeypatch.setattr(FrameCache, "get_image_path", _raise)

        ok, err = win._copy_cached_media_to_clipboard("fake.pdf")

        assert not ok
        assert "pypdfium2" in err


# ---------------------------------------------------------------------------
# take_media_screenshot
# ---------------------------------------------------------------------------

class TestTakeMediaScreenshot:
    @pytest.mark.parametrize(
        "media_type,ext",
        [
            (MediaType.PDF, "pdf"),
            (MediaType.SVG, "svg"),
            (MediaType.HTML, "html"),
        ],
    )
    def test_document_type_copies_to_clipboard_and_shows_success(
        self, window_with_dir, monkeypatch, media_type, ext
    ):
        """For PDF/SVG/HTML active media, take_media_screenshot copies the FrameCache
        raster to the clipboard and shows a success notification."""
        win, media_dir = window_with_dir
        fake_doc = os.path.join(media_dir, f"doc.{ext}")  # need not exist on disk
        png_path = win.file_browser.get_files()[0]

        monkeypatch.setattr(
            win.media_navigator, "get_active_media_filepath", lambda: fake_doc
        )
        monkeypatch.setattr(win.media_frame, "has_time_based_media", lambda: False)
        # Patch at the source so the local import inside take_media_screenshot picks it up.
        monkeypatch.setattr(
            "utils.media_utils.get_media_type_for_path", lambda p: media_type
        )
        # Give the clipboard copy a real PNG to load.
        FrameCache.cache[fake_doc] = png_path

        successes = []
        monkeypatch.setattr(win.app_actions, "success", lambda msg: successes.append(msg))

        win.take_media_screenshot()

        assert successes, "expected app_actions.success() to be called"
        img = QApplication.clipboard().image()
        assert not img.isNull()

    def test_document_type_shows_error_alert_when_clipboard_fails(
        self, window_with_dir, monkeypatch
    ):
        """When the clipboard copy fails, an error alert is shown and success is not called."""
        win, media_dir = window_with_dir
        fake_pdf = os.path.join(media_dir, "broken.pdf")

        monkeypatch.setattr(
            win.media_navigator, "get_active_media_filepath", lambda: fake_pdf
        )
        monkeypatch.setattr(win.media_frame, "has_time_based_media", lambda: False)
        monkeypatch.setattr(
            "utils.media_utils.get_media_type_for_path", lambda p: MediaType.PDF
        )
        # Both cache sources return nothing → copy will fail.
        monkeypatch.setattr(FrameCache, "get_image_path", lambda p: None)

        alerts = []
        monkeypatch.setattr(
            win.notification_ctrl,
            "alert",
            lambda title, msg, kind=None: alerts.append((title, kind)),
        )
        successes = []
        monkeypatch.setattr(win.app_actions, "success", lambda msg: successes.append(msg))

        win.take_media_screenshot()

        assert alerts, "expected an error alert"
        assert alerts[0][1] == "error"
        assert not successes

    def test_plain_raster_shows_warning(self, window_with_dir, monkeypatch):
        """A plain raster image (not video, not PDF/SVG/HTML) triggers a user warning."""
        win, _ = window_with_dir
        png_path = win.file_browser.get_files()[0]  # .png → MediaType.IMAGE naturally

        monkeypatch.setattr(
            win.media_navigator, "get_active_media_filepath", lambda: png_path
        )
        monkeypatch.setattr(win.media_frame, "has_time_based_media", lambda: False)

        warnings = []
        monkeypatch.setattr(win.app_actions, "warn", lambda msg: warnings.append(msg))

        win.take_media_screenshot()

        assert warnings, "expected app_actions.warn() for a plain raster"

    def test_no_active_path_is_no_op(self, window_with_dir, monkeypatch):
        """When there is no active media file, take_media_screenshot does nothing."""
        win, _ = window_with_dir
        monkeypatch.setattr(
            win.media_navigator, "get_active_media_filepath", lambda: None
        )
        alerts = []
        monkeypatch.setattr(
            win.notification_ctrl, "alert", lambda *a, **k: alerts.append(a)
        )
        warnings = []
        monkeypatch.setattr(win.app_actions, "warn", lambda msg: warnings.append(msg))

        win.take_media_screenshot()  # must not raise

        assert not alerts
        assert not warnings

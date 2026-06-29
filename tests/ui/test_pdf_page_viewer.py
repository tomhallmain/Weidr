"""
Tests for PdfPageViewer and the MediaFrame / AppWindow PDF integration.

Covers:
- PdfPageViewer.navigate(): forward, backward, clamping at boundaries
- PdfPageViewer.deactivate(): resets state and hides label
- PdfPageViewer.current_jpeg_path(): tracks the displayed page JPEG
- MediaFrame.show_media PDF dispatch (with pypdfium2 mocked)
- MediaFrame.pdf_navigate / pdf_current_page_path delegation
- _copy_cached_media_to_clipboard: prefers current page over page-0 cache
- interactive_crop: routes PDF through current page JPEG
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from image.frame_cache import FrameCache
from ui.app_window.pdf_page_viewer import PdfPageViewer
from utils.constants import MediaType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_viewer(media_frame):
    """Return a PdfPageViewer wired to *media_frame* with _show_image_in_view stubbed."""
    media_frame._show_image_in_view = MagicMock()
    viewer = PdfPageViewer(media_frame)
    return viewer


@contextmanager
def _mock_frame_cache(path: str, jpeg: str, total_pages: int):
    """Patch FrameCache at the source so _render() sees mocked get_pdf_page + stats."""
    stats = MagicMock()
    stats.total_items = total_pages
    with patch.object(FrameCache, "get_pdf_page", return_value=jpeg), \
         patch.object(FrameCache, "media_stats_cache", {path: stats}):
        yield


def _activate_viewer(viewer, path="fake.pdf", total_pages=5):
    """Simulate a successful render of page 0 so the viewer is 'active'."""
    jpeg = f"/tmp/{path}_page_0.jpg"
    with _mock_frame_cache(path, jpeg, total_pages):
        viewer.show(path)
    return jpeg


# ---------------------------------------------------------------------------
# PdfPageViewer unit tests
# ---------------------------------------------------------------------------

class TestPdfPageViewerState:
    def test_initially_inactive(self, media_frame):
        viewer = _make_viewer(media_frame)
        assert not viewer.is_active

    def test_active_after_successful_show(self, media_frame):
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer)
        assert viewer.is_active

    def test_current_jpeg_path_none_when_inactive(self, media_frame):
        viewer = _make_viewer(media_frame)
        assert viewer.current_jpeg_path() is None

    def test_current_jpeg_path_set_after_show(self, media_frame):
        viewer = _make_viewer(media_frame)
        jpeg = _activate_viewer(viewer, path="doc.pdf")
        assert viewer.current_jpeg_path() == jpeg

    def test_deactivate_clears_state(self, media_frame):
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer)
        viewer.deactivate()
        assert not viewer.is_active
        assert viewer.current_jpeg_path() is None

    def test_deactivate_hides_label(self, media_frame):
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, total_pages=3)
        viewer.deactivate()
        assert not viewer._label.isVisible()


class TestPdfPageViewerNavigation:
    def test_navigate_forward_increments_page(self, media_frame):
        path = "fake.pdf"
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, path=path, total_pages=5)

        with _mock_frame_cache(path, "/tmp/page_1.jpg", 5):
            viewer.navigate(1)

        assert viewer._page_index == 1

    def test_navigate_backward_decrements_page(self, media_frame):
        path = "fake.pdf"
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, path=path, total_pages=5)

        for page in [1, 2]:
            with _mock_frame_cache(path, f"/tmp/page_{page}.jpg", 5):
                viewer.navigate(1)

        with _mock_frame_cache(path, "/tmp/page_1.jpg", 5):
            viewer.navigate(-1)

        assert viewer._page_index == 1

    def test_navigate_forward_clamped_at_last_page(self, media_frame):
        path = "fake.pdf"
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, path=path, total_pages=2)

        with _mock_frame_cache(path, "/tmp/page_1.jpg", 2):
            viewer.navigate(1)
        assert viewer._page_index == 1

        with _mock_frame_cache(path, "/tmp/page_1.jpg", 2):
            viewer.navigate(1)
        assert viewer._page_index == 1

    def test_navigate_backward_clamped_at_first_page(self, media_frame):
        path = "fake.pdf"
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, path=path, total_pages=3)

        with _mock_frame_cache(path, "/tmp/page_0.jpg", 3):
            viewer.navigate(-1)
        assert viewer._page_index == 0

    def test_navigate_no_op_when_inactive(self, media_frame):
        viewer = _make_viewer(media_frame)
        with patch.object(FrameCache, "get_pdf_page") as mock_gpdf:
            viewer.navigate(1)
            mock_gpdf.assert_not_called()

    def test_navigate_updates_current_jpeg_path(self, media_frame):
        path = "fake.pdf"
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, path=path, total_pages=3)

        new_jpeg = "/tmp/page_1.jpg"
        with _mock_frame_cache(path, new_jpeg, 3):
            viewer.navigate(1)

        assert viewer.current_jpeg_path() == new_jpeg

    def test_failed_render_leaves_page_unchanged(self, media_frame):
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, total_pages=3)

        with patch.object(FrameCache, "get_pdf_page", side_effect=OSError("disk full")):
            viewer.navigate(1)

        assert viewer._page_index == 0

    def test_label_hidden_for_single_page_pdf(self, media_frame):
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, total_pages=1)
        assert not viewer._label.isVisible()

    def test_label_visible_for_multipage_pdf(self, media_frame):
        viewer = _make_viewer(media_frame)
        _activate_viewer(viewer, total_pages=3)
        QApplication.processEvents()
        assert viewer._label.isVisible()


# ---------------------------------------------------------------------------
# MediaFrame.pdf_navigate / pdf_current_page_path
# ---------------------------------------------------------------------------

class TestMediaFramePdfDelegation:
    def test_pdf_navigate_no_op_when_no_viewer(self, media_frame):
        media_frame.pdf_navigate(1)  # must not raise

    def test_pdf_current_page_path_none_when_no_viewer(self, media_frame):
        assert media_frame.pdf_current_page_path() is None

    def test_pdf_navigate_delegates_to_viewer(self, media_frame):
        mock_viewer = MagicMock()
        media_frame._pdf_viewer = mock_viewer
        media_frame.pdf_navigate(1)
        mock_viewer.navigate.assert_called_once_with(1)

    def test_pdf_current_page_path_delegates_to_viewer(self, media_frame):
        mock_viewer = MagicMock()
        mock_viewer.current_jpeg_path.return_value = "/tmp/page_2.jpg"
        media_frame._pdf_viewer = mock_viewer
        assert media_frame.pdf_current_page_path() == "/tmp/page_2.jpg"

    def test_show_media_pdf_deactivates_previous_viewer(self, media_frame, monkeypatch):
        """Switching away from PDF deactivates the viewer."""
        mock_viewer = MagicMock()
        mock_viewer.is_active = True
        media_frame._pdf_viewer = mock_viewer

        monkeypatch.setattr("ui.app_window.media_frame.has_imported_pypdfium2", False)
        with patch("os.path.exists", return_value=True):
            media_frame.show_media("image.png")

        mock_viewer.deactivate.assert_called()


# ---------------------------------------------------------------------------
# _copy_cached_media_to_clipboard: current page preferred over page-0 cache
# ---------------------------------------------------------------------------

class TestCopyClipboardCurrentPage:
    def test_uses_current_pdf_page_not_page_zero(self, window_with_dir, monkeypatch):
        """When a PDF viewer is active on a non-zero page, that page's JPEG is
        copied — not the page-0 entry in FrameCache.cache."""
        win, media_dir = window_with_dir
        import os
        import tempfile

        from PIL import Image

        page0_path = os.path.join(tempfile.gettempdir(), "weidr_test_page0.png")
        page3_path = os.path.join(tempfile.gettempdir(), "weidr_test_page3.png")
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(page0_path)
        Image.new("RGB", (8, 8), color=(0, 255, 0)).save(page3_path)

        fake_pdf = "multi.pdf"
        FrameCache.cache[fake_pdf] = page0_path

        monkeypatch.setattr(win.media_frame, "pdf_current_page_path", lambda: page3_path)

        ok, err = win._copy_cached_media_to_clipboard(fake_pdf)
        assert ok, err

        from PySide6.QtGui import QColor
        clipboard_img = QApplication.clipboard().image()
        pix = QColor(clipboard_img.pixel(0, 0))
        assert pix.green() > pix.red(), "Expected current page (green) not page-0 (red)"

    def test_falls_back_to_cache_when_no_pdf_viewer_active(self, window_with_dir, monkeypatch):
        """When pdf_current_page_path() returns None, the normal cache path is used."""
        win, _ = window_with_dir
        png_path = win.file_browser.get_files()[0]

        monkeypatch.setattr(win.media_frame, "pdf_current_page_path", lambda: None)
        FrameCache.cache["fallback.pdf"] = png_path

        ok, err = win._copy_cached_media_to_clipboard("fallback.pdf")
        assert ok, err


# ---------------------------------------------------------------------------
# interactive_crop: PDF source path uses current page
# ---------------------------------------------------------------------------

class TestInteractiveCropPdfPage:
    def test_crop_uses_current_pdf_page_not_get_image_path(
        self, window_with_dir, monkeypatch, bypass_password
    ):
        """interactive_crop should use pdf_current_page_path() for PDFs, not
        FrameCache.get_image_path(), so the crop operates on the visible page."""
        win, media_dir = window_with_dir
        import os
        import tempfile

        from PIL import Image

        current_page_jpeg = os.path.join(tempfile.gettempdir(), "weidr_current_page.jpg")
        page0_jpeg = os.path.join(tempfile.gettempdir(), "weidr_page0.jpg")
        for p in (current_page_jpeg, page0_jpeg):
            Image.new("RGB", (8, 8)).save(p)

        fake_pdf = os.path.join(media_dir, "test.pdf")
        open(fake_pdf, "wb").close()

        monkeypatch.setattr(win, "media_path", fake_pdf)
        monkeypatch.setattr(
            "utils.media_utils.get_media_type_for_path", lambda p: MediaType.PDF
        )
        monkeypatch.setattr(
            win.media_frame, "pdf_current_page_path", lambda: current_page_jpeg
        )
        get_image_path_calls = []
        monkeypatch.setattr(
            FrameCache, "get_image_path",
            lambda p: get_image_path_calls.append(p) or page0_jpeg,
        )
        monkeypatch.setattr(win.media_frame, "_show_image_in_view", MagicMock())
        monkeypatch.setattr(win.media_frame._graphics_view, "start_crop_mode", MagicMock())

        win.window_launcher.interactive_crop()

        assert not get_image_path_calls, (
            "FrameCache.get_image_path was called but should have been bypassed "
            "in favour of pdf_current_page_path()"
        )

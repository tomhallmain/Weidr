"""
PdfPageViewer — manages PDF and ePub page-by-page navigation within a MediaFrameQt.

All page rendering logic and page state live here; MediaFrameQt only calls
``show(path)`` on new files and ``handle_key(key)`` for Up/Down navigation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QLabel

from ui.app_style import AppStyle
from utils.logging_setup import get_logger
from utils.media_utils import is_epub_path
from utils.translations import _

if TYPE_CHECKING:
    from ui.app_window.media_frame import MediaFrameQt

logger = get_logger("pdf_page_viewer")

_LABEL_STYLE = (
    f"background: rgba(0,0,0,160); color: {AppStyle.FG_COLOR}; "
    "border-radius: 4px; padding: 4px 10px; font-size: 11pt;"
)


class _EpubBuildWorker(QThread):
    """Builds one ePub's derived PDF (FrameCache.get_epub_pdf) off the UI thread."""

    finished_build = Signal(str)

    def __init__(self, epub_path: str, parent=None):
        super().__init__(parent)
        self._epub_path = epub_path

    def run(self) -> None:
        from image.frame_cache import FrameCache
        try:
            FrameCache.get_epub_pdf(self._epub_path)
        except Exception as exc:
            logger.warning("ePub build failed path=%s: %s", self._epub_path, exc)
        self.finished_build.emit(self._epub_path)


class PdfPageViewer:
    """Controller that renders PDF and ePub pages into a MediaFrameQt on demand.

    Page JPEGs are written to FrameCache's temp directory and reused on
    revisits; the PDF document itself is not kept open between renders.

    An ePub is paged through a PDF rendered from the whole book. Until that
    build lands, the cover is shown, the label has no page total, and Up/Down
    presses are queued and applied once it finishes.
    """

    def __init__(self, media_frame: MediaFrameQt) -> None:
        self._frame = media_frame
        self._path: str | None = None
        self._page_index: int = 0
        self._total_pages: int = 0
        self._current_jpeg: str | None = None
        self._building_path: str | None = None
        self._queued_delta: int = 0
        # Running workers are kept referenced until they finish; a QThread
        # destroyed while running aborts the process.
        self._workers: set[_EpubBuildWorker] = set()

        self._label = QLabel(media_frame)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(_LABEL_STYLE)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._label.hide()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._total_pages > 0

    @property
    def is_building(self) -> bool:
        return self._building_path is not None

    def show(self, path: str) -> None:
        """Open *path* at page 0."""
        self._path = path
        self._page_index = 0
        self._building_path = None
        self._queued_delta = 0
        if is_epub_path(path) and self._start_epub_build(path):
            return
        self._render(0)

    def navigate(self, delta: int) -> None:
        """Advance by *delta* pages (+1 forward, -1 back). No-op when inactive."""
        if self.is_building:
            self._queued_delta += 1 if delta > 0 else -1
            return
        if not self.is_active:
            return
        target = (min(self._page_index + 1, self._total_pages - 1) if delta > 0
                  else max(self._page_index - 1, 0))
        self._render(target)

    def current_jpeg_path(self) -> str | None:
        """Return the JPEG path of the currently displayed page, or None if inactive."""
        return self._current_jpeg if self.is_active else None

    def deactivate(self) -> None:
        """Hide the page label and mark as inactive. A pending ePub build's result is discarded."""
        self._total_pages = 0
        self._current_jpeg = None
        self._building_path = None
        self._queued_delta = 0
        self._label.hide()

    def reposition_label(self) -> None:
        """Re-place the page label after the parent widget is resized."""
        if self._label.isVisible():
            self._place_label()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_epub_build(self, path: str) -> bool:
        """Show the cover and build the ePub's PDF in the background.

        Returns False when there is nothing to wait for (already built, or known
        to be unpageable) and the caller should render page 0 directly.
        """
        from image.frame_cache import FrameCache
        if FrameCache.get_cached_epub_pdf(path) or FrameCache.is_epub_unpageable(path):
            return False
        cover = FrameCache.get_epub_cover_path(path)
        if FrameCache.is_epub_unpageable(path):
            return False
        self._building_path = path
        if cover:
            self._frame._show_image_in_view(cover)
        else:
            self._frame._show_placeholder(_("Rendering ePub…"))
        self._label.setText(_("Page {0} / …").format(1))
        self._label.adjustSize()
        self._place_label()
        self._label.show()
        self._label.raise_()

        worker = _EpubBuildWorker(path, parent=self._frame)
        worker.finished_build.connect(self._on_epub_build_finished)
        worker.finished.connect(lambda w=worker: self._release_worker(w))
        self._workers.add(worker)
        worker.start()
        return True

    def _release_worker(self, worker: _EpubBuildWorker) -> None:
        self._workers.discard(worker)
        worker.deleteLater()

    def _on_epub_build_finished(self, path: str) -> None:
        if path != self._building_path or path != self._path:
            return  # the user moved on; the build stays cached for a revisit
        from image.frame_cache import FrameCache
        self._building_path = None
        queued = self._queued_delta
        self._queued_delta = 0
        self._label.hide()
        stats = FrameCache.media_stats_cache.get(path)
        total = (stats.total_items or 1) if stats else 1
        self._render(max(0, min(queued, total - 1)))

    def _render(self, page_index: int) -> None:
        from image.frame_cache import FrameCache
        try:
            jpeg_path = FrameCache.get_pdf_page(self._path, page_index)
        except Exception as exc:
            logger.warning(
                "PDF page render failed path=%s page=%s: %s", self._path, page_index, exc
            )
            return
        stats = FrameCache.media_stats_cache.get(self._path)
        self._total_pages = (stats.total_items or 0) if stats else 0
        self._page_index = page_index
        self._current_jpeg = jpeg_path
        self._frame._show_image_in_view(jpeg_path)
        self._update_label()

    def _update_label(self) -> None:
        if self._total_pages < 2:
            self._label.hide()
            return
        self._label.setText(
            _("Page {0} / {1}").format(self._page_index + 1, self._total_pages)
        )
        self._label.adjustSize()
        self._place_label()
        self._label.show()
        self._label.raise_()

    def _place_label(self) -> None:
        hint = self._label.sizeHint()
        x = max(0, (self._frame.width() - hint.width()) // 2)
        y = max(0, self._frame.height() - hint.height() - 14)
        self._label.setGeometry(x, y, hint.width(), hint.height())

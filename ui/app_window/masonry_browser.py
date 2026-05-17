"""
MasonryBrowser -- thumbnail grid panel for browse mode.

Shows a masonry-style (variable-height, multi-column) grid of file thumbnails.
Thumbnails are decoded asynchronously via QThreadPool so the UI stays responsive.
Tile count is capped (TILE_CAP) to keep memory bounded; this is the §4.3
"bounded window" approach described in docs/masonry-layout-prospect.md and is
expected to be replaced with full virtualization in a later phase.

Wheel events are consumed by the QScrollArea so they scroll the grid rather
than triggering AppWindow's file-navigation handler.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, QObject, QRunnable, QSize, Signal, QThreadPool, QTimer
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.app_style import AppStyle
from utils.logging_setup import get_logger

try:
    from PIL import Image as PilImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from image.frame_cache import FrameCache
    _FRAME_CACHE_AVAILABLE = True
except ImportError:
    _FRAME_CACHE_AVAILABLE = False

logger = get_logger("masonry_browser")

# Maximum number of tiles rendered per populate() call.
TILE_CAP = 500
# Maximum dimension (px) for thumbnail decoding.
THUMB_MAX_DIM = 220
# Default number of columns.
DEFAULT_COLUMNS = 4
# Spacing between tiles (px).
TILE_MARGIN = 6
# Max concurrent thumbnail decode threads. Kept small to avoid saturating the
# global pool and to limit concurrent I/O on slow/external drives.
_THUMB_MAX_THREADS = 4

# Dedicated thread pool for thumbnail decoding — isolated from globalInstance()
# so we can call clear() on repopulate without affecting other app components.
_thumb_pool: Optional[QThreadPool] = None


def _get_thumb_pool() -> QThreadPool:
    global _thumb_pool
    if _thumb_pool is None:
        _thumb_pool = QThreadPool()
        _thumb_pool.setMaxThreadCount(_THUMB_MAX_THREADS)
    return _thumb_pool


# ---------------------------------------------------------------------------
# Async thumbnail loading
# ---------------------------------------------------------------------------

class _ThumbnailSignals(QObject):
    # filepath, QImage (null QImage on failure)
    loaded = Signal(str, object)


class _ThumbnailLoader(QRunnable):
    """
    Decode one thumbnail on a pool thread and emit a QImage via signal.

    Strategy (in order):
      1. QImageReader — fast, covers JPEG/PNG/GIF/BMP/WEBP.
      2. Pillow — covers HEIC, AVIF, TIFF and other formats Qt may miss.
      3. FrameCache.get_image_path — extracts a still from video containers.
    Emits a null QImage on total failure so callers show a placeholder.
    """

    def __init__(self, filepath: str, max_dim: int) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._filepath = filepath
        self._max_dim = max_dim
        self.signals = _ThumbnailSignals()

    def run(self) -> None:
        qimage = self._load()
        self.signals.loaded.emit(self._filepath, qimage)

    def _load(self) -> QImage:
        path = self._filepath

        # --- QImageReader (fast path) ---
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        if reader.canRead():
            sz = reader.size()
            if sz.isValid() and sz.width() > 0 and sz.height() > 0:
                scale = self._max_dim / max(sz.width(), sz.height())
                if scale < 1.0:
                    reader.setScaledSize(QSize(int(sz.width() * scale), int(sz.height() * scale)))
            img = reader.read()
            if not img.isNull():
                return img

        # --- Pillow fallback ---
        if _PIL_AVAILABLE:
            try:
                pil = PilImage.open(path)
                pil.thumbnail((self._max_dim, self._max_dim * 2))
                pil = pil.convert("RGBA")
                data = pil.tobytes("raw", "RGBA")
                return QImage(data, pil.width, pil.height, pil.width * 4,
                              QImage.Format.Format_RGBA8888)
            except Exception:
                pass

        # --- FrameCache fallback (video first frame) ---
        if _FRAME_CACHE_AVAILABLE:
            try:
                preview_path = FrameCache.get_image_path(path)
                if preview_path and os.path.isfile(preview_path):
                    reader2 = QImageReader(preview_path)
                    reader2.setAutoTransform(True)
                    img2 = reader2.read()
                    if not img2.isNull():
                        return img2.scaled(
                            self._max_dim, self._max_dim * 2,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
            except Exception:
                pass

        return QImage()  # null — caller shows placeholder


# ---------------------------------------------------------------------------
# Individual tile
# ---------------------------------------------------------------------------

class MasonryTile(QFrame):
    """
    Single cell in the masonry grid.

    Displays a thumbnail (loaded asynchronously) and the file basename.
    Emits ``activated(filepath)`` on left-click.
    """

    activated = Signal(str)

    _ACTIVE_STYLE = (
        f"MasonryTile {{ border: 2px solid {AppStyle.PROGRESS_CHUNK}; "
        f"background-color: {AppStyle.BG_BUTTON}; border-radius: 4px; }}"
    )
    _INACTIVE_STYLE = (
        f"MasonryTile {{ border: 1px solid {AppStyle.BORDER_COLOR}; "
        f"background-color: {AppStyle.BG_COLOR}; border-radius: 4px; }}"
    )

    def __init__(self, filepath: str, tile_width: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._filepath = filepath
        self._tile_width = tile_width
        self._cancelled = False

        self.setFixedWidth(tile_width)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._INACTIVE_STYLE)
        self.setToolTip(filepath)

        inner = QVBoxLayout(self)
        inner.setContentsMargins(3, 3, 3, 3)
        inner.setSpacing(2)

        # Image area — starts as a square placeholder, resizes after load
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setFixedSize(tile_width - 6, tile_width - 6)
        self._img_label.setStyleSheet(f"background-color: {AppStyle.BG_INPUT};")
        inner.addWidget(self._img_label)

        # Filename — allow more characters for wider tiles (~6 px/char at 10 px font)
        name = os.path.basename(filepath)
        max_chars = max(20, (tile_width - 12) // 6)
        if len(name) > max_chars:
            name = name[:max_chars - 1] + "…"
        self._name_label = QLabel(name)
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 10px;")
        self._name_label.setWordWrap(False)
        inner.addWidget(self._name_label)

        # Kick off async decode on the dedicated masonry pool
        loader = _ThumbnailLoader(filepath, THUMB_MAX_DIM)
        loader.signals.loaded.connect(self._on_thumbnail_loaded)
        _get_thumb_pool().start(loader)

    @property
    def filepath(self) -> str:
        return self._filepath

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(self._ACTIVE_STYLE if active else self._INACTIVE_STYLE)

    def _on_thumbnail_loaded(self, filepath: str, qimage: object) -> None:
        if self._cancelled:
            return
        if not isinstance(qimage, QImage) or qimage.isNull():
            return
        pixmap = QPixmap.fromImage(qimage)
        w = self._tile_width - 6
        # Always scale to tile width — scales both down and up to fill the cell
        pixmap = pixmap.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation)
        self._img_label.setPixmap(pixmap)
        # Adjust label height to match the actual thumbnail aspect ratio
        h = max(30, min(pixmap.height(), THUMB_MAX_DIM * 2))
        self._img_label.setFixedSize(w, h)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self._filepath)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Masonry container
# ---------------------------------------------------------------------------

class MasonryBrowser(QScrollArea):
    """
    Scrollable masonry thumbnail grid.

    Tile count is capped at TILE_CAP to keep memory bounded (phase 1).
    Files are distributed across N columns using a shortest-column-first
    heuristic.  Since tile heights are not known until thumbnails load,
    the initial placement approximates masonry; the final layout is correct
    once all thumbnails have loaded and the column QVBoxLayouts have settled.

    Signals
    -------
    tile_activated(filepath) : emitted when the user clicks a tile.
    """

    tile_activated = Signal(str)

    def __init__(self, columns: int = DEFAULT_COLUMNS,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._columns = columns
        self._tiles: list[MasonryTile] = []
        self._current_file: Optional[str] = None

        # Canvas widget that holds the column strip
        self._canvas = QWidget()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        canvas_layout = QHBoxLayout(self._canvas)
        canvas_layout.setContentsMargins(TILE_MARGIN, TILE_MARGIN, TILE_MARGIN, TILE_MARGIN)
        canvas_layout.setSpacing(TILE_MARGIN)

        # One QWidget per column, each with a top-aligned VBoxLayout
        self._col_widgets: list[QWidget] = []
        self._col_layouts: list[QVBoxLayout] = []
        self._col_heights: list[int] = []
        for _ in range(columns):
            col = QWidget()
            col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            col_layout = QVBoxLayout(col)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(TILE_MARGIN)
            col_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self._col_widgets.append(col)
            self._col_layouts.append(col_layout)
            self._col_heights.append(0)
            canvas_layout.addWidget(col)

        self.setWidget(self._canvas)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet(f"QScrollArea {{ background-color: {AppStyle.MEDIA_BG}; border: none; }}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, filepaths: list[str], current_file: Optional[str] = None) -> None:
        """
        Rebuild the grid from *filepaths* (already sorted by the file browser).

        Capped at TILE_CAP entries.  *current_file* is highlighted and scrolled
        into view after the first layout pass.
        """
        self._clear()
        self._current_file = current_file
        self._col_heights = [0] * self._columns

        tile_width = self._compute_tile_width()
        capped = filepaths[:TILE_CAP]

        # Suppress intermediate repaints and layout passes during bulk creation.
        self._canvas.setUpdatesEnabled(False)
        try:
            for filepath in capped:
                col_idx = self._col_heights.index(min(self._col_heights))
                tile = MasonryTile(filepath, tile_width, parent=self._col_widgets[col_idx])
                tile.activated.connect(self.tile_activated)
                self._col_layouts[col_idx].addWidget(tile)
                self._tiles.append(tile)
                # Approximate height for placement heuristic (square placeholder + label)
                self._col_heights[col_idx] += tile_width + 20 + TILE_MARGIN
        finally:
            self._canvas.setUpdatesEnabled(True)

        self._highlight_current()
        # Defer scroll-to-current until after the layout pass
        QTimer.singleShot(0, self._scroll_to_current)

    def update_current(self, current_file: Optional[str]) -> None:
        """Update which tile appears highlighted without repopulating."""
        self._current_file = current_file
        self._highlight_current()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_tile_width(self) -> int:
        vw = self.viewport().width()
        if vw < 10:
            vw = 800  # fallback before first show
        usable = vw - TILE_MARGIN * (self._columns + 1)
        return max(80, usable // self._columns)

    def _clear(self) -> None:
        # Drop queued-but-not-started tasks before touching the tile list so
        # cancelled tasks do not race with the new populate() call.
        _get_thumb_pool().clear()
        self._canvas.setUpdatesEnabled(False)
        try:
            for tile in self._tiles:
                tile._cancelled = True
                tile.setParent(None)  # detaches from column layout immediately
                tile.deleteLater()
            self._tiles.clear()
        finally:
            self._canvas.setUpdatesEnabled(True)
        self._col_heights = [0] * self._columns

    def _highlight_current(self) -> None:
        for tile in self._tiles:
            tile.set_active(tile.filepath == self._current_file)

    def _scroll_to_current(self) -> None:
        for tile in self._tiles:
            if tile.filepath == self._current_file:
                self.ensureWidgetVisible(tile)
                return

"""
Media frame (PySide6): images, videos, GIFs, PDFs.
Uses QGraphicsView for static images (pan/zoom), QMovie for animated rasters, VLC for video.
"""

import os
import platform
import threading
import time
import warnings

from PySide6.QtWidgets import (
    QFrame,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
    QApplication,
    QDialog,
)
from PySide6.QtCore import Qt, QRectF, QSize, QPoint, QRect, QEvent, QTimer, Signal, QObject, QThread, Slot
from PySide6.QtGui import QColor, QImage, QPixmap, QImageReader, QPainter, QCursor, QMovie

from lib.blur_overlay_qt import BlurOverlay
from ui.app_style import AppStyle
from ui.app_window.media_controls_overlay import MediaControlsOverlay, OVERLAY_HEIGHT
from utils.config import config
from utils.logging_setup import get_logger
from image.frame_cache import FrameCache
from utils.media_utils import (
    is_audio_for_display,
    is_video_for_display,
    is_video_path_by_extension,
    is_video_container_signature,
)
from utils.utils import Utils
from utils.translations import I18N

_ = I18N._
logger = get_logger("media_frame_qt")

# Optional: Pillow for formats Qt may not support (HEIC, AVIF, etc.)
try:
    from PIL import Image
    from utils.pillow_plugins import ensure_pillow_plugins_registered
    ensure_pillow_plugins_registered()
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import vlc
    _VLC_AVAILABLE = True
except ImportError:
    _VLC_AVAILABLE = False

# Holds VLC Instance objects whose release() must be deferred because a hung
# stop() thread is still running inside libvlc.  Keeps the C-level instance
# alive (refcount > 0) so the thread doesn't access freed memory.  Entries are
# never explicitly released; they leak until process exit, which is intentional.
_vlc_instances_pending_cleanup: list = []

try:
    import shiboken6
    _SHIBOKEN_AVAILABLE = True
except ImportError:
    _SHIBOKEN_AVAILABLE = False


_MATROSKA_EXTENSIONS = {".webm", ".mkv", ".mka", ".mks"}

# Minimum wall-clock seconds that must elapse before a VLC State.Ended is
# treated as a real end for slideshow advancement.  Prevents buffer-deadlock
# videos (which report Ended in milliseconds with zero playback time) from
# causing the slideshow to blow past them instantly.
_VLC_INSTANT_END_MIN_SECONDS = 1.0

# Matroska paths where video_stop() previously timed out, indicating a missing
# Cues element.  show_video() skips VLC and shows a placeholder for these paths
# on all subsequent loads.  Populated lazily on first timeout; never cleared.
_matroska_missing_cues_paths: set = set()


class VideoUI:
    """Placeholder for video state (path, active)."""

    def __init__(self, filepath, has_video=True, has_cues=True):
        self.filepath = filepath
        self.active = False
        self.has_video = bool(has_video)
        self.has_cues = bool(has_cues)


def _probe_has_video_stream(path: str) -> bool:
    """Return True if the file contains at least one video stream.

    Checks with pyav first, then OpenCV. Falls back to True (old behaviour)
    when neither library is available, so the caller is never worse off than
    before this guard was added.
    """
    try:
        import av as _av
        container = _av.open(path, metadata_errors="ignore")
        try:
            return bool(container.streams.video)
        finally:
            container.close()
    except Exception:
        pass
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(path)
        try:
            return cap.isOpened() and cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) > 0
        finally:
            cap.release()
    except Exception:
        pass
    return True




def scale_dims(dims, max_dims, maximize=False):
    """Return (width, height) to fit dims inside max_dims. If maximize, fill when smaller."""
    x, y = dims[0], dims[1]
    max_x, max_y = max_dims[0], max_dims[1]
    if x <= max_x and y <= max_y:
        if maximize:
            if x < max_x:
                return (int(x * max_y / y), max_y)
            elif y < max_y:
                return (max_x, int(y * max_x / x))
        return (x, y)
    elif x <= max_x:
        return (int(x * max_y / y), max_y)
    elif y <= max_y:
        return (max_x, int(y * max_x / x))
    else:
        x_scale = max_x / x
        y_scale = max_y / y
        if x_scale < y_scale:
            return (int(x * x_scale), int(y * x_scale))
        return (int(x * y_scale), int(y * y_scale))


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView wrapper that provides modular wheel-zoom and drag-pan."""

    def __init__(
        self,
        parent=None,
        *,
        zoom_step: float = 1.15,
        min_zoom: float = 0.05,
        max_zoom: float = 20.0,
        zoom_modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.ControlModifier,
    ):
        super().__init__(parent)
        self._zoom_step = float(zoom_step)
        self._min_zoom = float(min_zoom)
        self._max_zoom = float(max_zoom)
        self._zoom_modifier = zoom_modifier
        self._zoom_factor = 1.0
        self._has_user_zoom = False
        self._interaction_enabled = False
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)
        if self._interaction_enabled:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def reset_interaction(self, *, reset_transform: bool = True) -> None:
        if reset_transform:
            self.resetTransform()
        self._zoom_factor = 1.0
        self._has_user_zoom = False

    def fit_item(self, item: QGraphicsPixmapItem, aspect_mode=Qt.AspectRatioMode.KeepAspectRatio) -> None:
        if item is None:
            return
        pix = item.pixmap()
        if pix.isNull():
            return
        self.resetTransform()
        self.fitInView(item, aspect_mode)
        self._zoom_factor = 1.0
        self._has_user_zoom = False

    def is_user_zoom_active(self) -> bool:
        return self._has_user_zoom

    def wheelEvent(self, event):
        if not self._interaction_enabled:
            super().wheelEvent(event)
            return
        if not (event.modifiers() & self._zoom_modifier):
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return

        direction = self._zoom_step if delta > 0 else (1.0 / self._zoom_step)
        new_zoom = self._zoom_factor * direction
        new_zoom = max(self._min_zoom, min(new_zoom, self._max_zoom))
        if new_zoom == self._zoom_factor:
            event.accept()
            return
        factor = new_zoom / self._zoom_factor

        old_transform_anchor = self.transformationAnchor()
        old_resize_anchor = self.resizeAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(old_transform_anchor)
        self.setResizeAnchor(old_resize_anchor)

        self._zoom_factor = new_zoom
        self._has_user_zoom = abs(self._zoom_factor - 1.0) > 1e-6
        event.accept()


class _ImageDecodeWorkerSignals(QObject):
    decoded = Signal(int, str, object)  # request_id, path, QImage
    failed = Signal(int, str, str)  # request_id, path, message


class _ImageDecodeWorker(QThread):
    """Decode a full-resolution image in background for progressive promotion."""

    def __init__(self, request_id: int, path: str):
        super().__init__()
        self.request_id = int(request_id)
        self.path = path
        self.signals = _ImageDecodeWorkerSignals()

    def run(self):
        try:
            reader = QImageReader(self.path)
            img = reader.read()
            if img.isNull():
                self.signals.failed.emit(self.request_id, self.path, reader.errorString())
                return
            self.signals.decoded.emit(self.request_id, self.path, img)
        except Exception as e:
            self.signals.failed.emit(self.request_id, self.path, str(e))


class MediaFrame(QFrame):
    """
    Display media (static images with pan/zoom via QGraphicsView, animated rasters via QMovie,
    video via VLC). Provides winId() for VLC embedding (used by muse/playback.py).
    """
    seek_requested = Signal(int)
    play_pause_requested = Signal()
    volume_requested = Signal(int)
    mute_requested = Signal()

    def __init__(self, parent=None, fill_canvas=False):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"background-color: {AppStyle.MEDIA_BG};")
        self.setMouseTracking(True)

        self.fill_canvas = fill_canvas
        self.path = "."
        self.imscale = 1.0
        self.imwidth = 0
        self.imheight = 0
        self.media_displayed = False

        self._image = None  # QImage or PIL Image when loaded
        self._video_ui = None  # VideoUI when showing video
        self._video_started_monotonic: float = 0.0
        self._current_pixmap = None  # keep reference

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._graphics_view = ZoomableGraphicsView(self)
        self._graphics_view.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, True
        )
        self._graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._graphics_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._graphics_view.setStyleSheet(f"background-color: {AppStyle.MEDIA_BG};")
        self._graphics_view.set_interaction_enabled(False)
        self._scene = QGraphicsScene(self)
        self._graphics_view.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        layout.addWidget(self._graphics_view)

        self._placeholder_label = QLabel("", self)
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        layout.addWidget(self._placeholder_label)
        self._placeholder_label.hide()

        self._gif_label = QLabel(self)
        self._gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gif_label.hide()
        layout.addWidget(self._gif_label)
        self._gif_movie = None
        self._gif_is_animated = False
        self._gif_frame_delays = []
        self._gif_timeline = []
        self._gif_total_duration_ms = 0

        if _VLC_AVAILABLE:
            self.vlc_instance = vlc.Instance()
            self.vlc_media_player = self.vlc_instance.media_player_new()
            self.vlc_media_player.video_set_mouse_input(False)
            self.vlc_media_player.video_set_key_input(False)
            if config.media_volume_use_eq:
                self._vlc_eq = vlc.AudioEqualizer()
                self.vlc_media_player.set_equalizer(self._vlc_eq)
            else:
                self._vlc_eq = None
            self.vlc_media = None
        else:
            self.vlc_instance = None
            self.vlc_media_player = None
            self._vlc_eq = None
            self.vlc_media = None
        self._hung_stop_thread: threading.Thread | None = None

        self._controls_overlay = MediaControlsOverlay(self)
        self._controls_overlay.seek_requested.connect(self.seek_requested.emit)
        self._controls_overlay.play_pause_requested.connect(self.play_pause_requested.emit)
        self._controls_overlay.volume_changed.connect(self.volume_requested.emit)
        self._controls_overlay.mute_toggled.connect(self.mute_requested.emit)

        self._blur_overlay = BlurOverlay(self)
        self._blur_overlay.attach(self)
        self._pending_blur_path: str | None = None
        self._window_filter_installed = False
        self._mouse_inside = False
        self._last_cursor_pos = None
        self._last_known_volume = 100
        self._last_known_muted = False

        self._mouse_poll_timer = QTimer(self)
        self._mouse_poll_timer.setInterval(100)
        self._mouse_poll_timer.timeout.connect(self._poll_mouse_position)
        self._mouse_poll_timer.start()

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(250)
        self._playback_timer.timeout.connect(self._update_vlc_playback_progress)
        self._vlc_disposed = False
        self._image_request_id = 0
        self._decode_worker = None
        self._empty_directory_message = _(
            "No displayable files found in this directory for the current settings."
        )

    def request_blur(self, path: str) -> None:
        """Record that the blur overlay should be shown when path is displayed."""
        self._pending_blur_path = path

    def apply_pending_blur(self, path: str) -> None:
        """Apply or remove the blur overlay after show_media has rendered path."""
        if self._pending_blur_path == path:
            self._blur_overlay.sync_geometry(self)
            # grab() works for all Qt-rendered content (images and the black
            # placeholder shown for blurred video — see _show_black_frame).
            snapshot = self.grab()
            if self._blur_overlay.is_blur_active():
                self._blur_overlay.update_blur(snapshot=snapshot)
            else:
                self._blur_overlay.show_blur(snapshot=snapshot)
        else:
            self._blur_overlay.hide_blur()
        self._pending_blur_path = None

    def set_background_color(self, background_color):
        color = background_color or AppStyle.MEDIA_BG
        self.setStyleSheet(f"background-color: {color};")
        self._graphics_view.setStyleSheet(f"background-color: {color};")

    def _is_animated_image_candidate(self, path: str) -> bool:
        """Return True for formats that may carry animation frames."""
        if not path:
            return False
        path_lower = path.lower()
        return path_lower.endswith(
            (".gif", ".webp", ".apng", ".jpg", ".jpeg", ".jpe", ".jfif")
        )

    def _large_image_dim_threshold(self) -> int:
        return max(1, int(getattr(config, "large_image_dim_threshold_px", 5000)))

    def _large_preview_overscan(self) -> float:
        return max(1.0, float(getattr(config, "large_image_preview_overscan", 1.5)))

    def _large_preview_max_dim(self) -> int:
        return max(512, int(getattr(config, "large_image_preview_max_dim", 4096)))

    def _large_hq_downscale_enabled(self) -> bool:
        return bool(getattr(config, "large_image_enable_hq_idle_downscale", True))

    def _large_hq_ratio_threshold(self) -> float:
        return max(1.1, float(getattr(config, "large_image_hq_downscale_ratio_threshold", 1.8)))

    def _large_enable_full_res_promotion(self) -> bool:
        return bool(getattr(config, "large_image_enable_full_res_promotion", True))

    def _promotion_min_free_ram_gb(self) -> float:
        return max(0.0, float(getattr(config, "large_image_promotion_min_free_ram_gb", 1.0)))

    def _promotion_max_estimated_mb(self) -> int:
        return max(64, int(getattr(config, "large_image_promotion_max_estimated_mb", 512)))

    def _promotion_available_ram_fraction(self) -> float:
        value = float(getattr(config, "large_image_promotion_available_ram_fraction", 0.25))
        return min(max(value, 0.05), 0.9)

    def _next_request_id(self) -> int:
        self._image_request_id += 1
        return self._image_request_id

    def _cancel_decode_worker(self) -> None:
        worker = self._decode_worker
        self._decode_worker = None
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait(10)
        except Exception:
            pass

    def _invalidate_pending_image_promotion(self) -> None:
        self._next_request_id()
        self._cancel_decode_worker()

    def _probe_image_size(self, path: str) -> tuple[int, int]:
        reader = QImageReader(path)
        size = reader.size()
        if size.width() > 0 and size.height() > 0:
            return int(size.width()), int(size.height())
        if _PIL_AVAILABLE:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with Image.open(path) as pil_img:
                        return int(pil_img.width), int(pil_img.height)
            except Exception:
                pass
        return 0, 0

    def _is_large_image_dims(self, dims: tuple[int, int]) -> bool:
        w, h = dims
        if w <= 0 or h <= 0:
            return False
        threshold = self._large_image_dim_threshold()
        return w > threshold or h > threshold

    def _preview_target_size(self, source_dims: tuple[int, int]) -> QSize:
        src_w, src_h = source_dims
        if src_w <= 0 or src_h <= 0:
            return QSize()
        viewport_size = self._graphics_view.viewport().size()
        view_w = max(1, viewport_size.width())
        view_h = max(1, viewport_size.height())
        overscan = self._large_preview_overscan()
        max_dim = self._large_preview_max_dim()
        target_w = min(int(view_w * overscan), max_dim, src_w)
        target_h = min(int(view_h * overscan), max_dim, src_h)
        return QSize(max(1, target_w), max(1, target_h))

    def _should_use_hq_downscale(self, source_dims: tuple[int, int], target_size: QSize | None) -> bool:
        if not self._large_hq_downscale_enabled() or target_size is None:
            return False
        src_w, src_h = source_dims
        if src_w <= 0 or src_h <= 0:
            return False
        target_w = max(1, int(target_size.width()))
        target_h = max(1, int(target_size.height()))
        ratio = min(src_w / target_w, src_h / target_h)
        return ratio >= self._large_hq_ratio_threshold()

    def _qimage_from_pillow(self, pil_img: "Image.Image") -> QImage:
        rgb = pil_img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        # RGB888 rows are not guaranteed to be 32-bit aligned, so pass
        # explicit bytes-per-line to avoid sheared/corrupted rendering.
        bytes_per_line = int(rgb.width) * 3
        qimg = QImage(data, rgb.width, rgb.height, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg.copy()

    def _qt_image_allocation_limit_mb(self) -> int:
        getter = getattr(QImageReader, "allocationLimit", None)
        if callable(getter):
            try:
                limit_mb = int(getter())
                if limit_mb > 0:
                    return limit_mb
            except Exception:
                pass
        # Qt default allocation guard is typically 256MB.
        return 256

    def _load_image_to_qimage(self, path, target_size: QSize | None = None, prefer_hq_downscale: bool = False, source_dims: tuple[int, int] | None = None):
        """Load path to QImage; support optional preview scaling and HQ downscale fallback."""
        reader = QImageReader(path)
        if target_size is not None and target_size.width() > 0 and target_size.height() > 0:
            reader.setScaledSize(target_size)
        img = reader.read()
        if not img.isNull():
            if not (_PIL_AVAILABLE and prefer_hq_downscale and source_dims and self._should_use_hq_downscale(source_dims, target_size)):
                return img
        if _PIL_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    with Image.open(path) as pil_img:
                        if target_size is not None and target_size.width() > 0 and target_size.height() > 0:
                            if prefer_hq_downscale and source_dims and self._should_use_hq_downscale(source_dims, target_size):
                                resampling = getattr(Image, "Resampling", Image).LANCZOS
                                pil_img = pil_img.convert("RGB").resize((int(target_size.width()), int(target_size.height())), resampling)
                                logger.debug(
                                    "Large-image HQ preview decode via PIL Lanczos: path=%s target=%sx%s",
                                    path,
                                    target_size.width(),
                                    target_size.height(),
                                )
                                return self._qimage_from_pillow(pil_img)
                            pil_img.thumbnail((int(target_size.width()), int(target_size.height())))
                        return self._qimage_from_pillow(pil_img)
                except Exception as e:
                    if "truncated" in str(e):
                        time.sleep(0.25)
                        with Image.open(path) as pil_img:
                            if target_size is not None and target_size.width() > 0 and target_size.height() > 0:
                                resampling = getattr(Image, "Resampling", Image).LANCZOS
                                pil_img = pil_img.convert("RGB").resize((int(target_size.width()), int(target_size.height())), resampling)
                                logger.debug(
                                    "Large-image truncated retry via PIL Lanczos: path=%s target=%sx%s",
                                    path,
                                    target_size.width(),
                                    target_size.height(),
                                )
                            return self._qimage_from_pillow(pil_img)
                    if not img.isNull():
                        return img
                    raise
        return img if not img.isNull() else QImage()

    def _display_qimage(self, qimg: QImage, source_dims: tuple[int, int], preserve_view: bool = False):
        if qimg.isNull():
            return
        previous_center = None
        previous_transform = None
        if preserve_view and self._current_pixmap is not None and not self._current_pixmap.isNull():
            viewport_center = self._graphics_view.viewport().rect().center()
            previous_center = self._graphics_view.mapToScene(viewport_center)
            previous_transform = self._graphics_view.transform()
        pix = QPixmap.fromImage(qimg)
        self._current_pixmap = pix
        self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        if source_dims[0] > 0 and source_dims[1] > 0:
            self.imwidth, self.imheight = int(source_dims[0]), int(source_dims[1])
        else:
            self.imwidth, self.imheight = qimg.width(), qimg.height()
        self._image = qimg
        self._graphics_view.set_interaction_enabled(True)
        if preserve_view and previous_center is not None and previous_transform is not None:
            self._graphics_view.setTransform(previous_transform)
            self._graphics_view.centerOn(previous_center)
        else:
            self._apply_image_scale_mode()
        self._graphics_view.show()
        self._gif_label.hide()
        self._placeholder_label.hide()
        self._controls_overlay.set_audio_controls_visible(True)
        self.media_displayed = True

    def _can_promote_large_image(self, source_dims: tuple[int, int]) -> bool:
        if not self._large_enable_full_res_promotion():
            return False
        src_w, src_h = source_dims
        if src_w <= 0 or src_h <= 0:
            return False
        available_ram_gb = Utils.calculate_available_ram()
        if available_ram_gb < self._promotion_min_free_ram_gb():
            return False
        estimated_bytes = int(src_w) * int(src_h) * 4
        max_estimated_bytes = self._promotion_max_estimated_mb() * 1024 * 1024
        available_budget_bytes = int(available_ram_gb * (1024 ** 3) * self._promotion_available_ram_fraction())
        qt_limit_bytes = int(self._qt_image_allocation_limit_mb() * 1024 * 1024 * 0.9)
        allowed = min(max_estimated_bytes, available_budget_bytes, qt_limit_bytes)
        return estimated_bytes <= allowed

    def _start_large_image_promotion(self, path: str, request_id: int, source_dims: tuple[int, int]) -> None:
        self._cancel_decode_worker()
        if not self._can_promote_large_image(source_dims):
            return
        worker = _ImageDecodeWorker(request_id=request_id, path=path)
        worker.signals.decoded.connect(self._on_promotion_decoded)
        worker.signals.failed.connect(self._on_promotion_failed)
        self._decode_worker = worker
        worker.start()

    @Slot(int, str, object)
    def _on_promotion_decoded(self, request_id: int, path: str, image_obj: object) -> None:
        if request_id != self._image_request_id:
            return
        if path != self.path:
            return
        if isinstance(self._video_ui, VideoUI) or self._gif_movie is not None:
            return
        if not isinstance(image_obj, QImage) or image_obj.isNull():
            return
        self._display_qimage(image_obj, (image_obj.width(), image_obj.height()), preserve_view=True)
        self._decode_worker = None

    @Slot(int, str, str)
    def _on_promotion_failed(self, request_id: int, path: str, _message: str) -> None:
        if request_id == self._image_request_id and path == self.path:
            logger.warning("Large-image promotion decode failed: path=%s request_id=%s message=%s", path, request_id, _message)
            self._decode_worker = None

    def _has_active_overlay_media(self):
        return isinstance(self._video_ui, VideoUI) or (self._gif_movie is not None and self._gif_is_animated)

    def _apply_image_scale_mode(self):
        """Scale only when needed, unless fill-canvas is enabled."""
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        viewport_size = self._graphics_view.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return
        pix_w = self._current_pixmap.width()
        pix_h = self._current_pixmap.height()
        if pix_w <= 0 or pix_h <= 0:
            return
        should_fit = self.fill_canvas or pix_w > viewport_size.width() or pix_h > viewport_size.height()
        if should_fit:
            self._graphics_view.fit_item(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self._graphics_view.reset_interaction()

    def _update_gif_scale_mode(self):
        if self._gif_movie is None:
            return
        frame_size = self._gif_movie.frameRect().size()
        if frame_size.width() <= 0 or frame_size.height() <= 0:
            return
        # On the first-ever GIF load the label can still report a tiny default
        # size before layout settles, which would lock QMovie scaling too small.
        label_w = self._gif_label.width()
        label_h = self._gif_label.height()
        if label_w <= 1 or label_h <= 1:
            content_rect = self.contentsRect()
            label_w = max(label_w, content_rect.width(), self.width())
            label_h = max(label_h, content_rect.height(), self.height())
        max_dims = (max(1, label_w), max(1, label_h))
        target_w, target_h = scale_dims(
            (frame_size.width(), frame_size.height()),
            max_dims,
            maximize=self.fill_canvas,
        )
        if target_w == frame_size.width() and target_h == frame_size.height():
            self._gif_movie.setScaledSize(QSize())
        else:
            self._gif_movie.setScaledSize(QSize(target_w, target_h))

    def _analyze_animated_image(self, path):
        frame_count = 0
        delays = []
        reader = QImageReader(path)
        if not reader.supportsAnimation():
            return frame_count, delays
        while True:
            frame = reader.read()
            if frame.isNull():
                break
            frame_count += 1
            delay = int(reader.nextImageDelay() or 100)
            delays.append(max(delay, 20))
            if not reader.canRead():
                break
        return frame_count, delays

    def _reset_gif_state(self):
        self._gif_is_animated = False
        self._gif_frame_delays = []
        self._gif_timeline = []
        self._gif_total_duration_ms = 0

    def _build_gif_timeline(self, frame_delays):
        timeline = [0]
        elapsed = 0
        for delay in frame_delays:
            elapsed += max(int(delay), 20)
            timeline.append(elapsed)
        return timeline, elapsed

    def _teardown_gif_movie(self):
        """Fully detach and release QMovie so GIF files are not locked on Windows."""
        movie = self._gif_movie or self._gif_label.movie()
        if movie is not None:
            try:
                movie.stop()
            except Exception:
                pass
            try:
                self._gif_label.setMovie(None)
            except Exception:
                pass
            if _SHIBOKEN_AVAILABLE:
                try:
                    shiboken6.delete(movie)
                except Exception:
                    try:
                        movie.deleteLater()
                    except Exception:
                        pass
            else:
                try:
                    movie.deleteLater()
                except Exception:
                    pass
        self._gif_movie = None
        self._reset_gif_state()
        self._gif_label.clear()
        self._gif_label.hide()
        app = QApplication.instance()
        if app is not None:
            # Flush deferred-delete events so file handles are dropped immediately.
            app.processEvents()

    def _update_gif_overlay_progress(self):
        if self._gif_movie is None or not self._gif_is_animated or self._gif_total_duration_ms <= 0:
            return
        frame_idx = max(int(self._gif_movie.currentFrameNumber()), 0)
        frame_idx = min(frame_idx, max(len(self._gif_timeline) - 2, 0))
        current_ms = self._gif_timeline[frame_idx] if self._gif_timeline else 0
        self.update_playback_progress(current_ms, self._gif_total_duration_ms)
        self.set_playback_paused(self._gif_movie.state() != QMovie.MovieState.Running)

    def _show_image_in_view(self, path):
        """Load image from path, display at full resolution in QGraphicsView.

        The view's ``SmoothPixmapTransform`` render hint handles the
        down-scaling to screen size in a single pass, avoiding the
        quality loss of a double-scale (pre-scale then fitInView).
        """
        if not path or path == "." or not os.path.exists(path):
            return
        request_id = self._next_request_id()
        self._cancel_decode_worker()
        source_dims = self._probe_image_size(path)
        if self._is_large_image_dims(source_dims):
            preview_target = self._preview_target_size(source_dims)
            qimg = self._load_image_to_qimage(
                path,
                target_size=preview_target,
                prefer_hq_downscale=True,
                source_dims=source_dims,
            )
            if qimg.isNull():
                qimg = self._load_image_to_qimage(path)
            if qimg.isNull():
                logger.warning("Large-image load failed (both preview and fallback decode): path=%s", path)
                return
            self._display_qimage(qimg, source_dims)
            self._start_large_image_promotion(path, request_id, source_dims)
            return

        qimg = self._load_image_to_qimage(path)
        if qimg.isNull():
            return
        self._display_qimage(qimg, source_dims)

    def _show_animated_media(self, path) -> bool:
        if not path or path == "." or not os.path.exists(path):
            return False
        self.clear()
        frame_count, frame_delays = self._analyze_animated_image(path)
        is_animated = frame_count > 1
        if not is_animated:
            return False
        self._gif_movie = QMovie(path)
        if not self._gif_movie.isValid():
            self._gif_movie = None
            self._show_placeholder(
                _("Animated media format is not supported for playback: ")
                + os.path.basename(path)
            )
            return True
        # Random-access seeking is unreliable without frame caching.
        self._gif_movie.setCacheMode(QMovie.CacheMode.CacheAll)
        self._gif_is_animated = True
        self._gif_frame_delays = frame_delays or [100] * max(frame_count, 1)
        self._gif_timeline, self._gif_total_duration_ms = self._build_gif_timeline(self._gif_frame_delays)
        self._gif_label.setMovie(self._gif_movie)
        self._gif_movie.jumpToFrame(0)
        self._gif_movie.frameChanged.connect(lambda _idx: self._update_gif_overlay_progress())
        self._graphics_view.hide()
        self._placeholder_label.hide()
        self._gif_label.show()
        self._update_gif_scale_mode()
        self._gif_movie.start()
        # Re-apply scaling once the event loop processes layout updates.
        QTimer.singleShot(0, self._update_gif_scale_mode)
        self._controls_overlay.set_audio_controls_visible(False)
        self._image = None
        self._video_ui = None
        self._current_pixmap = None
        self.media_displayed = True
        self.on_track_changed()
        self._update_gif_overlay_progress()
        self._playback_timer.start()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlay()
        if not self.media_displayed or isinstance(self._video_ui, VideoUI):
            return
        if self._gif_movie is not None:
            self._update_gif_scale_mode()
        else:
            if not self._graphics_view.is_user_zoom_active():
                self._apply_image_scale_mode()

    def show_media(self, path):
        """Show media at path. Dispatches to show_video for video, QMovie for animated rasters."""
        self._invalidate_pending_image_promotion()
        if isinstance(self._video_ui, VideoUI):
            self.video_stop()
        if self._gif_movie is not None:
            self._teardown_gif_movie()
        self.path = path or "."
        if not path or path == "." or path.strip() == "" or not os.path.exists(path):
            self.clear()
            return
        if is_audio_for_display(path):
            from ui.app_window.audio_playback import show_in_frame
            show_in_frame(self, path)
            return
        # Video dispatch: use VLC if available, otherwise show placeholder
        if is_video_for_display(path):
            if self._pending_blur_path == path:
                self._show_black_frame()
                return
            if _VLC_AVAILABLE and self.vlc_media_player:
                # Broken single-frame stream (buffer-deadlock): show the one
                # extractable frame as a static image and let VLC handle audio
                # + end-of-file in the background without painting over it.
                # Not applied to visually-inferred pseudostatic videos, which
                # may change outside the sampled windows.
                if FrameCache.is_single_frame_stream(path):
                    frame_path = FrameCache.get_any_cached_sampled_frame(path)
                    if frame_path:
                        self._show_image_in_view(frame_path)
                        self.show_video(path, freeze_frame=True)
                        return
                self.show_video(path)
            else:
                self._show_placeholder(_("Video: ") + os.path.basename(path))
            return
        if self._is_animated_image_candidate(path) and self._show_animated_media(path):
            return
        self._video_ui = None
        self.imscale = 1.0
        try:
            self._show_image_in_view(self.path)
        except Exception as e:
            logger.warning("Failed to render media path=%s error=%s", self.path, e)
            self._show_placeholder(_("Unable to display this file: ") + os.path.basename(path))

    def set_fill_canvas(self, fill_canvas: bool):
        self.fill_canvas = bool(fill_canvas)
        if not self.media_displayed or isinstance(self._video_ui, VideoUI):
            return
        if self._gif_movie is not None:
            self._update_gif_scale_mode()
        else:
            self._apply_image_scale_mode()

    def show_video(self, path, freeze_frame: bool = False):
        """Play video in this frame (VLC embeds via winId()).

        When freeze_frame is True the caller has already displayed a static
        image; VLC plays audio only (video output suppressed via :no-video)
        and drives end-of-file detection without painting over the graphics
        view.  clear() and graphics-view teardown are skipped.
        """
        if not _VLC_AVAILABLE or not self.vlc_media_player:
            return
        if (
            not is_audio_for_display(path)
            and not is_video_path_by_extension(path)
            and not is_video_container_signature(path)
        ):
            return
        if not freeze_frame:
            self.clear()
            self._graphics_view.set_interaction_enabled(False)
            self._graphics_view.reset_interaction()
        has_video = _probe_has_video_stream(path)
        has_cues = path not in _matroska_missing_cues_paths
        self._video_ui = VideoUI(path, has_video=has_video, has_cues=has_cues)
        self.path = path
        # Normal video: attach window handle so libvlc renders into this widget.
        # Audio-only or freeze_frame: skip window handle — no video surface needed.
        if has_video and not freeze_frame:
            self.ensure_video_frame()
        self.vlc_media = self.vlc_instance.media_new(path)
        if freeze_frame:
            # Suppress VLC video decode/output so it doesn't paint over the
            # static frame.  Audio track and end-of-file signalling still work.
            self.vlc_media.add_option(":no-video")
        self.vlc_media_player.set_media(self.vlc_media)
        self._video_started_monotonic = time.monotonic()
        if self.vlc_media_player.play() == -1:
            raise Exception("Failed to play video")
        if not freeze_frame:
            if has_video:
                self._graphics_view.hide()
                self._placeholder_label.hide()
            else:
                # Audio-only container: no window handle → no video output needed.
                self._graphics_view.hide()
                self._placeholder_label.setText(_("Audio: ") + os.path.basename(path))
                self._placeholder_label.show()
        self._controls_overlay.set_audio_controls_visible(True)
        # VLC renders into the native window handle and may set a busy cursor
        # at the OS level.  Force an arrow cursor so the user doesn't see a
        # spinner while a video is simply playing.
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.on_track_changed()
        self._controls_overlay.set_no_seek_index(not has_cues)
        self._sync_overlay_volume_state(force=True)
        self._playback_timer.start()

    def ensure_video_frame(self):
        """Set the window id for VLC video output."""
        if not _VLC_AVAILABLE or not self.vlc_media_player:
            return
        wid = int(self.winId()) if self.winId() else 0
        if not wid:
            return
        if platform.system() == "Windows":
            self.vlc_media_player.set_hwnd(wid)
        elif platform.system() == "Darwin":
            self.vlc_media_player.set_nsobject(wid)
        else:
            self.vlc_media_player.set_xwindow(wid)

    def video_display(self):
        """Start video playback (after ensure_video_frame)."""
        if not _VLC_AVAILABLE or not self.vlc_media_player or not self.path:
            return
        if isinstance(self._video_ui, VideoUI) and self._video_ui.has_video:
            self.ensure_video_frame()
        self.vlc_media = self.vlc_instance.media_new(self.path)
        self.vlc_media_player.set_media(self.vlc_media)
        if self.vlc_media_player.play() == -1:
            raise Exception("Failed to play video")
        self.on_track_changed()
        if isinstance(self._video_ui, VideoUI):
            self._controls_overlay.set_no_seek_index(not self._video_ui.has_cues)
        self._sync_overlay_volume_state(force=True)
        self._playback_timer.start()

    def closeEvent(self, event):
        self.dispose_vlc()
        super().closeEvent(event)

    def video_stop(self):
        if _VLC_AVAILABLE and self.vlc_media_player:
            player = self.vlc_media_player
            # libvlc_media_player_stop() is synchronous in libvlc 3.x: it blocks
            # until the internal decode loop exits.  For WEBM files without a Cues
            # (seek-index) element, VLC performs a sequential scan and that loop can
            # stall indefinitely — deadlocking the UI thread.  Run stop() in a daemon
            # thread and give it a generous timeout; if it hangs, abandon the player
            # and create a fresh one so the application stays responsive.
            _t = threading.Thread(target=player.stop, daemon=True)
            _t.start()
            _t.join(timeout=3.0)
            if _t.is_alive():
                logger.warning(
                    "VLC stop() timed out for %s — Matroska container likely missing "
                    "Cues element; abandoning hung player and creating a fresh instance",
                    self.path,
                )
                if self.path and os.path.splitext(self.path)[1].lower() in _MATROSKA_EXTENSIONS:
                    _matroska_missing_cues_paths.add(self.path)
                self._hung_stop_thread = _t
                self._replace_vlc_player()
            else:
                try:
                    player.set_media(None)
                except Exception:
                    pass
        if _VLC_AVAILABLE and self.vlc_media:
            try:
                self.vlc_media.release()
            except Exception:
                pass
            self.vlc_media = None
        self._video_ui = None
        self.unsetCursor()
        self.on_playback_stopped()
        self._playback_timer.stop()

    def _replace_vlc_player(self):
        """Spin up a fresh media player after the previous one failed to stop cleanly."""
        if not _VLC_AVAILABLE or self.vlc_instance is None:
            self.vlc_media_player = None
            return
        try:
            new_player = self.vlc_instance.media_player_new()
            new_player.video_set_mouse_input(False)
            new_player.video_set_key_input(False)
            if self._vlc_eq is not None:
                new_player.set_equalizer(self._vlc_eq)
            self.vlc_media_player = new_player
        except Exception:
            self.vlc_media_player = None

    def dispose_vlc(self):
        """Fully tear down VLC resources for this widget instance."""
        if self._vlc_disposed:
            return
        self._vlc_disposed = True
        self._playback_timer.stop()
        try:
            self.video_stop()
        except Exception:
            pass
        if _VLC_AVAILABLE and self.vlc_media_player:
            try:
                # Detach video output from this window handle before release.
                if platform.system() == "Windows":
                    self.vlc_media_player.set_hwnd(0)
            except Exception:
                pass
            try:
                self.vlc_media_player.release()
            except Exception:
                pass
            self.vlc_media_player = None
        if _VLC_AVAILABLE and self.vlc_instance:
            t = self._hung_stop_thread
            if t is not None and t.is_alive():
                t.join(timeout=5.0)
            self._hung_stop_thread = None
            if t is not None and t.is_alive():
                # libvlc_media_player_stop() is still running inside libvlc.
                # Calling libvlc_release() now would free memory the thread is
                # actively using, causing a segfault.  Keep the Python wrapper
                # alive in a module-level list so GC doesn't collect it (and
                # call libvlc_release() via __del__) before the thread exits.
                _vlc_instances_pending_cleanup.append(self.vlc_instance)
            else:
                try:
                    self.vlc_instance.release()
                except Exception:
                    pass
            self.vlc_instance = None
        if _VLC_AVAILABLE and self._vlc_eq is not None:
            try:
                self._vlc_eq.release()
            except Exception:
                pass
            self._vlc_eq = None

    def video_pause(self):
        if self._gif_movie is not None and self._gif_is_animated:
            self._gif_movie.setPaused(True)
            self.set_playback_paused(True)
            return
        if _VLC_AVAILABLE and self.vlc_media_player:
            self.vlc_media_player.set_pause(1)
            self.set_playback_paused(True)

    def video_play(self):
        if self._gif_movie is not None and self._gif_is_animated:
            if self._gif_movie.state() == QMovie.MovieState.NotRunning:
                self._gif_movie.start()
            else:
                self._gif_movie.setPaused(False)
            self.set_playback_paused(False)
            self._playback_timer.start()
            return
        if not _VLC_AVAILABLE or not self.vlc_media_player:
            return
        state = self.vlc_media_player.get_state()
        if state in (vlc.State.Stopped, vlc.State.Ended):
            self.video_display()
        else:
            self.vlc_media_player.set_pause(0)
        self.set_playback_paused(False)
        self._playback_timer.start()

    def video_toggle_pause(self):
        if self._gif_movie is not None and self._gif_is_animated:
            if self._gif_movie.state() == QMovie.MovieState.Running:
                self.video_pause()
            else:
                self.video_play()
            return
        if not _VLC_AVAILABLE or not self.vlc_media_player:
            return
        if self.vlc_media_player.is_playing():
            self.video_pause()
        else:
            self.video_play()

    def pause_video_if_playing(self) -> bool:
        """Pause playback if media is currently playing. Returns True if paused."""
        if self._gif_movie is not None and self._gif_is_animated:
            if self._gif_movie.state() == QMovie.MovieState.Running:
                self.video_pause()
                return True
        if _VLC_AVAILABLE and self.vlc_media_player and self.vlc_media_player.is_playing():
            self.video_pause()
            return True
        return False

    def get_video_playback_ms(self) -> tuple[int, int]:
        """(position_ms, duration_ms) for embedded VLC video; (0, 0) otherwise."""
        return self.slideshow_vlc_time_ms()

    def has_time_based_media(self) -> bool:
        return isinstance(self._video_ui, VideoUI) or (self._gif_movie is not None and self._gif_is_animated)

    def is_slideshow_animated_raster_active(self) -> bool:
        """True when an animated GIF/WebP is playing (slideshow poll uses this)."""
        return self._gif_movie is not None and self._gif_is_animated

    def slideshow_vlc_time_ms(self) -> tuple[int, int]:
        """VLC playback (position_ms, duration_ms); (0, 0) when not on an embedded VLC track."""
        if not _VLC_AVAILABLE or not isinstance(self._video_ui, VideoUI) or not self.vlc_media_player:
            return (0, 0)
        duration_ms = max(int(self.vlc_media_player.get_length() or 0), 0)
        current_ms = max(int(self.vlc_media_player.get_time() or 0), 0)
        return (current_ms, duration_ms)

    def slideshow_vlc_has_ended(self) -> bool:
        if not _VLC_AVAILABLE or not isinstance(self._video_ui, VideoUI) or not self.vlc_media_player:
            return False
        try:
            if self.vlc_media_player.get_state() != vlc.State.Ended:
                return False
            # Guard against buffer-deadlock videos that enter Ended state in
            # milliseconds with zero playback time — without this they would
            # blow through the slideshow instantly.  Legitimate short videos
            # advance via the cur_ms >= dur_ms - tolerance path before this
            # fires, so the guard only holds truly broken/instant-end videos.
            elapsed = time.monotonic() - self._video_started_monotonic
            return elapsed >= _VLC_INSTANT_END_MIN_SECONDS
        except Exception:
            return False

    def slideshow_vlc_natural_dwell_supported(self) -> bool:
        """True when VLC can drive intrinsic-length dwell for the current video."""
        return bool(
            _VLC_AVAILABLE
            and isinstance(self._video_ui, VideoUI)
            and self.vlc_media_player
        )

    def slideshow_animated_total_duration_ms(self) -> int:
        """Total duration of the active animated GIF/WebP timeline, or 0."""
        if not self.is_slideshow_animated_raster_active():
            return 0
        return max(int(self._gif_total_duration_ms), 0)

    def save_media_screenshot(self, output_path: str) -> tuple[bool, str]:
        """Save screenshot for active time-based media (video or animated GIF)."""
        if not output_path:
            return False, _("No output path provided.")

        if isinstance(self._video_ui, VideoUI):
            if not _VLC_AVAILABLE or not self.vlc_media_player:
                return False, _("VLC player is unavailable.")
            try:
                ret = self.vlc_media_player.video_take_snapshot(0, output_path, 0, 0)
                if ret != 0:
                    return False, _("VLC snapshot operation failed.")
                for _ in range(20):
                    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                        return True, ""
                    time.sleep(0.05)
                return False, _("Snapshot file was not created.")
            except Exception as e:
                return False, str(e)

        if self._gif_movie is not None and self._gif_is_animated:
            img = self._gif_movie.currentImage()
            if img.isNull():
                pix = self._gif_label.pixmap()
                if pix is not None and not pix.isNull():
                    img = pix.toImage()
            if img.isNull():
                return False, _("No GIF frame available for screenshot.")
            try:
                ok = img.save(output_path)
            except Exception as e:
                return False, str(e)
            if not ok:
                return False, _("Failed to save GIF frame image.")
            return True, ""

        return False, _("No time-based media is currently active.")

    def video_take_screenshot(self):
        if _VLC_AVAILABLE and self.vlc_media_player:
            try:
                self.vlc_media_player.video_take_snapshot(0, "", 0, 0)
            except Exception:
                pass

    def video_seek(self, pos):
        if _VLC_AVAILABLE and self.vlc_media_player:
            self.vlc_media_player.set_position(pos)

    def video_seek_ms(self, position_ms: int):
        if self._gif_movie is not None and self._gif_is_animated and self._gif_total_duration_ms > 0:
            bounded = max(0, min(int(position_ms), int(self._gif_total_duration_ms)))
            frame_idx = 0
            for idx in range(len(self._gif_timeline) - 1):
                if self._gif_timeline[idx] <= bounded < self._gif_timeline[idx + 1]:
                    frame_idx = idx
                    break
            else:
                frame_idx = max(len(self._gif_timeline) - 2, 0)
            was_running = self._gif_movie.state() == QMovie.MovieState.Running
            if was_running:
                self._gif_movie.setPaused(True)
            jumped = self._gif_movie.jumpToFrame(frame_idx)
            if not jumped:
                # Fallback for back-seeks on some backends.
                self._gif_movie.stop()
                self._gif_movie.start()
                self._gif_movie.setPaused(True)
                self._gif_movie.jumpToFrame(frame_idx)
            self._update_gif_overlay_progress()
            if was_running:
                self._gif_movie.setPaused(False)
            return
        if not _VLC_AVAILABLE or not self.vlc_media_player:
            return
        duration_ms = self.vlc_media_player.get_length()
        if duration_ms <= 0:
            return
        bounded = max(0, min(int(position_ms), int(duration_ms)))
        self.vlc_media_player.set_time(bounded)

    def _apply_volume_eq(self) -> None:
        """Apply per-instance volume/mute via equalizer preamp + audio_set_mute.

        audio_set_volume is process-global on WASAPI/DirectSound; the equalizer
        runs inside each player's own DSP chain, making it the only truly
        per-player volume control available in-process.

        Volume (1-100) maps to preamp 0 dB → -20 dB using a linear-in-dB taper.
        Mute is handled solely by audio_set_mute; the equalizer always reflects
        the current volume level.

        Known limitations:
        - The equalizer preamp floor is -20 dB (≈10% amplitude), so the slider
          cannot reduce volume to true silence on its own; vol=0 is still faintly
          audible. Use the mute control for guaranteed silence.
        - Applying equalizer changes via set_equalizer() has a slight processing
          delay inside VLC's DSP chain, causing volume transitions to feel
          marginally slower than a direct audio_set_volume call would.
        """
        if not _VLC_AVAILABLE or self._vlc_eq is None or self.vlc_media_player is None:
            return
        vol = self._last_known_volume
        # Linear-in-dB taper: each 1% step = 0.2 dB, giving uniform perceived loudness change.
        preamp = (vol / 100.0) * 20.0 - 20.0
        self._vlc_eq.set_preamp(preamp)
        self.vlc_media_player.set_equalizer(self._vlc_eq)
        self.vlc_media_player.audio_set_mute(self._last_known_muted)

    def set_volume(self, volume: int):
        bounded = max(0, min(int(volume), 100))
        if config.media_volume_use_eq:
            if bounded > 0 and self._last_known_muted:
                self._last_known_muted = False
            self._last_known_volume = bounded
            self._apply_volume_eq()
        else:
            if _VLC_AVAILABLE and self.vlc_media_player:
                self.vlc_media_player.audio_set_volume(bounded)
                if bounded > 0 and self.vlc_media_player.audio_get_mute():
                    self.vlc_media_player.audio_set_mute(False)
            self._last_known_volume = bounded
            self._last_known_muted = self.is_muted()
        self._sync_overlay_volume_state(force=True)

    def get_volume(self) -> int:
        if not config.media_volume_use_eq and _VLC_AVAILABLE and self.vlc_media_player:
            volume = int(self.vlc_media_player.audio_get_volume() or 0)
            if volume >= 0:
                return volume
        return self._last_known_volume

    def set_mute(self, muted: bool):
        self._last_known_muted = bool(muted)
        if config.media_volume_use_eq:
            self._apply_volume_eq()
        elif _VLC_AVAILABLE and self.vlc_media_player:
            self.vlc_media_player.audio_set_mute(bool(muted))
        self._sync_overlay_volume_state(force=True)

    def toggle_mute(self):
        if config.media_volume_use_eq:
            self._last_known_muted = not self._last_known_muted
            self._apply_volume_eq()
        elif _VLC_AVAILABLE and self.vlc_media_player:
            self.vlc_media_player.audio_toggle_mute()
        else:
            self._last_known_muted = not self._last_known_muted
        self._sync_overlay_volume_state(force=True)

    def is_muted(self) -> bool:
        if not config.media_volume_use_eq and _VLC_AVAILABLE and self.vlc_media_player:
            return bool(self.vlc_media_player.audio_get_mute())
        return self._last_known_muted

    def clear(self):
        self._pending_blur_path = None
        self._blur_overlay.hide_blur()
        self._invalidate_pending_image_promotion()
        if isinstance(self._video_ui, VideoUI):
            self.video_stop()
        self._teardown_gif_movie()
        self._video_ui = None
        self._graphics_view.set_interaction_enabled(False)
        self._graphics_view.reset_interaction()
        self._scene.clear()
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._current_pixmap = None
        self._image = None
        self.media_displayed = False
        self._graphics_view.show()
        self._placeholder_label.setText("")
        self._placeholder_label.hide()
        self._controls_overlay.set_audio_controls_visible(True)
        self._controls_overlay.set_no_seek_index(False)
        self._controls_overlay.dismiss()

    def _show_black_frame(self) -> None:
        """Show a solid black pixmap without starting VLC.

        Used as a temporary stand-in when a blur prevalidation targets a video
        file, since the blur overlay cannot render above VLC's native surface.
        TODO: replace with a proper solution once VLC z-order is resolved.
        """
        self._video_ui = None
        black = QPixmap(self.size())
        black.fill(QColor(0, 0, 0))
        self._pixmap_item.setPixmap(black)
        self.media_displayed = True

    def _show_placeholder(self, text: str) -> None:
        """Clear the view and show a text placeholder (e.g. for unsupported video)."""
        self.clear()
        self._placeholder_label.setText(text)
        self._placeholder_label.show()

    def show_empty_directory_message(self) -> None:
        """Show the empty-directory message in the main media frame."""
        self.clear()
        self._placeholder_label.setText(self._empty_directory_message)
        self._placeholder_label.show()

    def hide_empty_directory_message(self) -> None:
        """Hide the empty-directory message when displayable files exist."""
        if self._placeholder_label.text() == self._empty_directory_message:
            self._placeholder_label.setText("")
            self._placeholder_label.hide()

    def release_media(self):
        self._invalidate_pending_image_promotion()
        if isinstance(self._video_ui, VideoUI):
            self.video_stop()
        self._teardown_gif_movie()
        if self._image is not None:
            self._image = None
        self._graphics_view.set_interaction_enabled(False)
        self._graphics_view.reset_interaction()
        self._current_pixmap = None
        self._pixmap_item.setPixmap(QPixmap())
        self.media_displayed = False
        self._controls_overlay.set_audio_controls_visible(True)
        self._controls_overlay.dismiss()

    def focus(self, refresh_media=False):
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        if refresh_media and self.path:
            self.show_media(self.path)

    def redraw_figures(self):
        """Dummy for compatibility with children that override."""
        pass

    def _position_overlay(self):
        """Place the controls overlay at the bottom of the frame in global coords."""
        h = OVERLAY_HEIGHT
        bottom_left = self.mapToGlobal(QPoint(0, self.height() - h))
        self._controls_overlay.setGeometry(bottom_left.x(), bottom_left.y(), self.width(), h)

    def _ensure_window_filter(self):
        if self._window_filter_installed:
            return
        top = self.window()
        if top and top is not self:
            top.installEventFilter(self)
            self._window_filter_installed = True

    def eventFilter(self, watched, event):
        etype = event.type()
        if etype in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.WindowStateChange):
            self._position_overlay()
        return super().eventFilter(watched, event)

    def moveEvent(self, event):
        super().moveEvent(event)
        self._position_overlay()

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_window_filter()
        self._position_overlay()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._mouse_inside = False
        self._controls_overlay.dismiss()

    def _poll_mouse_position(self):
        if not self.isVisible() or not self._has_active_overlay_media():
            return
        app = QApplication.instance()
        top_window = self.window()
        active_window = app.activeWindow() if app is not None else None

        if self._has_visible_child_dialog(top_window):
            self._mouse_inside = False
            self._controls_overlay.dismiss()
            self._last_cursor_pos = QCursor.pos()
            return

        # Hide controls whenever a different top-level window is active
        # (e.g. a dialog opened from this window), so the overlay never
        # obscures dialog content.
        if active_window is None or active_window is not top_window:
            self._mouse_inside = False
            self._controls_overlay.dismiss()
            self._last_cursor_pos = QCursor.pos()
            return

        cursor = QCursor.pos()
        frame_rect = QRect(self.mapToGlobal(QPoint(0, 0)), self.size())
        overlay_geo = self._controls_overlay.geometry()
        inside = frame_rect.contains(cursor) or overlay_geo.contains(cursor)
        moved = self._last_cursor_pos is None or cursor != self._last_cursor_pos

        if inside:
            self._mouse_inside = True
            # Movement within the media area is the activity signal.
            # If the cursor stays still, the overlay auto-hides.
            if moved:
                self._position_overlay()
                self._controls_overlay.show_overlay()
        elif not inside and self._mouse_inside:
            self._mouse_inside = False
            self._controls_overlay.hide_overlay()
        self._last_cursor_pos = cursor

    def _has_visible_child_dialog(self, top_window) -> bool:
        """Return True when any visible QDialog is owned by this window."""
        app = QApplication.instance()
        if app is None or top_window is None:
            return False
        for widget in app.topLevelWidgets():
            if widget is top_window or widget is self._controls_overlay:
                continue
            if not widget.isVisible() or not isinstance(widget, QDialog):
                continue
            parent = widget.parentWidget()
            while parent is not None:
                if parent is top_window:
                    return True
                parent = parent.parentWidget()
        return False

    def _update_vlc_playback_progress(self):
        """Poll active media (VLC or animated GIF) and update overlay."""
        if isinstance(self._video_ui, VideoUI):
            if not _VLC_AVAILABLE or not self.vlc_media_player:
                return
            duration_ms = max(int(self.vlc_media_player.get_length() or 0), 0)
            current_ms = max(int(self.vlc_media_player.get_time() or 0), 0)
            if duration_ms > 0:
                self.update_playback_progress(current_ms, duration_ms)
            self.set_playback_paused(not bool(self.vlc_media_player.is_playing()))
            self._sync_overlay_volume_state()
            return
        if self._gif_movie is not None and self._gif_is_animated:
            self._update_gif_overlay_progress()

    def _sync_overlay_volume_state(self, force: bool = False):
        """Push volume state to the overlay controls.

        In EQ mode the cached _last_known_* values are authoritative, so they
        are pushed directly.  In global-volume mode VLC state is polled first
        so the overlay stays in sync with any changes made outside this class.
        """
        if config.media_volume_use_eq:
            if force:
                self._controls_overlay.set_volume_state(self._last_known_volume, self._last_known_muted)
        else:
            volume = self.get_volume()
            muted = self.is_muted()
            if force or volume != self._last_known_volume or muted != self._last_known_muted:
                self._last_known_volume = volume
                self._last_known_muted = muted
                self._controls_overlay.set_volume_state(volume, muted)

    def get_media_frame_handle(self):
        """Return window id for VLC embedding (muse/playback.py)."""
        wid = self.winId()
        return int(wid) if wid else None

    # ------------------------------------------------------------------
    # Playback progress & overlay
    # ------------------------------------------------------------------
    def update_playback_progress(self, current_ms: int, duration_ms: int):
        self._controls_overlay.update_progress(current_ms, duration_ms)

    def set_playback_paused(self, paused: bool):
        self._controls_overlay.set_paused(paused)

    def on_track_changed(self):
        self._controls_overlay.on_track_changed()

    def on_playback_stopped(self):
        self._controls_overlay.on_playback_stopped()

"""
Shared helpers for classifying video (and related) media paths.

Centralizes extension checks, container sniffing, and a few shared media helpers.
"""

from __future__ import annotations

import functools
import os

from utils.audio_media import is_audio_for_display, is_audio_path_by_extension
from utils.config import config
from utils.constants import MediaType

# Used when ``config.video_types`` is missing or empty (matches media_frame fallback).
DEFAULT_VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".wmv", ".mov", ".flv",
    ".webm", ".m4v", ".ogv", ".mpeg", ".mpg",
)

# Matroska/WebM containers where VLC stop() may hang without a Cues index.
MATROSKA_EXTENSIONS = frozenset({".webm", ".mkv", ".mka", ".mks"})

# Extensions that may use QMovie when Qt reports animation (see MediaFrame).
ANIMATED_IMAGE_SUFFIXES = (
    ".gif", ".webp", ".apng", ".jpg", ".jpeg", ".jpe", ".jfif",
)


def get_video_extensions() -> tuple[str, ...]:
    """Configured video extensions, lowercased. Missing attribute uses :data:`DEFAULT_VIDEO_EXTENSIONS`; an empty list stays empty."""
    vt = getattr(config, "video_types", None)
    if vt is None:
        return DEFAULT_VIDEO_EXTENSIONS
    return tuple(str(e).lower() for e in vt)


def is_video_path_by_extension(path: str) -> bool:
    """
    True if *path*'s suffix matches a configured video extension.

    Does not require the path to exist or ``enable_videos`` to be set — use for
    routing/display logic (e.g. media frame, frame cache when combined with
    ``enable_videos``).
    """
    if not path:
        return False
    if is_audio_path_by_extension(path):
        return False
    path_lower = path.lower()
    return any(path_lower.endswith(ext) for ext in get_video_extensions())


def is_video_container_signature(path: str) -> bool:
    """
    Detect common video containers from file signatures, regardless of extension.

    Useful for mislabeled files (e.g. MP4 payload with a wrong suffix).
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError:
        return False
    if len(head) < 12:
        return False

    # ISO BMFF / MP4 family: [size:4][ftyp:4][major_brand:4]
    if head[4:8] == b"ftyp":
        major_brand = head[8:12].lower()
        image_brands = {
            b"heic", b"heix", b"hevc", b"hevx",
            b"mif1", b"msf1",
            b"avif", b"avis",
        }
        if major_brand in image_brands:
            return False

        compatible = head[12:64].lower()
        video_markers = (
            b"isom", b"iso2", b"avc1", b"hvc1", b"hev1",
            b"mp41", b"mp42", b"m4v ", b"3gp", b"qt  ",
        )
        return major_brand in video_markers or any(m in compatible for m in video_markers)

    # WebM / Matroska (EBML)
    if head.startswith(b"\x1A\x45\xDF\xA3"):
        return True

    # Ogg container
    if head.startswith(b"OggS"):
        return True

    return False


def is_video_for_display(path: str) -> bool:
    """Extension match or container signature (media frame / VLC routing)."""
    if is_audio_path_by_extension(path):
        return False
    return is_video_path_by_extension(path) or is_video_container_signature(path)


def get_media_type_for_path(path: str) -> MediaType:
    """
    Classify *path* by suffix and config flags (videos, GIF, PDF, SVG, HTML).

    Returns :data:`~utils.constants.MediaType.UNCONFIGURED` when *path* is missing or
    not a string, when a video extension is present but videos are disabled, or when
    the suffix matches GIF/PDF/SVG/HTML but that category is disabled in config.

    Otherwise returns a concrete type; generic raster/unknown extensions map to ``IMAGE``.
    """
    if not path or not isinstance(path, str):
        return MediaType.UNCONFIGURED

    lower = path.lower()

    if is_audio_path_by_extension(path):
        if getattr(config, "enable_audio", False):
            return MediaType.AUDIO
        return MediaType.UNCONFIGURED

    if is_video_path_by_extension(path):
        if config.enable_videos:
            return MediaType.VIDEO
        return MediaType.UNCONFIGURED

    if lower.endswith(".gif"):
        if config.enable_gifs:
            return MediaType.GIF
        return MediaType.UNCONFIGURED

    if lower.endswith(".pdf"):
        if config.enable_pdfs:
            return MediaType.PDF
        return MediaType.UNCONFIGURED

    if lower.endswith(".svg"):
        if config.enable_svgs:
            return MediaType.SVG
        return MediaType.UNCONFIGURED

    if lower.endswith(".html") or lower.endswith(".htm"):
        if config.enable_html:
            return MediaType.HTML
        return MediaType.UNCONFIGURED

    return MediaType.IMAGE


def is_video_file(path: str) -> bool:
    """
    True when *path* is an existing file, videos are enabled, and the suffix is a
    configured video type — suitable for file operations (e.g. strip audio).
    """
    if not path or not os.path.isfile(path):
        return False
    if not config.enable_videos:
        return False
    if is_audio_path_by_extension(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in set(get_video_extensions())


@functools.lru_cache(maxsize=128)
def _pdf_page_count_cached(path: str, mtime_ns: int) -> int:
    try:
        import pypdfium2 as pdfium  # type: ignore[import-untyped]
    except ImportError:
        return 0
    try:
        pdf = pdfium.PdfDocument(path)
        try:
            return max(0, len(pdf))
        finally:
            close = getattr(pdf, "close", None)
            if callable(close):
                close()
    except Exception:
        return 0


def get_pdf_page_count(path: str) -> int:
    """Page count for a PDF file, or 0 if missing/unreadable (callers gate on config flags)."""
    if not path or not path.lower().endswith(".pdf"):
        return 0
    if not os.path.isfile(path):
        return 0
    try:
        mtime_ns = int(os.stat(path).st_mtime_ns)
    except OSError:
        return 0
    return _pdf_page_count_cached(os.path.normcase(os.path.abspath(path)), mtime_ns)


def is_animated_image_candidate(path: str) -> bool:
    """True for paths that may carry animation frames (GIF/WebP/APNG, etc.)."""
    if not path:
        return False
    return path.lower().endswith(ANIMATED_IMAGE_SUFFIXES)


def scale_dims(
    dims: tuple[int, int],
    max_dims: tuple[int, int],
    maximize: bool = False,
) -> tuple[int, int]:
    """Return (width, height) to fit *dims* inside *max_dims*. If *maximize*, fill when smaller."""
    x, y = dims[0], dims[1]
    max_x, max_y = max_dims[0], max_dims[1]
    if x <= max_x and y <= max_y:
        if maximize:
            if x < max_x:
                return (int(x * max_y / y), max_y)
            if y < max_y:
                return (max_x, int(y * max_x / x))
        return (x, y)
    if x <= max_x:
        return (int(x * max_y / y), max_y)
    if y <= max_y:
        return (max_x, int(y * max_x / x))
    x_scale = max_x / x
    y_scale = max_y / y
    if x_scale < y_scale:
        return (int(x * x_scale), int(y * x_scale))
    return (int(x * y_scale), int(y * y_scale))


def large_image_dim_threshold() -> int:
    return max(1, int(getattr(config, "large_image_dim_threshold_px", 5000)))


def large_preview_overscan() -> float:
    return max(1.0, float(getattr(config, "large_image_preview_overscan", 1.5)))


def large_preview_max_dim() -> int:
    return max(512, int(getattr(config, "large_image_preview_max_dim", 4096)))


def large_hq_downscale_enabled() -> bool:
    return bool(getattr(config, "large_image_enable_hq_idle_downscale", True))


def large_hq_downscale_ratio_threshold() -> float:
    return max(1.1, float(getattr(config, "large_image_hq_downscale_ratio_threshold", 1.8)))


def large_image_full_res_promotion_enabled() -> bool:
    return bool(getattr(config, "large_image_enable_full_res_promotion", True))


def large_image_promotion_min_free_ram_gb() -> float:
    return max(0.0, float(getattr(config, "large_image_promotion_min_free_ram_gb", 1.0)))


def large_image_promotion_max_estimated_mb() -> int:
    return max(64, int(getattr(config, "large_image_promotion_max_estimated_mb", 512)))


def large_image_promotion_available_ram_fraction() -> float:
    value = float(getattr(config, "large_image_promotion_available_ram_fraction", 0.25))
    return min(max(value, 0.05), 0.9)


def is_large_image_dims(dims: tuple[int, int]) -> bool:
    w, h = dims
    if w <= 0 or h <= 0:
        return False
    threshold = large_image_dim_threshold()
    return w > threshold or h > threshold


def get_image_dimensions(path: str) -> tuple[int, int] | None:
    """Return (width, height) for a PIL-readable image, or None if the file
    cannot be opened as an image (video, audio, missing, corrupt, etc.)."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def is_classifier_dynamic_media_path(path: str) -> bool:
    """True when *path* is an existing video, GIF, or PDF file with that type enabled in config."""
    if not path or not os.path.isfile(path):
        return False
    lower = path.lower()
    if config.enable_videos and is_video_path_by_extension(path):
        return True
    if config.enable_gifs and lower.endswith(".gif"):
        return True
    if config.enable_pdfs and lower.endswith(".pdf"):
        return True
    return False

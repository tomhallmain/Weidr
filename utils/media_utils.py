"""
Shared helpers for classifying video (and related) media paths.

Centralizes extension checks, container sniffing, and a few shared media helpers.
"""

from __future__ import annotations

import functools
import os

from utils.config import config
from utils.constants import MediaType

# Used when ``config.video_types`` is missing or empty (matches media_frame fallback).
DEFAULT_VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".wmv", ".mov", ".flv",
    ".webm", ".m4v", ".ogv", ".mpeg", ".mpg",
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

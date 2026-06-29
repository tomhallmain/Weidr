import hashlib
import os
import tempfile
import asyncio
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

# Quieter libav*/swscale stderr from OpenCV's bundled FFmpeg when supported.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except (AttributeError, cv2.error):
    pass

has_imported_pypdfium2 = False
try:
    import pypdfium2 as pdfium
    has_imported_pypdfium2 = True
except ImportError:
    pass

has_imported_cairosvg = False
try:
    import cairosvg
    has_imported_cairosvg = True
except ImportError:
    pass

has_imported_pyppeteer = False
try:
    from pyppeteer import launch
    has_imported_pyppeteer = True
except ImportError:
    pass

has_imported_pyav = False
try:
    import av

    has_imported_pyav = True
    try:
        av.logging.set_level(av.logging.ERROR)
    except (AttributeError, ValueError):
        pass
except ImportError:
    av = None  # type: ignore[misc, assignment]

from utils.config import config
from utils.logging_setup import get_logger
from utils.constants import CompareMediaType
from utils.media_utils import is_video_path_by_extension

logger = get_logger("frame_cache")


@dataclass
class MediaStats:
    """Lightweight metadata stored in :attr:`FrameCache.media_stats_cache` per media path."""
    media_type: str = "other"               # "video", "gif", "pdf", or "other"
    total_items: Optional[int] = None       # total frames (video/GIF) or pages (PDF)
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    frames_are_pseudostatic: Optional[bool] = None  # None = not yet tested
    single_frame_stream: bool = False       # True only for broken single-frame streams


@dataclass
class SeekPosition:
    """A resolved seek target derived from a TriggerFrameResult."""
    kind: str   # "ms" for video/GIF, "page" for PDF
    value: int  # milliseconds (video/GIF) or 0-based page number (PDF)


def _stats_media_type_for_sampled_video_path(media_path: str) -> str:
    """``media_type`` value stored in :attr:`FrameCache.media_stats_cache` for debug / UI."""
    return "gif" if media_path.lower().endswith(".gif") else "video"


# Bumps sample cache keys when extraction semantics change (invalidates in-memory entries).
_VIDEO_SAMPLE_CACHE_REV = "pyav2_gif_dynamic"

# Only emit a sampling-incomplete WARNING when this fraction of planned targets are missing.
# Small gaps (e.g. 1-2 frames past end-of-stream) are normal and silenced; large gaps
# (e.g. 19/20 from a buffer-deadlock video) are worth surfacing.
_SAMPLING_WARN_MISSING_RATIO = 0.5


def _stable_media_path_hash(media_path: str) -> str:
    """Deterministic ASCII-safe name component from absolute media path (for temp output files)."""
    n = os.path.normpath(os.path.abspath(media_path))
    return hashlib.sha256(n.encode("utf-8")).hexdigest()


def _make_sample_cache_key(media_path: str, ratio: float) -> str:
    min_sample_count = config.dynamic_media_min_sample_count
    max_sample_frames = config.dynamic_media_max_sample_frames
    max_sample_pages = config.dynamic_media_max_sample_pages
    return (
        f"{media_path}|{ratio:.4f}|min:{min_sample_count}"
        f"|maxf:{max_sample_frames}|maxp:{max_sample_pages}|{_VIDEO_SAMPLE_CACHE_REV}"
    )


def _open_video_capture(path: str) -> cv2.VideoCapture:
    """
    Prefer FFmpeg backend for file-backed video; default (e.g. MSMF on Windows) often
    mishandles H.264/HEVC seek and returns blank frames after CAP_PROP_POS_FRAMES.
    """
    apis: List[int] = []
    ff = getattr(cv2, "CAP_FFMPEG", None)
    if ff is not None:
        apis.append(int(ff))
    any_api = getattr(cv2, "CAP_ANY", 0)
    if any_api not in apis:
        apis.append(int(any_api))

    last_cap: Optional[cv2.VideoCapture] = None
    for api in apis:
        cap = cv2.VideoCapture(path, api)
        last_cap = cap
        if cap.isOpened():
            _configure_video_capture(cap)
            return cap
    cap = last_cap if last_cap is not None else cv2.VideoCapture(path)
    if cap.isOpened():
        _configure_video_capture(cap)
    return cap


def _configure_video_capture(cap: cv2.VideoCapture) -> None:
    """
    Mitigate broken or noisy FFmpeg decoding (swscale slice errors, blank frames)
    from hardware paths and odd RGB conversion.
    """
    if not cap.isOpened():
        return
    cc = getattr(cv2, "CAP_PROP_CONVERT_RGB", None)
    if cc is not None:
        try:
            cap.set(int(cc), 1.0)
        except cv2.error:
            pass
    ha_prop = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
    ha_none = getattr(cv2, "VIDEO_ACCELERATION_NONE", None)
    if ha_prop is not None and ha_none is not None:
        try:
            cap.set(int(ha_prop), int(ha_none))
        except cv2.error:
            pass


def _normalize_decoded_frame(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Ensure uint8 BGR and even width/height. Odd dimensions and exotic dtypes often
    trigger libswscale \"Slice parameters … are invalid\" and black or corrupt output.
    """
    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return None
    if frame.ndim < 2:
        return None
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        mx = float(np.nanmax(frame)) if frame.size else 0.0
        if mx <= 1.0:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(np.asarray(frame, dtype=np.float64), 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3:
        ch = int(frame.shape[2])
        if ch == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif ch == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif ch != 3:
            return None
    h, w = int(frame.shape[0]), int(frame.shape[1])
    ph, pw = h % 2, w % 2
    if ph or pw:
        frame = cv2.copyMakeBorder(frame, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return frame


def _is_likely_decoder_blank(frame: np.ndarray) -> bool:
    """
    Detect near-uniform black frames that commonly appear when random frame seek fails
    but decode still "succeeds". Deliberately ignores legitimately very dark scenes by
    requiring simultaneously low mean, low variance, and low channel peaks.
    """
    if frame is None or frame.size == 0:
        return True
    if frame.ndim < 2:
        return True
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_m = cv2.mean(gray)[0]
    _unused, stddev = cv2.meanStdDev(gray)
    std_m = float(stddev[0, 0])
    peak = float(np.max(gray))
    return peak < 6.0 and mean_m < 2.5 and std_m < 3.0


def _frames_are_visually_pseudostatic(
    frame_paths: List[str],
    diff_threshold: float = 0.03,
) -> bool:
    """True if every checked pair of JPEG frame paths appears visually identical.

    Checks up to three spread pairs via mean absolute pixel difference on
    64×64 thumbnails.  Returns False as soon as any pair exceeds
    *diff_threshold* (fraction of 255 per channel).  Requires at least two
    actual JPEG paths.
    """
    jpg_paths = [p for p in frame_paths if p.lower().endswith((".jpg", ".jpeg"))]
    if len(jpg_paths) < 2:
        return False

    n = len(jpg_paths)
    pairs: List[Tuple[int, int]] = [(0, 1)]
    if n > 2:
        pairs.append((0, n - 1))
    if n > 3:
        pairs.append((n // 2, n - 1))

    seen: set = set()
    for i, j in pairs:
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        img1 = cv2.imread(jpg_paths[i])
        img2 = cv2.imread(jpg_paths[j])
        if img1 is None or img2 is None:
            return False
        t1 = cv2.resize(img1, (64, 64)).astype(np.float32)
        t2 = cv2.resize(img2, (64, 64)).astype(np.float32)
        if np.abs(t1 - t2).mean() / 255.0 >= diff_threshold:
            return False

    return True


def _read_frame_via_imageio(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Fallback decode path using imageio (own FFmpeg pipeline) when OpenCV swscale fails."""
    try:
        import imageio.v3 as iio
    except ImportError:
        return None
    try:
        arr = iio.imread(video_path, index=int(frame_index))
    except Exception as e:
        logger.debug(
            "imageio frame read failed for %s index=%s: %s",
            video_path,
            frame_index,
            e,
        )
        return None
    if arr is None or arr.size == 0:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3:
        c = arr.shape[2]
        if c == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif c >= 3:
            arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        elif c == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return _normalize_decoded_frame(arr)


def _opencv_first_decoded_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """Single read + normalize; no blank-frame scan (fast thumbnail path)."""
    ret, raw = cap.read()
    if not ret or raw is None:
        return False, None
    frame = _normalize_decoded_frame(raw)
    return (frame is not None, frame)


def _first_substantive_frame(
    cap: cv2.VideoCapture,
    video_path: Optional[str] = None,
    max_frames: int = 360,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Read forward until the first non-blank frame or EOF (when ``config.debug2``)."""
    for _unused in range(max(1, max_frames)):
        ret, raw = cap.read()
        if not ret or raw is None:
            break
        frame = _normalize_decoded_frame(raw)
        if frame is not None and not _is_likely_decoder_blank(frame):
            return True, frame
    if video_path:
        io_frame = _read_frame_via_imageio(video_path, 0)
        if io_frame is not None and not _is_likely_decoder_blank(io_frame):
            return True, io_frame
    return False, None


def _pyav_video_stats(path: str) -> Tuple[int, float, Optional[float]]:
    """Frame count (estimate if needed), average FPS, and stream duration in seconds."""
    assert av is not None
    container = av.open(path, metadata_errors="ignore")
    try:
        if not container.streams.video:
            return 0, 0.0, None
        s = container.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else 0.0
        duration_s: Optional[float] = None
        if s.duration is not None:
            duration_s = float(s.duration * s.time_base)
        total = 0
        if getattr(s, "frames", None) not in (None, 0):
            total = int(s.frames)
        elif duration_s is not None and fps > 0:
            total = max(0, int(duration_s * fps))
        return total, fps, duration_s
    finally:
        container.close()


def _adjust_duration_for_file_size(
    max_duration_seconds: float,
    file_size_mb: float,
    max_size_mb: float,
) -> float:
    """
    Scale *max_duration_seconds* down when the file is larger than *max_size_mb*.

    The scale factor is ``max_size_mb / file_size_mb``, so a file twice the
    threshold receives half the duration cap.  The result is floored at 30 s
    so the cap never becomes trivially short.  Returns *max_duration_seconds*
    unchanged when size-based scaling is disabled (``max_size_mb <= 0``), the
    file is within the threshold, or the duration cap itself is already disabled.
    """
    if max_size_mb <= 0 or max_duration_seconds <= 0 or file_size_mb <= max_size_mb:
        return max_duration_seconds
    scale = max_size_mb / file_size_mb
    return max(30.0, max_duration_seconds * scale)


def _apply_duration_cap(
    total_frames: int,
    fps: float,
    duration_seconds: Optional[float],
    max_duration_seconds: float,
) -> int:
    """
    Return the effective frame count for sampling, capped to *max_duration_seconds*.

    When ``max_duration_seconds <= 0`` or the video is shorter than the cap,
    *total_frames* is returned unchanged so sampling covers the full video.
    Otherwise the returned value is ``min(total_frames, fps * max_duration_seconds)``,
    confining sample indices to the first N seconds of the file.
    """
    if max_duration_seconds <= 0 or total_frames <= 0:
        return total_frames
    known_duration = duration_seconds if duration_seconds is not None else (
        total_frames / fps if fps > 0 else None
    )
    if known_duration is None or known_duration <= max_duration_seconds:
        return total_frames
    if fps <= 0:
        return total_frames
    return max(1, int(fps * max_duration_seconds))


def _pyav_first_decoded_bgr(video_path: str) -> Optional[np.ndarray]:
    """First successfully decoded frame only (fast; no blank-frame scan)."""
    assert av is not None
    container = av.open(video_path, metadata_errors="ignore")
    try:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        for frame in container.decode(stream):
            try:
                raw = frame.to_ndarray(format="bgr24")
            except Exception:
                continue
            bgr = _normalize_decoded_frame(np.asarray(raw))
            if bgr is not None:
                return bgr
    except Exception:
        return None
    finally:
        container.close()
    return None


def _pyav_first_substantive_bgr(video_path: str, max_frames: int = 360) -> Optional[np.ndarray]:
    assert av is not None
    container = av.open(video_path, metadata_errors="ignore")
    try:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        for i, frame in enumerate(container.decode(stream)):
            if i >= max_frames:
                break
            try:
                bgr = frame.to_ndarray(format="bgr24")
            except Exception:
                continue
            bgr = _normalize_decoded_frame(np.asarray(bgr))
            if bgr is not None and not _is_likely_decoder_blank(bgr):
                return bgr
    except Exception:
        return None
    finally:
        container.close()
    return None


class FrameCache:
    """
    A cache for extracting and storing the first frame from various media types (videos, GIFs, PDFs, SVGs, HTMLs).
    This helps improve performance by avoiding repeated frame extraction operations.

    TODO support getting an "average" frame from a video, or at least an average embedding
    TODO(GC): Implement bounded cache eviction / periodic cleanup for sampled
    dynamic-media frames. Current sampled cache can grow large with many media
    files and high sampling limits.
    """
    temporary_directory = tempfile.TemporaryDirectory(prefix="tmp_comp_frames")
    cache: Dict[str, str] = {}  # Maps media_path to cached image path
    sampled_cache: Dict[str, List[str]] = {}  # Maps media_path|sample_ratio to sampled frame paths
    media_stats_cache: Dict[str, MediaStats] = {}  # Maps media_path to lightweight stats

    @classmethod
    def _write_cv2_jpeg(cls, frame: np.ndarray, path: str) -> Optional[str]:
        if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]) and os.path.isfile(path):
            return os.path.abspath(path)
        logger.debug("cv2.imwrite failed for %s", path)
        return None

    @classmethod
    def _write_pil_image(cls, pil_image, path: str, **save_kw) -> Optional[str]:
        try:
            pil_image.save(path, **save_kw)
        except OSError as e:
            logger.debug("PIL save failed for %s: %s", path, e)
            return None
        if os.path.isfile(path):
            return os.path.abspath(path)
        return None

    @classmethod
    def get_any_cached_sampled_frame(cls, media_path: str) -> Optional[str]:
        """Return the first valid JPEG from any cached sample set for *media_path*.

        Scans all sampled_cache entries whose key starts with the media path,
        so callers don't need to know which sample ratio was used.  Returns
        None when nothing is cached yet or every cached path no longer exists.
        """
        prefix = media_path + "|"
        for key, paths in cls.sampled_cache.items():
            if key.startswith(prefix):
                for p in paths:
                    if p.lower().endswith((".jpg", ".jpeg")) and os.path.isfile(p):
                        return p
        return None

    @classmethod
    def get_cached_sampled_frame_paths_if_any(
        cls, media_path: str, sample_ratio: float
    ) -> Optional[List[str]]:
        """Return sampled frame paths if :meth:`stream_frame_samples` has already materialized them."""
        try:
            ratio = float(sample_ratio)
        except Exception:
            ratio = 0.1
        ratio = max(0.0, min(1.0, ratio))
        cache_key = _make_sample_cache_key(media_path, ratio)
        if cache_key in cls.sampled_cache:
            return list(cls.sampled_cache[cache_key])
        return None

    @classmethod
    def get_image_path(cls, media_path: str) -> str:
        """
        Get the image path for a media file. If it's a video/GIF/PDF/SVG/HTML, extracts the first frame.
        Otherwise returns the original path.

        Args:
            media_path: Path to the media file

        Returns:
            Path to the image file (either original or extracted frame)
        """
        media_path_lower = media_path.lower()

        # Check for SVG files first (since they're the simplest to convert)
        if media_path_lower.endswith('.svg'):
            if config.enable_svgs:
                if has_imported_cairosvg:
                    return cls.get_first_frame(media_path, CompareMediaType.SVG)
                else:
                    raise ImportError("Unable to convert SVG to PNG: cairosvg is not installed")
            else:
                return media_path

        # Check for PDF files next
        if media_path_lower.endswith('.pdf'):
            if config.enable_pdfs:
                if has_imported_pypdfium2:
                    return cls.get_first_frame(media_path, CompareMediaType.PDF)
                else:
                    raise ImportError("Unable to extract PDF frame: pypdfium2 is not installed")
            else:
                return media_path

        # Check for HTML files
        if media_path_lower.endswith('.html') or media_path_lower.endswith('.htm'):
            if config.enable_html:
                if has_imported_pyppeteer:
                    return cls.get_first_frame(media_path, CompareMediaType.HTML)
                else:
                    raise ImportError("Unable to convert HTML to image: pyppeteer is not installed")
            else:
                return media_path

        # Check for video types from config (which may be dynamic)
        if config.enable_videos and is_video_path_by_extension(media_path):
            return cls.get_first_frame(media_path, CompareMediaType.VIDEO)

        return media_path

    @classmethod
    def get_first_frame(cls, media_path: str, media_type: CompareMediaType) -> str:
        """
        Get the first frame from a media file, using cache if available.

        Args:
            media_path: Path to the media file
            media_type: Type of media (from CompareMediaType enum)

        Returns:
            Path to the extracted frame image
        """
        if media_path not in cls.cache:
            cls.set_first_frame(media_path, media_type)
        return cls.cache[media_path]

    @classmethod
    def set_first_frame(cls, media_path: str, media_type: CompareMediaType) -> None:
        """
        Extract and cache the first frame from a media file.

        Args:
            media_path: Path to the media file
            media_type: Type of media (from CompareMediaType enum)
        """
        try:
            if media_type == CompareMediaType.PDF:
                cls._extract_pdf_frame(media_path)
            elif media_type == CompareMediaType.SVG:
                cls._extract_svg_frame(media_path)
            elif media_type == CompareMediaType.HTML:
                cls._extract_html_frame(media_path)
            else:
                cls._extract_video_frame(media_path)
        except Exception as e:
            logger.error(f"Error extracting frame from {media_path}: {str(e)}")
            # Fallback to original path if extraction fails
            cls.cache[media_path] = media_path

    @classmethod
    def get_pdf_page(cls, pdf_path: str, page_index: int) -> str:
        """Return the temp JPEG path for *page_index* of *pdf_path*, rendering on demand.

        Caches in the temp directory; repeated calls for the same page are instant.
        Also populates ``media_stats_cache`` with the total page count and, for
        page 0, updates ``cls.cache[pdf_path]`` so masonry thumbnails stay current.
        """
        mh = _stable_media_path_hash(pdf_path)
        page_path = os.path.join(cls.temporary_directory.name, f"{mh}_page_{page_index}.jpg")
        if os.path.isfile(page_path):
            if pdf_path not in cls.media_stats_cache:
                pdf = pdfium.PdfDocument(pdf_path)
                cls.media_stats_cache[pdf_path] = MediaStats(media_type="pdf", total_items=len(pdf))
            if page_index == 0:
                cls.cache[pdf_path] = page_path
            return page_path
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
        if n_pages == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        if page_index >= n_pages:
            raise IndexError(f"Page {page_index} out of range for {pdf_path} ({n_pages} pages)")
        cls.media_stats_cache[pdf_path] = MediaStats(media_type="pdf", total_items=n_pages)
        image = pdf[page_index].render(scale=4, fill_color=(255, 255, 255, 255)).to_pil()
        resolved = cls._write_pil_image(image, page_path, quality=95)
        if resolved is None:
            raise OSError(f"Could not write PDF page {page_index} for {pdf_path}")
        if page_index == 0:
            cls.cache[pdf_path] = resolved
        return resolved

    @classmethod
    def _extract_pdf_frame(cls, pdf_path: str) -> None:
        """Extract the first page from a PDF as an image (thumbnail path)."""
        try:
            logger.info(f"Extracting first page from PDF: {pdf_path}")
            cls.get_pdf_page(pdf_path, 0)
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    @classmethod
    def _extract_svg_frame(cls, svg_path: str) -> None:
        """
        Convert an SVG file to a PNG image.

        Args:
            svg_path: Path to the SVG file
        """
        try:
            logger.info(f"Converting SVG to PNG: {svg_path}")
            mh = _stable_media_path_hash(svg_path)
            frame_path = os.path.join(cls.temporary_directory.name, f"{mh}.png")
            
            # Convert SVG to PNG using cairosvg.
            # NOTE: passing background_color="white" here would fix transparent-background
            # SVGs on dark UIs, but many SVGs intentionally use transparency for compositing
            # so it is left unset for now.
            cairosvg.svg2png(url=svg_path, write_to=frame_path)
            cls.cache[svg_path] = frame_path
        except Exception as e:
            logger.error(f"Error processing SVG {svg_path}: {str(e)}")
            raise

    @classmethod
    def _extract_html_frame(cls, html_path: str) -> None:
        """
        Convert an HTML file to a PDF and then extract its first page as an image.

        Args:
            html_path: Path to the HTML file
        """
        try:
            logger.info(f"Converting HTML to image: {html_path}")
            # First convert HTML to PDF
            mh = _stable_media_path_hash(html_path)
            pdf_path = os.path.join(cls.temporary_directory.name, f"{mh}_from_html.pdf")
            
            # Convert HTML to PDF using Pyppeteer
            async def convert_html_to_pdf():
                browser = await launch(headless=True)
                page = await browser.newPage()
                
                # Read the HTML file
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Set the content and wait for network idle
                await page.setContent(html_content, {'waitUntil': 'networkidle0'})
                
                # Generate PDF with good quality settings
                await page.pdf({
                    'path': pdf_path,
                    'format': 'A4',
                    'printBackground': True,
                    'margin': {
                        'top': '0',
                        'right': '0',
                        'bottom': '0',
                        'left': '0'
                    }
                })
                
                await browser.close()
            
            # Run the async function
            asyncio.get_event_loop().run_until_complete(convert_html_to_pdf())
            
            # Now extract the first page as an image using our existing PDF extraction
            cls._extract_pdf_frame(pdf_path)
            
            # Update cache to point to the HTML file instead of the temporary PDF
            cls.cache[html_path] = cls.cache[pdf_path]
            del cls.cache[pdf_path]
            
        except Exception as e:
            logger.error(f"Error processing HTML {html_path}: {str(e)}")
            raise

    @classmethod
    def _extract_video_frame(cls, video_path: str) -> None:
        """
        Extract the first frame from a video/GIF file.

        Uses PyAV (FFmpeg bindings) when available — far more reliable than OpenCV's
        bundled decoder for HEVC/odd dimensions/exotic pixel formats.

        When ``config.debug2`` is true, skips past uniform-black decoder glitches by
        scanning for a substantive frame (slower). When false, uses the first decoded
        frame only.

        Args:
            video_path: Path to the video/GIF file
        """
        logger.info(f"Extracting first frame from video: {video_path}")
        if has_imported_pyav:
            try:
                total_frames, fps, duration_s = _pyav_video_stats(video_path)
                if total_frames <= 0:
                    cap = _open_video_capture(video_path)
                    try:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if fps <= 0:
                            fps = float(cap.get(cv2.CAP_PROP_FPS))
                    finally:
                        cap.release()
                duration_seconds = duration_s
                if duration_seconds is None and fps > 0 and total_frames > 0:
                    duration_seconds = total_frames / fps
                cls.media_stats_cache[video_path] = MediaStats(
                    media_type=_stats_media_type_for_sampled_video_path(video_path),
                    total_items=total_frames if total_frames > 0 else None,
                    duration_seconds=duration_seconds,
                    fps=fps if fps > 0 else None,
                )
                if config.debug2:
                    frame = _pyav_first_substantive_bgr(video_path)
                else:
                    frame = _pyav_first_decoded_bgr(video_path)
                if frame is not None:
                    mh = _stable_media_path_hash(video_path)
                    frame_path = os.path.join(cls.temporary_directory.name, f"{mh}_first.jpg")
                    resolved = cls._write_cv2_jpeg(frame, frame_path)
                    if resolved is None:
                        raise OSError(f"Could not write or resolve video frame path: {frame_path}")
                    cls.cache[video_path] = resolved
                    return
            except Exception as e:
                logger.warning(
                    "PyAV first-frame extract failed for %s (%s); using OpenCV",
                    video_path,
                    e,
                )

        cap = _open_video_capture(video_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            duration_seconds = None
            if fps and fps > 0 and total_frames > 0:
                duration_seconds = total_frames / fps
            cls.media_stats_cache[video_path] = MediaStats(
                media_type=_stats_media_type_for_sampled_video_path(video_path),
                total_items=total_frames if total_frames > 0 else None,
                duration_seconds=duration_seconds,
                fps=fps if fps > 0 else None,
            )
            if config.debug2:
                ok, frame = _first_substantive_frame(cap, video_path=video_path)
            else:
                ok, frame = _opencv_first_decoded_frame(cap)
            if not ok or frame is None:
                raise ValueError("Could not read a frame from the video")

            mh = _stable_media_path_hash(video_path)
            frame_path = os.path.join(cls.temporary_directory.name, f"{mh}_first.jpg")

            resolved = cls._write_cv2_jpeg(frame, frame_path)
            if resolved is None:
                raise ValueError("Could not write extracted frame to disk")
            cls.cache[video_path] = resolved
        finally:
            cap.release()

    @classmethod
    def stream_frame_samples(
        cls,
        media_path: str,
        sample_ratio: float = 0.1,
        detect_pseudostatic: bool = False,
        max_samples: Optional[int] = None,
    ) -> Tuple[int, Iterator[str]]:
        """
        Lazily produce sampled frame paths for video/GIF/PDF (or a single still path).

        Returns ``(planned_slot_count, iterator)``. *planned_slot_count* is the number
        of sampling slots (``len(frame_indices)`` or ``1`` for fallback), used for
        early-exit thresholds before all frames are decoded. The iterator decodes,
        writes JPEGs, and yields paths one at a time so consumers can stop early.

        On full consumption, results are stored in ``sampled_cache`` (same as
        :meth:`get_frame_samples`). If the consumer stops early, the partial result
        is not cached.

        Falls back to ``get_image_path`` for non-dynamic media.

        max_samples: when provided, overrides the config cap on how many frames/pages
        are sampled. Pass a very large value (e.g. ``2**31 - 1``) to sample all
        available frames — useful for interactive seek operations where precision
        matters more than speed. Results are cached under a separate key so normal
        batch scans are not affected.
        """
        media_path_lower = media_path.lower()
        is_video = config.enable_videos and is_video_path_by_extension(media_path)
        is_gif = config.enable_gifs and media_path_lower.endswith(".gif")
        is_pdf = media_path_lower.endswith(".pdf") and config.enable_pdfs and has_imported_pypdfium2
        if not is_video and not is_gif and not is_pdf:
            p = cls.get_image_path(media_path)
            return 1, iter([p])

        try:
            ratio = float(sample_ratio)
        except Exception:
            ratio = 0.1
        ratio = max(0.0, min(1.0, ratio))
        min_sample_count = config.dynamic_media_min_sample_count
        max_sample_frames = max_samples if max_samples is not None else config.dynamic_media_max_sample_frames
        max_sample_pages = max_samples if max_samples is not None else config.dynamic_media_max_sample_pages
        max_sample_duration_seconds = config.dynamic_media_max_sample_duration_seconds
        max_sample_size_mb = config.dynamic_media_max_sample_size_mb
        # When max_samples is overridden, use a distinct cache key so normal scans
        # are not polluted with denser results (and vice-versa).
        if max_samples is not None:
            cache_key = f"{_make_sample_cache_key(media_path, ratio)}|n{max_samples}"
        else:
            cache_key = _make_sample_cache_key(media_path, ratio)

        if cache_key in cls.sampled_cache:
            cached = cls.sampled_cache[cache_key]
            if detect_pseudostatic:
                cached_stats = cls.media_stats_cache.get(media_path)
                if cached_stats is None or cached_stats.frames_are_pseudostatic is None:
                    real = [p for p in cached if p.lower().endswith((".jpg", ".jpeg"))]
                    if len(real) >= 2:
                        is_pseudo = _frames_are_visually_pseudostatic(real)
                        if cached_stats is None:
                            cached_stats = MediaStats()
                            cls.media_stats_cache[media_path] = cached_stats
                        cached_stats.frames_are_pseudostatic = is_pseudo
            return len(cached), iter(cached)

        if is_video or is_gif:
            try:
                file_size_mb = os.path.getsize(media_path) / (1024 * 1024)
            except OSError:
                file_size_mb = 0.0
            effective_duration = _adjust_duration_for_file_size(
                max_sample_duration_seconds, file_size_mb, max_sample_size_mb
            )
            size_cap_logged = effective_duration < max_sample_duration_seconds and max_sample_duration_seconds > 0
            if size_cap_logged:
                logger.debug(
                    "Reducing duration cap from %.0fs to %.0fs for large file (%.0f MB) for %s",
                    max_sample_duration_seconds, effective_duration, file_size_mb, media_path,
                )
            return cls._stream_video_frame_samples_dispatch(
                media_path,
                ratio,
                min_sample_count=min_sample_count,
                max_sample_count=max_sample_frames,
                max_duration_seconds=effective_duration,
                suppress_cap_log=size_cap_logged,
                cache_key=cache_key,
                detect_pseudostatic=detect_pseudostatic,
            )
        return cls._stream_pdf_sample_pages(
            media_path,
            ratio,
            min_sample_count=min_sample_count,
            max_sample_count=max_sample_pages,
            cache_key=cache_key,
        )

    @classmethod
    def get_frame_samples(cls, media_path: str, sample_ratio: float = 0.1) -> List[str]:
        """
        Get sampled frame image paths for dynamic media (currently video/GIF/PDF).

        Materializes :meth:`stream_frame_samples` (full decode). Falls back to
        ``get_image_path`` when sampling is not applicable or fails.
        """
        _unused, path_iter = cls.stream_frame_samples(media_path, sample_ratio)
        sampled_paths = list(path_iter)
        media_path_lower = media_path.lower()
        is_video = config.enable_videos and is_video_path_by_extension(media_path)
        is_gif = config.enable_gifs and media_path_lower.endswith(".gif")
        is_pdf = media_path_lower.endswith(".pdf") and config.enable_pdfs and has_imported_pypdfium2
        if len(sampled_paths) == 0 and (is_video or is_gif or is_pdf):
            sampled_paths = [media_path]
            try:
                ratio = float(sample_ratio)
            except Exception:
                ratio = 0.1
            ratio = max(0.0, min(1.0, ratio))
            cache_key = _make_sample_cache_key(media_path, ratio)
            cls.sampled_cache[cache_key] = sampled_paths
        return sampled_paths

    @classmethod
    def _stream_video_frame_samples_dispatch(
        cls,
        video_path: str,
        sample_ratio: float,
        min_sample_count: int,
        max_sample_count: int,
        max_duration_seconds: float,
        suppress_cap_log: bool,
        cache_key: str,
        detect_pseudostatic: bool = False,
    ) -> Tuple[int, Iterator[str]]:
        if has_imported_pyav:
            try:
                return cls._stream_pyav_video_frame_samples(
                    video_path,
                    sample_ratio,
                    min_sample_count=min_sample_count,
                    max_sample_count=max_sample_count,
                    max_duration_seconds=max_duration_seconds,
                    suppress_cap_log=suppress_cap_log,
                    cache_key=cache_key,
                    detect_pseudostatic=detect_pseudostatic,
                )
            except Exception as e:
                logger.warning(
                    "PyAV video sampling failed for %s (%s); falling back to OpenCV",
                    video_path,
                    e,
                )
        return cls._opencv_stream_video_frame_samples(
            video_path,
            sample_ratio,
            min_sample_count=min_sample_count,
            max_sample_count=max_sample_count,
            max_duration_seconds=max_duration_seconds,
            suppress_cap_log=suppress_cap_log,
            cache_key=cache_key,
            detect_pseudostatic=detect_pseudostatic,
        )

    @classmethod
    def _stream_pyav_video_frame_samples(
        cls,
        video_path: str,
        sample_ratio: float,
        min_sample_count: int,
        max_sample_count: int,
        max_duration_seconds: float,
        suppress_cap_log: bool,
        cache_key: str,
        detect_pseudostatic: bool = False,
    ) -> Tuple[int, Iterator[str]]:
        assert av is not None
        total_frames, fps, duration_s = _pyav_video_stats(video_path)
        if total_frames <= 0:
            cap = _open_video_capture(video_path)
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps <= 0:
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
            finally:
                cap.release()

        duration_seconds = duration_s
        if duration_seconds is None and fps > 0 and total_frames > 0:
            duration_seconds = total_frames / fps
        cls.media_stats_cache[video_path] = MediaStats(
            media_type=_stats_media_type_for_sampled_video_path(video_path),
            total_items=total_frames if total_frames > 0 else None,
            duration_seconds=duration_seconds,
            fps=fps if fps > 0 else None,
        )
        effective_frames = _apply_duration_cap(total_frames, fps, duration_seconds, max_duration_seconds)
        if effective_frames < total_frames and not suppress_cap_log:
            logger.debug(
                "Capping video sampling to first %.0fs (%.0fs total) for %s",
                max_duration_seconds, duration_seconds, video_path,
            )
        frame_indices = cls._compute_sample_indices(
            total_items=effective_frames,
            sample_ratio=sample_ratio,
            min_sample_count=min_sample_count,
            max_sample_count=max_sample_count,
        )
        if len(frame_indices) == 0:

            def gen_empty() -> Iterator[str]:
                cls.sampled_cache[cache_key] = [video_path]
                yield video_path

            return 1, gen_empty()

        media_hash = _stable_media_path_hash(video_path)
        planned = len(frame_indices)

        def gen() -> Iterator[str]:
            accumulated: List[str] = []
            completed = False
            first_real_frame: Optional[str] = None
            inline_check_done = False
            frames_pseudostatic: Optional[bool] = None
            is_single_frame = False
            try:
                for path in cls._iter_pyav_video_sample_paths(
                    video_path,
                    frame_indices,
                    media_hash,
                    accumulated,
                ):
                    yield path
                    # Inline short-circuit: compare frames 0 and 1 as soon as available.
                    # If they differ we know immediately the video is not pseudostatic.
                    # Only runs when the caller opted in via detect_pseudostatic.
                    if detect_pseudostatic and not inline_check_done and path.lower().endswith((".jpg", ".jpeg")):
                        if first_real_frame is None:
                            first_real_frame = path
                        else:
                            inline_check_done = True
                            if not _frames_are_visually_pseudostatic([first_real_frame, path]):
                                frames_pseudostatic = False
                if len(accumulated) == 0:
                    accumulated.append(video_path)
                    yield video_path
                completed = True
            finally:
                # Runs on natural exhaustion AND on early close() — ensures pseudostatic
                # status is stored even when the consumer exits before the last frame.
                if detect_pseudostatic:
                    if frames_pseudostatic is None:
                        if inline_check_done:
                            # First two frames were identical; confirm with a spread check.
                            real = [p for p in accumulated if p.lower().endswith((".jpg", ".jpeg"))]
                            if len(real) >= 2:
                                frames_pseudostatic = _frames_are_visually_pseudostatic(real)
                        elif first_real_frame is not None:
                            # Only one real frame was ever decoded (buffer-deadlock / broken
                            # stream).  Mark it pseudostatic AND flag it as a single-frame
                            # stream so callers can distinguish this from visually-inferred
                            # pseudostatic videos.
                            frames_pseudostatic = True
                            is_single_frame = True
                    if frames_pseudostatic is not None:
                        mstats = cls.media_stats_cache.get(video_path)
                        if mstats is None:
                            mstats = MediaStats()
                            cls.media_stats_cache[video_path] = mstats
                        mstats.frames_are_pseudostatic = frames_pseudostatic
                        if is_single_frame:
                            mstats.single_frame_stream = True
                if completed:
                    cls.sampled_cache[cache_key] = accumulated

        return planned, gen()

    @classmethod
    def _iter_pyav_video_sample_paths(
        cls,
        video_path: str,
        frame_indices: List[int],
        media_hash: str,
        accumulated: List[str],
    ) -> Iterator[str]:
        assert av is not None
        targets = sorted(set(frame_indices))
        want = set(targets)
        idx = 0
        max_target = max(targets)
        scan_limit = min(
            max(max_target + 10_000, len(targets) * 200),
            800_000,
        )
        lookahead_cap = 128

        container = av.open(video_path, metadata_errors="ignore")
        try:
            if not container.streams.video:
                raise ValueError("no video stream")
            stream = container.streams.video[0]
            try:
                stream.thread_type = "AUTO"
            except Exception:
                pass
            decoder = container.decode(stream)
            while want and idx < scan_limit:
                if idx not in want:
                    try:
                        next(decoder)
                    except StopIteration:
                        break
                    idx += 1
                    continue

                chosen: Optional[np.ndarray] = None
                extra = 0
                try:
                    av_frame = next(decoder)
                except StopIteration:
                    break
                try:
                    raw = av_frame.to_ndarray(format="bgr24")
                except Exception as e:
                    logger.debug("PyAV to_ndarray failed at %s for %s: %s", idx, video_path, e)
                    want.discard(idx)
                    idx += 1
                    continue

                frame = _normalize_decoded_frame(np.asarray(raw))
                if frame is not None and not _is_likely_decoder_blank(frame):
                    chosen = frame
                else:
                    for _unused in range(lookahead_cap):
                        try:
                            av2 = next(decoder)
                        except StopIteration:
                            break
                        extra += 1
                        try:
                            raw2 = av2.to_ndarray(format="bgr24")
                        except Exception:
                            continue
                        f2 = _normalize_decoded_frame(np.asarray(raw2))
                        if f2 is not None and not _is_likely_decoder_blank(f2):
                            chosen = f2
                            break

                if chosen is not None:
                    frame_path = os.path.join(
                        cls.temporary_directory.name,
                        f"{media_hash}_sample_{idx}.jpg",
                    )
                    resolved = cls._write_cv2_jpeg(chosen, frame_path)
                    if resolved is None:
                        logger.debug(
                            "Could not write sampled frame JPEG at index %s for %s",
                            idx,
                            video_path,
                        )
                    else:
                        accumulated.append(resolved)
                        yield resolved
                else:
                    logger.debug(
                        "Degenerate PyAV sampled frame at index %s for %s (skipped)",
                        idx,
                        video_path,
                    )
                want.discard(idx)
                idx += 1 + extra
        finally:
            container.close()

        if want and len(want) / len(targets) >= _SAMPLING_WARN_MISSING_RATIO:
            logger.warning(
                "Video sampling incomplete for %s — missing %s/%s indices (decoder ended or scan cap)",
                video_path,
                len(want),
                len(targets),
            )

    @classmethod
    def _opencv_stream_video_frame_samples(
        cls,
        video_path: str,
        sample_ratio: float,
        min_sample_count: int,
        max_sample_count: int,
        max_duration_seconds: float,
        suppress_cap_log: bool,
        cache_key: str,
        detect_pseudostatic: bool = False,
    ) -> Tuple[int, Iterator[str]]:
        cap = _open_video_capture(video_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            duration_seconds = None
            if fps and fps > 0 and total_frames > 0:
                duration_seconds = total_frames / fps
            cls.media_stats_cache[video_path] = MediaStats(
                media_type=_stats_media_type_for_sampled_video_path(video_path),
                total_items=total_frames if total_frames > 0 else None,
                duration_seconds=duration_seconds,
                fps=fps if fps > 0 else None,
            )
            effective_frames = _apply_duration_cap(total_frames, fps, duration_seconds, max_duration_seconds)
            if effective_frames < total_frames and not suppress_cap_log:
                logger.debug(
                    "Capping video sampling to first %.0fs (%.0fs total) for %s",
                    max_duration_seconds, duration_seconds, video_path,
                )
            frame_indices = cls._compute_sample_indices(
                total_items=effective_frames,
                sample_ratio=sample_ratio,
                min_sample_count=min_sample_count,
                max_sample_count=max_sample_count,
            )
        except Exception as e:
            logger.warning(f"Error extracting sampled frames from {video_path}: {e}")
            try:
                cap.release()
            except Exception:
                pass

            def gen_video_fail() -> Iterator[str]:
                cls.sampled_cache[cache_key] = [video_path]
                yield video_path

            return 1, gen_video_fail()

        if len(frame_indices) == 0:
            cap.release()

            def gen_empty() -> Iterator[str]:
                cls.sampled_cache[cache_key] = [video_path]
                yield video_path

            return 1, gen_empty()

        media_hash = _stable_media_path_hash(video_path)
        planned = len(frame_indices)

        def gen() -> Iterator[str]:
            accumulated: List[str] = []
            completed = False
            first_real_frame: Optional[str] = None
            inline_check_done = False
            frames_pseudostatic: Optional[bool] = None
            is_single_frame = False
            try:
                for path in cls._iter_opencv_video_sample_paths(
                    cap,
                    video_path,
                    frame_indices,
                    media_hash,
                    accumulated,
                ):
                    yield path
                    if detect_pseudostatic and not inline_check_done and path.lower().endswith((".jpg", ".jpeg")):
                        if first_real_frame is None:
                            first_real_frame = path
                        else:
                            inline_check_done = True
                            if not _frames_are_visually_pseudostatic([first_real_frame, path]):
                                frames_pseudostatic = False
                if len(accumulated) == 0:
                    accumulated.append(video_path)
                    yield video_path
                completed = True
            finally:
                if detect_pseudostatic:
                    if frames_pseudostatic is None:
                        if inline_check_done:
                            real = [p for p in accumulated if p.lower().endswith((".jpg", ".jpeg"))]
                            if len(real) >= 2:
                                frames_pseudostatic = _frames_are_visually_pseudostatic(real)
                        elif first_real_frame is not None:
                            frames_pseudostatic = True
                            is_single_frame = True
                    if frames_pseudostatic is not None:
                        mstats = cls.media_stats_cache.get(video_path)
                        if mstats is None:
                            mstats = MediaStats()
                            cls.media_stats_cache[video_path] = mstats
                        mstats.frames_are_pseudostatic = frames_pseudostatic
                        if is_single_frame:
                            mstats.single_frame_stream = True
                cap.release()
                if completed:
                    cls.sampled_cache[cache_key] = accumulated

        return planned, gen()

    @classmethod
    def _iter_opencv_video_sample_paths(
        cls,
        cap: cv2.VideoCapture,
        video_path: str,
        frame_indices: List[int],
        media_hash: str,
        accumulated: List[str],
    ) -> Iterator[str]:
        """
        OpenCV VideoCapture sequential decode (fallback when PyAV is unavailable).
        Appends each path to *accumulated* (same list the outer generator caches).
        """
        targets = sorted(set(frame_indices))
        want = set(targets)
        idx = 0
        max_target = max(targets)
        total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scan_limit = max(
            max_target + 10_000,
            total_reported + 5000 if total_reported > 0 else max_target + 5000,
            len(targets) * 200,
        )
        scan_limit = min(scan_limit, 800_000)
        lookahead_cap = 128

        while want and idx < scan_limit:
            ret, raw = cap.read()
            if not ret or raw is None:
                break
            if idx not in want:
                idx += 1
                continue

            frame = _normalize_decoded_frame(raw)
            chosen: Optional[np.ndarray] = None
            extra = 0
            if frame is not None and not _is_likely_decoder_blank(frame):
                chosen = frame
            else:
                for _unused in range(lookahead_cap):
                    r2, raw2 = cap.read()
                    extra += 1
                    if not r2 or raw2 is None:
                        break
                    f2 = _normalize_decoded_frame(raw2)
                    if f2 is not None and not _is_likely_decoder_blank(f2):
                        chosen = f2
                        break

            if chosen is None:
                io_frame = _read_frame_via_imageio(video_path, idx)
                if io_frame is not None and not _is_likely_decoder_blank(io_frame):
                    chosen = io_frame
                    logger.debug(
                        "Using imageio fallback for %s frame index %s",
                        video_path,
                        idx,
                    )

            if chosen is not None:
                frame_path = os.path.join(
                    cls.temporary_directory.name,
                    f"{media_hash}_sample_{idx}.jpg",
                )
                resolved = cls._write_cv2_jpeg(chosen, frame_path)
                if resolved is None:
                    logger.debug(
                        "Could not write sampled frame JPEG at index %s for %s",
                        idx,
                        video_path,
                    )
                else:
                    accumulated.append(resolved)
                    yield resolved
            else:
                logger.debug(
                    "Degenerate sampled frame at index %s for %s (skipped)",
                    idx,
                    video_path,
                )
            want.discard(idx)
            idx += 1 + extra

        if want and len(want) / len(targets) >= _SAMPLING_WARN_MISSING_RATIO:
            logger.warning(
                "Video sampling incomplete for %s — missing %s/%s indices (decoder ended or scan cap)",
                video_path,
                len(want),
                len(targets),
            )

    @classmethod
    def get_dynamic_media_stats(cls, media_path: str) -> Optional[MediaStats]:
        """Return lightweight metadata useful for debug logging, or None if not yet cached."""
        return cls.media_stats_cache.get(media_path)

    @classmethod
    def slot_index_to_seek_position(
        cls,
        media_path: str,
        slot_index: int,
        total_planned_slots: int,
    ) -> Optional["SeekPosition"]:
        """Convert a slot_index from find_first_trigger_slot() into a SeekPosition.

        Requires that stream_frame_samples() has already been called for media_path
        (which find_first_trigger_slot() always does internally).

        Returns None when stats are unavailable or FPS is unknown.
        """
        stats = cls.media_stats_cache.get(media_path)
        if stats is None or total_planned_slots <= 0 or not stats.total_items:
            return None

        step = max(1, stats.total_items // total_planned_slots)
        actual_index = slot_index * step

        if stats.media_type == "pdf":
            return SeekPosition(kind="page", value=actual_index)

        fps = stats.fps or 0.0
        if fps <= 0:
            if stats.duration_seconds and stats.duration_seconds > 0:
                # Fallback: interpolate from duration
                frac = slot_index / max(1, total_planned_slots - 1)
                ms = int(frac * stats.duration_seconds * 1000)
                return SeekPosition(kind="ms", value=ms)
            return None

        return SeekPosition(kind="ms", value=int(actual_index / fps * 1000))

    @classmethod
    def is_pseudostatic_dynamic_media(cls, media_path: str) -> bool:
        """True if the media was determined to be pseudo-static after stream_frame_samples ran.

        Returns True when:
        - total_items (frames or pages) is known and <= 1, OR
        - all sampled frames were visually identical (content comparison during streaming)

        Requires stream_frame_samples to have fully consumed its iterator for media_path.
        """
        stats = cls.media_stats_cache.get(media_path)
        if stats is None:
            return False
        if stats.total_items is not None and stats.total_items <= 1:
            return True
        return bool(stats.frames_are_pseudostatic)

    @classmethod
    def is_single_frame_stream(cls, media_path: str) -> bool:
        """True only when video sampling yielded exactly one real frame.

        Distinguishes broken/buffer-deadlock streams (where the decoder dies
        after the first frame) from videos whose sampled frames merely happen
        to look visually identical.  Only set when stream_frame_samples ran
        with detect_pseudostatic=True and the single-frame path was triggered.
        """
        stats = cls.media_stats_cache.get(media_path)
        return stats is not None and stats.single_frame_stream

    @classmethod
    def _stream_pdf_sample_pages(
        cls,
        pdf_path: str,
        sample_ratio: float,
        min_sample_count: int,
        max_sample_count: int,
        cache_key: str,
    ) -> Tuple[int, Iterator[str]]:
        pdf = None
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf)
            cls.media_stats_cache[pdf_path] = MediaStats(
                media_type="pdf", total_items=total_pages
            )
            page_indices = cls._compute_sample_indices(
                total_items=total_pages,
                sample_ratio=sample_ratio,
                min_sample_count=min_sample_count,
                max_sample_count=max_sample_count,
            )
        except Exception as e:
            logger.warning(f"Error opening PDF for sampling {pdf_path}: {e}")
            if pdf is not None:
                try:
                    pdf.close()
                except Exception:
                    pass

            def gen_open_fail() -> Iterator[str]:
                cls.sampled_cache[cache_key] = [pdf_path]
                yield pdf_path

            return 1, gen_open_fail()

        if len(page_indices) == 0:
            try:
                pdf.close()
            except Exception:
                pass

            def gen_no_pages() -> Iterator[str]:
                cls.sampled_cache[cache_key] = [pdf_path]
                yield pdf_path

            return 1, gen_no_pages()

        media_hash = _stable_media_path_hash(pdf_path)
        planned = len(page_indices)
        pdf_ref = pdf

        def gen() -> Iterator[str]:
            accumulated: List[str] = []
            completed = False
            try:
                for page_index in page_indices:
                    page = pdf_ref[page_index]
                    image = page.render(scale=4, fill_color=(255, 255, 255, 255)).to_pil()
                    page_path = os.path.join(
                        cls.temporary_directory.name,
                        f"{media_hash}_sample_page_{page_index}.jpg",
                    )
                    resolved = cls._write_pil_image(image, page_path, quality=95)
                    if resolved is None:
                        logger.debug(
                            "Could not write sampled PDF page %s for %s",
                            page_index,
                            pdf_path,
                        )
                        continue
                    accumulated.append(resolved)
                    yield resolved
                if len(accumulated) == 0:
                    accumulated.append(pdf_path)
                    yield pdf_path
                completed = True
            except Exception as e:
                logger.warning(f"Error extracting sampled PDF pages from {pdf_path}: {e}")
            finally:
                try:
                    pdf_ref.close()
                except Exception:
                    pass
                if completed:
                    cls.sampled_cache[cache_key] = accumulated

        return planned, gen()

    @staticmethod
    def _compute_sample_indices(
        total_items: int,
        sample_ratio: float,
        min_sample_count: int,
        max_sample_count: int,
    ) -> List[int]:
        if total_items <= 0:
            return []
        sample_count = max(1, int(total_items * sample_ratio))
        sample_count = max(sample_count, max(1, min_sample_count))
        sample_count = min(sample_count, max(1, max_sample_count))
        sample_count = min(sample_count, total_items)
        step = max(1, total_items // sample_count)
        indices = list(range(0, total_items, step))[:sample_count]
        if len(indices) == 0:
            return [0]
        return indices

    @classmethod
    def get_cache_dir(cls) -> str:
        """Shared temp directory used for extracted frames, samples, and cover art."""
        return cls.temporary_directory.name

    @classmethod
    def get_cached_path(cls, media_path: str) -> Optional[str]:
        """
        Return the cached temp path for a media file (e.g. SVG -> PNG path), if any.
        Returns None if the media is not in the cache or is not a type that uses a temp file.
        """
        return cls.cache.get(media_path)

    @classmethod
    def get_media_path_for_cached(cls, maybe_cached_path: str) -> Optional[str]:
        """
        Resolve a cached frame/page path back to the original media path.

        Returns:
            The original media path if *maybe_cached_path* is known in the cache,
            otherwise None.
        """
        if not maybe_cached_path:
            return None
        for media_path, cached_path in cls.cache.items():
            if cached_path == maybe_cached_path:
                return media_path
        for key, sampled_paths in cls.sampled_cache.items():
            for sampled_path in sampled_paths:
                if sampled_path == maybe_cached_path:
                    media_path = key.split("|", 1)[0]
                    return media_path
        return None

    @classmethod
    def remove_from_cache(cls, media_path: str, delete_temp_file: bool = False) -> None:
        """
        Remove a media path from the cache. If it had a temp file (e.g. generated PNG for SVG),
        optionally delete that temp file from disk. Call this before moving/deleting the source
        file so the temp file is not left behind and no handles are held.
        """
        temp_path = cls.cache.pop(media_path, None)
        cls.media_stats_cache.pop(media_path, None)
        if delete_temp_file and temp_path and os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Removed cached temp file: {temp_path}")
            except OSError as e:
                logger.warning(f"Could not remove cached temp file {temp_path}: {e}")
        sampled_keys = [k for k in cls.sampled_cache.keys() if k.startswith(f"{media_path}|")]
        for key in sampled_keys:
            sampled_paths = cls.sampled_cache.pop(key, [])
            if delete_temp_file:
                for sampled_path in sampled_paths:
                    if sampled_path and os.path.isfile(sampled_path):
                        try:
                            os.remove(sampled_path)
                        except OSError as e:
                            logger.warning(f"Could not remove sampled cache file {sampled_path}: {e}")

    @classmethod
    def clear(cls) -> None:
        """Clear the frame cache."""
        cls.cache.clear()
        cls.sampled_cache.clear()
        cls.media_stats_cache.clear()

    @classmethod
    def cleanup(cls) -> None:
        """Clean up temporary files and directory."""
        cls.clear()
        cls.temporary_directory.cleanup()
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="tmp_comp_frames")


"""
Classic slideshow rules for time-based and PDF media.

Static images and “plain” dynamic items (caps set to 0) advance on the main
``slideshow_interval_seconds`` timer.

When per-type caps are non-zero (see ``utils.config``):

- **Video, N > 0:** advance after at most N seconds of playback (VLC time when
  available), or sooner if the clip ends. If VLC is unavailable, wall-clock
  time on the placeholder is used.

- **Video, N < 0** (e.g. ``-1``): dwell until **intrinsic end** of the clip
  (VLC reports end state, or playback time reaches reported duration). If VLC
  is not driving the file (placeholder / no player), the main interval timer
  still advances that slide.

- **Animated GIF / WebP, N > 0:** advance after N seconds of wall-clock time.

- **Animated GIF / WebP, N < 0:** dwell for **one full animation cycle**
  (total GIF timeline length). Static single-frame GIFs keep interval-based
  behavior.

- **PDF:** single raster page in the viewer. **N > 0:** dwell
  ``slideshow_interval_seconds * N`` (reading budget). **N <= 0:** use the
  normal slideshow interval only (no extra dwell rule).
"""

from __future__ import annotations

import time
from typing import Any

from utils.audio_media import is_audio_for_display
from utils.config import config
from utils.media_utils import get_pdf_page_count, is_video_for_display

# VLC duration/time can lag slightly behind the true end.
_SLIDESHOW_END_TOLERANCE_MS = 250


def _is_video_or_audio_for_display(path: str) -> bool:
    """Audio plays through the same VideoUI/VLC plumbing as video, so the
    video dwell rules (finite cap / intrinsic end) apply identically."""
    return is_video_for_display(path) or is_audio_for_display(path)


def _pdf_effective_page_budget(path: str, pages_cfg: int) -> int:
    """
    Pages used for dwell = interval * budget.

    * pages_cfg > 0: fixed N
    * pages_cfg < 0: intrinsic page count from disk (0 if unreadable)
    * pages_cfg == 0: no PDF dwell rule (caller should not use poll for PDF)
    """
    if pages_cfg > 0:
        return pages_cfg
    if pages_cfg < 0:
        return get_pdf_page_count(path)
    return 0


def skip_classic_slideshow_primary_tick(media_frame: Any, path: str | None) -> bool:
    """
    When True, the main slideshow interval must not advance this slide; the
    poll timer handles it (finite cap, intrinsic video end, or PDF page budget).
    """
    if not path:
        return False
    v_cap = float(config.slideshow_dynamic_video_max_seconds)
    if _is_video_or_audio_for_display(path):
        if v_cap > 0:
            return True
        if v_cap < 0 and getattr(
            media_frame, "slideshow_vlc_natural_dwell_supported", lambda: False
        )():
            return True
        return False
    if path.lower().endswith(".pdf") and config.enable_pdfs:
        pages_cfg = int(config.slideshow_dynamic_pdf_max_pages)
        return _pdf_effective_page_budget(path, pages_cfg) > 0
    g_cap = float(config.slideshow_dynamic_gif_max_seconds)
    if g_cap > 0 and getattr(
        media_frame, "is_slideshow_animated_raster_active", lambda: False
    )():
        return True
    if g_cap < 0:
        total_ms = int(
            getattr(media_frame, "slideshow_animated_total_duration_ms", lambda: 0)()
        )
        if total_ms > 0 and getattr(
            media_frame, "is_slideshow_animated_raster_active", lambda: False
        )():
            return True
    return False


def slideshow_poll_should_run(media_frame: Any, path: str | None) -> bool:
    """Whether the short-interval poll timer should be active."""
    if not path:
        return False
    v_cap = float(config.slideshow_dynamic_video_max_seconds)
    if _is_video_or_audio_for_display(path):
        if v_cap > 0:
            return True
        if v_cap < 0 and getattr(
            media_frame, "slideshow_vlc_natural_dwell_supported", lambda: False
        )():
            return True
        return False
    if path.lower().endswith(".pdf") and config.enable_pdfs:
        pages_cfg = int(config.slideshow_dynamic_pdf_max_pages)
        return _pdf_effective_page_budget(path, pages_cfg) > 0
    g_cap = float(config.slideshow_dynamic_gif_max_seconds)
    if not getattr(media_frame, "is_slideshow_animated_raster_active", lambda: False)():
        return False
    if g_cap > 0:
        return True
    if g_cap < 0:
        return int(
            getattr(media_frame, "slideshow_animated_total_duration_ms", lambda: 0)()
        ) > 0
    return False


def should_advance_slideshow_poll(
    media_frame: Any,
    path: str | None,
    started_monotonic: float | None,
) -> bool:
    """Return True if classic slideshow should move to the next file now."""
    if not path or started_monotonic is None:
        return False
    now = time.monotonic()

    cap_v = float(config.slideshow_dynamic_video_max_seconds)
    if cap_v < 0 and _is_video_or_audio_for_display(path):
        if not getattr(
            media_frame, "slideshow_vlc_natural_dwell_supported", lambda: False
        )():
            return False
        if getattr(media_frame, "slideshow_vlc_has_ended", lambda: False)():
            return True
        cur_ms, dur_ms = getattr(media_frame, "slideshow_vlc_time_ms", lambda: (0, 0))()
        if dur_ms > 0 and cur_ms >= max(0, dur_ms - _SLIDESHOW_END_TOLERANCE_MS):
            return True
        return False

    if cap_v > 0 and _is_video_or_audio_for_display(path):
        if getattr(media_frame, "slideshow_vlc_has_ended", lambda: False)():
            return True
        cur_ms, dur_ms = getattr(media_frame, "slideshow_vlc_time_ms", lambda: (0, 0))()
        cap_ms = int(cap_v * 1000)
        if dur_ms > 0:
            return cur_ms >= min(dur_ms, cap_ms)
        if cur_ms > 0:
            return cur_ms >= cap_ms
        return (now - started_monotonic) >= cap_v

    cap_g = float(config.slideshow_dynamic_gif_max_seconds)
    if cap_g > 0 and getattr(media_frame, "is_slideshow_animated_raster_active", lambda: False)():
        return (now - started_monotonic) >= cap_g

    if cap_g < 0 and getattr(media_frame, "is_slideshow_animated_raster_active", lambda: False)():
        total_ms = int(
            getattr(media_frame, "slideshow_animated_total_duration_ms", lambda: 0)()
        )
        if total_ms <= 0:
            return False
        return (now - started_monotonic) >= (total_ms / 1000.0)

    pages_cfg = int(config.slideshow_dynamic_pdf_max_pages)
    if path.lower().endswith(".pdf") and config.enable_pdfs:
        eff = _pdf_effective_page_budget(path, pages_cfg)
        if eff <= 0:
            return False
        interval = max(float(config.slideshow_interval_seconds), 0.001)
        dwell = interval * eff
        return (now - started_monotonic) >= dwell

    return False

"""PEEK-driven "essential frame" selection for video and GIF.

PEEK (https://github.com/momentslab/peek) is an optional dependency that
scores candidate frames of a video and picks the ``k`` most essential ones.
It returns indices/timestamps only -- it does not write image files -- so the
decode-and-write half lives in ``image/frame_extraction.py``, shared with the
selection strategies that need no model at all. This module is only PEEK: the
guarded import, the pipeline cache, the density caps and the selection call.

Two density caps apply before any decoding happens, both config-driven
(``utils/config.py``): ``peek_max_candidate_frames`` bounds how many candidate
frames PEEK itself samples and scores (derived into an effective ``fps``), and
``peek_max_frames_per_media`` bounds how many of PEEK's selected frames are
actually written to disk, regardless of the ``k`` a caller asked for -- itself
scaled up from ``peek_default_k`` for longer videos
(``peek_minutes_per_extra_frame``) rather than staying flat, before that same
cap is applied. A separate ``peek_max_video_duration_seconds`` threshold
refuses extraction outright on videos judged too long for reliable results (0
disables it).
"""

from __future__ import annotations

import os
from typing import List, Optional

from image.frame_extraction import (
    FrameExtraction,
    extract_frames_at_timestamps,
    frame_output_paths,
    is_frame_extraction_eligible,
    media_duration_seconds,
    resolve_target_dir,
    signatures_to_skip,
)
from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("peek_frame_selector")

PEEK_SUFFIX = "_peek_"


# ---------------------------------------------------------------------------
# PEEK pipeline (lazy import, cached load)
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {}  # (variant, device) -> (encoder, scorer, resolved_device)


def _load_peek():
    try:
        from peek.inference import load_peek_pipeline, select_frames_from_video
    except ImportError as e:
        raise RuntimeError(
            "peek is required for frame detection. "
            "Install with: pip install git+https://github.com/momentslab/peek"
        ) from e
    return load_peek_pipeline, select_frames_from_video


def _resolve_device(device: str) -> str:
    """Mirrors ImageClassifierWrapper._get_device: 'auto' picks CUDA if available."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_pipeline(device: str, variant: str = "s0"):
    resolved_device = _resolve_device(device)
    cache_key = (variant, resolved_device)
    cached = _pipeline_cache.get(cache_key)
    if cached is not None:
        return cached
    load_peek_pipeline, _select = _load_peek()
    encoder, scorer, resolved = load_peek_pipeline(variant=variant, device=resolved_device)
    result = (encoder, scorer, resolved)
    _pipeline_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

def _effective_candidate_fps(duration_seconds: Optional[float], requested_fps: float) -> float:
    """Cap requested_fps so duration_seconds * fps never exceeds the configured
    candidate ceiling. Falls back to requested_fps when duration is unknown."""
    max_candidates = max(1, int(config.peek_max_candidate_frames))
    if not duration_seconds or duration_seconds <= 0:
        return requested_fps
    max_fps = max_candidates / duration_seconds
    return min(requested_fps, max_fps)


def _scaled_default_k(duration_seconds: Optional[float]) -> int:
    """peek_default_k, plus one extra frame per peek_minutes_per_extra_frame
    of video -- longer videos get proportionally more extracted frames
    instead of a flat count, still subject to peek_max_frames_per_media.
    Only applies when the caller didn't pass an explicit k."""
    base = int(config.peek_default_k)
    per_extra = float(config.peek_minutes_per_extra_frame)
    if not duration_seconds or duration_seconds <= 0 or per_extra <= 0:
        return base
    extra = int((duration_seconds / 60.0) // per_extra)
    return base + extra


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# The name the batch and MCP callers already import.
PeekFrameExtraction = FrameExtraction


def select_peek_timestamps(
    media_path: str,
    k: Optional[int] = None,
    fps: Optional[float] = None,
    device: Optional[str] = None,
) -> List[float]:
    """The timestamps PEEK considers most essential, in seconds.

    Raises RuntimeError if peek is not installed, or if *media_path*'s duration
    exceeds peek_max_video_duration_seconds (extraction on very long media is
    treated as unreliable rather than attempted).
    """
    duration_seconds = media_duration_seconds(media_path)

    max_duration = float(config.peek_max_video_duration_seconds)
    if max_duration > 0 and duration_seconds and duration_seconds > max_duration:
        raise RuntimeError(
            _(
                "{0} is {1:.0f}s long, over the {2:.0f}s PEEK extraction limit "
                "(peek_max_video_duration_seconds) -- extraction on media this "
                "long is treated as unreliable."
            ).format(os.path.basename(media_path), duration_seconds, max_duration)
        )

    k = _scaled_default_k(duration_seconds) if k is None else k
    fps = config.peek_default_fps if fps is None else fps
    device = config.peek_device if device is None else device
    effective_k = max(1, min(int(k), int(config.peek_max_frames_per_media)))
    effective_fps = _effective_candidate_fps(duration_seconds, fps)

    _unused_loader, select_frames_from_video = _load_peek()
    encoder, scorer, resolved_device = _get_pipeline(device)

    from pathlib import Path
    output = select_frames_from_video(
        Path(media_path), encoder=encoder, scorer=scorer, device=resolved_device,
        k=effective_k, fps=effective_fps,
    )
    return list(getattr(output, "selected_timestamps_sec", []) or [])


def extract_peek_frames(
    media_path: str,
    k: Optional[int] = None,
    fps: Optional[float] = None,
    target_dir: Optional[str] = None,
    device: Optional[str] = None,
    seen=None,
) -> PeekFrameExtraction:
    """Run PEEK on *media_path* and write its selected frames as PNGs, each
    tagged with a related_image pointer back at *media_path*.

    Raises RuntimeError if peek is not installed, if *media_path* is not an
    eligible video/GIF file, or if it is over the configured duration limit.
    """
    if not is_frame_extraction_eligible(media_path):
        raise RuntimeError(_("Not a video or GIF file: {0}").format(media_path))

    timestamps = select_peek_timestamps(media_path, k=k, fps=fps, device=device)
    if not timestamps:
        return FrameExtraction(media_path=media_path, frames_written=[])

    out_dir = resolve_target_dir(media_path, target_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_paths = frame_output_paths(media_path, out_dir, len(timestamps), PEEK_SUFFIX)
    # A frame another strategy already pulled out of this file is not written
    # again; this run's own names are exempt, so a rerun still replaces them.
    # Names stay positional per timestamp, so a skipped one leaves a gap.
    # A combined run passes its shared set in, so the frames the other
    # strategies just took count too.
    if seen is None:
        seen = signatures_to_skip(media_path, out_dir, out_paths)
    writes = extract_frames_at_timestamps(media_path, timestamps, out_paths, seen)

    return FrameExtraction(
        media_path=media_path,
        frames_written=writes.written,
        duplicates_skipped=writes.duplicates,
    )

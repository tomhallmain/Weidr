"""PEEK-driven "essential frame" extraction for video and GIF.

PEEK (https://github.com/momentslab/peek) is an optional dependency that
scores candidate frames of a video and picks the ``k`` most essential ones.
It returns indices/timestamps only -- it does not write image files -- so
this module also does the actual decode-and-write step, using this
project's existing ``av``/PIL decode paths directly rather than through a
live Qt/VLC player, so it works identically for a single file, a directory
batch, and a headless session with no window open at all.

Two density caps apply before any decoding happens, both config-driven
(``utils/config.py``): ``peek_max_candidate_frames`` bounds how many
candidate frames PEEK itself samples and scores (derived into an effective
``fps``), and ``peek_max_frames_per_media`` bounds how many of PEEK's
selected frames are actually written to disk, regardless of the ``k`` a
caller asked for -- itself scaled up from ``peek_default_k`` for longer
videos (``peek_minutes_per_extra_frame``) rather than staying flat, before
that same cap is applied. A separate ``peek_max_video_duration_seconds``
threshold refuses extraction outright on videos judged too long for
reliable results (0 disables it).

Frames are written as PNG, not JPEG, specifically so each one can carry a
``related_image`` PNG text chunk pointing back at its source file --
PIL exposes PNG text chunks through ``Image.info`` on read, which is
exactly what ``image.image_data_extractor.ImageDataExtractor.get_related_image_path``
already checks (``RELATED_IMAGE_KEY = "related_image"``) to answer "what is
this image derived from" for the related-image window and downstream/related
navigation. JPEG has no equivalent PIL-readable arbitrary-text-chunk
mechanism, so it cannot carry this tag in a form that reader understands.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from utils.config import config
from utils.logging_setup import get_logger
from utils.media_utils import is_video_path_by_extension
from utils.translations import _

logger = get_logger("peek_frame_selector")

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]


def is_peek_eligible_media_path(path: str) -> bool:
    """True for an existing video or GIF file -- PEEK's own decode target.

    Not the same set as ``is_classifier_dynamic_media_path`` (video/GIF/PDF):
    PDF has no frames to decode via PEEK's video-codec-based pipeline, so it
    is never eligible here regardless of that flag.
    """
    if not path or not os.path.isfile(path):
        return False
    lower = path.lower()
    return is_video_path_by_extension(path) or lower.endswith(".gif")


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
# Candidate-density cap
# ---------------------------------------------------------------------------

def _video_duration_and_fps(video_path: str) -> Tuple[Optional[float], Optional[float]]:
    """(duration_seconds, fps), either of which may be None if unavailable.

    A best-effort probe: an unreadable/corrupt file yields (None, None)
    rather than raising, since a failed density-cap estimate should not by
    itself stop the caller from reaching the actual (and more informative)
    decode failure further down.
    """
    if av is None:
        return None, None
    try:
        container = av.open(video_path, metadata_errors="ignore")
    except Exception:
        return None, None
    try:
        if not container.streams.video:
            return None, None
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        duration_s = float(stream.duration * stream.time_base) if stream.duration is not None else None
        return duration_s, fps
    except Exception:
        return None, None
    finally:
        container.close()


def _gif_frame_timeline_ms(gif_path: str) -> List[int]:
    """Cumulative start time (ms) of each frame, from each frame's own duration.

    A GIF frame missing/zero duration defaults to 100ms, matching PIL's own
    convention for how such frames are commonly displayed.
    """
    from PIL import Image

    starts: List[int] = []
    elapsed = 0
    with Image.open(gif_path) as im:
        n_frames = int(getattr(im, "n_frames", 1) or 1)
        for i in range(n_frames):
            im.seek(i)
            starts.append(elapsed)
            elapsed += int(im.info.get("duration") or 100)
    return starts


def _effective_candidate_fps(duration_seconds: Optional[float], requested_fps: float) -> float:
    """Cap requested_fps so duration_seconds * fps never exceeds the configured
    candidate ceiling. Falls back to requested_fps when duration is unknown."""
    max_candidates = max(1, int(config.peek_max_candidate_frames))
    if not duration_seconds or duration_seconds <= 0:
        return requested_fps
    max_fps = max_candidates / duration_seconds
    return min(requested_fps, max_fps)


# ---------------------------------------------------------------------------
# Frame-at-timestamp decode and write (Qt-free, no live player involved)
# ---------------------------------------------------------------------------

def _write_png_with_related_image(pil_image, out_path: str, source_path: str) -> bool:
    """Write *pil_image* as PNG, tagging it with a related_image text chunk
    pointing at *source_path* -- the same key
    image.image_data_extractor.ImageDataExtractor.RELATED_IMAGE_KEY reads."""
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("related_image", source_path)
    try:
        pil_image.save(out_path, "PNG", pnginfo=info)
    except Exception:
        logger.exception("Failed to write PNG frame to %s", out_path)
        return False
    return os.path.isfile(out_path)


def _extract_video_frames_at_timestamps(
    video_path: str, timestamps_sec: List[float], out_paths: List[str],
) -> List[str]:
    """Seek to each timestamp and write the first decoded frame at/after it.

    One av.open() call, re-seeking within it per timestamp -- simpler and
    more robust than trying to serve multiple targets off a single forward
    decode pass, and k is always small (capped by peek_max_frames_per_media),
    so repeated seeks are cheap relative to one full linear decode of a long
    video would be.
    """
    if av is None:
        raise RuntimeError(_("av is required to decode video frames but is not installed."))
    from PIL import Image

    written: List[str] = []
    container = av.open(video_path, metadata_errors="ignore")
    try:
        if not container.streams.video:
            return written
        stream = container.streams.video[0]
        for seconds, out_path in zip(timestamps_sec, out_paths):
            try:
                container.seek(int(max(0.0, seconds) * 1_000_000))
                chosen = None
                for frame in container.decode(stream):
                    if frame.time is None or frame.time >= seconds:
                        chosen = frame
                        break
                if chosen is None:
                    continue
                pil_frame = Image.fromarray(chosen.to_ndarray(format="rgb24"))
                if _write_png_with_related_image(pil_frame, out_path, video_path):
                    written.append(out_path)
            except Exception:
                logger.exception("Failed to extract video frame at %.3fs from %s", seconds, video_path)
    finally:
        container.close()
    return written


def _extract_gif_frames_at_timestamps(
    gif_path: str, timestamps_sec: List[float], out_paths: List[str],
) -> List[str]:
    from PIL import Image

    written: List[str] = []
    timeline = _gif_frame_timeline_ms(gif_path)
    if not timeline:
        return written
    with Image.open(gif_path) as im:
        for seconds, out_path in zip(timestamps_sec, out_paths):
            try:
                target_ms = max(0.0, seconds) * 1000.0
                frame_idx = 0
                for i, start_ms in enumerate(timeline):
                    if start_ms <= target_ms:
                        frame_idx = i
                    else:
                        break
                im.seek(frame_idx)
                if _write_png_with_related_image(im.convert("RGB"), out_path, gif_path):
                    written.append(out_path)
            except Exception:
                logger.exception("Failed to extract GIF frame at %.3fs from %s", seconds, gif_path)
    return written


def _resolve_target_dir(media_path: str, target_dir: Optional[str]) -> str:
    if target_dir:
        return target_dir
    if config.peek_output_directory and not config.peek_save_to_same_dir:
        return config.peek_output_directory
    return os.path.dirname(media_path) or "."


def _frame_output_paths(media_path: str, out_dir: str, count: int) -> List[str]:
    stem = os.path.splitext(os.path.basename(media_path))[0]
    return [os.path.join(out_dir, f"{stem}_peek_{i}.png") for i in range(count)]


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

@dataclass
class PeekFrameExtraction:
    media_path: str
    frames_written: List[str]


def extract_peek_frames(
    media_path: str,
    k: Optional[int] = None,
    fps: Optional[float] = None,
    target_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> PeekFrameExtraction:
    """Run PEEK on *media_path* and write its selected frames as PNGs, each
    tagged with a related_image pointer back at *media_path*.

    Raises RuntimeError if peek is not installed, if *media_path* is not an
    eligible video/GIF file, or if its duration exceeds
    peek_max_video_duration_seconds (extraction on very long media is
    treated as unreliable rather than attempted).
    """
    if not is_peek_eligible_media_path(media_path):
        raise RuntimeError(_("Not a video or GIF file: {0}").format(media_path))

    is_gif = media_path.lower().endswith(".gif")
    if is_gif:
        timeline = _gif_frame_timeline_ms(media_path)
        duration_seconds = (timeline[-1] / 1000.0) if timeline else None
    else:
        duration_seconds, _fps_unused = _video_duration_and_fps(media_path)

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
    timestamps = list(getattr(output, "selected_timestamps_sec", []) or [])
    if not timestamps:
        return PeekFrameExtraction(media_path=media_path, frames_written=[])

    out_dir = _resolve_target_dir(media_path, target_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_paths = _frame_output_paths(media_path, out_dir, len(timestamps))

    if is_gif:
        written = _extract_gif_frames_at_timestamps(media_path, timestamps, out_paths)
    else:
        written = _extract_video_frames_at_timestamps(media_path, timestamps, out_paths)

    return PeekFrameExtraction(media_path=media_path, frames_written=written)

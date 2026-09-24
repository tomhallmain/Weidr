"""Writing frames out of a video or GIF, at timestamps somebody else chose.

Everything here is about the mechanics shared by every way of choosing which
frames to extract: what counts as an extractable file, how its duration and
frame timeline are read, how a timestamp becomes a decoded frame on disk, and
where that file goes.

Choosing the timestamps is a separate job. ``image/peek_frame_selector.py``
does it with the optional ``peek`` model; first/last/trigger selection needs no
optional dependency at all. This module must therefore never import
``peek_frame_selector`` at module level, or the cheap strategies would stop
working wherever ``peek`` is not installed.

Frames are written as PNG, not JPEG, specifically so each one can carry a
``related_image`` PNG text chunk pointing back at its source file -- PIL
exposes PNG text chunks through ``Image.info`` on read, which is exactly what
``image.image_data_extractor.ImageDataExtractor.get_related_image_path``
already checks (``RELATED_IMAGE_KEY = "related_image"``) to answer "what is
this image derived from" for the related-image window and downstream/related
navigation. JPEG has no equivalent PIL-readable arbitrary-text-chunk
mechanism, so it cannot carry this tag in a form that reader understands.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

from utils.config import config
from utils.logging_setup import get_logger
from utils.media_utils import is_video_path_by_extension
from utils.translations import _

logger = get_logger("frame_extraction")

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]


def is_frame_extraction_eligible(path: str) -> bool:
    """True for an existing video or GIF file -- something with real frames.

    Not the same set as ``is_classifier_dynamic_media_path`` (video/GIF/PDF/ePub):
    a PDF or ePub is paginated rather than framed and has no codec to decode, so
    it is never eligible here regardless of that flag.
    """
    if not path or not os.path.isfile(path):
        return False
    lower = path.lower()
    return is_video_path_by_extension(path) or lower.endswith(".gif")


def is_gif_path(path: str) -> bool:
    return bool(path) and path.lower().endswith(".gif")


# ---------------------------------------------------------------------------
# Reading a file's timeline
# ---------------------------------------------------------------------------

def video_duration_and_fps(video_path: str) -> Tuple[Optional[float], Optional[float]]:
    """(duration_seconds, fps), either of which may be None if unavailable.

    A best-effort probe: an unreadable/corrupt file yields (None, None) rather
    than raising, since a failed estimate should not by itself stop the caller
    from reaching the actual (and more informative) decode failure further
    down.
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


def gif_frame_timeline_ms(gif_path: str) -> List[int]:
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


def media_duration_seconds(media_path: str) -> Optional[float]:
    """Duration of a video or GIF, or None when it cannot be read."""
    if is_gif_path(media_path):
        timeline = gif_frame_timeline_ms(media_path)
        return (timeline[-1] / 1000.0) if timeline else None
    duration_seconds, _fps_unused = video_duration_and_fps(media_path)
    return duration_seconds


# ---------------------------------------------------------------------------
# Decode and write (Qt-free, no live player involved)
# ---------------------------------------------------------------------------

def write_png_with_related_image(pil_image, out_path: str, source_path: str) -> bool:
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


@dataclass
class FrameWrites:
    """What one write pass produced."""
    written: List[str] = field(default_factory=list)
    duplicates: int = 0


def frame_signature(pil_image) -> str:
    """A content hash of one decoded frame.

    PNG is lossless and every decode of the same frame yields the same raster,
    so this is stable across runs and across strategies -- two strategies that
    land on one frame produce one signature.
    """
    return hashlib.sha1(pil_image.convert("RGB").tobytes()).hexdigest()


def signatures_to_skip(
    media_path: str, out_dir: str, ignore_paths=(),
) -> Optional[Set[str]]:
    """Frames already extracted from *media_path* into *out_dir*.

    Returns None when duplicate-skipping is switched off, which every writer
    reads as "write everything".

    *ignore_paths* are the files this call is about to write: a strategy rerun
    replaces its own output rather than treating it as an existing frame and
    skipping everything. Only PNGs whose ``related_image`` tag names this same
    source count, so frames extracted from a different file never suppress
    one of this file's.
    """
    if not config.frame_extraction_skip_duplicate_frames:
        return None

    from PIL import Image

    signatures: Set[str] = set()
    source_key = os.path.normcase(os.path.abspath(media_path))
    ignore = {os.path.normcase(os.path.abspath(p)) for p in ignore_paths}
    stem = os.path.splitext(os.path.basename(media_path))[0]
    try:
        entries = os.listdir(out_dir)
    except OSError:
        return signatures
    for name in entries:
        if not name.lower().endswith(".png") or not name.startswith(stem):
            continue
        path = os.path.join(out_dir, name)
        if os.path.normcase(os.path.abspath(path)) in ignore:
            continue
        try:
            with Image.open(path) as existing:
                tagged = existing.info.get("related_image")
                if not tagged or os.path.normcase(os.path.abspath(tagged)) != source_key:
                    continue
                signatures.add(frame_signature(existing))
        except Exception:
            logger.debug("Could not read an existing frame at %s", path, exc_info=True)
    return signatures


def _is_duplicate(pil_image, seen: Optional[Set[str]]) -> bool:
    """True when this exact frame was already extracted; records it if not."""
    if seen is None:
        return False
    signature = frame_signature(pil_image)
    if signature in seen:
        return True
    seen.add(signature)
    return False


def extract_video_frames_at_timestamps(
    video_path: str, timestamps_sec: List[float], out_paths: List[str],
    seen: Optional[Set[str]] = None,
) -> FrameWrites:
    """Seek to each timestamp and write the first decoded frame at/after it.

    One av.open() call, re-seeking within it per timestamp -- simpler and more
    robust than trying to serve multiple targets off a single forward decode
    pass, and the number of frames asked for is always small, so repeated seeks
    are cheap relative to one full linear decode of a long video.
    """
    if av is None:
        raise RuntimeError(_("av is required to decode video frames but is not installed."))
    from PIL import Image

    result = FrameWrites()
    container = av.open(video_path, metadata_errors="ignore")
    try:
        if not container.streams.video:
            return result
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
                if _is_duplicate(pil_frame, seen):
                    result.duplicates += 1
                    continue
                if write_png_with_related_image(pil_frame, out_path, video_path):
                    result.written.append(out_path)
            except Exception:
                logger.exception("Failed to extract video frame at %.3fs from %s", seconds, video_path)
    finally:
        container.close()
    return result


def extract_gif_frames_at_timestamps(
    gif_path: str, timestamps_sec: List[float], out_paths: List[str],
    seen: Optional[Set[str]] = None,
) -> FrameWrites:
    from PIL import Image

    result = FrameWrites()
    timeline = gif_frame_timeline_ms(gif_path)
    if not timeline:
        return result
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
                frame = im.convert("RGB")
                if _is_duplicate(frame, seen):
                    result.duplicates += 1
                    continue
                if write_png_with_related_image(frame, out_path, gif_path):
                    result.written.append(out_path)
            except Exception:
                logger.exception("Failed to extract GIF frame at %.3fs from %s", seconds, gif_path)
    return result


def extract_frames_at_timestamps(
    media_path: str, timestamps_sec: List[float], out_paths: List[str],
    seen: Optional[Set[str]] = None,
) -> FrameWrites:
    """Write *media_path*'s frames at *timestamps_sec*, video or GIF alike."""
    if is_gif_path(media_path):
        return extract_gif_frames_at_timestamps(media_path, timestamps_sec, out_paths, seen)
    return extract_video_frames_at_timestamps(media_path, timestamps_sec, out_paths, seen)


# ---------------------------------------------------------------------------
# Where the output goes
# ---------------------------------------------------------------------------

def resolve_target_dir(media_path: str, target_dir: Optional[str]) -> str:
    if target_dir:
        return target_dir
    if config.peek_output_directory and not config.peek_save_to_same_dir:
        return config.peek_output_directory
    return os.path.dirname(media_path) or "."


def frame_output_paths(media_path: str, out_dir: str, count: int, suffix: str) -> List[str]:
    """``<stem><suffix><i>.png`` for *count* frames, numbered from 0."""
    stem = os.path.splitext(os.path.basename(media_path))[0]
    return [os.path.join(out_dir, f"{stem}{suffix}{i}.png") for i in range(count)]


def single_frame_output_path(media_path: str, out_dir: str, suffix: str) -> str:
    """``<stem><suffix>.png`` -- for strategies that produce one frame."""
    stem = os.path.splitext(os.path.basename(media_path))[0]
    return os.path.join(out_dir, f"{stem}{suffix}.png")


# ---------------------------------------------------------------------------
# Strategies: which frames to take
# ---------------------------------------------------------------------------

STRATEGY_PEEK = "peek"
STRATEGY_FIRST = "first"
STRATEGY_LAST = "last"
STRATEGY_TRIGGER = "trigger"
STRATEGIES = (STRATEGY_PEEK, STRATEGY_FIRST, STRATEGY_LAST, STRATEGY_TRIGGER)

FIRST_SUFFIX = "_first"
LAST_SUFFIX = "_last"
TRIGGER_SUFFIX = "_trigger_"


@dataclass
class FrameExtraction:
    media_path: str
    frames_written: List[str]
    # Frames this run decoded but did not write, because the same frame had
    # already been extracted from this source -- by an earlier run or by
    # another strategy.
    duplicates_skipped: int = 0
    # Strategies that could not run, as readable reasons. A combined run
    # reports them and carries on with the rest rather than failing whole.
    errors: List[str] = field(default_factory=list)


def extract_frames(
    media_path: str,
    strategy: str = STRATEGY_PEEK,
    *,
    k: Optional[int] = None,
    fps: Optional[float] = None,
    target_dir: Optional[str] = None,
    action_name: Optional[str] = None,
    kind: str = "classifier_action",
    start_slot: int = 0,
    sample_ratio: Optional[float] = None,
    seen: Optional[Set[str]] = None,
) -> FrameExtraction:
    """Write frames of *media_path* chosen by *strategy*.

    Every strategy produces the same kind of output: PNGs tagged with a
    related_image pointer back at the source. They differ only in which
    timestamps they pick -- ``peek`` scores candidates with the optional peek
    model, ``first`` and ``last`` take the ends, and ``trigger`` takes the
    frame a named classifier action or prevalidation fires on.

    *k*/*fps* apply to ``peek``; *action_name*/*kind*/*start_slot*/
    *sample_ratio* apply to ``trigger``. Raises RuntimeError for an unknown
    strategy, an ineligible file, or a strategy that cannot run (peek not
    installed, an unreadable duration, an unknown action). A scan that simply
    finds nothing is not an error: it writes no frames.

    *seen* is the duplicate-skip set. Given one, this call uses it instead of
    reading the directory -- that is how extract_frames_all shares a single
    set across strategies. Left None, each strategy builds its own from the
    frames already on disk.
    """
    if strategy not in STRATEGIES:
        raise RuntimeError(
            _("Unknown frame extraction strategy: {0}").format(strategy)
        )
    if not is_frame_extraction_eligible(media_path):
        raise RuntimeError(_("Not a video or GIF file: {0}").format(media_path))

    if strategy == STRATEGY_PEEK:
        # Lazy: peek is optional, and the other strategies must work without it.
        from image.peek_frame_selector import extract_peek_frames

        return extract_peek_frames(
            media_path, k=k, fps=fps, target_dir=target_dir, seen=seen,
        )

    out_dir = resolve_target_dir(media_path, target_dir)
    os.makedirs(out_dir, exist_ok=True)

    if strategy == STRATEGY_FIRST:
        out_path = single_frame_output_path(media_path, out_dir, FIRST_SUFFIX)
        writes = extract_frames_at_timestamps(
            media_path, [0.0], [out_path],
            seen if seen is not None else signatures_to_skip(media_path, out_dir, [out_path]),
        )
    elif strategy == STRATEGY_LAST:
        writes = _extract_last_frame(media_path, out_dir, seen)
    else:
        writes = _extract_trigger_frame(
            media_path, out_dir, action_name, kind, start_slot, sample_ratio, seen,
        )
    return FrameExtraction(
        media_path=media_path,
        frames_written=writes.written,
        duplicates_skipped=writes.duplicates,
    )


def enabled_strategies(action_name: Optional[str] = None) -> List[str]:
    """The strategies one extraction run uses, in the order they run.

    The deterministic ones come first so that PEEK, which picks several
    frames, is the one that yields a frame another strategy already took
    rather than the other way round.

    ``trigger`` needs an action to scan for: without one configured (and none
    passed in) it is left out rather than asking, so a run never stops to be
    configured.
    """
    ordered: List[str] = []
    if config.frame_extraction_use_first_frame:
        ordered.append(STRATEGY_FIRST)
    if config.frame_extraction_use_last_frame:
        ordered.append(STRATEGY_LAST)
    if config.frame_extraction_use_trigger and (
        action_name or config.frame_extraction_trigger_action
    ):
        ordered.append(STRATEGY_TRIGGER)
    # PEEK joins when it is switched on; whether the package is installed is
    # found out by running it, and a failure there does not stop the rest.
    if config.enable_peek_frame_detection:
        ordered.append(STRATEGY_PEEK)
    return ordered


def extract_frames_all(
    media_path: str,
    *,
    strategies: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
    fps: Optional[float] = None,
    target_dir: Optional[str] = None,
    action_name: Optional[str] = None,
    kind: Optional[str] = None,
    start_slot: int = 0,
    sample_ratio: Optional[float] = None,
) -> FrameExtraction:
    """Run every enabled strategy over one file, in one pass.

    This is what "extract frames" means to a user: one action, covering every
    way of choosing a frame that is switched on, with nothing to pick per
    file or per directory. The strategies share one duplicate-skip set, so a
    frame two of them both land on is written once.

    A strategy that cannot run -- peek not installed, a duration that cannot
    be read, an action that is not registered -- is recorded in ``errors`` and
    skipped. The others still produce their frames.
    """
    if not is_frame_extraction_eligible(media_path):
        raise RuntimeError(_("Not a video or GIF file: {0}").format(media_path))

    chosen = list(strategies) if strategies is not None else enabled_strategies(action_name)
    combined = FrameExtraction(media_path=media_path, frames_written=[])
    if not chosen:
        return combined

    # One set for the whole run, starting empty: these strategies dedupe
    # against each other, and a rerun replaces the output of all of them, so
    # there is nothing on disk this run should defer to.
    seen: Optional[Set[str]] = set() if config.frame_extraction_skip_duplicate_frames else None

    for strategy in chosen:
        try:
            outcome = extract_frames(
                media_path, strategy, k=k, fps=fps, target_dir=target_dir,
                action_name=action_name,
                kind=kind or config.frame_extraction_trigger_kind,
                start_slot=start_slot, sample_ratio=sample_ratio, seen=seen,
            )
        except Exception as e:
            logger.warning("%s extraction failed for %s: %s", strategy, media_path, e)
            combined.errors.append(f"{strategy}: {e}")
            continue
        combined.frames_written.extend(outcome.frames_written)
        combined.duplicates_skipped += outcome.duplicates_skipped
    return combined


def _extract_last_frame(
    media_path: str, out_dir: str, seen: Optional[Set[str]] = None,
) -> FrameWrites:
    out_path = single_frame_output_path(media_path, out_dir, LAST_SUFFIX)
    if seen is None:
        seen = signatures_to_skip(media_path, out_dir, [out_path])
    if is_gif_path(media_path):
        timeline = gif_frame_timeline_ms(media_path)
        if not timeline:
            return FrameWrites()
        # The mapping in extract_gif_frames_at_timestamps takes the last frame
        # whose start is at or before the target, so the final start hits it.
        return extract_gif_frames_at_timestamps(
            media_path, [timeline[-1] / 1000.0], [out_path], seen
        )
    return _extract_last_video_frame(media_path, out_path, seen)


def _extract_last_video_frame(
    video_path: str, out_path: str, seen: Optional[Set[str]] = None,
) -> FrameWrites:
    """Write the final decodable frame.

    Seeking to the duration and taking the first frame at or after it finds
    nothing -- there is no such frame -- so this seeks back by
    ``frame_extraction_last_frame_lookback_seconds`` and keeps the last frame
    decoded from there.
    """
    duration, _fps_unused = video_duration_and_fps(video_path)
    if not duration or duration <= 0:
        raise RuntimeError(
            _(
                "{0}'s duration could not be read, so its last frame cannot be "
                "located without decoding the whole file."
            ).format(os.path.basename(video_path))
        )
    if av is None:
        raise RuntimeError(_("av is required to decode video frames but is not installed."))
    from PIL import Image

    lookback = max(0.0, float(config.frame_extraction_last_frame_lookback_seconds))
    container = av.open(video_path, metadata_errors="ignore")
    try:
        if not container.streams.video:
            return FrameWrites()
        stream = container.streams.video[0]
        start = max(0.0, duration - lookback)
        chosen = _decode_last_frame_from(container, stream, start)
        if chosen is None and start > 0.0:
            # The lookback landed past the last decodable frame -- a duration
            # that overstates the stream, or one that ends early. Correct but
            # slow, and only reached for short or damaged files.
            chosen = _decode_last_frame_from(container, stream, 0.0)
        if chosen is None:
            return FrameWrites()
        pil_frame = Image.fromarray(chosen.to_ndarray(format="rgb24"))
        if _is_duplicate(pil_frame, seen):
            return FrameWrites(duplicates=1)
        if write_png_with_related_image(pil_frame, out_path, video_path):
            return FrameWrites(written=[out_path])
        return FrameWrites()
    finally:
        container.close()


def _decode_last_frame_from(container, stream, start_seconds: float):
    try:
        container.seek(int(start_seconds * 1_000_000))
        last = None
        for frame in container.decode(stream):
            last = frame
        return last
    except Exception:
        logger.exception("Failed to decode the last frame from %.3fs", start_seconds)
        return None


def _extract_trigger_frame(
    media_path: str,
    out_dir: str,
    action_name: Optional[str],
    kind: str,
    start_slot: int,
    sample_ratio: Optional[float],
    seen: Optional[Set[str]] = None,
) -> FrameWrites:
    """Write the frame a classifier action or prevalidation triggers on.

    The scan and the slot-to-position mapping are compare/trigger_scan.py's,
    the same ones the Seek to Trigger tab and the find_trigger tool use.
    """
    action_name = action_name or config.frame_extraction_trigger_action
    if not action_name:
        raise RuntimeError(
            _(
                "No classifier action or prevalidation to scan for. Name one in "
                "frame_extraction_trigger_action, or pass one to this call."
            )
        )
    from compare.trigger_scan import find_trigger

    try:
        report = find_trigger(
            action_name, kind, media_path,
            start_slot=start_slot, sample_ratio=sample_ratio,
        )
    except ValueError as e:
        # Caller error (unknown action, wrong media type, an action that
        # cannot run), as distinct from a scan that found nothing.
        raise RuntimeError(str(e))

    if not report.get("matched"):
        return FrameWrites()

    position = report.get("position")
    if not position or position.get("kind") != "ms":
        raise RuntimeError(
            _(
                "{0} triggered on {1}, but the position of that frame in the "
                "file could not be resolved."
            ).format(action_name, os.path.basename(media_path))
        )
    seconds = float(position["value"]) / 1000.0
    out_path = single_frame_output_path(
        media_path, out_dir, f"{TRIGGER_SUFFIX}{report.get('slot_index', 0)}"
    )
    if seen is None:
        seen = signatures_to_skip(media_path, out_dir, [out_path])
    return extract_frames_at_timestamps(media_path, [seconds], [out_path], seen)

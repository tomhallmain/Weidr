"""
Frame-sampled matching of dynamic media (video, GIF, PDF, ePub), shared by
classifier actions/prevalidations, the trigger-frame seek, compare filters and
classifier pipeline conditions so one file is judged the same way everywhere.

Frames come from FrameCache.stream_frame_samples(). The media matches once at
least ceil(planned_slots * positive_ratio) sampled frames match; sampling
stops early on success, or once the remaining slots can no longer reach that
count. match_media() adds the still-image path, so callers don't branch on the
media type themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

from utils.logging_setup import get_logger

logger = get_logger("dynamic_media_sampling")

# frame path -> (is_match, matched category or None)
FrameMatcher = Callable[[str], Tuple[bool, Optional[str]]]


DEFAULT_RATIO = 0.1


def normalize_ratio(value, default: float = DEFAULT_RATIO) -> float:
    """*value* as a float clamped to [0, 1]; *default* when it isn't a number."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass(frozen=True)
class FrameSampling:
    """How much of a dynamic media file to sample, and how many samples must match."""
    sample_ratio: float = DEFAULT_RATIO
    positive_ratio: float = DEFAULT_RATIO


@dataclass
class DynamicSampleResult:
    planned_slots: int
    required_positive_count: int
    positive_count: int = 0
    processed_samples: int = 0
    last_processed_index: int = -1
    threshold_met: bool = False
    reached_last_sample: bool = False
    # Category of the last matching frame that reported one.
    matched_category: Optional[str] = None
    # Slot index and path of the first matching frame.
    first_match_index: Optional[int] = None
    first_match_path: Optional[str] = None


@dataclass
class MediaMatchResult:
    matched: bool
    matched_category: Optional[str] = None
    # The frame sampling behind the result; None when the file was judged as one image.
    sampling: Optional[DynamicSampleResult] = None


def count_sample_matches(
    frames: Iterable[str],
    sample_count: int,
    required_positive_count: int,
    frame_matcher: FrameMatcher,
    slot_offset: int = 0,
) -> DynamicSampleResult:
    """Run *frame_matcher* over *frames* until *required_positive_count* match, or
    until the frames left of *sample_count* can no longer reach that count.

    Slot indices in the result are offset by *slot_offset* (for a scan that
    starts part-way through the samples). A frame whose matcher raises counts as
    processed but not matching. *frames* is closed afterwards if it can be (a
    generator holding decoded media open).
    """
    result = DynamicSampleResult(
        planned_slots=sample_count,
        required_positive_count=required_positive_count,
    )
    try:
        for idx, sampled_path in enumerate(frames):
            try:
                result.processed_samples += 1
                result.last_processed_index = slot_offset + idx
                is_match, matched_category = frame_matcher(sampled_path)
                if is_match:
                    result.positive_count += 1
                    if result.first_match_index is None:
                        result.first_match_index = slot_offset + idx
                        result.first_match_path = sampled_path
                    if matched_category:
                        result.matched_category = matched_category
                    if result.positive_count >= result.required_positive_count:
                        result.threshold_met = True
                        break
                remaining_samples = sample_count - (idx + 1)
                if result.positive_count + remaining_samples < result.required_positive_count:
                    break
            except Exception as e:
                logger.debug(f"Sample frame evaluation failed for {sampled_path}: {e}")
        else:
            # Consumed every yielded sample (may be fewer than sample_count).
            result.reached_last_sample = True
    finally:
        close = getattr(frames, "close", None)
        if callable(close):
            close()
    return result


def evaluate_dynamic_media(
    media_path: str,
    frame_matcher: FrameMatcher,
    sample_ratio: float,
    positive_ratio: float,
    detect_pseudostatic: bool = False,
) -> Optional[DynamicSampleResult]:
    """Sample *media_path* and count frames *frame_matcher* accepts.

    Returns None when FrameCache plans no samples, so the caller can fall
    back to treating the file as a still. A frame whose matcher raises counts
    as processed but not matching.
    """
    from image.frame_cache import FrameCache
    planned_slots, sample_iter = FrameCache.stream_frame_samples(
        media_path,
        sample_ratio=sample_ratio,
        detect_pseudostatic=detect_pseudostatic,
    )
    if planned_slots <= 0:
        return None
    return count_sample_matches(
        sample_iter, planned_slots, math.ceil(planned_slots * positive_ratio), frame_matcher
    )


def match_media(
    media_path: str,
    frame_matcher: FrameMatcher,
    sampling: FrameSampling,
    detect_pseudostatic: bool = False,
) -> MediaMatchResult:
    """Match *media_path*: dynamic media (video, GIF, PDF, ePub with their types
    enabled) by frame sampling; anything else, or dynamic media with no samples
    planned, as the one image FrameCache resolves for it (first frame or page,
    SVG/HTML render, or the file itself).

    Exceptions from *frame_matcher* on that single image propagate; on sampled
    frames they count as non-matching frames.
    """
    # Looked up per call so a patched utils.media_utils function applies.
    from utils.media_utils import is_classifier_dynamic_media_path
    if is_classifier_dynamic_media_path(media_path):
        result = evaluate_dynamic_media(
            media_path, frame_matcher, sampling.sample_ratio, sampling.positive_ratio,
            detect_pseudostatic=detect_pseudostatic,
        )
        if result is not None:
            return MediaMatchResult(result.threshold_met, result.matched_category, result)
    from image.frame_cache import FrameCache
    is_match, matched_category = frame_matcher(FrameCache.get_image_path(media_path))
    return MediaMatchResult(bool(is_match), matched_category)

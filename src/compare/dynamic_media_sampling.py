"""
Frame-sampled matching of dynamic media (video, GIF, PDF, ePub), shared by
classifier actions/prevalidations and compare filters so one file is judged
the same way everywhere.

Frames come from FrameCache.stream_frame_samples(). The media matches once at
least ceil(planned_slots * positive_ratio) sampled frames match; sampling
stops early on success, or once the remaining slots can no longer reach that
count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from image.frame_cache import FrameCache
from utils.logging_setup import get_logger

logger = get_logger("dynamic_media_sampling")

# frame path -> (is_match, matched category or None)
FrameMatcher = Callable[[str], Tuple[bool, Optional[str]]]


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
    planned_slots, sample_iter = FrameCache.stream_frame_samples(
        media_path,
        sample_ratio=sample_ratio,
        detect_pseudostatic=detect_pseudostatic,
    )
    if planned_slots <= 0:
        return None

    result = DynamicSampleResult(
        planned_slots=planned_slots,
        required_positive_count=math.ceil(planned_slots * positive_ratio),
    )
    try:
        for idx, sampled_path in enumerate(sample_iter):
            try:
                result.processed_samples += 1
                result.last_processed_index = idx
                is_match, matched_category = frame_matcher(sampled_path)
                if is_match:
                    result.positive_count += 1
                    if matched_category:
                        result.matched_category = matched_category
                    if result.positive_count >= result.required_positive_count:
                        result.threshold_met = True
                        break
                remaining_samples = planned_slots - (idx + 1)
                if result.positive_count + remaining_samples < result.required_positive_count:
                    break
            except Exception as e:
                logger.debug(f"Sample frame evaluation failed for {sampled_path}: {e}")
        else:
            # Consumed every yielded sample (may be fewer than planned_slots).
            result.reached_last_sample = True
    finally:
        close = getattr(sample_iter, "close", None)
        if callable(close):
            close()
    return result

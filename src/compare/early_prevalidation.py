"""Early prevalidation: during a slideshow, prevalidate the upcoming slide while
the current one is still on screen, so advancing does not wait on it.

Only dynamic media (video, GIF, paged documents: frame sampling) with an
applicable prevalidation is evaluated. Static images are left to the normal
prevalidation on arrival, model checks included. The walk ends at the first
other candidate or the first candidate that would be shown. Results reach the
normal prevalidation on arrival through its own caches and the file system (a
moved or deleted file is gone); ``prevalidate_media(early=True)`` leaves COPY,
NOTIFY and prevalidation pipelines to that arrival pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from compare.classifier_actions_manager import ClassifierActionsManager, prevalidation_action_skips
from utils.audio_media import is_audio_for_display
from utils.constants import ClassifierActionType
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("early_prevalidation")


class StopReason:
    ABORTED = "aborted"
    CHEAP = "cheap"
    SHOWN = "shown"
    LIMIT = "limit"
    ERROR = "error"


@dataclass
class EarlyRunResult:
    evaluated: List[str] = field(default_factory=list)
    stop_reason: str = StopReason.LIMIT
    stopped_at: Optional[str] = None


def is_costly_candidate(media_path: str, base_dir: Optional[str]) -> bool:
    """Whether to prevalidate *media_path* ahead of time while browsing
    *base_dir*: dynamic media with an applicable prevalidation. Reads only
    the path and the rule configuration, never the file."""
    if is_audio_for_display(media_path):
        return False
    if base_dir in ClassifierActionsManager.directories_to_exclude:
        return False
    if media_path in ClassifierActionsManager.user_prevalidation_overrides:
        return False
    if not ClassifierActionsManager.is_dynamic_prevalidation_media(media_path):
        return False
    return bool(ClassifierActionsManager.applicable_prevalidations(media_path, base_dir))


def upcoming_candidates(files: Sequence[str], current_index: int, max_candidates: int) -> List[str]:
    """Files after *current_index* in slideshow order, wrapping around but
    never back to the current file, at most *max_candidates* of them."""
    count = len(files)
    if count < 2 or max_candidates <= 0 or not 0 <= current_index < count:
        return []
    limit = min(max_candidates, count - 1)
    return [files[(current_index + step) % count] for step in range(1, limit + 1)]


def run_early_prevalidation(
    candidates: Sequence[str],
    *,
    prevalidate: Callable[[str], Optional[ClassifierActionType]],
    is_costly: Callable[[str], bool],
    is_hidden: Callable[[str], bool],
    should_abort: Callable[[], bool],
) -> EarlyRunResult:
    """Walk *candidates* in order and prevalidate each costly one until one
    would be shown, a cheap one is reached, or the candidates run out.

    Candidates that are gone or hidden are passed over, the same as the
    slideshow's own skip loop would. *should_abort* is checked before each
    candidate; a running prevalidation is never interrupted.
    """
    result = EarlyRunResult()
    for media_path in candidates:
        if should_abort():
            result.stop_reason = StopReason.ABORTED
            return result
        if not os.path.isfile(media_path) or is_hidden(media_path):
            continue
        cached_hit, cached_action = ClassifierActionsManager.peek_session_cached_result(media_path)
        if cached_hit:
            if prevalidation_action_skips(cached_action):
                continue
            result.stop_reason = StopReason.SHOWN
            result.stopped_at = media_path
            return result
        if not is_costly(media_path):
            result.stop_reason = StopReason.CHEAP
            result.stopped_at = media_path
            return result
        try:
            action = prevalidate(media_path)
        except Exception:
            logger.exception(_("Early prevalidation failed for {0}").format(media_path))
            result.stop_reason = StopReason.ERROR
            result.stopped_at = media_path
            return result
        result.evaluated.append(media_path)
        if is_hidden(media_path) or prevalidation_action_skips(action):
            continue
        result.stop_reason = StopReason.SHOWN
        result.stopped_at = media_path
        return result
    return result

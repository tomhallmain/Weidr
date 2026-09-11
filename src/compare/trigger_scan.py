"""Seek to Trigger's scan, independent of any display.

A scan looks through a media file's sampled frames for the first one a
ClassifierAction (or Prevalidation) triggers on. The Qt Seek to Trigger tab
runs it on a worker thread and seeks its player to the result; the MCP
find_trigger tool reports the result as data. A still image is a one-slot
scan, and a still that doesn't match still carries the classifier's ranked
predictions -- what the model saw is the useful part of checking a still.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from compare.classifier_action import ClassifierAction, TriggerDetail, TriggerFrameResult
from compare.classifier_actions_manager import ClassifierActionsManager
from image.frame_cache import FrameCache
from utils.media_utils import is_classifier_dynamic_media_path

TRIGGER_ACTION_KINDS = ("classifier_action", "prevalidation")


@dataclass
class TriggerScan:
    """Outcome of one scan."""
    result: Optional[TriggerFrameResult]      # None when nothing triggered
    samples_scanned: int                      # 1 for a still
    no_match_detail: Optional[TriggerDetail] = None   # a non-matching still's ranked predictions


def scan_for_trigger(
    action: ClassifierAction, media_path: str,
    start_slot: int = 0, sample_ratio: Optional[float] = None,
) -> TriggerScan:
    """Find the first slot at or after *start_slot* that *action* triggers on.

    When nothing triggers from *start_slot* on, retries from slot 0, so
    repeatedly asking for the next trigger loops back to the first one. A
    result with slot_index < start_slot therefore means the scan wrapped.
    """
    result = action.find_first_trigger_slot(media_path, start_slot=start_slot, sample_ratio=sample_ratio)
    if result is None and start_slot > 0:
        result = action.find_first_trigger_slot(media_path, start_slot=0, sample_ratio=sample_ratio)
    if result is not None:
        return TriggerScan(result=result, samples_scanned=result.total_planned_slots)
    if not is_classifier_dynamic_media_path(media_path):
        return TriggerScan(result=None, samples_scanned=1,
                           no_match_detail=action.describe_image_prediction(media_path))
    stats = FrameCache.get_dynamic_media_stats(media_path)
    return TriggerScan(result=None, samples_scanned=getattr(stats, "total_items", 0) or 0)


def list_trigger_actions() -> dict:
    """The actions find_trigger can scan with, by kind."""
    return {
        "classifier_actions": [_describe_action(a) for a in ClassifierActionsManager.classifier_actions],
        "prevalidations": [_describe_action(p) for p in ClassifierActionsManager.prevalidations],
    }


def find_trigger(
    action_name: str, kind: str, media_path: str,
    start_slot: int = 0, sample_ratio: Optional[float] = None,
) -> dict:
    """Scan *media_path* with the named action and report the result as data.

    Raises ValueError for input the caller got wrong (unknown kind or name,
    missing file, an action that doesn't apply to this media type or can't
    run), as distinct from a scan that ran and found nothing.
    """
    if not os.path.isfile(media_path):
        raise ValueError(f"not a file: {media_path}")
    if start_slot < 0:
        raise ValueError("start_slot must be 0 or greater")
    if sample_ratio is not None and not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be greater than 0 and at most 1")
    action = _resolve_action(action_name, kind)
    if not action.media_type_allowed(media_path):
        raise ValueError(f"{kind} {action_name!r} does not apply to this media type")
    if not action.can_run:
        raise ValueError(f"{kind} {action_name!r} cannot run: {action.initialization_error}")

    scan = scan_for_trigger(action, media_path, start_slot=start_slot, sample_ratio=sample_ratio)
    result = scan.result
    if not is_classifier_dynamic_media_path(media_path):
        detail = result.detail if result is not None else scan.no_match_detail
        return {"media_kind": "still", "matched": result is not None, "detail": _detail_dict(detail)}
    if result is None:
        return {"media_kind": "dynamic", "matched": False, "samples_scanned": scan.samples_scanned}
    position = FrameCache.slot_index_to_seek_position(
        media_path, result.slot_index, result.total_planned_slots,
    )
    return {
        "media_kind": "dynamic",
        "matched": True,
        "slot_index": result.slot_index,
        "total_planned_slots": result.total_planned_slots,
        "wrapped": start_slot > 0 and result.slot_index < start_slot,
        "position": {"kind": position.kind, "value": position.value} if position is not None else None,
        "detail": _detail_dict(result.detail),
    }


def _resolve_action(action_name: str, kind: str) -> ClassifierAction:
    if kind == "classifier_action":
        actions = ClassifierActionsManager.classifier_actions
    elif kind == "prevalidation":
        actions = ClassifierActionsManager.prevalidations
    else:
        raise ValueError(f"kind must be one of {', '.join(TRIGGER_ACTION_KINDS)}, not {kind}")
    action = next((a for a in actions if a.name == action_name), None)
    if action is None:
        raise ValueError(f"no {kind} named {action_name!r}")
    if kind == "prevalidation":
        # The one-time setup prevalidate_media runs before first use: resolves
        # profiles, loads prototypes, and sets can_run from validation.
        ClassifierActionsManager._prevalidations_post_init()
    return action


def _describe_action(action: ClassifierAction) -> dict:
    types = action.applies_to_media_types
    return {
        "name": action.name,
        "applies_to_media_types": [t.value for t in types] if types is not None else None,
    }


def _detail_dict(detail: Optional[TriggerDetail]) -> Optional[dict]:
    if detail is None:
        return None
    ranked = detail.top_predictions
    return {
        "trigger_type": detail.trigger_type,
        "category": detail.category,
        # Scores can be numpy floats, which JSON can't encode.
        "top_predictions": (
            [{"category": str(category), "score": float(score)} for category, score in ranked]
            if ranked is not None else None
        ),
    }

"""Tests for compare/trigger_scan.py: the Seek to Trigger scan and the
plain-data report the MCP find_trigger tool returns.

The action is a stand-in scripted with find_first_trigger_slot results, so
no model is loaded; whether a path counts as dynamic media, and FrameCache's
stats/seek lookups, are patched per test.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from compare import trigger_scan
from compare.classifier_action import TriggerDetail, TriggerFrameResult
from compare.classifier_actions_manager import ClassifierActionsManager
from image.frame_cache import SeekPosition


class _Action:
    """Duck-typed ClassifierAction: returns scripted scan results in order."""

    def __init__(self, name="Cats", results=(), no_match_detail=None,
                 allowed=True, can_run=True, applies_to_media_types=None):
        self.name = name
        self._results = list(results)
        self._no_match_detail = no_match_detail
        self._allowed = allowed
        self.can_run = can_run
        self.initialization_error = None if can_run else "model missing"
        self.applies_to_media_types = applies_to_media_types
        self.scan_calls = []

    def find_first_trigger_slot(self, media_path, start_slot=0, sample_ratio=None):
        self.scan_calls.append((start_slot, sample_ratio))
        return self._results.pop(0) if self._results else None

    def describe_image_prediction(self, media_path):
        return self._no_match_detail

    def media_type_allowed(self, media_path):
        return self._allowed


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    return str(path)


@pytest.fixture
def as_dynamic(monkeypatch):
    monkeypatch.setattr(trigger_scan, "is_classifier_dynamic_media_path", lambda p: True)


@pytest.fixture
def as_still(monkeypatch):
    monkeypatch.setattr(trigger_scan, "is_classifier_dynamic_media_path", lambda p: False)


@pytest.fixture
def actions(monkeypatch):
    def install(classifier_actions=(), prevalidations=()):
        monkeypatch.setattr(ClassifierActionsManager, "classifier_actions", list(classifier_actions))
        monkeypatch.setattr(ClassifierActionsManager, "prevalidations", list(prevalidations))
    return install


def _detail():
    return TriggerDetail(
        trigger_type="image_classifier", category="cat",
        top_predictions=[("cat", np.float32(0.9)), ("dog", np.float32(0.1))],
    )


# ---------------------------------------------------------------------------
# scan_for_trigger
# ---------------------------------------------------------------------------

def test_scan_retries_from_the_start_when_nothing_triggers_after_start_slot(media, as_dynamic):
    found = TriggerFrameResult(slot_index=2, total_planned_slots=10, frame_path="f2.png")
    action = _Action(results=[None, found])
    scan = trigger_scan.scan_for_trigger(action, media, start_slot=5)
    assert scan.result is found
    assert action.scan_calls == [(5, None), (0, None)]


def test_scan_of_a_non_matching_still_carries_the_ranked_predictions(media, as_still):
    detail = _detail()
    scan = trigger_scan.scan_for_trigger(_Action(no_match_detail=detail), media)
    assert scan.result is None
    assert scan.samples_scanned == 1
    assert scan.no_match_detail is detail


def test_scan_of_non_matching_dynamic_media_reports_the_frame_count(media, as_dynamic, monkeypatch):
    monkeypatch.setattr(trigger_scan.FrameCache, "get_dynamic_media_stats",
                        classmethod(lambda cls, p: SimpleNamespace(total_items=48)))
    scan = trigger_scan.scan_for_trigger(_Action(), media)
    assert scan.result is None
    assert scan.samples_scanned == 48
    assert scan.no_match_detail is None


# ---------------------------------------------------------------------------
# find_trigger report
# ---------------------------------------------------------------------------

def test_dynamic_match_reports_slot_position_and_json_safe_detail(media, as_dynamic, actions, monkeypatch):
    found = TriggerFrameResult(slot_index=3, total_planned_slots=12, frame_path="f3.png", detail=_detail())
    actions(classifier_actions=[_Action(results=[found])])
    monkeypatch.setattr(trigger_scan.FrameCache, "slot_index_to_seek_position",
                        classmethod(lambda cls, p, slot, total: SeekPosition(kind="ms", value=4500)))

    report = trigger_scan.find_trigger("Cats", "classifier_action", media)

    assert report == {
        "media_kind": "dynamic", "matched": True, "slot_index": 3, "total_planned_slots": 12,
        "wrapped": False, "position": {"kind": "ms", "value": 4500},
        "detail": {
            "trigger_type": "image_classifier", "category": "cat",
            "top_predictions": [
                {"category": "cat", "score": pytest.approx(0.9)},
                {"category": "dog", "score": pytest.approx(0.1)},
            ],
        },
    }
    json.dumps(report)  # numpy scores were converted


def test_dynamic_match_found_by_wrapping_is_flagged(media, as_dynamic, actions, monkeypatch):
    found = TriggerFrameResult(slot_index=1, total_planned_slots=12, frame_path="f1.png")
    actions(classifier_actions=[_Action(results=[None, found])])
    monkeypatch.setattr(trigger_scan.FrameCache, "slot_index_to_seek_position",
                        classmethod(lambda cls, p, slot, total: None))

    report = trigger_scan.find_trigger("Cats", "classifier_action", media, start_slot=4)

    assert report["wrapped"] is True
    assert report["position"] is None


def test_dynamic_no_match_reports_samples_scanned(media, as_dynamic, actions, monkeypatch):
    actions(classifier_actions=[_Action()])
    monkeypatch.setattr(trigger_scan.FrameCache, "get_dynamic_media_stats",
                        classmethod(lambda cls, p: SimpleNamespace(total_items=30)))
    assert trigger_scan.find_trigger("Cats", "classifier_action", media) == {
        "media_kind": "dynamic", "matched": False, "samples_scanned": 30,
    }


def test_still_no_match_still_reports_detail(media, as_still, actions):
    actions(classifier_actions=[_Action(no_match_detail=_detail())])
    report = trigger_scan.find_trigger("Cats", "classifier_action", media)
    assert report["media_kind"] == "still"
    assert report["matched"] is False
    assert report["detail"]["category"] == "cat"


def test_prevalidation_kind_runs_the_lazy_setup_first(media, as_still, actions, monkeypatch):
    calls = []
    monkeypatch.setattr(ClassifierActionsManager, "_prevalidations_post_init",
                        staticmethod(lambda: calls.append(True)))
    actions(prevalidations=[_Action(name="NoCats")])
    trigger_scan.find_trigger("NoCats", "prevalidation", media)
    assert calls == [True]


@pytest.mark.parametrize("kwargs, installed", [
    ({"action_name": "Cats", "kind": "bogus"}, _Action()),
    ({"action_name": "Missing", "kind": "classifier_action"}, _Action()),
    ({"action_name": "Cats", "kind": "classifier_action"}, _Action(allowed=False)),
    ({"action_name": "Cats", "kind": "classifier_action"}, _Action(can_run=False)),
    ({"action_name": "Cats", "kind": "classifier_action", "start_slot": -1}, _Action()),
    ({"action_name": "Cats", "kind": "classifier_action", "sample_ratio": 1.5}, _Action()),
])
def test_caller_errors_raise_value_error(media, as_still, actions, kwargs, installed):
    actions(classifier_actions=[installed])
    with pytest.raises(ValueError):
        trigger_scan.find_trigger(media_path=media, **kwargs)
    assert installed.scan_calls == []


def test_missing_file_raises_value_error(tmp_path, actions):
    actions(classifier_actions=[_Action()])
    with pytest.raises(ValueError):
        trigger_scan.find_trigger("Cats", "classifier_action", str(tmp_path / "nope.mp4"))


def test_list_trigger_actions(actions):
    from utils.constants import CompareMediaType

    actions(
        classifier_actions=[_Action(name="Cats")],
        prevalidations=[_Action(name="NoVideo", applies_to_media_types=[CompareMediaType.IMAGE])],
    )
    assert trigger_scan.list_trigger_actions() == {
        "classifier_actions": [{"name": "Cats", "applies_to_media_types": None}],
        "prevalidations": [{"name": "NoVideo", "applies_to_media_types": [CompareMediaType.IMAGE.value]}],
    }

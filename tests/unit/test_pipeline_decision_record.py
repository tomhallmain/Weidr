"""
Unit tests for per-file decision records (compare.pipeline_decision_record) and
their capture by the pipeline runner.

Records are opt-in per pipeline via record_node_verdicts; the off case is
covered here too, since leaving existing runs untouched is the point of the
flag defaulting to False.

All external ML dependencies are mocked, matching test_classifier_pipeline_runner.
"""

import pytest

from compare.action_callbacks import ActionCallbacks
from compare.classifier_pipeline import (
    ClassifierPipeline,
    EmbeddingCondition,
    GroupCondition,
    NodeOutcome,
    OutcomeType,
    PipelineNode,
)
from compare.classifier_pipeline_runner import run_pipeline
from compare.pipeline_decision_record import (
    DECISION_RECORD_VERSION,
    build_decision_record,
)
from compare.pipeline_run_report import PipelineRunReport, PipelineRunStats
from utils.constants import ClassifierActionType
from utils.translations import _

IMAGE = "/fake/image.jpg"


def _node(name, condition, on_match=None, on_no_match=None):
    return PipelineNode(
        name=name,
        condition=condition,
        on_match=on_match or NodeOutcome.continue_(),
        on_no_match=on_no_match or NodeOutcome.accept(),
    )


def _pipeline(*nodes, record=True, **kwargs):
    return ClassifierPipeline(
        name="test", nodes=list(nodes), record_node_verdicts=record, **kwargs
    )


def _patch_embedding(monkeypatch, result):
    """Patch CLIP text comparison. *result* is a bool, or a callable taking the
    positives list so different nodes can resolve differently."""
    import compare.compare_embeddings_clip as clip_mod
    if callable(result):
        fn = lambda path, pos, neg, thresh: result(pos)
    else:
        fn = lambda *a, **kw: result
    monkeypatch.setattr(clip_mod.CompareEmbeddingClip, "multi_text_compare", staticmethod(fn))


# ---------------------------------------------------------------------------
# build_decision_record
# ---------------------------------------------------------------------------

class TestBuildDecisionRecord:
    def test_shape(self):
        rec = build_decision_record(
            "p", IMAGE, ClassifierActionType.NOTIFY, {"n1": True}, {"n1": 0.75}
        )
        assert rec["version"] == DECISION_RECORD_VERSION
        assert rec["pipeline_name"] == "p"
        assert rec["path"] == IMAGE
        assert rec["action"] == "NOTIFY"
        assert rec["node_verdicts"] == {"n1": {"matched": True, "score": 0.75}}

    def test_action_none_when_no_action_fired(self):
        rec = build_decision_record("p", IMAGE, None, {}, {})
        assert rec["action"] is None

    def test_missing_score_becomes_none(self):
        rec = build_decision_record("p", IMAGE, None, {"n1": False}, {})
        assert rec["node_verdicts"]["n1"] == {"matched": False, "score": None}

    def test_string_score_preserved(self):
        """Several conditions return a descriptive string rather than a float
        (matched media type, matched filename pattern) — it is their only score."""
        rec = build_decision_record("p", IMAGE, None, {"n1": True}, {"n1": "image"})
        assert rec["node_verdicts"]["n1"]["score"] == "image"

    def test_bool_score_dropped(self):
        """bool is an int subclass; a True/False 'score' duplicates matched."""
        rec = build_decision_record("p", IMAGE, None, {"n1": True}, {"n1": True})
        assert rec["node_verdicts"]["n1"]["score"] is None

    def test_unserializable_score_becomes_none(self):
        rec = build_decision_record("p", IMAGE, None, {"n1": True}, {"n1": object()})
        assert rec["node_verdicts"]["n1"]["score"] is None

    def test_source_dicts_not_aliased(self):
        results, scores = {"n1": True}, {"n1": 0.5}
        rec = build_decision_record("p", IMAGE, None, results, scores)
        results["n2"] = False
        assert "n2" not in rec["node_verdicts"]

    def test_record_is_json_serializable(self):
        import json
        rec = build_decision_record(
            "p", IMAGE, ClassifierActionType.MOVE, {"a": True, "b": False},
            {"a": 0.9, "b": "pattern"},
        )
        assert json.loads(json.dumps(rec)) == rec


# ---------------------------------------------------------------------------
# Capture during a run
# ---------------------------------------------------------------------------

class TestRunPipelineCapture:
    def test_disabled_by_default_emits_nothing(self, monkeypatch):
        _patch_embedding(monkeypatch, True)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])), record=False)
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        assert report.decisions() == []

    def test_enabled_emits_one_record(self, monkeypatch):
        _patch_embedding(monkeypatch, True)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])))
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        assert len(report.decisions()) == 1
        assert report.decisions()[0]["path"] == IMAGE

    def test_no_report_is_not_an_error(self, monkeypatch):
        """record_node_verdicts with no report to write into must not raise."""
        _patch_embedding(monkeypatch, True)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])))
        assert run_pipeline(p, IMAGE, ActionCallbacks()) is None

    def test_records_are_not_messages(self, monkeypatch):
        """Decision records must stay out of the message list, or the completion
        report and per-seed summaries would be buried under one line per file."""
        _patch_embedding(monkeypatch, True)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])))
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        assert report.decisions()
        assert report.messages() == []

    def test_captures_action_that_fired(self, monkeypatch):
        _patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=NodeOutcome(OutcomeType.EXECUTE,
                                       action_type=ClassifierActionType.NOTIFY))
        )
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=lambda *a, **kw: None),
                     report=report)
        assert report.decisions()[0]["action"] == "NOTIFY"

    def test_captures_accept_as_no_action(self, monkeypatch):
        _patch_embedding(monkeypatch, False)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])))
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        rec = report.decisions()[0]
        assert rec["action"] is None
        assert rec["node_verdicts"]["n1"]["matched"] is False

    def test_captures_reject_action(self, monkeypatch):
        _patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=NodeOutcome(OutcomeType.REJECT)),
            default_reject_action=ClassifierActionType.SKIP,
        )
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=lambda *a, **kw: None),
                     report=report)
        assert report.decisions()[0]["action"] == "SKIP"

    def test_captures_default_action_when_nodes_exhausted(self, monkeypatch):
        _patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]), on_match=NodeOutcome.continue_()),
            default_action=ClassifierActionType.SKIP,
        )
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=lambda *a, **kw: None),
                     report=report)
        assert report.decisions()[0]["action"] == "SKIP"

    def test_only_evaluated_nodes_appear(self, monkeypatch):
        """A node the walk halted before never ran, so it is absent rather than
        recorded as a no-match — 'did not run' and 'ran and did not match' are
        different facts."""
        _patch_embedding(monkeypatch, True)
        p = _pipeline(
            _node("n1", EmbeddingCondition(["x"]),
                  on_match=NodeOutcome(OutcomeType.EXECUTE,
                                       action_type=ClassifierActionType.NOTIFY)),
            _node("n2", EmbeddingCondition(["y"])),
        )
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(notify_callback=lambda *a, **kw: None),
                     report=report)
        verdicts = report.decisions()[0]["node_verdicts"]
        assert "n1" in verdicts
        assert "n2" not in verdicts

    def test_group_children_recorded_under_composite_keys(self, monkeypatch):
        """Group children are stored by the runner as '<group>/<child>'; the
        record surfaces them verbatim so a per-member verdict is readable."""
        _patch_embedding(monkeypatch, lambda pos: pos == ["yes"])
        group = GroupCondition(
            operator="AND",
            nodes=[
                PipelineNode(name="c_yes", condition=EmbeddingCondition(["yes"])),
                PipelineNode(name="c_no", condition=EmbeddingCondition(["no"])),
            ],
        )
        p = _pipeline(_node("g", group))
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        verdicts = report.decisions()[0]["node_verdicts"]
        assert verdicts["g"]["matched"] is False           # AND over children
        assert verdicts["g/c_yes"]["matched"] is True
        assert verdicts["g/c_no"]["matched"] is False

    def test_one_record_per_file_across_a_batch(self, monkeypatch):
        _patch_embedding(monkeypatch, True)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])))
        report = PipelineRunReport()
        for path in ("/fake/a.jpg", "/fake/b.jpg", "/fake/c.jpg"):
            run_pipeline(p, path, ActionCallbacks(), report=report)
        assert [r["path"] for r in report.decisions()] == [
            "/fake/a.jpg", "/fake/b.jpg", "/fake/c.jpg"
        ]

    def test_inactive_pipeline_records_nothing(self, monkeypatch):
        """No node ran, so there is no verdict to record."""
        _patch_embedding(monkeypatch, True)
        p = _pipeline(_node("n1", EmbeddingCondition(["x"])))
        p.is_active = False
        report = PipelineRunReport()
        run_pipeline(p, IMAGE, ActionCallbacks(), report=report)
        assert report.decisions() == []


# ---------------------------------------------------------------------------
# PipelineRunReport decision channel
# ---------------------------------------------------------------------------

class TestReportDecisionChannel:
    def test_starts_empty(self):
        assert PipelineRunReport().decisions() == []
        assert PipelineRunReport().decision_count() == 0

    def test_add_and_count(self):
        report = PipelineRunReport()
        report.add_decision({"path": "a"})
        report.add_decision({"path": "b"})
        assert report.decision_count() == 2

    def test_decisions_returns_a_copy(self):
        report = PipelineRunReport()
        report.add_decision({"path": "a"})
        report.decisions().append({"path": "b"})
        assert report.decision_count() == 1

    def test_clear_clears_decisions(self):
        report = PipelineRunReport()
        report.add_decision({"path": "a"})
        report.clear()
        assert report.decisions() == []

    def test_decisions_do_not_affect_has_messages(self):
        report = PipelineRunReport()
        report.add_decision({"path": "a"})
        assert report.has_messages() is False

    def test_completion_report_shows_count_not_records(self):
        report = PipelineRunReport()
        for i in range(3):
            report.add_decision({"path": f"/fake/{i}.jpg"})
        text = report.format_completion_report(PipelineRunStats(pipeline_name="p"))
        assert _("Decision records captured: {0}").format(3) in text
        assert "/fake/0.jpg" not in text

    def test_completion_report_omits_line_when_no_decisions(self):
        report = PipelineRunReport()
        text = report.format_completion_report(PipelineRunStats(pipeline_name="p"))
        assert _("Decision records captured: {0}").format(0) not in text

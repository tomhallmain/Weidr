"""Tests for compare/pipeline_run_service.py: registering, ending, stopping
and resuming batch runs. The batch is replaced by fakes; the history is the
in-memory process singleton (reset per test by the root conftest)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from compare import pipeline_run_service as svc
from compare.classifier_pipeline import ClassifierPipelines
from compare.pipeline_run_control import CANCEL_SHUTDOWN, RunCheckpoint
from compare.pipeline_run_history import RunStatus, pipeline_run_history
from compare.pipeline_run_report import PipelineRunStats
from utils.path_utils import canonical_path
from files.directory_profile import DirectoryProfile


class _Pipeline:
    def __init__(self, name="Sorter", nodes=1):
        self.name = name
        self.nodes = nodes

    def to_dict(self):
        return {"name": self.name, "nodes": self.nodes}


def _fake_batch(monkeypatch, *, errors=0, cancelled=False, raises=None, hook=None):
    calls = []

    def run(pipeline, directories, **kwargs):
        calls.append({"directories": list(directories), **kwargs})
        if hook is not None:
            hook(kwargs["run_control"])
        if raises is not None:
            raise raises
        stats = PipelineRunStats(pipeline_name=pipeline.name, files_evaluated=5, errors=errors,
                                 action_counts={"MOVE": 2}, cancelled=cancelled)
        return SimpleNamespace(stats=stats, summary="summary", cancelled=cancelled,
                               dump_path="/logs/dump.json")

    monkeypatch.setattr(svc.pipeline_batch, "run_pipeline_over_directories", run)
    return calls


def _begin(dirs=("/data/a",), **kw):
    kw.setdefault("profile_name", "Photos")
    kw.setdefault("source", "gui")
    return svc.begin_run(kw.pop("pipeline", _Pipeline()), list(dirs), **kw)


class TestBeginRun:
    def test_registers_a_running_record(self):
        active = _begin()
        record = pipeline_run_history.get(active.run_id)
        assert record.status is RunStatus.RUNNING
        assert record.directories == ["/data/a"]
        assert record.pipeline_fingerprint is not None
        assert svc.is_active(active.run_id)

    def test_refuses_overlap_with_a_running_run(self):
        _begin(["/data"])
        with pytest.raises(svc.RunRefused):
            _begin(["/data/a/b"])
        # The GUI asks the user and may go ahead.
        assert _begin(["/data/a/b"], allow_running_overlap=True)

    def test_unrelated_directories_are_not_refused(self):
        _begin(["/data/a"])
        assert _begin(["/data/b"])

    def test_resume_requires_a_resumable_run(self):
        active = _begin()
        with pytest.raises(svc.RunRefused):
            _begin(resume_from=active.run_id, allow_running_overlap=True)
        with pytest.raises(svc.RunRefused):
            _begin(resume_from="missing")


class TestExecuteRun:
    @pytest.mark.parametrize("errors, cancelled, expected", [
        (0, False, RunStatus.COMPLETED),
        (2, False, RunStatus.COMPLETED_WITH_ERRORS),
        (0, True, RunStatus.CANCELLED),
    ])
    def test_status_from_outcome(self, monkeypatch, errors, cancelled, expected):
        _fake_batch(monkeypatch, errors=errors, cancelled=cancelled)
        active = _begin()
        svc.execute_run(active)
        record = pipeline_run_history.get(active.run_id)
        assert record.status is expected
        assert record.files_evaluated == 5
        assert record.action_counts == {"MOVE": 2}
        assert record.summary == "summary"
        assert record.dump_path == "/logs/dump.json"
        assert record.is_resumable == (expected is RunStatus.CANCELLED)
        assert not svc.is_active(active.run_id)
        assert active.done.is_set()

    def test_shutdown_cancel_is_recorded_as_interrupted(self, monkeypatch):
        _fake_batch(monkeypatch, cancelled=True,
                    hook=lambda control: control.request_cancel(CANCEL_SHUTDOWN))
        active = _begin()
        svc.execute_run(active)
        assert pipeline_run_history.get(active.run_id).status is RunStatus.INTERRUPTED

    def test_batch_exception_is_recorded_and_reraised(self, monkeypatch):
        _fake_batch(monkeypatch, raises=RuntimeError("disk gone"))
        active = _begin()
        with pytest.raises(RuntimeError):
            svc.execute_run(active)
        record = pipeline_run_history.get(active.run_id)
        assert record.status is RunStatus.FAILED
        assert record.error == "disk gone"
        assert record.is_resumable
        assert not svc.is_active(active.run_id)

    def test_batch_gets_the_run_control(self, monkeypatch):
        calls = _fake_batch(monkeypatch)
        active = _begin(["/x", "/y"])
        svc.execute_run(active, generation_type="G")
        assert calls[0]["run_control"] is active.control
        assert calls[0]["directories"] == ["/x", "/y"]
        assert calls[0]["generation_type"] == "G"
        assert calls[0]["profile_name"] == "Photos"

    def test_abandon_marks_failed(self):
        active = _begin()
        svc.abandon(active, "thread did not start")
        assert pipeline_run_history.get(active.run_id).status is RunStatus.FAILED
        assert not svc.is_active(active.run_id)


class TestStop:
    def test_request_stop_cancels_and_marks(self):
        active = _begin()
        assert svc.request_stop(active.run_id)
        assert active.control.is_cancelled()
        assert pipeline_run_history.get(active.run_id).stop_requested

    def test_request_stop_unknown_run(self):
        assert not svc.request_stop("missing")

    def test_shutdown_cancels_and_waits(self, monkeypatch):
        released = threading.Event()

        def hook(control):
            while not control.is_cancelled():
                released.wait(0.01)

        _fake_batch(monkeypatch, cancelled=True, hook=hook)
        active = _begin()
        worker = threading.Thread(target=svc.execute_run, args=(active,))
        worker.start()
        svc.shutdown(timeout=5)
        worker.join(5)
        assert active.control.cancel_reason == CANCEL_SHUTDOWN
        assert pipeline_run_history.get(active.run_id).status is RunStatus.INTERRUPTED


class TestResume:
    @pytest.fixture
    def cancelled_run(self, monkeypatch):
        pipeline = _Pipeline()
        monkeypatch.setattr(ClassifierPipelines, "pipelines", [pipeline])
        monkeypatch.setattr(DirectoryProfile, "directory_profiles",
                            [DirectoryProfile("Photos", ["/data/a"])])
        _fake_batch(monkeypatch, cancelled=True)
        active = _begin(pipeline=pipeline)
        svc.execute_run(active)
        return pipeline, active.run_id

    def test_check_resume_ok(self, cancelled_run, monkeypatch):
        pipeline, run_id = cancelled_run
        monkeypatch.setattr(svc.os.path, "isdir", lambda d: True)
        check = svc.check_resume(run_id)
        assert check.problem is None
        assert check.pipeline is pipeline
        assert not check.pipeline_changed and not check.profile_changed
        assert check.missing_directories == []

    def test_check_resume_reports_changes(self, cancelled_run, monkeypatch):
        pipeline, run_id = cancelled_run
        pipeline.nodes = 2
        monkeypatch.setattr(DirectoryProfile, "directory_profiles",
                            [DirectoryProfile("Photos", ["/data/other"])])
        check = svc.check_resume(run_id)
        assert check.problem is None
        assert check.pipeline_changed
        assert check.profile_changed
        assert check.missing_directories == ["/data/a"]

    def test_check_resume_missing_pipeline(self, cancelled_run, monkeypatch):
        from utils.translations import _
        _pipeline, run_id = cancelled_run
        monkeypatch.setattr(ClassifierPipelines, "pipelines", [])
        assert svc.check_resume(run_id).problem == _("Pipeline '{0}' no longer exists.").format("Sorter")

    def test_resume_seeds_the_control_and_links_records(self, cancelled_run, monkeypatch):
        pipeline, run_id = cancelled_run
        pipeline_run_history.write_checkpoint(run_id, RunCheckpoint(
            completed_directories=[canonical_path("/data/a")], files_evaluated=9))
        active = _begin(pipeline=pipeline, resume_from=run_id)
        assert active.control.resumed_from == run_id
        assert active.control.is_directory_completed("/data/a")
        assert active.control.seed_counters()[0] == 9
        original = pipeline_run_history.get(run_id)
        assert original.resumed_by == active.run_id
        assert not original.is_resumable
        assert pipeline_run_history.get(active.run_id).files_evaluated == 9
        with pytest.raises(svc.RunRefused):
            _begin(pipeline=pipeline, resume_from=run_id, allow_running_overlap=True)
        assert svc.check_resume(run_id).problem is not None


def test_plan_run_reports_last_running_and_resumable(monkeypatch):
    _fake_batch(monkeypatch, cancelled=True)
    done = _begin(["/data"])
    svc.execute_run(done)
    running = _begin(["/data/a/x"], pipeline=_Pipeline("Other"), allow_running_overlap=True)

    plan = svc.plan_run("Sorter", ["/data/a"])
    assert plan.last_run.run_id == done.run_id
    assert not plan.last_run_same_set
    assert plan.last_run_overlap == [("/data/a", "/data")]
    assert [r.run_id for r in plan.running] == [running.run_id]
    # Resuming is offered only for the same directory set.
    assert plan.resumable is None
    assert svc.plan_run("Sorter", ["/data"]).resumable.run_id == done.run_id

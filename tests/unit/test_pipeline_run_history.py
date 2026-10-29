"""Tests for compare/pipeline_run_history.py: the persistent run history,
the registry of running runs, crash reconciliation and checkpoints.

Encryption is replaced by plain file I/O so no key material is touched.
"""

import json
import os

import pytest

from compare import pipeline_run_history as prh
from compare.pipeline_run_control import RunCheckpoint, RunProgress
from compare.pipeline_run_history import (
    PipelineRunHistory,
    PipelineRunRecord,
    RunOwner,
    RunStatus,
    pipeline_fingerprint,
)


@pytest.fixture
def plain_encryption(monkeypatch):
    import utils.encryptor as enc

    def _encrypt(data, service_name, app_identifier, output_path, *a, **kw):
        with open(output_path, "wb") as f:
            f.write(data)

    def _decrypt(path, service_name, app_identifier):
        with open(path, "rb") as f:
            return f.read()

    monkeypatch.setattr(enc, "encrypt_data_to_file", _encrypt)
    monkeypatch.setattr(enc, "decrypt_data_from_file", _decrypt)


@pytest.fixture
def dead_pids(monkeypatch):
    """pids in the returned set report as not running."""
    import psutil
    dead = set()
    real = psutil.pid_exists
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid not in dead and real(pid))
    return dead


def _record(history, name="P", dirs=("/data/a",), **kw):
    return PipelineRunRecord.new(name, list(dirs), profile_name="prof", source="gui",
                                 owner=history.owner, **kw)


def _foreign(name="P", dirs=("/data/a",), pid=999_999, status=RunStatus.RUNNING, started=100.0):
    owner = RunOwner(token="other", pid=pid, create_time=None)
    r = PipelineRunRecord.new(name, list(dirs), profile_name="prof", source="mcp", owner=owner)
    r.status = status
    r.started_at = r.updated_at = started
    return r


def _write_index(records):
    path = prh._index_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": prh.FORMAT_VERSION, "runs": [r.to_json() for r in records]}, f)


def _read_index():
    with open(prh._index_path(), encoding="utf-8") as f:
        return {r["run_id"]: r for r in json.load(f)["runs"]}


class TestRecord:
    def test_json_round_trip(self):
        h = PipelineRunHistory()
        r = _record(h)
        r.status = RunStatus.CANCELLED
        r.action_counts = {"MOVE": 2}
        assert PipelineRunRecord.from_json(r.to_json()) == r

    def test_malformed_record_is_dropped(self):
        assert PipelineRunRecord.from_json({"run_id": "x"}) is None
        assert PipelineRunRecord.from_json({**_record(PipelineRunHistory()).to_json(),
                                            "status": "bogus"}) is None

    def test_resumable_needs_unfinished_status_checkpoint_and_no_successor(self):
        r = _record(PipelineRunHistory())
        r.status, r.has_checkpoint = RunStatus.INTERRUPTED, True
        assert r.is_resumable
        r.resumed_by = "next"
        assert not r.is_resumable
        r.resumed_by, r.status = None, RunStatus.COMPLETED
        assert not r.is_resumable

    def test_status_labels_are_translated(self):
        from utils.translations import _
        assert RunStatus.CANCELLED.label() == _("Cancelled")


def test_fingerprint_ignores_name_and_active_flag():
    class _P:
        def __init__(self, **d):
            self.d = d

        def to_dict(self):
            return dict(self.d)

    a = pipeline_fingerprint(_P(name="a", is_active=True, nodes=[1]))
    assert a == pipeline_fingerprint(_P(name="b", is_active=False, nodes=[1]))
    assert a != pipeline_fingerprint(_P(name="a", nodes=[2]))
    assert pipeline_fingerprint(object()) is None


class TestPersistence:
    def test_store_before_load_writes_nothing(self, plain_encryption):
        h = PipelineRunHistory()
        h.add_run(_record(h))
        h.store()
        assert not os.path.exists(prh._index_path())

    def test_round_trip(self, plain_encryption):
        h = PipelineRunHistory()
        h.load()
        r = _record(h)
        h.add_run(r)
        h.finish_run(r.run_id, RunStatus.COMPLETED, summary="done")
        fresh = PipelineRunHistory()
        fresh.load()
        got = fresh.get(r.run_id)
        assert got.status is RunStatus.COMPLETED and got.summary == "done"
        assert got.directories == ["/data/a"]
        assert not [n for n in os.listdir(os.path.dirname(prh._index_path())) if n.endswith(".tmp")]

    def test_unreadable_index_is_set_aside(self, plain_encryption):
        path = prh._index_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00not json")
        h = PipelineRunHistory()
        h.load()
        assert os.path.exists(path + ".unreadable")
        assert h.records() == []

    def test_checkpoint_written_and_dropped_on_completion(self, plain_encryption):
        h = PipelineRunHistory()
        h.load()
        r = _record(h)
        h.add_run(r)
        h.write_checkpoint(r.run_id, RunCheckpoint(files_evaluated=3))
        assert os.path.exists(prh._progress_path(r.run_id))
        assert h.get(r.run_id).has_checkpoint
        h.finish_run(r.run_id, RunStatus.COMPLETED)
        assert not os.path.exists(prh._progress_path(r.run_id))
        assert h.read_checkpoint(r.run_id) is None

    def test_cancelled_run_keeps_final_checkpoint(self, plain_encryption):
        h = PipelineRunHistory()
        h.load()
        r = _record(h)
        h.add_run(r)
        h.finish_run(r.run_id, RunStatus.CANCELLED,
                     counters=RunCheckpoint(files_evaluated=4, completed_directories=["x"]))
        fresh = PipelineRunHistory()
        fresh.load()
        assert fresh.get(r.run_id).is_resumable
        assert fresh.read_checkpoint(r.run_id).files_evaluated == 4

    def test_orphan_progress_files_removed_on_load(self, plain_encryption):
        orphan = prh._progress_path("orphan")
        os.makedirs(os.path.dirname(orphan), exist_ok=True)
        open(orphan, "wb").close()
        PipelineRunHistory().load()
        assert not os.path.exists(orphan)

    def test_prune_keeps_running_and_newest_finished(self, plain_encryption, monkeypatch):
        monkeypatch.setattr(prh, "MAX_FINISHED_RECORDS", 2)
        h = PipelineRunHistory()
        h.load()
        ids = []
        for i in range(4):
            r = _record(h)
            r.started_at = float(i)
            h.add_run(r)
            h.finish_run(r.run_id, RunStatus.COMPLETED)
            ids.append(r.run_id)
        running = _record(h)
        running.started_at = -1.0
        h.add_run(running)
        kept = {r.run_id for r in h.records()}
        assert kept == {ids[2], ids[3], running.run_id}


class TestReconciliation:
    def test_dead_foreign_running_record_becomes_interrupted(self, plain_encryption, dead_pids):
        dead_pids.add(999_999)
        foreign = _foreign()
        _write_index([foreign])
        h = PipelineRunHistory()
        h.load()
        got = h.get(foreign.run_id)
        assert got.status is RunStatus.INTERRUPTED
        assert got.finished_at == foreign.updated_at
        assert _read_index()[foreign.run_id]["status"] == RunStatus.INTERRUPTED.value

    def test_live_foreign_running_record_stays_running(self, plain_encryption):
        foreign = _foreign(pid=os.getpid())
        _write_index([foreign])
        h = PipelineRunHistory()
        h.load()
        assert h.get(foreign.run_id).status is RunStatus.RUNNING
        assert not h.is_own(h.get(foreign.run_id))

    def test_own_running_record_is_never_reconciled(self, plain_encryption):
        h = PipelineRunHistory()
        h.load()
        r = _record(h)
        h.add_run(r)
        assert h.records()[0].status is RunStatus.RUNNING

    def test_write_keeps_records_other_processes_added(self, plain_encryption):
        h = PipelineRunHistory()
        h.load()
        foreign = _foreign(status=RunStatus.COMPLETED)
        _write_index([foreign])  # another process wrote after our load
        h.add_run(_record(h))
        assert foreign.run_id in _read_index()

    def test_record_removed_by_other_process_is_dropped(self, plain_encryption):
        foreign = _foreign(status=RunStatus.COMPLETED)
        _write_index([foreign])
        h = PipelineRunHistory()
        h.load()
        _write_index([])  # the other process removed it
        h.sync_from_disk()
        assert h.get(foreign.run_id) is None


class TestQueries:
    def _finished(self, h, name, dirs, started, status=RunStatus.COMPLETED):
        r = _record(h, name=name, dirs=dirs)
        r.started_at = started
        h.add_run(r)
        h.finish_run(r.run_id, status, counters=RunCheckpoint() if status is not RunStatus.COMPLETED else None)
        return r

    def test_last_overlapping_matches_same_pipeline_and_containment(self):
        h = PipelineRunHistory()
        old = self._finished(h, "P", ["/data"], 1.0)
        self._finished(h, "Q", ["/data/a"], 3.0)
        newer = self._finished(h, "P", ["/data/a/b"], 2.0)
        assert h.last_overlapping("P", ["/data/a"]).run_id == newer.run_id
        assert h.last_overlapping("P", ["/other"]) is None
        assert h.last_overlapping("P", ["/data/z"]).run_id == old.run_id

    def test_running_overlapping_any_pipeline(self):
        h = PipelineRunHistory()
        r = _record(h, name="Q", dirs=["/data/a"])
        h.add_run(r)
        assert [x.run_id for x in h.running_overlapping(["/data"])] == [r.run_id]
        assert h.running_overlapping(["/elsewhere"]) == []

    def test_latest_resumable_requires_same_directory_set(self):
        h = PipelineRunHistory()
        r = self._finished(h, "P", ["/data/a", "/data/b"], 1.0, status=RunStatus.INTERRUPTED)
        assert h.latest_resumable("P", ["/data/b", "/data/a/"]).run_id == r.run_id
        assert h.latest_resumable("P", ["/data/a"]) is None
        assert h.latest_resumable().run_id == r.run_id

    def test_link_resume_retires_original(self):
        h = PipelineRunHistory()
        r = self._finished(h, "P", ["/data/a"], 1.0, status=RunStatus.CANCELLED)
        assert h.get(r.run_id).is_resumable
        h.link_resume(r.run_id, "next")
        assert not h.get(r.run_id).is_resumable
        assert h.get(r.run_id).resumed_by == "next"

    def test_remove_refuses_running(self):
        h = PipelineRunHistory()
        running = _record(h)
        h.add_run(running)
        assert not h.remove(running.run_id)
        done = self._finished(h, "P", ["/x"], 1.0)
        assert h.remove(done.run_id)
        assert h.get(done.run_id) is None

    def test_update_progress_and_stop_request(self):
        h = PipelineRunHistory()
        r = _record(h)
        h.add_run(r)
        h.update_progress(r.run_id, RunProgress(
            directories_total=1, directories_done=0, current_directory="/data/a",
            current_dir_total=10, current_dir_done=4, files_evaluated=4,
            files_skipped=0, errors=1, action_counts={"MOVE": 1}))
        h.mark_stop_requested(r.run_id)
        got = h.get(r.run_id)
        assert (got.current_dir_done, got.errors, got.stop_requested) == (4, 1, True)

"""Tests for compare/pipeline_run_control.py: checkpoints, resume seeds,
pending generate/scramble work and cancellation."""

import pytest

from compare import pipeline_run_control as prc
from compare.pipeline_run_control import (
    CANCEL_SHUTDOWN,
    CANCEL_USER,
    PipelineRunControl,
    RunCheckpoint,
)
from utils.path_utils import canonical_path


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(prc.time, "monotonic", c)
    return c


def _control(dirs, **kw):
    checkpoints, progress = [], []
    control = PipelineRunControl(
        "run1", dirs, checkpoint_sink=checkpoints.append, progress_sink=progress.append, **kw)
    return control, checkpoints, progress


def _done(control, path, n, stems=None):
    control.file_done(path, processed_stems=stems, files_evaluated=n, errors=0,
                      action_counts={"(no action)": n}, files_by_directory={})


def test_directory_boundaries_checkpoint_immediately(tmp_path, clock):
    d = str(tmp_path / "a")
    control, checkpoints, _ = _control([d])
    control.directory_started(d, 3)
    assert checkpoints[-1].current_directory == canonical_path(d)
    control.directory_finished(d)
    cp = checkpoints[-1]
    assert cp.completed_directories == [canonical_path(d)]
    assert cp.current_directory is None and cp.processed_files == []


def test_file_checkpoints_are_throttled(tmp_path, clock):
    d = str(tmp_path / "a")
    control, checkpoints, progress = _control([d])
    control.directory_started(d, 1000)
    start = len(checkpoints)
    for i in range(prc.CHECKPOINT_MIN_FILES):
        _done(control, str(tmp_path / "a" / f"{i}.png"), i + 1)
    # Enough files but not enough time.
    assert len(checkpoints) == start
    clock.now += prc.CHECKPOINT_MIN_SECONDS
    _done(control, str(tmp_path / "a" / "x.png"), 101)
    assert len(checkpoints) == start + 1
    assert len(checkpoints[-1].processed_files) == prc.CHECKPOINT_MIN_FILES + 1
    # Progress goes out on every file.
    assert progress[-1].current_dir_done == prc.CHECKPOINT_MIN_FILES + 1


def test_max_interval_checkpoints_a_slow_run(tmp_path, clock):
    d = str(tmp_path / "a")
    control, checkpoints, _ = _control([d])
    control.directory_started(d, 5)
    start = len(checkpoints)
    clock.now += prc.CHECKPOINT_MAX_SECONDS
    _done(control, str(tmp_path / "a" / "1.png"), 1)
    assert len(checkpoints) == start + 1


def test_seed_skips_processed_files_only_in_its_directory(tmp_path, clock):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    seed = RunCheckpoint(
        completed_directories=[canonical_path(a)],
        current_directory=canonical_path(b),
        processed_files=[canonical_path(str(tmp_path / "b" / "1.png"))],
        processed_stems=["stem1"],
        files_evaluated=7, errors=1, files_skipped=0,
        action_counts={"MOVE": 2}, files_by_directory={a: 6},
    )
    control, _, _ = _control([a, b], seed=seed)
    assert control.is_directory_completed(a + "/")
    assert not control.is_directory_completed(b)
    assert control.seed_processed_stems() == {"stem1"}
    assert control.seed_counters() == (7, 1, {"MOVE": 2}, {a: 6})

    control.directory_started(b, 2)
    assert control.is_file_processed(str(tmp_path / "b" / "1.png"))
    assert not control.is_file_processed(str(tmp_path / "b" / "2.png"))
    assert control.files_skipped == 1
    # The skipped file stays recorded as processed in this run's checkpoints.
    assert canonical_path(str(tmp_path / "b" / "1.png")) in control.snapshot().processed_files


def test_pending_generates_kept_until_flush(tmp_path, clock):
    d = str(tmp_path / "a")
    control, checkpoints, _ = _control([d])
    control.directory_started(d, 2)
    control.note_generate("p1", "_x", None)
    control.note_scramble("p2", "m")
    snap = control.snapshot()
    assert snap.pending_generates == [["p1", "_x", None]]
    assert snap.pending_scrambles == [["p2", "m"]]

    control.generates_flushed()
    start = len(checkpoints)
    _done(control, str(tmp_path / "a" / "1.png"), 1)
    # A flush makes the next file checkpoint regardless of throttling.
    assert len(checkpoints) == start + 1
    assert checkpoints[-1].pending_generates == []
    assert checkpoints[-1].pending_scrambles == [["p2", "m"]]


def test_carried_work_comes_from_the_seed():
    seed = RunCheckpoint(pending_generates=[["p", None, "t"]], pending_scrambles=[["q", None]])
    control = PipelineRunControl("r", [], seed=seed)
    assert control.carried_generates() == [("p", None, "t")]
    assert control.carried_scrambles() == [("q", None)]


def test_cancel_keeps_first_reason():
    control = PipelineRunControl("r", [])
    assert not control.is_cancelled()
    control.request_cancel(CANCEL_SHUTDOWN)
    control.request_cancel(CANCEL_USER)
    assert control.is_cancelled()
    assert control.cancel_reason == CANCEL_SHUTDOWN


def test_checkpoint_json_round_trip():
    cp = RunCheckpoint(completed_directories=["a"], current_directory="b",
                       processed_files=["b/1"], processed_stems=None,
                       files_evaluated=3, errors=1, files_skipped=2,
                       action_counts={"MOVE": 1}, files_by_directory={"a": 2},
                       pending_generates=[["p", None, None]], pending_scrambles=[],
                       updated_at=5.0)
    assert RunCheckpoint.from_json(cp.to_json()) == cp

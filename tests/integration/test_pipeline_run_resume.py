"""Stopping and resuming a real batch run (run_pipeline_over_directories with
a PipelineRunControl) over small PNGs.

The pipelines are inactive, so run_pipeline() returns None per file without
evaluating anything: these exercise the batch driver's skip/cancel/seed
handling rather than any classifier.
"""

import json

import pytest
from PIL import Image

from compare import classifier_pipeline_batch as pipeline_batch
from compare.classifier_pipeline import ClassifierPipeline
from compare.pipeline_run_control import PipelineRunControl


def _png(path) -> None:
    Image.new("RGB", (4, 4), (128, 128, 128)).save(path, format="PNG")


@pytest.fixture
def two_dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for i in range(3):
        _png(a / f"a{i}.png")
    for i in range(2):
        _png(b / f"b{i}.png")
    return [str(a), str(b)]


def _pipeline():
    p = ClassifierPipeline(name="ResumeBatch", is_active=False)
    p.nodes = []
    return p


def _cancelling_control(run_id, dirs, after_files, **kw):
    checkpoints = []
    holder = {}

    def progress(p):
        if p.files_evaluated >= after_files:
            holder["control"].request_cancel()

    control = PipelineRunControl(run_id, dirs, checkpoint_sink=checkpoints.append,
                                 progress_sink=progress, **kw)
    holder["control"] = control
    return control, checkpoints


def test_cancel_stops_after_the_current_file(two_dirs):
    control, checkpoints = _cancelling_control("r1", two_dirs, after_files=2)
    outcome = pipeline_batch.run_pipeline_over_directories(
        _pipeline(), two_dirs, write_dump=False, run_control=control)
    assert outcome.cancelled
    assert outcome.stats.cancelled
    assert outcome.stats.files_evaluated == 2
    final = checkpoints[-1]
    assert final.completed_directories == []
    assert len(final.processed_files) == 2


def test_resume_evaluates_exactly_the_rest(two_dirs):
    control, checkpoints = _cancelling_control("r1", two_dirs, after_files=2)
    pipeline_batch.run_pipeline_over_directories(
        _pipeline(), two_dirs, write_dump=False, run_control=control)

    resumed = PipelineRunControl("r2", two_dirs, seed=checkpoints[-1], resumed_from="r1")
    outcome = pipeline_batch.run_pipeline_over_directories(
        _pipeline(), two_dirs, write_dump=False, run_control=resumed)

    assert not outcome.cancelled
    # Counters are cumulative over the original and the resumed run.
    assert outcome.stats.files_evaluated == 5
    assert outcome.stats.files_skipped == 2
    assert outcome.stats.resumed_from == "r1"
    assert outcome.stats.files_by_directory == {two_dirs[0]: 3, two_dirs[1]: 2}


def test_completed_directories_are_not_gathered_again(two_dirs, monkeypatch):
    control, checkpoints = _cancelling_control("r1", two_dirs, after_files=4)
    pipeline_batch.run_pipeline_over_directories(
        _pipeline(), two_dirs, write_dump=False, run_control=control)
    assert len(checkpoints[-1].completed_directories) == 1

    from compare import base_compare
    gathered = []
    real_gather = base_compare.gather_files

    def recording_gather(directory, *a, **kw):
        gathered.append(directory)
        return real_gather(directory, *a, **kw)

    monkeypatch.setattr(base_compare, "gather_files", recording_gather)
    resumed = PipelineRunControl("r2", two_dirs, seed=checkpoints[-1])
    outcome = pipeline_batch.run_pipeline_over_directories(
        _pipeline(), two_dirs, write_dump=False, run_control=resumed)
    assert gathered == [two_dirs[1]]
    assert outcome.stats.files_evaluated == 5
    assert outcome.stats.files_skipped == 1


def test_without_run_control_results_are_unchanged(two_dirs):
    outcome = pipeline_batch.run_pipeline_over_directories(_pipeline(), two_dirs, write_dump=False)
    assert outcome.stats.files_evaluated == 5
    assert not outcome.cancelled
    assert outcome.stats.run_id is None
    assert outcome.dump_path is None


def test_dump_records_the_run(two_dirs):
    control, _checkpoints = _cancelling_control("r1", two_dirs, after_files=1, started_at=123.0)
    outcome = pipeline_batch.run_pipeline_over_directories(
        _pipeline(), two_dirs, run_control=control)
    assert outcome.dump_path is not None
    with open(outcome.dump_path, encoding="utf-8") as f:
        dump = json.load(f)
    assert dump["run"] == {"run_id": "r1", "status": "cancelled", "started_at": 123.0,
                           "resumed_from": None}
    assert dump["stats"]["cancelled"] is True
    assert dump["stats"]["run_id"] == "r1"

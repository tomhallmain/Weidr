"""A pipeline batch run works with no Qt at all.

Running a pipeline over directories is one of the things an automated caller
must be able to do without the GUI. The orchestration -- file ordering, the
cross-directory stem-group gate, the per-directory tallies and the run report
-- used to live inside the pipelines tab widget, so a headless caller could
only have reimplemented it. It now lives in compare.classifier_pipeline_batch,
which this drives directly.

The pipelines are inactive on purpose: run_pipeline() then returns None per
file without evaluating any condition, so these exercise the batch driver
itself rather than any classifier or model.
"""

import json

import pytest
from PIL import Image

from compare import classifier_pipeline_batch as pipeline_batch
from compare.classifier_pipeline import ClassifierPipeline


def _png(path) -> None:
    Image.new("RGB", (4, 4), (128, 128, 128)).save(path, format="PNG")


@pytest.fixture
def two_dirs(tmp_path):
    """Two directories, three PNGs and two, plus a file gather_files ignores."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for i in range(3):
        _png(a / f"a{i}.png")
    for i in range(2):
        _png(b / f"b{i}.png")
    (a / "notes.txt").write_text("ignored", encoding="utf-8")
    return a, b


def _pipeline(name="HeadlessBatch"):
    p = ClassifierPipeline(name=name, is_active=False)
    p.nodes = []
    return p


class TestHeadlessBatchRun:
    def test_no_qapplication_exists(self):
        # The premise of this file: if something pulled a QApplication into
        # being, it would stop proving anything.
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is not None:
            pytest.skip(
                "a QApplication already exists -- another package's tests ran "
                "first in this process, so the premise cannot be checked here"
            )

    def test_walks_every_directory_and_counts_files(self, two_dirs):
        a, b = two_dirs
        outcome = pipeline_batch.run_pipeline_over_directories(
            _pipeline(), [str(a), str(b)], write_dump=False,
        )
        assert outcome.stats.files_evaluated == 5
        assert outcome.stats.files_by_directory == {str(a): 3, str(b): 2}
        assert outcome.stats.errors == 0

    def test_non_image_files_are_not_evaluated(self, two_dirs):
        # notes.txt sits in a/ but is not an image type, so gather_files skips it.
        a, _b = two_dirs
        outcome = pipeline_batch.run_pipeline_over_directories(
            _pipeline(), [str(a)], write_dump=False,
        )
        assert outcome.stats.files_evaluated == 3

    def test_stats_carry_the_run_identity(self, two_dirs):
        a, _b = two_dirs
        outcome = pipeline_batch.run_pipeline_over_directories(
            _pipeline("NamedRun"), [str(a)],
            profile_name="SomeProfile", write_dump=False,
        )
        assert outcome.stats.pipeline_name == "NamedRun"
        assert outcome.stats.profile_name == "SomeProfile"
        assert outcome.stats.directories == [str(a)]
        assert isinstance(outcome.summary, str) and outcome.summary

    def test_inactive_pipeline_fires_no_actions(self, two_dirs):
        a, _b = two_dirs
        outcome = pipeline_batch.run_pipeline_over_directories(
            _pipeline(), [str(a)], write_dump=False,
        )
        assert outcome.generates == []
        assert outcome.scrambles == []
        assert set(outcome.stats.action_counts) == {"(no action)"}

    def test_empty_directory_list_is_harmless(self):
        outcome = pipeline_batch.run_pipeline_over_directories(
            _pipeline(), [], write_dump=False,
        )
        assert outcome.stats.files_evaluated == 0
        assert outcome.stats.files_by_directory == {}


class TestDumpIsOptional:
    """The run dump is read back by rerun-last, so both states must hold.

    No log-directory setup here: the autouse isolated_singletons fixture
    already redirects get_log_dir() to a per-test temp directory, precisely so
    a run dump cannot reach the real user log path.
    """

    def test_write_dump_false_writes_no_dump(self, two_dirs):
        # A headless caller running repeatedly should be able to opt out.
        a, _b = two_dirs
        pipeline = _pipeline("SkipDump")
        pipeline_batch.run_pipeline_over_directories(
            pipeline, [str(a)], write_dump=False,
        )
        assert pipeline_batch.find_latest_dump(pipeline) is None

    def test_write_dump_true_writes_a_readable_dump(self, two_dirs):
        a, _b = two_dirs
        pipeline = _pipeline("DumpMe")

        pipeline_batch.run_pipeline_over_directories(
            pipeline, [str(a)], write_dump=True,
        )

        dump = pipeline_batch.find_latest_dump(pipeline)
        assert dump is not None
        parsed = json.loads(dump.read_text(encoding="utf-8"))
        assert parsed["stats"]["pipeline_name"] == "DumpMe"
        assert parsed["stats"]["files_evaluated"] == 3

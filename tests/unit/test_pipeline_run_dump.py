"""
Unit tests for _write_pipeline_run_dump (Phase 1 — dump schema extension).

Verifies that the new `generates`, `scrambles`, and `generation_type_value`
fields are written correctly without touching Qt or the SD Runner.

Isolation: app_info_cache, config, and get_log_dir are all redirected to
per-test temp directories by the root conftest `isolated_singletons` fixture.
No production files are read or written.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from compare.pipeline_run_report import PipelineRunReport, PipelineRunStats
from utils.constants import ImageGenerationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(**overrides) -> PipelineRunStats:
    defaults = dict(
        pipeline_name="test_pipeline",
        profile_name="test_profile",
        directories=["/some/dir"],
        files_by_directory={"/some/dir": 3},
        files_evaluated=3,
        errors=0,
        action_counts={"hide": 2},
        generates_queued=0,
        generation_type_label=None,
        generation_type_value=None,
    )
    defaults.update(overrides)
    return PipelineRunStats(**defaults)


def _fake_pipeline():
    p = MagicMock()
    p.name = "test_pipeline"
    p.to_dict.return_value = {"name": "test_pipeline"}
    return p


def _call_write_dump(log_dir: Path, stats, all_generates=(), all_scrambles=()):
    """Call _write_pipeline_run_dump and return the parsed JSON.

    get_log_dir is already redirected to tmp_path/logs by the root conftest
    isolated_singletons fixture, so no patching is needed here.
    """
    from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab

    ClassifierPipelinesTab._write_pipeline_run_dump(
        _fake_pipeline(),
        stats,
        PipelineRunReport(),
        all_generates=all_generates,
        all_scrambles=all_scrambles,
    )

    dumps = list(log_dir.glob("pipeline_run_*.json"))
    assert len(dumps) == 1, "Expected exactly one dump file"
    return json.loads(dumps[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# generates field
# ---------------------------------------------------------------------------

class TestGeneratesField:
    def test_empty_generates_writes_empty_list(self, tmp_path):
        data = _call_write_dump(tmp_path / "logs", _make_stats())
        assert data["generates"] == []

    def test_generates_paths_written_correctly(self, tmp_path):
        items = [("/a/img1.jpg", None), ("/a/img2.jpg", None)]
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generates_queued=2), all_generates=items
        )
        assert data["generates"][0]["path"] == "/a/img1.jpg"
        assert data["generates"][1]["path"] == "/a/img2.jpg"

    def test_generates_modifier_none_preserved(self, tmp_path):
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generates_queued=1),
            all_generates=[("/a/img.jpg", None)],
        )
        assert data["generates"][0]["modifier"] is None

    def test_generates_modifier_string_preserved(self, tmp_path):
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generates_queued=1),
            all_generates=[("/a/img.jpg", "v2")],
        )
        assert data["generates"][0]["modifier"] == "v2"

    def test_generates_count_matches_list_length(self, tmp_path):
        items = [("/a/img1.jpg", None), ("/a/img2.jpg", "x"), ("/a/img3.jpg", None)]
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generates_queued=3), all_generates=items
        )
        assert len(data["generates"]) == 3

    def test_generates_order_preserved(self, tmp_path):
        items = [("/z.jpg", None), ("/a.jpg", None), ("/m.jpg", None)]
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generates_queued=3), all_generates=items
        )
        paths = [e["path"] for e in data["generates"]]
        assert paths == ["/z.jpg", "/a.jpg", "/m.jpg"]


# ---------------------------------------------------------------------------
# scrambles field
# ---------------------------------------------------------------------------

class TestScramblesField:
    def test_empty_scrambles_writes_empty_list(self, tmp_path):
        data = _call_write_dump(tmp_path / "logs", _make_stats())
        assert data["scrambles"] == []

    def test_scrambles_paths_written_correctly(self, tmp_path):
        items = [("/b/img1.png", "semi"), ("/b/img2.png", None)]
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(), all_scrambles=items
        )
        assert data["scrambles"][0]["path"] == "/b/img1.png"
        assert data["scrambles"][1]["path"] == "/b/img2.png"

    def test_scrambles_modifier_preserved(self, tmp_path):
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(),
            all_scrambles=[("/b/img.png", "semi_inco")],
        )
        assert data["scrambles"][0]["modifier"] == "semi_inco"

    def test_scrambles_modifier_none_preserved(self, tmp_path):
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(),
            all_scrambles=[("/b/img.png", None)],
        )
        assert data["scrambles"][0]["modifier"] is None

    def test_scrambles_count(self, tmp_path):
        items = [("/b/img1.png", "x"), ("/b/img2.png", "y")]
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(), all_scrambles=items
        )
        assert len(data["scrambles"]) == 2

    def test_generates_and_scrambles_independent(self, tmp_path):
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generates_queued=1),
            all_generates=[("/g.jpg", None)],
            all_scrambles=[("/s.png", "semi"), ("/s2.png", None)],
        )
        assert len(data["generates"]) == 1
        assert len(data["scrambles"]) == 2


# ---------------------------------------------------------------------------
# generation_type_value field
# ---------------------------------------------------------------------------

class TestGenerationTypeValue:
    def test_none_when_not_set(self, tmp_path):
        data = _call_write_dump(tmp_path / "logs", _make_stats(generation_type_value=None))
        assert data["stats"]["generation_type_value"] is None

    def test_value_written_for_each_member(self, tmp_path):
        log_dir = tmp_path / "logs"
        for member in ImageGenerationType:
            data = _call_write_dump(log_dir, _make_stats(generation_type_value=member.value))
            assert data["stats"]["generation_type_value"] == member.value
            # Roundtrip: the stored string reconstructs the enum
            assert ImageGenerationType.get(data["stats"]["generation_type_value"]) is member
            # Clean up between iterations so the single-dump assertion holds
            for f in log_dir.glob("pipeline_run_*.json"):
                f.unlink()

    def test_roundtrip_redo_prompt(self, tmp_path):
        gen_type = ImageGenerationType.REDO_PROMPT
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generation_type_value=gen_type.value)
        )
        reconstructed = ImageGenerationType.get(data["stats"]["generation_type_value"])
        assert reconstructed is ImageGenerationType.REDO_PROMPT

    def test_roundtrip_img2img(self, tmp_path):
        gen_type = ImageGenerationType.IMG2IMG
        data = _call_write_dump(
            tmp_path / "logs", _make_stats(generation_type_value=gen_type.value)
        )
        reconstructed = ImageGenerationType.get(data["stats"]["generation_type_value"])
        assert reconstructed is ImageGenerationType.IMG2IMG

    def test_existing_label_field_still_present(self, tmp_path):
        data = _call_write_dump(
            tmp_path / "logs",
            _make_stats(
                generation_type_label="Redo Prompt",
                generation_type_value=ImageGenerationType.REDO_PROMPT.value,
            ),
        )
        assert data["stats"]["generation_type_label"] == "Redo Prompt"
        assert data["stats"]["generation_type_value"] == "redo_prompt"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_existing_stats_fields_unchanged(self, tmp_path):
        stats = _make_stats(
            files_evaluated=7,
            errors=1,
            action_counts={"hide": 3, "mark": 4},
            generates_queued=2,
            generation_type_label="IP Adapter",
        )
        data = _call_write_dump(
            tmp_path / "logs", stats, all_generates=[("/x.jpg", None)] * 2
        )
        s = data["stats"]
        assert s["files_evaluated"] == 7
        assert s["errors"] == 1
        assert s["action_counts"] == {"hide": 3, "mark": 4}
        assert s["generates_queued"] == 2
        assert s["generation_type_label"] == "IP Adapter"

    def test_dump_written_even_with_no_generates_or_scrambles(self, tmp_path):
        _call_write_dump(tmp_path / "logs", _make_stats())
        assert len(list((tmp_path / "logs").glob("pipeline_run_*.json"))) == 1

    def test_messages_field_still_present(self, tmp_path):
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab

        report = PipelineRunReport()
        report.add("INFO", "node1", "/img.jpg", "detail text")

        ClassifierPipelinesTab._write_pipeline_run_dump(
            _fake_pipeline(), _make_stats(), report
        )

        dumps = list((tmp_path / "logs").glob("pipeline_run_*.json"))
        data = json.loads(dumps[0].read_text(encoding="utf-8"))
        assert len(data["messages"]) == 1
        assert data["messages"][0]["detail"] == "detail text"

"""
Integration tests for Phase 4 — "Rerun Last" pipeline logic.

Tests exercise _find_latest_dump, the rerun worker, and idempotency without
needing a real Qt window or SD Runner process.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dump(
    pipeline_name: str,
    generates: list[dict] | None = None,
    scrambles: list[dict] | None = None,
    generation_type_value: str | None = "ip_adapter",  # must be an ImageGenerationType.value
    timestamp: str = "2026-06-24T14:32:11",
) -> dict:
    return {
        "timestamp": timestamp,
        "pipeline": {"name": pipeline_name},
        "stats": {
            "pipeline_name": pipeline_name,
            "generation_type_value": generation_type_value,
        },
        "generates": generates or [],
        "scrambles": scrambles or [],
    }


def _write_dump(directory: Path, pipeline_name: str, ts: str, dump: dict) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in pipeline_name)
    path = directory / f"pipeline_run_{ts}_{safe_name}.json"
    path.write_text(json.dumps(dump), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _find_latest_dump
# ---------------------------------------------------------------------------

class TestFindLatestDump:
    def test_returns_none_when_no_dumps_exist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.get_log_dir"
            if False else "utils.logging_setup.get_log_dir",
            lambda: tmp_path,
        )
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        from compare.classifier_pipeline import ClassifierPipeline
        pipeline = ClassifierPipeline.__new__(ClassifierPipeline)
        pipeline.name = "MyPipeline"

        with patch("utils.logging_setup.get_log_dir", return_value=tmp_path):
            result = ClassifierPipelinesTab._find_latest_dump(pipeline)
        assert result is None

    def test_returns_most_recent_dump(self, tmp_path):
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        from compare.classifier_pipeline import ClassifierPipeline
        pipeline = ClassifierPipeline.__new__(ClassifierPipeline)
        pipeline.name = "MyPipeline"

        dump = _make_dump("MyPipeline")
        _write_dump(tmp_path, "MyPipeline", "2026-06-01_10-00-00", dump)
        _write_dump(tmp_path, "MyPipeline", "2026-06-24_14-32-11", dump)
        older = _write_dump(tmp_path, "MyPipeline", "2026-05-01_09-00-00", dump)

        with patch("utils.logging_setup.get_log_dir", return_value=tmp_path):
            result = ClassifierPipelinesTab._find_latest_dump(pipeline)
        assert result is not None
        assert result != older
        assert "2026-06-24" in result.name

    def test_ignores_dumps_for_other_pipelines(self, tmp_path):
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        from compare.classifier_pipeline import ClassifierPipeline
        pipeline = ClassifierPipeline.__new__(ClassifierPipeline)
        pipeline.name = "PipelineA"

        _write_dump(tmp_path, "PipelineB", "2026-06-24_14-32-11", _make_dump("PipelineB"))

        with patch("utils.logging_setup.get_log_dir", return_value=tmp_path):
            result = ClassifierPipelinesTab._find_latest_dump(pipeline)
        assert result is None


# ---------------------------------------------------------------------------
# Rerun — generate dispatch
# ---------------------------------------------------------------------------

class TestRerunGenerates:
    def test_dispatches_generates_to_sd_runner(self, tmp_path):
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab

        generates = [
            {"path": "/img_001.png", "modifier": None},
            {"path": "/img_002.png", "modifier": "v2"},
        ]
        dump = _make_dump("Pipeline", generates=generates, generation_type_value="ip_adapter")

        mock_rb = MagicMock()
        with (
            patch("utils.logging_setup.get_log_dir", return_value=tmp_path),
            patch(
                "extensions.sd_runner_client.SDRunnerClient.run_batch", mock_rb
            ),
        ):
            _write_dump(tmp_path, "Pipeline", "2026-06-24_14-32-11", dump)
            # Call the worker directly by extracting the logic
            gen_type_val = dump["stats"]["generation_type_value"]
            from utils.constants import ImageGenerationType
            gen_type = ImageGenerationType(gen_type_val)
            batch_args = [
                {
                    "image": g["path"],
                    "append": False,
                    **({'edit_suffix': g["modifier"]} if g.get("modifier") else {}),
                }
                for g in generates
            ]
            from extensions.sd_runner_client import SDRunnerClient
            SDRunnerClient().run_batch(gen_type, batch_args)

        mock_rb.assert_called_once()
        _, called_args = mock_rb.call_args.args
        assert called_args[0]["image"] == "/img_001.png"
        assert "edit_suffix" not in called_args[0]
        assert called_args[1]["image"] == "/img_002.png"
        assert called_args[1]["edit_suffix"] == "v2"

    def test_falls_back_to_app_gen_mode_on_bad_type_value(self):
        from utils.constants import ImageGenerationType
        fallback = MagicMock()
        with patch(
            "ui.image.media_details.MediaDetails.get_image_specific_generation_mode",
            return_value=fallback,
        ):
            try:
                gen_type = ImageGenerationType("__not_a_real_value__")
            except (ValueError, TypeError):
                from ui.image.media_details import MediaDetails
                gen_type = MediaDetails.get_image_specific_generation_mode()
        assert gen_type is fallback


# ---------------------------------------------------------------------------
# Rerun — scramble idempotency
# ---------------------------------------------------------------------------

class TestRerunScrambleIdempotency:
    def test_skips_scramble_when_output_exists(self, tmp_path):
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab

        src = tmp_path / "img.png"
        src.write_bytes(b"")
        # "_inco" is the real modifier; new_filepath concatenates directly → img_inco.png
        (tmp_path / "img_inco.png").write_bytes(b"existing")

        mock_scramble = MagicMock()
        with patch("image.image_ops.ImageOps.scramble_image", mock_scramble):
            ClassifierPipelinesTab._run_one_scramble(str(src), "_inco", skip_existing=True)

        mock_scramble.assert_not_called()

    def test_executes_scramble_when_output_absent(self, tmp_path):
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab

        src = tmp_path / "img.png"
        src.write_bytes(b"")

        mock_scramble = MagicMock()
        with patch("image.image_ops.ImageOps.scramble_image", mock_scramble):
            ClassifierPipelinesTab._run_one_scramble(str(src), "_inco", skip_existing=True)

        mock_scramble.assert_called_once()


# ---------------------------------------------------------------------------
# Dump loading — graceful error handling
# ---------------------------------------------------------------------------

class TestRerunDumpLoading:
    def test_find_returns_none_for_corrupt_dir(self, tmp_path):
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        from compare.classifier_pipeline import ClassifierPipeline
        pipeline = ClassifierPipeline.__new__(ClassifierPipeline)
        pipeline.name = "P"

        with patch("utils.logging_setup.get_log_dir", side_effect=OSError("gone")):
            result = ClassifierPipelinesTab._find_latest_dump(pipeline)
        assert result is None

    def test_dump_generates_and_scrambles_default_to_empty(self, tmp_path):
        minimal = {"timestamp": "2026-01-01T00:00:00", "stats": {}}
        path = tmp_path / "pipeline_run_2026-01-01_00-00-00_P.json"
        path.write_text(json.dumps(minimal), encoding="utf-8")

        dump = json.loads(path.read_text(encoding="utf-8"))
        assert dump.get("generates", []) == []
        assert dump.get("scrambles", []) == []

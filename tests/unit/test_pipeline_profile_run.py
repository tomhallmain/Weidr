"""Tests for compare/pipeline_profile_run.py: running a pipeline over a
directory profile the way the Pipelines tab's Run on Profile does.

run_pipeline_over_directories is replaced by a recorder, so no media is
classified; runs still go through a real ThreadedTaskRunner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from compare import pipeline_profile_run as ppr
from compare.classifier_pipeline import ClassifierPipeline, ClassifierPipelines, PrevalidationPipeline
from compare.pipeline_run_report import PipelineRunStats
from files.directory_profile import DirectoryProfile
from utils.constants import ImageGenerationType


class _Pipeline:
    """Duck-typed pipeline for run tests."""

    def __init__(self, name="Sorter", generation_type=None, generates=False):
        self.name = name
        self.is_active = True
        self.generation_type = generation_type
        self._generates = generates

    def has_generate_action(self):
        return self._generates


@pytest.fixture
def installed(monkeypatch):
    def install(pipelines=(), profiles=()):
        monkeypatch.setattr(ClassifierPipelines, "pipelines", list(pipelines))
        monkeypatch.setattr(DirectoryProfile, "directory_profiles", list(profiles))
    return install


@pytest.fixture
def batch_calls(monkeypatch):
    calls = []

    def fake_run(pipeline, directories, **kwargs):
        calls.append({"pipeline": pipeline, "directories": list(directories), **kwargs})
        stats = PipelineRunStats(pipeline_name=pipeline.name, files_evaluated=3,
                                 action_counts={"(no action)": 3})
        return SimpleNamespace(stats=stats, summary="3 files evaluated")

    monkeypatch.setattr(ppr.pipeline_batch, "run_pipeline_over_directories", fake_run)
    return calls


def _start(runs, pipeline_name="Sorter", profile_name="Photos", **overrides):
    kwargs = dict(
        continue_without_sd_runner=False,
        fallback_generation_type=ImageGenerationType.IP_ADAPTER,
        hide_callback=None, notify_callback=None, blur_callback=None,
    )
    kwargs.update(overrides)
    result = runs.start(pipeline_name, profile_name, **kwargs)
    assert runs._runner.wait(5)
    return result


def test_run_goes_over_the_profile_directories_and_records_the_outcome(installed, batch_calls):
    from utils.app_info_cache import app_info_cache

    installed(pipelines=[_Pipeline()], profiles=[DirectoryProfile("Photos", ["/p/a", "/p/b"])])
    notified = []
    runs = ppr.PipelineProfileRuns()

    started = _start(runs, notify_callback=notified.append)

    assert started == {"status": "started", "pipeline": "Sorter", "profile": "Photos",
                       "directories": ["/p/a", "/p/b"], "warnings": []}
    (call,) = batch_calls
    assert call["directories"] == ["/p/a", "/p/b"]
    assert call["profile_name"] == "Photos"
    assert call["generation_type"] == ImageGenerationType.IP_ADAPTER
    status = runs.status()
    assert status["running"] is False
    assert status["stats"]["files_evaluated"] == 3
    assert status["summary"] == "3 files evaluated"
    assert status["error"] is None
    assert notified[-1] == "3 files evaluated"  # the completion notice the tab also sends
    assert app_info_cache.get_meta(ppr.last_profile_meta_key("Sorter"), "") == "Photos"


def test_profile_defaults_to_the_one_selected_in_the_pipelines_tab(installed, batch_calls):
    from utils.app_info_cache import app_info_cache

    installed(pipelines=[_Pipeline()], profiles=[DirectoryProfile("Videos", ["/v"])])
    app_info_cache.set_meta(ppr.SELECTED_PROFILE_META_KEY, "Videos")
    started = _start(ppr.PipelineProfileRuns(), profile_name=None)
    assert started["profile"] == "Videos"
    assert batch_calls[0]["directories"] == ["/v"]


def test_pipeline_generation_type_wins_over_the_fallback(installed, batch_calls):
    installed(pipelines=[_Pipeline(generation_type=ImageGenerationType.REDO_PROMPT)],
              profiles=[DirectoryProfile("Photos", ["/p"])])
    _start(ppr.PipelineProfileRuns())
    assert batch_calls[0]["generation_type"] == ImageGenerationType.REDO_PROMPT


def test_run_error_is_recorded(installed, monkeypatch):
    def failing_run(pipeline, directories, **kwargs):
        raise RuntimeError("classifier missing")

    monkeypatch.setattr(ppr.pipeline_batch, "run_pipeline_over_directories", failing_run)
    installed(pipelines=[_Pipeline()], profiles=[DirectoryProfile("Photos", ["/p"])])
    runs = ppr.PipelineProfileRuns()
    _start(runs)
    status = runs.status()
    assert "classifier missing" in status["error"]
    assert status["stats"] is None
    assert status["finished_at"] is not None


def test_generate_pipeline_is_refused_while_sd_runner_is_unreachable(installed, batch_calls, monkeypatch):
    from extensions import sd_runner_client

    monkeypatch.setattr(sd_runner_client.SDRunnerClient, "is_reachable", staticmethod(lambda: False))
    installed(pipelines=[_Pipeline(generates=True)], profiles=[DirectoryProfile("Photos", ["/p"])])
    runs = ppr.PipelineProfileRuns()
    with pytest.raises(ValueError):
        runs.start("Sorter", "Photos", continue_without_sd_runner=False,
                   fallback_generation_type=None, hide_callback=None,
                   notify_callback=None, blur_callback=None)
    assert batch_calls == []

    started = _start(runs, continue_without_sd_runner=True)
    assert started["warnings"]
    assert len(batch_calls) == 1


@pytest.mark.parametrize("pipeline_name, profile_name", [
    ("Missing", "Photos"),
    ("Sorter", "Missing"),
    ("Sorter", None),  # nothing selected in the tab either
])
def test_unknown_or_missing_names_raise_value_error(installed, batch_calls, pipeline_name, profile_name):
    installed(pipelines=[_Pipeline()], profiles=[DirectoryProfile("Photos", ["/p"])])
    with pytest.raises(ValueError):
        ppr.PipelineProfileRuns().start(
            pipeline_name, profile_name, continue_without_sd_runner=False,
            fallback_generation_type=None, hide_callback=None,
            notify_callback=None, blur_callback=None,
        )
    assert batch_calls == []


def test_second_run_is_refused_while_one_is_running(installed, batch_calls, monkeypatch):
    installed(pipelines=[_Pipeline()], profiles=[DirectoryProfile("Photos", ["/p"])])
    runs = ppr.PipelineProfileRuns()
    monkeypatch.setattr(runs._runner, "is_running", lambda: True)
    with pytest.raises(ValueError):
        runs.start("Sorter", "Photos", continue_without_sd_runner=False,
                   fallback_generation_type=None, hide_callback=None,
                   notify_callback=None, blur_callback=None)


def test_list_pipelines_and_profiles(installed):
    installed(
        pipelines=[ClassifierPipeline(name="Sorter"), PrevalidationPipeline(name="Gate", is_active=False)],
        profiles=[DirectoryProfile("Photos", ["/p"])],
    )
    listing = ppr.list_pipelines()
    assert listing["pipelines"] == [
        {"name": "Sorter", "is_active": True, "kind": "action", "last_profile": None},
        {"name": "Gate", "is_active": False, "kind": "prevalidation", "last_profile": None},
    ]
    assert listing["selected_profile"] is None
    assert ppr.list_directory_profiles() == {"profiles": [{"name": "Photos", "directories": ["/p"]}]}


def test_loaded_pipelines_are_not_reloaded(installed, monkeypatch):
    """A loaded list is current; reloading would replace objects the GUI holds."""
    installed(pipelines=[ClassifierPipeline(name="Sorter")])
    loads = []
    monkeypatch.setattr(ClassifierPipelines, "load", staticmethod(lambda: loads.append(True)))
    ppr.ensure_pipelines_loaded()
    assert loads == []


def test_image_specific_generation_type():
    assert ImageGenerationType.image_specific(ImageGenerationType.IP_ADAPTER) == ImageGenerationType.IP_ADAPTER
    assert ImageGenerationType.image_specific(ImageGenerationType.RENOISER) == ImageGenerationType.CONTROL_NET

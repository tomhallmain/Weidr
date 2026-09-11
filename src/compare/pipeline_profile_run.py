"""Run a classifier pipeline over a directory profile, the way the Pipelines
tab's "Run on Profile" button does, without its dialogs.

The tab asks two questions before a run: a confirmation, which a caller
here has already answered by asking for the run, and "SD Runner is not
reachable, continue anyway?", which the caller answers up front with
continue_without_sd_runner. Everything else follows the tab: the same
profile, directories, last-profile record, generation type fallback, batch
call and completion notification.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Callable, Optional

from compare import classifier_pipeline_batch as pipeline_batch
from compare.classifier_pipeline import ClassifierPipelines, PrevalidationPipeline
from files.directory_profile import DirectoryProfile
from utils.app_info_cache import app_info_cache
from utils.background_runner import ThreadedTaskRunner
from utils.logging_setup import get_logger

logger = get_logger("pipeline_profile_run")

# The profile selected in the Pipelines tab's "Run on profile" combo.
SELECTED_PROFILE_META_KEY = "classifier_pipelines_profile"


def last_profile_meta_key(pipeline_name: str) -> str:
    """Where the profile a pipeline was last run on is recorded."""
    return f"pipeline_last_profile:{pipeline_name}"


def ensure_pipelines_loaded() -> None:
    """Load pipelines only if none are loaded yet.

    A loaded list is already current (edits in the GUI update it and store
    right away), and ClassifierPipelines.load() replaces every object in it,
    including ones the GUI holds references to.
    """
    if not ClassifierPipelines.get_all_pipelines():
        ClassifierPipelines.load()


def list_pipelines() -> dict:
    ensure_pipelines_loaded()
    return {
        "pipelines": [
            {
                "name": p.name,
                "is_active": p.is_active,
                "kind": "prevalidation" if isinstance(p, PrevalidationPipeline) else "action",
                "last_profile": app_info_cache.get_meta(last_profile_meta_key(p.name), "") or None,
            }
            for p in ClassifierPipelines.get_all_pipelines()
        ],
        "selected_profile": app_info_cache.get_meta(SELECTED_PROFILE_META_KEY, "") or None,
    }


def list_directory_profiles() -> dict:
    return {
        "profiles": [
            {"name": p.name, "directories": list(p.directories)}
            for p in DirectoryProfile.directory_profiles
        ],
    }


class PipelineProfileRuns:
    """At most one pipeline run started this way at a time, and the latest
    run's outcome. The Pipelines tab starts its runs independently."""

    def __init__(self) -> None:
        self._runner = ThreadedTaskRunner()
        self._lock = threading.Lock()          # guards _status
        self._start_lock = threading.Lock()    # makes check-then-start atomic
        self._status: dict = {"running": False, "pipeline": None}

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "running": self._runner.is_running()}

    def start(
        self, pipeline_name: str, profile_name: Optional[str], *,
        continue_without_sd_runner: bool,
        fallback_generation_type,
        hide_callback: Optional[Callable],
        notify_callback: Optional[Callable],
        blur_callback: Optional[Callable],
    ) -> dict:
        """Start the run on a background thread and return what was started.

        *profile_name* defaults to the Pipelines tab's selected profile.
        *fallback_generation_type* is used when the pipeline has none of its
        own; the caller resolves it (MediaDetails.get_image_specific_generation_mode()
        in the GUI). Raises ValueError when the run can't start.
        """
        with self._start_lock:
            return self._start(
                pipeline_name, profile_name,
                continue_without_sd_runner=continue_without_sd_runner,
                fallback_generation_type=fallback_generation_type,
                hide_callback=hide_callback, notify_callback=notify_callback,
                blur_callback=blur_callback,
            )

    def _start(
        self, pipeline_name, profile_name, *, continue_without_sd_runner,
        fallback_generation_type, hide_callback, notify_callback, blur_callback,
    ) -> dict:
        from files.marked_files import MarkedFiles

        if self._runner.is_running():
            raise ValueError("a pipeline run is already in progress")
        ensure_pipelines_loaded()
        pipeline = ClassifierPipelines.get_pipeline_by_name(pipeline_name)
        if pipeline is None:
            raise ValueError(f"no pipeline named {pipeline_name!r}")
        if not profile_name:
            profile_name = app_info_cache.get_meta(SELECTED_PROFILE_META_KEY, "") or None
            if not profile_name:
                raise ValueError("no profile_name given and none is selected in the Pipelines tab")
        profile = DirectoryProfile.get_profile_by_name(profile_name)
        if profile is None:
            raise ValueError(f"no directory profile named {profile_name!r}")
        directories = list(profile.directories)

        warnings = []
        if pipeline.has_generate_action():
            from extensions.sd_runner_client import SDRunnerClient
            if not SDRunnerClient.is_reachable():
                if not continue_without_sd_runner:
                    raise ValueError(
                        f"pipeline {pipeline_name!r} has GENERATE actions but SD Runner is not "
                        "reachable; pass continue_without_sd_runner=true to run anyway"
                    )
                warnings.append(
                    "SD Runner is not reachable: generated images won't be produced "
                    "unless it starts before the run completes"
                )

        generation_type = (
            pipeline.generation_type if pipeline.generation_type is not None
            else fallback_generation_type
        )
        app_info_cache.set_meta(last_profile_meta_key(pipeline.name), profile_name)

        def run() -> None:
            outcome = pipeline_batch.run_pipeline_over_directories(
                pipeline, directories,
                generation_type=generation_type,
                profile_name=profile_name,
                hide_callback=hide_callback,
                notify_callback=notify_callback,
                add_mark_callback=MarkedFiles.add_mark_if_not_present,
                blur_callback=blur_callback,
            )
            with self._lock:
                self._status["stats"] = dataclasses.asdict(outcome.stats)
                self._status["summary"] = outcome.summary
            if notify_callback is not None:
                try:
                    notify_callback(outcome.summary)
                except Exception:
                    logger.exception("Pipeline completion notification failed")

        def on_error(message: str) -> None:
            with self._lock:
                self._status["error"] = message

        def on_finished() -> None:
            with self._lock:
                self._status["finished_at"] = time.time()

        started = {
            "pipeline": pipeline.name,
            "profile": profile_name,
            "directories": directories,
            "warnings": warnings,
        }
        with self._lock:
            self._status = {
                **started, "started_at": time.time(), "finished_at": None,
                "stats": None, "summary": None, "error": None,
            }
        self._runner.start(run, on_error=on_error, on_finished=on_finished)
        return {"status": "started", **started}


# One per process: pipelines and profiles are process-wide, so is the run.
pipeline_runs = PipelineProfileRuns()

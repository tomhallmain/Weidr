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


def history_warnings(pipeline_name: str, directories) -> list:
    """English notes on earlier and running runs that touch *directories*."""
    from compare import pipeline_run_service
    from compare.pipeline_run_history import format_run_time

    plan = pipeline_run_service.plan_run(pipeline_name, directories)
    notes = []
    last = plan.last_run
    if last is not None:
        where = "these directories" if plan.last_run_same_set else "overlapping directories"
        notes.append(
            f"pipeline {pipeline_name!r} last ran on {where} at {format_run_time(last.started_at)} "
            f"({last.status.value}, profile {last.profile_name!r}, run {last.run_id})"
        )
    for r in plan.running:
        notes.append(
            f"pipeline {r.pipeline_name!r} is running on overlapping directories "
            f"(run {r.run_id}, started {format_run_time(r.started_at)})"
        )
    if plan.resumable is not None:
        notes.append(
            f"run {plan.resumable.run_id} on these directories did not finish "
            f"({plan.resumable.status.value}); resume_pipeline can continue it"
        )
    return notes


class PipelineProfileRuns:
    """At most one pipeline run started this way at a time, and the latest
    run's outcome. Runs go through pipeline_run_service like the Pipelines
    tab's, so they share one history and refuse to start on directories a
    running run (from either) overlaps."""

    def __init__(self) -> None:
        self._runner = ThreadedTaskRunner()
        self._lock = threading.Lock()          # guards _status
        self._start_lock = threading.Lock()    # makes check-then-start atomic
        self._status: dict = {"running": False, "pipeline": None, "run_id": None}

    def status(self) -> dict:
        with self._lock:
            status = {**self._status, "running": self._runner.is_running()}
        run_id = status.get("run_id")
        if run_id:
            from compare.pipeline_run_history import pipeline_run_history
            record = pipeline_run_history.get(run_id)
            status["progress"] = record.to_public_dict() if record is not None else None
        return status

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

    def resume(
        self, run_id: Optional[str], *,
        continue_without_sd_runner: bool,
        fallback_generation_type,
        hide_callback: Optional[Callable],
        notify_callback: Optional[Callable],
        blur_callback: Optional[Callable],
    ) -> dict:
        """Continue the unfinished run *run_id* (default: the newest
        resumable one) over its recorded directories, skipping the files it
        already processed. Raises ValueError when it can't be resumed."""
        from compare import pipeline_run_service
        from compare.pipeline_run_history import pipeline_run_history

        with self._start_lock:
            if self._runner.is_running():
                raise ValueError("a pipeline run is already in progress")
            ensure_pipelines_loaded()
            pipeline_run_history.sync_from_disk()
            if not run_id:
                latest = pipeline_run_history.latest_resumable()
                if latest is None:
                    raise ValueError("no unfinished run to resume")
                run_id = latest.run_id
            check = pipeline_run_service.check_resume(run_id)
            if check.problem:
                raise ValueError(f"run {run_id!r} cannot be resumed: {check.problem}")
            if check.pipeline_changed:
                raise ValueError(
                    f"pipeline {check.record.pipeline_name!r} was edited since run {run_id!r}; "
                    "start a new run instead"
                )
            pipeline = check.pipeline
            errors = pipeline.validate()
            if errors:
                raise pipeline_batch.PipelineValidationError(pipeline.name, errors)
            warnings = [f"directory no longer exists: {d}" for d in check.missing_directories]
            return self._launch(
                pipeline, check.record.profile_name, list(check.record.directories),
                warnings=warnings, resume_from=run_id,
                continue_without_sd_runner=continue_without_sd_runner,
                fallback_generation_type=fallback_generation_type,
                hide_callback=hide_callback, notify_callback=notify_callback,
                blur_callback=blur_callback,
            )

    def stop(self, run_id: Optional[str] = None) -> dict:
        """Ask a run active in this process to stop after its current file.
        *run_id* defaults to the run started here last, else the only
        active run. Raises ValueError when there is nothing to stop."""
        from compare import pipeline_run_service

        if not run_id:
            with self._lock:
                run_id = self._status.get("run_id") if self._runner.is_running() else None
            if not run_id:
                active = pipeline_run_service.active_run_ids()
                if len(active) != 1:
                    raise ValueError(
                        "no pipeline run is active" if not active
                        else "several pipeline runs are active; pass run_id"
                    )
                run_id = active[0]
        if not pipeline_run_service.request_stop(run_id):
            raise ValueError(f"run {run_id!r} is not active in this process")
        return {"status": "stopping", "run_id": run_id}

    def history(self, limit: int = 50) -> dict:
        from compare.pipeline_run_history import pipeline_run_history

        pipeline_run_history.sync_from_disk()
        records = pipeline_run_history.records()[:max(0, int(limit))]
        return {"runs": [r.to_public_dict() for r in records]}

    def _start(
        self, pipeline_name, profile_name, *, continue_without_sd_runner,
        fallback_generation_type, hide_callback, notify_callback, blur_callback,
    ) -> dict:
        if self._runner.is_running():
            raise ValueError("a pipeline run is already in progress")
        ensure_pipelines_loaded()
        pipeline = ClassifierPipelines.get_pipeline_by_name(pipeline_name)
        if pipeline is None:
            raise ValueError(f"no pipeline named {pipeline_name!r}")
        # Checked here too so the caller hears of it, not only the run thread.
        errors = pipeline.validate()
        if errors:
            raise pipeline_batch.PipelineValidationError(pipeline.name, errors)
        if not profile_name:
            profile_name = app_info_cache.get_meta(SELECTED_PROFILE_META_KEY, "") or None
            if not profile_name:
                raise ValueError("no profile_name given and none is selected in the Pipelines tab")
        profile = DirectoryProfile.get_profile_by_name(profile_name)
        if profile is None:
            raise ValueError(f"no directory profile named {profile_name!r}")
        directories = list(profile.directories)
        return self._launch(
            pipeline, profile_name, directories,
            warnings=history_warnings(pipeline.name, directories), resume_from=None,
            continue_without_sd_runner=continue_without_sd_runner,
            fallback_generation_type=fallback_generation_type,
            hide_callback=hide_callback, notify_callback=notify_callback,
            blur_callback=blur_callback,
        )

    def _launch(
        self, pipeline, profile_name, directories, *, warnings, resume_from,
        continue_without_sd_runner, fallback_generation_type,
        hide_callback, notify_callback, blur_callback,
    ) -> dict:
        from compare import pipeline_run_service
        from files.marked_files import MarkedFiles

        warnings = list(warnings)
        if pipeline.has_generate_action():
            from extensions.sd_runner_client import SDRunnerClient
            if not SDRunnerClient.is_reachable():
                if not continue_without_sd_runner:
                    raise ValueError(
                        f"pipeline {pipeline.name!r} has GENERATE actions but SD Runner is not "
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
        # RunRefused is a ValueError: overlap with a running run.
        active = pipeline_run_service.begin_run(
            pipeline, directories, profile_name=profile_name, source="mcp",
            resume_from=resume_from,
        )
        if profile_name:
            app_info_cache.set_meta(last_profile_meta_key(pipeline.name), profile_name)

        def run() -> None:
            outcome = pipeline_run_service.execute_run(
                active,
                generation_type=generation_type,
                hide_callback=hide_callback,
                notify_callback=notify_callback,
                add_mark_callback=MarkedFiles.add_mark_if_not_present,
                blur_callback=blur_callback,
            )
            with self._lock:
                self._status["stats"] = dataclasses.asdict(outcome.stats)
                self._status["summary"] = outcome.summary
                self._status["cancelled"] = getattr(outcome, "cancelled", False)
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
            "run_id": active.run_id,
            "pipeline": pipeline.name,
            "profile": profile_name,
            "directories": directories,
            "warnings": warnings,
        }
        if resume_from:
            started["resumed_from"] = resume_from
        with self._lock:
            self._status = {
                **started, "started_at": time.time(), "finished_at": None,
                "stats": None, "summary": None, "error": None, "cancelled": False,
            }
        try:
            self._runner.start(run, on_error=on_error, on_finished=on_finished)
        except Exception as e:
            pipeline_run_service.abandon(active, str(e))
            raise
        return {"status": "started", **started}


# One per process: pipelines and profiles are process-wide, so is the run.
pipeline_runs = PipelineProfileRuns()

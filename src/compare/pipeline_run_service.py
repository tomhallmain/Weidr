"""Start, stop and resume batch pipeline runs, for the GUI and MCP alike.

Every batch run goes through begin_run() and execute_run(), so each one is
recorded in pipeline_run_history and can be stopped from anywhere in this
process. The caller owns the worker thread: begin_run() registers the run
synchronously (and may refuse it), execute_run() is the blocking body the
caller's thread runs.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from compare import classifier_pipeline_batch as pipeline_batch
from compare import pipeline_run_history as prh
from compare.pipeline_run_control import (
    CANCEL_SHUTDOWN,
    CANCEL_USER,
    PipelineRunControl,
)
from compare.pipeline_run_history import PipelineRunRecord, RunStatus, pipeline_fingerprint
from utils.logging_setup import get_logger
from utils.path_utils import overlapping_pairs, same_directory_set
from utils.translations import _

logger = get_logger("pipeline_run_service")


class RunRefused(ValueError):
    """A run that must not start (overlap with a running run, nothing to
    resume, ...). The message is English, like the MCP layer's errors."""


@dataclass
class RunPlan:
    """What the history says about running *pipeline* on some directories."""

    last_run: Optional[PipelineRunRecord] = None
    last_run_same_set: bool = False
    last_run_overlap: List[Tuple[str, str]] = field(default_factory=list)
    running: List[PipelineRunRecord] = field(default_factory=list)
    resumable: Optional[PipelineRunRecord] = None


@dataclass
class ResumeCheck:
    record: Optional[PipelineRunRecord]
    pipeline: object = None
    pipeline_changed: bool = False
    missing_directories: List[str] = field(default_factory=list)
    # The profile of that name is gone or now lists other directories; the
    # run's recorded directories are used either way.
    profile_changed: bool = False
    problem: Optional[str] = None


@dataclass
class ActiveRun:
    run_id: str
    pipeline: object
    directories: List[str]
    profile_name: Optional[str]
    control: PipelineRunControl
    done: threading.Event = field(default_factory=threading.Event)


_registry_lock = threading.RLock()
_active: Dict[str, ActiveRun] = {}


def _history():
    # Looked up per call so a test can swap the singleton.
    return prh.pipeline_run_history


def plan_run(pipeline_name: str, directories: Sequence[str]) -> RunPlan:
    history = _history()
    history.sync_from_disk()
    directories = list(directories)
    plan = RunPlan(
        running=history.running_overlapping(directories),
        resumable=history.latest_resumable(pipeline_name, directories),
    )
    last = history.last_overlapping(pipeline_name, directories)
    if last is not None:
        plan.last_run = last
        plan.last_run_same_set = same_directory_set(last.directories, directories)
        if not plan.last_run_same_set:
            plan.last_run_overlap = overlapping_pairs(directories, last.directories)
    return plan


def begin_run(
    pipeline,
    directories: Sequence[str],
    *,
    profile_name: Optional[str],
    source: str,
    resume_from: Optional[str] = None,
    allow_running_overlap: bool = False,
) -> ActiveRun:
    """Record and register a run of *pipeline* over *directories*.

    With *resume_from*, the new run continues that run's checkpoint; pass
    that run's recorded directories. Raises RunRefused when the directories
    overlap a running run (unless *allow_running_overlap*) or when
    *resume_from* can't be resumed.
    """
    history = _history()
    directories = [str(d) for d in directories]
    with _registry_lock:
        history.sync_from_disk()
        if not allow_running_overlap:
            running = history.running_overlapping(directories)
            if running:
                r = running[0]
                raise RunRefused(
                    f"pipeline {r.pipeline_name!r} is already running on overlapping "
                    f"directories (run {r.run_id})"
                )
        seed = None
        if resume_from is not None:
            original = history.get(resume_from)
            if original is None or not original.is_resumable:
                raise RunRefused(f"run {resume_from!r} cannot be resumed")
            seed = history.read_checkpoint(resume_from)
            if seed is None:
                raise RunRefused(f"run {resume_from!r} has no checkpoint to resume from")

        record = PipelineRunRecord.new(
            pipeline.name, directories,
            profile_name=profile_name, source=source, owner=history.owner,
            pipeline_fingerprint=pipeline_fingerprint(pipeline),
            resumed_from=resume_from,
        )
        if seed is not None:
            record.files_evaluated = seed.files_evaluated
            record.files_skipped = seed.files_skipped
            record.errors = seed.errors
            record.action_counts = dict(seed.action_counts)
            record.directories_done = len(seed.completed_directories)
        run_id = record.run_id
        control = PipelineRunControl(
            run_id, directories, seed=seed,
            checkpoint_sink=lambda cp: history.write_checkpoint(run_id, cp),
            progress_sink=lambda progress: history.update_progress(run_id, progress),
            started_at=record.started_at,
            resumed_from=resume_from,
        )
        history.add_run(record)
        if seed is not None:
            # The new run holds the checkpoint before the original lets go of it.
            history.write_checkpoint(run_id, seed)
            history.link_resume(resume_from, run_id)
        active = ActiveRun(run_id=run_id, pipeline=pipeline, directories=directories,
                           profile_name=profile_name, control=control)
        _active[run_id] = active
    logger.info("Pipeline %r: run %s started on %d director%s%s", pipeline.name, run_id,
                len(directories), "y" if len(directories) == 1 else "ies",
                f" (resuming {resume_from})" if resume_from else "")
    return active


def execute_run(
    active: ActiveRun,
    *,
    generation_type=None,
    hide_callback: Optional[Callable] = None,
    notify_callback: Optional[Callable] = None,
    add_mark_callback: Optional[Callable] = None,
    blur_callback: Optional[Callable] = None,
):
    """Run the batch for *active* and record how it ended. Blocking.
    Returns the batch outcome; re-raises what the batch raised."""
    history = _history()
    control = active.control
    try:
        outcome = pipeline_batch.run_pipeline_over_directories(
            active.pipeline, active.directories,
            generation_type=generation_type,
            profile_name=active.profile_name,
            hide_callback=hide_callback,
            notify_callback=notify_callback,
            add_mark_callback=add_mark_callback,
            blur_callback=blur_callback,
            run_control=control,
        )
    except Exception as e:
        history.finish_run(active.run_id, RunStatus.FAILED,
                           error=str(e) or type(e).__name__, counters=control.snapshot())
        raise
    else:
        status = _record_outcome(active, outcome)
        logger.info("Pipeline %r: run %s ended: %s", getattr(active.pipeline, "name", "?"),
                    active.run_id, status.value)
        return outcome
    finally:
        with _registry_lock:
            _active.pop(active.run_id, None)
        active.done.set()


def _record_outcome(active: ActiveRun, outcome) -> RunStatus:
    control = active.control
    # The batch's stats are the authority for the counters; a test double
    # for the batch may return an outcome without the newer fields.
    stats = getattr(outcome, "stats", None)
    counters = control.snapshot()
    if stats is not None:
        counters.files_evaluated = getattr(stats, "files_evaluated", counters.files_evaluated)
        counters.errors = getattr(stats, "errors", counters.errors)
        counters.action_counts = dict(getattr(stats, "action_counts", counters.action_counts))
    if getattr(outcome, "cancelled", False):
        status = RunStatus.INTERRUPTED if control.cancel_reason == CANCEL_SHUTDOWN else RunStatus.CANCELLED
    elif counters.errors:
        status = RunStatus.COMPLETED_WITH_ERRORS
    else:
        status = RunStatus.COMPLETED
    _history().finish_run(active.run_id, status,
                          summary=getattr(outcome, "summary", None),
                          dump_path=getattr(outcome, "dump_path", None),
                          counters=counters)
    return status


def abandon(active: ActiveRun, error: str) -> None:
    """Record a registered run whose worker never started as failed."""
    with _registry_lock:
        _active.pop(active.run_id, None)
    active.done.set()
    _history().finish_run(active.run_id, RunStatus.FAILED, error=error)


def check_resume(run_id: str) -> ResumeCheck:
    """Whether the run *run_id* can be resumed, and what changed since."""
    from compare.classifier_pipeline import ClassifierPipelines
    from files.directory_profile import DirectoryProfile

    record = _history().get(run_id)
    if record is None:
        return ResumeCheck(record=None, problem=_("This run is no longer in the history."))
    check = ResumeCheck(record=record)
    if record.resumed_by is not None:
        check.problem = _("This run was already resumed.")
        return check
    if not record.is_resumable:
        check.problem = _("Only cancelled, failed or interrupted runs with saved progress can be resumed.")
        return check
    pipeline = ClassifierPipelines.get_pipeline_by_name(record.pipeline_name)
    if pipeline is None:
        check.problem = _("Pipeline '{0}' no longer exists.").format(record.pipeline_name)
        return check
    check.pipeline = pipeline
    current = pipeline_fingerprint(pipeline)
    check.pipeline_changed = (
        current is not None and record.pipeline_fingerprint is not None
        and current != record.pipeline_fingerprint
    )
    check.missing_directories = [d for d in record.directories if not os.path.isdir(d)]
    if record.profile_name:
        profile = DirectoryProfile.get_profile_by_name(record.profile_name)
        check.profile_changed = (
            profile is None or not same_directory_set(profile.directories, record.directories)
        )
    return check


def request_stop(run_id: str) -> bool:
    """Ask the run *run_id* to stop after its current file. False when it
    isn't a run active in this process."""
    with _registry_lock:
        active = _active.get(run_id)
    if active is None:
        return False
    active.control.request_cancel(CANCEL_USER)
    _history().mark_stop_requested(run_id)
    return True


def active_run_ids() -> List[str]:
    with _registry_lock:
        return list(_active)


def is_active(run_id: str) -> bool:
    with _registry_lock:
        return run_id in _active


def shutdown(timeout: float = 3.0) -> None:
    """Stop every active run for process exit and wait up to *timeout*
    seconds in total. A run that doesn't end in time is left to the next
    start's reconciliation, which records it as interrupted."""
    with _registry_lock:
        actives = list(_active.values())
    for active in actives:
        active.control.request_cancel(CANCEL_SHUTDOWN)
    deadline = time.monotonic() + timeout
    for active in actives:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not active.done.wait(remaining):
            logger.warning("Pipeline run %s still running at shutdown", active.run_id)


def reset() -> None:
    """Cancel and forget every active run (tests)."""
    with _registry_lock:
        actives = list(_active.values())
        _active.clear()
    for active in actives:
        active.control.request_cancel(CANCEL_SHUTDOWN)

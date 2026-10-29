"""Qt-free execution of a classifier pipeline over whole directories.

run_pipeline_over_directories() is the entry point; the rest are the pieces it
assembles -- generate/scramble dispatch state, and the run dump written
afterwards. Nothing here is presentation or Qt, so the GUI and a headless
caller share one implementation: the file ordering, the cross-directory
stem-group gate and the reporting rules exist once, not per caller.

run_one_scramble is looked up as a module global at dispatch time rather than
captured, so patching it on this module affects work already queued.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import _

if TYPE_CHECKING:
    from compare.classifier_pipeline import ClassifierPipeline
    from compare.pipeline_run_control import PipelineRunControl

logger = get_logger("classifier_pipeline_batch")


class PipelineValidationError(ValueError):
    """A batch run refused because the pipeline fails validate(); *errors*
    are validate()'s messages."""

    def __init__(self, pipeline_name: str, errors: list[str]):
        self.errors = list(errors)
        super().__init__(
            _("Pipeline {0} cannot run until these are fixed:\n{1}").format(
                pipeline_name, "\n".join(self.errors)
            )
        )


@dataclass
class PipelineRunOutcome:
    """What a finished (completed or cancelled) batch run produced."""

    stats: Any
    report: Any
    summary: str
    generates: list
    scrambles: list
    cancelled: bool = False
    dump_path: Optional[str] = None


def run_one_scramble(
    path: str, modifier: str | None, skip_existing: bool = False
) -> None:
    from image.image_ops import ImageOps
    ImageOps.scramble_by_modifier(path, modifier, skip_existing=skip_existing)


def make_scramble_batch_state(
    scramble_batch_size: int | None,
    on_flush: Optional[Callable[[], None]] = None,
) -> tuple[list, object, object]:
    """Build the scramble-execution state for a pipeline run.

    Returns (all_scrambles, on_scramble, execute_batch) where:
      all_scrambles   – grows with every call to on_scramble; written to the dump.
      on_scramble     – use as ActionCallbacks.scramble_callback.
      execute_batch   – call at end-of-run to flush any remainder; also called
                        automatically at each intermediate BATCH_SIZE threshold.

    scramble_batch_size=None means no intermediate flushes; one flush at end-of-run.
    Setting pipeline_scramble_batch_size=0 in config produces None (inline-like behaviour).
    on_flush, if given, is called after each non-empty flush.
    """
    all_scrambles: list[tuple[str, str | None]] = []
    flush_scrambles: list[tuple[str, str | None]] = []

    def execute_batch() -> None:
        if not flush_scrambles:
            return
        batch = list(flush_scrambles)
        flush_scrambles.clear()
        from concurrent.futures import ThreadPoolExecutor
        # 4 workers: modest parallelism without saturating I/O alongside the scan loop.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(run_one_scramble, path, modifier)
                for path, modifier in batch
            ]
            for f in futures:
                try:
                    f.result()
                except Exception:
                    logger.exception("Scramble batch item failed")
        if on_flush is not None:
            on_flush()

    def on_scramble(path: str, modifier: str | None = None) -> None:
        all_scrambles.append((path, modifier))
        flush_scrambles.append((path, modifier))
        if scramble_batch_size is not None and len(flush_scrambles) >= scramble_batch_size:
            execute_batch()

    return all_scrambles, on_scramble, execute_batch


def make_generate_batch_state(
    generation_type,
    generate_batch_size: int | None,
    on_flush: Optional[Callable[[], None]] = None,
) -> tuple[list, object, object]:
    """Build the generate-dispatch state for a pipeline run.

    Returns (all_generates, on_generate, dispatch_batch) where:
      all_generates   – grows with every call to on_generate; written to the dump.
      on_generate     – use as ActionCallbacks.generate_callback.
      dispatch_batch  – call at the end-of-run to flush any remainder; also
                        called automatically at each intermediate BATCH_SIZE threshold.

    generate_batch_size=None disables intermediate flushes; one dispatch happens
    at end-of-run.  Setting pipeline_generate_batch_size=0 in config produces None.
    on_flush, if given, is called after each non-empty dispatch, failed or not.
    """
    all_generates: list[tuple[str, str | None]] = []
    flush_generates: list[tuple[str, str | None, str | None]] = []

    def dispatch_batch() -> None:
        if not flush_generates:
            return
        batch = list(flush_generates)
        flush_generates.clear()
        batch_args = [
            {
                'image': path,
                'append': False,
                **({'edit_suffix': suffix} if suffix else {}),
                **({'target_dir': tdir} if tdir else {}),
            }
            for path, suffix, tdir in batch
        ]
        try:
            from extensions.sd_runner_client import SDRunnerClient
            SDRunnerClient().run_batch(generation_type, batch_args)
        except Exception:
            logger.exception(
                "Intermediate generate batch failed; items in all_generates for rerun"
            )
        if on_flush is not None:
            on_flush()

    def on_generate(path: str, edit_suffix: str | None = None, target_dir: str | None = None) -> None:
        all_generates.append((path, edit_suffix))
        flush_generates.append((path, edit_suffix, target_dir))
        if generate_batch_size is not None and len(flush_generates) >= generate_batch_size:
            dispatch_batch()

    return all_generates, on_generate, dispatch_batch


def find_latest_dump(pipeline: "ClassifierPipeline"):
    """Return the most recent dump Path for *pipeline*, or None if absent."""
    try:
        from utils.logging_setup import get_log_dir
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in pipeline.name
        )
        return max(get_log_dir().glob(f"pipeline_run_*_{safe_name}.json"), default=None)
    except Exception:
        return None


def write_pipeline_run_dump(
    pipeline, stats, report,
    all_generates=(), all_scrambles=(),
    run_info: Optional[dict] = None,
) -> Optional[Path]:
    """Write the run's JSON dump to the log directory and return its path,
    or None if writing failed. *run_info* (run id, status, start time, ...)
    is written as the dump's "run" block when given."""
    try:
        import json
        from datetime import datetime
        from utils.logging_setup import get_log_dir
        dump = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": pipeline.to_dict(),
            "stats": {
                "pipeline_name": stats.pipeline_name,
                "profile_name": stats.profile_name,
                "directories": stats.directories,
                "files_by_directory": stats.files_by_directory,
                "files_evaluated": stats.files_evaluated,
                "errors": stats.errors,
                "action_counts": stats.action_counts,
                "generates_queued": stats.generates_queued,
                "generation_type_label": stats.generation_type_label,
                "generation_type_value": stats.generation_type_value,
                "category_map": stats.category_map,
                "cancelled": getattr(stats, "cancelled", False),
                "run_id": getattr(stats, "run_id", None),
                "resumed_from": getattr(stats, "resumed_from", None),
                "files_skipped": getattr(stats, "files_skipped", 0),
            },
            "generates": [
                {"path": path, "modifier": modifier}
                for path, modifier in all_generates
            ],
            "scrambles": [
                {"path": path, "modifier": modifier}
                for path, modifier in all_scrambles
            ],
            "messages": [
                {
                    "severity": m.severity,
                    "node": m.node,
                    "image_path": m.image_path,
                    "detail": m.detail,
                    "data": m.data,
                }
                for m in report.messages()
            ],
            # Empty unless the pipeline sets record_node_verdicts; one entry
            # per evaluated file, already JSON-ready (see
            # compare.pipeline_decision_record).
            "decisions": report.decisions(),
        }
        if run_info is not None:
            dump["run"] = dict(run_info)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in pipeline.name)
        dump_path = get_log_dir() / f"pipeline_run_{ts}_{safe_name}.json"
        dump_path.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
        logger.info("Pipeline run data written to %s", dump_path)
        return dump_path
    except Exception:
        logger.exception("Failed to write pipeline run dump")
        return None


def run_pipeline_over_directories(
    pipeline: "ClassifierPipeline",
    directories: Sequence[str],
    *,
    generation_type=None,
    profile_name: Optional[str] = None,
    hide_callback: Optional[Callable] = None,
    notify_callback: Optional[Callable] = None,
    add_mark_callback: Optional[Callable] = None,
    blur_callback: Optional[Callable] = None,
    write_dump: bool = True,
    run_control: Optional["PipelineRunControl"] = None,
) -> PipelineRunOutcome:
    """Evaluate *pipeline* against every file in *directories*.

    This is the whole batch run: it builds the generate/scramble dispatch state
    from config, assembles ActionCallbacks around the caller's presentation
    callbacks, walks the directories, flushes both batches at the end, and
    writes the run dump. Callers supply only what differs between them -- the
    GUI its notification callbacks, a headless caller none or its own -- so the
    ordering, stem-group and reporting rules exist once rather than per caller.

    generation_type is resolved by the caller because the GUI falls back to an
    application-wide setting the pipeline itself does not know about.

    Blocking and single-threaded; run it on a worker thread if the caller must
    stay responsive. Raises PipelineValidationError, before touching any file,
    when the pipeline fails validate().

    *run_control* makes the run stoppable and resumable: it is checked for
    cancellation before every file, told about each processed file and each
    finished directory, and, when seeded from an earlier run's checkpoint,
    supplies the directories and files to skip plus the counters, stem-group
    gate and undispatched generate/scramble work to carry over. A cancelled
    run still dispatches the work its processed files decided on and writes
    its dump; outcome.cancelled tells it apart.
    """
    errors = pipeline.validate()
    if errors:
        raise PipelineValidationError(pipeline.name, errors)

    from compare.action_callbacks import ActionCallbacks
    from compare.base_compare import gather_files
    from compare.classifier_pipeline_runner import run_pipeline
    from compare.pipeline_run_report import PipelineRunReport, PipelineRunStats
    from files.related_image import (
        clear_base_stem_dir_cache,
        clear_generate_gate_cache,
        extract_filename_base_stem,
    )
    from utils.constants import ClassifierActionType

    directories = list(directories)

    # 0 means "no intermediate flush" (single batch at end-of-run).
    _gen_cfg = config.pipeline_generate_batch_size
    _scr_cfg = config.pipeline_scramble_batch_size
    # all_generates/all_scrambles are the full audit record: written to the
    # dump and used by rerun-last, regardless of how often the batch flushed.
    all_generates, on_generate, dispatch_generate_batch = make_generate_batch_state(
        generation_type, _gen_cfg if _gen_cfg > 0 else None,
        on_flush=run_control.generates_flushed if run_control is not None else None,
    )
    all_scrambles, on_scramble, execute_scramble_batch = make_scramble_batch_state(
        _scr_cfg if _scr_cfg > 0 else None,
        on_flush=run_control.scrambles_flushed if run_control is not None else None,
    )
    if run_control is not None:
        base_on_generate, base_on_scramble = on_generate, on_scramble

        # Noted before the base callback, which may flush (and clear) at once.
        def on_generate(path, edit_suffix=None, target_dir=None):
            run_control.note_generate(path, edit_suffix, target_dir)
            base_on_generate(path, edit_suffix, target_dir)

        def on_scramble(path, modifier=None):
            run_control.note_scramble(path, modifier)
            base_on_scramble(path, modifier)

    callbacks = ActionCallbacks(
        hide_callback=hide_callback,
        notify_callback=notify_callback,
        add_mark_callback=add_mark_callback,
        blur_callback=blur_callback,
        generate_callback=on_generate,
        scramble_callback=on_scramble,
    )

    clear_base_stem_dir_cache()
    clear_generate_gate_cache()
    report = PipelineRunReport()
    total = 0
    errors = 0
    actions: dict[str, int] = {}
    files_by_directory: dict[str, int] = {}
    # One set shared across every directory in the run, so a stem group
    # spanning two scanned directories is still only evaluated once.
    # None disables the stem-group gate entirely (every file evaluated).
    processed_stems = set() if pipeline.dedupe_stem_groups else None
    cancelled = False

    if run_control is not None:
        total, errors, actions, files_by_directory = run_control.seed_counters()
        if processed_stems is not None:
            processed_stems = run_control.seed_processed_stems() or set()
        for path, edit_suffix, target_dir in run_control.carried_generates():
            on_generate(path, edit_suffix, target_dir)
        for path, modifier in run_control.carried_scrambles():
            on_scramble(path, modifier)

    for directory in directories:
        if run_control is not None:
            if run_control.is_cancelled():
                cancelled = True
                break
            if run_control.is_directory_completed(directory):
                continue
        files = pipeline.sort_files_for_run(list(gather_files(directory)))
        files_by_directory[directory] = len(files)
        logger.info(
            "Pipeline %r: scanning %s — %d file(s)", pipeline.name, directory, len(files)
        )
        if run_control is not None:
            run_control.directory_started(directory, len(files))
        for image_path in files:
            if run_control is not None:
                if run_control.is_cancelled():
                    cancelled = True
                    break
                if run_control.is_file_processed(image_path):
                    continue
            try:
                msg_snapshot = report.message_count()
                result = run_pipeline(
                    pipeline, image_path, callbacks,
                    base_directory=directory, report=report,
                    processed_stems=processed_stems,
                )
                total += 1
                key = result.value if isinstance(result, ClassifierActionType) else "(no action)"
                actions[key] = actions.get(key, 0) + 1
                if not config.debug:
                    base_stem = extract_filename_base_stem(image_path)
                    file_stem = os.path.splitext(os.path.basename(image_path))[0]
                    if base_stem and file_stem.lower() == base_stem.lower():
                        logger.info(
                            "Pipeline %r: %s",
                            pipeline.name,
                            report.format_seed_summary(image_path, result, msg_snapshot),
                        )
            except Exception:
                errors += 1
                logger.exception("Pipeline run error on %s", image_path)
            if run_control is not None:
                # A file that raised counts as processed too, so resuming
                # doesn't retry it forever.
                run_control.file_done(
                    image_path, processed_stems=processed_stems,
                    files_evaluated=total, errors=errors,
                    action_counts=actions, files_by_directory=files_by_directory,
                )
        if cancelled:
            break
        if run_control is not None:
            run_control.directory_finished(directory)

    if cancelled:
        logger.info("Pipeline %r: run cancelled after %d file(s)", pipeline.name, total)

    gen_label = generation_type.get_text() if generation_type is not None else None
    stats = PipelineRunStats(
        pipeline_name=pipeline.name,
        profile_name=profile_name,
        directories=directories,
        files_by_directory=files_by_directory,
        files_evaluated=total,
        errors=errors,
        action_counts=actions,
        generates_queued=len(all_generates),
        generation_type_label=gen_label,
        generation_type_value=generation_type.value if generation_type is not None else None,
        category_map=dict(pipeline.category_map or {}),
        cancelled=cancelled,
        run_id=run_control.run_id if run_control is not None else None,
        resumed_from=run_control.resumed_from if run_control is not None else None,
        files_skipped=run_control.files_skipped if run_control is not None else 0,
    )
    summary = report.format_completion_report(stats)
    logger.info("\n%s", summary)
    dispatch_generate_batch()   # flush generate remainder
    execute_scramble_batch()    # flush scramble remainder
    dump_path = None
    if write_dump:
        run_info = None
        if run_control is not None:
            run_info = {
                "run_id": run_control.run_id,
                "status": "cancelled" if cancelled else "completed",
                "started_at": run_control.started_at,
                "resumed_from": run_control.resumed_from,
            }
        dump_path = write_pipeline_run_dump(
            pipeline, stats, report, all_generates, all_scrambles, run_info=run_info)
    if run_control is not None:
        run_control.final_checkpoint()

    return PipelineRunOutcome(
        stats=stats, report=report, summary=summary,
        generates=all_generates, scrambles=all_scrambles,
        cancelled=cancelled,
        dump_path=str(dump_path) if dump_path is not None else None,
    )

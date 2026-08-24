#!/usr/bin/env python3
"""Runnable reference for driving Weidr from a script or agent, with no UI.

Run it:
    python scripts/agent_headless_demo.py            # run every demo
    python scripts/agent_headless_demo.py --api      # print the API notes only
    python scripts/agent_headless_demo.py --list     # list demo names
    python scripts/agent_headless_demo.py --only compare
    python scripts/agent_headless_demo.py --keep     # keep the sandbox to inspect

Every demo works inside a throwaway directory this script creates and fills
with generated images, and points the app cache at that directory too. Nothing
here touches media you own or the pipelines you have saved -- which matters,
because parts of this API move and delete files for real. Your config is read
but never written, so what the demo sees still reflects your settings.

================================================================================
API NOTES  (this is the part to read before writing your own script)
================================================================================

SETUP
-----
The core packages -- compare/, files/, image/ -- import no Qt. What they do
expect is an "app actions" object: the callback bundle the UI normally
supplies. Build the Qt-free one:

    from utils.headless_app_actions import build_headless_app_actions
    app_actions = build_headless_app_actions({
        "get_base_dir": lambda: my_dir,      # supply these two
        "is_compare_running": lambda: False,
    })

Messages become log lines and display calls become no-ops. Two of the eleven
domain actions are genuinely reached from the Qt-free packages -- get_base_dir
and is_compare_running -- and both are cheap to answer, as above. The rest are
only ever called by the UI; leave them out. Calling one you did not supply
raises HeadlessActionUnavailable naming it, rather than silently doing nothing.

A question that would open a dialog (alert with yes/no) answers False without a
user. That is deliberate: declining is the safe answer, so an engine that asks
"proceed?" stops rather than guessing yes.

BROWSING AND NAVIGATION            files.file_browser.FileBrowser
-----------------------------------------------------------------
    fb = FileBrowser(directory)
    fb.set_directory(path)      -> list[str]   also refreshes
    fb.get_files()              -> list[str]   filtered by config.file_types
    fb.current_file()           -> str | None
    fb.next_file()              -> str         wraps at the end
    fb.previous_file()          -> str         wraps at the start
    fb.last_file()              -> str
    fb.go_to_file(path)         -> None
    fb.go_to_index(n)           -> str         *** 1-based, not 0-based ***
    fb.random_file()            -> str
    fb.get_cursor()             -> int         0-based position

Two things bite here. go_to_index() counts from 1, so index 1 is the first
file. And a freshly set directory leaves the cursor at -1, a pre-position
sentinel: the first next_file() therefore lands on the first file, but calling
previous_file() straight away steps to -2 and returns the second-from-last
file. Position the cursor before stepping backwards.

The navigation entries on the app-actions contract (go_to_file,
show_next_media) answer False without a screen. FileBrowser is the interface
to use instead -- those exist to render, which is not something you want.

MARKS, MOVING, COPYING             files.marked_files.MarkedFiles
-----------------------------------------------------------------
Marks are process-wide class state, not per-window:

    MarkedFiles.add_mark_if_not_present(path, app_actions) -> bool
    MarkedFiles.file_marks                                  -> list[str]
    MarkedFiles.clear_file_marks(app_actions)               -> None

    MarkedFiles.move_marks_to_dir_static(
        app_actions,
        target_dir=...,
        move_func=Utils.move_file,   # or Utils.copy_file to copy
    ) -> (some_files_already_present, exceptions_present)

Note the return: two booleans about *problems*, not a success flag. Both False
means the transfer went through cleanly. Successfully moved files are removed
from file_marks; ones that failed stay marked.

DELETING                           files.marked_files.MarkedFiles
-----------------------------------------------------------------
    MarkedFiles.delete_file_static(
        path, app_actions,
        toast=True, manual_delete=True, is_directory=False,
    ) -> bool

Honours the configured trash folder: with config.delete_instantly False and a
trash folder set, the file is moved there rather than destroyed. Returns
whether it ended up deleted; failures are reported, not raised.

COMPARISONS                        compare.compare_manager.CompareManager
-----------------------------------------------------------------
    manager = CompareManager(
        master=None,                          # no window
        app_actions=app_actions,
        get_base_dir=lambda: base_dir,
        responsiveness=NullResponsiveness(),  # nothing to keep responsive
    )
    manager.set_primary_mode(CompareMode.COLOR_MATCHING)
    manager.run(args)
    manager._primary_wrapper().file_groups    # {group_index: {path: distance}}

Two traps in CompareArgs:

  * mode must be Mode.GROUP for a grouping run. With Mode.SEARCH the engine
    asks for confirmation first, the headless port declines, and the run
    returns having done nothing at all.
  * compare_threshold means different things per mode. It defaults to
    embedding_similarity_threshold (0.9); CompareColors reads the same field
    as a LAB colour distance, where the useful value is config
    .color_diff_threshold (15). Set it to match your mode or nothing groups.

COLOR_MATCHING needs no ML model, which is why this demo uses it. The
embedding modes download and load a model on first use.

To run a comparison off the calling thread, use the Qt-free runner:

    from utils.background_runner import ThreadedTaskRunner
    ThreadedTaskRunner().start(manager.run, [args],
                               on_finished=..., on_error=...)

CLASSIFIER PIPELINES
-----------------------------------------------------------------
Build and persist (compare.classifier_pipeline):

    p = ClassifierPipeline(name="...", is_active=True)
    p.nodes = [PipelineNode(name=..., condition=..., on_match=..., on_no_match=...)]
    ClassifierPipelines.add_pipeline(p)
    ClassifierPipelines.store()          # -> app_info_cache
    ClassifierPipelines.load()           # in a later session
    ClassifierPipelines.get_pipeline_by_name("...")

A pipeline with is_active False, or with no nodes, evaluates to None for every
file -- it is skipped, not an error.

Run it over directories (compare.classifier_pipeline_batch):

    outcome = run_pipeline_over_directories(
        pipeline, [dir_a, dir_b],
        add_mark_callback=MarkedFiles.add_mark_if_not_present,
        write_dump=True,
    )
    outcome.stats     # PipelineRunStats: files_evaluated, action_counts, ...
    outcome.summary   # the human-readable report
    outcome.generates / outcome.scrambles

This is the same function the UI's "Run" button uses, so file ordering, the
cross-directory duplicate gate and the run report behave identically. Pass
your own presentation callbacks or none. write_dump=False skips the JSON run
record, which is what "Rerun Last" would otherwise read back.

DIRECTORY-WIDE MEDIA OPERATIONS    image.directory_ops
-----------------------------------------------------------------
Each operation comes in two halves so you can ask what would happen before
doing it -- the UI uses the first half to fill its confirmation dialog:

    survey = survey_jpg_conversion(files)          # counts, candidates
    convert_files_to_jpg(survey, overwrite_existing=False)

    survey_svg_conversion(files)      / convert_svgs_to_png(survey, ...)
    survey_image_scaling(files, side) / scale_images(survey)
    survey_video_metadata_strip(files)/ strip_video_metadata(survey)

Surveys touch nothing. The decision the dialog would collect is a plain
argument, so there is no callback back into a user interface.

files.directory_ops.move_directory_contents_then_delete(src, dst) merges one
directory into another and removes it; an entry whose name already exists at
the destination is skipped and left behind.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402


def _say(heading, *lines):
    print(f"\n--- {heading} " + "-" * max(0, 60 - len(heading)))
    for line in lines:
        print(f"    {line}")


def _png(path, color=(120, 120, 120), size=(64, 64)):
    Image.new("RGB", size, color).save(str(path), format="PNG")
    return str(path)


def _sandbox():
    """A throwaway directory with a few generated images. Never user media.

    Also redirects the app cache into it. The pipeline demo persists a
    pipeline, and app_info_cache resolves its location once, in its
    constructor -- so this has to be set before anything imports it. Nothing at
    this module's top level does, and the demos import lazily, so setting it
    here is early enough. Your real cache is left alone either way.
    """
    root = tempfile.mkdtemp(prefix="weidr_agent_demo_")
    source = os.path.join(root, "source")
    target = os.path.join(root, "target")
    cache = os.path.join(root, "cache")
    os.makedirs(source)
    os.makedirs(target)
    os.makedirs(cache)
    os.environ["WEIDR_CACHE_DIR"] = cache
    for i, color in enumerate([(200, 30, 30), (200, 40, 35), (30, 30, 200)]):
        _png(os.path.join(source, f"sample_{i}.png"), color)
    _png(os.path.join(source, "keep_me.png"), (30, 200, 30))
    return root, source, target


def _actions(base_dir):
    from utils.headless_app_actions import build_headless_app_actions
    return build_headless_app_actions({
        "get_base_dir": lambda: base_dir,
        "is_compare_running": lambda: False,
    })


def demo_browse(source, target):
    """Open a directory and move the cursor around."""
    from files.file_browser import FileBrowser

    fb = FileBrowser(source)
    fb.set_directory(source)
    files = fb.get_files()
    _say("browse", f"{len(files)} file(s) found", *[os.path.basename(f) for f in files])
    if not files:
        print("    (config.file_types excludes .png -- nothing to navigate)")
        return
    _say("navigate",
         f"next_file()     -> {os.path.basename(fb.next_file())}",
         f"go_to_index(2)  -> {os.path.basename(fb.go_to_index(2))}   (1-based!)",
         f"cursor is now   -> {fb.get_cursor()}                       (0-based)",
         f"last_file()     -> {os.path.basename(fb.last_file())}")


def demo_marks(source, target):
    """Mark files and move them to another directory."""
    from files.file_browser import FileBrowser
    from files.marked_files import MarkedFiles
    from utils.utils import Utils

    app_actions = _actions(source)
    fb = FileBrowser(source)
    fb.set_directory(source)
    to_mark = [f for f in fb.get_files() if "sample_" in os.path.basename(f)]
    if not to_mark:
        _say("marks", "no sample files visible under the current config -- skipped")
        return

    MarkedFiles.clear_file_marks(app_actions)
    for path in to_mark:
        MarkedFiles.add_mark_if_not_present(path, app_actions)
    _say("marks", f"marked {len(MarkedFiles.file_marks)} file(s)")

    already_present, had_errors = MarkedFiles.move_marks_to_dir_static(
        app_actions, target_dir=target, move_func=Utils.move_file,
    )
    _say("move",
         f"name collisions at target: {already_present}",
         f"errors during transfer   : {had_errors}",
         f"target now holds         : {sorted(os.listdir(target))}",
         f"marks remaining          : {len(MarkedFiles.file_marks)}")


def demo_delete(source, target):
    """Delete one file, honouring the configured trash folder."""
    from files.marked_files import MarkedFiles

    victim = os.path.join(source, "keep_me.png")
    if not os.path.isfile(victim):
        _say("delete", "nothing to delete -- skipped")
        return
    deleted = MarkedFiles.delete_file_static(victim, _actions(source), toast=False)
    _say("delete",
         f"delete_file_static -> {deleted}",
         f"still on disk?      {os.path.isfile(victim)}",
         "(with a trash folder configured the file is moved there, not destroyed)")


def demo_compare(source, target):
    """Group visually similar files. COLOR_MATCHING needs no ML model."""
    from compare.compare_args import CompareArgs
    from compare.compare_manager import CompareManager
    from utils.config import config
    from utils.constants import CompareMode, Mode
    from utils.ui_responsiveness import NullResponsiveness

    scope = target if os.listdir(target) else source
    app_actions = _actions(scope)
    manager = CompareManager(
        master=None, app_actions=app_actions,
        get_base_dir=lambda: scope, responsiveness=NullResponsiveness(),
    )
    manager.set_primary_mode(CompareMode.COLOR_MATCHING)
    manager.run(CompareArgs(
        base_dir=scope,
        mode=Mode.GROUP,                  # not SEARCH -- see the API notes
        compare_mode=CompareMode.COLOR_MATCHING,
        recursive=False,
        store_checkpoints=False,
        app_actions=app_actions,
        compare_threshold=config.color_diff_threshold,   # per-mode! see notes
    ))
    groups = manager._primary_wrapper().file_groups
    _say("compare", f"scope: {scope}", f"{len(groups)} group(s) formed")
    for index, group in groups.items():
        print(f"    group {index}: {[os.path.basename(p) for p in group]}")


def demo_pipeline(source, target):
    """Build, persist, reload and run a classifier pipeline."""
    from compare.classifier_pipeline import ClassifierPipeline, ClassifierPipelines
    from compare.classifier_pipeline_batch import run_pipeline_over_directories
    from compare.classifier_pipeline_conditions import FilenameContainsCondition
    from compare.classifier_pipeline_nodes import NodeOutcome, OutcomeType, PipelineNode
    from utils.constants import ClassifierActionType

    marked = []

    pipeline = ClassifierPipeline(name="AgentDemoPipeline", is_active=True)
    pipeline.nodes = [PipelineNode(
        name="mark_samples",
        # Filename matching needs no model, so this runs anywhere.
        condition=FilenameContainsCondition(patterns=["sample_"]),
        on_match=NodeOutcome(OutcomeType.EXECUTE,
                             action_type=ClassifierActionType.ADD_MARK),
        on_no_match=NodeOutcome.accept(),
    )]

    ClassifierPipelines.add_pipeline(pipeline)
    ClassifierPipelines.store()
    ClassifierPipelines.pipelines = []          # prove it round-trips
    ClassifierPipelines.load()
    restored = ClassifierPipelines.get_pipeline_by_name("AgentDemoPipeline")
    _say("pipeline", f"stored and reloaded: {restored is not None}")

    scope = target if os.listdir(target) else source
    outcome = run_pipeline_over_directories(
        restored or pipeline, [scope],
        add_mark_callback=lambda path: marked.append(path),
        write_dump=False,
    )
    _say("pipeline run",
         f"files evaluated : {outcome.stats.files_evaluated}",
         f"actions fired   : {outcome.stats.action_counts}",
         f"marked by rule  : {[os.path.basename(p) for p in marked]}")

    ClassifierPipelines.remove_pipeline("AgentDemoPipeline")
    ClassifierPipelines.store()


def demo_directory_ops(source, target):
    """Ask what a directory-wide conversion would do, then do it."""
    from image import directory_ops

    scope = target if os.listdir(target) else source
    files = [os.path.join(scope, n) for n in sorted(os.listdir(scope))]

    survey = directory_ops.survey_jpg_conversion(files)
    _say("survey (changes nothing)",
         f"convertible images   : {len(survey.image_files)}",
         f"not yet JPG          : {len(survey.convert_candidates)}",
         f"targets already there: {survey.existing_target_count}")

    result = directory_ops.convert_files_to_jpg(survey, overwrite_existing=False)
    _say("convert",
         f"converted: {result.converted}   skipped: {result.skipped_existing}   failed: {result.failed}",
         f"directory now: {sorted(os.listdir(scope))}")


DEMOS = {
    "browse": demo_browse,
    "marks": demo_marks,
    "delete": demo_delete,
    "compare": demo_compare,
    "pipeline": demo_pipeline,
    "directory-ops": demo_directory_ops,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Runnable reference for driving Weidr without its UI.",
    )
    parser.add_argument("--api", action="store_true",
                        help="print the API notes from this file's docstring and exit")
    parser.add_argument("--list", action="store_true", help="list demo names and exit")
    parser.add_argument("--only", metavar="NAME", help="run a single demo")
    parser.add_argument("--keep", action="store_true",
                        help="keep the sandbox directory instead of deleting it")
    args = parser.parse_args(argv)

    if args.api:
        print(__doc__)
        return 0
    if args.list:
        for name, fn in DEMOS.items():
            print(f"{name:15} {fn.__doc__.splitlines()[0]}")
        return 0

    selected = DEMOS if args.only is None else {args.only: DEMOS.get(args.only)}
    if args.only is not None and selected[args.only] is None:
        parser.error(f"unknown demo {args.only!r}; try --list")

    root, source, target = _sandbox()
    print(f"sandbox: {root}")
    print("(generated files only -- no media of yours is read or written)")
    try:
        for name, fn in selected.items():
            print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
            fn(source, target)
    finally:
        if args.keep:
            print(f"\nsandbox kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
            print("\nsandbox removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

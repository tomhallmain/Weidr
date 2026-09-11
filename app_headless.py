#!/usr/bin/env python3
"""Weidr -- persistent headless entry point.

No QApplication, no window: this process holds one browsing/comparison
session open for its lifetime and serves it over MCP
(extensions/mcp_server.py), for an agent to drive with no GUI involved.

This is the persistent counterpart to scripts/agent_headless_demo.py, which
builds the same kind of session but throws it away after one script run.
Both rest on the same Qt-free layer: files/file_browser.py,
compare/compare_manager.py, files/marked_files.py, and
utils/headless_app_actions.py.

Run it:
    python app_headless.py --base-dir /path/to/media --port 6200
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from compare.compare_args import CompareArgs
from compare.compare_manager import CompareManager
from extensions.mcp_server import MCPServerExtension
from files.file_browser import FileBrowser
from files.marked_files import MarkedFiles
from files.skip_aware_navigation import advance_past_skipped
from utils.background_runner import ThreadedTaskRunner
from utils.config import config
from utils.constants import CompareMode, ImageGenerationType, Mode
from utils.headless_app_actions import build_headless_app_actions
from utils.logging_setup import get_logger
from utils.ui_responsiveness import NullResponsiveness

logger = get_logger("app_headless")

# Mirrors ui/image/media_details.py's MediaDetails.get_image_specific_generation_mode()
# fallback for a type that isn't one of the four "sticky" ones a Qt session
# can select interactively. There's no interactive selection here, so this is
# the only type headless generation requests use.
_DEFAULT_IMAGE_GENERATION_TYPE = ImageGenerationType.CONTROL_NET


class _CompareMaster:
    """The part of an AppWindow that CompareWrapper reads off its master:
    the app mode (kept current through the set_mode action), the file
    browser whose sort orders a group complement, and a repaint hook.
    """

    def __init__(self, file_browser: FileBrowser):
        self.file_browser = file_browser
        self.mode = Mode.BROWSE

    def set_mode(self, mode: Mode, do_update: bool = True) -> None:
        self.mode = mode

    def update(self) -> None:
        pass


class HeadlessMCPSession:
    """Session backed by one Qt-free FileBrowser/CompareManager pair.

    Implements the same interface extensions/mcp_server.py calls through for
    a live AppWindow (ui/app_window/mcp_session_qt.py), but directly against
    the Qt-free layer -- there is no display to update, so none of these
    methods do anything beyond the underlying state change.
    """

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._runner = ThreadedTaskRunner()
        self._file_browser = FileBrowser(base_dir)
        self._file_browser.set_directory(base_dir)
        self._compare_master = _CompareMaster(self._file_browser)
        self._actions = build_headless_app_actions({
            "get_base_dir": lambda: self._base_dir,
            "is_compare_running": lambda: self._runner.is_running(),
            "set_mode": self._compare_master.set_mode,
            # Reached by prevalidation/pipeline HIDE and GENERATE rules through
            # AppActions._build_callbacks; unsupplied, a matching rule would
            # raise HeadlessActionUnavailable out of skip_media() mid-next_file.
            "hide_media": self._hide_media,
            "run_image_generation": self._generate_for_callback,
        })
        self._compare_manager = CompareManager(
            master=self._compare_master, app_actions=self._actions,
            get_base_dir=lambda: self._base_dir,
            responsiveness=NullResponsiveness(),
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def get_current_file(self) -> Optional[str]:
        return self._file_browser.current_file()

    def next_file(self) -> Optional[str]:
        start = self._file_browser.current_file()
        candidate = self._file_browser.next_file()
        return advance_past_skipped(
            self._file_browser, self._compare_manager.skip_media,
            backward=False, start=start, current=candidate,
        )

    def previous_file(self) -> Optional[str]:
        start = self._file_browser.current_file()
        candidate = self._file_browser.previous_file()
        return advance_past_skipped(
            self._file_browser, self._compare_manager.skip_media,
            backward=True, start=start, current=candidate,
        )

    def get_index(self) -> "tuple[Optional[int], int]":
        files = self._file_browser.get_files()
        current = self.get_current_file()
        index = files.index(current) + 1 if current in files else None
        return index, len(files)

    def go_to_file(self, path: str) -> Optional[str]:
        return self._file_browser.find(search_text=path, exact_match=True)

    def go_to_index(self, index: int) -> Optional[str]:
        try:
            return self._file_browser.go_to_index(index)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------
    def get_base_dir(self) -> Optional[str]:
        return self._base_dir

    def set_base_dir(self, path: str) -> None:
        self._base_dir = path
        self._file_browser.set_directory(path)

    # ------------------------------------------------------------------
    # Marks / file operations
    # ------------------------------------------------------------------
    def list_marks(self) -> list:
        return list(MarkedFiles.file_marks)

    def toggle_mark(self, path: Optional[str]) -> "tuple[bool, str]":
        filepath = path if path is not None else self.get_current_file()
        if not filepath:
            raise ValueError("no file to mark")
        try:
            marked = MarkedFiles.toggle_mark(filepath, self._actions)
        except Exception as e:
            raise ValueError(str(e))
        return marked, filepath

    def go_to_mark(self, backward: bool) -> Optional[str]:
        try:
            marked_file, _wrapped = MarkedFiles.advance_mark_cursor(backward=backward)
        except Exception as e:
            raise ValueError(str(e))
        return self.go_to_file(marked_file)

    def clear_marks(self) -> int:
        cleared = len(MarkedFiles.file_marks)
        if not MarkedFiles.clear_file_marks(self._actions):
            raise ValueError("marks are locked while a transfer is in progress")
        return cleared

    def add_marks_series(self) -> dict:
        """Mark the run of files between the last existing mark and the
        current file, in file-listing order (`FileBrowser.select_series`).

        There is no headless equivalent of the Qt hotkey's compare-mode
        variants (series within the current compare group, or every
        matched file) -- headless has no "mode" concept to branch on, so
        this only ever does the browse-listing version.
        """
        if not MarkedFiles.file_marks:
            raise ValueError("no existing mark to start the series from")
        current = self.get_current_file()
        if not current:
            raise ValueError("no current file")
        if current in MarkedFiles.file_marks:
            return {"added": 0, "marks": list(MarkedFiles.file_marks)}
        files = self._file_browser.select_series(MarkedFiles.file_marks[-1], current)
        try:
            added = MarkedFiles.add_series(files, self._actions)
        except Exception as e:
            raise ValueError(str(e))
        return {"added": added, "marks": list(MarkedFiles.file_marks)}

    def move_marks(self, target_dir: str, copy: bool) -> dict:
        from utils.utils import Utils

        if not copy and self.is_compare_running():
            raise ValueError("cannot move marks while a compare is running")
        if MarkedFiles.is_performing_action:
            raise ValueError("a marks transfer is already in progress")

        already_present, had_errors = MarkedFiles.move_marks_to_dir_static(
            self._actions, target_dir=target_dir,
            move_func=Utils.copy_file if copy else Utils.move_file,
        )
        return {
            "already_present": already_present,
            "had_errors": had_errors,
            "marks_remaining": list(MarkedFiles.file_marks),
        }

    def delete_file(self, path: str) -> None:
        MarkedFiles.delete_file_static(path, self._actions, toast=False, manual_delete=True)

    def hide_current_file(self, path: Optional[str]) -> None:
        filepath = path if path is not None else self.get_current_file()
        if filepath is not None:
            self._hide_media(filepath)
        self.next_file()

    def _hide_media(self, media_path: str) -> None:
        # No navigation: the HIDE callback's caller is a skip loop that
        # already steps past the file, or a batch run that shouldn't move.
        if media_path not in self._compare_manager.hidden_media:
            self._compare_manager.hidden_media.append(media_path)

    # ------------------------------------------------------------------
    # Directory-wide operations
    #
    # Each wraps an existing survey/execute pair from image/directory_ops.py
    # -- already Qt-free, already taking an explicit decision (overwrite or
    # not) instead of reading one from a dialog -- so no new logic here
    # beyond picking the file scope and calling it. Mirrors the shape
    # scripts/agent_headless_demo.py's demo_directory_ops already uses.
    # ------------------------------------------------------------------
    def convert_to_jpg(self, overwrite_existing: bool) -> dict:
        from image import directory_ops

        files = self._file_browser.get_files()
        survey = directory_ops.survey_jpg_conversion(files)
        result = directory_ops.convert_files_to_jpg(survey, overwrite_existing=overwrite_existing)
        if result.converted > 0:
            self._file_browser.refresh()
        return {"converted": result.converted, "failed": result.failed, "skipped_existing": result.skipped_existing}

    def convert_svg_to_png(self, overwrite_existing: bool) -> dict:
        from image import directory_ops

        files = self._file_browser.get_files()
        survey = directory_ops.survey_svg_conversion(files)
        result = directory_ops.convert_svgs_to_png(survey, overwrite_existing=overwrite_existing)
        if result.converted > 0:
            self._file_browser.refresh()
        return {"converted": result.converted, "failed": result.failed, "skipped_existing": result.skipped_existing}

    def scale_images(self, target_side: int) -> dict:
        from image import directory_ops

        files = self._file_browser.get_files()
        survey = directory_ops.survey_image_scaling(files, target_side)
        result = directory_ops.scale_images(survey)
        if result.scaled > 0:
            self._file_browser.refresh()
        return {"scaled": result.scaled, "skipped": result.skipped, "failed": result.failed}

    def strip_video_metadata(self) -> dict:
        from image import directory_ops

        files = self._file_browser.get_files()
        survey = directory_ops.survey_video_metadata_strip(files)
        result = directory_ops.strip_video_metadata(survey)
        if result.written > 0:
            self._file_browser.refresh()
        return {"written": result.written, "failed": result.failed}

    def extract_peek_frames(
        self, media_path: Optional[str] = None, k: Optional[int] = None,
        fps: Optional[float] = None, target_dir: Optional[str] = None,
    ) -> dict:
        from image.peek_frame_selector import extract_peek_frames

        path = media_path or self.get_current_file()
        if not path:
            raise ValueError("no current file to extract frames from")
        try:
            outcome = extract_peek_frames(path, k=k, fps=fps, target_dir=target_dir)
        except RuntimeError as e:
            raise ValueError(str(e))
        if outcome.frames_written:
            self._file_browser.refresh()
        return {"media_path": path, "frames_written": outcome.frames_written}

    def extract_peek_frames_batch(
        self, k: Optional[int] = None, fps: Optional[float] = None,
        target_dir: Optional[str] = None,
    ) -> dict:
        from image import directory_ops

        files = self._file_browser.get_files()
        survey = directory_ops.survey_peek_extraction(files)
        result = directory_ops.extract_peek_frames_for_directory(
            survey, k=k, fps=fps, target_dir=target_dir,
        )
        if result.frames_written > 0:
            self._file_browser.refresh()
        return {
            "extracted": result.extracted,
            "frames_written": result.frames_written,
            "failed": result.failed,
            "skipped": result.skipped,
        }

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    def _resolve_compare_mode(self, mode: str) -> CompareMode:
        try:
            return CompareMode[mode]
        except KeyError:
            raise ValueError(f"unknown compare mode: {mode}")

    def _compare_threshold(self, compare_mode: CompareMode) -> float:
        # COLOR_MATCHING reads compare_threshold as a LAB colour distance
        # rather than an embedding similarity -- see
        # scripts/agent_headless_demo.py's API notes on this same gotcha.
        return (
            config.color_diff_threshold
            if compare_mode == CompareMode.COLOR_MATCHING
            else config.embedding_similarity_threshold
        )

    def _resolve_run_mode(self, run_mode: str) -> Mode:
        resolved = Mode.__members__.get(run_mode)
        if resolved not in (Mode.GROUP, Mode.GROUP_COMPLEMENT):
            raise ValueError(f"run_mode must be GROUP or GROUP_COMPLEMENT, not {run_mode}")
        return resolved

    def _start_compare(self, compare_mode: CompareMode, compare_args: CompareArgs,
                       complement: bool = False) -> None:
        try:
            self._runner.start(
                self._run_compare_task, [compare_mode, compare_args, complement],
                on_error=lambda msg: logger.error("Compare run failed: %s", msg),
            )
        except RuntimeError:
            raise ValueError("a compare is already running")

    def _run_compare_task(self, compare_mode: CompareMode, compare_args: CompareArgs,
                          complement: bool) -> None:
        # CompareManager.run compares with its own primary mode and ignores
        # compare_args.compare_mode. Switched here, on the runner thread, so a
        # start refused because a run is in flight leaves that run's mode alone.
        if compare_mode != self._compare_manager.compare_mode:
            self._compare_manager.set_compare_mode(compare_mode)
        self._compare_manager.run(compare_args)
        if complement:
            self._compare_manager.enter_complement_mode()

    def run_compare(self, mode: str, find_duplicates: bool, run_mode: str = "GROUP") -> None:
        """*run_mode* GROUP_COMPLEMENT runs the same GROUP compare, then
        switches to the files it left ungrouped, as the GUI's "View ungrouped
        files" button does. That button exists only after a plain GROUP run,
        so find_duplicates is refused with it. If every file was grouped the
        session stays in GROUP mode (compare_results' run_mode says which).
        """
        compare_mode = self._resolve_compare_mode(mode)
        complement = self._resolve_run_mode(run_mode) == Mode.GROUP_COMPLEMENT
        if complement and find_duplicates:
            raise ValueError("run_mode GROUP_COMPLEMENT cannot be combined with find_duplicates")
        compare_args = CompareArgs(
            base_dir=self._base_dir,
            mode=Mode.GROUP,
            compare_mode=compare_mode,
            recursive=False,
            store_checkpoints=False,
            app_actions=self._actions,
            compare_threshold=self._compare_threshold(compare_mode),
        )
        compare_args.find_duplicates = find_duplicates
        self._start_compare(compare_mode, compare_args, complement=complement)

    def run_search(
        self,
        mode: str,
        search_text: Optional[str] = None,
        search_text_negative: Optional[str] = None,
        search_media_path: Optional[str] = None,
        negative_search_media_path: Optional[str] = None,
    ) -> None:
        if not any([search_text, search_text_negative, search_media_path, negative_search_media_path]):
            raise ValueError(
                "run_search needs at least one of search_text, search_text_negative, "
                "search_media_path, negative_search_media_path"
            )
        compare_mode = self._resolve_compare_mode(mode)
        compare_args = CompareArgs(
            base_dir=self._base_dir,
            mode=Mode.SEARCH,
            compare_mode=compare_mode,
            recursive=False,
            store_checkpoints=False,
            app_actions=self._actions,
            compare_threshold=self._compare_threshold(compare_mode),
            search_text=search_text,
            search_text_negative=search_text_negative,
            search_media_path=search_media_path,
        )
        # Not a CompareArgs constructor parameter -- only settable as an
        # attribute after construction.
        compare_args.negative_search_media_path = negative_search_media_path
        self._start_compare(compare_mode, compare_args)

    def is_compare_running(self) -> bool:
        return self._runner.is_running()

    def get_compare_mode(self) -> str:
        return self._compare_manager.compare_mode.name

    def compare_results(self) -> dict:
        """Results with files that no longer exist left out.

        Nothing here updates the compare state when files are deleted or
        moved -- the Qt window does that through its refresh/delete handlers,
        and headless has neither -- so gone files are filtered at read time
        (one stat per listed path). As on a Qt delete, a group that losing
        members leaves with fewer than two is dropped; a group the engine
        itself returned with one member is kept.
        """
        cm = self._compare_manager
        file_groups = {}
        for index, group in cm.file_groups.items():
            present = {path: score for path, score in group.items() if os.path.exists(path)}
            if len(present) == len(group) or len(present) > 1:
                file_groups[index] = present
        return {
            "has_compare": cm.has_compare(),
            "run_mode": self._compare_master.mode.name,
            "file_groups": file_groups,
            "files_matched": [path for path in cm.files_matched if os.path.exists(path)],
        }

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def run_image_generation(
        self, edit_suffix: Optional[str], target_dir: Optional[str],
        media_path: Optional[str] = None,
    ) -> None:
        """*media_path* defaults to the current file -- the MCP tool never
        passes it explicitly. The prevalidation GENERATE callback does: the
        file a prevalidation rule matched is not necessarily the file
        currently displayed (or, headlessly, the file browser's current
        cursor position), so it has to be threaded through rather than
        assumed.
        """
        from extensions.sd_runner_client import SDRunnerClient

        if media_path is None:
            media_path = self.get_current_file()
        if media_path is None:
            raise ValueError("no current file to generate from")
        SDRunnerClient().run(
            _DEFAULT_IMAGE_GENERATION_TYPE, media_path,
            edit_suffix=edit_suffix, target_dir=target_dir,
        )

    def _generate_for_callback(
        self, media_path: Optional[str] = None, edit_suffix: Optional[str] = None,
        suppress_toast: bool = False, target_dir: Optional[str] = None,
    ) -> None:
        """The run_image_generation action, in the keyword shape
        AppActions._build_callbacks calls it with. suppress_toast is
        display-only; there is no toast to suppress here."""
        self.run_image_generation(edit_suffix, target_dir, media_path=media_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a Weidr browsing/comparison session over MCP, with no GUI.",
    )
    parser.add_argument("--base-dir", required=True, help="directory to browse")
    parser.add_argument("--host", default=None, help="default: config.mcp_server_host")
    parser.add_argument("--port", type=int, default=None, help="default: config.mcp_server_port")
    parser.add_argument("--token", default=None, help="default: config.mcp_server_token")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.base_dir):
        parser.error(f"not a directory: {args.base_dir}")

    session = HeadlessMCPSession(args.base_dir)
    server = MCPServerExtension(
        session_resolver=lambda: session,
        host=args.host, port=args.port, token=args.token,
    )
    refusal = server.refuses_to_start()
    if refusal:
        logger.error("MCP server not started: %s", refusal)
        return 1

    logger.info("Serving MCP for %s", args.base_dir)
    started = server.start()
    return 0 if started else 1


if __name__ == "__main__":
    sys.exit(main())

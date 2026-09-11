"""Adapts a live AppWindow to the session interface extensions/mcp_server.py
calls through.

Every method here is a thin call through the window's existing
``app_actions`` -- already wrapped onto the GUI thread by
``AppWindow._build_app_actions()`` -- so this file does no thread-marshaling
of its own. Kept out of extensions/mcp_server.py so that module stays free
of any Qt dependency; app_headless.py provides the other implementation of
the same session interface, against a Qt-free FileBrowser/CompareManager
pair instead of a live window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from compare.compare_args import CompareArgs
from utils.config import config
from utils.constants import CompareMode, Mode

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow


class QtWindowMCPSession:
    """Session backed by one open ``AppWindow``."""

    def __init__(self, window: "AppWindow"):
        self._window = window
        self._actions = window.app_actions

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def get_current_file(self) -> Optional[str]:
        return self._actions.get_active_media_filepath()

    def next_file(self) -> Optional[str]:
        self._actions.show_next_media()
        return self.get_current_file()

    def previous_file(self) -> Optional[str]:
        self._actions.show_prev_media()
        return self.get_current_file()

    def get_index(self) -> "tuple[Optional[int], int]":
        files = self._window.file_browser.get_files()
        current = self.get_current_file()
        index = files.index(current) + 1 if current in files else None
        return index, len(files)

    def go_to_file(self, path: str) -> Optional[str]:
        found = self._actions.go_to_file(search_text=path)
        return self.get_current_file() if found else None

    def go_to_index(self, index: int) -> Optional[str]:
        found = self._actions.go_to_file_by_index(index)
        return self.get_current_file() if found else None

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------
    def get_base_dir(self) -> Optional[str]:
        return self._actions.get_base_dir()

    def set_base_dir(self, path: str) -> None:
        self._actions.set_base_dir(path)

    # ------------------------------------------------------------------
    # Marks / file operations
    # ------------------------------------------------------------------
    def list_marks(self) -> list:
        from files.marked_files import MarkedFiles
        return list(MarkedFiles.file_marks)

    def toggle_mark(self, path: Optional[str]) -> "tuple[bool, str]":
        from files.marked_files import MarkedFiles

        filepath = path if path is not None else self.get_current_file()
        if not filepath:
            raise ValueError("no file to mark")
        try:
            marked = MarkedFiles.toggle_mark(filepath, self._actions)
        except Exception as e:
            raise ValueError(str(e))
        return marked, filepath

    def go_to_mark(self, backward: bool) -> Optional[str]:
        from files.marked_files import MarkedFiles

        try:
            marked_file, _wrapped = MarkedFiles.advance_mark_cursor(backward=backward)
        except Exception as e:
            raise ValueError(str(e))
        return self.go_to_file(marked_file)

    def clear_marks(self) -> int:
        from files.marked_files import MarkedFiles

        cleared = len(MarkedFiles.file_marks)
        if not MarkedFiles.clear_file_marks(self._actions):
            raise ValueError("marks are locked while a transfer is in progress")
        return cleared

    def add_marks_series(self) -> dict:
        """Mark the run of files between the last existing mark and the
        current file, in file-listing order.

        Deliberately narrower than the Qt hotkey this is drawn from
        (`FileMarksController.add_all_marks_from_last_or_current_group`),
        which also has a compare-mode variant (series within the current
        compare group, or -- Alt held -- every matched file). This always
        uses `FileBrowser.select_series`, browse-listing order, regardless
        of what mode the window is actually in.
        """
        from files.marked_files import MarkedFiles

        if not MarkedFiles.file_marks:
            raise ValueError("no existing mark to start the series from")
        current = self.get_current_file()
        if not current:
            raise ValueError("no current file")
        if current in MarkedFiles.file_marks:
            return {"added": 0, "marks": list(MarkedFiles.file_marks)}
        files = self._window.file_browser.select_series(MarkedFiles.file_marks[-1], current)
        try:
            added = MarkedFiles.add_series(files, self._actions)
        except Exception as e:
            raise ValueError(str(e))
        return {"added": added, "marks": list(MarkedFiles.file_marks)}

    def move_marks(self, target_dir: str, copy: bool) -> dict:
        from files.marked_files import MarkedFiles
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
        self._actions.delete(path)

    def hide_current_file(self, path: Optional[str]) -> None:
        self._actions.hide_current_media(media_path=path)

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

        files = self._window.file_browser.get_files()
        survey = directory_ops.survey_jpg_conversion(files)
        result = directory_ops.convert_files_to_jpg(survey, overwrite_existing=overwrite_existing)
        if result.converted > 0:
            self._actions.refresh()
        return {"converted": result.converted, "failed": result.failed, "skipped_existing": result.skipped_existing}

    def convert_svg_to_png(self, overwrite_existing: bool) -> dict:
        from image import directory_ops

        files = self._window.file_browser.get_files()
        survey = directory_ops.survey_svg_conversion(files)
        result = directory_ops.convert_svgs_to_png(survey, overwrite_existing=overwrite_existing)
        if result.converted > 0:
            self._actions.refresh()
        return {"converted": result.converted, "failed": result.failed, "skipped_existing": result.skipped_existing}

    def scale_images(self, target_side: int) -> dict:
        from image import directory_ops

        files = self._window.file_browser.get_files()
        survey = directory_ops.survey_image_scaling(files, target_side)
        result = directory_ops.scale_images(survey)
        if result.scaled > 0:
            self._actions.refresh()
        return {"scaled": result.scaled, "skipped": result.skipped, "failed": result.failed}

    def strip_video_metadata(self) -> dict:
        from image import directory_ops

        files = self._window.file_browser.get_files()
        survey = directory_ops.survey_video_metadata_strip(files)
        result = directory_ops.strip_video_metadata(survey)
        if result.written > 0:
            self._actions.refresh()
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
            self._actions.refresh()
        return {"media_path": path, "frames_written": outcome.frames_written}

    def extract_peek_frames_batch(
        self, k: Optional[int] = None, fps: Optional[float] = None,
        target_dir: Optional[str] = None,
    ) -> dict:
        from image import directory_ops

        files = self._window.file_browser.get_files()
        survey = directory_ops.survey_peek_extraction(files)
        result = directory_ops.extract_peek_frames_for_directory(
            survey, k=k, fps=fps, target_dir=target_dir,
        )
        if result.frames_written > 0:
            self._actions.refresh()
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

    def run_compare(self, mode: str, find_duplicates: bool, run_mode: str = "GROUP") -> None:
        # GROUP_COMPLEMENT is headless-only: here it would have to be entered
        # after SearchController's worker finishes, which it offers no hook for.
        if run_mode != Mode.GROUP.name:
            raise ValueError(
                f"run_mode {run_mode} is not supported in an app window session; "
                "run GROUP and use the window's View ungrouped files button"
            )
        compare_mode = self._resolve_compare_mode(mode)
        # SearchController._run_compare overwrites CompareArgs.mode with the
        # window's own self._app.mode before running, so the mode this call
        # actually gets depends on that, not on what's set below -- set it
        # explicitly first so a stale SEARCH mode from earlier UI activity
        # doesn't leak into this run.
        self._actions.set_mode(Mode.GROUP)
        self._actions.run_compare(
            CompareArgs(
                base_dir=self.get_base_dir(),
                mode=Mode.GROUP,
                compare_mode=compare_mode,
                app_actions=self._actions,
                compare_threshold=self._compare_threshold(compare_mode),
            ),
            find_duplicates=find_duplicates,
        )

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
            base_dir=self.get_base_dir(),
            mode=Mode.SEARCH,
            compare_mode=compare_mode,
            app_actions=self._actions,
            compare_threshold=self._compare_threshold(compare_mode),
            search_text=search_text,
            search_text_negative=search_text_negative,
            search_media_path=search_media_path,
        )
        # Not a CompareArgs constructor parameter -- only settable as an
        # attribute after construction.
        compare_args.negative_search_media_path = negative_search_media_path
        # See run_compare's note: SearchController._run_compare reads the
        # window's own mode rather than trusting compare_args.mode, so it has
        # to be set here too.
        self._actions.set_mode(Mode.SEARCH)
        self._actions.run_compare(compare_args, find_duplicates=False)

    def is_compare_running(self) -> bool:
        return self._actions.is_compare_running()

    def get_compare_mode(self) -> str:
        return self._actions.get_compare_mode().name

    def compare_results(self) -> dict:
        cm = self._window.compare_manager
        return {
            "has_compare": cm.has_compare(),
            "run_mode": self._window.mode.name,
            "file_groups": dict(cm.file_groups),
            "files_matched": list(cm.files_matched),
        }

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def run_image_generation(self, edit_suffix: Optional[str], target_dir: Optional[str]) -> None:
        self._actions.run_image_generation(
            edit_suffix=edit_suffix, suppress_toast=True, target_dir=target_dir,
        )

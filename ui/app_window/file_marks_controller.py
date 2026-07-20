"""FileMarksController -- mark-related operations."""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import QApplication

from files.file_action import FileAction
from files.related_image import (
    get_downstream_related_images,
    get_downstream_files_for_sources,
    get_sources_with_downstream_in_dir,
    validate_related_image_suffix,
)
from files.marked_files import MarkedFiles
from ui.files.marked_file_mover_qt import MarkedFileMover
from ui.auth.password_utils import require_password
from utils.config import config
from utils.constants import Mode, ProtectedActions
from utils.logging_setup import get_logger
from utils.translations import _, marks_transfer_running_warn
from utils.utils import ModifierKey, Utils

if TYPE_CHECKING:
    from compare.compare_manager import CompareManager
    from files.file_browser import FileBrowser
    from ui.app_window.app_window import AppWindow
    from ui.app_window.media_navigator import MediaNavigator
logger = get_logger("file_marks_controller")


class FileMarksController:
    """
    Owns all mark-related operations: adding/removing marks, navigating
    to marked files, copying marks, opening the move-marks window,
    and running previous/permanent mark actions.
    """

    def __init__(
        self,
        app_window: AppWindow,
        file_browser: FileBrowser,
        compare_manager: CompareManager,
        media_navigator: MediaNavigator,
    ):
        self._app = app_window
        self._fb = file_browser
        self._cm = compare_manager
        self._nav = media_navigator

    # ==================================================================
    # Mark operations
    # ==================================================================
    def add_or_remove_mark(
        self, event=None, show_toast: bool = True, filepath: Optional[str] = None
    ) -> None:
        """Toggle a mark on the current (or specified) file."""
        if not MarkedFiles.guard_mark_mutation(
            self._app.app_actions, _("Add or remove a mark")
        ):
            return
        if filepath is None:
            filepath = self._app.media_path
        if self._app.delete_lock:
            warning = _("DELETE_LOCK_MARK_STOP")
            self._app.app_actions.warn(warning)
            raise Exception(warning)
            # NOTE: Exception prevents downstream events from using empty marks

        self._check_marks(min_mark_size=0)

        if filepath in MarkedFiles.file_marks:
            MarkedFiles.file_marks.remove(filepath)
            remaining = len(MarkedFiles.file_marks)
            if MarkedFiles.mark_cursor >= remaining:
                MarkedFiles.mark_cursor = -1
            if show_toast:
                self._app.notification_ctrl.toast(
                    _("Mark removed. Remaining: {0}").format(remaining)
                )
        else:
            MarkedFiles.file_marks.append(filepath)
            if show_toast:
                self._app.notification_ctrl.toast(
                    _("Mark added. Total set: {0}").format(len(MarkedFiles.file_marks))
                )

    def add_all_marks_from_last_or_current_group(self, event=None) -> None:
        """
        Add all files from the last or current group/series to the mark list.

        In BROWSE mode, selects files between the last mark and the current media file.
        In compare modes, Alt selects all matches; otherwise selects series.
        """
        if not MarkedFiles.guard_mark_mutation(
            self._app.app_actions, _("Add marks from group")
        ):
            return
        if self._app.mode == Mode.BROWSE:
            if self._app.media_path in MarkedFiles.file_marks:
                return
            self._check_marks()
            files = self._fb.select_series(
                start_file=MarkedFiles.file_marks[-1], end_file=self._app.media_path
            )
        else:
            alt_pressed = (
                Utils.modifier_key_pressed(event, keys_to_check=[ModifierKey.ALT])
                if event is not None
                else False
            )
            if alt_pressed:
                files = list(self._cm.files_matched)
            else:
                files = self._cm.select_series(
                    start_file=MarkedFiles.file_marks[-1], end_file=self._app.media_path
                )

        for _file in files:
            if _file not in MarkedFiles.file_marks:
                MarkedFiles.file_marks.append(_file)

        self._app.notification_ctrl.toast(
            _("Marks added. Total set: {0}").format(len(MarkedFiles.file_marks))
        )

    def go_to_mark(self, event=None) -> None:
        """Navigate to the next (or previous, if Alt is held) marked file."""
        self._check_marks()

        alt_pressed = (
            Utils.modifier_key_pressed(event, keys_to_check=[ModifierKey.ALT])
            if event is not None
            else False
        )
        MarkedFiles.mark_cursor += -1 if alt_pressed else 1
        if MarkedFiles.mark_cursor >= len(MarkedFiles.file_marks):
            MarkedFiles.mark_cursor = 0
            if len(MarkedFiles.file_marks) > 1:
                self._app.notification_ctrl.toast(_("First sorted mark"))

        marked_file = MarkedFiles.file_marks[MarkedFiles.mark_cursor]

        if self._app.mode == Mode.BROWSE:
            self._fb.go_to_file(marked_file)
            self._nav.create_media(marked_file)
            if len(MarkedFiles.file_marks) == 1:
                self._app.notification_ctrl.toast(_("Only one marked file set."))
        else:
            self._nav.go_to_file(search_text=os.path.basename(marked_file))

    def copy_marks_list(self, event=None) -> None:
        """Copy the list of marked files to the clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(str(MarkedFiles.file_marks))

    # ==================================================================
    # Move marks window
    # ==================================================================
    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def open_move_marks_window(
        self,
        event=None,
        open_gui: bool = True,
        override_marks: Optional[list[str]] = None,
        filepath: Optional[str] = None,
    ) -> None:
        """Open the move-marks window."""
        if override_marks is None:
            override_marks = []

        self._check_marks(min_mark_size=0)

        if filepath:
            if not os.path.exists(filepath):
                self._app.notification_ctrl.alert(
                    _("Invalid file path"),
                    _("The file path {0} is invalid.").format(filepath),
                    kind="error",
                )
                return
            if not MarkedFiles.add_mark_if_not_present(
                filepath, app_actions=self._app.app_actions
            ):
                return
        else:
            filepath = self._nav.get_active_media_filepath()

        if len(override_marks) > 0:
            if not MarkedFiles.guard_mark_mutation(
                self._app.app_actions, _("Add marks")
            ):
                return
            logger.debug(_("Including marks: {0}").format(override_marks))
            MarkedFiles.file_marks.extend(override_marks)

        current_media = filepath
        single_media = False
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(filepath=filepath)
            single_media = True

        try:
            MarkedFileMover.show_window(
                self._app,  # parent widget for the window
                open_gui,
                single_media,
                current_media,
                self._app.mode,
                self._app.app_actions,
                base_dir=self._app.get_base_dir(),
            )
        except Exception as e:
            self._app.notification_ctrl.handle_error(
                str(e), title="Marked Files Window Error"
            )

    # ==================================================================
    # Quick-action mark operations
    # ==================================================================
    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_previous_marks_action(self, event=None) -> None:
        """Re-run the previously used marks action."""
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        prev = FileAction.get_history_action(start_index=0)
        MarkedFileMover.run_marks_action_with_progress(
            self._app, self._nav.get_active_media_filepath(),
            MarkedFiles.run_previous_action, prev.action if prev else None,
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_penultimate_marks_action(self, event=None) -> None:
        """Re-run the second-to-last marks action."""
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        prev = FileAction.get_history_action(start_index=1)
        MarkedFileMover.run_marks_action_with_progress(
            self._app, self._nav.get_active_media_filepath(),
            MarkedFiles.run_penultimate_action, prev.action if prev else None,
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_antepenultimate_marks_action(self, event=None) -> None:
        """Re-run the third-to-last marks action."""
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        prev = FileAction.get_history_action(start_index=2)
        MarkedFileMover.run_marks_action_with_progress(
            self._app, self._nav.get_active_media_filepath(),
            MarkedFiles.run_antepenultimate_action, prev.action if prev else None,
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_permanent_marks_action(self, event=None) -> None:
        """Run the permanently-configured marks action."""
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        perm = FileAction.permanent_action
        MarkedFileMover.run_marks_action_with_progress(
            self._app, self._nav.get_active_media_filepath(),
            MarkedFiles.run_permanent_action, perm.action if perm else None,
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_hotkey_marks_action(
        self, number: int, shift_pressed: bool = False
    ) -> None:
        """
        Run the hotkey-bound marks action for the given digit.

        The digit and shift state are captured by the ``KeyBindingManager``
        closure rather than extracted from a UI event.
        """
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        hotkey = FileAction.hotkey_actions.get(number)
        move_func = hotkey.get_action(do_flip=shift_pressed) if hotkey else None
        MarkedFileMover.run_marks_action_with_progress(
            self._app, self._nav.get_active_media_filepath(),
            partial(MarkedFiles.run_hotkey_action, number=number, shift_key_pressed=shift_pressed),
            move_func,
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_file_action_set(self, event=None) -> None:
        """Execute the currently selected file action set on the current file."""
        from files.file_action_set import FileActionSets
        if MarkedFiles.is_transfer_running():
            self._app.app_actions.warn(
                marks_transfer_running_warn(_("Run file action set"))
            )
            return
        selected = FileActionSets.get_selected_actions()
        if not selected:
            self._app.app_actions.warn(_("No file action set actions selected."))
            return
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        marks_snapshot = list(MarkedFiles.file_marks)
        current_media = self._nav.get_active_media_filepath()
        copy_actions = [s for s in selected if not s.is_move()]
        move_actions = [s for s in selected if s.is_move()]
        total = len(marks_snapshot)
        for step in copy_actions:
            progress, callback = MarkedFileMover.build_marks_progress(
                self._app, total, Utils.copy_file
            )
            MarkedFiles.move_marks_to_dir_static(
                self._app.app_actions,
                target_dir=step.target,
                move_func=Utils.copy_file,
                files=marks_snapshot,
                single_image=(len(marks_snapshot) == 1),
                current_media=current_media,
                get_target_dir_callback=MarkedFileMover.get_target_directory,
                progress_callback=callback,
            )
            if progress is not None:
                progress.setValue(total)
        for step in move_actions:
            progress, callback = MarkedFileMover.build_marks_progress(
                self._app, total, Utils.move_file
            )
            MarkedFiles.move_marks_to_dir_static(
                self._app.app_actions,
                target_dir=step.target,
                move_func=Utils.move_file,
                files=marks_snapshot,
                single_image=(len(marks_snapshot) == 1),
                current_media=current_media,
                get_target_dir_callback=MarkedFileMover.get_target_directory,
                progress_callback=callback,
            )
            if progress is not None:
                progress.setValue(total)

    def create_interpolation_gif(self, event=None) -> None:
        """Create a smooth optical-flow interpolation GIF from exactly 2 marked files."""
        MarkedFileMover.create_interpolation_gif_from_marks(self._app.app_actions)

    def _check_marks(self, min_mark_size: int = 1) -> None:
        """Validate that enough marks exist for the intended operation."""
        if len(MarkedFiles.file_marks) < min_mark_size:
            exception_text = _("NO_MARKS_SET").format(
                len(MarkedFiles.file_marks), min_mark_size
            )
            self._app.app_actions.warn(exception_text)
            raise Exception(exception_text)

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def revert_last_marks_change(self, event=None) -> None:
        """Undo the last marks change."""
        if not config.use_file_paths_json:
            MarkedFileMover.undo_move_marks(self._app.get_base_dir(), self._app.app_actions)

    # ==================================================================
    # Related media / downstream marks
    # ==================================================================
    def _notify_related_result(self, message: str, action_label: str, **data) -> None:
        """Surface a related-image action outcome to the related images
        window's result area (no-op when it isn't open)."""
        self._app.app_actions.notify_related_images_result(
            message, action_label=action_label, data=data or None
        )

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def set_marks_from_downstream_related_images(
        self,
        event=None,
        base_dir: Optional[str] = None,
        media_to_use: Optional[str] = None,
    ) -> None:
        """Set marks from downstream related images found in another directory."""
        from ui.image.media_details import MediaDetails
        from ui.app_window.window_manager import WindowManager

        if base_dir is None:
            window, dirs = WindowManager.get_other_window_or_self_dir(
                self._app, allow_current_window=True
            )
            if window is None:
                self._app.window_launcher.open_recent_directory_window(
                    extra_callback_args=(self.set_marks_from_downstream_related_images, dirs)
                )
                return
            base_dir = dirs[0]
        else:
            window = WindowManager.get_window(base_dir=base_dir)

        if media_to_use is None:
            media_to_use = (
                self._app.media_path
                if len(MarkedFiles.file_marks) != 1
                else MarkedFiles.file_marks[0]
            )

        if self._app.check_many_files(window, action="find related media"):
            return

        downstream_related_images = get_downstream_related_images(
            media_to_use, base_dir, self._app.app_actions, force_refresh=True
        )
        if downstream_related_images is not None:
            if not MarkedFiles.guard_mark_mutation(
                self._app.app_actions, _("Set marks from related media")
            ):
                return
            MarkedFiles.file_marks = downstream_related_images
            message = _("{0} file marks set").format(len(downstream_related_images))
            self._app.notification_ctrl.toast(message)
            self._notify_related_result(
                message, _("Set marks from downstream images"),
                found=len(downstream_related_images), base_dir=base_dir,
            )
            window.file_marks_ctrl.go_to_mark()
            window.media_frame.setFocus()
        else:
            self._notify_related_result(
                _("No downstream related images found in") + f"\n{base_dir}",
                _("Set marks from downstream images"),
                found=0, base_dir=base_dir,
            )

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def set_marks_from_downstream_related_images_all_windows(self, event=None) -> None:
        """Set marks from downstream related images across all open windows.

        Multi-target generalization of set_marks_from_downstream_related_images:
        every open window's base directory (current window included,
        de-duplicated) is searched and the results are unioned. Oversized
        unconfirmed directories are skipped with a note in the summary rather
        than aborting the whole search.
        """
        from ui.app_window.window_manager import WindowManager

        media_to_use = (
            self._app.media_path
            if len(MarkedFiles.file_marks) != 1
            else MarkedFiles.file_marks[0]
        )
        if not media_to_use:
            self._app.notification_ctrl.toast(_("No active media to search from."))
            return

        searched_dirs: list[str] = []
        skipped_dirs: list[str] = []
        seen_dirs: set = set()
        found: list[str] = []
        seen_files: set = set()
        first_owner_window = None
        for window in WindowManager.get_open_windows():
            base_dir = getattr(window, "base_dir", None)
            if not base_dir or not os.path.isdir(base_dir):
                continue
            norm_dir = os.path.normpath(os.path.abspath(base_dir))
            if norm_dir in seen_dirs:
                continue
            seen_dirs.add(norm_dir)
            # Same size condition as check_many_files, but non-interactive:
            # skip-with-note instead of prompting/aborting per window.
            if (not window.file_browser.has_confirmed_dir()
                    and window.file_browser.is_slow_total_files(threshold=2000)):
                skipped_dirs.append(base_dir)
                continue
            downstream = get_downstream_related_images(
                media_to_use, base_dir, self._app.app_actions,
                force_refresh=True, quiet=True,
            )
            searched_dirs.append(base_dir)
            for f in downstream or []:
                if f not in seen_files:
                    seen_files.add(f)
                    found.append(f)
            if downstream and first_owner_window is None:
                first_owner_window = window

        if not found:
            message = _("No downstream related images found across {0} open window(s).").format(
                len(searched_dirs))
            if skipped_dirs:
                message += "\n" + _("Skipped {0} large unconfirmed director(ies).").format(
                    len(skipped_dirs))
            self._app.notification_ctrl.toast(message)
            self._notify_related_result(
                message, _("Search all open windows"),
                found=0, searched_dirs=searched_dirs, skipped_dirs=skipped_dirs,
            )
            return

        if not MarkedFiles.guard_mark_mutation(
            self._app.app_actions, _("Set marks from related media (all windows)")
        ):
            return
        MarkedFiles.file_marks = found
        summary = _("{0} file marks set from {1} window(s)").format(
            len(found), len(searched_dirs))
        if skipped_dirs:
            summary += "\n" + _("Skipped {0} large unconfirmed director(ies).").format(
                len(skipped_dirs))
        self._app.notification_ctrl.toast(summary)
        self._notify_related_result(
            summary, _("Search all open windows"),
            found=len(found), searched_dirs=searched_dirs, skipped_dirs=skipped_dirs,
        )
        if first_owner_window is not None:
            first_owner_window.file_marks_ctrl.go_to_mark()
            first_owner_window.media_frame.setFocus()

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def mark_sources_with_downstream_in_dir(
        self,
        event=None,
        base_dir: Optional[str] = None,
    ) -> None:
        """Mark all files in the current dir that have a downstream image in base_dir."""
        from ui.image.media_details import MediaDetails

        if base_dir is None:
            self._app.window_launcher.open_recent_directory_window(
                extra_callback_args=(self.mark_sources_with_downstream_in_dir, [])
            )
            return

        source_paths = self._fb.filepaths
        if not source_paths:
            self._app.notification_ctrl.toast(_("No files in current directory."))
            return

        sources = get_sources_with_downstream_in_dir(source_paths, base_dir)

        if not sources:
            message = _("No source files with downstream images found in\n{0}").format(base_dir)
            self._app.notification_ctrl.toast(message)
            self._notify_related_result(
                message, _("Mark sources with downstream"), found=0, base_dir=base_dir)
            return

        if not MarkedFiles.guard_mark_mutation(
            self._app.app_actions, _("Mark sources with downstream images")
        ):
            return

        MarkedFiles.file_marks = sources
        message = _("{0} source file(s) marked").format(len(sources))
        self._app.notification_ctrl.toast(message)
        self._notify_related_result(
            message, _("Mark sources with downstream"),
            found=len(sources), base_dir=base_dir)
        self.go_to_mark()
        self._app.media_frame.setFocus()

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def mark_downstream_files_in_dir(
        self,
        event=None,
        base_dir: Optional[str] = None,
    ) -> None:
        """Mark all files in base_dir that are downstream of any file in the current dir."""
        from ui.image.media_details import MediaDetails
        from ui.app_window.window_manager import WindowManager

        if base_dir is None:
            self._app.window_launcher.open_recent_directory_window(
                extra_callback_args=(self.mark_downstream_files_in_dir, [])
            )
            return

        source_paths = self._fb.filepaths
        if not source_paths:
            self._app.notification_ctrl.toast(_("No files in current directory."))
            return

        downstream = get_downstream_files_for_sources(source_paths, base_dir)

        if not downstream:
            message = _("No downstream files found in\n{0}").format(base_dir)
            self._app.notification_ctrl.toast(message)
            self._notify_related_result(
                message, _("Mark downstream files"), found=0, base_dir=base_dir)
            return

        if not MarkedFiles.guard_mark_mutation(
            self._app.app_actions, _("Mark downstream files")
        ):
            return

        MarkedFiles.file_marks = downstream
        message = _("{0} downstream file(s) marked").format(len(downstream))
        self._app.notification_ctrl.toast(message)
        self._notify_related_result(
            message, _("Mark downstream files"),
            found=len(downstream), base_dir=base_dir)
        window = WindowManager.get_window(base_dir=base_dir)
        if window is not None:
            window.file_marks_ctrl.go_to_mark()
            window.media_frame.setFocus()

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def set_marked_file_as_related_to_current(self, event=None) -> None:
        """Rename the single marked file to make it a named derivative of the current media.

        The marked file is renamed to <current_stem>_<suffix><marked_ext>, which
        is sufficient for the filename-pattern fallback in get_origin_basename() to
        resolve the relationship without any metadata rewrite.
        """
        from PySide6.QtWidgets import QInputDialog

        if not MarkedFiles.guard_mark_mutation(
            self._app.app_actions, _("Set related image")
        ):
            return

        if len(MarkedFiles.file_marks) != 1:
            self._app.notification_ctrl.toast(
                _("Exactly one file must be marked for this operation.")
            )
            return

        marked_path = MarkedFiles.file_marks[0]
        current_media = self._app.media_path

        if not current_media or not os.path.isfile(current_media):
            self._app.notification_ctrl.toast(_("No valid current media file."))
            return
        if not os.path.isfile(marked_path):
            self._app.notification_ctrl.toast(_("Marked file no longer exists."))
            return

        current_stem = os.path.splitext(os.path.basename(current_media))[0]
        marked_ext = os.path.splitext(marked_path)[1]
        marked_dir = os.path.dirname(marked_path)

        suffix, ok = QInputDialog.getText(
            self._app,
            _("Set as Related Image"),
            _(
                "Enter a suffix (1–8 alphanumeric characters, not purely numeric).\n"
                "The marked file will be renamed to:\n"
                "{0}_<suffix>{1}"
            ).format(current_stem, marked_ext),
        )
        if not ok:
            return

        suffix = suffix.strip()
        error = validate_related_image_suffix(suffix)
        if error:
            self._app.notification_ctrl.alert(
                _("Invalid Suffix"), error, kind="error"
            )
            return

        new_basename = f"{current_stem}_{suffix}{marked_ext}"
        new_path = os.path.join(marked_dir, new_basename)

        if os.path.exists(new_path):
            self._app.notification_ctrl.alert(
                _("File Already Exists"),
                _("A file named {0} already exists in that directory.").format(new_basename),
                kind="error",
            )
            return

        try:
            with Utils.file_operation_lock:
                os.rename(marked_path, new_path)
        except OSError as e:
            self._app.notification_ctrl.alert(
                _("Rename Failed"), str(e), kind="error"
            )
            return

        MarkedFiles.file_marks[0] = new_path
        self._app.notification_ctrl.toast(_("Renamed to {0}").format(new_basename))
        self._app.app_actions.refresh(removed_files=[marked_path])

"""
FileMarksController -- mark-related operations.

Extracted from: add_or_remove_mark, _add_all_marks_from_last_or_current_group,
go_to_mark, copy_marks_list, _check_marks, open_move_marks_window,
run_previous_marks_action, run_penultimate_marks_action,
run_permanent_marks_action, run_hotkey_marks_action, revert_last_marks_change,
set_marks_from_downstream_related_images.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import QApplication

from files.marked_files import MarkedFiles
from ui.files.marked_file_mover_qt import MarkedFileMover
from ui.auth.password_utils import require_password
from utils.config import config
from utils.constants import Mode, ProtectedActions
from utils.logging_setup import get_logger
from utils.translations import I18N
from utils.utils import ModifierKey, Utils

if TYPE_CHECKING:
    from compare.compare_manager import CompareManager
    from files.file_browser import FileBrowser
    from ui.app_window.app_window import AppWindow
    from ui.app_window.media_navigator import MediaNavigator

_ = I18N._
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
        """
        Toggle a mark on the current (or specified) file.

        Ported from App.add_or_remove_mark.
        """
        if filepath is None:
            filepath = self._app.img_path
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

        In BROWSE mode, selects files between the last mark and the current image.
        In compare modes, Alt selects all matches; otherwise selects series.

        Ported from App._add_all_marks_from_last_or_current_group.
        """
        if self._app.mode == Mode.BROWSE:
            if self._app.img_path in MarkedFiles.file_marks:
                return
            self._check_marks()
            files = self._fb.select_series(
                start_file=MarkedFiles.file_marks[-1], end_file=self._app.img_path
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
                    start_file=MarkedFiles.file_marks[-1], end_file=self._app.img_path
                )

        for _file in files:
            if _file not in MarkedFiles.file_marks:
                MarkedFiles.file_marks.append(_file)

        self._app.notification_ctrl.toast(
            _("Marks added. Total set: {0}").format(len(MarkedFiles.file_marks))
        )

    def go_to_mark(self, event=None) -> None:
        """
        Navigate to the next (or previous, if Alt is held) marked file.

        Ported from App.go_to_mark.
        """
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
        """
        Copy the list of marked files to the clipboard.

        Ported from App.copy_marks_list.
        """
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
        """
        Open the move-marks window.

        Ported from App.open_move_marks_window.
        """
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
            MarkedFiles.add_mark_if_not_present(filepath)
        else:
            filepath = self._nav.get_active_media_filepath()

        if len(override_marks) > 0:
            logger.debug(_("Including marks: {0}").format(override_marks))
            MarkedFiles.file_marks.extend(override_marks)

        current_image = filepath
        single_image = False
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(filepath=filepath)
            single_image = True

        try:
            MarkedFileMover.show_window(
                self._app,  # parent widget for the window
                open_gui,
                single_image,
                current_image,
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
        """
        Re-run the previously used marks action.

        Ported from App.run_previous_marks_action.
        """
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        MarkedFiles.run_previous_action(
            self._app.app_actions, self._nav.get_active_media_filepath(), ui_class=MarkedFileMover
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_penultimate_marks_action(self, event=None) -> None:
        """
        Re-run the second-to-last marks action.

        Ported from App.run_penultimate_marks_action.
        """
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        MarkedFiles.run_penultimate_action(
            self._app.app_actions, self._nav.get_active_media_filepath(), ui_class=MarkedFileMover
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_antepenultimate_marks_action(self, event=None) -> None:
        """
        Re-run the third-to-last marks action.
        """
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        MarkedFiles.run_antepenultimate_action(
            self._app.app_actions, self._nav.get_active_media_filepath(), ui_class=MarkedFileMover
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_permanent_marks_action(self, event=None) -> None:
        """
        Run the permanently-configured marks action.

        Ported from App.run_permanent_marks_action.
        """
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        MarkedFiles.run_permanent_action(
            self._app.app_actions, self._nav.get_active_media_filepath(), ui_class=MarkedFileMover
        )

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def run_hotkey_marks_action(
        self, number: int, shift_pressed: bool = False
    ) -> None:
        """
        Run the hotkey-bound marks action for the given digit.

        In the Qt port the digit and shift state are captured by the
        ``KeyBindingManager`` closure rather than extracted from a Tkinter UI event.

        Ported from App.run_hotkey_marks_action.
        """
        if len(MarkedFiles.file_marks) == 0:
            self.add_or_remove_mark(show_toast=False)
        MarkedFiles.run_hotkey_action(
            self._app.app_actions,
            self._nav.get_active_media_filepath(),
            number,
            shift_pressed,
            ui_class=MarkedFileMover
        )

    def _check_marks(self, min_mark_size: int = 1) -> None:
        """
        Validate that enough marks exist for the intended operation.

        Ported from App._check_marks.
        """
        if len(MarkedFiles.file_marks) < min_mark_size:
            exception_text = _("NO_MARKS_SET").format(
                len(MarkedFiles.file_marks), min_mark_size
            )
            self._app.app_actions.warn(exception_text)
            raise Exception(exception_text)

    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def revert_last_marks_change(self, event=None) -> None:
        """
        Undo the last marks change.

        Ported from App.revert_last_marks_change.
        """
        if not config.use_file_paths_json:
            MarkedFileMover.undo_move_marks(self._app.get_base_dir(), self._app.app_actions)

    # ==================================================================
    # Related images / downstream marks
    # ==================================================================
    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def set_marks_from_downstream_related_images(
        self,
        event=None,
        base_dir: Optional[str] = None,
        image_to_use: Optional[str] = None,
    ) -> None:
        """
        Set marks from downstream related images found in another directory.

        Ported from App.set_marks_from_downstream_related_images.
        """
        from ui.image.image_details_qt import ImageDetails
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

        if image_to_use is None:
            image_to_use = (
                self._app.img_path
                if len(MarkedFiles.file_marks) != 1
                else MarkedFiles.file_marks[0]
            )

        if self._app.check_many_files(window, action="find related images"):
            return

        downstream_related_images = ImageDetails.get_downstream_related_images(
            image_to_use, base_dir, self._app.app_actions, force_refresh=True
        )
        if downstream_related_images is not None:
            MarkedFiles.file_marks = downstream_related_images
            self._app.notification_ctrl.toast(
                _("{0} file marks set").format(len(downstream_related_images))
            )
            window.file_marks_ctrl.go_to_mark()
            window.media_frame.setFocus()

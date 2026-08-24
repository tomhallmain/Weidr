"""
FileOpsController -- delete, hide, copy, and file-manipulation operations.

Also owns the periodic file-check timer, which monitors the file system
for changes and refreshes the file list when needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from ui.auth.password_utils import require_password
from utils.config import config
from utils.constants import ActionType, Mode, ProtectedActions, Sort, SortBy
from utils.logging_setup import get_logger
from utils.running_tasks_registry import start_thread
from utils.translations import _, compare_running_warn
from utils.utils import Utils

if TYPE_CHECKING:
    from compare.compare_manager import CompareManager
    from files.file_browser import FileBrowser
    from ui.app_window.app_window import AppWindow
    from ui.app_window.media_navigator import MediaNavigator
logger = get_logger("file_ops_controller")

_RANDOMIZE_FILENAMES_LOG_BASENAME = "randomize_filenames.log"


class FileOpsController:
    """
    Owns delete, hide, copy, and file-manipulation operations
    on the currently viewed media. Also owns the periodic file-check timer.
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

        # Periodic file-check timer
        self._file_check_timer: Optional[QTimer] = None

    # ==================================================================
    # Periodic file check
    # ==================================================================
    def start_file_check_timer(self) -> None:
        """
        Start a periodic timer that refreshes the file list if files changed.

        Replaces the async ``check_files`` coroutine + ``@periodic`` decorator
        from the original App class.
        """
        interval_ms = int(self._app.file_check_config.interval_seconds * 1000)
        if interval_ms <= 0:
            return

        self._file_check_timer = QTimer()
        self._file_check_timer.timeout.connect(self._on_file_check)
        self._file_check_timer.start(interval_ms)

    def stop_file_check_timer(self) -> None:
        if self._file_check_timer is not None:
            self._file_check_timer.stop()
            self._file_check_timer = None

    def _on_file_check(self) -> None:
        """Called on the main thread by QTimer."""
        try:
            show_new_media = self._app.slideshow_config.show_new_media
            if not self._fb.checking_files and not show_new_media:
                # checking_files is False to avoid rechecking large directories, but
                # the new-media slideshow's whole purpose is watching for new files,
                # so it overrides that skip.
                return
            if self._app.mode != Mode.BROWSE:
                return
            base_dir = self._app.get_base_dir()
            if base_dir and base_dir != "":
                self._app.refresh(
                    show_new_media=show_new_media,
                    from_file_check=True,
                )
        except Exception as e:
            logger.debug(f"Error in file check: {e}")

    # ==================================================================
    # Delete operations
    # ==================================================================
    @require_password(ProtectedActions.DELETE_MEDIA)
    def delete_media(self, event=None) -> None:
        """Delete the currently displayed media file from the filesystem."""
        from files.marked_files import MarkedFiles

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("delete files")))
            return

        if self._app.delete_lock:
            self._app.app_actions.warn(_("DELETE_LOCK"))
            return

        if self._app.mode == Mode.BROWSE:
            self._fb.checking_files = False
            filepath = self._fb.current_file()
            if filepath:
                self._app.release_media_canvas()
                self._handle_delete(filepath)
                MarkedFiles.handle_file_removal(filepath)
                self._fb.refresh(
                    refresh_cursor=False,
                    removed_files=[filepath],
                    direction=self._app.direction,
                    file_check=True,
                )
                self._nav.last_chosen_direction_func()
            self._fb.checking_files = True
            return

        if self._app.mode == Mode.GROUP_COMPLEMENT:
            filepath = self._nav.get_active_media_filepath()
            if filepath is not None:
                MarkedFiles.handle_file_removal(filepath)
                self._app.release_media_canvas()
                self._handle_delete(filepath)
                if self._cm.has_compare():
                    self._cm.compare().remove_from_groups([filepath])
                self._cm.remove_from_complement(filepath)
            return

        is_toggled_search_media = self._nav.is_toggled_search_media()

        if len(self._cm.files_matched) == 0 and not is_toggled_search_media:
            self._app.app_actions.warn(_("Invalid action, no files found to delete"))
            return
        elif is_toggled_search_media and (
            self._cm.search_media_path is None
            or self._cm.search_media_path == ""
        ):
            self._app.app_actions.warn(_("Invalid action, search media not found"))
            return

        filepath = self._nav.get_active_media_filepath()

        if filepath is not None:
            MarkedFiles.handle_file_removal(filepath)
            if filepath == self._cm.search_media_path:
                self._cm.search_media_path = None
            self._app.release_media_canvas()
            self._handle_delete(filepath)
            if self._cm.has_compare():
                self._cm.compare().remove_from_groups([filepath])
            self._cm._update_groups_for_removed_file(
                self._app.mode,
                self._cm.current_group_index,
                self._cm.match_index,
                show_next_media=self._app.direction,
            )
            self._cm._sync_result_after_deletion(filepath)
        else:
            self._app.notification_ctrl.handle_error(
                _("Failed to delete current file, unable to get valid filepath")
            )

    def _handle_delete(
        self,
        filepath: str,
        toast: bool = True,
        manual_delete: bool = True,
        is_directory: bool = False,
    ) -> None:
        """Execute a delete operation on the given file or directory."""
        from files.marked_files import MarkedFiles

        MarkedFiles.delete_file_static(
            filepath,
            self._app.app_actions,
            toast=toast,
            manual_delete=manual_delete,
            is_directory=is_directory,
        )

    # Expose as the AppActions-compatible name
    handle_delete = _handle_delete

    def delete_current_base_dir(self, event=None) -> None:
        """
        Delete or trash the entire current base directory, with an option to
        move its contents to another directory first.
        """
        from files.marked_files import MarkedFiles
        from files.recent_directories import RecentDirectories
        from utils.app_info_cache import app_info_cache
        from ui.app_window.window_manager import WindowManager

        base_dir = self._app.get_base_dir()
        if not base_dir or base_dir == "." or base_dir.strip() == "" or not os.path.isdir(base_dir):
            self._app.notification_ctrl.alert(
                _("Invalid directory"), _("No valid base directory to delete"), kind="warning"
            )
            return

        open_window_dirs = [
            w.get_base_dir()
            for w in WindowManager.get_open_windows()
            if w.window_id != self._app.window_id and w.get_base_dir()
        ]

        try:
            replacement_dir = RecentDirectories.find_replacement_directory(base_dir, open_window_dirs)
        except ValueError as e:
            self._app.notification_ctrl.alert(_("Cannot Delete Directory"), str(e), kind="warning")
            return

        file_summary = self._fb.get_file_type_summary_for_directory(recursive=True)
        alert_message = (
            _("Are you sure you want to delete the directory and all contents?")
            + "\n\n" + str(base_dir) + "\n\n"
            + _("Contents to be deleted:") + "\n" + file_summary
        )

        # Three-option dialog: Delete / Move Files / Cancel (high-severity styled)
        clicked = self._app.notification_ctrl.alert(
            _("Confirm Delete Directory"),
            alert_message,
            kind="askokcancel",
            severity="high",
            buttons=[
                (_("Delete"), "destructive"),
                (_("Move Files"), "action"),
                (_("Cancel"), "reject"),
            ],
        )
        if not clicked:
            return

        move_files = clicked == _("Move Files")
        target_dir: str | None = None

        if move_files:
            from lib.fast_directory_picker_qt import get_existing_directory
            target_dir = get_existing_directory(
                self._app,
                _("Select target directory for directory contents"),
                initial_dir=str(os.path.dirname(base_dir)),
            )
            if not target_dir:
                return

        logger.info(f"Setting base directory to {replacement_dir} before deleting {base_dir}")
        self._app.sidebar_panel.update_base_dir_display(replacement_dir)
        self._app.set_base_dir(replacement_dir)

        # Close other windows using this base directory
        for win in WindowManager.get_open_windows()[:]:
            if win.window_id != self._app.window_id and win.base_dir == base_dir:
                try:
                    win.on_closing()
                except Exception as e:
                    logger.error(f"Error closing window for deleted directory: {e}")

        MarkedFiles.remove_marks_for_base_dir(base_dir, self._app.app_actions)

        try:
            RecentDirectories.remove_directory(base_dir)
            app_info_cache.clear_directory_cache(base_dir)
            app_info_cache.store()
            if move_files and target_dir:
                self._move_dir_contents_then_delete(base_dir, target_dir)
            else:
                self._handle_delete(base_dir, toast=True, manual_delete=True, is_directory=True)
        except Exception as e:
            self._app.notification_ctrl.handle_error(str(e), title=_("Delete Directory Error"))
            return

        if move_files and target_dir:
            self._app.notification_ctrl.toast(
                _("Directory {0} contents moved to {1} and directory deleted.").format(base_dir, target_dir),
                time_in_seconds=10,
            )
        else:
            self._app.notification_ctrl.toast(
                _("Directory {0} deleted.").format(base_dir), time_in_seconds=10
            )

    def _move_dir_contents_then_delete(self, base_dir: str, target_dir: str) -> None:
        """Move all contents of base_dir into target_dir, then delete base_dir."""
        from files.directory_ops import move_directory_contents_then_delete

        item_name = os.path.basename(base_dir)
        self._app.notification_ctrl.title_notify(
            _("Moving directory contents: {0}").format(item_name),
            action_type=ActionType.REMOVE_FILE,
        )
        move_directory_contents_then_delete(base_dir, target_dir)

    # ==================================================================
    # Hide operations
    # ==================================================================
    def hide_current_media(self, event=None, media_path: Optional[str] = None) -> None:
        """Hide the current media from the file list."""
        filepath = self._nav.get_active_media_filepath() if media_path is None else media_path
        if filepath is not None and filepath not in self._cm.hidden_media:
            self._cm.hidden_media.append(filepath)
        if media_path is None:
            self._app.notification_ctrl.toast(_("Hid current media.\nTo unhide, press Shift+B."))
        self._nav.show_next_media()

    def clear_hidden_media(self, event=None) -> None:
        """Clear the list of hidden media files."""
        self._cm.hidden_media.clear()
        self._app.notification_ctrl.toast(_("Cleared all hidden media."))

    # ==================================================================
    # Copy operations
    # ==================================================================
    def copy_media_path(self, filepath: Optional[str] = None) -> None:
        """Copy the file path to the clipboard."""
        if filepath is None:
            filepath = self._nav.get_active_media_filepath()
        if filepath is None:
            return
        if sys.platform == "win32":
            filepath = os.path.normpath(filepath)
            if config.escape_backslash_filepaths:
                filepath = filepath.replace("\\", "\\\\")
        clipboard = QApplication.clipboard()
        clipboard.setText(filepath)
        self._app.notification_ctrl.toast(_("Copied filepath to clipboard"))

    def copy_media_basename(self, filepath: Optional[str] = None) -> None:
        """Copy the file basename to the clipboard."""
        if filepath is None:
            filepath = self._nav.get_active_media_filepath()
        if filepath is None:
            return
        basename = os.path.basename(filepath)
        clipboard = QApplication.clipboard()
        clipboard.setText(basename)
        self._app.notification_ctrl.toast(_("Copied filename to clipboard"))

    def convert_directory_images_to_jpg(self, event=None) -> None:
        """
        Convert all non-JPG images in the current file-browser scope to JPG.
        """
        from image import directory_ops

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("convert files")))
            return

        base_dir = self._app.get_base_dir()
        if not base_dir or not os.path.isdir(base_dir):
            self._app.app_actions.warn(_("No valid base directory to convert"))
            return

        # Force-refresh at the AppWindow layer to align with existing refresh
        # guards/thread marshaling before collecting files for conversion.
        self._app.app_actions.refresh(file_check=False)

        # For conversion, process by creation-time using cached SortableFile metadata.
        files = self._fb.get_files_sorted_for_operation(
            sort_by=SortBy.CREATION_TIME,
            sort=Sort.ASC,
        )
        if len(files) == 0:
            self._app.notification_ctrl.toast(_("No files found to convert"))
            return

        survey = directory_ops.survey_jpg_conversion(files)

        if survey.has_nothing_to_do():
            self._app.notification_ctrl.toast(_("No image files found to convert"))
            return

        none_found = " " + _("(none to skip)") if survey.existing_target_count == 0 else ""
        choice = self._app.app_actions.alert(
            _("Convert Directory Images to JPG"),
            _(
                "Confirm convert images in current directory scope.\n\n{0}\n\n"
                "- Existing JPG/JPEG files in scope: {1}\n"
                "- Non-JPG files with existing JPG targets: {2}\n\n"
                "Overwrite existing JPG files (no-EXIF conversion output), or skip "
                "conversion for files with target conflicts{3}?"
            ).format(
                base_dir, survey.existing_jpg_count, survey.existing_target_count, none_found
            ),
            kind="askyesnocancel",
            yes_text=_("Overwrite existing JPG"),
            no_text=_("Skip conflicts"),
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return

        result = directory_ops.convert_files_to_jpg(
            survey, overwrite_existing=choice == QMessageBox.StandardButton.Yes
        )

        if result.converted > 0:
            self._app.refresh()

        if result.failed == 0 and result.skipped_existing == 0:
            self._app.notification_ctrl.toast(
                _("Converted {0} files to JPG").format(result.converted)
            )
        else:
            self._app.app_actions.warn(
                _(
                    "Converted {0} files to JPG, {1} failed, {2} skipped existing"
                ).format(result.converted, result.failed, result.skipped_existing)
            )

    def convert_directory_svg_to_png(self, event=None) -> None:
        """
        Convert all SVG files in the current file-browser scope to PNG.
        """
        from image import directory_ops

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("convert files")))
            return

        base_dir = self._app.get_base_dir()

        if not base_dir or not os.path.isdir(base_dir):
            self._app.app_actions.warn(_("No valid base directory to convert"))
            return

        # Force-refresh at the AppWindow layer
        self._app.app_actions.refresh(file_check=False)
        
        # Get files sorted by creation time
        files = self._fb.get_files_sorted_for_operation(
            sort_by=SortBy.CREATION_TIME,
            sort=Sort.ASC,
        )
        
        if len(files) == 0:
            self._app.notification_ctrl.toast(_("No files found to convert"))
            return
            
        survey = directory_ops.survey_svg_conversion(files)

        if survey.has_nothing_to_do():
            self._app.notification_ctrl.toast(_("No SVG files found to convert"))
            return

        count_target_pngs = survey.existing_target_count
        none_found = " " + _("(none to skip)") if count_target_pngs == 0 else ""
        choice = self._app.app_actions.alert(
            _("Convert Directory SVG to PNG"),
            _(
                "Convert SVGs in the current directory scope to PNG?\n\n"
                "{0}\n\n"
                "{1} SVG(s) already have a PNG at the matching output path.\n\n"
                "Convert all SVGs and overwrite any existing PNGs, or skip "
                "conversion for SVGs with target conflicts{2}?"
            ).format(base_dir, count_target_pngs, none_found),
            kind="askyesnocancel",
            yes_text=_("Overwrite existing PNG"),
            no_text=_("Skip conflicts"),
        )
        
        if choice == QMessageBox.StandardButton.Cancel:
            return
            
        result = directory_ops.convert_svgs_to_png(
            survey, overwrite_existing=choice == QMessageBox.StandardButton.Yes
        )

        if result.converted > 0:
            self._app.refresh()

        if result.failed == 0 and result.skipped_existing == 0:
            self._app.notification_ctrl.toast(
                _("Converted {0} SVG files to PNG").format(result.converted)
            )
        else:
            self._app.app_actions.warn(
                _(
                    "Converted {0} files, {1} failed, {2} skipped existing"
                ).format(result.converted, result.failed, result.skipped_existing)
            )

    def scale_directory_images(self, event=None) -> None:
        """Scale all images in the current directory scope to an equivalent pixel area."""
        from PySide6.QtWidgets import QInputDialog

        from image import directory_ops

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("scale images")))
            return

        base_dir = self._app.get_base_dir()
        if not base_dir or not os.path.isdir(base_dir):
            self._app.app_actions.warn(_("No valid base directory to scale images"))
            return

        target_side, ok = QInputDialog.getInt(
            self._app,
            _("Scale Directory Images"),
            _("Equivalent square side (pixels).\n"
              "Images are scaled so their total pixel count matches this value squared.\n"
              "Example: 320 → target area = 320×320 = 102 400 px"),
            320,  # value
            1,  # minValue
            65535,  # maxValue
        )
        if not ok:
            return

        self._app.app_actions.refresh(file_check=False)
        files = self._fb.get_files_sorted_for_operation(
            sort_by=SortBy.CREATION_TIME,
            sort=Sort.ASC,
        )
        if not files:
            self._app.notification_ctrl.toast(_("No files found"))
            return

        survey = directory_ops.survey_image_scaling(files, target_side)
        if survey.has_no_candidates():
            self._app.notification_ctrl.toast(_("No image files found to scale"))
            return

        if survey.nothing_to_scale():
            self._app.notification_ctrl.toast(
                _("{0} image(s) are already at or below {1}²={2} px — nothing to scale").format(
                    len(survey.candidates), target_side, survey.target_pixels,
                )
            )
            return

        choice = self._app.app_actions.alert(
            _("Scale Directory Images"),
            _(
                "Scale images in:\n{0}\n\n"
                "Target area: {1}×{1} = {2} px (aspect ratio preserved)\n\n"
                "Images in scope: {3}\n"
                "Already within limit: {4}\n"
                "Will be scaled down: {5}\n\n"
                "This modifies image files in place. Proceed?"
            ).format(base_dir, target_side, survey.target_pixels,
                     len(survey.candidates), survey.already_within, survey.to_scale),
            kind="askokcancel",
        )
        if choice != QMessageBox.StandardButton.Ok:
            return

        result = directory_ops.scale_images(survey)

        if result.scaled > 0:
            self._app.refresh()

        if result.failed == 0:
            self._app.notification_ctrl.toast(
                _("Scaled {0} image(s) to ~{1}² px, {2} already within limit").format(
                    result.scaled, target_side, result.skipped,
                )
            )
        else:
            self._app.app_actions.warn(
                _("Scaled {0} image(s) to ~{1}² px, {2} failed, {3} skipped").format(
                    result.scaled, target_side, result.failed, result.skipped,
                )
            )

    # ==================================================================
    # Replace / group operations
    # ==================================================================
    def replace_current_media_with_search_media(self) -> None:
        """Overwrite the current media file with the search media file."""
        if (
            self._app.mode != Mode.SEARCH
            or len(self._cm.files_matched) == 0
            or not os.path.exists(str(self._cm.search_media_path))
        ):
            return

        _filepath = self._cm.current_match()
        filepath = Utils.get_valid_file(self._app.get_base_dir(), _filepath)

        if filepath is None:
            self._app.notification_ctrl.handle_error(
                _("Invalid target filepath for replacement: ") + "None"
            )
            return

        os.rename(str(self._cm.search_media_path), filepath)
        self._app.notification_ctrl.toast(_("Moved search media to ") + filepath)

    def handle_remove_files_from_groups(self, files: list[str]) -> None:
        """Remove the given files from compare groups."""
        # Only move flows reach this method (deletes update groups directly in
        # delete_media), and moves can be undone — snapshot the group state
        # first so the compare-result change can be undone along with the move
        # (see AppWindow.restore_compare_state_for_undone_move).
        self._cm.capture_removal_undo_snapshot(files, self._app.mode)
        current_media = self._cm.current_match()
        if self._app.mode == Mode.GROUP_COMPLEMENT:
            # _get_file_group_map / _update_groups_for_removed_file assume the
            # removed file belongs to the group at current_group_index, which
            # still points at whatever real group was last being browsed
            # (deliberately left untouched -- see
            # CompareWrapper.enter_complement_mode). Going through that path
            # here would risk corrupting the live complement list or the real
            # group data, so route every file through remove_from_complement
            # instead, same as the GROUP_COMPLEMENT branch in delete_media.
            for filepath in files:
                if filepath == self._cm.search_media_path:
                    self._cm.search_media_path = None
                self._cm.remove_from_complement(filepath)
                self._cm._sync_result_after_deletion(filepath)
            return
        for filepath in files:
            if filepath == self._cm.search_media_path:
                self._cm.search_media_path = None
            show_next_media = self._app.direction if current_media == filepath else None
            file_group_map = self._cm._get_file_group_map(self._app.mode)
            try:
                group_indexes = file_group_map[filepath]
                self._cm._update_groups_for_removed_file(
                    self._app.mode,
                    group_indexes[0],
                    group_indexes[1],
                    set_group=False,
                    show_next_media=show_next_media,
                )
            except KeyError:
                pass
            self._cm._sync_result_after_deletion(filepath)

    # ==================================================================
    # External file operations
    # ==================================================================
    def open_media_location(self, event=None) -> None:
        """Open the file's directory in the system file manager."""
        filepath = self._nav.get_active_media_filepath()
        if filepath is not None:
            is_video = self._app.media_frame.pause_video_if_playing() if hasattr(self._app.media_frame, "pause_video_if_playing") else False
            self._app.notification_ctrl.toast(_("Opening media file: {0}").format(filepath))
            Utils.open_media_file(filepath, is_video=is_video)
        else:
            self._app.notification_ctrl.handle_error(
                _("Failed to open current media file, unable to get valid filepath")
            )

    def strip_audio_from_current_video(self, event=None) -> None:
        """
        Create a sibling copy of the current video with audio removed (ffmpeg stream copy).
        """
        from image.video_ops import VideoOps
        from utils.media_utils import is_video_file

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("strip audio")))
            return

        filepath = self._nav.get_active_media_filepath()
        if not filepath:
            return
        if not is_video_file(filepath):
            self._app.app_actions.warn(_("Not a video file"))
            return
        if not VideoOps.find_ffmpeg_executable():
            self._app.app_actions.warn(_("ffmpeg not found on PATH. Install ffmpeg to strip audio."))
            return

        planned_out = VideoOps.default_output_path_copy_without_audio(filepath)
        if not self._app.app_actions.alert(
            _("Save copy without audio"),
            _(
                "Create a new file with all audio streams removed (stream copy, no re-encode). "
                "The original file will not be changed.\n\n"
                "Output:\n{0}"
            ).format(planned_out),
            kind="askyesno",
        ):
            return

        if hasattr(self._app.media_frame, "pause_video_if_playing"):
            self._app.media_frame.pause_video_if_playing()

        try:
            out_path = VideoOps.copy_video_without_audio(filepath)
        except Exception as e:
            logger.warning("Strip audio failed: %s", e)
            self._app.app_actions.warn(str(e))
            return

        self._app.refresh()
        self._app.notification_ctrl.toast(_("Saved copy without audio: {0}").format(out_path))

    def copy_current_video_without_metadata(self, event=None) -> None:
        """
        Create a sibling copy of the current video with container metadata stripped (ffmpeg remux).
        """
        from image.video_ops import VideoOps
        from utils.media_utils import is_video_file

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("copy video")))
            return

        filepath = self._nav.get_active_media_filepath()
        if not filepath:
            return
        if not is_video_file(filepath):
            self._app.app_actions.warn(_("Not a video file"))
            return
        if not VideoOps.find_ffmpeg_executable():
            self._app.app_actions.warn(_("ffmpeg not found on PATH."))
            return

        planned_out = VideoOps.default_output_path_copy_without_metadata(filepath)
        if not self._app.app_actions.alert(
            _("Save copy without metadata"),
            _(
                "Create a new file with container tags and chapters removed (stream copy, no re-encode). "
                "The original file will not be changed.\n\n"
                "Output:\n{0}"
            ).format(planned_out),
            kind="askyesno",
        ):
            return

        if hasattr(self._app.media_frame, "pause_video_if_playing"):
            self._app.media_frame.pause_video_if_playing()

        try:
            out_path = VideoOps.copy_video_without_metadata(filepath)
        except Exception as e:
            logger.warning("Copy without metadata failed: %s", e)
            self._app.app_actions.warn(str(e))
            return

        self._app.refresh()
        self._app.notification_ctrl.toast(_("Saved copy without metadata: {0}").format(out_path))

    def cut_current_video_at_playback_position(self, event=None) -> None:
        """
        Trim the active video at the current VLC playback position, writing a new
        sibling file.  The user chooses which half to keep via a confirmation dialog.
        Runs ffmpeg on a background thread so the UI stays responsive.
        """
        from image.video_ops import VideoCutSide, VideoOps
        from utils.media_utils import is_video_file

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("cut video")))
            return

        filepath = self._nav.get_active_media_filepath()
        if not filepath:
            return
        if not is_video_file(filepath):
            self._app.app_actions.warn(_("Not a video file"))
            return
        if not VideoOps.find_ffmpeg_executable():
            self._app.app_actions.warn(_("ffmpeg not found on PATH. Install ffmpeg to cut video."))
            return

        pos_ms, dur_ms = self._app.media_frame.get_video_playback_ms()
        if dur_ms <= 0:
            self._app.app_actions.warn(_("Video duration is unknown. Seek to a position first."))
            return
        if pos_ms <= 0:
            self._app.app_actions.warn(_("Cannot cut at the very start. Seek to a later position."))
            return
        if pos_ms >= dur_ms:
            self._app.app_actions.warn(_("Cannot cut at or past the end of the video."))
            return

        def _fmt(ms: int) -> str:
            total_s, frac = divmod(ms, 1000)
            m, s = divmod(total_s, 60)
            return f"{m:02d}:{s:02d}.{frac:03d}"

        beginning_dur = _fmt(pos_ms)
        end_dur = _fmt(dur_ms - pos_ms)
        planned_before = VideoOps.default_output_path_cut(filepath, VideoCutSide.KEEP_BEGINNING, pos_ms)
        planned_after = VideoOps.default_output_path_cut(filepath, VideoCutSide.KEEP_END, pos_ms)

        choice = self._app.notification_ctrl.alert(
            _("Cut video at current position"),
            _(
                "Cut at {0} of {1}\n\n"
                "Keep beginning — writes ~{2} of footage (start → cut)\n"
                "  Output: {3}\n\n"
                "Keep end — writes ~{4} of footage (cut → end)\n"
                "  Output: {5}\n\n"
                "Note: stream copy snaps to the nearest keyframe.\n"
                "The original file is not modified."
            ).format(
                _fmt(pos_ms), _fmt(dur_ms),
                beginning_dur, planned_before,
                end_dur, planned_after,
            ),
            kind="askyesnocancel",
            yes_text=_("Keep beginning"),
            no_text=_("Keep end"),
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return
        side = VideoCutSide.KEEP_BEGINNING if choice == QMessageBox.StandardButton.Yes else VideoCutSide.KEEP_END

        if hasattr(self._app.media_frame, "pause_video_if_playing"):
            self._app.media_frame.pause_video_if_playing()

        _cutting = [True]

        def _do_cut():
            try:
                out_path = VideoOps.cut_video_at_ms(filepath, pos_ms, side, dur_ms)
                self._app.notification_ctrl.toast(_("Cut video saved: {0}").format(out_path))
                self._app.refresh()
            except Exception as e:
                logger.warning("Cut video failed: %s", e)
                self._app.notification_ctrl.toast(_("Cut video failed: {0}").format(e))
            finally:
                _cutting[0] = False

        start_thread(_do_cut, use_asyncio=False)

    def mute_current_video_audio_at_playback_position(self, event=None) -> None:
        """
        Silence the audio for a short window starting at the current VLC playback
        position, writing a new sibling file. The window's duration defaults to
        1 second but can be adjusted. Runs ffmpeg on a background thread so the
        UI stays responsive.
        """
        from PySide6.QtWidgets import QInputDialog
        from image.video_ops import VideoOps
        from utils.media_utils import is_video_file

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("mute video audio")))
            return

        filepath = self._nav.get_active_media_filepath()
        if not filepath:
            return
        if not is_video_file(filepath):
            self._app.app_actions.warn(_("Not a video file"))
            return
        if not VideoOps.find_ffmpeg_executable():
            self._app.app_actions.warn(_("ffmpeg not found on PATH. Install ffmpeg to mute audio."))
            return

        pos_ms, dur_ms = self._app.media_frame.get_video_playback_ms()
        if dur_ms <= 0:
            self._app.app_actions.warn(_("Video duration is unknown. Seek to a position first."))
            return
        if pos_ms >= dur_ms:
            self._app.app_actions.warn(_("Cannot mute at or past the end of the video."))
            return

        duration_ms, ok = QInputDialog.getInt(
            self._app,
            _("Mute Audio at Current Position"),
            _("Duration to mute, in milliseconds (starting at the current position):"),
            1000,  # value
            1,  # minValue
            60000,  # maxValue
        )
        if not ok:
            return

        start_ms = pos_ms
        end_ms = start_ms + duration_ms

        def _fmt(ms: int) -> str:
            total_s, frac = divmod(ms, 1000)
            m, s = divmod(total_s, 60)
            return f"{m:02d}:{s:02d}.{frac:03d}"

        planned_out = VideoOps.default_output_path_mute_audio(filepath, start_ms, end_ms)

        proceed = self._app.notification_ctrl.alert(
            _("Mute audio at current position"),
            _(
                "Silence audio from {0} to {1}\n\n"
                "Output: {2}\n\n"
                "Video is copied unchanged; only the audio track is re-encoded.\n"
                "The original file is not modified."
            ).format(_fmt(start_ms), _fmt(end_ms), planned_out),
            kind="askokcancel",
        )
        if not proceed:
            return

        if hasattr(self._app.media_frame, "pause_video_if_playing"):
            self._app.media_frame.pause_video_if_playing()

        def _do_mute():
            try:
                out_path = VideoOps.mute_audio_range_ms(filepath, start_ms, end_ms)
                self._app.notification_ctrl.toast(_("Muted audio saved: {0}").format(out_path))
                self._app.refresh()
            except Exception as e:
                logger.warning("Mute audio failed: %s", e)
                self._app.notification_ctrl.toast(_("Mute audio failed: {0}").format(e))

        start_thread(_do_mute, use_asyncio=False)

    def copy_directory_videos_without_metadata(self, event=None) -> None:
        """
        For each video in the current file-browser scope, write a sibling copy with
        container metadata stripped (same behavior as single-file \"Save copy without metadata\").
        """
        from image import directory_ops
        from image.video_ops import VideoOps

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("copy videos without metadata")))
            return

        if not getattr(config, "enable_videos", True):
            self._app.app_actions.warn(_("Video support is disabled in configuration"))
            return

        base_dir = self._app.get_base_dir()
        if not base_dir or not os.path.isdir(base_dir):
            self._app.app_actions.warn(_("No valid base directory"))
            return

        if not VideoOps.find_ffmpeg_executable():
            self._app.app_actions.warn(_("ffmpeg not found on PATH."))
            return

        self._app.app_actions.refresh(file_check=False)

        files = self._fb.get_files_sorted_for_operation(
            sort_by=SortBy.CREATION_TIME,
            sort=Sort.ASC,
        )
        survey = directory_ops.survey_video_metadata_strip(files)
        if survey.has_nothing_to_do():
            self._app.notification_ctrl.toast(_("No video files found in this directory scope"))
            return

        if not self._app.app_actions.alert(
            _("Save copies without metadata (videos in directory)"),
            _(
                "Create new files with container tags and chapters removed (stream copy) for each video.\n\n"
                "Directory:\n{0}\n\nVideos to process: {1}"
            ).format(base_dir, len(survey.videos)),
            kind="askyesno",
        ):
            return

        result = directory_ops.strip_video_metadata(survey)

        if result.written > 0:
            self._app.refresh()
        if result.failed == 0:
            self._app.notification_ctrl.toast(
                _("Wrote {0} video copy(ies) without metadata").format(result.written)
            )
        else:
            self._app.app_actions.warn(
                _("Finished: {0} succeeded, {1} failed").format(result.written, result.failed)
            )

    def open_image_in_gimp(self, event=None) -> None:
        """Open the current image in GIMP."""
        config.validate_and_find_gimp()
        if not config.gimp_exe_loc:
            self._app.notification_ctrl.handle_error(
                _("GIMP integration is not configured. Please set 'gimp_exe_loc' in config.json."),
                title=_("GIMP Integration Error"),
            )
            return

        if self._app.delete_lock:
            filepath = self._app.prev_media_path
        else:
            filepath = self._nav.get_active_media_filepath()

        if filepath is not None:
            from extensions.gimp.gimp_wrapper import open_image_in_gimp_wrapper
            from files.marked_files import MarkedFiles
            open_image_in_gimp_wrapper(
                filepath,
                config.gimp_exe_loc,
                self._is_slow_total_files_for_gimp,
                self._app.app_actions,
            )
            MarkedFiles.gimp_opened_in_last_action = True
        else:
            self._app.notification_ctrl.handle_error(
                _("Failed to open current file in GIMP, unable to get valid filepath")
            )

    def _is_slow_total_files_for_gimp(self) -> bool:
        """
        Keep GIMP wrapper behavior tuned for earlier temp-dir fallback.
        """
        return self._fb.is_slow_total_files(threshold=1000)

    def run_refacdir(self, event=None) -> None:
        """Run the RefacDir client on the current image."""
        from extensions.refacdir_client import RefacDirClient

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("run RefacDir")))
            return

        refacdir_client = RefacDirClient()
        refacdir_client.run(self._app.media_path)
        self._app.notification_ctrl.toast(_("Running refacdir"))

    @staticmethod
    def _randomize_filenames_script_path() -> str:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return os.path.join(repo_root, "scripts", "randomize_filenames.py")

    def run_randomize_filenames(self, event=None) -> None:
        """Spawn randomize_filenames.py on the base directory (dry run or execute)."""
        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("randomize filenames")))
            return

        base_dir = self._app.get_base_dir()
        if not base_dir or not os.path.isdir(base_dir):
            self._app.app_actions.warn(_("No valid base directory for randomize filenames"))
            return

        script_path = self._randomize_filenames_script_path()
        if not os.path.isfile(script_path):
            self._app.app_actions.warn(
                _("Randomize filenames script not found:\n{0}").format(script_path)
            )
            return

        log_path = os.path.join(base_dir, _RANDOMIZE_FILENAMES_LOG_BASENAME)
        short_dir = Utils.get_relative_dirpath(base_dir, levels=2) or base_dir

        btn_dry_run = _("Dry run")
        btn_execute = _("Execute")
        choice = self._app.notification_ctrl.alert(
            _("Randomize Filenames"),
            _(
                "Randomize media filenames under the entire base directory tree.\n\n"
                "{0}\n\n"
                "Log file:\n{1}\n\n"
                "Dry run — plan renames only; no files are changed.\n"
                "Execute — apply renames and update the mapping cache (irreversible)."
            ).format(base_dir, log_path),
            kind="askokcancel",
            severity="high",
            buttons=[
                (btn_dry_run, "action"),
                (btn_execute, "destructive"),
                (_("Cancel"), "reject"),
            ],
        )
        if not choice:
            return
        execute = choice == btn_execute

        if execute and not self._app.notification_ctrl.alert(
            _("Confirm Randomize Filenames — Execute"),
            _(
                "This will rename media files on disk under:\n\n{0}\n\n"
                "Output log:\n{1}\n\n"
                "This cannot be undone from Weidr. Proceed?"
            ).format(base_dir, log_path),
            kind="askokcancel",
            severity="high",
        ):
            return

        cmd = [
            sys.executable,
            script_path,
            base_dir,
            "--verbose",
            "--log-file",
            log_path,
        ]
        cmd.append("--execute" if execute else "--dry-run")

        def spawn_worker() -> None:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW
            try:
                subprocess.Popen(
                    cmd,
                    cwd=base_dir,
                    stdin=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            except OSError as e:
                logger.error("Failed to start randomize_filenames: %s", e)
                self._app.notification_ctrl.handle_error(
                    _("Failed to start randomize filenames: {0}").format(e)
                )
                return

        start_thread(spawn_worker, use_asyncio=False)

        if execute:
            self._app.notification_ctrl.toast(
                _("Randomizing filenames under {0} — see {1}").format(short_dir, log_path)
            )
        else:
            self._app.notification_ctrl.toast(
                _("Randomize filenames dry run started for {0} — see {1}").format(
                    short_dir, log_path
                )
            )

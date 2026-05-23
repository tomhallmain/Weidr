import os
import sys
from typing import Tuple, Optional, Callable

from files.file_action import FileAction
from image.frame_cache import FrameCache
from image.image_ops import ImageOps
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.constants import ActionType
from utils.logging_setup import get_logger
from utils.translations import I18N, compare_running_warn
from utils.utils import Utils

_ = I18N._
logger = get_logger("marked_files")


class MarkedFiles():
    file_marks = []
    mark_cursor = -1
    mark_target_dirs = []
    previous_marks = []
    last_moved_image = None
    last_set_target_dir = None
    file_browser = None # a file browser for test_is_in_directory

    # For file operations that take a while because they involve many files, pressing Ctrl+Z while they are
    # running should not attempt to undo the action before the currently running one.
    is_performing_action = False
    is_cancelled_action = False

    # Track if GIMP was opened in the last action (for delete source file check)
    gimp_opened_in_last_action = False

    # Unable to undo a delete action.
    delete_lock = False
    # Per-target preference: True means this target requires GUI mode
    # (prevents accidental non-GUI quick targeting).
    mark_target_require_gui = {}

    @staticmethod
    def load_target_dirs():
        MarkedFiles.set_target_dirs(app_info_cache.get_meta("marked_file_target_dirs", default_val=[]))
        MarkedFiles.set_target_dir_require_gui(
            app_info_cache.get_meta("marked_file_target_require_gui", default_val={})
        )

    @staticmethod
    def set_target_dirs(target_dirs):
        MarkedFiles.mark_target_dirs = target_dirs
        for d in MarkedFiles.mark_target_dirs[:]:
            if not os.path.isdir(d):
                # The external drive this reference is pointing to may not be mounted, might still be valid
                if sys.platform == "win32" and not d.startswith("C:\\"):
                    base_dir = os.path.split("\\")[0] + "\\"
                    if not os.path.isdir(base_dir):
                        continue
                MarkedFiles.mark_target_dirs.remove(d)
                logger.warning(f"Removed stale target directory reference: {d}")
        MarkedFiles._cleanup_target_dir_require_gui()

    @staticmethod
    def store_target_dirs():
        app_info_cache.set_meta("marked_file_target_dirs", MarkedFiles.mark_target_dirs)
        app_info_cache.set_meta(
            "marked_file_target_require_gui", MarkedFiles.mark_target_require_gui
        )

    @staticmethod
    def _normalize_target_dir(dirpath: Optional[str]) -> str:
        if not dirpath:
            return ""
        return os.path.normcase(os.path.normpath(dirpath))

    @staticmethod
    def set_target_dir_require_gui(target_require_gui) -> None:
        """
        Accept a raw cache payload and keep only valid/known target directories.
        """
        if not isinstance(target_require_gui, dict):
            target_require_gui = {}
        normalized_payload = {}
        for dirpath, is_required in target_require_gui.items():
            if not isinstance(dirpath, str):
                continue
            normalized_payload[MarkedFiles._normalize_target_dir(dirpath)] = bool(is_required)
        MarkedFiles.mark_target_require_gui = normalized_payload
        MarkedFiles._cleanup_target_dir_require_gui()

    @staticmethod
    def _cleanup_target_dir_require_gui() -> None:
        valid_dirs = {
            MarkedFiles._normalize_target_dir(d)
            for d in MarkedFiles.mark_target_dirs
        }
        for dirpath in list(MarkedFiles.mark_target_require_gui.keys()):
            if dirpath not in valid_dirs:
                del MarkedFiles.mark_target_require_gui[dirpath]

    @staticmethod
    def is_target_dir_require_gui(dirpath: Optional[str]) -> bool:
        normalized = MarkedFiles._normalize_target_dir(dirpath)
        if normalized == "":
            return False
        return bool(MarkedFiles.mark_target_require_gui.get(normalized, False))

    @staticmethod
    def set_single_target_require_gui(dirpath: Optional[str], require_gui: bool) -> None:
        normalized = MarkedFiles._normalize_target_dir(dirpath)
        if normalized == "":
            return
        if require_gui:
            MarkedFiles.mark_target_require_gui[normalized] = True
        elif normalized in MarkedFiles.mark_target_require_gui:
            del MarkedFiles.mark_target_require_gui[normalized]

    @staticmethod
    def add_mark_if_not_present(filepath):
        """
        Add a file to the marks list if it's not already present.
        Returns True if the file was added, False if it was already present.
        """
        if filepath not in MarkedFiles.file_marks:
            MarkedFiles.file_marks.append(filepath)
            return True
        return False

    @staticmethod
    def set_delete_lock(delete_lock=True):
        MarkedFiles.delete_lock = delete_lock

    @staticmethod
    def clear_file_marks(toast_callback):
        MarkedFiles.file_marks = []
        toast_callback(_("Marks cleared."))

    @staticmethod
    def _paths_match(path_a: Optional[str], path_b: Optional[str]) -> bool:
        if not path_a or not path_b:
            return False
        try:
            if os.path.exists(path_a) and os.path.exists(path_b):
                return os.path.samefile(path_a, path_b)
        except Exception:
            pass
        norm_a = os.path.normcase(os.path.normpath(path_a))
        norm_b = os.path.normcase(os.path.normpath(path_b))
        return norm_a == norm_b

    @staticmethod
    def set_current_marks_from_previous(toast_callback):
        for f in MarkedFiles.previous_marks:
            if f not in MarkedFiles.file_marks and os.path.exists(f):
                MarkedFiles.file_marks.append(f)
        toast_callback(_("Set current marks from previous.") + "\n" + _("Total set: {0}").format(len(MarkedFiles.file_marks)))

    @staticmethod
    def run_previous_action(app_actions, current_media=None, ui_class=None, progress_callback=None):
        previous_action = FileAction.get_history_action(start_index=0)
        if previous_action is None:
            return False, False
        if ui_class is None:
            raise Exception("ui_class is required to get the target directory for undo move marked files.")
        return MarkedFiles.move_marks_to_dir_static(app_actions,
                                             target_dir=previous_action.target,
                                             move_func=previous_action.action,
                                             single_image=(len(MarkedFiles.file_marks)==1),
                                             current_media=current_media,
                                             get_target_dir_callback=ui_class.get_target_directory,
                                             progress_callback=progress_callback,
                                             )

    @staticmethod
    def run_penultimate_action(app_actions, current_media=None, ui_class=None, progress_callback=None):
        penultimate_action = FileAction.get_history_action(start_index=1)
        if penultimate_action is None:
            return False, False
        if ui_class is None:
            raise Exception("ui_class is required to get the target directory for undo move marked files.")
        return MarkedFiles.move_marks_to_dir_static(app_actions,
                                             target_dir=penultimate_action.target,
                                             move_func=penultimate_action.action,
                                             single_image=(len(MarkedFiles.file_marks)==1),
                                             current_media=current_media,
                                             get_target_dir_callback=ui_class.get_target_directory,
                                             progress_callback=progress_callback,
                                             )

    @staticmethod
    def run_antepenultimate_action(app_actions, current_media=None, ui_class=None, progress_callback=None):
        antepenultimate_action = FileAction.get_history_action(start_index=2)
        if antepenultimate_action is None:
            return False, False
        if ui_class is None:
            raise Exception("ui_class is required to get the target directory for undo move marked files.")
        return MarkedFiles.move_marks_to_dir_static(app_actions,
                                             target_dir=antepenultimate_action.target,
                                             move_func=antepenultimate_action.action,
                                             single_image=(len(MarkedFiles.file_marks)==1),
                                             current_media=current_media,
                                             get_target_dir_callback=ui_class.get_target_directory,
                                             progress_callback=progress_callback,
                                             )

    @staticmethod
    def run_permanent_action(app_actions, current_media=None, ui_class=None, progress_callback=None):
        if not FileAction.permanent_action:
            app_actions.toast(_("NO_MARK_TARGET_SET"))
            return False, False
        if ui_class is None:
            raise Exception("ui_class is required to get the target directory for undo move marked files.")
        return MarkedFiles.move_marks_to_dir_static(app_actions,
                                             target_dir=FileAction.permanent_action.target,
                                             move_func=FileAction.permanent_action.action,
                                             single_image=(len(MarkedFiles.file_marks)==1),
                                             current_media=current_media,
                                             get_target_dir_callback=ui_class.get_target_directory,
                                             progress_callback=progress_callback,
                                             )

    @staticmethod
    def run_hotkey_action(app_actions, current_media=None, number=-1, shift_key_pressed=False, ui_class=None, progress_callback=None):
        assert number in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if number not in FileAction.hotkey_actions:
            app_actions.toast(_("NO_HOTKEY_ACTION_SET").format(number, number))
            return False, False
        if ui_class is None:
            raise Exception("ui_class is required to get the target directory for undo move marked files.")
        file_action = FileAction.hotkey_actions[number]
        return MarkedFiles.move_marks_to_dir_static(app_actions,
                                             target_dir=file_action.target,
                                             move_func=file_action.get_action(do_flip=shift_key_pressed),
                                             single_image=(len(MarkedFiles.file_marks)==1),
                                             current_media=current_media,
                                             get_target_dir_callback=ui_class.get_target_directory,
                                             progress_callback=progress_callback,
                                             )


    @staticmethod
    def move_marks_to_dir_static(
        app_actions,
        target_dir=None,
        move_func=Utils.move_file,
        files=None,
        single_image=False,
        current_media=None,
        get_base_dir_callback=None,
        get_target_dir_callback=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[bool, bool]:
        """
        Move or copy the marked files to the target directory.
        """
        is_moving = move_func == Utils.move_file
        if is_moving and app_actions is not None and app_actions.is_compare_running():
            app_actions.warn(compare_running_warn(_("move files")))
            return False, False

        MarkedFiles.is_performing_action = True
        some_files_already_present = False
        action_part1 = _("Moving") if is_moving else _("Copying")
        MarkedFiles.previous_marks.clear()
        files_to_move = MarkedFiles.file_marks if files is None else files
        action = FileAction(move_func, target_dir, MarkedFiles.file_marks)
        if len(files_to_move) > 1:
            logger.warning(f"{action_part1} {len(files_to_move)} files to directory: {target_dir}")
        exceptions = {}
        invalid_files = []
        set_last_moved_file = False
        files_done = 0
        total_files = len(files_to_move)
        for marked_file in files_to_move:
            if MarkedFiles.is_cancelled_action:
                break
            # Resolve source path for SVG: move/copy either the SVG or the generated PNG per config
            source_path = marked_file
            moved_svg_as_png = False
            if config.enable_svgs and marked_file.lower().endswith(".svg"):
                cached_png = FrameCache.get_cached_path(marked_file)
                if cached_png and os.path.isfile(cached_png):
                    if config.marked_file_svg_move_type == "png":
                        if is_moving and MarkedFiles._paths_match(current_media, marked_file) and app_actions:
                            app_actions.release_media_canvas()
                        source_path = cached_png
                        moved_svg_as_png = True
                    elif is_moving:
                        # Move SVG: release media and remove temp PNG so we don't hold handles
                        if MarkedFiles._paths_match(current_media, marked_file) and app_actions:
                            app_actions.release_media_canvas()
                        FrameCache.remove_from_cache(marked_file, delete_temp_file=True)
            new_filename = os.path.join(target_dir, os.path.basename(source_path))
            if not set_last_moved_file:
                MarkedFiles.last_moved_image = new_filename
                set_last_moved_file = True
            success, result = MarkedFiles._process_single_file_operation(
                marked_file, target_dir, move_func, new_filename, current_media, app_actions,
                overwrite_existing=config.move_marks_overwrite_existing_file,
                source_path=source_path
            )

            if success:
                action.add_file(result)
                MarkedFiles.previous_marks.append(marked_file)
                if moved_svg_as_png and is_moving:
                    FrameCache.remove_from_cache(marked_file, delete_temp_file=False)
                    if app_actions:
                        try:
                            app_actions.delete(marked_file, toast=False, manual_delete=False)
                        except Exception as e:
                            logger.warning(f"Failed to remove SVG after moving PNG: {marked_file} - {e}")
            else:
                exceptions[marked_file] = (result, new_filename)  # result is error message
                if not os.path.exists(marked_file):
                    invalid_files.append(marked_file)
            files_done += 1
            if progress_callback is not None:
                progress_callback(files_done, total_files)
        if MarkedFiles.is_cancelled_action:
            MarkedFiles.is_cancelled_action = False
            MarkedFiles.is_performing_action = False
            logger.warning(f"Cancelled {action_part1} to {target_dir}")
            if len(MarkedFiles.previous_marks) > 0:
                MarkedFiles.undo_move_marks(app_actions.get_base_dir(), app_actions, get_base_dir_callback, get_target_dir_callback)
            return False, False
        if len(exceptions) < len(files_to_move):
            FileAction.update_history(action)
            action_type = ActionType.MOVE_FILE if is_moving else ActionType.COPY_FILE
            target_dir_name = Utils.get_relative_dirpath(target_dir, levels=2)
            if is_moving:
                message = _("Moved {0} files to {1}").format(len(files_to_move) - len(exceptions), target_dir_name)
            else:
                message = _("Copied {0} files to {1}").format(len(files_to_move) - len(exceptions), target_dir_name)
            logger.warning(message.replace("\n", " "))
            app_actions.title_notify(message, base_message=target_dir_name, action_type=action_type)
            MarkedFiles.delete_lock = False
        MarkedFiles.file_marks.clear()
        exceptions_present = len(exceptions) > 0
        if exceptions_present:
            action_part3 = "move" if is_moving else "copy"
            logger.error(f"Failed to {action_part3} some files:")
            names_are_short = False
            matching_files = False
            content_matching_files = False
            for marked_file, exc_tuple in exceptions.items():
                error_msg = exc_tuple[0]
                target_filepath = exc_tuple[1]
                logger.error(error_msg)
                if marked_file not in invalid_files:
                    if not config.clear_marks_with_errors_after_move and not single_image:
                        # Just in case some of them failed to move for whatever reason.
                        MarkedFiles.file_marks.append(marked_file)
                    if error_msg.startswith("File already exists"):
                        if Utils.calculate_hash(marked_file) == Utils.calculate_hash(target_filepath):
                            matching_files = True
                            logger.info(f"File hashes match: {marked_file} <> {target_filepath}")
                            if is_moving and marked_file != target_filepath:
                                # Check if we should delete the source file (or warn instead if mistake detected)
                                if MarkedFiles._check_delete_source_file(marked_file, target_dir, target_filepath, app_actions):
                                    # The other effect of this operation would have been to remove the
                                    # file from source, so try to do that
                                    MarkedFiles._auto_delete_source_file(marked_file, current_media, app_actions)
                        elif ImageOps.compare_image_content_without_exif(marked_file, target_filepath):
                            # Hash comparison failed, but check if image content is identical
                            # (different EXIF data but same visual content)
                            logger.info(f"File hashes differ but image content matches: {marked_file} <> {target_filepath}")
                            logger.info("Replacing target file with source file (source has more EXIF data)")
                            try:
                                # Replace target with source file (source has more information)
                                success, result = MarkedFiles._process_single_file_operation(
                                    marked_file, os.path.dirname(target_filepath), move_func, target_filepath, current_media, app_actions,
                                    overwrite_existing=True
                                )
                                if success:
                                    content_matching_files = True
                                    logger.info("Replaced target file with source: " + marked_file)
                                    # Remove from exceptions since it was successfully handled
                                    del exceptions[marked_file]
                                    # Add to successful operations
                                    action.add_file(target_filepath)
                                    MarkedFiles.previous_marks.append(marked_file)
                                else:
                                    error_text = f"Failed to replace target file with source: {marked_file} - {result}"
                                    logger.warning(error_text)
                                    app_actions.title_notify(error_text)
                            except Exception as e:
                                error_text = f"Failed to replace target file with source: {marked_file} - {e}"
                                logger.warning(error_text)
                                app_actions.title_notify(error_text)
                        elif len(os.path.basename(marked_file)) < 13 and not names_are_short:
                            names_are_short = True
                        if not some_files_already_present:
                            some_files_already_present = True
                            # Copy first file path to clipboard if no matching files
                            if not matching_files:
                                try:
                                    app_actions.copy_media_path(target_filepath)
                                    logger.info(f"Copied first target file path to clipboard: {target_filepath}")
                                except Exception as e:
                                    logger.warning(f"Failed to copy file path to clipboard: {e}")
            if some_files_already_present:
                if config.clear_marks_with_errors_after_move and not single_image:
                    logger.info("Cleared invalid marks by config option")
                warning = _("Existing filenames match!")
                if matching_files:
                    warning += "\n" + _("WARNING: Exact file match.")
                if content_matching_files:
                    warning += "\n" + _("INFO: Target files with different EXIF data replaced.")
                if names_are_short:
                    warning += "\n" + _("WARNING: Short filenames.")
                app_actions.warn(warning)
        MarkedFiles.is_performing_action = False
        if len(MarkedFiles.previous_marks) > 0:
            MarkedFiles.last_set_target_dir = target_dir
            if is_moving:
                app_actions.refresh(removed_files=list(MarkedFiles.previous_marks))
            else:
                app_actions.refresh()
            if not exceptions_present:
                app_actions.refocus()
        return some_files_already_present, exceptions_present

    @staticmethod
    def undo_move_marks(base_dir, app_actions, get_base_dir_callback=None, get_target_dir_callback=None):
        """
        Undo the previous move/copy operation.
        """
        if MarkedFiles.is_performing_action:
            MarkedFiles.is_cancelled_action = True
            return
        if MarkedFiles.delete_lock:
            return
        is_moving_back = FileAction.action_history[0].action == Utils.move_file
        action_part1 = _("Moving back") if is_moving_back else _("Removing")
        action_part2 = _("Moved back") if is_moving_back else _("Removed")
        if get_target_dir_callback is None:
            raise Exception("get_target_dir_callback is required to get the target directory for undo move marked files.")
        target_dir, target_was_valid = get_target_dir_callback(MarkedFiles.last_set_target_dir, None, app_actions)
        if not target_was_valid:
            raise Exception(f"{action_part1} previously marked files failed, somehow previous target directory invalid:  {target_dir}")
        if base_dir is None:
            if get_base_dir_callback is None:
                raise Exception("get_base_dir_callback is required to get the base directory for undo move marked files.")
            base_dir = get_base_dir_callback()
        if base_dir is None or base_dir == "" or not os.path.isdir(base_dir):
            raise Exception("Failed to get valid base directory for undo move marked files.")
        logger.warning(f"Undoing action: {action_part1} {len(MarkedFiles.previous_marks)} files from directory:\n{MarkedFiles.last_set_target_dir}")
        exceptions = {}
        invalid_files = []
        action = FileAction.action_history[0]
        for i, marked_file in enumerate(MarkedFiles.previous_marks):
            # previous_marks holds source paths (e.g. foo.svg); action.new_files holds the paths we
            # actually created in the target. If we moved the generated PNG instead of the SVG,
            # the file in the target is foo.png, not foo.svg, so we must use action.new_files[i]
            # to know which file to move back. Both lists are appended in lockstep on success,
            # so they are normally the same length; the fallback is for legacy or edge cases.
            if i < len(action.new_files):
                expected_new_filepath = action.new_files[i]
            else:
                expected_new_filepath = os.path.join(target_dir, os.path.basename(marked_file))
            try:
                if is_moving_back:
                    # Move the file back to its original place.
                    Utils.move_file(expected_new_filepath, base_dir, overwrite_existing=config.move_marks_overwrite_existing_file)
                else:
                    # Remove the file.
                    os.remove(expected_new_filepath)
                logger.info(f"{action_part2} file from {target_dir}: {os.path.basename(expected_new_filepath)}")
            except Exception as e:
                exceptions[marked_file] = str(e)
                if is_moving_back:
                    if not os.path.exists(marked_file):
                        invalid_files.append(expected_new_filepath)
                elif os.path.exists(expected_new_filepath):
                    invalid_files.append(expected_new_filepath)
        if len(exceptions) < len(MarkedFiles.previous_marks):
            if is_moving_back:
                message = _("Moved back {0} files from {1}").format(len(MarkedFiles.previous_marks) - len(exceptions), target_dir)
            else:
                message = _("Removed {0} files from {1}").format(len(MarkedFiles.previous_marks) - len(exceptions), target_dir)
            app_actions.toast(message)
        MarkedFiles.previous_marks.clear()
        if len(exceptions) > 0:
            for marked_file in exceptions.keys():
                if marked_file not in invalid_files:
                    MarkedFiles.previous_marks.append(marked_file)  # Just in case some of them failed to move for whatever reason.
            action_part3 = "move" if is_moving_back else "copy"
            raise Exception(f"Failed to {action_part3} some files: {exceptions}")
        app_actions.refresh()

    @staticmethod
    def test_in_directory_static(app_actions, target_dir=None, single_image=False) -> bool:
        """
        Check if the marked files are in the target directory.
        """
        MarkedFiles.is_performing_action = True
        if len(MarkedFiles.file_marks) > 1:
            logger.info(f"Checking if {len(MarkedFiles.file_marks)} files are in directory: {target_dir}")
        found_files = []
        for marked_file in MarkedFiles.file_marks:
            new_filename = os.path.join(target_dir, os.path.basename(marked_file))
            if os.path.isfile(new_filename):
                logger.warning(f"{marked_file} is already present in {target_dir}")
                found_files.append((marked_file, new_filename))
#        MarkedFiles.file_marks.clear() MAYBE use if not config.clear_marks_with_errors_after_move and not single_image
        names_are_short = False
        matching_files = 0
        content_matching_files = 0
        for marked_file, new_filename in found_files:
            if Utils.calculate_hash(marked_file) == Utils.calculate_hash(new_filename):
                matching_files += 1
                logger.info(f"File hashes match: {marked_file} <> {new_filename}")
            elif ImageOps.compare_image_content_without_exif(marked_file, new_filename):
                # Hash comparison failed, but check if image content is identical
                # (different EXIF data but same visual content)
                content_matching_files += 1
                logger.info(f"File hashes differ but image content matches: {marked_file} <> {new_filename}")
            elif len(os.path.basename(marked_file)) < 13 and not names_are_short:
                names_are_short = True
        if len(found_files) > 0:
            # if config.clear_marks_with_errors_after_move and not single_image:
            #     logger.info("Cleared invalid marks by config option")
            warning = _("Existing filenames found!")
            if matching_files == len(MarkedFiles.file_marks):
                warning += "\n" + _("WARNING: All file hashes match.")
            elif matching_files > 0:
                warning += "\n" + _("WARNING: %s of %s file hashes match.").format(matching_files, len(MarkedFiles.file_marks))
            if content_matching_files > 0:
                warning += "\n" + _("INFO: %s files have identical content but different EXIF data.").format(content_matching_files)
            if (matching_files + content_matching_files) == len(MarkedFiles.file_marks):
                warning += "\n" + _("WARNING: All files are either identical or have matching content.")
            if names_are_short:
                warning += "\n" + _("WARNING: Short filenames.")
            app_actions.warn(warning)
#            MarkedFiles.last_set_target_dir = target_dir
        else:
            app_actions.toast(_("No existing filenames found."))
        app_actions.refocus()
        return len(found_files) > 0

    @staticmethod
    def handle_file_removal(filepath):
        if filepath in MarkedFiles.file_marks:
            filepath_index = MarkedFiles.file_marks.index(filepath)
            if filepath_index < MarkedFiles.mark_cursor:
                MarkedFiles.mark_cursor -= 1
            elif filepath_index == len(MarkedFiles.file_marks) - 1:
                MarkedFiles.mark_cursor = 0
            MarkedFiles.file_marks.remove(filepath)

    @staticmethod
    def remove_marks_for_base_dir(base_dir, app_actions):
        if len(MarkedFiles.file_marks) > 0 and base_dir and base_dir != "":
            removed_count = 0
            i = 0
            while i < len(MarkedFiles.file_marks):
                marked_file = MarkedFiles.file_marks[i]
                # NOTE we don't necessarily want to remove files that are in subdirectories of the base dir here
                if os.path.dirname(marked_file) == base_dir:
                    MarkedFiles.file_marks.remove(marked_file)
                    removed_count += 1
                    if MarkedFiles.mark_cursor >= len(MarkedFiles.file_marks):
                        MarkedFiles.mark_cursor = 0
                    elif MarkedFiles.mark_cursor > i:
                        MarkedFiles.mark_cursor -= 1
                else:
                    i += 1
            if len(MarkedFiles.file_marks) == 0:
                app_actions.toast(_("Marks cleared."))
            elif removed_count > 0:
                app_actions.toast(_("Removed {0} marks").format(removed_count))

    @staticmethod
    def _check_delete_source_file(marked_file: str, target_dir: str, target_filepath: str, app_actions) -> bool:
        """
        Check if we should delete the source file after a move operation.
        If the previous action was a COPY to the same directory with the same file,
        this indicates the user likely made a mistake, so we ask for confirmation.
        
        Returns:
            True if we should delete the source file, False if we should not delete.
        """
        # If GIMP was opened in the last action, skip the check (no copy operation occurred)
        if MarkedFiles.gimp_opened_in_last_action:
            MarkedFiles.gimp_opened_in_last_action = False
            return True
        
        should_check = False
        
        if len(FileAction.action_history) == 0:
            MarkedFiles.gimp_opened_in_last_action = False
            return True
        
        previous_action = FileAction.action_history[0]
        if previous_action.target == target_dir:
            # Check if the same file was involved in the previous action
            if (marked_file in previous_action.original_marks or 
                target_filepath in previous_action.new_files):
                should_check = True
        
        if should_check:
            # User likely made a mistake - ask if they want to continue with deletion
            warning_message = _(
                "WARNING: You just copied this file to this directory, and now you're trying to move it here.\n\n"
                "This would delete the original file. If this was a mistake, please cancel this operation.\n\n"
                "File: {0}\n"
                "Target: {1}\n\n"
                "Do you want to continue and delete the source file?"
            ).format(os.path.basename(marked_file), target_dir)
            if app_actions.alert(_("Potential Mistake Detected"), warning_message, kind="askokcancel"):
                # User chose to continue with deletion
                logger.warning(f"User confirmed deletion after copy-then-move detection for {marked_file}")
                MarkedFiles.gimp_opened_in_last_action = False
                return True
            else:
                # User cancelled - don't delete
                logger.warning(f"User cancelled deletion after copy-then-move detection for {marked_file}")
                MarkedFiles.gimp_opened_in_last_action = False
                return False
        
        MarkedFiles.gimp_opened_in_last_action = False
        return True

    @staticmethod
    def _auto_delete_source_file(
        marked_file: str,
        current_media: Optional[str] = None,
        app_actions=None,
    ) -> None:
        """
        Auto-delete the source file after a move operation when the target file already exists.
        This simulates what would have happened if the move had succeeded.
        """
        try:
            if MarkedFiles._paths_match(current_media, marked_file):
                app_actions.release_media_canvas()
            app_actions.delete(marked_file)
            if marked_file in MarkedFiles.file_marks:
                MarkedFiles.file_marks.remove(marked_file)
            app_actions.warn(_("Removed marked file from source: {0}").format(marked_file))
        except Exception as e:
            error_text = f"Failed to remove marked file from source: {marked_file} - {e}"
            logger.warning(error_text)
            app_actions.title_notify(error_text)

    @staticmethod
    def _process_single_file_operation(
        marked_file: str,
        target_dir: str,
        move_func: Callable,
        new_filename: str,
        current_media: Optional[str] = None,
        app_actions=None,
        overwrite_existing: bool = False,
        source_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Process a single file operation using a thread-safe lock.
        When source_path is set (e.g. for SVG->PNG move), that path is moved/copied instead of marked_file.
        new_filename is ignored; the destination path is computed from source_path.

        Returns:
            Tuple[bool, str]: (success, new_filename)
        """
        actual_source = source_path if source_path is not None else marked_file
        new_filename = os.path.join(target_dir, os.path.basename(actual_source))
        is_moving = move_func == Utils.move_file

        try:
            # Release media canvas BEFORE acquiring the lock.
            # release_media_canvas() calls QApplication.processEvents(), which
            # drains Qt's event queue synchronously on the main thread.  If
            # processEvents() fires the periodic file-check timer while the
            # lock is already held, file_browser._gather_files() tries to
            # re-acquire the same non-re-entrant threading.Lock and the main
            # thread deadlocks with itself.  Moving the canvas release out of
            # the locked section avoids that re-entrancy entirely.
            if is_moving and current_media == marked_file:
                if app_actions:
                    app_actions.release_media_canvas()

            # Use lock to ensure thread-safe file operations
            with Utils.file_operation_lock:
                # Perform the actual file operation on the resolved source
                move_func(
                    actual_source,
                    target_dir,
                    overwrite_existing=overwrite_existing,
                )

            action_part2 = _("Moved") if is_moving else _("Copied")
            logger.info(f"{action_part2} file to {new_filename}")
            return True, new_filename

        except Exception as e:
            return False, str(e)
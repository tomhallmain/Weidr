"""
WindowLauncher -- opens secondary windows and dialogs.

A thin class where each method creates the appropriate dialog/window.
Extracted from: all open_*_window methods, get_media_details,
get_help_and_config.

All window imports now point to the PySide6 (_qt) versions.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from ui.auth.password_utils import require_password, check_session_expired
from utils.constants import ClassifierActionType, Mode, ProtectedActions
from utils.logging_setup import get_logger, set_logger_level
from utils.translations import _, compare_running_warn

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow
logger = get_logger("window_launcher")


class WindowLauncher:
    """
    Opens every secondary window / dialog. Keeps the "which windows exist"
    knowledge in one place and makes window implementations easy to swap.
    """

    def __init__(self, app_window: AppWindow):
        self._app = app_window
        self._go_to_file_window = None
        self._directory_notes_window = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _handle_error(self, error: Exception, title: str = "Window Error") -> None:
        self._app.notification_ctrl.handle_error(str(error), title=title)

    # ------------------------------------------------------------------
    # Navigation windows
    # ------------------------------------------------------------------
    def open_go_to_file_window(self, event=None) -> None:
        """Open the go-to-file search window."""
        try:
            if self._go_to_file_window is not None:
                try:
                    if self._go_to_file_window.isVisible():
                        self._go_to_file_window.raise_()
                        self._go_to_file_window.activateWindow()
                        return
                except (RuntimeError, AttributeError):
                    self._go_to_file_window = None

            from ui.files.go_to_file_qt import GoToFile
            self._go_to_file_window = GoToFile(self._app, self._app.app_actions)
            self._go_to_file_window.show()
        except Exception as e:
            self._handle_error(e, "Go To File Window Error")

    def open_go_to_file_with_current_media(self, event=None) -> None:
        """Open go-to-file pre-populated with the current media name."""
        try:
            if self._go_to_file_window is not None:
                try:
                    if self._go_to_file_window.isVisible():
                        self._go_to_file_window.update_with_current_media()
                        return
                except (RuntimeError, AttributeError):
                    self._go_to_file_window = None

            from ui.files.go_to_file_qt import GoToFile
            self._go_to_file_window = GoToFile(self._app, self._app.app_actions)
            self._go_to_file_window.show()
            self._go_to_file_window.update_with_current_media(focus=True)
        except Exception as e:
            self._handle_error(e, "Go To File Window Error")

    # ------------------------------------------------------------------
    # Directory windows
    # ------------------------------------------------------------------
    def open_recent_directory_window(
        self,
        event=None,
        open_gui: bool = True,
        run_compare_media: Optional[str] = None,
        extra_callback_args: Optional[list[Any]] = None,
    ) -> None:
        """Open the recent directories window."""
        try:
            from ui.files.recent_directory_window_qt import RecentDirectoryWindow

            window = RecentDirectoryWindow(
                self._app,
                open_gui,
                self._app.app_actions,
                base_dir=self._app.get_base_dir(),
                run_compare_media=run_compare_media,
                extra_callback_args=extra_callback_args,
            )
            window.show()
        except Exception as e:
            self._handle_error(e, "Recent Directory Window Error")

    def open_favorites_window(self, event=None) -> None:
        """Open the favorites directory window."""
        try:
            from ui.files.favorites_window_qt import FavoritesWindow
            window = FavoritesWindow(self._app, self._app.app_actions)
            window.show()
        except Exception as e:
            self._handle_error(e, "Favorites Window Error")

    def open_directory_notes_window(self, event=None) -> None:
        """Open the directory notes window for the current base directory."""
        try:
            base_dir = self._app.get_base_dir()
            if not base_dir:
                self._app.notification_ctrl.toast(_("Please set a base directory first"))
                return

            # Re-use existing window if still open for the same directory.
            # If the directory changed, close the stale window and open a fresh one.
            if self._directory_notes_window is not None:
                try:
                    if self._directory_notes_window.isVisible():
                        if self._directory_notes_window._base_dir == base_dir:
                            self._directory_notes_window.raise_()
                            self._directory_notes_window.activateWindow()
                            return
                        else:
                            self._directory_notes_window.close()
                except (RuntimeError, AttributeError):
                    pass
                self._directory_notes_window = None

            from ui.files.directory_notes_window_qt import DirectoryNotesWindow
            self._directory_notes_window = DirectoryNotesWindow(
                self._app, self._app.app_actions, base_dir
            )
            self._directory_notes_window.show()
        except Exception as e:
            self._handle_error(e, "Directory Notes Window Error")

    # ------------------------------------------------------------------
    # Settings / configuration windows
    # ------------------------------------------------------------------
    def open_compare_settings_window(self, event=None) -> None:
        """Open the compare settings window."""
        try:
            from ui.compare.compare_settings_window_qt import CompareSettingsWindow
            CompareSettingsWindow.open(
                parent=self._app,
                compare_manager=self._app.compare_manager,
                set_file_filter=self._app.sidebar_panel.file_filter_entry.setText,
            )
        except Exception as e:
            self._handle_error(e, "Compare Settings Window Error")

    @require_password(ProtectedActions.EDIT_PREVALIDATIONS)
    def open_hf_model_manager_window(self, event=None) -> None:
        """Open the HF Hub model manager window."""
        try:
            from ui.compare.hf_model_manager_window_qt import HfModelManagerWindow
            HfModelManagerWindow.show_window(self._app, self._app.app_actions)
        except Exception as e:
            self._handle_error(e, "HF Hub Model Manager Window Error")

    @require_password(ProtectedActions.CONFIGURE_MEDIA_TYPES)
    def open_type_configuration_window(self, event=None) -> None:
        """Open the file type configuration window."""
        from ui.files.type_configuration_window_qt import TypeConfigurationWindow
        TypeConfigurationWindow.show(master=self._app, app_actions=self._app.app_actions)

    @require_password(ProtectedActions.EDIT_PREVALIDATIONS)
    def open_prevalidations_window(self, event=None) -> None:
        """Open the prevalidations window (goes to the prevalidations tab)."""
        from utils.config import config as _config
        if not _config.enable_prevalidations:
            return
        try:
            from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
            ClassifierManagementWindow.show_window(self._app, self._app.app_actions)
            mgmt = ClassifierManagementWindow._instance
            if mgmt and hasattr(mgmt, '_tabs'):
                mgmt._tabs.setCurrentIndex(1)
        except Exception as e:
            self._handle_error(e, "Prevalidations Window Error")

    @require_password(ProtectedActions.EDIT_PREVALIDATIONS, ProtectedActions.RUN_PREVALIDATIONS)
    def open_classifier_actions_window(self, event=None) -> None:
        """Open the classifier management window (classifier actions tab)."""
        try:
            from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
            ClassifierManagementWindow.show_window(self._app, self._app.app_actions)
            mgmt = ClassifierManagementWindow._instance
            if mgmt and hasattr(mgmt, '_tabs'):
                mgmt._tabs.setCurrentIndex(0)
        except Exception as e:
            self._handle_error(e, "Classifier Actions Window Error")

    # ------------------------------------------------------------------
    # File operations windows
    # ------------------------------------------------------------------
    def open_file_actions_window(self, event=None) -> None:
        """Open the file actions window."""
        try:
            from files.marked_files import MarkedFiles
            from ui.files.file_actions_window_qt import FileActionsWindow
            from ui.image.media_details import MediaDetails
            window = FileActionsWindow(
                self._app,
                self._app.app_actions,
                MediaDetails.open_temp_media_canvas,
                MarkedFiles.move_marks_to_dir_static,
            )
            window.show()
        except Exception as e:
            self._handle_error(e, "File Actions Window Error")

    @require_password(ProtectedActions.VIEW_FILE_ACTIONS)
    def open_file_action_sets_window(self, event=None) -> None:
        """Open the file action sets window."""
        try:
            from files.marked_files import MarkedFiles
            from ui.files.file_action_sets_window_qt import FileActionSetsWindow
            if FileActionSetsWindow._instance is not None:
                try:
                    if FileActionSetsWindow._instance.isVisible():
                        FileActionSetsWindow._instance.raise_()
                        FileActionSetsWindow._instance.activateWindow()
                        return
                except (RuntimeError, AttributeError):
                    FileActionSetsWindow._instance = None
            window = FileActionSetsWindow(
                self._app,
                self._app.app_actions,
                MarkedFiles.move_marks_to_dir_static,
            )
            window.show()
        except Exception as e:
            self._handle_error(e, "File Action Sets Window Error")

    # ------------------------------------------------------------------
    # Auth / admin windows
    # ------------------------------------------------------------------
    @require_password(ProtectedActions.ACCESS_ADMIN)
    def open_password_admin_window(self, event=None) -> None:
        """Open the password administration window."""
        try:
            from ui.auth.password_admin_window import PasswordAdminWindow
            PasswordAdminWindow(self._app, self._app.app_actions)
        except Exception as e:
            self._handle_error(e, "Password Admin Window Error")

    # ------------------------------------------------------------------
    # Info windows
    # ------------------------------------------------------------------
    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def open_media_details(
        self,
        event=None,
        media_path: Optional[str] = None,
        manually_keyed: bool = True,
    ) -> None:
        """
        Open the media details / metadata inspector window.

        Manages the singleton MediaDetails window reference stored on AppActions.
        """
        from ui.image.media_details import MediaDetails

        app_actions = self._app.app_actions

        # Close existing window if the session expired
        if app_actions.media_details_window() is not None:
            if check_session_expired(ProtectedActions.VIEW_MEDIA_DETAILS):
                app_actions.media_details_window().close_windows()
                app_actions.set_media_details_window(None)

        preset_media_path = True
        if media_path is None:
            media_path = self._app.media_path
            preset_media_path = False

        if not media_path:
            return

        # Build index text
        if preset_media_path:
            index_text = _("(Open this media as part of a directory to see index details.)")
        elif self._app.mode == Mode.BROWSE or self._app.is_compare_running():
            index_text = self._app.file_browser.get_index_details()
        else:
            cm = self._app.compare_manager
            _index = cm.match_index + 1
            len_matched = len(cm.files_matched)
            if self._app.mode == Mode.GROUP:
                len_groups = len(cm.file_groups)
                group_idx = cm.current_group_index + 1
                index_text = f"{_index} of {len_matched} (Group {group_idx} of {len_groups})"
            elif self._app.mode == Mode.SEARCH and self._app.is_toggled_view_matches:
                index_text = f"{_index} of {len_matched} ({self._app.file_browser.get_index_details()})"
            else:
                index_text = ""

        existing = app_actions.media_details_window()
        if existing is not None and not existing.has_closed:
            if existing.do_refresh:
                existing.update_media_details(media_path, index_text)
            if manually_keyed:
                existing.focus()
            else:
                # Keep browsing focus in the main app while details auto-refreshes.
                self._app.refocus()
        else:
            try:
                details_win = MediaDetails(
                    self._app, media_path, index_text,
                    app_actions, do_refresh=not preset_media_path,
                    take_focus=manually_keyed,
                )
                details_win.show()
                app_actions.set_media_details_window(details_win)
            except Exception as e:
                self._handle_error(e, "Image Details Error")

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def copy_prompt(self, event=None) -> None:
        """Copy the AI prompt from the currently viewed image."""
        from ui.image.media_details import MediaDetails
        MediaDetails.copy_prompt_no_break_static(
            self._app.media_navigator.get_active_media_filepath(),
            self._app,
            self._app.app_actions,
        )

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def show_related_media(self, event=None) -> None:
        """Show a related media file to the current one."""
        from ui.image.media_details import MediaDetails
        MediaDetails.show_related_image(
            master=self._app,
            image_path=self._app.media_path,
            app_actions=self._app.app_actions,
        )

    def get_help_and_config(self, event=None) -> None:
        """Open the help and configuration window."""
        try:
            from ui.help_and_config_qt import HelpAndConfig
            dialog = HelpAndConfig(parent=self._app, position_parent=self._app)
            dialog.show()
        except Exception as e:
            self._app.notification_ctrl.alert(
                "Help & Config Error", str(e), kind="error"
            )

    # ------------------------------------------------------------------
    # Secondary compare window
    # ------------------------------------------------------------------
    def open_secondary_compare_window(
        self, event=None, run_compare_media: Optional[str] = None
    ) -> None:
        """Open a new secondary window and optionally start a compare."""
        if run_compare_media is None:
            self.open_recent_directory_window(run_compare_media="")
        elif not os.path.isfile(run_compare_media):
            self._app.notification_ctrl.alert(
                _("No media selected"),
                _("No media file was selected for comparison"),
            )
        else:
            self.open_recent_directory_window(run_compare_media=self._app.media_path)

    # ------------------------------------------------------------------
    # Prevalidations (action, not window)
    # ------------------------------------------------------------------
    @require_password(ProtectedActions.RUN_PREVALIDATIONS)
    def run_prevalidations_for_base_dir(self, event=None) -> None:
        """Run all prevalidations on every file in the current directory."""
        from ui.compare.prevalidations_tab_qt import PrevalidationsTab

        fb = self._app.file_browser
        if fb.is_slow_total_files(threshold=100, use_sortable_files=True):
            ok = self._app.notification_ctrl.alert(
                _("Many Files"),
                _("Are you sure you want to run all prevalidations on directory {0} ? "
                  "This may take a while.").format(self._app.get_base_dir()),
                kind="askokcancel",
            )
            if not ok:
                logger.info("User canceled prevalidations task")
                return

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("run prevalidations")))
            return

        logger.warning("Running prevalidations for " + self._app.get_base_dir())
        from PySide6.QtWidgets import QApplication
        from files.marked_files import MarkedFiles

        # Same sidebar badge as dynamic (video/GIF/PDF) prevalidation on navigate
        # (compare_wrapper._run_dynamic_prevalidation_with_spinner); here the work
        # runs on the main thread with processEvents, so the spinner still animates.
        total_files = fb.count()
        self._app.app_actions.start_loading_spinner(force=True)
        self._app.app_actions.start_progress_bar()
        try:
            directory_was_excluded = PrevalidationsTab.remove_directory_from_exclusion_list(self._app.get_base_dir())
            files_checked = 0
            outcomes = 0
            moves = 0
            copies = 0
            deletes = 0
            errors = 0
            for media_path in fb.get_files():
                files_checked += 1
                try:
                    # No blur_callback: this pass does not navigate to each file, so a
                    # display-only BLUR action would have nothing to apply to.
                    result = PrevalidationsTab.prevalidate(
                        media_path,
                        self._app.get_base_dir,
                        self._app.file_ops_ctrl.hide_current_media,
                        self._app.notification_ctrl.title_notify,
                        MarkedFiles.add_mark_if_not_present,
                        force=False,  # TODO optionally allow force=True via different keybind
                    )
                    if result is not None:
                        outcomes += 1
                        if result == ClassifierActionType.MOVE:
                            moves += 1
                        elif result == ClassifierActionType.COPY:
                            copies += 1
                        elif result == ClassifierActionType.DELETE:
                            deletes += 1
                except Exception as e:
                    errors += 1
                    logger.error(e)
                self._app.app_actions._set_label_state(
                    _("Prevalidations: {0} / {1}").format(files_checked, total_files)
                )
                # Keep the UI responsive during long-running prevalidation
                QApplication.processEvents()
            if directory_was_excluded:
                PrevalidationsTab.add_directory_to_exclusion_list(self._app.get_base_dir())

            # Rescan from disk so counts and paths match any moves/deletes (fast once
            # the directory is loaded; *force* bypasses the incremental-load refresh guard).
            self._app.refresh(force=True)
            self._app.notification_ctrl.toast(
                _(
                    "All prevalidations in this directory have finished.\n"
                    "Files checked: {0} — outcomes: {1} — moves: {2} — copies: {3} — deletes: {4} — errors: {5}"
                ).format(files_checked, outcomes, moves, copies, deletes, errors),
                time_in_seconds=10,
            )
        finally:
            self._app.app_actions.stop_loading_spinner()
            self._app.app_actions.stop_progress_bar()

    @require_password(ProtectedActions.RUN_PREVALIDATIONS)
    def toggle_prevalidations(self, event=None) -> None:
        """Toggle prevalidations on or off."""
        from utils.config import config as _config
        _config.enable_prevalidations = not _config.enable_prevalidations
        self._app.notification_ctrl.toast(
            _("Prevalidations now running") if _config.enable_prevalidations
            else _("Prevalidations turned off")
        )

    def toggle_extra_debug_logging(self, event=None) -> None:
        """Toggle the extra-verbose debug logging flag."""
        from utils.config import config as _config
        _config.debug = not _config.debug
        _config.debug2 = not _config.debug2
        set_logger_level(_config.debug)
        self._app.notification_ctrl.toast(
            _("Extra debug logging enabled") if _config.debug2
            else _("Extra debug logging disabled")
        )

    # ------------------------------------------------------------------
    # Directory note operations
    # ------------------------------------------------------------------
    def toggle_directory_note_mark(self, event=None) -> None:
        """Toggle a file's marked status in directory notes."""
        from files.directory_notes import DirectoryNotes

        media_path = self._app.media_navigator.get_active_media_filepath()
        if not media_path:
            return

        base_dir = self._app.get_base_dir()
        if DirectoryNotes.is_marked_file(base_dir, media_path):
            DirectoryNotes.remove_marked_file(base_dir, media_path)
            self._app.notification_ctrl.toast(
                _("Removed from directory notes: {0}").format(os.path.basename(media_path))
            )
        else:
            DirectoryNotes.add_marked_file(base_dir, media_path)
            self._app.notification_ctrl.toast(
                _("Added to directory notes: {0}").format(os.path.basename(media_path))
            )

        # Refresh the notes window if it is open
        if self._directory_notes_window is not None:
            try:
                if self._directory_notes_window.isVisible():
                    self._directory_notes_window._refresh()
            except (RuntimeError, AttributeError):
                pass

    def edit_file_note(self, event=None) -> None:
        """Open a dialog to edit the note for the current file."""
        from files.directory_notes import DirectoryNotes
        from PySide6.QtWidgets import (
            QDialog, QLabel, QPlainTextEdit, QPushButton,
            QHBoxLayout, QVBoxLayout,
        )
        from ui.app_style import AppStyle

        media_path = self._app.media_navigator.get_active_media_filepath()
        if not media_path:
            return

        base_dir = self._app.get_base_dir()
        current_note = DirectoryNotes.get_file_note(base_dir, media_path) or ""

        dialog = QDialog(self._app)
        dialog.setWindowTitle(_("Edit Note - {0}").format(os.path.basename(media_path)))
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)
        path_label = QLabel(media_path, dialog)
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        note_edit = QPlainTextEdit(dialog)
        note_edit.setPlainText(current_note)
        layout.addWidget(note_edit)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton(_("Save"), dialog)
        cancel_btn = QPushButton(_("Cancel"), dialog)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def save_note():
            new_note = note_edit.toPlainText().strip()
            DirectoryNotes.set_file_note(base_dir, media_path, new_note)
            self._app.notification_ctrl.toast(
                _("Note saved for: {0}").format(os.path.basename(media_path))
            )
            dialog.accept()
            if self._directory_notes_window is not None:
                try:
                    if self._directory_notes_window.isVisible():
                        self._directory_notes_window._refresh()
                except (RuntimeError, AttributeError):
                    pass

        save_btn.clicked.connect(save_note)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()


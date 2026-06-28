"""
MarkedFiles window -- move/copy marked files to target directories (PySide6).

Self-contained: class-level state, persistence, action runners, core file
operations, and two-mode UI (scrollable directory list or translucent
non-GUI mode).

Uses ``FileAction`` from ``files.file_action`` and
``FileActionsWindow`` from ``ui.files.file_actions_window_qt``.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QProgressDialog, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from files.marked_files import MarkedFiles
from files.file_action import FileAction
from image.frame_cache import FrameCache
from lib.fast_directory_picker_qt import get_existing_directory
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from ui.auth.password_utils import require_password
from utils.app_actions import AppActions
from utils.config import config
from utils.constants import Mode, ProtectedActions
from utils.logging_setup import get_logger
from utils.translations import _, compare_running_warn, marks_transfer_running_warn
from utils.utils import Utils
logger = get_logger("marked_file_mover_qt")


class MarkedFileMover(SmartDialog):
    """
    Move / copy / delete marked files to target directories.

    Two window modes:

    * **GUI** (``is_gui=True``, Ctrl+M): full action bar + scrollable
      directory list.
    * **Translucent** (``is_gui=False``, Ctrl+K): tiny semi-transparent
      dialog with no widgets; the user types to filter, presses Enter.
    """

    _current_window: Optional[MarkedFileMover] = None
    # Runtime-only memory for recovering dropped first-letter filters.
    # key: normalized user filter text, value: resolved target directory.
    _session_filter_target_map: dict[str, str] = {}

    MAX_HEIGHT: int = 900
    COL_0_WIDTH: int = 600
    LARGE_FILE_OP_CONFIRM_THRESHOLD: int = 25
    LARGE_FILE_OP_SHOW_PROGRESS_THRESHOLD: int = 100

    @staticmethod
    def build_marks_progress(parent, total: int, move_func=None):
        """
        Return (QProgressDialog, callback) for large operations, or (None, None) when
        the count is below LARGE_FILE_OP_CONFIRM_THRESHOLD.  The callback updates the
        dialog and sets MarkedFiles.is_cancelled_action if the user cancels.
        """
        if total < MarkedFileMover.LARGE_FILE_OP_SHOW_PROGRESS_THRESHOLD:
            return None, None
        if move_func == Utils.copy_file:
            label = _("Copying {0} files...").format(total)
        elif move_func == Utils.move_file:
            label = _("Moving {0} files...").format(total)
        else:
            label = _("Processing {0} files...").format(total)
        progress = QProgressDialog(label, _("Cancel"), 0, total, parent)
        progress.setWindowTitle(_("File Operation"))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(250)
        progress.setValue(0)

        # One update per percent (~100 steps total).
        _step = max(1, total // 100)

        def _callback(done: int, _total: int) -> None:
            if done < _step or done % _step == 0 or done == _total:
                progress.setValue(done)
            if progress.wasCanceled():
                MarkedFiles.is_cancelled_action = True

        return progress, _callback

    @staticmethod
    def run_marks_action_with_progress(app, current_media, action_func, move_func) -> None:
        """
        Build a progress dialog for the current marks, call action_func, then
        ensure the dialog is closed.  action_func must accept
        (app_actions, current_media=, *, ui_class=, progress_callback=).
        """
        total = len(MarkedFiles.file_marks)
        progress, callback = MarkedFileMover.build_marks_progress(app, total, move_func)
        action_func(
            app.app_actions,
            current_media=current_media,
            ui_class=MarkedFileMover,
            progress_callback=callback,
        )
        if progress is not None:
            progress.setValue(total)

    # ==================================================================
    # Static helper methods
    # ==================================================================
    @staticmethod
    def get_target_directory(
        target_dir, starting_target, app_actions, parent=None
    ):
        """Validate *target_dir* or ask the user to pick one."""
        if target_dir:
            if os.path.isdir(target_dir):
                return target_dir, True
            else:
                if target_dir in MarkedFiles.mark_target_dirs:
                    MarkedFiles.mark_target_dirs.remove(target_dir)
                app_actions.warn(
                    _("Invalid directory: {0}").format(target_dir)
                )
        target_dir = get_existing_directory(
            parent,
            _("Select target directory for marked files"),
            starting_target or "",
            quick_access_locations=[(app_actions.get_base_dir(), _("Current Base Directory"))],
        )
        return target_dir, False

    @staticmethod
    def undo_move_marks(target_dir, app_actions) -> None:
        """Undo the previous move/copy operation."""
        def get_base_dir_callback():
            base_dir = get_existing_directory(
                None,
                _("Where should the marked files have gone?"),
                target_dir or "",
                quick_access_locations=[(app_actions.get_base_dir(), _("Current Base Directory"))],
            )
            return base_dir
        return MarkedFiles.undo_move_marks(
            target_dir, app_actions,
            get_base_dir_callback=get_base_dir_callback,
            get_target_dir_callback=MarkedFileMover.get_target_directory
        )

    # ==================================================================
    # Factory
    # ==================================================================
    @staticmethod
    def show_window(
        master,
        is_gui: bool,
        single_image,
        current_media,
        app_mode,
        app_actions,
        base_dir: str = ".",
    ):
        """Create or focus the MarkedFiles dialog. Returns the instance."""
        if MarkedFileMover._current_window is not None:
            try:
                win = MarkedFileMover._current_window
                if win.isVisible():
                    win.setWindowTitle(_("Move {0} Marked File(s)").format(len(MarkedFiles.file_marks)))
                    win.setWindowOpacity(1.0)
                    if win.isMinimized():
                        win.showNormal()
                    win.raise_()
                    win.activateWindow()
                    return win
            except Exception:
                MarkedFileMover._current_window = None
            else:
                # Closed / hidden instance still parented to *master* — drop it before
                # constructing another or orphaned dialogs accumulate (same as modal leaks).
                old = MarkedFileMover._current_window
                MarkedFileMover._current_window = None
                old.deleteLater()

        window = MarkedFileMover(
            master,
            is_gui,
            single_image,
            current_media,
            app_mode,
            app_actions,
            base_dir,
        )
        window.show()
        return window

    # ==================================================================
    # Construction
    # ==================================================================
    def __init__(
        self,
        master: QWidget,
        is_gui: bool,
        single_image,
        current_media,
        app_mode,
        app_actions: AppActions,
        base_dir: str = ".",
    ) -> None:
        geometry = "600x500" if is_gui else "300x100"
        super().__init__(
            parent=master,
            position_parent=master,
            title=_("Move {0} Marked File(s)").format(len(MarkedFiles.file_marks)),
            geometry=geometry,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        MarkedFileMover._current_window = self

        self._is_gui = is_gui
        self._single_image = single_image
        self._current_media = current_media
        self._app_mode = app_mode
        self._app_actions = app_actions
        self._base_dir = os.path.normpath(base_dir)
        self._filter_text: str = ""
        self._filtered_target_dirs: list[str] = (
            MarkedFiles.mark_target_dirs[:]
        )
        self._is_sorted_by_embedding = False
        self._used_filter_fallback = False

        if MarkedFiles.last_set_target_dir and os.path.isdir(
            MarkedFiles.last_set_target_dir
        ):
            self._starting_target = MarkedFiles.last_set_target_dir
        else:
            self._starting_target = base_dir

        self._do_set_permanent_mark_target = False
        self._do_set_hotkey_action = -1

        if not is_gui:
            self.setWindowOpacity(0.3)
        else:
            self._build_gui()

        # -- keyboard shortcuts -------------------------------------------
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(
            self.close_windows
        )
        QShortcut(QKeySequence("Shift+Delete"), self).activated.connect(
            self._delete_marked_files
        )
        QShortcut(QKeySequence("Ctrl+T"), self).activated.connect(
            self._set_permanent_mark_target
        )
        # Deliberately not Ctrl+S: that is the global slideshow toggle; a window-level
        # Ctrl+S shortcut here stole the chord and fired sort unintentionally. Use a
        # distinct tri-modifier chord (Ctrl+Shift+O is unused globally).
        QShortcut(QKeySequence("Ctrl+Shift+O"), self).activated.connect(
            self._sort_target_dirs_by_embedding
        )
        QShortcut(QKeySequence("Ctrl+H"), self).activated.connect(
            self._open_hotkey_actions_window
        )
        QShortcut(QKeySequence("Ctrl+M"), self).activated.connect(
            self._clear_manual_filter
        )
        QShortcut(QKeySequence("Ctrl+K"), self).activated.connect(
            self._clear_manual_filter
        )

        QTimer.singleShot(1, self.activateWindow)

    # ==================================================================
    # GUI building
    # ==================================================================
    def _build_gui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # -- action bar ---------------------------------------------------
        bar = QHBoxLayout()
        bar.setSpacing(4)

        new_lbl = QLabel(_("New target:"))
        new_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        bar.addWidget(new_lbl)

        move_new_btn = QPushButton(_("MOVE"))
        move_new_btn.setFocusPolicy(Qt.NoFocus)
        move_new_btn.clicked.connect(
            lambda: self._handle_target_directory(move_func=Utils.move_file)
        )
        bar.addWidget(move_new_btn)

        copy_new_btn = QPushButton(_("COPY"))
        copy_new_btn.setFocusPolicy(Qt.NoFocus)
        copy_new_btn.clicked.connect(
            lambda: self._handle_target_directory(move_func=Utils.copy_file)
        )
        bar.addWidget(copy_new_btn)

        del_btn = QPushButton(_("DELETE"))
        del_btn.setFocusPolicy(Qt.NoFocus)
        del_btn.clicked.connect(self._delete_marked_files)
        bar.addWidget(del_btn)

        add_parent_btn = QPushButton(_("Add from parent"))
        add_parent_btn.setFocusPolicy(Qt.NoFocus)
        add_parent_btn.clicked.connect(self._set_target_dirs_from_dir)
        bar.addWidget(add_parent_btn)

        clear_btn = QPushButton(_("Clear targets"))
        clear_btn.setFocusPolicy(Qt.NoFocus)
        clear_btn.clicked.connect(self._clear_target_dirs)
        bar.addWidget(clear_btn)

        interp_gif_btn = QPushButton(_("Interpolation GIF"))
        interp_gif_btn.setFocusPolicy(Qt.NoFocus)
        interp_gif_btn.clicked.connect(lambda _checked=False: self._create_interpolation_gif_from_marks())
        bar.addWidget(interp_gif_btn)

        pdf_btn = QPushButton(_("Create PDF"))
        pdf_btn.setFocusPolicy(Qt.NoFocus)
        # QPushButton.clicked emits a bool(checked) argument; ignore it so the
        # PDF API never mistakes it for an output path.
        pdf_btn.clicked.connect(lambda _checked=False: self._create_pdf_from_marks())
        bar.addWidget(pdf_btn)

        diff_pdf_btn = QPushButton(_("Diff PDFs"))
        diff_pdf_btn.setFocusPolicy(Qt.NoFocus)
        diff_pdf_btn.clicked.connect(
            lambda _checked=False: self._create_diff_pdf_from_marks()
        )
        bar.addWidget(diff_pdf_btn)

        root.addLayout(bar)

        # -- scroll area for directory rows -------------------------------
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFocusPolicy(Qt.NoFocus)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {AppStyle.BG_COLOR}; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_content.setFocusPolicy(Qt.NoFocus)
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(2)
        self._scroll.setWidget(self._scroll_content)
        root.addWidget(self._scroll, 1)

        # -- filter indicator ---------------------------------------------
        self._filter_label = QLabel("")
        self._filter_label.setStyleSheet("color: orange; font-style: italic;")
        self._filter_label.setVisible(False)
        root.addWidget(self._filter_label)

        self._rebuild_directory_rows()

    def _rebuild_directory_rows(self) -> None:
        """Clear and rebuild the scrollable directory list."""
        _clear_layout(self._scroll_layout)

        for target_dir in self._filtered_target_dirs:
            row = QHBoxLayout()

            dir_label = QLabel(target_dir)
            dir_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            dir_label.setWordWrap(True)
            row.addWidget(dir_label, 1)

            move_btn = QPushButton(_("Move"))
            move_btn.setFocusPolicy(Qt.NoFocus)
            move_btn.clicked.connect(
                lambda _=False, d=target_dir: self._move_marks_to_dir(target_dir=d)
            )
            row.addWidget(move_btn)

            copy_btn = QPushButton(_("Copy"))
            copy_btn.setFocusPolicy(Qt.NoFocus)
            copy_btn.clicked.connect(
                lambda _=False, d=target_dir: self._move_marks_to_dir(
                    target_dir=d, move_func=Utils.copy_file
                )
            )
            row.addWidget(copy_btn)

            toggle_lock_text = (
                _("Unlock")
                if MarkedFiles.is_target_dir_require_gui(target_dir)
                else _("Lock")
            )
            lock_btn = QPushButton(toggle_lock_text)
            lock_btn.setFocusPolicy(Qt.NoFocus)
            lock_btn.setToolTip(_("Toggle GUI requirement for this target directory"))
            lock_btn.clicked.connect(
                lambda _=False, d=target_dir: self._toggle_target_require_gui(d)
            )
            row.addWidget(lock_btn)

            remove_btn = QPushButton("\u00d7")  # multiplication sign
            remove_btn.setFocusPolicy(Qt.NoFocus)
            remove_btn.setFixedWidth(28)
            remove_btn.setToolTip(_("Remove this target directory"))
            remove_btn.clicked.connect(
                lambda _=False, d=target_dir: self._remove_single_target(d)
            )
            row.addWidget(remove_btn)

            self._scroll_layout.addLayout(row)

        self._scroll_layout.addStretch()

    # ==================================================================
    # Instance action methods
    # ==================================================================
    def _handle_target_directory(
        self, target_dir=None, move_func=Utils.move_file
    ) -> Optional[str]:
        """Validate/ask for target dir, add to list, trigger action."""
        target_dir, target_was_valid = MarkedFileMover.get_target_directory(
            target_dir, self._starting_target, self._app_actions, parent=self
        )
        if not target_dir or not os.path.isdir(target_dir):
            # Keep this window open when directory picking is cancelled/invalid.
            # Clear any typed filter so the full list is immediately visible.
            self._clear_manual_filter()
            return None

        if target_was_valid:
            return target_dir

        target_dir = os.path.normpath(target_dir)
        if target_dir not in MarkedFiles.mark_target_dirs:
            MarkedFiles.mark_target_dirs.append(target_dir)
            MarkedFiles.mark_target_dirs.sort()

        if move_func is not None:
            self._move_marks_to_dir(target_dir=target_dir, move_func=move_func)
        else:
            self._test_is_in_directory(target_dir=target_dir)
        return target_dir

    def _clear_manual_filter(self) -> None:
        """Clear typed filter text and refresh the current directory list."""
        self._filter_text = ""
        self._apply_filter()

    def _record_filter_resolution_for_session(self, target_dir: Optional[str]) -> None:
        """
        Store filter->target mapping for this app session only.
        Skip if filter is too short or if fallback logic was used.
        """
        if target_dir is None:
            return
        filter_key = self._filter_text.strip().lower()
        if len(filter_key) < 2:
            return
        if self._used_filter_fallback:
            return
        if target_dir not in self._filtered_target_dirs:
            return
        MarkedFileMover._session_filter_target_map[filter_key] = target_dir

    def _toggle_target_require_gui(self, target_dir: str) -> None:
        require_gui = not MarkedFiles.is_target_dir_require_gui(target_dir)
        MarkedFiles.set_single_target_require_gui(target_dir, require_gui)
        if self._is_gui:
            self._rebuild_directory_rows()
        status_text = _("locked") if require_gui else _("unlocked")
        self._app_actions.toast(
            _("Target directory {0}: {1}").format(target_dir, status_text)
        )

    def _ensure_target_allowed_for_current_mode(self, target_dir: Optional[str]) -> bool:
        if target_dir is None:
            return False
        if self._is_gui:
            return True
        if not MarkedFiles.is_target_dir_require_gui(target_dir):
            return True
        self._clear_manual_filter()
        self._app_actions.alert(
            _("Target is locked"),
            _(
                "This target requires the GUI Marked File Mover window.\n\n"
                "Open the GUI mover (Ctrl+M) to confirm this destination."
            ),
            kind="warning",
            master=self,
        )
        return False

    def _move_marks_to_dir(
        self, target_dir=None, move_func=Utils.move_file
    ) -> None:
        target_dir = self._handle_target_directory(
            target_dir=target_dir, move_func=None  # prevent recursion
        )
        if target_dir is None:
            return
        if not self._ensure_target_allowed_for_current_mode(target_dir):
            return
        self._record_filter_resolution_for_session(target_dir)
        if not self._confirm_large_file_operation(target_dir, move_func):
            return
        if (
            config.debug
            and self._filter_text
            and self._filter_text.strip() != ""
        ):
            logger.debug(f"Filtered by string: {self._filter_text}")

        if self._do_set_permanent_mark_target:
            FileAction.set_permanent_action(
                target_dir, move_func, self._app_actions.toast
            )
            self._do_set_permanent_mark_target = False

        if self._do_set_hotkey_action > -1:
            FileAction.set_hotkey_action(
                self._do_set_hotkey_action,
                target_dir,
                move_func,
                self._app_actions.toast,
            )
            self._do_set_hotkey_action = -1

        total = len(MarkedFiles.file_marks)
        progress, callback = MarkedFileMover.build_marks_progress(self, total, move_func)
        MarkedFiles.move_marks_to_dir_static(
            self._app_actions,
            target_dir=target_dir,
            move_func=move_func,
            single_image=self._single_image,
            current_media=self._current_media,
            progress_callback=callback,
        )
        if progress is not None:
            progress.setValue(total)
            from ui.sound_player import play_sound
            QTimer.singleShot(0, lambda: play_sound("success"))
        self.close_windows()

    def _confirm_large_file_operation(self, target_dir: str, move_func) -> bool:
        """Ask for confirmation before very large move/copy operations."""
        marked_count = len(MarkedFiles.file_marks)
        if marked_count < self.LARGE_FILE_OP_CONFIRM_THRESHOLD:
            return True

        is_move = move_func == Utils.move_file
        if is_move:
            warning_text = _(
                "You are about to move {0} marked files to:\n\n{1}\n\n"
                "Please confirm this destination is correct."
            ).format(marked_count, target_dir)
        else:
            warning_text = _(
                "You are about to copy {0} marked files to:\n\n{1}\n\n"
                "Please confirm this destination is correct."
            ).format(marked_count, target_dir)
        return bool(
            self._app_actions.alert(
                _("Confirm Large File Operation"),
                warning_text,
                kind="askokcancel",
                severity="high",
                master=self,
            )
        )

    def _delete_marked_files(self) -> None:
        if MarkedFiles.is_transfer_running():
            self._app_actions.warn(
                marks_transfer_running_warn(_("delete marked files"))
            )
            return
        if self._app_actions.is_compare_running():
            self._app_actions.warn(compare_running_warn(_("delete files")))
            return

        if not self._app_actions.alert(
            _("Confirm Delete"),
            _("Deleting {0} marked files - Are you sure you want to proceed?").format(
                len(MarkedFiles.file_marks)
            ),
            kind="askokcancel",
            severity="high" if len(MarkedFiles.file_marks) > 5 else "normal",
            master=self,
        ):
            return

        if self._current_media and self._current_media in MarkedFiles.file_marks:
            self._app_actions.release_media_canvas()

        removed_files: list[str] = []
        failed_to_delete: list[str] = []
        total = len(MarkedFiles.file_marks)
        # build_marks_progress uses move_func=None → "Processing N files..." label.
        # Override label and strip the cancel button: deletion is not reversible mid-run.
        progress, _unused = MarkedFileMover.build_marks_progress(self, total, move_func=None)
        if progress is not None:
            progress.setLabelText(_("Deleting {0} files...").format(total))
            progress.setCancelButton(None)

        _step = max(1, total // 100)
        for i, filepath in enumerate(MarkedFiles.file_marks):
            try:
                if config.enable_svgs and filepath.lower().endswith(".svg"):
                    FrameCache.remove_from_cache(
                        filepath, delete_temp_file=True
                    )
                self._app_actions.delete(filepath, manual_delete=False)
                removed_files.append(filepath)
            except Exception as e:
                logger.error(f"Failed to delete {filepath}: {e}")
                if os.path.exists(filepath):
                    failed_to_delete.append(filepath)
            done = i + 1
            if progress is not None and (done % _step == 0 or done == total):
                progress.setValue(done)

        MarkedFiles.file_marks.clear()
        if failed_to_delete:
            MarkedFiles.file_marks.extend(failed_to_delete)
            self._app_actions.alert(
                _("Delete Failed"),
                _("Failed to delete {0} files - check log for details.").format(len(failed_to_delete)),
                kind="warning",
                master=self,
            )
        else:
            self._app_actions.warn(_("Deleted {0} marked files.").format(len(removed_files)))

        self._app_actions.refresh(removed_files=removed_files if self._app_mode != Mode.BROWSE else [])
        self.close_windows()

    def _remove_single_target(self, target_dir: str) -> None:
        """Remove a single directory from the target list."""
        if target_dir in MarkedFiles.mark_target_dirs:
            MarkedFiles.mark_target_dirs.remove(target_dir)
        MarkedFiles.set_single_target_require_gui(target_dir, False)
        if target_dir in self._filtered_target_dirs:
            self._filtered_target_dirs.remove(target_dir)
        if self._is_gui:
            self._rebuild_directory_rows()

    def _clear_target_dirs(self) -> None:
        MarkedFiles.mark_target_dirs.clear()
        MarkedFiles.mark_target_require_gui.clear()
        self._filtered_target_dirs.clear()
        if self._is_gui:
            self._rebuild_directory_rows()

    def _set_target_dirs_from_dir(self) -> None:
        parent_dir = get_existing_directory(
            self,
            _("Select parent directory for target directories"),
            self._starting_target or "",
            quick_access_locations=[(self._base_dir, _("Current Base Directory"))],
        )
        if not parent_dir or not os.path.isdir(parent_dir):
            return

        for name in os.listdir(parent_dir):
            dirpath = os.path.normpath(os.path.join(parent_dir, name))
            if os.path.isdir(dirpath) and dirpath != self._base_dir:
                if dirpath not in MarkedFiles.mark_target_dirs:
                    MarkedFiles.mark_target_dirs.append(dirpath)

        MarkedFiles.mark_target_dirs.sort()
        self._filtered_target_dirs = MarkedFiles.mark_target_dirs[:]
        self._filter_text = ""
        if self._is_gui:
            self._rebuild_directory_rows()
            self._filter_label.setVisible(False)

    @require_password(ProtectedActions.SET_HOTKEY_ACTIONS)
    def _open_hotkey_actions_window(self) -> None:
        try:
            from ui.files.hotkey_actions_window_qt import HotkeyActionsWindow

            win = HotkeyActionsWindow(
                self,
                self._app_actions,
                self._set_permanent_mark_target,
                self.set_hotkey_action,
            )
            win.show()
        except Exception as e:
            self._app_actions.alert(
                _("Error"),
                "Error opening hotkey actions window: " + str(e),
                master=self,
            )

    @require_password(
        ProtectedActions.SET_HOTKEY_ACTIONS,
        custom_text=_(
            "WARNING: This action sets hotkey actions that will be used "
            "for future file operations. You may have accidentally triggered "
            "this shortcut due to a sticky Control key. Please confirm you "
            "want to proceed."
        ),
        allow_unauthenticated=False,
    )
    def _set_permanent_mark_target(self) -> None:
        self._do_set_permanent_mark_target = True
        logger.debug("Setting permanent mark target hotkey action")
        self._app_actions.toast(_("Recording next mark target and action."))

    def set_hotkey_action(self, event=None, hotkey_override=None) -> None:
        assert event is not None or hotkey_override is not None
        self._do_set_hotkey_action = int(hotkey_override) if hotkey_override is not None else -1
        logger.debug(f"Doing set hotkey action: {self._do_set_hotkey_action}")
        self._app_actions.toast(_("Recording next mark target and action."))

    def _sort_target_dirs_by_embedding(self) -> None:
        """Reorder filtered targets by CLIP similarity (current image vs preset texts)."""
        if not self._is_gui:
            return
        if len(self._filtered_target_dirs) < 2:
            return

        pivot_image = self._current_media
        if not pivot_image or not isinstance(pivot_image, str) or not os.path.isfile(
            pivot_image
        ):
            logger.debug(
                "Sort by embedding skipped: invalid current image path (%r)",
                pivot_image,
            )
            return

        from compare.compare_embeddings_clip import CompareEmbeddingClip

        embedding_texts: dict[str, str] = {}
        for d in self._filtered_target_dirs:
            embedding_text = self._get_embedding_text_for_dirpath(d)
            if embedding_text is not None and embedding_text.strip() != "":
                embedding_texts[d] = embedding_text

        if not embedding_texts:
            logger.debug(
                "Sort by embedding skipped: no filtered targets matched text "
                "embedding search presets"
            )
            return

        similarities = CompareEmbeddingClip.single_text_compare(
            pivot_image, embedding_texts
        )
        self._filtered_target_dirs = [
            dirpath
            for dirpath, _unused in sorted(similarities.items(), key=lambda x: -x[1])
        ]
        self._is_sorted_by_embedding = True
        self._rebuild_directory_rows()
        self._app_actions.toast(
            _("Sorted directories by embedding comparison.")
        )

    def _get_embedding_text_for_dirpath(self, dirpath: str) -> Optional[str]:
        basename = os.path.basename(dirpath)
        for text in config.text_embedding_search_presets:
            if basename == text or re.search(f"(^|_| ){text}($|_| )", basename):
                logger.info(f"Found embeddable directory for text {text}: {dirpath}")
                return text
        return None

    def _create_interpolation_gif_from_marks(self) -> None:
        MarkedFileMover.create_interpolation_gif_from_marks(self._app_actions)

    @staticmethod
    def create_interpolation_gif_from_marks(app_actions: "AppActions") -> None:
        """Create a smooth optical-flow interpolation GIF from exactly 2 marked files."""
        marks = MarkedFiles.file_marks
        if len(marks) != 2:
            app_actions.warn(
                _("Exactly 2 marked files are required to create an interpolation GIF "
                  "({0} currently marked).").format(len(marks))
            )
            return
        from image.image_ops import ImageOps
        from utils.running_tasks_registry import start_thread

        path_a, path_b = marks[0], marks[1]

        def _run():
            try:
                output = ImageOps.create_interpolation_gif(path_a, path_b)
                app_actions.toast(
                    _("Interpolation GIF saved: {0}").format(os.path.basename(output))
                )
            except Exception as e:
                logger.error("Interpolation GIF failed: %s", e)
                app_actions.warn(
                    _("Failed to create interpolation GIF: {0}").format(str(e))
                )

        start_thread(_run, use_asyncio=False)

    def _create_pdf_from_marks(self, output_path=None) -> None:
        from files.pdf_creator import PDFCreator
        from ui.files.pdf_options_window_qt import PDFOptionsWindow
        from PySide6.QtWidgets import QFileDialog

        # Defensive guard for any direct signal wiring that might pass a bool.
        if isinstance(output_path, bool):
            output_path = None
        save_cancelled = False

        def _save_file_dialog(default_dir, default_name):
            nonlocal save_cancelled
            path, _filter = QFileDialog.getSaveFileName(
                self,
                _("Save PDF as"),
                os.path.join(default_dir, default_name + ".pdf"),
                "PDF files (*.pdf)",
            )
            if not path:
                save_cancelled = True
                return None
            return path

        def pdf_callback(options):
            ok = PDFCreator.create_pdf_from_files(
                MarkedFiles.file_marks, self._app_actions, output_path, options,
                save_file_callback=_save_file_dialog,
            )
            if not ok and not save_cancelled:
                # Ensure failures are visible even if lower-level code only logs.
                self._app_actions.alert(
                    _("Error"),
                    _("Failed to create PDF from marked files."),
                    kind="error",
                    master=self,
                )

        PDFOptionsWindow.show(self, self._app_actions, pdf_callback)

    def _create_diff_pdf_from_marks(self) -> None:
        from files.pdf_creator import PDFCreator
        from ui.files.pdf_options_window_qt import PDFOptionsWindow

        initial_filename = PDFCreator.get_default_diff_filename(
            MarkedFiles.file_marks
        )

        def diff_callback(options):
            PDFCreator.create_diff_pdf_from_files(
                MarkedFiles.file_marks,
                self._app_actions,
                options=options,
            )

        PDFOptionsWindow.show(
            self,
            self._app_actions,
            diff_callback,
            initial_filename=initial_filename,
        )

    def _test_is_in_directory(
        self, target_dir=None, shift: bool = False
    ) -> None:
        target_dir = self._handle_target_directory(target_dir=target_dir, move_func=None)
        if target_dir is None:
            return
        if not self._ensure_target_allowed_for_current_mode(target_dir):
            return
        self._record_filter_resolution_for_session(target_dir)
        if (
            config.debug
            and self._filter_text
            and self._filter_text.strip() != ""
        ):
            logger.debug(f"Filtered by string: {self._filter_text}")

        if shift:
            self._find_is_downstream_related_image_in_directory(
                target_dir=target_dir
            )
        else:
            MarkedFiles.test_in_directory_static(
                self._app_actions,
                target_dir=target_dir,
                single_image=self._single_image,
            )
        self.close_windows()

    def _do_action_test_is_in_directory(
        self, *, ctrl: bool = False, alt: bool = False, shift: bool = False
    ) -> None:
        target_dir = None
        if alt:
            penultimate_action = FileAction.get_history_action(start_index=1)
            if penultimate_action is not None and os.path.isdir(penultimate_action.target):
                target_dir = penultimate_action.target
        elif len(self._filtered_target_dirs) == 0 or ctrl:
            self._handle_target_directory(move_func=None)
            return
        else:
            if (
                len(self._filtered_target_dirs) == 1
                or self._filter_text.strip() != ""
                or self._is_sorted_by_embedding
            ):
                target_dir = self._filtered_target_dirs[0]
            else:
                target_dir = MarkedFiles.last_set_target_dir

        if target_dir is None:
            self._handle_target_directory(move_func=None)
        else:
            self._test_is_in_directory(target_dir=target_dir, shift=shift)

    def _find_is_downstream_related_image_in_directory(self, target_dir: str) -> None:
        from files.file_browser import FileBrowser
        from image.image_data_extractor import image_data_extractor

        if (
            MarkedFiles.file_browser is None
            or MarkedFiles.file_browser.directory != target_dir
            or not MarkedFiles.file_browser.recursive
        ):
            MarkedFiles.file_browser = FileBrowser(
                directory=target_dir, recursive=True
            )
        MarkedFiles.file_browser._gather_files(files=None)

        marked_file_basenames = [os.path.basename(f) for f in MarkedFiles.file_marks]
        downstream_related_images: list[str] = []
        for path in MarkedFiles.file_browser.filepaths:
            if path in MarkedFiles.file_marks:
                continue
            related_image_path = image_data_extractor.get_related_image_path(path)
            if related_image_path is not None:
                if related_image_path in MarkedFiles.file_marks:
                    downstream_related_images.append(path)
                else:
                    file_basename = os.path.basename(related_image_path)
                    if (
                        len(file_basename) > 10
                        and file_basename in marked_file_basenames
                    ):
                        downstream_related_images.append(path)

        if downstream_related_images:
            for image in downstream_related_images:
                logger.warning(f"Downstream related image found: {image}")
            self._app_actions.toast(
                _("Found {0} downstream related images").format(
                    len(downstream_related_images)
                )
            )
        else:
            self._app_actions.toast(_("No downstream related images found"))

    # ==================================================================
    # Paging
    # ==================================================================
    def _page_up(self) -> None:
        paging_len = max(1, len(self._filtered_target_dirs) // 10)
        idx = len(self._filtered_target_dirs) - paging_len
        self._filtered_target_dirs = (
            self._filtered_target_dirs[idx:]
            + self._filtered_target_dirs[:idx]
        )
        if self._is_gui:
            self._rebuild_directory_rows()

    def _page_down(self) -> None:
        paging_len = max(1, len(self._filtered_target_dirs) // 10)
        self._filtered_target_dirs = (
            self._filtered_target_dirs[paging_len:]
            + self._filtered_target_dirs[:paging_len]
        )
        if self._is_gui:
            self._rebuild_directory_rows()

    # ==================================================================
    # Filtering (4-pass ranked matching)
    # ==================================================================
    def _apply_filter(self) -> None:
        if self._filter_text:
            self._filter_label.setText(
                _("Filter: {0}").format(self._filter_text)
            ) if self._is_gui else None
            if self._is_gui:
                self._filter_label.setVisible(True)
        else:
            if self._is_gui:
                self._filter_label.setVisible(False)

        ft = self._filter_text.strip().lower()
        if not ft:
            self._filtered_target_dirs = MarkedFiles.mark_target_dirs[:]
            self._used_filter_fallback = False
        else:
            temp: list[str] = []
            dirs = MarkedFiles.mark_target_dirs
            self._used_filter_fallback = False

            # Pass 1: exact basename match
            for d in dirs:
                basename = os.path.basename(os.path.normpath(d))
                if basename.lower() == ft:
                    temp.append(d)

            # Pass 2: basename starts-with
            for d in dirs:
                if d not in temp:
                    basename = os.path.basename(os.path.normpath(d))
                    if basename.lower().startswith(ft):
                        temp.append(d)

            # Pass 3: parent directory starts-with
            for d in dirs:
                if d not in temp:
                    dirname = os.path.basename(
                        os.path.dirname(os.path.normpath(d))
                    )
                    if dirname and dirname.lower().startswith(ft):
                        temp.append(d)

            # Pass 4: substring match in basename
            for d in dirs:
                if d not in temp:
                    basename = os.path.basename(os.path.normpath(d))
                    if basename and (
                        f" {ft}" in basename.lower()
                        or f"_{ft}" in basename.lower()
                    ):
                        temp.append(d)

            # Lost-first-letter fallback:
            # if no normal matches, compare against previous submitted filter keys.
            if not temp:
                for previous_filter, resolved_dir in MarkedFileMover._session_filter_target_map.items():
                    if len(previous_filter) > 1 and previous_filter[1:] == ft:
                        if resolved_dir in dirs and resolved_dir not in temp:
                            temp.append(resolved_dir)
                if temp:
                    self._used_filter_fallback = True

            self._filtered_target_dirs = temp

        if self._is_gui:
            self._rebuild_directory_rows()

    # ==================================================================
    # Enter-key action dispatch
    # ==================================================================
    def _do_action(
        self, *, shift: bool = False, ctrl: bool = False, alt: bool = False
    ) -> None:
        move_func = Utils.copy_file if shift else Utils.move_file

        if alt:
            penultimate_action = FileAction.get_history_action(start_index=1)
            if penultimate_action is not None and os.path.isdir(penultimate_action.target):
                self._move_marks_to_dir(target_dir=penultimate_action.target, move_func=move_func)
        elif len(self._filtered_target_dirs) == 0 or ctrl:
            self._handle_target_directory(move_func=move_func)
        else:
            if (
                len(self._filtered_target_dirs) == 1
                or self._filter_text.strip() != ""
                or self._is_sorted_by_embedding
            ):
                target_dir = self._filtered_target_dirs[0]
            else:
                target_dir = MarkedFiles.last_set_target_dir
            self._move_marks_to_dir(target_dir=target_dir, move_func=move_func)

    # ==================================================================
    # Key / mouse event handling
    # ==================================================================
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        modifiers = event.modifiers()

        # Return / Enter -> do action
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._do_action(
                shift=bool(modifiers & Qt.ShiftModifier),
                ctrl=bool(modifiers & Qt.ControlModifier),
                alt=bool(modifiers & Qt.AltModifier),
            )
            return

        # Page Up / Down
        if key == Qt.Key_PageUp:
            self._page_up()
            return
        if key == Qt.Key_PageDown:
            self._page_down()
            return

        # Up / Down arrows -> roll list
        if key == Qt.Key_Down and self._filtered_target_dirs:
            self._filtered_target_dirs = (
                self._filtered_target_dirs[1:]
                + [self._filtered_target_dirs[0]]
            )
            if self._is_gui:
                self._rebuild_directory_rows()
            return
        if key == Qt.Key_Up and self._filtered_target_dirs:
            self._filtered_target_dirs = (
                [self._filtered_target_dirs[-1]]
                + self._filtered_target_dirs[:-1]
            )
            if self._is_gui:
                self._rebuild_directory_rows()
            return

        # Backspace -> trim filter
        if key == Qt.Key_Backspace:
            if self._filter_text:
                self._filter_text = self._filter_text[:-1]
                self._apply_filter()
            return

        # Ignore modifier-only or Ctrl/Alt combos (let shortcuts handle)
        if modifiers & (Qt.ControlModifier | Qt.AltModifier):
            super().keyPressEvent(event)
            return
        if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
            super().keyPressEvent(event)
            return

        # Printable text -> filter
        text = event.text()
        if not text:
            # Some platform/Qt configurations return empty text for Shift+punctuation
            # (e.g. Shift+Minus → underscore on Windows).  Qt key codes for printable
            # ASCII match their ord() values, so chr(key) recovers the character.
            candidate = chr(key) if 32 <= key <= 126 else ""
            if candidate.isprintable():
                text = candidate
        if text and text.isprintable() and text not in ("/", "\\"):
            self._filter_text += text
            self._apply_filter()
            return

        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton:
            self._delete_marked_files()
            return
        if event.button() == Qt.RightButton:
            mods = event.modifiers()
            self._do_action_test_is_in_directory(
                ctrl=bool(mods & Qt.ControlModifier),
                alt=bool(mods & Qt.AltModifier),
                shift=bool(mods & Qt.ShiftModifier),
            )
            return
        super().mousePressEvent(event)

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def close_windows(self) -> None:
        self.close()

    def reject(self) -> None:  # noqa: N802
        if MarkedFileMover._current_window is self:
            MarkedFileMover._current_window = None
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if MarkedFileMover._current_window is self:
            MarkedFileMover._current_window = None
        if (
            self._single_image is not None
            and len(MarkedFiles.file_marks) == 1
        ):
            MarkedFiles.file_marks.clear()
            self._app_actions.toast(_("Cleared marked file"))
        super().closeEvent(event)


# ======================================================================
# Helpers
# ======================================================================
def _clear_layout(layout) -> None:
    """Recursively remove all items from a QLayout."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        sub = item.layout()
        if sub is not None:
            _clear_layout(sub)

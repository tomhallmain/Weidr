"""
FileActionsWindow -- file action history and setup (PySide6).

Self-contained: ``FileAction`` integration, module-level setup helpers,
persistence, static state management, and Qt UI.

Uses a **Load More** pattern for the action history list so only a small
initial page of rows is created on open, keeping widget count low.
Statistics always reflect the full (or today-filtered) collection.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from files.file_action import FileAction
from lib.multi_display_qt import SmartWindow
from ui.app_style import AppStyle
from utils.app_actions import AppActions
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.constants import FileActionKind
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

logger = get_logger("file_actions_window_qt")



# ======================================================================
# FileActionsWindow
# ======================================================================
class FileActionsWindow(SmartWindow):
    """
    Window displaying completed file actions with statistics,
    filter-by-typing, and a paginated action history ("Load More").
    """

    _instance: Optional[FileActionsWindow] = None

    # -- pagination -------------------------------------------------------
    INITIAL_PAGE_SIZE: int = 20
    PAGE_SIZE: int = 20

    # -- layout -----------------------------------------------------------
    COL_0_WIDTH: int = 600
    MAX_DISPLAY_DIRS: int = 6

    # ==================================================================
    # Construction
    # ==================================================================
    def __init__(
        self,
        app_master: QWidget,
        app_actions: AppActions,
        view_image_callback: Callable,
        move_marks_callback: Callable,
        geometry: str = "900x1400",
    ) -> None:
        super().__init__(
            persistent_parent=None,
            position_parent=app_master,
            title=_("File Actions"),
            geometry=geometry,
            respect_title_bar=True,
        )
        FileActionsWindow._instance = self
        self._app_master = app_master
        self._app_actions = app_actions
        self._view_image_callback = view_image_callback
        self._move_marks_callback = move_marks_callback
        self._filter_text: str = ""
        self._filtered_history: list[FileAction] = FileAction.action_history[:]
        self._show_today_only: bool = False
        self._visible_count: int = self.INITIAL_PAGE_SIZE
        self._action_type_filter: FileActionKind | None = None  # None means "all"
        self._initiator_filter: bool | None = None  # None = any, False = user, True = auto

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # -- header row ---------------------------------------------------
        header = QHBoxLayout()
        title_lbl = QLabel(_("File Action History"))
        title_lbl.setStyleSheet(
            f"font-size: 12pt; font-weight: bold; color: {AppStyle.FG_COLOR};"
        )
        header.addWidget(title_lbl)
        header.addStretch()

        self._type_combo = QComboBox()
        for label, kind in [
            (_("All"), None),
            (_("Move"), FileActionKind.MOVE),
            (_("Copy"), FileActionKind.COPY),
            (_("Delete"), FileActionKind.DELETE),
        ]:
            self._type_combo.addItem(label, kind)
        self._type_combo.currentIndexChanged.connect(self._on_type_filter_changed)
        header.addWidget(self._type_combo)

        self._initiator_combo = QComboBox()
        for label, value in [
            (_("Any"), None),
            (_("User"), False),
            (_("Auto"), True),
        ]:
            self._initiator_combo.addItem(label, value)
        self._initiator_combo.currentIndexChanged.connect(self._on_initiator_filter_changed)
        header.addWidget(self._initiator_combo)

        search_btn = QPushButton(_("Search Media"))
        # clicked emits bool(checked); ignore so it is never passed as media_path.
        search_btn.clicked.connect(lambda _checked=False: self._search_for_active_media())
        header.addWidget(search_btn)

        clear_btn = QPushButton(_("Clear History"))
        clear_btn.clicked.connect(self._clear_action_history)
        header.addWidget(clear_btn)
        root.addLayout(header)

        # -- scroll area --------------------------------------------------
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {AppStyle.BG_COLOR}; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(4)
        self._scroll.setWidget(self._scroll_content)
        root.addWidget(self._scroll, 1)

        # -- filter indicator (hidden by default) -------------------------
        self._filter_label = QLabel("")
        self._filter_label.setStyleSheet("color: orange; font-style: italic;")
        self._filter_label.setVisible(False)
        root.addWidget(self._filter_label)

        # -- initial content build ----------------------------------------
        self._rebuild_content()

        # -- keyboard shortcuts -------------------------------------------
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence("Shift+Escape"), self).activated.connect(self.close)
        QShortcut(QKeySequence("Shift+A"), self).activated.connect(
            self._search_for_active_media
        )

        # Focus the scroll area so key events are captured
        QTimer.singleShot(1, self._scroll.setFocus)

    # ==================================================================
    # Content rebuild
    # ==================================================================
    def _rebuild_content(self) -> None:
        """Clear and rebuild the entire scrollable content area."""
        _clear_layout(self._scroll_layout)

        # Statistics section
        stats_widget = self._build_statistics()
        if stats_widget is not None:
            self._scroll_layout.addWidget(stats_widget)

            # Separator
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            self._scroll_layout.addWidget(sep)

        # Action rows (paginated)
        self._build_action_rows()

        # Empty-state hint when filters exclude everything
        if not self._filtered_history and FileAction.action_history:
            kind_labels = {
                FileActionKind.MOVE: _("moves"),
                FileActionKind.COPY: _("copies"),
                FileActionKind.DELETE: _("deletes"),
            }
            if self._action_type_filter is not None and not self._filter_text and not self._show_today_only:
                msg = _("No {0} in history.").format(kind_labels[self._action_type_filter])
            else:
                msg = _("No actions match the current filter.")
            hint = QLabel(msg)
            hint.setStyleSheet(
                f"color: {AppStyle.FG_COLOR}; font-style: italic; padding: 8px;"
            )
            hint.setAlignment(Qt.AlignCenter)
            self._scroll_layout.addWidget(hint)

        # "Load More" button
        remaining = len(self._filtered_history) - self._visible_count
        if remaining > 0:
            load_more = QPushButton(
                _("Load More ({0} remaining)").format(remaining)
            )
            load_more.setStyleSheet(
                f"color: {AppStyle.FG_COLOR}; padding: 6px 12px; margin-top: 4px;"
            )
            load_more.clicked.connect(self._load_more)
            self._scroll_layout.addWidget(load_more, 0, Qt.AlignCenter)

        self._scroll_layout.addStretch()

    # ------------------------------------------------------------------
    # Statistics section
    # ------------------------------------------------------------------
    def _build_statistics(self) -> QFrame | None:
        stats = FileAction.get_action_statistics(today_only=self._show_today_only, kind=self._action_type_filter, auto=self._initiator_filter)
        if not stats:
            return None

        # When a single type is active, show only that column (Total would equal
        # it, so it is omitted). When showing all, show the full breakdown.
        if self._action_type_filter is None:
            col_defs = [
                (_("Moved"), "moved"),
                (_("Copied"), "copied"),
                (_("Deleted"), "deleted"),
                (_("Total"), "total"),
            ]
        else:
            col_defs = [
                {
                    FileActionKind.MOVE: (_("Moved"), "moved"),
                    FileActionKind.COPY: (_("Copied"), "copied"),
                    FileActionKind.DELETE: (_("Deleted"), "deleted"),
                }[self._action_type_filter]
            ]

        sorted_stats = sorted(
            stats.items(), key=lambda x: x[1]["total"], reverse=True
        )
        display_stats = sorted_stats[: self.MAX_DISPLAY_DIRS]
        remaining_stats = sorted_stats[self.MAX_DISPLAY_DIRS :]

        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {AppStyle.BG_COLOR}; border: none; }}"
        )
        grid = QGridLayout(frame)
        grid.setContentsMargins(5, 5, 5, 5)
        grid.setSpacing(2)

        # col 0 = directory, cols 1..n = data; toggle shares the last data col
        total_cols = 1 + len(col_defs)
        grid.setColumnStretch(0, 3)
        for i in range(1, total_cols):
            grid.setColumnStretch(i, 1)

        # Title + toggle button (row 0)
        kind_stat_labels = {
            FileActionKind.MOVE: _("Move Statistics"),
            FileActionKind.COPY: _("Copy Statistics"),
            FileActionKind.DELETE: _("Delete Statistics"),
        }
        if self._show_today_only:
            title_text = _("Today's File Actions")
        elif self._action_type_filter is not None:
            title_text = kind_stat_labels[self._action_type_filter]
        else:
            title_text = _("File Action Statistics")
        title = QLabel(title_text)
        title.setStyleSheet(f"font-weight: bold; color: {AppStyle.FG_COLOR};")
        title.setAlignment(Qt.AlignCenter)
        grid.addWidget(title, 0, 0, 1, total_cols - 1)  # span all but last data col

        toggle_btn = QPushButton(
            _("All Time") if self._show_today_only else _("Today Only")
        )
        toggle_btn.clicked.connect(self._toggle_statistics_view)
        grid.addWidget(toggle_btn, 0, total_cols - 1, Qt.AlignRight)  # shares last data col

        # Headers (row 1)
        dir_hdr = QLabel(_("Target Directory"))
        dir_hdr.setStyleSheet(f"font-weight: bold; color: {AppStyle.FG_COLOR};")
        grid.addWidget(dir_hdr, 1, 0, Qt.AlignLeft)
        for col_offset, (header_text, _key) in enumerate(col_defs):
            lbl = QLabel(header_text)
            lbl.setStyleSheet(f"font-weight: bold; color: {AppStyle.FG_COLOR};")
            grid.addWidget(lbl, 1, 1 + col_offset, Qt.AlignRight)

        # Data rows
        for i, (target_dir, counts) in enumerate(display_stats):
            row = i + 2
            target_display = Utils.get_relative_dirpath(target_dir, levels=2)
            if len(target_display) > 30:
                target_display = Utils.get_centrally_truncated_string(target_display, 30)
            self._add_stat_cell(grid, target_display, row, 0, Qt.AlignLeft)
            for col_offset, (_header, key) in enumerate(col_defs):
                self._add_stat_cell(grid, str(counts[key]), row, 1 + col_offset, Qt.AlignRight)

        # "etc." row
        if remaining_stats:
            row = len(display_stats) + 2
            self._add_stat_cell(
                grid, _("... and {0} more").format(len(remaining_stats)), row, 0, Qt.AlignLeft
            )
            for col_offset, (_header, key) in enumerate(col_defs):
                rem_val = sum(c[key] for _unused, c in remaining_stats)
                self._add_stat_cell(grid, str(rem_val), row, 1 + col_offset, Qt.AlignRight)

        return frame

    @staticmethod
    def _add_stat_cell(grid: QGridLayout, text: str, row: int, col: int, align) -> None:
        lbl = QLabel("-" if text == "0" else text)
        lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(lbl, row, col, align)

    # ------------------------------------------------------------------
    # Action history rows (paginated)
    # ------------------------------------------------------------------
    def _build_action_rows(self) -> None:
        visible = self._filtered_history[: self._visible_count]
        last_action: FileAction | None = None
        last_target: str | None = None
        current_group_layout: QVBoxLayout | None = None

        for action in visible:
            need_header = (
                action != last_action
                or len(action.new_files) != 1
                or (last_action is not None and len(last_action.new_files) != 1)
            )

            is_delete = action.is_delete_action()
            file_count = len(action.relevant_files)

            if need_header:
                # Insert a separator only when the target directory changes
                if last_target is not None and action.target != last_target:
                    sep = QFrame()
                    sep.setFixedHeight(1)
                    sep.setStyleSheet(
                        f"background: {AppStyle.BORDER_COLOR};"
                    )
                    self._scroll_layout.addWidget(sep)

                group = QFrame()
                group.setStyleSheet(
                    f"QFrame {{ background: {AppStyle.BG_COLOR}; "
                    f"padding: 2px 0; }}"
                )
                current_group_layout = QVBoxLayout(group)
                current_group_layout.setContentsMargins(4, 4, 4, 4)
                current_group_layout.setSpacing(2)
                self._scroll_layout.addWidget(group)

                # Header row
                header = QHBoxLayout()

                action_text = Utils.get_relative_dirpath(action.target, levels=2)
                if file_count > 1:
                    action_text += _(" ({0} files)").format(file_count)
                if action.auto:
                    action_text += " " + _("(auto)")

                dir_lbl = QLabel(action_text)
                dir_lbl.setWordWrap(True)
                dir_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
                header.addWidget(dir_lbl, 1)

                if is_delete:
                    type_lbl = QLabel(_("Delete"))
                elif action.is_move_action():
                    type_lbl = QLabel(_("Move"))
                else:
                    type_lbl = QLabel(_("Copy"))
                type_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
                header.addWidget(type_lbl)

                # Only show header-level Undo/Modify when the group
                # has multiple files; for single-file groups the
                # file row already provides identical buttons.
                if not is_delete and len(action.new_files) > 1:
                    undo_btn = QPushButton(_("Undo"))
                    undo_btn.clicked.connect(
                        lambda _=False, a=action: self._undo(a)
                    )
                    header.addWidget(undo_btn)

                    modify_btn = QPushButton(_("Modify"))
                    modify_btn.clicked.connect(
                        lambda _=False, a=action: self._modify(a)
                    )
                    header.addWidget(modify_btn)

                current_group_layout.addLayout(header)
                last_target = action.target

            last_action = action

            # File rows
            for filename in action.relevant_files:
                file_row = QHBoxLayout()

                display_name = os.path.basename(filename)
                if len(display_name) > 50:
                    display_name = Utils.get_centrally_truncated_string(
                        display_name, 50
                    )

                name_lbl = QLabel(display_name)
                name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
                name_lbl.setWordWrap(True)
                file_row.addWidget(name_lbl, 1)

                if not is_delete:
                    view_btn = QPushButton(_("View"))
                    view_btn.clicked.connect(
                        lambda _=False, p=filename: self._view(p)
                    )
                    file_row.addWidget(view_btn)

                copy_btn = QPushButton(_("Copy Filename"))
                copy_btn.clicked.connect(
                    lambda _=False, p=filename: self._copy_filename(p)
                )
                file_row.addWidget(copy_btn)

                if not is_delete:
                    undo_file_btn = QPushButton(_("Undo"))
                    undo_file_btn.clicked.connect(
                        lambda _=False, a=action, p=filename: self._undo(
                            a, specific_image=p
                        )
                    )
                    file_row.addWidget(undo_file_btn)

                    modify_file_btn = QPushButton(_("Modify"))
                    modify_file_btn.clicked.connect(
                        lambda _=False, p=filename: self._modify(p)
                    )
                    file_row.addWidget(modify_file_btn)

                if current_group_layout is not None:
                    current_group_layout.addLayout(file_row)

    # ==================================================================
    # Actions
    # ==================================================================
    def _view(self, media_path: str) -> None:
        if not os.path.isfile(media_path):
            self._app_actions.warn(
                _("File not found: ") + os.path.basename(media_path)
            )
            return
        try:
            self._view_image_callback(
                master=self._app_master,
                media_path=media_path,
                app_actions=self._app_actions,
            )
        except Exception as e:
            self._app_actions.toast(
                _("Error opening media: ") + str(e)
            )

    def _undo(self, action: FileAction, specific_image: str | None = None) -> None:
        # If this was an automated prevalidation action, confirm that the user
        # wants to create an override so it won't fire on these files again.
        override_paths: list[str] = []
        if action.auto:
            if specific_image is not None:
                try:
                    idx = action.new_files.index(specific_image)
                    override_paths = [action.original_marks[idx]]
                except (ValueError, IndexError):
                    override_paths = []
            else:
                override_paths = list(action.original_marks)

            confirmed = self._app_actions.alert(
                _("Override Prevalidation?"),
                _(
                    "This action was performed automatically by a prevalidation. "
                    "Undoing it will create a prevalidation override for this file.\n\n"
                    "Proceed?"
                ),
                kind="askyesno",
                master=self,
            )
            if not confirmed:
                return

        invoked_move_callback = False
        if specific_image is not None:
            if not os.path.isfile(specific_image):
                error_text = _("Image does not exist: ") + specific_image
                self._app_actions.alert(
                    _("File Action Error"), error_text, master=self
                )
                raise Exception(error_text)
            if action.is_move_action():
                original_directory = action.get_original_directory()
                self._move_marks_callback(
                    self._app_actions,
                    target_dir=original_directory,
                    move_func=Utils.move_file,
                    files=[specific_image],
                    single_image=True,
                )
                invoked_move_callback = True
            else:
                os.remove(specific_image)
        else:
            if not action.any_new_files_exist():
                error_text = _("Images not found")
                self._app_actions.alert(
                    _("File Action Error"), error_text, master=self
                )
                raise Exception(error_text)
            if action.is_move_action():
                original_directory = action.get_original_directory()
                self._move_marks_callback(
                    self._app_actions,
                    target_dir=original_directory,
                    move_func=Utils.move_file,
                    files=action.new_files,
                    single_image=(len(action.new_files) == 1),
                )
                invoked_move_callback = True
            else:
                action.remove_new_files()

        if override_paths:
            from compare.classifier_actions_manager import ClassifierActionsManager
            for path in override_paths:
                ClassifierActionsManager.record_prevalidation_override(path)

        if invoked_move_callback:
            # MarkedFiles.move_marks_to_dir_static() can refocus the main window.
            # Restore this dialog's keyboard focus for Esc/typing workflows.
            QTimer.singleShot(0, self._restore_dialog_focus)

    def _restore_dialog_focus(self) -> None:
        if not self.isVisible():
            return
        self.raise_()
        self.activateWindow()
        self._scroll.setFocus()

    def _modify(self, image_path_or_action) -> None:
        # TODO: implement this (matches original stub)
        pass

    def _copy_filename(self, filepath: str) -> None:
        if not filepath:
            return
        filename = os.path.basename(filepath)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(filename)
        if hasattr(self._app_actions, "toast"):
            self._app_actions.toast(f"Copied filename: {filename}")

    # ==================================================================
    # Type filter
    # ==================================================================
    def _on_type_filter_changed(self, _index: int) -> None:
        self._action_type_filter = self._type_combo.currentData()
        self._update_filter_label()
        self._refresh_filtered_history()

    def _filter_by_action_type(self, actions: list[FileAction]) -> list[FileAction]:
        kind = self._action_type_filter
        if kind is None:
            return actions
        return [a for a in actions if a.action_kind() == kind]

    # ==================================================================
    # Initiator filter
    # ==================================================================
    def _on_initiator_filter_changed(self, _index: int) -> None:
        self._initiator_filter = self._initiator_combo.currentData()
        self._update_filter_label()
        self._refresh_filtered_history()

    def _filter_by_initiator(self, actions: list[FileAction]) -> list[FileAction]:
        if self._initiator_filter is None:
            return actions
        return [a for a in actions if a.auto == self._initiator_filter]

    # ==================================================================
    # Unified filter pipeline
    # ==================================================================
    def _refresh_filtered_history(self) -> None:
        actions = FileAction.action_history[:]
        actions = self._filter_by_action_type(actions)
        actions = self._filter_by_initiator(actions)
        if self._show_today_only:
            actions = [a for a in actions if a.is_today()]
        ft = self._filter_text.strip().lower()
        if ft:
            actions = self._filter_by_directory_text(actions, ft)
        self._filtered_history = actions
        self._visible_count = self.INITIAL_PAGE_SIZE
        self._rebuild_content()

    def _filter_by_directory_text(
        self, actions: list[FileAction], ft: str
    ) -> list[FileAction]:
        temp: list[FileAction] = []
        # Pass 1: directory basename exact match
        for action in actions:
            basename = os.path.basename(os.path.normpath(action.target))
            if basename.lower() == ft:
                temp.append(action)
        # Pass 2: directory basename starts-with
        for action in actions:
            if not FileAction._is_matching_action_in_list(temp, action):
                basename = os.path.basename(os.path.normpath(action.target))
                if basename.lower().startswith(ft):
                    temp.append(action)
        # Pass 3: parent directory starts-with
        for action in actions:
            if not FileAction._is_matching_action_in_list(temp, action):
                dirname = os.path.basename(
                    os.path.dirname(os.path.normpath(action.target))
                )
                if dirname and dirname.lower().startswith(ft):
                    temp.append(action)
        # Pass 4: substring match in basename
        for action in actions:
            if not FileAction._is_matching_action_in_list(temp, action):
                basename = os.path.basename(os.path.normpath(action.target))
                if basename and (
                    f" {ft}" in basename.lower()
                    or f"_{ft}" in basename.lower()
                ):
                    temp.append(action)
        return temp

    def _update_filter_label(self) -> None:
        parts = []
        if self._filter_text:
            parts.append(self._filter_text)
        if self._action_type_filter is not None:
            kind_labels = {
                FileActionKind.MOVE: _("moves only"),
                FileActionKind.COPY: _("copies only"),
                FileActionKind.DELETE: _("deletes only"),
            }
            parts.append(kind_labels[self._action_type_filter])
        if self._initiator_filter is not None:
            parts.append(_("auto only") if self._initiator_filter else _("user only"))
        if parts:
            self._filter_label.setText(_("Filter: {}").format(" — ".join(parts)))
            self._filter_label.setVisible(True)
        else:
            self._filter_label.setVisible(False)

    # ==================================================================
    # Load More / Clear / Toggle
    # ==================================================================
    def _load_more(self) -> None:
        self._visible_count += self.PAGE_SIZE
        self._rebuild_content()

    def _clear_action_history(self) -> None:
        FileAction.action_history.clear()
        self._filtered_history.clear()
        self._action_type_filter = None
        self._type_combo.setCurrentIndex(0)
        self._initiator_filter = None
        self._initiator_combo.setCurrentIndex(0)
        self._filter_text = ""
        self._update_filter_label()
        self._visible_count = self.INITIAL_PAGE_SIZE
        self._rebuild_content()

    def _toggle_statistics_view(self) -> None:
        self._show_today_only = not self._show_today_only
        self._refresh_filtered_history()

    # ==================================================================
    # Search for active media
    # ==================================================================
    def _search_for_active_media(self, media_path: str | None = None) -> None:
        if media_path is not None and not isinstance(media_path, str):
            media_path = None
        if media_path is None:
            media_path = self._app_actions.get_active_media_filepath()
            if media_path is None:
                raise Exception("No active media")

        media_path = os.path.normpath(media_path)
        search_basename = os.path.basename(media_path).lower()
        basename_no_ext = os.path.splitext(search_basename)[0].lower()

        temp: list[FileAction] = []
        # Pass 1: exact path match
        for action in FileAction.action_history:
            for f in action.relevant_files:
                if f == media_path:
                    temp.append(action)
                    break
        # Pass 2: basename match
        for action in FileAction.action_history:
            if action not in temp:
                for f in action.relevant_files:
                    if os.path.basename(os.path.normpath(f)).lower() == search_basename:
                        temp.append(action)
                        break
        # Pass 3: basename prefix match
        for action in FileAction.action_history:
            if action not in temp:
                for f in action.relevant_files:
                    if os.path.basename(os.path.normpath(f)).lower().startswith(
                        basename_no_ext
                    ):
                        temp.append(action)
                        break

        self._filtered_history = self._filter_by_action_type(temp)
        self._visible_count = self.INITIAL_PAGE_SIZE
        self._rebuild_content()

    # ==================================================================
    # Filter by typing
    # ==================================================================
    def _apply_filter(self) -> None:
        self._update_filter_label()
        self._refresh_filtered_history()

    # ==================================================================
    # Enter-key action dispatch
    # ==================================================================
    def _do_action(self, *, shift: bool, ctrl: bool, alt: bool) -> None:
        """Act on the first filtered action based on modifier keys."""
        if not self._filtered_history:
            return
        if len(self._filtered_history) != 1 and not self._filter_text.strip():
            return

        action = self._filtered_history[0]
        if alt:
            self._undo(action)
        elif ctrl:
            self._modify(action)
        else:
            if not action.is_delete_action() and action.new_files:
                self._view(action.new_files[0])

    # ==================================================================
    # Key event handling
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

        # Backspace -> trim filter
        if key == Qt.Key_Backspace:
            if self._filter_text:
                self._filter_text = self._filter_text[:-1]
                self._apply_filter()
            return

        # Ignore when Ctrl or Alt held (let shortcuts handle)
        if modifiers & (Qt.ControlModifier | Qt.AltModifier):
            super().keyPressEvent(event)
            return

        # Ignore Shift alone (but allow shift+char for uppercase)
        if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
            super().keyPressEvent(event)
            return

        # Printable text -> append to filter
        text = event.text()
        if text and text.isprintable():
            self._filter_text += text
            self._apply_filter()
            return

        super().keyPressEvent(event)

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def close_windows(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        FileActionsWindow._instance = None
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

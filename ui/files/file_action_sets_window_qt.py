from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMenu, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from files.file_action import FileAction
from files.file_action_set import ActionSet, ActionStep, FileActionSets
from lib.multi_display_qt import SmartDialog, SmartWindow
from ui.app_style import AppStyle
from ui.auth.password_utils import require_password
from utils.app_actions import AppActions
from utils.constants import ProtectedActions
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils
logger = get_logger("file_action_sets_window_qt")


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        sub = item.layout()
        if sub is not None:
            _clear_layout(sub)


# ======================================================================
# Save-preset dialog
# ======================================================================
class _SavePresetDialog(SmartDialog):
    """
    Dialog for naming a preset before saving. Lists existing presets so
    the user can click one to pre-fill the name field (overwrite flow).
    """

    def __init__(self, parent: QWidget, initial_name: str = "") -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Save Preset"),
            geometry="480x360",
            center=True,
        )
        self.chosen_name: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        name_lbl = QLabel(_("Preset name:"))
        name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(name_lbl)

        self._name_edit = QLineEdit(initial_name)
        self._name_edit.setPlaceholderText(_("Enter a name…"))
        self._name_edit.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._name_edit.returnPressed.connect(self._on_save)
        root.addWidget(self._name_edit)

        if FileActionSets.action_sets:
            overwrite_lbl = QLabel(_("Or click an existing preset to overwrite it:"))
            overwrite_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-style: italic;")
            root.addWidget(overwrite_lbl)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet(
                f"QScrollArea {{ border: 1px solid {AppStyle.BORDER_COLOR}; background: {AppStyle.BG_COLOR}; }}"
            )
            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(4, 4, 4, 4)
            content_layout.setSpacing(3)
            for action_set in FileActionSets.action_sets:
                content_layout.addWidget(self._make_preset_row(action_set))
            content_layout.addStretch()
            scroll.setWidget(content)
            root.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton(_("Save"))
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        root.addLayout(btn_row)

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.reject)
        self._name_edit.setFocus()

    def _make_preset_row(self, action_set: ActionSet) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {AppStyle.BG_COLOR}; border: 1px solid {AppStyle.BORDER_COLOR}; "
            f"border-radius: 3px; }}"
        )
        frame.setCursor(Qt.PointingHandCursor)
        row = QHBoxLayout(frame)
        row.setContentsMargins(6, 3, 6, 3)
        row.setSpacing(6)

        name_lbl = QLabel(f"<b>{action_set.name}</b>")
        name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        row.addWidget(name_lbl)

        summary_lbl = QLabel(action_set.summary())
        summary_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 9pt;")
        summary_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(summary_lbl, 1)

        frame.mousePressEvent = lambda _evt, n=action_set.name: self._prefill(n)
        return frame

    def _prefill(self, name: str) -> None:
        self._name_edit.setText(name)
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _on_save(self) -> None:
        name = self._name_edit.text().strip()
        if name:
            self.chosen_name = name
            self.accept()


# ======================================================================
# Main window
# ======================================================================
class FileActionSetsWindow(SmartWindow):
    _instance: Optional[FileActionSetsWindow] = None

    def __init__(
        self,
        app_master: QWidget,
        app_actions: AppActions,
        move_marks_callback: Callable,
        geometry: str = "700x800",
    ) -> None:
        super().__init__(
            persistent_parent=None,
            position_parent=app_master,
            title=_("File Action Sets"),
            geometry=geometry,
            respect_title_bar=True,
        )
        FileActionSetsWindow._instance = self
        self._app_master = app_master
        self._app_actions = app_actions
        self._move_marks_callback = move_marks_callback
        self._active_preset_index: Optional[int] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # -- actions header -----------------------------------------------
        actions_header = QHBoxLayout()
        actions_lbl = QLabel(_("Actions"))
        actions_lbl.setStyleSheet(f"font-size: 11pt; font-weight: bold; color: {AppStyle.FG_COLOR};")
        actions_header.addWidget(actions_lbl)
        actions_header.addStretch()

        add_hotkeys_btn = QPushButton(_("Add from Hotkeys"))
        add_hotkeys_btn.clicked.connect(lambda: self._add_from_hotkeys(add_hotkeys_btn))
        actions_header.addWidget(add_hotkeys_btn)

        add_recent_btn = QPushButton(_("Add from Recent ▾"))
        add_recent_btn.clicked.connect(lambda: self._show_recent_menu(add_recent_btn))
        actions_header.addWidget(add_recent_btn)

        root.addLayout(actions_header)

        # -- actions scroll area ------------------------------------------
        self._actions_scroll = QScrollArea()
        self._actions_scroll.setWidgetResizable(True)
        self._actions_scroll.setMinimumHeight(130)
        self._actions_scroll.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {AppStyle.BORDER_COLOR}; background: {AppStyle.BG_COLOR}; }}"
        )
        self._actions_content = QWidget()
        self._actions_layout = QVBoxLayout(self._actions_content)
        self._actions_layout.setContentsMargins(4, 4, 4, 4)
        self._actions_layout.setSpacing(4)
        self._actions_scroll.setWidget(self._actions_content)
        root.addWidget(self._actions_scroll, 3)

        # -- action buttons row -------------------------------------------
        action_row = QHBoxLayout()

        save_preset_btn = QPushButton(_("Save as Preset…"))
        save_preset_btn.clicked.connect(self._save_as_preset)
        action_row.addWidget(save_preset_btn)

        action_row.addStretch()

        execute_btn = QPushButton(_("Execute"))
        execute_btn.setStyleSheet(
            f"QPushButton {{ font-weight: bold; padding: 6px 20px; color: {AppStyle.FG_COLOR}; }}"
        )
        execute_btn.clicked.connect(self._execute)
        action_row.addWidget(execute_btn)

        root.addLayout(action_row)

        # -- separator ----------------------------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {AppStyle.BORDER_COLOR};")
        root.addWidget(sep)

        # -- presets section ----------------------------------------------
        presets_lbl = QLabel(_("Presets"))
        presets_lbl.setStyleSheet(f"font-size: 11pt; font-weight: bold; color: {AppStyle.FG_COLOR};")
        root.addWidget(presets_lbl)

        self._presets_scroll = QScrollArea()
        self._presets_scroll.setWidgetResizable(True)
        self._presets_scroll.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {AppStyle.BORDER_COLOR}; background: {AppStyle.BG_COLOR}; }}"
        )
        self._presets_content = QWidget()
        self._presets_layout = QVBoxLayout(self._presets_content)
        self._presets_layout.setContentsMargins(4, 4, 4, 4)
        self._presets_layout.setSpacing(4)
        self._presets_scroll.setWidget(self._presets_content)
        root.addWidget(self._presets_scroll, 1)

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

        self._rebuild_actions()
        self._rebuild_presets()

    # ==================================================================
    # Actions (pool) UI
    # ==================================================================
    def _rebuild_actions(self) -> None:
        _clear_layout(self._actions_layout)
        if not FileActionSets.all_actions:
            hint = QLabel(_("Add actions from Hotkeys or Recent to get started."))
            hint.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-style: italic;")
            self._actions_layout.addWidget(hint)
        for i, action in enumerate(FileActionSets.all_actions):
            selected = i in FileActionSets.selected_indices
            self._actions_layout.addWidget(self._make_action_card(i, action, selected))
        self._actions_layout.addStretch()

    def _make_action_card(self, index: int, action: ActionStep, selected: bool) -> QFrame:
        border = AppStyle.FG_COLOR if selected else AppStyle.BORDER_COLOR
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {AppStyle.BG_COLOR}; border: 1px solid {border}; "
            f"border-radius: 3px; padding: 2px; }}"
        )
        card.setCursor(Qt.PointingHandCursor)

        row = QHBoxLayout(card)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)

        type_lbl = QLabel(_("Move") if action.is_move() else _("Copy"))
        type_lbl.setFixedWidth(40)
        type_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-weight: bold;")
        row.addWidget(type_lbl)

        dir_display = Utils.get_relative_dirpath(action.target, levels=2) if action.target else _("(no directory)")
        dir_lbl = QLabel(dir_display)
        dir_lbl.setToolTip(action.target)
        dir_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        dir_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(dir_lbl, 1)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(28)
        del_btn.clicked.connect(lambda _=False, i=index: self._remove_from_pool(i))
        row.addWidget(del_btn)

        card.mousePressEvent = lambda evt, i=index: (
            self._toggle_selection(i) if evt.button() == Qt.LeftButton else None
        )
        return card

    def _toggle_selection(self, index: int) -> None:
        action = FileActionSets.all_actions[index]
        if index in FileActionSets.selected_indices:
            FileActionSets.selected_indices.remove(index)
        else:
            # At most one move action may be selected at a time
            if action.is_move():
                existing_move = FileActionSets.selected_move_index()
                if existing_move != -1:
                    FileActionSets.selected_indices.remove(existing_move)
            FileActionSets.selected_indices.append(index)
        FileActionSets.store()
        self._rebuild_actions()

    def _remove_from_pool(self, index: int) -> None:
        del FileActionSets.all_actions[index]
        # Remap selected indices: drop the removed index, shift higher ones down
        FileActionSets.selected_indices = [
            i - 1 if i > index else i
            for i in FileActionSets.selected_indices
            if i != index
        ]
        FileActionSets.store()
        self._rebuild_actions()

    # ==================================================================
    # Adding to pool
    # ==================================================================
    def _add_to_pool(self, action: str, target: str) -> None:
        FileActionSets.add_to_pool(action, target)
        FileActionSets.store()
        self._rebuild_actions()

    def _add_from_hotkeys(self, btn: QPushButton) -> None:
        hotkeys = FileAction.hotkey_actions
        permanent = FileAction.permanent_action
        if not hotkeys and permanent is None:
            self._app_actions.warn(_("No hotkey actions configured."))
            return
        menu = QMenu(self)
        if permanent is not None:
            action_label = _("Move") if permanent.is_move_action() else _("Copy")
            label = f"Ctrl+T: {action_label} → {Utils.get_relative_dirpath(permanent.target, levels=2)}"
            menu.addAction(label, lambda _=False, a=permanent: self._add_to_pool(
                FileAction.convert_action_to_text(a.action), a.target
            ))
        for number in sorted(hotkeys):
            action = hotkeys[number]
            action_label = _("Move") if action.is_move_action() else _("Copy")
            label = f"{number}: {action_label} → {Utils.get_relative_dirpath(action.target, levels=2)}"
            menu.addAction(label, lambda _=False, a=action: self._add_to_pool(
                FileAction.convert_action_to_text(a.action), a.target
            ))
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _show_recent_menu(self, btn: QPushButton) -> None:
        history = FileAction.action_history
        seen: set = set()
        unique: list = []
        for action in history:
            if action.is_delete_action():
                continue
            key = (FileAction.convert_action_to_text(action.action), action.target)
            if key not in seen:
                seen.add(key)
                unique.append(action)
        if not unique:
            self._app_actions.warn(_("No recent file actions found."))
            return
        menu = QMenu(self)
        for action in unique[:20]:
            action_text = FileAction.convert_action_to_text(action.action) or ""
            action_label = _("Move") if action_text == "move_file" else _("Copy")
            label = f"{action_label} → {Utils.get_relative_dirpath(action.target, levels=2)}"
            menu.addAction(label, lambda _=False, at=action_text, t=action.target: self._add_to_pool(at, t))
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    # ==================================================================
    # Presets UI
    # ==================================================================
    def _rebuild_presets(self) -> None:
        _clear_layout(self._presets_layout)
        if not FileActionSets.action_sets:
            empty_lbl = QLabel(_("No presets saved yet."))
            empty_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-style: italic;")
            self._presets_layout.addWidget(empty_lbl)
        for i, action_set in enumerate(FileActionSets.action_sets):
            self._presets_layout.addWidget(
                self._make_preset_row(i, action_set, active=(i == self._active_preset_index))
            )
        self._presets_layout.addStretch()

    def _make_preset_row(self, index: int, action_set: ActionSet, active: bool = False) -> QFrame:
        border = AppStyle.FG_COLOR if active else AppStyle.BORDER_COLOR
        row_frame = QFrame()
        row_frame.setStyleSheet(
            f"QFrame {{ background: {AppStyle.BG_COLOR}; border: 1px solid {border}; border-radius: 3px; padding: 2px; }}"
        )
        row_frame.setCursor(Qt.PointingHandCursor)

        row = QHBoxLayout(row_frame)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)

        name_lbl = QLabel(f"<b>{action_set.name}</b>")
        name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(name_lbl, 1)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(28)
        del_btn.clicked.connect(lambda _=False, i=index: self._delete_preset(i))
        row.addWidget(del_btn)

        row_frame.mousePressEvent = lambda evt, i=index: (
            self._load_preset(i) if evt.button() == Qt.LeftButton else None
        )
        return row_frame

    def _load_preset(self, index: int) -> None:
        action_set = FileActionSets.action_sets[index]
        # Ensure every step in the preset exists in the pool, adding if needed
        new_indices = []
        for step in action_set.steps:
            pool_index = FileActionSets.add_to_pool(step.action, step.target)
            new_indices.append(pool_index)
        FileActionSets.selected_indices = new_indices
        self._active_preset_index = index
        FileActionSets.store()
        self._rebuild_actions()
        self._rebuild_presets()

    def _delete_preset(self, index: int) -> None:
        del FileActionSets.action_sets[index]
        if self._active_preset_index is not None:
            if self._active_preset_index == index:
                self._active_preset_index = None
            elif self._active_preset_index > index:
                self._active_preset_index -= 1
        FileActionSets.store()
        self._rebuild_presets()

    def _save_as_preset(self) -> None:
        selected = FileActionSets.get_selected_actions()
        if not selected:
            self._app_actions.warn(_("No actions selected."))
            return

        initial = ""
        if self._active_preset_index is not None and self._active_preset_index < len(FileActionSets.action_sets):
            initial = FileActionSets.action_sets[self._active_preset_index].name

        dialog = _SavePresetDialog(self, initial_name=initial)
        if dialog.exec() != _SavePresetDialog.Accepted or not dialog.chosen_name:
            return

        name = dialog.chosen_name
        steps_copy = [ActionStep(s.action, s.target) for s in selected]

        for i, existing in enumerate(FileActionSets.action_sets):
            if existing.name == name:
                existing.steps = steps_copy
                self._active_preset_index = i
                FileActionSets.store()
                self._rebuild_presets()
                return

        FileActionSets.action_sets.append(ActionSet(name, steps_copy))
        self._active_preset_index = len(FileActionSets.action_sets) - 1
        FileActionSets.store()
        self._rebuild_presets()

    # ==================================================================
    # Execute
    # ==================================================================
    @require_password(ProtectedActions.RUN_FILE_ACTIONS)
    def _execute(self) -> None:
        from files.marked_files import MarkedFiles
        from utils.translations import marks_transfer_running_warn

        if MarkedFiles.is_transfer_running():
            self._app_actions.warn(
                marks_transfer_running_warn(_("run file action set"))
            )
            return

        selected = FileActionSets.get_selected_actions()
        if not selected:
            self._app_actions.warn(_("No actions selected."))
            return

        marks_snapshot = list(MarkedFiles.file_marks)
        if not marks_snapshot:
            self._app_actions.warn(_("No files are marked."))
            return

        copy_actions = [s for s in selected if not s.is_move()]
        move_actions = [s for s in selected if s.is_move()]

        errors = []
        for step in copy_actions:
            try:
                self._move_marks_callback(
                    self._app_actions,
                    target_dir=step.target,
                    move_func=Utils.copy_file,
                    files=marks_snapshot,
                    single_image=(len(marks_snapshot) == 1),
                )
            except Exception as e:
                errors.append(str(e))

        for step in move_actions:
            try:
                self._move_marks_callback(
                    self._app_actions,
                    target_dir=step.target,
                    move_func=Utils.move_file,
                    files=marks_snapshot,
                    single_image=(len(marks_snapshot) == 1),
                )
            except Exception as e:
                errors.append(str(e))

        if errors:
            self._app_actions.alert(
                _("File Action Sets Error"),
                "\n".join(errors),
                kind="error",
                master=self,
            )
        else:
            total = len(copy_actions) + len(move_actions)
            self._app_actions.toast(
                _("Executed {0} action(s) on {1} file(s).").format(total, len(marks_snapshot))
            )

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def reject(self) -> None:  # noqa: N802
        FileActionSetsWindow._instance = None
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        FileActionSetsWindow._instance = None
        super().closeEvent(event)

"""
FileInterceptorRulesWindow -- edit the ordered file-handling interceptor rules.

Singleton dialog listing rules in evaluation order with move-up/move-down
controls, since list position is the user's precedence control: the first
matching rule wins. Editing one rule opens a small modal form.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from files.file_interceptor_rule import (
    FileInterceptorRule,
    InterceptorAppliesTo,
    InterceptorBehavior,
    InterceptorTransformOp,
)
from files.file_interceptor_rules_manager import FileInterceptorRulesManager
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.app_actions import AppActions
from utils.constants import CompareMediaType
from utils.translations import _

#: Media types a rule can be restricted to. UNCONFIGURED is omitted -- it means
#: "unknown or disabled type", which is not something to write a rule against.
_SELECTABLE_MEDIA_TYPES = [
    CompareMediaType.IMAGE,
    CompareMediaType.GIF,
    CompareMediaType.VIDEO,
    CompareMediaType.PDF,
    CompareMediaType.SVG,
    CompareMediaType.HTML,
    CompareMediaType.AUDIO,
]


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip() != ""]


class _RuleEditDialog(SmartDialog):
    """Modal form for a single rule. Writes back into the rule on accept."""

    def __init__(self, parent: QWidget, rule: FileInterceptorRule) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Edit Interceptor Rule"),
            geometry="620x620",
            center=True,
        )
        self._rule = rule

        form = QFormLayout()

        self._name_edit = QLineEdit(rule.name)
        form.addRow(_("Name"), self._name_edit)

        self._active_check = QCheckBox(_("Rule is active"))
        self._active_check.setChecked(rule.is_active)
        form.addRow("", self._active_check)

        self._applies_combo = QComboBox()
        for applies_to in InterceptorAppliesTo:
            self._applies_combo.addItem(applies_to.get_translation(), applies_to)
        self._applies_combo.setCurrentIndex(
            self._applies_combo.findData(rule.applies_to)
        )
        form.addRow(_("Applies to"), self._applies_combo)

        self._target_dirs_edit = QLineEdit(", ".join(rule.match_target_dirs))
        self._target_dirs_edit.setToolTip(
            _("Comma-separated target directories. Leave empty to match any target.")
        )
        form.addRow(_("Target directories"), self._target_dirs_edit)

        self._subdirs_check = QCheckBox(_("Also match subdirectories"))
        self._subdirs_check.setChecked(rule.include_subdirectories)
        form.addRow("", self._subdirs_check)

        self._patterns_edit = QLineEdit(", ".join(rule.match_filename_patterns))
        self._patterns_edit.setToolTip(
            _("Comma-separated substrings. Leave empty to match any filename.")
        )
        form.addRow(_("Filename contains"), self._patterns_edit)

        self._case_check = QCheckBox(_("Filename match is case-sensitive"))
        self._case_check.setChecked(rule.filename_case_sensitive)
        form.addRow("", self._case_check)

        types_row = QVBoxLayout()
        types_hint = QLabel(_("Leave all unchecked to match any media type."))
        types_hint.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-style: italic;")
        types_row.addWidget(types_hint)
        self._type_checks: dict[CompareMediaType, QCheckBox] = {}
        selected_types = rule.match_media_types or []
        for media_type in _SELECTABLE_MEDIA_TYPES:
            check = QCheckBox(media_type.get_translation())
            check.setChecked(media_type in selected_types)
            self._type_checks[media_type] = check
            types_row.addWidget(check)
        types_widget = QWidget()
        types_widget.setLayout(types_row)
        form.addRow(_("Media types"), types_widget)

        self._behavior_combo = QComboBox()
        for behavior in InterceptorBehavior:
            self._behavior_combo.addItem(behavior.get_translation(), behavior)
        self._behavior_combo.setCurrentIndex(
            self._behavior_combo.findData(rule.behavior)
        )
        self._behavior_combo.currentIndexChanged.connect(self._update_behavior_fields)
        form.addRow(_("Behavior"), self._behavior_combo)

        self._block_message_edit = QLineEdit(rule.block_message)
        self._block_message_edit.setToolTip(
            _("Shown when this rule blocks a transfer. Optional.")
        )
        form.addRow(_("Block message"), self._block_message_edit)

        self._transform_combo = QComboBox()
        for op in InterceptorTransformOp:
            self._transform_combo.addItem(op.get_translation(), op)
        if rule.transform_op is not None:
            self._transform_combo.setCurrentIndex(
                self._transform_combo.findData(rule.transform_op)
            )
        form.addRow(_("Transform operation"), self._transform_combo)

        self._delete_original_check = QCheckBox(
            _("Remove the original after moving the transformed file")
        )
        self._delete_original_check.setChecked(rule.delete_original_after_transform)
        self._delete_original_check.setToolTip(
            _("Never applies to copy operations, only to moves.")
        )
        form.addRow("", self._delete_original_check)

        buttons = QHBoxLayout()
        buttons.addStretch()
        ok_btn = QPushButton(_("OK"))
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(ok_btn)
        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addStretch()
        outer.addLayout(buttons)

        self._update_behavior_fields()
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.reject)

    def _update_behavior_fields(self) -> None:
        is_transform = self._behavior_combo.currentData() == InterceptorBehavior.TRANSFORM
        self._transform_combo.setEnabled(is_transform)
        self._delete_original_check.setEnabled(is_transform)
        self._block_message_edit.setEnabled(not is_transform)

    def apply_to_rule(self) -> None:
        """Copy the form's values back onto the rule being edited."""
        rule = self._rule
        name = self._name_edit.text().strip()
        if name:
            rule.name = name
        rule.is_active = self._active_check.isChecked()
        rule.applies_to = self._applies_combo.currentData()
        rule.match_target_dirs = _split_csv(self._target_dirs_edit.text())
        rule.include_subdirectories = self._subdirs_check.isChecked()
        rule.match_filename_patterns = _split_csv(self._patterns_edit.text())
        rule.filename_case_sensitive = self._case_check.isChecked()
        selected = [mt for mt, check in self._type_checks.items() if check.isChecked()]
        rule.match_media_types = selected or None
        rule.behavior = self._behavior_combo.currentData()
        rule.block_message = self._block_message_edit.text().strip()
        rule.transform_op = self._transform_combo.currentData()
        rule.delete_original_after_transform = self._delete_original_check.isChecked()


class FileInterceptorRulesWindow(SmartDialog):
    """Ordered list of interceptor rules with add/edit/remove/reorder."""

    _instance: Optional["FileInterceptorRulesWindow"] = None

    def __init__(self, app_master: QWidget, app_actions: AppActions) -> None:
        super().__init__(
            parent=app_master,
            position_parent=app_master,
            title=_("File Handling Interceptor Rules"),
            geometry="900x700",
            respect_title_bar=True,
        )
        FileInterceptorRulesWindow._instance = self
        self._app_master = app_master
        self._app_actions = app_actions

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        hint = QLabel(
            _(
                "Rules are checked in order when marked files are moved or copied; "
                "the first matching rule applies."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-style: italic;")
        outer.addWidget(hint)

        bar = QHBoxLayout()
        add_btn = QPushButton(_("Add rule"))
        add_btn.setFocusPolicy(Qt.NoFocus)
        add_btn.clicked.connect(self._add_rule)
        bar.addWidget(add_btn)
        bar.addStretch()
        outer.addLayout(bar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFocusPolicy(Qt.NoFocus)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {AppStyle.BG_COLOR}; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(2)
        self._scroll.setWidget(self._scroll_content)
        outer.addWidget(self._scroll, 1)

        self._rebuild_rows()
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # Factory (singleton)
    # ------------------------------------------------------------------
    @classmethod
    def show_window(cls, parent: QWidget, app_actions: AppActions) -> None:
        if cls._instance is not None:
            try:
                if cls._instance.isVisible():
                    if cls._instance.isMinimized():
                        cls._instance.showNormal()
                    cls._instance._rebuild_rows()
                    cls._instance.raise_()
                    cls._instance.activateWindow()
                    return
                cls._instance = None
            except Exception:
                cls._instance = None
        win = cls(parent, app_actions)
        win.show()

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------
    def _rebuild_rows(self) -> None:
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

        rules = FileInterceptorRulesManager.rules
        if not rules:
            empty = QLabel(_("No interceptor rules configured."))
            empty.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            self._scroll_layout.addWidget(empty)

        for index, rule in enumerate(rules):
            row = QHBoxLayout()

            label = QLabel(f"{index + 1}. {rule.name}\n    {rule.summary()}")
            label.setWordWrap(True)
            color = AppStyle.FG_COLOR if rule.is_active else "gray"
            label.setStyleSheet(f"color: {color};")
            row.addWidget(label, 1)

            up_btn = QPushButton("↑")
            up_btn.setFixedWidth(30)
            up_btn.setFocusPolicy(Qt.NoFocus)
            up_btn.setToolTip(_("Move this rule earlier in the evaluation order"))
            up_btn.setEnabled(index > 0)
            up_btn.clicked.connect(lambda _c=False, i=index: self._move_rule(i, -1))
            row.addWidget(up_btn)

            down_btn = QPushButton("↓")
            down_btn.setFixedWidth(30)
            down_btn.setFocusPolicy(Qt.NoFocus)
            down_btn.setToolTip(_("Move this rule later in the evaluation order"))
            down_btn.setEnabled(index < len(rules) - 1)
            down_btn.clicked.connect(lambda _c=False, i=index: self._move_rule(i, 1))
            row.addWidget(down_btn)

            edit_btn = QPushButton(_("Edit"))
            edit_btn.setFocusPolicy(Qt.NoFocus)
            edit_btn.clicked.connect(lambda _c=False, i=index: self._edit_rule(i))
            row.addWidget(edit_btn)

            remove_btn = QPushButton("×")
            remove_btn.setFixedWidth(28)
            remove_btn.setFocusPolicy(Qt.NoFocus)
            remove_btn.setToolTip(_("Remove this rule"))
            remove_btn.clicked.connect(lambda _c=False, i=index: self._remove_rule(i))
            row.addWidget(remove_btn)

            self._scroll_layout.addLayout(row)

        self._scroll_layout.addStretch()

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                FileInterceptorRulesWindow._clear_layout(item.layout())

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _persist(self) -> None:
        FileInterceptorRulesManager.store_rules()
        self._rebuild_rows()

    def _add_rule(self) -> None:
        rule = FileInterceptorRule()
        dialog = _RuleEditDialog(self, rule)
        if dialog.exec():
            dialog.apply_to_rule()
            FileInterceptorRulesManager.rules.append(rule)
            self._persist()
            self._app_actions.toast(_("Added interceptor rule: {0}").format(rule.name))

    def _edit_rule(self, index: int) -> None:
        if not 0 <= index < len(FileInterceptorRulesManager.rules):
            return
        rule = FileInterceptorRulesManager.rules[index]
        dialog = _RuleEditDialog(self, rule)
        if dialog.exec():
            dialog.apply_to_rule()
            self._persist()

    def _remove_rule(self, index: int) -> None:
        if not 0 <= index < len(FileInterceptorRulesManager.rules):
            return
        removed = FileInterceptorRulesManager.rules.pop(index)
        self._persist()
        self._app_actions.toast(_("Removed interceptor rule: {0}").format(removed.name))

    def _move_rule(self, index: int, offset: int) -> None:
        rules = FileInterceptorRulesManager.rules
        new_index = index + offset
        if not (0 <= index < len(rules) and 0 <= new_index < len(rules)):
            return
        rules[index], rules[new_index] = rules[new_index], rules[index]
        self._persist()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        FileInterceptorRulesWindow._instance = None

    def reject(self) -> None:  # noqa: N802  (Escape key -- does NOT call closeEvent)
        self._on_close()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802  (X button -> reject())
        self._on_close()
        super().closeEvent(event)

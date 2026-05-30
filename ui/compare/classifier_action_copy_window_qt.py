"""
PySide6 port of compare/classifier_action_copy_window.py --
ClassifierActionCopyWindow.

Intermediary dialog for copying ClassifierAction and Prevalidation objects
into new instances of either type.

Non-UI imports:
  - ClassifierAction, Prevalidation from compare.classifier_action
  - ClassifierActionsManager from compare.classifier_actions_manager
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QWidget,
)

from compare.classifier_action import ClassifierAction, Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from lib.multi_display_qt import SmartDialog
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.constants import ClassifierActionClass
from utils.config import config
from utils.translations import _


class ClassifierActionCopyWindow(SmartDialog):
    """
    Intermediary dialog for copying ClassifierAction / Prevalidation objects.

    Allows:
      - ClassifierAction  -> new ClassifierAction
      - Prevalidation     -> new Prevalidation
      - ClassifierAction  -> new Prevalidation
      - Prevalidation     -> new ClassifierAction
    """

    _instance: Optional[ClassifierActionCopyWindow] = None

    def __init__(
        self,
        parent: QWidget,
        app_actions,
        source_item,
        source_type: str = "auto",
        refresh_classifier_actions_callback: Optional[Callable] = None,
        refresh_prevalidations_callback: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Copy Classifier Action / Prevalidation"),
            geometry="500x300",
        )
        ClassifierActionCopyWindow._instance = self

        self._app_actions = app_actions
        self._source_item = source_item
        self._refresh_ca_cb = refresh_classifier_actions_callback
        self._refresh_pv_cb = refresh_prevalidations_callback

        # Detect source type
        if source_type == "auto":
            if isinstance(source_item, Prevalidation):
                self._source_type = ClassifierActionClass.PREVALIDATION
            elif isinstance(source_item, ClassifierAction):
                self._source_type = ClassifierActionClass.CLASSIFIER_ACTION
            else:
                raise ValueError(
                    f"Unknown source item type: {type(source_item)}"
                )
        else:
            self._source_type = ClassifierActionClass.from_key(source_type)

        self._build_ui()

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_Return), self).activated.connect(
            self._copy_item
        )

        self._name_edit.setFocus()
        self._name_edit.selectAll()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(8)
        row = 0

        bold = QFont()
        bold.setBold(True)

        # -- Source information -------------------------------------------
        src_header = QLabel(_("Copying from:"))
        src_header.setFont(bold)
        src_header.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(src_header, row, 0, 1, 2, Qt.AlignLeft)
        row += 1

        src_name = QLabel(f"{_('Name')}: {self._source_item.name}")
        src_name.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(src_name, row, 0, 1, 2, Qt.AlignLeft)
        row += 1

        src_type = QLabel(
            f"{_('Type')}: {self._source_type.get_display_value()}"
        )
        src_type.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(src_type, row, 0, 1, 2, Qt.AlignLeft)
        row += 1

        # spacer
        grid.setRowMinimumHeight(row, 16)
        row += 1

        # -- Target type --------------------------------------------------
        tgt_header = QLabel(_("Copy to:"))
        tgt_header.setFont(bold)
        tgt_header.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(tgt_header, row, 0, 1, 2, Qt.AlignLeft)
        row += 1

        tgt_type_lbl = QLabel(_("Target Type:"))
        tgt_type_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(tgt_type_lbl, row, 0, Qt.AlignLeft)

        self._target_type_combo = QComboBox()
        self._target_type_combo.addItems([
            ClassifierActionClass.CLASSIFIER_ACTION.get_display_value(),
            ClassifierActionClass.PREVALIDATION.get_display_value(),
        ])
        self._target_type_combo.setCurrentText(
            self._source_type.get_display_value()
        )
        grid.addWidget(self._target_type_combo, row, 1)
        row += 1

        # -- New name -----------------------------------------------------
        name_lbl = QLabel(_("New Name:"))
        name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(name_lbl, row, 0, Qt.AlignLeft)

        self._name_edit = QLineEdit(self._generate_default_name())
        grid.addWidget(self._name_edit, row, 1)
        row += 1

        # spacer
        grid.setRowMinimumHeight(row, 16)
        row += 1

        # -- Buttons ------------------------------------------------------
        btn_row = QHBoxLayout()
        copy_btn = QPushButton(_("Copy"))
        copy_btn.clicked.connect(self._copy_item)
        btn_row.addWidget(copy_btn)

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(cancel_btn)

        btn_row.addStretch()
        grid.addLayout(btn_row, row, 0, 1, 2)
        row += 1

        grid.setRowStretch(row, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_target_class(self) -> ClassifierActionClass:
        return ClassifierActionClass.from_display_value(
            self._target_type_combo.currentText()
        )

    def _get_existing_names(self, target_class: ClassifierActionClass) -> list[str]:
        if target_class == ClassifierActionClass.PREVALIDATION:
            return [pv.name for pv in ClassifierActionsManager.prevalidations]
        return [ca.name for ca in ClassifierActionsManager.classifier_actions]

    def _generate_default_name(self) -> str:
        source_name = self._source_item.name
        target_class = self._get_target_class()
        existing = self._get_existing_names(target_class)

        if " Copy" in source_name:
            base_name = source_name.rsplit(" Copy", 1)[0]
            copy_num = 2
            while True:
                candidate = f"{base_name} Copy {copy_num}"
                if candidate not in existing:
                    return candidate
                copy_num += 1
        else:
            candidate = f"{source_name} Copy"
            if candidate not in existing:
                return candidate
            copy_num = 2
            while True:
                candidate = f"{source_name} Copy {copy_num}"
                if candidate not in existing:
                    return candidate
                copy_num += 1

    # ------------------------------------------------------------------
    # Copy action
    # ------------------------------------------------------------------
    def _copy_item(self) -> None:
        new_name = self._name_edit.text().strip()
        if not new_name:
            qt_alert(_("Error"), _("Name cannot be empty"), kind="warning", master=self)
            return

        target_class = self._get_target_class()
        existing = self._get_existing_names(target_class)

        if new_name in existing:
            qt_alert(
                _("Error"),
                _("A {0} with this name already exists").format(
                    target_class.get_display_value().lower()
                ),
                kind="warning",
                master=self,
            )
            return

        # Build new item from source dict
        source_dict = self._source_item.to_dict()
        source_dict["name"] = new_name

        if target_class == ClassifierActionClass.PREVALIDATION:
            if isinstance(self._source_item, Prevalidation):
                new_item = Prevalidation.from_dict(source_dict)
            else:
                if "profile_name" not in source_dict:
                    source_dict["profile_name"] = None
                new_item = Prevalidation.from_dict(source_dict)
        else:
            if isinstance(self._source_item, Prevalidation):
                source_dict.pop("profile_name", None)
                new_item = ClassifierAction.from_dict(source_dict)
            else:
                new_item = ClassifierAction.from_dict(source_dict)

        try:
            new_item.validate()
        except Exception as e:
            qt_alert(
                _("Invalid Copy Target"),
                str(e),
                kind="warning",
                master=self,
            )
            return

        # Add and open modify window
        if target_class == ClassifierActionClass.PREVALIDATION:
            if new_item not in ClassifierActionsManager.prevalidations:
                ClassifierActionsManager.prevalidations.insert(0, new_item)

            self.close()

            from ui.compare.prevalidations_tab_qt import PrevalidationModifyWindow

            refresh_cb = self._refresh_pv_cb or self._fallback_pv_refresh()
            PrevalidationModifyWindow._instance = PrevalidationModifyWindow(
                self.parentWidget(), self._app_actions, refresh_cb, new_item
            )
            PrevalidationModifyWindow._instance.show()
        else:
            if new_item not in ClassifierActionsManager.classifier_actions:
                ClassifierActionsManager.classifier_actions.insert(0, new_item)

            self.close()

            from ui.compare.classifier_management_window_qt import (
                ClassifierActionModifyWindow,
            )

            refresh_cb = self._refresh_ca_cb or self._fallback_ca_refresh()
            ClassifierActionModifyWindow._instance = ClassifierActionModifyWindow(
                self.parentWidget(), self._app_actions, refresh_cb, new_item
            )
            ClassifierActionModifyWindow._instance.show()

    # ------------------------------------------------------------------
    # Fallback refresh callbacks
    # ------------------------------------------------------------------
    def _fallback_pv_refresh(self) -> Callable:
        def refresh(prevalidation=None):
            if (
                prevalidation is not None
                and prevalidation not in ClassifierActionsManager.prevalidations
            ):
                ClassifierActionsManager.prevalidations.insert(0, prevalidation)
            try:
                from ui.compare.classifier_management_window_qt import (
                    ClassifierManagementWindow,
                )
                win = ClassifierManagementWindow._instance
                if win is not None and hasattr(win, "_prevalidations_tab"):
                    win._prevalidations_tab.refresh_prevalidations(prevalidation)
            except Exception:
                pass

        return refresh

    def _fallback_ca_refresh(self) -> Callable:
        def refresh(classifier_action=None):
            if (
                classifier_action is not None
                and classifier_action
                not in ClassifierActionsManager.classifier_actions
            ):
                ClassifierActionsManager.classifier_actions.insert(
                    0, classifier_action
                )
            try:
                from ui.compare.classifier_management_window_qt import (
                    ClassifierManagementWindow,
                )
                win = ClassifierManagementWindow._instance
                if win is not None and hasattr(win, "_classifier_actions_tab"):
                    win._classifier_actions_tab.refresh_classifier_actions(
                        classifier_action
                    )
            except Exception:
                pass

        return refresh

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        ClassifierActionCopyWindow._instance = None
        super().closeEvent(event)

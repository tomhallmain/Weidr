"""
AutoSortConfirmationWindow -- choose which auto-sorted categories confirm first.

Small singleton dialog holding one freeform, comma-separated category list.
Categories are plain user-typed strings with no central registry to offer as
checkboxes, so a text field is the honest control here.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from files.auto_sort_confirmation import AutoSortConfirmation
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.app_actions import AppActions
from utils.translations import _


class AutoSortConfirmationWindow(SmartDialog):

    _instance: Optional["AutoSortConfirmationWindow"] = None

    def __init__(self, app_master: QWidget, app_actions: AppActions) -> None:
        super().__init__(
            parent=app_master,
            position_parent=app_master,
            title=_("Auto-Sort Confirmation Categories"),
            geometry="640x260",
            center=True,
        )
        AutoSortConfirmationWindow._instance = self
        self._app_actions = app_actions

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        hint = QLabel(
            _(
                "When viewing the last automatically sorted media, ask for "
                "confirmation first if it was sorted into one of these categories.\n"
                "Categories are matched without regard to case."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(hint)

        label = QLabel(_("Categories (comma-separated):"))
        label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(label)

        self._categories_edit = QLineEdit(", ".join(AutoSortConfirmation.get_categories()))
        self._categories_edit.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._categories_edit.returnPressed.connect(self._save)
        root.addWidget(self._categories_edit)

        root.addStretch()

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.close)
        buttons.addWidget(cancel_btn)
        save_btn = QPushButton(_("Save"))
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)
        root.addLayout(buttons)

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    @classmethod
    def show_window(cls, parent: QWidget, app_actions: AppActions) -> None:
        if cls._instance is not None:
            try:
                if cls._instance.isVisible():
                    if cls._instance.isMinimized():
                        cls._instance.showNormal()
                    cls._instance.raise_()
                    cls._instance.activateWindow()
                    return
                cls._instance = None
            except Exception:
                cls._instance = None
        win = cls(parent, app_actions)
        win.show()

    def _save(self) -> None:
        categories = [
            part.strip()
            for part in self._categories_edit.text().split(",")
            if part.strip() != ""
        ]
        AutoSortConfirmation.set_categories(categories)
        AutoSortConfirmation.save()
        self._app_actions.toast(
            _("Confirmation set for {0} categories").format(
                len(AutoSortConfirmation.confirm_categories)
            )
        )
        self.close()

    def _on_close(self) -> None:
        AutoSortConfirmationWindow._instance = None

    def reject(self) -> None:  # noqa: N802  (Escape key -- does NOT call closeEvent)
        self._on_close()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802  (X button -> reject())
        self._on_close()
        super().closeEvent(event)

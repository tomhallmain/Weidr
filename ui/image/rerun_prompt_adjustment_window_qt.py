"""
Dialog to edit positive/negative prompts before a redo-prompt image generation run.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from image.image_data_extractor import image_data_extractor
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.translations import _


class RerunPromptAdjustmentWindow(SmartDialog):
    """Edit prompts before sending a redo-prompt request to SD Runner."""

    def __init__(
        self,
        parent: QWidget,
        media_path: str,
        positive_prompt: str,
        negative_prompt: str,
        dimensions: str = "720x560",
    ) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Rerun Prompt Adjustment") + " - " + os.path.basename(media_path),
            geometry=dimensions,
        )
        self._positive_prompt = positive_prompt
        self._negative_prompt = negative_prompt
        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        hint = QLabel(
            _(
                "Adjust prompts before rerunning. Changes are sent to SD Runner "
                "for this run only."
            )
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        layout.addWidget(hint)

        pos_label = QLabel(_("Positive prompt"))
        pos_label.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        layout.addWidget(pos_label)

        self._positive_edit = QPlainTextEdit()
        self._positive_edit.setPlainText(self._positive_prompt)
        self._positive_edit.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        layout.addWidget(self._positive_edit, stretch=2)

        neg_label = QLabel(_("Negative prompt"))
        neg_label.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        layout.addWidget(neg_label)

        self._negative_edit = QPlainTextEdit()
        self._negative_edit.setPlainText(self._negative_prompt)
        self._negative_edit.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        layout.addWidget(self._negative_edit, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch()

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        rerun_btn = QPushButton(_("Rerun"))
        rerun_btn.setDefault(True)
        rerun_btn.clicked.connect(self._on_rerun)
        buttons.addWidget(rerun_btn)

        layout.addLayout(buttons)

    def _bind_shortcuts(self) -> None:
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self.reject)

    def _on_rerun(self) -> None:
        self._positive_prompt = self._positive_edit.toPlainText()
        self._negative_prompt = self._negative_edit.toPlainText()
        self.accept()

    @property
    def positive_prompt(self) -> str:
        return self._positive_prompt

    @property
    def negative_prompt(self) -> str:
        return self._negative_prompt

    @staticmethod
    def prompt_adjustment(
        parent: QWidget,
        media_path: str,
    ) -> Optional[tuple[str, str]]:
        """
        Show the adjustment dialog modally.

        Returns ``(positive, negative)`` if the user confirms, else ``None``.
        """
        positive, negative = image_data_extractor.extract_positive_prompt_for_rerun(
            media_path
        )
        dialog = RerunPromptAdjustmentWindow(
            parent=parent,
            media_path=media_path,
            positive_prompt=positive,
            negative_prompt=negative,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.positive_prompt, dialog.negative_prompt

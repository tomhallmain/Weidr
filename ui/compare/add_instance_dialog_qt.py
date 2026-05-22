"""
Add-Instance dialog for CompareSettingsWindow.

Modal dialog that collects configuration for a new CompareManager instance:
  - Compare mode
  - Optional search text (positive / negative) — meaningful for embedding modes
  - Optional per-instance threshold override
  - Weight — shown only when the current combination logic is WEIGHTED
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from compare.compare_manager import CombinationLogic
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.constants import CompareMode
from utils.translations import I18N

_ = I18N._

MAX_INSTANCES = 10


class AddInstanceDialog(SmartDialog):
    """Modal dialog for adding a new compare instance."""

    def __init__(
        self,
        parent: QWidget,
        combination_logic: CombinationLogic,
        current_instance_count: int,
    ) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Add Compare Instance"),
            geometry="500x340",
            center=True,
        )
        self._result: Optional[dict] = None

        self._build_ui(combination_logic, current_instance_count)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.reject)

    # ------------------------------------------------------------------
    def _build_ui(self, combination_logic: CombinationLogic,
                  current_instance_count: int) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)

        def lbl(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            return l

        row = 0

        # Mode
        grid.addWidget(lbl(_("Compare mode:")), row, 0)
        self._mode_combo = QComboBox()
        for mode in CompareMode:
            self._mode_combo.addItem(mode.get_text(), userData=mode)
        grid.addWidget(self._mode_combo, row, 1)
        row += 1

        # Search text
        grid.addWidget(lbl(_("Search text:")), row, 0)
        self._search_text = QLineEdit()
        self._search_text.setPlaceholderText(_("(optional — for embedding modes)"))
        grid.addWidget(self._search_text, row, 1)
        row += 1

        # Negative search text
        grid.addWidget(lbl(_("Negative search text:")), row, 0)
        self._neg_text = QLineEdit()
        self._neg_text.setPlaceholderText(_("(optional)"))
        grid.addWidget(self._neg_text, row, 1)
        row += 1

        # Threshold override
        grid.addWidget(lbl(_("Threshold override:")), row, 0)
        self._threshold_edit = QLineEdit()
        self._threshold_edit.setPlaceholderText(_("default"))
        grid.addWidget(self._threshold_edit, row, 1)
        row += 1

        # Weight (WEIGHTED mode only)
        self._weight_lbl = lbl(_("Weight:"))
        grid.addWidget(self._weight_lbl, row, 0)
        self._weight_edit = QLineEdit("1.0")
        grid.addWidget(self._weight_edit, row, 1)
        row += 1

        is_weighted = combination_logic == CombinationLogic.WEIGHTED
        self._weight_lbl.setVisible(is_weighted)
        self._weight_edit.setVisible(is_weighted)

        outer.addLayout(grid)

        # Capacity note
        if current_instance_count >= MAX_INSTANCES:
            warn = QLabel(_("Maximum of %s instances reached.").format(MAX_INSTANCES))
            warn.setStyleSheet("color: #e06c75;")
            outer.addWidget(warn)

        outer.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        self._ok_btn = QPushButton(_("OK"))
        self._ok_btn.setDefault(True)
        self._ok_btn.setEnabled(current_instance_count < MAX_INSTANCES)
        self._ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(self._ok_btn)

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _on_ok(self) -> None:
        mode: CompareMode = self._mode_combo.currentData()
        search_text = self._search_text.text().strip() or None
        neg_text = self._neg_text.text().strip() or None

        threshold: Optional[float] = None
        t_str = self._threshold_edit.text().strip()
        if t_str:
            try:
                threshold = int(t_str) if mode == CompareMode.COLOR_MATCHING else float(t_str)
            except ValueError:
                pass

        weight = 1.0
        w_str = self._weight_edit.text().strip()
        if w_str:
            try:
                weight = float(w_str)
            except ValueError:
                pass

        self._result = {
            "mode": mode,
            "search_text": search_text,
            "search_text_negative": neg_text,
            "threshold": threshold,
            "weight": weight,
        }
        self.accept()

    # ------------------------------------------------------------------
    def get_result(self) -> Optional[dict]:
        """
        Returns the collected instance configuration, or None if the dialog
        was cancelled.  Keys: mode, search_text, search_text_negative,
        threshold, weight.
        """
        return self._result

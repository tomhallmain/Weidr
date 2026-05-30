"""
PySide6 port of the LookaheadWindow from compare/lookahead.py.

Only the UI class is ported here. The non-UI ``Lookahead`` data class
is imported from the original module per the reuse policy.

Non-UI imports:
  - Lookahead from compare.lookahead (reuse policy)
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QLabel, QLineEdit,
    QPushButton, QSlider, QWidget,
)

from compare.lookahead import Lookahead
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.config import config
from utils.translations import _
from utils.logging_setup import get_logger
logger = get_logger("lookahead_window_qt")


class LookaheadWindow(SmartDialog):
    """
    Create / edit dialog for a single Lookahead object.

    Fields:
      - Lookahead name (QLineEdit)
      - "Reference existing prevalidation" checkbox (QCheckBox)
      - Prevalidation name (QComboBox) **or** custom text (QLineEdit)
      - Threshold slider (QSlider 0-100, maps to 0.00-1.00)
      - Done button
    """

    _instance: Optional[LookaheadWindow] = None

    def __init__(
        self,
        parent: QWidget,
        app_actions,
        refresh_callback: Callable,
        lookahead: Optional[Lookahead] = None,
        dimensions: str = "500x450",
    ) -> None:
        self._is_edit = lookahead is not None
        self._lookahead = lookahead if lookahead is not None else Lookahead()
        self._original_name = self._lookahead.name if self._is_edit else None

        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Edit Lookahead") if self._is_edit else _("Create Lookahead"),
            geometry=dimensions,
        )
        LookaheadWindow._instance = self

        self._app_actions = app_actions
        self._refresh_callback = refresh_callback

        # -- fetch prevalidation names for combobox -----------------------
        from compare.classifier_actions_manager import ClassifierActionsManager
        self._existing_names = [
            pv.name for pv in ClassifierActionsManager.prevalidations
        ]

        self._build_ui()

        # -- keyboard shortcuts -------------------------------------------
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_Return), self).activated.connect(
            self._finalize_lookahead
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(8)
        row = 0

        # -- Lookahead name -----------------------------------------------
        name_lbl = QLabel(_("Lookahead Name"))
        name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(name_lbl, row, 0, Qt.AlignLeft)

        self._name_edit = QLineEdit(self._lookahead.name)
        grid.addWidget(self._name_edit, row, 1)
        row += 1

        # -- "Reference existing prevalidation" checkbox ------------------
        self._is_prevalidation_cb = QCheckBox(
            _("Reference existing prevalidation name")
        )
        self._is_prevalidation_cb.setChecked(self._lookahead.is_prevalidation_name)
        self._is_prevalidation_cb.stateChanged.connect(self._update_ui_for_type)
        grid.addWidget(self._is_prevalidation_cb, row, 1, Qt.AlignLeft)
        row += 1

        # -- Name-or-text label -------------------------------------------
        not_lbl = QLabel(_("Prevalidation Name or Custom Text"))
        not_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(not_lbl, row, 0, Qt.AlignLeft)

        # Combobox (prevalidation name selector)
        self._name_or_text_combo = QComboBox()
        self._name_or_text_combo.setEditable(True)
        self._name_or_text_combo.addItems(self._existing_names)
        self._name_or_text_combo.setCurrentText(self._lookahead.name_or_text)
        grid.addWidget(self._name_or_text_combo, row, 1)

        # Line edit (custom text)
        self._name_or_text_edit = QLineEdit(self._lookahead.name_or_text)
        grid.addWidget(self._name_or_text_edit, row, 1)

        self._update_ui_for_type()
        row += 1

        # -- Threshold slider ---------------------------------------------
        threshold_lbl = QLabel(_("Threshold"))
        threshold_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(threshold_lbl, row, 0, Qt.AlignLeft)

        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setRange(0, 100)
        self._threshold_slider.setValue(int(self._lookahead.threshold * 100))
        self._threshold_slider.setTickInterval(10)
        self._threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        grid.addWidget(self._threshold_slider, row, 1)

        self._threshold_value_lbl = QLabel(
            f"{self._threshold_slider.value()}%"
        )
        self._threshold_value_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(self._threshold_value_lbl, row, 2)
        self._threshold_slider.valueChanged.connect(
            lambda v: self._threshold_value_lbl.setText(f"{v}%")
        )
        row += 1

        # -- Done button --------------------------------------------------
        done_btn = QPushButton(_("Done"))
        done_btn.clicked.connect(self._finalize_lookahead)
        grid.addWidget(done_btn, row, 0)
        row += 1

        grid.setRowStretch(row, 1)

    # ------------------------------------------------------------------
    # Toggle combo / line-edit
    # ------------------------------------------------------------------
    def _update_ui_for_type(self) -> None:
        is_pv = self._is_prevalidation_cb.isChecked()
        self._name_or_text_combo.setVisible(is_pv)
        self._name_or_text_edit.setVisible(not is_pv)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def _finalize_lookahead(self) -> None:
        name = self._name_edit.text().strip()
        is_prevalidation_name = self._is_prevalidation_cb.isChecked()
        name_or_text = (
            self._name_or_text_combo.currentText().strip()
            if is_prevalidation_name
            else self._name_or_text_edit.text().strip()
        )

        if not name:
            logger.error("Lookahead name is required")
            return
        if not name_or_text:
            logger.error("Prevalidation name or custom text is required")
            return

        # Duplicate-name check
        if not self._is_edit:
            if Lookahead.get_lookahead_by_name(name) is not None:
                logger.error(f"Lookahead with name {name} already exists")
                return
        else:
            if name != self._original_name:
                if Lookahead.get_lookahead_by_name(name) is not None:
                    logger.error(f"Lookahead with name {name} already exists")
                    return

        threshold = self._threshold_slider.value() / 100.0

        # If prevalidation name, verify it exists
        if is_prevalidation_name and name_or_text not in self._existing_names:
            logger.warning(
                f"Prevalidation '{name_or_text}' not found, treating as custom text"
            )
            is_prevalidation_name = False

        self._lookahead.name = name
        self._lookahead.name_or_text = name_or_text
        self._lookahead.threshold = threshold
        self._lookahead.is_prevalidation_name = is_prevalidation_name

        if not self._is_edit:
            Lookahead.lookaheads.append(self._lookahead)
        else:
            for idx, lh in enumerate(Lookahead.lookaheads):
                if lh.name == self._original_name:
                    Lookahead.lookaheads[idx] = self._lookahead
                    break

            # Update references if name changed
            if self._original_name != name:
                from compare.classifier_actions_manager import ClassifierActionsManager
                for pv in ClassifierActionsManager.prevalidations:
                    if self._original_name in pv.lookahead_names:
                        idx_ref = pv.lookahead_names.index(self._original_name)
                        pv.lookahead_names[idx_ref] = name

        self.close()
        self._refresh_callback()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        LookaheadWindow._instance = None
        super().closeEvent(event)

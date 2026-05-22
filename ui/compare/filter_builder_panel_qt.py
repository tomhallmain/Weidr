"""
Filter builder panel — renders a CompareFilter tree as an editable list of
condition rows. Each row maps to a SizeFilter or ModelFilter leaf. Multiple
rows are combined via a top-level AND / OR / NOT operator into a
CompareFilterGroup.

Public interface:
    panel.get_filter() -> Optional[CompareFilter]
    panel.set_filter(f: Optional[CompareFilter]) -> None
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget, QLineEdit,
)

from compare.compare_filters import (
    CompareFilter, CompareFilterGroup, FilterOperator,
    ModelFilter, SizeFilter,
)
from ui.app_style import AppStyle
from utils.translations import I18N

_ = I18N._


class _FilterRow(QWidget):
    """A single condition row: [type] [sub-fields…] [✕]."""

    def __init__(self, on_remove: Callable[['_FilterRow'], None],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_remove = on_remove

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        # --- Filter-type selector ---
        self._type_combo = QComboBox()
        self._type_combo.addItems([_("Size"), _("Model")])
        self._type_combo.setFixedWidth(80)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        layout.addWidget(self._type_combo)

        # --- Size sub-widgets ---
        self._size_widget = QWidget()
        sl = QHBoxLayout(self._size_widget)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)

        self._size_sub = QComboBox()
        self._size_sub.addItems([_("exact"), _("min"), _("max")])
        self._size_sub.setFixedWidth(70)
        sl.addWidget(self._size_sub)

        self._size_w = QLineEdit()
        self._size_w.setFixedWidth(58)
        self._size_w.setPlaceholderText("W")
        sl.addWidget(self._size_w)

        cross = QLabel("×")
        cross.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        sl.addWidget(cross)

        self._size_h = QLineEdit()
        self._size_h.setFixedWidth(58)
        self._size_h.setPlaceholderText("H")
        sl.addWidget(self._size_h)

        self._tol_lbl = QLabel("±")
        self._tol_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        sl.addWidget(self._tol_lbl)

        self._size_tol = QLineEdit()
        self._size_tol.setFixedWidth(40)
        self._size_tol.setPlaceholderText("0")
        sl.addWidget(self._size_tol)

        self._size_sub.currentTextChanged.connect(self._on_size_sub_changed)
        layout.addWidget(self._size_widget)

        # --- Model sub-widgets ---
        self._model_widget = QWidget()
        ml = QHBoxLayout(self._model_widget)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(4)

        self._model_mode_combo = QComboBox()
        self._model_mode_combo.addItems([_("contains"), _("does not contain")])
        self._model_mode_combo.setFixedWidth(160)
        ml.addWidget(self._model_mode_combo)

        self._model_text = QLineEdit()
        self._model_text.setPlaceholderText(_("model names (space-separated)"))
        self._model_text.setMinimumWidth(180)
        ml.addWidget(self._model_text)

        self._model_match_any = QCheckBox(_("match any"))
        ml.addWidget(self._model_match_any)

        self._model_loras = QCheckBox(_("incl. LoRAs"))
        self._model_loras.setChecked(True)
        ml.addWidget(self._model_loras)

        layout.addWidget(self._model_widget)
        layout.addStretch()

        # --- Remove button ---
        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip(_("Remove this condition"))
        remove_btn.clicked.connect(lambda: self._on_remove(self))
        layout.addWidget(remove_btn)

        # Initial visibility
        self._on_type_changed(self._type_combo.currentText())

    # ------------------------------------------------------------------
    def _on_type_changed(self, type_text: str) -> None:
        is_size = (type_text == _("Size"))
        self._size_widget.setVisible(is_size)
        self._model_widget.setVisible(not is_size)

    def _on_size_sub_changed(self, sub_text: str) -> None:
        is_exact = (sub_text == _("exact"))
        self._tol_lbl.setVisible(is_exact)
        self._size_tol.setVisible(is_exact)

    # ------------------------------------------------------------------
    def get_filter(self) -> Optional[CompareFilter]:
        if self._type_combo.currentText() == _("Size"):
            return self._build_size_filter()
        return self._build_model_filter()

    def _build_size_filter(self) -> Optional[SizeFilter]:
        try:
            w = int(self._size_w.text().strip()) if self._size_w.text().strip() else None
            h = int(self._size_h.text().strip()) if self._size_h.text().strip() else None
        except ValueError:
            return None
        if w is None and h is None:
            return None
        size = (w or 0, h or 0)
        sub = self._size_sub.currentText()
        try:
            tol = int(self._size_tol.text().strip() or "0")
        except ValueError:
            tol = 0
        if sub == _("exact"):
            return SizeFilter(exact_size=size, size_tolerance=tol)
        if sub == _("min"):
            return SizeFilter(min_size=size)
        return SizeFilter(max_size=size)

    def _build_model_filter(self) -> Optional[ModelFilter]:
        raw = self._model_text.text().strip()
        if not raw:
            return None
        models = [m for m in raw.replace(",", " ").split() if m]
        if not models:
            return None
        mode = "exclude" if self._model_mode_combo.currentText() == _("does not contain") else "include"
        return ModelFilter(
            models=models,
            mode=mode,
            match_any=self._model_match_any.isChecked(),
            include_loras=self._model_loras.isChecked(),
        )

    # ------------------------------------------------------------------
    def set_filter(self, f: CompareFilter) -> None:
        if isinstance(f, SizeFilter):
            self._type_combo.setCurrentText(_("Size"))
            if f.exact_size:
                self._size_sub.setCurrentText(_("exact"))
                self._size_w.setText(str(f.exact_size[0]))
                self._size_h.setText(str(f.exact_size[1]))
                self._size_tol.setText(str(f.size_tolerance))
            elif f.min_size:
                self._size_sub.setCurrentText(_("min"))
                self._size_w.setText(str(f.min_size[0]))
                self._size_h.setText(str(f.min_size[1]))
            elif f.max_size:
                self._size_sub.setCurrentText(_("max"))
                self._size_w.setText(str(f.max_size[0]))
                self._size_h.setText(str(f.max_size[1]))
        elif isinstance(f, ModelFilter):
            self._type_combo.setCurrentText(_("Model"))
            if f.models:
                self._model_text.setText(" ".join(f.models))
            self._model_mode_combo.setCurrentText(
                _("does not contain") if f.mode == "exclude" else _("contains")
            )
            self._model_match_any.setChecked(f.match_any)
            self._model_loras.setChecked(f.include_loras)


class FilterBuilderPanel(QFrame):
    """Editable list of filter conditions that constructs a CompareFilter tree."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Operator row
        op_row = QHBoxLayout()
        op_lbl = QLabel(_("Match:"))
        op_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        op_row.addWidget(op_lbl)

        self._op_combo = QComboBox()
        self._op_combo.addItems(["AND", "OR", "NOT"])
        self._op_combo.setFixedWidth(80)
        op_row.addWidget(self._op_combo)
        op_row.addStretch()
        outer.addLayout(op_row)

        # Scrollable rows area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(80)

        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()
        scroll.setWidget(self._rows_widget)
        outer.addWidget(scroll, 1)

        # Add-condition button
        add_btn = QPushButton(_("+ Add condition"))
        add_btn.setFixedWidth(150)
        add_btn.clicked.connect(lambda: self._add_row())
        outer.addWidget(add_btn)

        self._rows: List[_FilterRow] = []

    # ------------------------------------------------------------------
    def _add_row(self, filter_obj: Optional[CompareFilter] = None) -> None:
        row = _FilterRow(on_remove=self._remove_row, parent=self._rows_widget)
        if filter_obj is not None:
            row.set_filter(filter_obj)
        # Insert before the trailing stretch item
        insert_pos = max(0, self._rows_layout.count() - 1)
        self._rows_layout.insertWidget(insert_pos, row)
        self._rows.append(row)

    def _remove_row(self, row: _FilterRow) -> None:
        if row in self._rows:
            self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    # ------------------------------------------------------------------
    def get_filter(self) -> Optional[CompareFilter]:
        """Build and return the current filter tree, or None if empty."""
        active = [r.get_filter() for r in self._rows]
        active = [f for f in active if f is not None and f.is_active()]
        if not active:
            return None
        if len(active) == 1:
            return active[0]
        op = FilterOperator[self._op_combo.currentText()]
        return CompareFilterGroup(operator=op, filters=active)

    def set_filter(self, f: Optional[CompareFilter]) -> None:
        """Populate rows from an existing filter tree."""
        for row in list(self._rows):
            self._rows_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

        if f is None:
            return
        if isinstance(f, CompareFilterGroup):
            self._op_combo.setCurrentText(f.operator.name)
            for child in f.filters:
                if isinstance(child, (SizeFilter, ModelFilter)):
                    self._add_row(child)
        elif isinstance(f, (SizeFilter, ModelFilter)):
            self._add_row(f)

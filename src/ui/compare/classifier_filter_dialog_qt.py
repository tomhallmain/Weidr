"""
Classifier-filter dialog for the compare settings' filter panel.

Modal dialog that edits one ClassifierFilter: classifier domain and model,
category selection (explicit categories or the model's positive groups),
include/exclude, an optional minimum confidence and the frame-sampling
ratios for video/GIF/PDF/ePub. The result only
replaces the filter row's value; CompareSettingsWindow's Apply hands it to
the CompareManager.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QRadioButton, QVBoxLayout,
    QWidget,
)

from compare.classifier_categories import model_strategy_positive_categories
from compare.compare_filters import (
    CLASSIFIER_DOMAIN_AUDIO, CLASSIFIER_DOMAIN_IMAGE,
    SELECTION_MODEL_STRATEGY, SELECTION_SELECTED_CATEGORIES,
    ClassifierFilter,
)
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.translations import _


def _model_configs(domain: str) -> list:
    if domain == CLASSIFIER_DOMAIN_AUDIO:
        from image.audio_classifier_manager import audio_classifier_manager
        return audio_classifier_manager.get_model_configs()
    from image.image_classifier_manager import image_classifier_manager
    return image_classifier_manager.get_model_configs()


def describe_classifier_filter(f: Optional[ClassifierFilter]) -> str:
    """One-line summary of *f* for the filter panel row."""
    if f is None or not f.is_active():
        return _("(not configured)")
    if f.selection_mode == SELECTION_MODEL_STRATEGY:
        selection = _("model strategy")
    else:
        selection = ", ".join(f.categories or [])
    if f.mode == "exclude":
        text = _("{0}: exclude {1}").format(f.classifier_name, selection)
    else:
        text = _("{0}: only {1}").format(f.classifier_name, selection)
    if f.min_confidence > 0:
        text += " " + _("(min. {0:.2f})").format(f.min_confidence)
    return text


class ClassifierFilterDialog(SmartDialog):
    """Modal dialog for editing a ClassifierFilter."""

    def __init__(self, parent: QWidget, initial: Optional[ClassifierFilter] = None) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Classifier Filter"),
            geometry="560x560",
            center=True,
        )
        self._result: Optional[ClassifierFilter] = None
        # Categories from *initial* to pre-check when its model is shown.
        self._pending_categories: List[str] = list(initial.categories or []) if initial else []

        self._build_ui()
        self._load_initial(initial)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.reject)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)

        def lbl(text: str) -> QLabel:
            l = QLabel(text)
            l.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            return l

        row = 0
        grid.addWidget(lbl(_("Classifier type:")), row, 0)
        self._domain_combo = QComboBox()
        self._domain_combo.addItem(_("Image classifier"), CLASSIFIER_DOMAIN_IMAGE)
        self._domain_combo.addItem(_("Audio classifier"), CLASSIFIER_DOMAIN_AUDIO)
        self._domain_combo.currentIndexChanged.connect(self._on_domain_changed)
        grid.addWidget(self._domain_combo, row, 1)
        row += 1

        grid.addWidget(lbl(_("Model:")), row, 0)
        self._model_combo = QComboBox()
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        grid.addWidget(self._model_combo, row, 1)
        row += 1

        grid.addWidget(lbl(_("Files:")), row, 0)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem(_("Only files in the selected categories"), "include")
        self._mode_combo.addItem(_("Exclude files in the selected categories"), "exclude")
        grid.addWidget(self._mode_combo, row, 1)
        row += 1

        grid.addWidget(lbl(_("Minimum confidence:")), row, 0)
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setDecimals(2)
        self._confidence_spin.setSpecialValueText(_("off"))
        grid.addWidget(self._confidence_spin, row, 1)
        row += 1

        # Video/GIF/PDF/ePub sampling, as in classifier actions.
        def ratio_spin(tooltip: str) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setToolTip(tooltip)
            return spin

        self._sample_ratio_lbl = lbl(_("Sample ratio:"))
        grid.addWidget(self._sample_ratio_lbl, row, 0)
        self._sample_ratio_spin = ratio_spin(
            _("Share of frames or pages sampled from videos, GIFs, PDFs and ePubs."))
        grid.addWidget(self._sample_ratio_spin, row, 1)
        row += 1

        self._positive_ratio_lbl = lbl(_("Positive ratio:"))
        grid.addWidget(self._positive_ratio_lbl, row, 0)
        self._positive_ratio_spin = ratio_spin(
            _("Share of sampled frames or pages that must be in the selected "
              "categories for the file to count as in them."))
        grid.addWidget(self._positive_ratio_spin, row, 1)
        row += 1
        outer.addLayout(grid)

        sel_row = QHBoxLayout()
        self._selected_radio = QRadioButton(_("Selected categories"))
        self._strategy_radio = QRadioButton(_("Model strategy (positive groups)"))
        self._selected_radio.setChecked(True)
        self._selected_radio.toggled.connect(self._on_selection_mode_changed)
        sel_row.addWidget(self._selected_radio)
        sel_row.addWidget(self._strategy_radio)
        sel_row.addStretch()
        outer.addLayout(sel_row)

        self._category_list = QListWidget()
        self._category_list.itemChanged.connect(lambda _item: self._update_ok_enabled())
        outer.addWidget(self._category_list, 1)

        self._warning_lbl = QLabel()
        self._warning_lbl.setWordWrap(True)
        self._warning_lbl.setStyleSheet("color: #e06c75;")
        self._warning_lbl.setVisible(False)
        outer.addWidget(self._warning_lbl)

        hint = QLabel(_(
            "Videos, GIFs, PDFs and ePubs are sampled like in classifier actions. "
            "Files the classifier cannot handle (e.g. audio files for an image model) "
            "count as not in the selected categories."
        ))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        outer.addWidget(hint)

        btn_row = QHBoxLayout()
        self._ok_btn = QPushButton(_("OK"))
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(self._ok_btn)
        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _load_initial(self, initial: Optional[ClassifierFilter]) -> None:
        defaults = initial or ClassifierFilter()
        self._sample_ratio_spin.setValue(defaults.sample_ratio)
        self._positive_ratio_spin.setValue(defaults.positive_ratio)
        if initial is None:
            self._on_domain_changed()
            return
        self._domain_combo.blockSignals(True)
        idx = self._domain_combo.findData(initial.domain)
        self._domain_combo.setCurrentIndex(max(idx, 0))
        self._domain_combo.blockSignals(False)
        self._update_sampling_enabled()
        self._populate_models()

        # Signals stay blocked so _on_model_changed runs once, below, while
        # the initial categories are still pending.
        name = (initial.classifier_name or "").strip()
        self._model_combo.blockSignals(True)
        model_idx = self._model_combo.findData(name)
        if model_idx < 0 and name:
            # Keep an unregistered model visible so the user sees what is
            # configured; the compare run reports it as an error.
            self._model_combo.addItem(_("{0} (not registered)").format(name), name)
            model_idx = self._model_combo.count() - 1
        if model_idx >= 0:
            self._model_combo.setCurrentIndex(model_idx)
        self._model_combo.blockSignals(False)
        self._on_model_changed()

        self._mode_combo.setCurrentIndex(max(self._mode_combo.findData(initial.mode), 0))
        self._confidence_spin.setValue(float(initial.min_confidence or 0.0))
        if initial.selection_mode == SELECTION_MODEL_STRATEGY and self._strategy_radio.isEnabled():
            self._strategy_radio.setChecked(True)

    def _populate_models(self) -> None:
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for cfg in _model_configs(self._domain_combo.currentData()):
            self._model_combo.addItem(cfg.model_name, cfg.model_name)
        self._model_combo.blockSignals(False)

    def _current_config(self):
        name = self._model_combo.currentData()
        for cfg in _model_configs(self._domain_combo.currentData()):
            if cfg.model_name == name:
                return cfg
        return None

    # ------------------------------------------------------------------
    def _on_domain_changed(self, *_args) -> None:
        self._populate_models()
        self._on_model_changed()
        self._update_sampling_enabled()

    def _update_sampling_enabled(self) -> None:
        # Frame sampling only applies to image classifiers.
        enabled = self._domain_combo.currentData() == CLASSIFIER_DOMAIN_IMAGE
        for w in (self._sample_ratio_lbl, self._sample_ratio_spin,
                  self._positive_ratio_lbl, self._positive_ratio_spin):
            w.setEnabled(enabled)

    def _on_model_changed(self, *_args) -> None:
        cfg = self._current_config()
        categories = list(cfg.model_categories) if cfg else []
        positives = model_strategy_positive_categories(cfg.positive_groups if cfg else None)

        pending = self._pending_categories
        self._pending_categories = []
        dropped = [c for c in pending if c not in categories]

        self._category_list.blockSignals(True)
        self._category_list.clear()
        for cat in categories:
            item = QListWidgetItem(cat)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if cat in pending else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, cat in positives)
            self._category_list.addItem(item)
        self._category_list.blockSignals(False)

        self._strategy_radio.setEnabled(bool(positives))
        if not positives and self._strategy_radio.isChecked():
            self._selected_radio.setChecked(True)

        if cfg is None and self._model_combo.count() == 0:
            self._show_warning(_("No classifier models of this type are configured."))
        elif cfg is None:
            self._show_warning(_("This model is not registered; the compare run will fail until it is."))
        elif dropped:
            self._show_warning(
                _("These categories are no longer part of the model and were removed: {0}")
                .format(", ".join(dropped)))
        else:
            self._show_warning(None)
        self._on_selection_mode_changed()

    def _on_selection_mode_changed(self, *_args) -> None:
        strategy = self._strategy_radio.isChecked()
        self._category_list.blockSignals(True)
        for i in range(self._category_list.count()):
            item = self._category_list.item(i)
            if strategy:
                is_positive = bool(item.data(Qt.ItemDataRole.UserRole))
                item.setCheckState(Qt.CheckState.Checked if is_positive else Qt.CheckState.Unchecked)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
        self._category_list.blockSignals(False)
        self._update_ok_enabled()

    def _show_warning(self, text: Optional[str]) -> None:
        self._warning_lbl.setText(text or "")
        self._warning_lbl.setVisible(bool(text))

    def _checked_categories(self) -> List[str]:
        return [
            self._category_list.item(i).text()
            for i in range(self._category_list.count())
            if self._category_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _update_ok_enabled(self) -> None:
        has_model = bool(self._model_combo.currentData())
        has_selection = self._strategy_radio.isChecked() or bool(self._checked_categories())
        self._ok_btn.setEnabled(has_model and has_selection)

    # ------------------------------------------------------------------
    def _on_ok(self) -> None:
        strategy = self._strategy_radio.isChecked()
        self._result = ClassifierFilter(
            classifier_name=self._model_combo.currentData() or "",
            domain=self._domain_combo.currentData(),
            selection_mode=SELECTION_MODEL_STRATEGY if strategy else SELECTION_SELECTED_CATEGORIES,
            categories=[] if strategy else self._checked_categories(),
            mode=self._mode_combo.currentData(),
            min_confidence=float(self._confidence_spin.value()),
            sample_ratio=float(self._sample_ratio_spin.value()),
            positive_ratio=float(self._positive_ratio_spin.value()),
        )
        self.accept()

    def get_result(self) -> Optional[ClassifierFilter]:
        """The edited filter, or None if the dialog was cancelled."""
        return self._result

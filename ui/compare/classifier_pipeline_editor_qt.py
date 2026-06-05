"""
Phase 4b + 4c: ClassifierPipelineEditorDialog

Modal SmartDialog for creating and editing ClassifierPipeline instances.
Launched from ClassifierPipelinesTab on New / Edit.

Layout:
  Top     — pipeline-level fields (name, active, type, profile, default actions)
  Middle  — QSplitter: node list (left) | node detail editor (right)
  Bottom  — read-only flow preview + Save / Cancel
"""

from __future__ import annotations

import copy
import os
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    ClassifierRankCondition,
    CompositeCondition,
    EmbeddingCondition,
    LookaheadCondition,
    NodeOutcome,
    NodeResultCondition,
    OutcomeType,
    PipelineNode,
    PrevalidationPipeline,
    PromptCondition,
    PrototypeCondition,
)
from files.directory_profile import DirectoryProfile
from lib.multi_display_qt import SmartDialog
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.constants import ClassifierActionType
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("classifier_pipeline_editor_qt")


# ---------------------------------------------------------------------------
# Helpers shared across panels
# ---------------------------------------------------------------------------

_CONDITION_ENTRIES = [
    ("embedding",       _("Embedding")),
    ("classifier_rank", _("Classifier Rank")),
    ("prototype",       _("Prototype")),
    ("prompt",          _("Prompt")),
    ("lookahead",       _("Lookahead")),
    ("node_result",     _("Prior Node Result")),
    ("composite",       _("Composite")),
]
_CONDITION_TYPES   = [k for k, _ in _CONDITION_ENTRIES]
_CONDITION_LABELS  = [v for _, v in _CONDITION_ENTRIES]

# Sub-condition types: composites cannot contain composites in the UI
_SUB_CONDITION_ENTRIES = _CONDITION_ENTRIES[:-1]
_SUB_CONDITION_TYPES   = [k for k, _ in _SUB_CONDITION_ENTRIES]
_SUB_CONDITION_LABELS  = [v for _, v in _SUB_CONDITION_ENTRIES]

_ACTION_OPTIONS = [_("(none)")] + [at.value for at in ClassifierActionType]
_OUTCOME_OPTIONS = [ot.value for ot in OutcomeType]
_FG = AppStyle.FG_COLOR
_BG = AppStyle.BG_COLOR


def _label(text: str, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    style = f"color: {_FG};"
    if bold:
        style += " font-weight: bold;"
    lbl.setStyleSheet(style)
    return lbl


def _action_from_text(text: str) -> Optional[ClassifierActionType]:
    if not text or text == _("(none)"):
        return None
    return ClassifierActionType.get_action(text)


# ---------------------------------------------------------------------------
# _StringListEditor — reusable list-with-add/remove widget
# ---------------------------------------------------------------------------

class _StringListEditor(QWidget):
    """QListWidget with a text entry and Add / Remove buttons below."""

    def __init__(self, placeholder: str = "", on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._list = QListWidget()
        self._list.setStyleSheet(f"color: {_FG}; background: {_BG};")
        self._list.setMaximumHeight(90)
        layout.addWidget(self._list)

        entry_row = QHBoxLayout()
        self._entry = QLineEdit()
        self._entry.setPlaceholderText(placeholder)
        self._entry.returnPressed.connect(self._add_item)
        entry_row.addWidget(self._entry, 1)

        add_btn = QPushButton(_("Add"))
        add_btn.setFixedWidth(50)
        add_btn.clicked.connect(self._add_item)
        entry_row.addWidget(add_btn)

        remove_btn = QPushButton(_("Remove"))
        remove_btn.setFixedWidth(65)
        remove_btn.clicked.connect(self._remove_selected)
        entry_row.addWidget(remove_btn)

        layout.addLayout(entry_row)

    def _add_item(self) -> None:
        text = self._entry.text().strip()
        if text:
            self._list.addItem(text)
            self._entry.clear()
            self._on_changed()

    def _remove_selected(self) -> None:
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
        self._on_changed()

    def set_items(self, items: list) -> None:
        self._list.clear()
        for it in items:
            self._list.addItem(str(it))

    def get_items(self) -> list:
        return [self._list.item(i).text() for i in range(self._list.count())]


# ---------------------------------------------------------------------------
# Condition panels
# ---------------------------------------------------------------------------

class _EmbeddingPanel(QWidget):
    condition_type = "embedding"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(4)

        self._positives = _StringListEditor(_("positive text"), on_changed=self._on_changed)
        form.addRow(_("Positives:"), self._positives)

        self._negatives = _StringListEditor(_("negative text"), on_changed=self._on_changed)
        form.addRow(_("Negatives:"), self._negatives)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0, 1.0)
        self._threshold.setSingleStep(0.01)
        self._threshold.setDecimals(3)
        self._threshold.setValue(0.23)
        self._threshold.valueChanged.connect(self._on_changed)
        form.addRow(_("Threshold:"), self._threshold)

    def load(self, condition) -> None:
        if isinstance(condition, EmbeddingCondition):
            self._positives.set_items(condition.positives)
            self._negatives.set_items(condition.negatives)
            self._threshold.setValue(condition.threshold)
        else:
            self._positives.set_items([])
            self._negatives.set_items([])
            self._threshold.setValue(0.23)

    def get_condition(self) -> EmbeddingCondition:
        return EmbeddingCondition(
            positives=self._positives.get_items(),
            negatives=self._negatives.get_items(),
            threshold=self._threshold.value(),
        )


class _ClassifierRankPanel(QWidget):
    condition_type = "classifier_rank"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(4)

        self._classifier_combo = QComboBox()
        self._populate_classifiers()
        self._classifier_combo.currentIndexChanged.connect(self._on_changed)
        form.addRow(_("Classifier:"), self._classifier_combo)

        self._categories = _StringListEditor(_("category name"), on_changed=self._on_changed)
        form.addRow(_("Categories:"), self._categories)

        rank_row = QHBoxLayout()
        self._min_rank = QSpinBox()
        self._min_rank.setRange(1, 20)
        self._min_rank.setValue(1)
        self._min_rank.valueChanged.connect(self._on_changed)
        rank_row.addWidget(_label(_("Min:")))
        rank_row.addWidget(self._min_rank)
        rank_row.addSpacing(8)
        self._max_rank = QSpinBox()
        self._max_rank.setRange(1, 20)
        self._max_rank.setValue(1)
        self._max_rank.valueChanged.connect(self._on_changed)
        rank_row.addWidget(_label(_("Max:")))
        rank_row.addWidget(self._max_rank)
        rank_row.addStretch()
        form.addRow(_("Rank range:"), rank_row)

        self._min_confidence = QDoubleSpinBox()
        self._min_confidence.setRange(0.0, 1.0)
        self._min_confidence.setSingleStep(0.01)
        self._min_confidence.setDecimals(3)
        self._min_confidence.setValue(0.0)
        self._min_confidence.valueChanged.connect(self._on_changed)
        form.addRow(_("Min confidence:"), self._min_confidence)

    def _populate_classifiers(self) -> None:
        self._classifier_combo.clear()
        try:
            from image.image_classifier_manager import image_classifier_manager
            names = list(image_classifier_manager.classifier_metadata.keys())
        except Exception:
            names = []
        if names:
            self._classifier_combo.addItems(names)
        else:
            self._classifier_combo.addItem(_("(no classifiers)"))

    def load(self, condition) -> None:
        if isinstance(condition, ClassifierRankCondition):
            idx = self._classifier_combo.findText(condition.classifier_name)
            if idx >= 0:
                self._classifier_combo.setCurrentIndex(idx)
            self._categories.set_items(condition.categories)
            self._min_rank.setValue(condition.min_rank)
            self._max_rank.setValue(condition.max_rank)
            self._min_confidence.setValue(condition.min_confidence)
        else:
            self._categories.set_items([])
            self._min_rank.setValue(1)
            self._max_rank.setValue(1)
            self._min_confidence.setValue(0.0)

    def get_condition(self) -> ClassifierRankCondition:
        return ClassifierRankCondition(
            classifier_name=self._classifier_combo.currentText(),
            categories=self._categories.get_items(),
            min_rank=self._min_rank.value(),
            max_rank=self._max_rank.value(),
            min_confidence=self._min_confidence.value(),
        )


class _PrototypePanel(QWidget):
    condition_type = "prototype"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(4)

        self._pos_dir = QLineEdit()
        self._pos_dir.setPlaceholderText(_("Positive prototype directory"))
        self._pos_dir.textChanged.connect(self._on_changed)
        pos_row = QHBoxLayout()
        pos_row.addWidget(self._pos_dir, 1)
        pos_browse = QPushButton(_("Browse…"))
        pos_browse.clicked.connect(lambda: self._browse(self._pos_dir))
        pos_row.addWidget(pos_browse)
        form.addRow(_("Positive dir:"), pos_row)

        self._neg_dir = QLineEdit()
        self._neg_dir.setPlaceholderText(_("Negative prototype directory (optional)"))
        self._neg_dir.textChanged.connect(self._on_changed)
        neg_row = QHBoxLayout()
        neg_row.addWidget(self._neg_dir, 1)
        neg_browse = QPushButton(_("Browse…"))
        neg_browse.clicked.connect(lambda: self._browse(self._neg_dir))
        neg_row.addWidget(neg_browse)
        form.addRow(_("Negative dir:"), neg_row)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(-1.0, 1.0)
        self._threshold.setSingleStep(0.01)
        self._threshold.setDecimals(3)
        self._threshold.setValue(0.5)
        self._threshold.valueChanged.connect(self._on_changed)
        form.addRow(_("Threshold:"), self._threshold)

        self._neg_lambda = QDoubleSpinBox()
        self._neg_lambda.setRange(0.0, 5.0)
        self._neg_lambda.setSingleStep(0.05)
        self._neg_lambda.setDecimals(2)
        self._neg_lambda.setValue(0.5)
        self._neg_lambda.valueChanged.connect(self._on_changed)
        form.addRow(_("Negative lambda:"), self._neg_lambda)

    def _browse(self, line_edit: QLineEdit) -> None:
        current = line_edit.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, _("Select directory"), current)
        if d:
            line_edit.setText(d)

    def load(self, condition) -> None:
        if isinstance(condition, PrototypeCondition):
            self._pos_dir.setText(condition.prototype_directory or "")
            self._neg_dir.setText(condition.negative_prototype_directory or "")
            self._threshold.setValue(condition.threshold)
            self._neg_lambda.setValue(condition.negative_lambda)
        else:
            self._pos_dir.setText("")
            self._neg_dir.setText("")
            self._threshold.setValue(0.5)
            self._neg_lambda.setValue(0.5)

    def get_condition(self) -> PrototypeCondition:
        return PrototypeCondition(
            prototype_directory=self._pos_dir.text().strip(),
            negative_prototype_directory=self._neg_dir.text().strip(),
            threshold=self._threshold.value(),
            negative_lambda=self._neg_lambda.value(),
        )


class _PromptPanel(QWidget):
    condition_type = "prompt"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        self._blacklist_cb = QCheckBox(_("Use global prompt blacklist"))
        self._blacklist_cb.stateChanged.connect(self._on_blacklist_toggled)
        layout.addWidget(self._blacklist_cb)

        prompts_lbl = _label(_("Prompts to match (ignored when blacklist is checked):"))
        layout.addWidget(prompts_lbl)

        self._prompts = _StringListEditor(_("prompt keyword"), on_changed=self._on_changed)
        layout.addWidget(self._prompts)

    def _on_blacklist_toggled(self, state: int) -> None:
        self._prompts.setEnabled(not bool(state))
        self._on_changed()

    def load(self, condition) -> None:
        if isinstance(condition, PromptCondition):
            self._blacklist_cb.setChecked(condition.use_blacklist)
            self._prompts.set_items(condition.prompts or [])
        else:
            self._blacklist_cb.setChecked(False)
            self._prompts.set_items([])
        self._prompts.setEnabled(not self._blacklist_cb.isChecked())

    def get_condition(self) -> PromptCondition:
        return PromptCondition(
            prompts=self._prompts.get_items(),
            use_blacklist=self._blacklist_cb.isChecked(),
        )


class _LookaheadPanel(QWidget):
    condition_type = "lookahead"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)

        self._combo = QComboBox()
        self._populate()
        self._combo.currentIndexChanged.connect(self._on_changed)
        form.addRow(_("Lookahead:"), self._combo)

    def _populate(self) -> None:
        self._combo.clear()
        try:
            from compare.lookahead import Lookahead
            names = [lh.name for lh in Lookahead.lookaheads]
        except Exception:
            names = []
        if names:
            self._combo.addItems(names)
        else:
            self._combo.addItem(_("(no lookaheads)"))

    def load(self, condition) -> None:
        self._populate()
        if isinstance(condition, LookaheadCondition):
            idx = self._combo.findText(condition.lookahead_name)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)

    def get_condition(self) -> LookaheadCondition:
        return LookaheadCondition(lookahead_name=self._combo.currentText())


class _NodeResultPanel(QWidget):
    condition_type = "node_result"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)

        self._node_combo = QComboBox()
        self._node_combo.currentIndexChanged.connect(self._on_changed)
        form.addRow(_("Prior node:"), self._node_combo)

        self._expected_cb = QCheckBox(_("Expected result: matched"))
        self._expected_cb.setChecked(True)
        self._expected_cb.stateChanged.connect(self._on_changed)
        form.addRow("", self._expected_cb)

        info = _label(
            _("Note: only nodes defined earlier in the pipeline are available.")
        )
        info.setWordWrap(True)
        form.addRow("", info)

    def set_prior_nodes(self, names: list) -> None:
        current = self._node_combo.currentText()
        self._node_combo.blockSignals(True)
        self._node_combo.clear()
        if names:
            self._node_combo.addItems(names)
            if current in names:
                self._node_combo.setCurrentText(current)
        else:
            self._node_combo.addItem(_("(no prior nodes)"))
        self._node_combo.blockSignals(False)

    def load(self, condition) -> None:
        if isinstance(condition, NodeResultCondition):
            idx = self._node_combo.findText(condition.node_name)
            if idx >= 0:
                self._node_combo.setCurrentIndex(idx)
            self._expected_cb.setChecked(condition.expected_result)
        else:
            self._expected_cb.setChecked(True)

    def get_condition(self) -> NodeResultCondition:
        return NodeResultCondition(
            node_name=self._node_combo.currentText(),
            expected_result=self._expected_cb.isChecked(),
        )


# ---------------------------------------------------------------------------
# Composite panel (Phase 4c) — inline list of sub-conditions
# ---------------------------------------------------------------------------

class _SubCondRow(QWidget):
    """A single sub-condition row inside the composite panel."""

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        self._prior_nodes: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        top_row = QHBoxLayout()
        type_lbl = _label(_("Type:"))
        top_row.addWidget(type_lbl)

        self._type_combo = QComboBox()
        self._type_combo.addItems(_SUB_CONDITION_LABELS)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        top_row.addWidget(self._type_combo, 1)

        root.addLayout(top_row)

        # Stacked panels — reuse the same panel classes, no composites
        self._stack = QStackedWidget()
        self._panels = [
            _EmbeddingPanel(on_changed=self._on_changed),
            _ClassifierRankPanel(on_changed=self._on_changed),
            _PrototypePanel(on_changed=self._on_changed),
            _PromptPanel(on_changed=self._on_changed),
            _LookaheadPanel(on_changed=self._on_changed),
            _NodeResultPanel(on_changed=self._on_changed),
        ]
        for p in self._panels:
            self._stack.addWidget(p)
        root.addWidget(self._stack)

    def _on_type_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        self._stack.currentWidget().load(None)
        if hasattr(self._stack.currentWidget(), "set_prior_nodes"):
            self._stack.currentWidget().set_prior_nodes(self._prior_nodes)
        self._on_changed()

    def set_prior_nodes(self, names: list) -> None:
        self._prior_nodes = names
        for p in self._panels:
            if hasattr(p, "set_prior_nodes"):
                p.set_prior_nodes(names)

    def load(self, condition) -> None:
        ctype = getattr(condition, "condition_type", "embedding")
        idx = _SUB_CONDITION_TYPES.index(ctype) if ctype in _SUB_CONDITION_TYPES else 0
        self._type_combo.blockSignals(True)
        self._type_combo.setCurrentIndex(idx)
        self._type_combo.blockSignals(False)
        self._stack.setCurrentIndex(idx)
        self._panels[idx].load(condition)

    def get_condition(self):
        idx = self._stack.currentIndex()
        return self._panels[idx].get_condition()


class _CompositePanel(QWidget):
    condition_type = "composite"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        self._prior_nodes: list = []
        # Each entry: (container_widget, _SubCondRow)
        self._row_data: list[tuple[QWidget, _SubCondRow]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 4)
        root.setSpacing(4)

        op_row = QHBoxLayout()
        op_row.addWidget(_label(_("Operator:")))
        self._op_combo = QComboBox()
        self._op_combo.addItems(["AND", "OR", "NOT", "XOR"])
        self._op_combo.currentIndexChanged.connect(self._on_changed)
        op_row.addWidget(self._op_combo)
        op_row.addStretch()
        root.addLayout(op_row)

        op_note = _label(
            _("AND/OR: ≥ 2 sub-conditions · NOT: exactly 1 · XOR: exactly 2")
        )
        op_note.setWordWrap(True)
        root.addWidget(op_note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: 1px solid {_FG}; background: {_BG}; }}")
        scroll.setMinimumHeight(180)
        self._sub_widget = QWidget()
        self._sub_layout = QVBoxLayout(self._sub_widget)
        self._sub_layout.setContentsMargins(4, 4, 4, 4)
        self._sub_layout.setSpacing(6)
        self._sub_layout.addStretch()
        scroll.setWidget(self._sub_widget)
        root.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        add_sub = QPushButton(_("Add sub-condition"))
        add_sub.clicked.connect(lambda: self._add_sub())
        btn_row.addWidget(add_sub)
        remove_sub = QPushButton(_("Remove last"))
        remove_sub.clicked.connect(self._remove_last)
        btn_row.addWidget(remove_sub)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _add_sub(self, condition=None) -> None:
        container = QWidget()
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)

        sub_row = _SubCondRow(on_changed=self._on_changed)
        sub_row.set_prior_nodes(self._prior_nodes)
        if condition is not None:
            sub_row.load(condition)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.clicked.connect(lambda: self._remove_row(container, sub_row))

        row_layout.addWidget(sub_row, 1)
        row_layout.addWidget(remove_btn)

        insert_pos = self._sub_layout.count() - 1
        self._sub_layout.insertWidget(insert_pos, container)
        self._row_data.append((container, sub_row))
        self._on_changed()

    def _remove_row(self, container: QWidget, sub_row: _SubCondRow) -> None:
        self._row_data = [(c, r) for c, r in self._row_data if r is not sub_row]
        container.deleteLater()
        self._on_changed()

    def _remove_last(self) -> None:
        if not self._row_data:
            return
        container, _ = self._row_data.pop()
        container.deleteLater()
        self._on_changed()

    def set_prior_nodes(self, names: list) -> None:
        self._prior_nodes = names
        for _, sub_row in self._row_data:
            sub_row.set_prior_nodes(names)

    def load(self, condition) -> None:
        while self._row_data:
            container, _ = self._row_data.pop()
            container.deleteLater()
        # Also sweep out any orphaned widgets above the stretch
        while self._sub_layout.count() > 1:
            item = self._sub_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if isinstance(condition, CompositeCondition):
            self._op_combo.setCurrentText(condition.operator)
            for sub in condition.sub_conditions:
                self._add_sub(sub)
        else:
            self._op_combo.setCurrentIndex(0)

    def get_condition(self) -> CompositeCondition:
        return CompositeCondition(
            operator=self._op_combo.currentText(),
            sub_conditions=[r.get_condition() for _, r in self._row_data],
        )


# ---------------------------------------------------------------------------
# Outcome editor widget
# ---------------------------------------------------------------------------

class _OutcomeEditorWidget(QGroupBox):
    """
    Compact editor for a single NodeOutcome (on_match or on_no_match).

    Shows: outcome type combo + dependent field:
      GOTO    → target node combo
      EXECUTE → action type combo + modifier line edit (for MOVE/COPY)
    """

    def __init__(self, label: str, on_changed: Callable = None, parent=None):
        super().__init__(label, parent)
        self.setStyleSheet(f"QGroupBox {{ color: {_FG}; }}")
        self._on_changed = on_changed or (lambda: None)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 6)
        layout.setSpacing(4)

        type_row = QHBoxLayout()
        type_row.addWidget(_label(_("Outcome:")))
        self._type_combo = QComboBox()
        self._type_combo.addItems(_OUTCOME_OPTIONS)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo, 1)
        layout.addLayout(type_row)

        # GOTO: node combo
        self._goto_row = QHBoxLayout()
        self._goto_row.addWidget(_label(_("Jump to node:")))
        self._goto_combo = QComboBox()
        self._goto_combo.currentIndexChanged.connect(self._on_changed)
        self._goto_row.addWidget(self._goto_combo, 1)
        layout.addLayout(self._goto_row)

        # EXECUTE: action combo + modifier
        self._exec_row = QVBoxLayout()
        action_row = QHBoxLayout()
        action_row.addWidget(_label(_("Action:")))
        self._action_combo = QComboBox()
        self._action_combo.addItems([at.value for at in ClassifierActionType])
        self._action_combo.currentIndexChanged.connect(self._on_action_changed)
        action_row.addWidget(self._action_combo, 1)
        self._exec_row.addLayout(action_row)

        self._modifier_row = QHBoxLayout()
        self._modifier_row.addWidget(_label(_("Target dir:")))
        self._modifier_edit = QLineEdit()
        self._modifier_edit.setPlaceholderText(_("Target directory for MOVE/COPY"))
        self._modifier_edit.textChanged.connect(self._on_changed)
        self._modifier_row.addWidget(self._modifier_edit, 1)
        mod_browse = QPushButton(_("Browse…"))
        mod_browse.clicked.connect(self._browse_modifier)
        self._modifier_row.addWidget(mod_browse)
        self._exec_row.addLayout(self._modifier_row)
        layout.addLayout(self._exec_row)

        self._update_dependent_visibility(self._type_combo.currentText())

    def _browse_modifier(self) -> None:
        current = self._modifier_edit.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, _("Select target directory"), current)
        if d:
            self._modifier_edit.setText(d)

    def _on_type_changed(self, _: int) -> None:
        self._update_dependent_visibility(self._type_combo.currentText())
        self._on_changed()

    def _on_action_changed(self, _: int) -> None:
        action_val = self._action_combo.currentText()
        needs_dir = action_val in (ClassifierActionType.MOVE.value, ClassifierActionType.COPY.value)
        for i in range(self._modifier_row.count()):
            item = self._modifier_row.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(needs_dir)
        self._on_changed()

    def _update_dependent_visibility(self, outcome_val: str) -> None:
        is_goto = outcome_val == OutcomeType.GOTO.value
        is_exec = outcome_val == OutcomeType.EXECUTE.value
        self._set_layout_visible(self._goto_row, is_goto)
        self._set_layout_visible(self._exec_row, is_exec)
        if is_exec:
            self._on_action_changed(0)

    @staticmethod
    def _set_layout_visible(layout, visible: bool) -> None:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item:
                if item.widget():
                    item.widget().setVisible(visible)
                elif item.layout():
                    _OutcomeEditorWidget._set_layout_visible(item.layout(), visible)

    def set_later_nodes(self, names: list) -> None:
        current = self._goto_combo.currentText()
        self._goto_combo.blockSignals(True)
        self._goto_combo.clear()
        if names:
            self._goto_combo.addItems(names)
            if current in names:
                self._goto_combo.setCurrentText(current)
        else:
            self._goto_combo.addItem(_("(no later nodes)"))
        self._goto_combo.blockSignals(False)

    def load(self, outcome: NodeOutcome, later_nodes: list) -> None:
        self.set_later_nodes(later_nodes)
        ot = outcome.outcome_type
        val = ot.value if isinstance(ot, OutcomeType) else str(ot)
        idx = self._type_combo.findText(val)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        if ot == OutcomeType.GOTO and outcome.target_node:
            idx2 = self._goto_combo.findText(outcome.target_node)
            if idx2 >= 0:
                self._goto_combo.setCurrentIndex(idx2)
        if ot == OutcomeType.EXECUTE:
            if outcome.action_type:
                idx3 = self._action_combo.findText(outcome.action_type.value)
                if idx3 >= 0:
                    self._action_combo.setCurrentIndex(idx3)
            self._modifier_edit.setText(outcome.action_modifier or "")

    def get_outcome(self) -> NodeOutcome:
        ot_val = self._type_combo.currentText()
        ot = OutcomeType(ot_val)
        if ot == OutcomeType.GOTO:
            return NodeOutcome(
                outcome_type=ot,
                target_node=self._goto_combo.currentText(),
            )
        if ot == OutcomeType.EXECUTE:
            action_val = self._action_combo.currentText()
            return NodeOutcome(
                outcome_type=ot,
                action_type=ClassifierActionType.get_action(action_val),
                action_modifier=self._modifier_edit.text().strip() or None,
            )
        return NodeOutcome(outcome_type=ot)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class ClassifierPipelineEditorDialog(SmartDialog):
    """Create or edit a ClassifierPipeline."""

    def __init__(
        self,
        parent: QWidget,
        app_actions,
        refresh_callback: Callable,
        pipeline: Optional[ClassifierPipeline] = None,
    ) -> None:
        self._is_edit = pipeline is not None
        self._original_name = pipeline.name if self._is_edit else None
        self._pipeline = copy.deepcopy(pipeline) if pipeline else ClassifierPipeline()
        self._app_actions = app_actions
        self._refresh_callback = refresh_callback
        self._current_node_idx: Optional[int] = None
        self._suppress_refresh = False

        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Edit Pipeline") if self._is_edit else _("New Pipeline"),
            geometry="1100x820",
        )

        self._build_ui()
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        root.addWidget(self._build_pipeline_group())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_node_list_pane())
        splitter.addWidget(self._build_node_editor_pane())
        splitter.setSizes([260, 840])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_flow_preview_group())

        btn_row = QHBoxLayout()
        save_btn = QPushButton(_("Save"))
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _build_pipeline_group(self) -> QGroupBox:
        box = QGroupBox(_("Pipeline"))
        box.setStyleSheet(f"QGroupBox {{ color: {_FG}; }}")
        form = QFormLayout(box)
        form.setContentsMargins(8, 14, 8, 8)
        form.setSpacing(6)

        p = self._pipeline

        self._name_edit = QLineEdit(p.name)
        self._name_edit.textChanged.connect(self._on_field_changed)
        form.addRow(_("Name:"), self._name_edit)

        self._desc_edit = QLineEdit(getattr(p, "description", "") or "")
        form.addRow(_("Description:"), self._desc_edit)

        self._active_cb = QCheckBox()
        self._active_cb.setChecked(p.is_active)
        form.addRow(_("Active:"), self._active_cb)

        self._type_combo = QComboBox()
        self._type_combo.addItems([_("General"), _("Prevalidation")])
        if isinstance(p, PrevalidationPipeline):
            self._type_combo.setCurrentIndex(1)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        form.addRow(_("Type:"), self._type_combo)

        profile_names = [pr.name for pr in DirectoryProfile.directory_profiles]
        self._profile_combo = QComboBox()
        self._profile_combo.addItems(profile_names or [_("(no profiles)")])
        if isinstance(p, PrevalidationPipeline) and p.profile_name:
            idx = self._profile_combo.findText(p.profile_name)
            if idx >= 0:
                self._profile_combo.setCurrentIndex(idx)
        self._profile_lbl = QLabel(_("Profile:"))
        self._profile_lbl.setStyleSheet(f"color: {_FG};")
        form.addRow(self._profile_lbl, self._profile_combo)

        self._default_action_combo = QComboBox()
        self._default_action_combo.addItems(_ACTION_OPTIONS)
        if p.default_action:
            self._default_action_combo.setCurrentText(p.default_action.value)
        form.addRow(_("Default action:"), self._default_action_combo)

        self._default_reject_combo = QComboBox()
        self._default_reject_combo.addItems(_ACTION_OPTIONS)
        if p.default_reject_action:
            self._default_reject_combo.setCurrentText(p.default_reject_action.value)
        form.addRow(_("Default reject action:"), self._default_reject_combo)

        self._on_type_changed()
        return box

    def _build_node_list_pane(self) -> QWidget:
        pane = QWidget()
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(4)

        hdr = _label(_("Nodes"), bold=True)
        lay.addWidget(hdr)

        self._node_list = QListWidget()
        self._node_list.setStyleSheet(
            f"color: {_FG}; background: {_BG}; selection-background-color: #444;"
        )
        self._node_list.currentRowChanged.connect(self._on_node_selected)
        lay.addWidget(self._node_list, 1)

        btn_row = QHBoxLayout()
        for label, slot in [
            (_("Add"), self._add_node),
            (_("Remove"), self._remove_node),
            (_("↑"), self._move_node_up),
            (_("↓"), self._move_node_down),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

        self._rebuild_node_list()
        # Pre-select first node
        if self._pipeline.nodes:
            self._node_list.setCurrentRow(0)
        return pane

    def _build_node_editor_pane(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {_BG}; }}")

        self._node_editor_widget = QWidget()
        ned = QVBoxLayout(self._node_editor_widget)
        ned.setContentsMargins(4, 0, 0, 0)
        ned.setSpacing(8)

        ned.addWidget(_label(_("Node Editor"), bold=True))

        # Node name
        name_form = QFormLayout()
        name_form.setSpacing(4)
        self._node_name_edit = QLineEdit()
        self._node_name_edit.setPlaceholderText(_("unique node name"))
        self._node_name_edit.textChanged.connect(self._on_node_name_changed)
        name_form.addRow(_("Node name:"), self._node_name_edit)
        ned.addLayout(name_form)

        # Condition type
        ctype_form = QFormLayout()
        ctype_form.setSpacing(4)
        self._condition_type_combo = QComboBox()
        self._condition_type_combo.addItems(_CONDITION_LABELS)
        self._condition_type_combo.currentIndexChanged.connect(self._on_condition_type_changed)
        ctype_form.addRow(_("Condition type:"), self._condition_type_combo)
        ned.addLayout(ctype_form)

        # Condition panels (stacked)
        changed_cb = self._on_field_changed
        self._embedding_panel      = _EmbeddingPanel(on_changed=changed_cb)
        self._classifier_rank_panel = _ClassifierRankPanel(on_changed=changed_cb)
        self._prototype_panel      = _PrototypePanel(on_changed=changed_cb)
        self._prompt_panel         = _PromptPanel(on_changed=changed_cb)
        self._lookahead_panel      = _LookaheadPanel(on_changed=changed_cb)
        self._node_result_panel    = _NodeResultPanel(on_changed=changed_cb)
        self._composite_panel      = _CompositePanel(on_changed=changed_cb)

        self._cond_panels = [
            self._embedding_panel, self._classifier_rank_panel,
            self._prototype_panel, self._prompt_panel,
            self._lookahead_panel, self._node_result_panel,
            self._composite_panel,
        ]

        self._condition_stack = QStackedWidget()
        for p in self._cond_panels:
            self._condition_stack.addWidget(p)
        ned.addWidget(self._condition_stack)

        # Outcome editors
        self._on_match_editor    = _OutcomeEditorWidget(_("On match:"), on_changed=changed_cb)
        self._on_no_match_editor = _OutcomeEditorWidget(_("On no-match:"), on_changed=changed_cb)
        ned.addWidget(self._on_match_editor)
        ned.addWidget(self._on_no_match_editor)

        ned.addStretch()
        scroll.setWidget(self._node_editor_widget)
        self._node_editor_widget.setEnabled(False)
        return scroll

    def _build_flow_preview_group(self) -> QGroupBox:
        box = QGroupBox(_("Flow Preview"))
        box.setStyleSheet(f"QGroupBox {{ color: {_FG}; }}")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 14, 8, 8)

        self._flow_preview = QPlainTextEdit()
        self._flow_preview.setReadOnly(True)
        self._flow_preview.setMaximumHeight(140)
        self._flow_preview.setStyleSheet(
            f"color: {_FG}; background: {_BG}; font-family: monospace; font-size: 10pt;"
        )
        lay.addWidget(self._flow_preview)
        self._refresh_flow_preview()
        return box

    # ------------------------------------------------------------------
    # Pipeline-level callbacks
    # ------------------------------------------------------------------

    def _on_type_changed(self) -> None:
        is_pv = self._type_combo.currentIndex() == 1
        self._profile_lbl.setVisible(is_pv)
        self._profile_combo.setVisible(is_pv)

    def _on_field_changed(self) -> None:
        if not self._suppress_refresh:
            self._refresh_flow_preview()

    # ------------------------------------------------------------------
    # Node list
    # ------------------------------------------------------------------

    def _rebuild_node_list(self) -> None:
        row = self._node_list.currentRow()
        self._node_list.blockSignals(True)
        self._node_list.clear()
        for node in self._pipeline.nodes:
            self._node_list.addItem(self._node_label(node))
        self._node_list.blockSignals(False)
        target = row if 0 <= row < len(self._pipeline.nodes) else (
            0 if self._pipeline.nodes else -1
        )
        if target >= 0:
            self._node_list.setCurrentRow(target)

    def _node_label(self, node: PipelineNode) -> str:
        ctype = getattr(node.condition, "condition_type", "?")
        match_summary = (
            node.on_match.outcome_type.value
            if isinstance(node.on_match.outcome_type, OutcomeType)
            else str(node.on_match.outcome_type)
        )
        no_match_summary = (
            node.on_no_match.outcome_type.value
            if isinstance(node.on_no_match.outcome_type, OutcomeType)
            else str(node.on_no_match.outcome_type)
        )
        return f"{node.name}  [{ctype}]  ✓{match_summary} / ✗{no_match_summary}"

    def _on_node_selected(self, row: int) -> None:
        self._flush_node_to_model()
        if row < 0 or row >= len(self._pipeline.nodes):
            self._current_node_idx = None
            self._node_editor_widget.setEnabled(False)
            return
        self._current_node_idx = row
        self._node_editor_widget.setEnabled(True)
        self._load_node_to_editor(row)

    def _load_node_to_editor(self, idx: int) -> None:
        self._suppress_refresh = True
        node = self._pipeline.nodes[idx]
        prior = [n.name for n in self._pipeline.nodes[:idx]]
        later = [n.name for n in self._pipeline.nodes[idx + 1:]]

        self._node_name_edit.setText(node.name)

        ctype = getattr(node.condition, "condition_type", "embedding")
        ct_idx = _CONDITION_TYPES.index(ctype) if ctype in _CONDITION_TYPES else 0
        self._condition_type_combo.setCurrentIndex(ct_idx)
        self._condition_stack.setCurrentIndex(ct_idx)
        self._cond_panels[ct_idx].load(node.condition)

        self._node_result_panel.set_prior_nodes(prior)
        self._composite_panel.set_prior_nodes(prior)

        self._on_match_editor.load(node.on_match, later)
        self._on_no_match_editor.load(node.on_no_match, later)

        self._suppress_refresh = False
        self._refresh_flow_preview()

    def _flush_node_to_model(self) -> None:
        idx = self._current_node_idx
        if idx is None or idx >= len(self._pipeline.nodes):
            return
        node = self._pipeline.nodes[idx]
        name = self._node_name_edit.text().strip()
        if name:
            node.name = name
        try:
            node.condition = self._cond_panels[self._condition_stack.currentIndex()].get_condition()
        except Exception:
            pass
        try:
            node.on_match = self._on_match_editor.get_outcome()
        except Exception:
            pass
        try:
            node.on_no_match = self._on_no_match_editor.get_outcome()
        except Exception:
            pass

    def _add_node(self) -> None:
        self._flush_node_to_model()
        node = PipelineNode(name=f"node_{len(self._pipeline.nodes) + 1}")
        self._pipeline.nodes.append(node)
        self._rebuild_node_list()
        self._node_list.setCurrentRow(len(self._pipeline.nodes) - 1)
        self._refresh_flow_preview()

    def _remove_node(self) -> None:
        idx = self._node_list.currentRow()
        if idx < 0 or idx >= len(self._pipeline.nodes):
            return
        self._current_node_idx = None
        del self._pipeline.nodes[idx]
        self._rebuild_node_list()
        if not self._pipeline.nodes:
            self._node_editor_widget.setEnabled(False)
        self._refresh_flow_preview()

    def _move_node_up(self) -> None:
        self._flush_node_to_model()
        idx = self._node_list.currentRow()
        if idx <= 0:
            return
        nodes = self._pipeline.nodes
        nodes[idx], nodes[idx - 1] = nodes[idx - 1], nodes[idx]
        self._current_node_idx = idx - 1
        self._rebuild_node_list()
        self._node_list.setCurrentRow(idx - 1)
        self._refresh_flow_preview()

    def _move_node_down(self) -> None:
        self._flush_node_to_model()
        idx = self._node_list.currentRow()
        if idx < 0 or idx >= len(self._pipeline.nodes) - 1:
            return
        nodes = self._pipeline.nodes
        nodes[idx], nodes[idx + 1] = nodes[idx + 1], nodes[idx]
        self._current_node_idx = idx + 1
        self._rebuild_node_list()
        self._node_list.setCurrentRow(idx + 1)
        self._refresh_flow_preview()

    # ------------------------------------------------------------------
    # Condition type change
    # ------------------------------------------------------------------

    def _on_node_name_changed(self, text: str) -> None:
        idx = self._current_node_idx
        if idx is None or idx >= len(self._pipeline.nodes):
            return
        self._pipeline.nodes[idx].name = text.strip()
        item = self._node_list.item(idx)
        if item:
            item.setText(self._node_label(self._pipeline.nodes[idx]))
        self._refresh_flow_preview()

    def _on_condition_type_changed(self, idx: int) -> None:
        self._condition_stack.setCurrentIndex(idx)
        panel = self._cond_panels[idx]
        panel.load(None)
        # Forward prior nodes context to NodeResult and Composite
        curr_idx = self._current_node_idx
        if curr_idx is not None:
            prior = [n.name for n in self._pipeline.nodes[:curr_idx]]
            if hasattr(panel, "set_prior_nodes"):
                panel.set_prior_nodes(prior)
        self._on_field_changed()

    # ------------------------------------------------------------------
    # Flow preview
    # ------------------------------------------------------------------

    def _refresh_flow_preview(self) -> None:
        if self._suppress_refresh:
            return
        # Soft-flush so preview reflects current widget state
        self._flush_node_to_model()
        try:
            self._flow_preview.setPlainText(self._pipeline.flow_preview())
        except Exception:
            self._flow_preview.setPlainText(_("(preview unavailable)"))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._flush_node_to_model()

        name = self._name_edit.text().strip()
        if not name:
            qt_alert(self, _("Validation Error"), _("Pipeline name is required."), kind="warning")
            return

        existing = {p.name for p in ClassifierPipelines.get_all_pipelines()}
        if self._is_edit:
            if name != self._original_name and name in existing:
                qt_alert(
                    self,
                    _("Validation Error"),
                    _("A pipeline named '{}' already exists.").format(name),
                    kind="warning",
                )
                return
        else:
            if name in existing:
                qt_alert(
                    self,
                    _("Validation Error"),
                    _("A pipeline named '{}' already exists.").format(name),
                    kind="warning",
                )
                return

        # Assemble final pipeline with correct type
        is_pv = self._type_combo.currentIndex() == 1
        if is_pv:
            if isinstance(self._pipeline, PrevalidationPipeline):
                final = self._pipeline
            else:
                d = self._pipeline.to_dict()
                d["pipeline_class"] = "prevalidation"
                d.setdefault("profile_name", None)
                final = PrevalidationPipeline.from_dict(d)
            profile_text = self._profile_combo.currentText()
            final.profile_name = (
                profile_text if profile_text and profile_text != _("(no profiles)") else None
            )
        else:
            if isinstance(self._pipeline, PrevalidationPipeline):
                d = self._pipeline.to_dict()
                d.pop("pipeline_class", None)
                d.pop("profile_name", None)
                final = ClassifierPipeline.from_dict(d)
            else:
                final = self._pipeline

        final.name = name
        final.description = self._desc_edit.text().strip()
        final.is_active = self._active_cb.isChecked()
        final.default_action = _action_from_text(self._default_action_combo.currentText())
        final.default_reject_action = _action_from_text(self._default_reject_combo.currentText())

        errors = final.validate()
        if errors:
            qt_alert(
                self,
                _("Validation Errors"),
                "\n".join(f"• {e}" for e in errors),
                kind="warning",
            )
            return

        if self._is_edit:
            ClassifierPipelines.remove_pipeline(self._original_name)
        ClassifierPipelines.add_pipeline(final)
        ClassifierPipelines.store()
        self.close()
        self._refresh_callback()

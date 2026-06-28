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
import math
import os
from typing import Callable, Optional

from PySide6.QtCore import Qt, QEvent, QObject, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsScene, QGraphicsView,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    ClassifierRankCondition,
    CompositeCondition,
    EmbeddingCondition,
    GroupCondition,
    GroupChildResultCondition,
    LookaheadCondition,
    FilenameContainsCondition,
    NodeOutcome,
    NodeResultCondition,
    OutcomeType,
    PipelineNode,
    PrevalidationPipeline,
    PromptCondition,
    PrototypeCondition,
    BaseStemMatchCondition,
    UnknownSuffixCondition,
    RelatedImageCondition,
)
from files.directory_profile import DirectoryProfile
from lib.multi_display_qt import SmartDialog
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.constants import ClassifierActionType, ImageGenerationType
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("classifier_pipeline_editor_qt")


# ---------------------------------------------------------------------------
# Helpers shared across panels
# ---------------------------------------------------------------------------

_CONDITION_ENTRIES = [
    ("embedding",           _("Embedding")),
    ("classifier_rank",     _("Classifier Rank")),
    ("prototype",           _("Prototype")),
    ("prompt",              _("Prompt")),
    ("filename_contains",   _("Filename Contains")),
    ("related_image",       _("Related Image Exists")),
    ("base_stem_match",     _("Base Stem Match")),
    ("unknown_suffix",      _("Unknown Suffix Guard")),
    ("lookahead",           _("Lookahead")),
    ("node_result",         _("Prior Node Result")),
    ("group_child_result",  _("Group Child Result")),
    ("composite",           _("Composite")),
    ("group",               _("Group")),
]
_CONDITION_TYPES   = [k for k, _ in _CONDITION_ENTRIES]
_CONDITION_LABELS  = [v for _, v in _CONDITION_ENTRIES]

# Sub-condition types: no composite, no group, no group_child_result (need outer context)
_FLAT_TYPES = {"composite", "group", "group_child_result"}
_SUB_CONDITION_ENTRIES = [(k, v) for k, v in _CONDITION_ENTRIES if k not in _FLAT_TYPES]
_SUB_CONDITION_TYPES   = [k for k, _ in _SUB_CONDITION_ENTRIES]
_SUB_CONDITION_LABELS  = [v for _, v in _SUB_CONDITION_ENTRIES]

_FG = AppStyle.FG_COLOR
_BG = AppStyle.BG_COLOR

# Graph layout constants (all in scene pixels)
_NODE_W         = 320
_NODE_H         = 80
_NODE_X         = 10
_NODE_VSTEP     = 90   # top-of-node to top-of-next-node
_GOTO_OFFSET_M  = 42   # bezier bulge for on_match GOTO
_GOTO_OFFSET_NM = 68   # bezier bulge for on_no_match GOTO


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


def _fill_outcome_combo(combo: QComboBox) -> None:
    combo.clear()
    for ot in OutcomeType:
        combo.addItem(ot.display(), ot)


def _fill_action_combo(combo: QComboBox) -> None:
    combo.clear()
    for at in ClassifierActionType:
        combo.addItem(at.get_translation(), at)


def _fill_default_action_combo(combo: QComboBox) -> None:
    combo.clear()
    combo.addItem(_("(none)"), None)
    for at in ClassifierActionType:
        combo.addItem(at.get_translation(), at)


def _combo_set_data(combo: QComboBox, value) -> None:
    idx = combo.findData(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def _outcome_from_combo(combo: QComboBox) -> OutcomeType:
    data = combo.currentData()
    if isinstance(data, OutcomeType):
        return data
    return OutcomeType.get(combo.currentText())


def _action_from_combo(combo: QComboBox) -> ClassifierActionType:
    data = combo.currentData()
    if isinstance(data, ClassifierActionType):
        return data
    return ClassifierActionType.get_action(combo.currentText())


def _trunc(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _arrowhead(scene: "QGraphicsScene", x: float, y: float, angle_rad: float,
               pen: "QPen", size: int = 8) -> None:
    """Draw a two-line arrowhead tip at (x, y) pointing in direction angle_rad."""
    a1 = angle_rad + math.pi - 0.4
    a2 = angle_rad + math.pi + 0.4
    scene.addLine(x, y, x + size * math.cos(a1), y + size * math.sin(a1), pen)
    scene.addLine(x, y, x + size * math.cos(a2), y + size * math.sin(a2), pen)


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
        add_btn.clicked.connect(self._add_item)
        entry_row.addWidget(add_btn)

        remove_btn = QPushButton(_("Remove"))
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
# _StringPairListEditor — name → value two-column list editor
# ---------------------------------------------------------------------------

class _StringPairListEditor(QWidget):
    """QListWidget whose entries each hold a (name, value) pair, displayed as 'Name → value'.

    Used for category_map: the user enters a human-readable category name and its
    corresponding filesystem suffix (e.g. 'Apple' / '_apple').
    """

    _SEP = " → "   # " → "

    def __init__(
        self,
        name_placeholder: str = "",
        value_placeholder: str = "",
        on_changed: Callable = None,
        parent=None,
    ):
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
        self._name_entry = QLineEdit()
        self._name_entry.setPlaceholderText(name_placeholder)
        self._name_entry.returnPressed.connect(self._add_item)
        entry_row.addWidget(self._name_entry, 2)

        entry_row.addWidget(_label(self._SEP.strip()))

        self._value_entry = QLineEdit()
        self._value_entry.setPlaceholderText(value_placeholder)
        self._value_entry.returnPressed.connect(self._add_item)
        entry_row.addWidget(self._value_entry, 2)

        add_btn = QPushButton(_("Add"))
        add_btn.clicked.connect(self._add_item)
        entry_row.addWidget(add_btn)

        remove_btn = QPushButton(_("Remove"))
        remove_btn.clicked.connect(self._remove_selected)
        entry_row.addWidget(remove_btn)

        layout.addLayout(entry_row)

    def _add_item(self) -> None:
        name = self._name_entry.text().strip()
        value = self._value_entry.text().strip()
        if name and value:
            self._list.addItem(f"{name}{self._SEP}{value}")
            self._name_entry.clear()
            self._value_entry.clear()
            self._on_changed()

    def _remove_selected(self) -> None:
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
        self._on_changed()

    def set_items(self, d: dict) -> None:
        self._list.clear()
        for k, v in d.items():
            self._list.addItem(f"{k}{self._SEP}{v}")

    def get_items(self) -> dict:
        result: dict = {}
        for i in range(self._list.count()):
            text = self._list.item(i).text()
            if self._SEP in text:
                k, v = text.split(self._SEP, 1)
                k, v = k.strip(), v.strip()
                if k and v:
                    result[k] = v
        return result


# ---------------------------------------------------------------------------
# _AdaptiveStack — QStackedWidget that reports the *current* page's size hint
# so the surrounding layout doesn't reserve space for the tallest hidden page.
# ---------------------------------------------------------------------------

class _AdaptiveStack(QStackedWidget):
    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w else super().minimumSizeHint()


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

        self._inherit_categories = QCheckBox(_("Inherit from pipeline category map"))
        self._inherit_categories.setChecked(False)
        self._inherit_categories.stateChanged.connect(self._on_inherit_toggled)
        form.addRow(_("Categories:"), self._inherit_categories)

        self._categories = _StringListEditor(_("category name"), on_changed=self._on_changed)
        form.addRow("", self._categories)

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

    def _on_inherit_toggled(self, state: int) -> None:
        self._categories.setEnabled(not bool(state))
        self._on_changed()

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
            self._inherit_categories.setChecked(condition.inherit_categories)
            self._categories.set_items(condition.categories)
            self._categories.setEnabled(not condition.inherit_categories)
            self._min_rank.setValue(condition.min_rank)
            self._max_rank.setValue(condition.max_rank)
            self._min_confidence.setValue(condition.min_confidence)
        else:
            self._inherit_categories.setChecked(False)
            self._categories.set_items([])
            self._categories.setEnabled(True)
            self._min_rank.setValue(1)
            self._max_rank.setValue(1)
            self._min_confidence.setValue(0.0)

    def get_condition(self) -> ClassifierRankCondition:
        return ClassifierRankCondition(
            classifier_name=self._classifier_combo.currentText(),
            inherit_categories=self._inherit_categories.isChecked(),
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


class _FilenameContainsPanel(QWidget):
    condition_type = "filename_contains"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        layout.addWidget(_label(_("Filename patterns (match if any substring found in filename):")))

        self._patterns = _StringListEditor(_("pattern"), on_changed=self._on_changed)
        layout.addWidget(self._patterns)

        self._case_sensitive_cb = QCheckBox(_("Case-sensitive"))
        self._case_sensitive_cb.stateChanged.connect(self._on_changed)
        layout.addWidget(self._case_sensitive_cb)

    def load(self, condition) -> None:
        if isinstance(condition, FilenameContainsCondition):
            self._patterns.set_items(condition.patterns or [])
            self._case_sensitive_cb.setChecked(condition.case_sensitive)
        else:
            self._patterns.set_items([])
            self._case_sensitive_cb.setChecked(False)

    def get_condition(self) -> FilenameContainsCondition:
        return FilenameContainsCondition(
            patterns=self._patterns.get_items(),
            case_sensitive=self._case_sensitive_cb.isChecked(),
        )


class _RelatedImagePanel(QWidget):
    condition_type = "related_image"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(4)

        self._edit_suffix = QLineEdit()
        self._edit_suffix.setPlaceholderText(_("e.g. _edit"))
        self._edit_suffix.textChanged.connect(self._on_changed)
        form.addRow(_("Edit suffix:"), self._edit_suffix)

        self._use_configured_dirs = QCheckBox(
            _("Search configured related-image directories when no directory is set below")
        )
        self._use_configured_dirs.setChecked(True)
        self._use_configured_dirs.stateChanged.connect(self._on_changed)
        form.addRow("", self._use_configured_dirs)

        self._search_dir = QLineEdit()
        self._search_dir.setPlaceholderText(_("(empty = use configured dirs or pipeline working directory)"))
        self._search_dir.textChanged.connect(self._on_changed)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._search_dir, 1)
        browse_btn = QPushButton(_("Browse…"))
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(browse_btn)
        form.addRow(_("Search directory:"), dir_row)

        self._count_threshold = QSpinBox()
        self._count_threshold.setRange(1, 999)
        self._count_threshold.setValue(1)
        self._count_threshold.valueChanged.connect(self._on_changed)
        form.addRow(_("Count threshold:"), self._count_threshold)

        note = _label(
            _("Matches when the downstream file count with this suffix "
              "is below the threshold (i.e. room to generate more). "
              "When 'Search directory' is empty and the checkbox is checked, "
              "all configured related-image directories are searched; "
              "generation is suppressed if the file is found in any of them.")
        )
        note.setWordWrap(True)
        form.addRow("", note)

    def _browse(self) -> None:
        current = self._search_dir.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, _("Select search directory"), current)
        if d:
            self._search_dir.setText(d)

    def load(self, condition) -> None:
        if isinstance(condition, RelatedImageCondition):
            self._edit_suffix.setText(condition.edit_suffix)
            self._search_dir.setText(condition.search_directory)
            self._count_threshold.setValue(condition.count_threshold)
            self._use_configured_dirs.setChecked(condition.use_configured_search_directories)
        else:
            self._edit_suffix.setText("")
            self._search_dir.setText("")
            self._count_threshold.setValue(1)
            self._use_configured_dirs.setChecked(True)

    def get_condition(self) -> RelatedImageCondition:
        return RelatedImageCondition(
            edit_suffix=self._edit_suffix.text().strip(),
            search_directory=self._search_dir.text().strip(),
            count_threshold=self._count_threshold.value(),
            use_configured_search_directories=self._use_configured_dirs.isChecked(),
        )


class _BaseStemMatchPanel(QWidget):
    condition_type = "base_stem_match"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(6)

        self._require_match = QCheckBox(_("Pass when found (uncheck to pass when not found)"))
        self._require_match.setChecked(True)
        self._require_match.stateChanged.connect(self._on_changed)
        form.addRow(_("Match mode:"), self._require_match)

        self._suffix_filter = QLineEdit()
        self._suffix_filter.setPlaceholderText(_("e.g. _a, _b  (comma-separated; empty = any)"))
        self._suffix_filter.textChanged.connect(self._on_changed)
        form.addRow(_("Suffix filter:"), self._suffix_filter)

        self._use_working_directory = QCheckBox(
            _("Search pipeline working directory when no directory is set below")
        )
        self._use_working_directory.setChecked(False)
        self._use_working_directory.stateChanged.connect(self._on_changed)
        form.addRow("", self._use_working_directory)

        dir_row = QHBoxLayout()
        self._search_directory = QLineEdit()
        self._search_directory.setPlaceholderText(
            _("(empty = use configured related-image dirs or pipeline working directory; see checkbox)"))
        self._search_directory.setToolTip(
            _("When empty, searches config: directories_to_search_for_related_images, "
              "or the pipeline working directory if the checkbox above is checked.")
        )
        self._search_directory.textChanged.connect(self._on_changed)
        dir_row.addWidget(self._search_directory)
        browse_btn = QPushButton(_("Browse…"))
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        form.addRow(_("Search directory:"), dir_row)

        self._max_stem_group_size = QSpinBox()
        self._max_stem_group_size.setRange(-1, 9999)
        self._max_stem_group_size.setValue(0)
        self._max_stem_group_size.setToolTip(
            _("Overflow detection: pass when stem group exceeds this many files. "
              "-1 = auto-compute from pipeline category count (len(categories) + 1), "
              "works regardless of search directory. "
              "0 = disabled (or auto-computed when no search directory is set). "
              "When non-zero, match mode is ignored.")
        )
        self._max_stem_group_size.valueChanged.connect(self._on_changed)
        form.addRow(_("Max stem group size:"), self._max_stem_group_size)

    def _browse_dir(self) -> None:
        current = self._search_directory.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, _("Select search directory"), current)
        if d:
            self._search_directory.setText(d)

    def load(self, condition) -> None:
        if isinstance(condition, BaseStemMatchCondition):
            self._require_match.setChecked(condition.require_match)
            self._suffix_filter.setText(", ".join(condition.suffix_filter))
            self._use_working_directory.setChecked(condition.use_working_directory)
            self._search_directory.setText(condition.search_directory)
            self._max_stem_group_size.setValue(condition.max_stem_group_size)
        else:
            self._require_match.setChecked(True)
            self._suffix_filter.setText("")
            self._use_working_directory.setChecked(False)
            self._search_directory.setText("")
            self._max_stem_group_size.setValue(0)

    def get_condition(self) -> BaseStemMatchCondition:
        raw = self._suffix_filter.text()
        suffixes = [s.strip() for s in raw.split(",") if s.strip()]
        return BaseStemMatchCondition(
            require_match=self._require_match.isChecked(),
            suffix_filter=suffixes,
            use_working_directory=self._use_working_directory.isChecked(),
            search_directory=self._search_directory.text().strip(),
            max_stem_group_size=self._max_stem_group_size.value(),
        )


class _UnknownSuffixPanel(QWidget):
    condition_type = "unknown_suffix"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(4)

        self._expected_suffixes = QLineEdit()
        self._expected_suffixes.setPlaceholderText(_("e.g. _a, _ani, _animal, _b, _c, _d  (comma-separated)"))
        self._expected_suffixes.textChanged.connect(self._on_changed)
        form.addRow(_("Expected suffixes:"), self._expected_suffixes)

        self._use_working_directory = QCheckBox(
            _("Search pipeline working directory when no directory is set below")
        )
        self._use_working_directory.setChecked(False)
        self._use_working_directory.stateChanged.connect(self._on_changed)
        form.addRow("", self._use_working_directory)

        self._search_dir = QLineEdit()
        self._search_dir.setPlaceholderText(
            _("(empty = use configured related-image dirs or pipeline working directory; see checkbox)"))
        self._search_dir.textChanged.connect(self._on_changed)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._search_dir, 1)
        browse_btn = QPushButton(_("Browse…"))
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(browse_btn)
        form.addRow(_("Search directory:"), dir_row)

        self._classifier_name = QLineEdit()
        self._classifier_name.setPlaceholderText(_("(empty = no inference; unknown files always block)"))
        self._classifier_name.textChanged.connect(self._on_changed)
        form.addRow(_("Classifier (inference):"), self._classifier_name)

        self._inference_threshold = QDoubleSpinBox()
        self._inference_threshold.setRange(0.0, 1.0)
        self._inference_threshold.setSingleStep(0.05)
        self._inference_threshold.setDecimals(2)
        self._inference_threshold.setValue(0.85)
        self._inference_threshold.valueChanged.connect(self._on_changed)
        form.addRow(_("Inference threshold:"), self._inference_threshold)

        note = _label(
            _("Returns True when an unrecognised-suffix file exists that the classifier "
              "cannot resolve. Wrap in CompositeCondition(NOT) as a guard node: "
              "on_match=CONTINUE, on_no_match=REJECT. "
              "The seed image (no suffix) is never flagged.")
        )
        note.setWordWrap(True)
        form.addRow("", note)

    def _browse(self) -> None:
        current = self._search_dir.text() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, _("Select search directory"), current)
        if d:
            self._search_dir.setText(d)

    def load(self, condition) -> None:
        if isinstance(condition, UnknownSuffixCondition):
            self._expected_suffixes.setText(", ".join(condition.expected_suffixes))
            self._use_working_directory.setChecked(condition.use_base_directory)
            self._search_dir.setText(condition.search_directory)
            self._classifier_name.setText(condition.classifier_name)
            self._inference_threshold.setValue(condition.inference_threshold)
        else:
            self._expected_suffixes.setText("")
            self._use_working_directory.setChecked(False)
            self._search_dir.setText("")
            self._classifier_name.setText("")
            self._inference_threshold.setValue(0.85)

    def get_condition(self) -> UnknownSuffixCondition:
        raw = self._expected_suffixes.text()
        suffixes = [s.strip() for s in raw.split(",") if s.strip()]
        return UnknownSuffixCondition(
            expected_suffixes=suffixes,
            use_base_directory=self._use_working_directory.isChecked(),
            search_directory=self._search_dir.text().strip(),
            classifier_name=self._classifier_name.text().strip(),
            inference_threshold=self._inference_threshold.value(),
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


class _GroupChildResultPanel(QWidget):
    """Check which child inside a prior group node matched."""
    condition_type = "group_child_result"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        self._group_children: dict[str, list[str]] = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 4, 0, 4)
        form.setSpacing(4)

        self._group_combo = QComboBox()
        self._group_combo.currentIndexChanged.connect(self._on_group_changed)
        form.addRow(_("Group node:"), self._group_combo)

        self._child_combo = QComboBox()
        self._child_combo.currentIndexChanged.connect(self._on_changed)
        form.addRow(_("Child node:"), self._child_combo)

        self._expected_cb = QCheckBox(_("Expected result: matched"))
        self._expected_cb.setChecked(True)
        self._expected_cb.stateChanged.connect(self._on_changed)
        form.addRow("", self._expected_cb)

        info = _label(_("Only prior group nodes and their children appear here."))
        info.setWordWrap(True)
        form.addRow("", info)

    def _on_group_changed(self, _: int) -> None:
        self._populate_children()
        self._on_changed()

    def _populate_children(self) -> None:
        name = self._group_combo.currentText()
        children = self._group_children.get(name, [])
        self._child_combo.blockSignals(True)
        self._child_combo.clear()
        self._child_combo.addItems(children if children else [_("(no children)")])
        self._child_combo.blockSignals(False)

    def set_prior_group_nodes(self, group_children: "dict[str, list[str]]") -> None:
        """group_children: {group_node_name: [child_node_names]}"""
        self._group_children = group_children
        current_group = self._group_combo.currentText()
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        names = list(group_children.keys())
        self._group_combo.addItems(names if names else [_("(no prior group nodes)")])
        if current_group in names:
            self._group_combo.setCurrentText(current_group)
        self._group_combo.blockSignals(False)
        self._populate_children()

    def load(self, condition) -> None:
        if isinstance(condition, GroupChildResultCondition):
            idx = self._group_combo.findText(condition.group_node_name)
            if idx >= 0:
                self._group_combo.setCurrentIndex(idx)
                self._populate_children()
            idx2 = self._child_combo.findText(condition.child_node_name)
            if idx2 >= 0:
                self._child_combo.setCurrentIndex(idx2)
            self._expected_cb.setChecked(condition.expected_result)
        else:
            self._expected_cb.setChecked(True)

    def get_condition(self) -> GroupChildResultCondition:
        return GroupChildResultCondition(
            group_node_name=self._group_combo.currentText(),
            child_node_name=self._child_combo.currentText(),
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
            _FilenameContainsPanel(on_changed=self._on_changed),
            _RelatedImagePanel(on_changed=self._on_changed),
            _BaseStemMatchPanel(on_changed=self._on_changed),
            _UnknownSuffixPanel(on_changed=self._on_changed),
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


class _GroupPanel(QWidget):
    """
    Editor for a GroupCondition: an ordered list of named child nodes, each
    with its own condition (no composite/group nesting).  Child outcomes are
    not edited here — only the condition matters; routing is handled by the
    outer pipeline node.
    """
    condition_type = "group"

    def __init__(self, on_changed: Callable = None, parent=None):
        super().__init__(parent)
        self._on_changed = on_changed or (lambda: None)
        self._children: list[PipelineNode] = []
        self._current_idx: Optional[int] = None
        self._suppress = False
        self._prior_nodes: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 4)
        root.setSpacing(6)

        op_row = QHBoxLayout()
        op_row.addWidget(_label(_("Operator:")))
        self._op_combo = QComboBox()
        self._op_combo.addItems(["OR", "AND"])
        self._op_combo.currentIndexChanged.connect(self._on_changed)
        op_row.addWidget(self._op_combo)
        op_row.addStretch()
        root.addLayout(op_row)

        root.addWidget(_label(
            _("OR: match if any child matches  ·  AND: match if all children match")
        ))

        root.addWidget(_label(_("Child nodes:"), bold=True))
        self._child_list = QListWidget()
        self._child_list.setStyleSheet(
            f"color: {_FG}; background: {_BG}; selection-background-color: #444;"
        )
        self._child_list.setMaximumHeight(90)
        self._child_list.currentRowChanged.connect(self._on_child_selected)
        root.addWidget(self._child_list)

        btn_row = QHBoxLayout()
        for lbl, slot in [
            (_("Add"),    self._add_child),
            (_("Remove"), self._remove_child),
            (_("↑"),      self._move_up),
            (_("↓"),      self._move_down),
        ]:
            b = QPushButton(lbl)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Per-child editor (name + condition)
        self._child_editor = QWidget()
        child_form = QFormLayout(self._child_editor)
        child_form.setContentsMargins(0, 4, 0, 0)
        child_form.setSpacing(4)

        self._child_name_edit = QLineEdit()
        self._child_name_edit.setPlaceholderText(_("unique child name"))
        self._child_name_edit.textChanged.connect(self._on_child_name_changed)
        child_form.addRow(_("Child name:"), self._child_name_edit)

        self._child_ctype_combo = QComboBox()
        self._child_ctype_combo.addItems(_SUB_CONDITION_LABELS)
        self._child_ctype_combo.currentIndexChanged.connect(self._on_child_ctype_changed)
        child_form.addRow(_("Condition:"), self._child_ctype_combo)
        root.addWidget(self._child_editor)

        # Stacked condition panels (flat — no group/composite nesting)
        changed_cb = self._on_changed
        self._child_panels = [
            _EmbeddingPanel(on_changed=changed_cb),
            _ClassifierRankPanel(on_changed=changed_cb),
            _PrototypePanel(on_changed=changed_cb),
            _PromptPanel(on_changed=changed_cb),
            _FilenameContainsPanel(on_changed=changed_cb),
            _RelatedImagePanel(on_changed=changed_cb),
            _BaseStemMatchPanel(on_changed=changed_cb),
            _UnknownSuffixPanel(on_changed=changed_cb),
            _LookaheadPanel(on_changed=changed_cb),
            _NodeResultPanel(on_changed=changed_cb),
        ]
        self._child_stack = QStackedWidget()
        for p in self._child_panels:
            self._child_stack.addWidget(p)
        root.addWidget(self._child_stack)

        self._child_editor.setEnabled(False)
        self._child_stack.setEnabled(False)

    # --- prior-node context forwarded to NodeResult panels inside children ---

    def set_prior_nodes(self, names: list) -> None:
        self._prior_nodes = names
        for p in self._child_panels:
            if hasattr(p, "set_prior_nodes"):
                p.set_prior_nodes(names)

    # --- child list helpers -------------------------------------------------

    def _child_label(self, child: PipelineNode) -> str:
        ctype = getattr(child.condition, "condition_type", "?")
        return f"{child.name}  [{ctype}]"

    def _rebuild_child_list(self) -> None:
        row = self._child_list.currentRow()
        self._child_list.blockSignals(True)
        self._child_list.clear()
        for child in self._children:
            self._child_list.addItem(self._child_label(child))
        self._child_list.blockSignals(False)
        target = row if 0 <= row < len(self._children) else (0 if self._children else -1)
        if target >= 0:
            self._child_list.setCurrentRow(target)

    def _flush_child(self) -> None:
        idx = self._current_idx
        if idx is None or idx >= len(self._children):
            return
        child = self._children[idx]
        name = self._child_name_edit.text().strip()
        if name:
            child.name = name
        try:
            child.condition = self._child_panels[self._child_stack.currentIndex()].get_condition()
        except Exception:
            pass

    def _on_child_selected(self, row: int) -> None:
        self._flush_child()
        if row < 0 or row >= len(self._children):
            self._current_idx = None
            self._child_editor.setEnabled(False)
            self._child_stack.setEnabled(False)
            return
        self._current_idx = row
        self._child_editor.setEnabled(True)
        self._child_stack.setEnabled(True)
        self._suppress = True
        child = self._children[row]
        self._child_name_edit.setText(child.name)
        ctype = getattr(child.condition, "condition_type", "embedding")
        ct_idx = _SUB_CONDITION_TYPES.index(ctype) if ctype in _SUB_CONDITION_TYPES else 0
        self._child_ctype_combo.setCurrentIndex(ct_idx)
        self._child_stack.setCurrentIndex(ct_idx)
        self._child_panels[ct_idx].load(child.condition)
        self._suppress = False
        self._on_changed()

    def _on_child_name_changed(self, text: str) -> None:
        idx = self._current_idx
        if idx is None or idx >= len(self._children) or self._suppress:
            return
        self._children[idx].name = text.strip()
        item = self._child_list.item(idx)
        if item:
            item.setText(self._child_label(self._children[idx]))
        self._on_changed()

    def _on_child_ctype_changed(self, idx: int) -> None:
        if self._suppress:
            return
        self._child_stack.setCurrentIndex(idx)
        self._child_panels[idx].load(None)
        self._on_changed()

    def _add_child(self) -> None:
        self._flush_child()
        child = PipelineNode(name=f"child_{len(self._children) + 1}")
        self._children.append(child)
        self._rebuild_child_list()
        self._child_list.setCurrentRow(len(self._children) - 1)
        self._on_changed()

    def _remove_child(self) -> None:
        idx = self._child_list.currentRow()
        if idx < 0 or idx >= len(self._children):
            return
        self._current_idx = None
        del self._children[idx]
        self._rebuild_child_list()
        if not self._children:
            self._child_editor.setEnabled(False)
            self._child_stack.setEnabled(False)
        self._on_changed()

    def _move_up(self) -> None:
        self._flush_child()
        idx = self._child_list.currentRow()
        if idx <= 0:
            return
        self._children[idx], self._children[idx - 1] = (
            self._children[idx - 1], self._children[idx]
        )
        self._current_idx = idx - 1
        self._rebuild_child_list()
        self._child_list.setCurrentRow(idx - 1)
        self._on_changed()

    def _move_down(self) -> None:
        self._flush_child()
        idx = self._child_list.currentRow()
        if idx < 0 or idx >= len(self._children) - 1:
            return
        self._children[idx], self._children[idx + 1] = (
            self._children[idx + 1], self._children[idx]
        )
        self._current_idx = idx + 1
        self._rebuild_child_list()
        self._child_list.setCurrentRow(idx + 1)
        self._on_changed()

    # --- load / get ---------------------------------------------------------

    def load(self, condition) -> None:
        self._current_idx = None
        if isinstance(condition, GroupCondition):
            self._op_combo.setCurrentText(condition.operator)
            self._children = [copy.deepcopy(c) for c in condition.nodes]
        else:
            self._op_combo.setCurrentIndex(0)
            self._children = []
        self._rebuild_child_list()
        self._child_editor.setEnabled(False)
        self._child_stack.setEnabled(False)

    def get_condition(self) -> GroupCondition:
        self._flush_child()
        return GroupCondition(operator=self._op_combo.currentText(), nodes=list(self._children))


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
        _fill_outcome_combo(self._type_combo)
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
        _fill_action_combo(self._action_combo)
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
        action = self._action_combo.currentData()
        needs_dir = isinstance(action, ClassifierActionType) and action.requires_target_directory()
        for i in range(self._modifier_row.count()):
            item = self._modifier_row.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(needs_dir)
        self._on_changed()

    def _update_dependent_visibility(self, outcome_val: str) -> None:
        try:
            ot = _outcome_from_combo(self._type_combo)
        except ValueError:
            try:
                ot = OutcomeType(outcome_val)
            except ValueError:
                return
        self._set_layout_visible(self._goto_row, ot.requires_target_node)
        self._set_layout_visible(self._exec_row, ot.requires_action)
        if ot.requires_action:
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
        if not isinstance(ot, OutcomeType):
            ot = OutcomeType(ot)
        _combo_set_data(self._type_combo, ot)
        if ot.requires_target_node and outcome.target_node:
            idx2 = self._goto_combo.findText(outcome.target_node)
            if idx2 >= 0:
                self._goto_combo.setCurrentIndex(idx2)
        if ot.requires_action:
            if outcome.action_type:
                _combo_set_data(self._action_combo, outcome.action_type)
            self._modifier_edit.setText(outcome.action_modifier or "")
        self._update_dependent_visibility(self._type_combo.currentText())

    def get_outcome(self) -> NodeOutcome:
        ot = _outcome_from_combo(self._type_combo)
        if ot.requires_target_node:
            return NodeOutcome(
                outcome_type=ot,
                target_node=self._goto_combo.currentText(),
            )
        if ot.requires_action:
            return NodeOutcome(
                outcome_type=ot,
                action_type=_action_from_combo(self._action_combo),
                action_modifier=self._modifier_edit.text().strip() or None,
            )
        return NodeOutcome(outcome_type=ot)


# ---------------------------------------------------------------------------
# Wheel-scroll guard — prevents accidental value changes on unfocused combos
# ---------------------------------------------------------------------------

class _WheelGuard(QObject):
    """Event filter: discards wheel events on QComboBox widgets that lack focus."""
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and not watched.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(watched, event)


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
        self._node_editor_widget: Optional[QWidget] = None

        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Edit Pipeline") if self._is_edit else _("New Pipeline"),
            geometry="1280x1050",
            respect_title_bar=True,
        )

        self._wheel_guard = _WheelGuard(self)
        self._guarded_combos: set = set()
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

        # _rebuild_node_list already selected row 0, but _on_selection_changed
        # returned early because _node_editor_widget didn't exist yet.
        # All widgets (_node_editor_widget, _flow_preview) are now built —
        # call directly to initialize _current_node_idx and load the editor.
        if self._pipeline.nodes:
            self._on_selection_changed()

        self._install_wheel_guard()

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

        self._desc_edit = QPlainTextEdit(getattr(p, "description", "") or "")
        self._desc_edit.setFixedHeight(60)
        self._desc_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        form.addRow(_("Description:"), self._desc_edit)

        _active_row_layout = QHBoxLayout()
        self._active_cb = QCheckBox(_("Active"))
        self._active_cb.setChecked(p.is_active)
        _active_row_layout.addWidget(self._active_cb)
        self._move_to_working_dir_cb = QCheckBox(_("Move generated to working dir"))
        self._move_to_working_dir_cb.setChecked(p.move_to_working_dir)
        self._move_to_working_dir_cb.setToolTip(
            _("When enabled, GENERATE actions pass the current working directory "
              "to sd-runner so generated files land there instead of the default output folder.")
        )
        _active_row_layout.addWidget(self._move_to_working_dir_cb)
        _active_row_layout.addStretch(1)
        form.addRow("", _active_row_layout)

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
        _fill_default_action_combo(self._default_action_combo)
        if p.default_action:
            _combo_set_data(self._default_action_combo, p.default_action)
        form.addRow(_("Default action:"), self._default_action_combo)

        self._default_reject_combo = QComboBox()
        _fill_default_action_combo(self._default_reject_combo)
        if p.default_reject_action:
            _combo_set_data(self._default_reject_combo, p.default_reject_action)
        form.addRow(_("Default reject action:"), self._default_reject_combo)

        self._gen_type_combo = QComboBox()
        self._gen_type_combo.addItem(_("(Use global setting)"), None)
        _excluded = {ImageGenerationType.CANCEL, ImageGenerationType.REVERT_TO_SIMPLE_GEN}
        for gt in ImageGenerationType:
            if gt not in _excluded:
                self._gen_type_combo.addItem(gt.get_text(), gt)
        if p.generation_type is not None:
            idx = self._gen_type_combo.findData(p.generation_type)
            if idx >= 0:
                self._gen_type_combo.setCurrentIndex(idx)
        form.addRow(_("Generation type:"), self._gen_type_combo)

        self._category_map_editor = _StringPairListEditor(
            name_placeholder=_("category name, e.g. Apple"),
            value_placeholder=_("suffix, e.g. _apple"),
            on_changed=self._on_category_map_changed,
        )
        self._category_map_editor.set_items(p.category_map)
        form.addRow(_("Category map:"), self._category_map_editor)

        self._seed_category_combo = QComboBox()
        self._seed_category_combo.setToolTip(
            _("The category this pipeline generates from seed images. "
              "Used in the run confirmation dialog to label the operation.")
        )
        self._seed_category_combo.currentTextChanged.connect(self._on_field_changed)
        form.addRow(_("Seed category:"), self._seed_category_combo)
        self._refresh_seed_category_combo(p.seed_category)

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
        self._node_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._node_list.itemSelectionChanged.connect(self._on_selection_changed)
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

        group_row = QHBoxLayout()
        self._group_btn = QPushButton(_("Group Selected →"))
        self._group_btn.setToolTip(
            _("Convert the selected nodes into children of a new Group node "
              "(select 2 or more with Ctrl+click)")
        )
        self._group_btn.setEnabled(False)
        self._group_btn.clicked.connect(self._group_selected_nodes)
        group_row.addWidget(self._group_btn)

        fill_btn = QPushButton(_("Fill from Map"))
        fill_btn.setToolTip(
            _("Append one Generate node per category-map entry. "
              "Each node checks that no derivative exists yet and that the target "
              "directory lacks this stem, then generates with the category suffix. "
              "Set the target directory per node after generating.")
        )
        fill_btn.clicked.connect(self._fill_nodes_from_category_map)
        group_row.addWidget(fill_btn)

        group_row.addStretch()
        lay.addLayout(group_row)

        self._rebuild_node_list()
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

        # Node name + enabled toggle
        name_form = QFormLayout()
        name_form.setSpacing(4)
        self._node_name_edit = QLineEdit()
        self._node_name_edit.setPlaceholderText(_("unique node name"))
        self._node_name_edit.textChanged.connect(self._on_node_name_changed)
        name_form.addRow(_("Node name:"), self._node_name_edit)

        self._node_enabled_cb = QCheckBox()
        self._node_enabled_cb.setChecked(True)
        self._node_enabled_cb.stateChanged.connect(self._on_node_enabled_changed)
        name_form.addRow(_("Enabled:"), self._node_enabled_cb)
        ned.addLayout(name_form)

        # Condition type
        ctype_form = QFormLayout()
        ctype_form.setSpacing(4)
        self._condition_type_combo = QComboBox()
        self._condition_type_combo.addItems(_CONDITION_LABELS)
        self._condition_type_combo.currentIndexChanged.connect(self._on_condition_type_changed)
        ctype_form.addRow(_("Condition type:"), self._condition_type_combo)
        ned.addLayout(ctype_form)

        # Condition panels (stacked) — order must match _CONDITION_TYPES exactly
        changed_cb = self._on_field_changed
        self._embedding_panel           = _EmbeddingPanel(on_changed=changed_cb)
        self._classifier_rank_panel     = _ClassifierRankPanel(on_changed=changed_cb)
        self._prototype_panel           = _PrototypePanel(on_changed=changed_cb)
        self._prompt_panel              = _PromptPanel(on_changed=changed_cb)
        self._filename_contains_panel   = _FilenameContainsPanel(on_changed=changed_cb)
        self._related_image_panel       = _RelatedImagePanel(on_changed=changed_cb)
        self._base_stem_match_panel     = _BaseStemMatchPanel(on_changed=changed_cb)
        self._unknown_suffix_panel      = _UnknownSuffixPanel(on_changed=changed_cb)
        self._lookahead_panel           = _LookaheadPanel(on_changed=changed_cb)
        self._node_result_panel         = _NodeResultPanel(on_changed=changed_cb)
        self._group_child_result_panel  = _GroupChildResultPanel(on_changed=changed_cb)
        self._composite_panel           = _CompositePanel(on_changed=changed_cb)
        self._group_panel               = _GroupPanel(on_changed=changed_cb)

        self._cond_panels = [
            self._embedding_panel, self._classifier_rank_panel,
            self._prototype_panel, self._prompt_panel,
            self._filename_contains_panel, self._related_image_panel,
            self._base_stem_match_panel,
            self._unknown_suffix_panel,
            self._lookahead_panel, self._node_result_panel,
            self._group_child_result_panel,
            self._composite_panel,
            self._group_panel,
        ]

        self._condition_stack = _AdaptiveStack()
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

        self._flow_scene = QGraphicsScene()
        self._flow_view = QGraphicsView(self._flow_scene)
        self._flow_view.setMinimumHeight(160)
        self._flow_view.setMaximumHeight(220)
        self._flow_view.setRenderHint(QPainter.Antialiasing)
        self._flow_view.setStyleSheet(
            f"QGraphicsView {{ background: #1e1e1e; border: 1px solid {_FG}; }}"
        )
        self._flow_view.setDragMode(QGraphicsView.ScrollHandDrag)
        lay.addWidget(self._flow_view)
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

    def _on_category_map_changed(self) -> None:
        current = self._seed_category_combo.currentText()
        self._refresh_seed_category_combo(current)
        self._on_field_changed()

    def _refresh_seed_category_combo(self, select: str = "") -> None:
        self._seed_category_combo.blockSignals(True)
        self._seed_category_combo.clear()
        self._seed_category_combo.addItem(_("(none)"), "")
        for category in self._category_map_editor.get_items():
            self._seed_category_combo.addItem(category, category)
        idx = self._seed_category_combo.findData(select)
        self._seed_category_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._seed_category_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Node list
    # ------------------------------------------------------------------

    def _rebuild_node_list(self) -> None:
        row = self._node_list.currentRow()
        self._node_list.blockSignals(True)
        self._node_list.clear()
        for node in self._pipeline.nodes:
            item = QListWidgetItem(self._node_label(node))
            if not node.enabled:
                item.setForeground(QColor("#888888"))
            self._node_list.addItem(item)
        self._node_list.blockSignals(False)
        target = row if 0 <= row < len(self._pipeline.nodes) else (
            0 if self._pipeline.nodes else -1
        )
        if target >= 0:
            self._node_list.setCurrentRow(target)

    def _node_label(self, node: PipelineNode) -> str:
        ctype = getattr(node.condition, "condition_type", "?")
        match_summary = node.on_match.display_summary()
        no_match_summary = node.on_no_match.display_summary()
        label = f"{node.name}  [{ctype}]  ✓{match_summary} / ✗{no_match_summary}"
        return f"[{_('disabled')}]  {label}" if not node.enabled else label

    def _on_selection_changed(self) -> None:
        if self._node_editor_widget is None:
            return
        selected = sorted(idx.row() for idx in self._node_list.selectedIndexes())
        n = len(selected)
        self._group_btn.setEnabled(n >= 2)
        if n == 1:
            row = selected[0]
            if row == self._current_node_idx and self._node_editor_widget.isEnabled():
                return
            self._flush_node_to_model()
            self._current_node_idx = row
            self._node_editor_widget.setEnabled(True)
            self._load_node_to_editor(row)
        else:
            self._flush_node_to_model()
            self._current_node_idx = None
            self._node_editor_widget.setEnabled(False)

    def _group_selected_nodes(self) -> None:
        """Convert the currently selected nodes into children of a new GroupCondition node."""
        self._flush_node_to_model()
        selected = sorted(idx.row() for idx in self._node_list.selectedIndexes())
        if len(selected) < 2:
            return

        # Collect the nodes being grouped and the names of nodes that will disappear
        nodes_to_group = [self._pipeline.nodes[i] for i in selected]
        absorbed_names = {n.name for n in nodes_to_group}

        # Build child PipelineNodes: preserve name + condition; outcomes are ignored by runner
        children = [
            PipelineNode(
                name=node.name,
                condition=copy.deepcopy(node.condition),
            )
            for node in nodes_to_group
        ]

        # Pick a name for the outer group node
        base = f"group_{nodes_to_group[0].name}"
        existing = {n.name for n in self._pipeline.nodes}
        name = base
        counter = 2
        while name in existing:
            name = f"{base}_{counter}"
            counter += 1

        # Carry forward the first selected node's outcomes, but clear any GOTO targets
        # that pointed into the now-absorbed set (they'd become invalid).
        def _safe_outcome(outcome: NodeOutcome) -> NodeOutcome:
            if (outcome.outcome_type == OutcomeType.GOTO
                    and outcome.target_node in absorbed_names):
                return NodeOutcome(OutcomeType.CONTINUE)
            return copy.deepcopy(outcome)

        group_node = PipelineNode(
            name=name,
            condition=GroupCondition(operator="OR", nodes=children),
            on_match=_safe_outcome(nodes_to_group[0].on_match),
            on_no_match=_safe_outcome(nodes_to_group[0].on_no_match),
        )

        # Replace selected nodes with the single group node (insert at first position)
        insert_at = selected[0]
        for i in reversed(selected):
            del self._pipeline.nodes[i]
        self._pipeline.nodes.insert(insert_at, group_node)

        self._current_node_idx = None
        self._rebuild_node_list()
        self._node_list.setCurrentRow(insert_at)
        self._refresh_flow_preview()

    def _fill_nodes_from_category_map(self) -> None:
        category_map = self._category_map_editor.get_items()
        if not category_map:
            qt_alert(self, _("Fill from Map"),
                     _("The category map is empty. Add categories before generating nodes."))
            return

        covered = {
            category
            for category, suffix in category_map.items()
            if any(n.is_category_generate_node(suffix) for n in self._pipeline.nodes)
        }
        if covered:
            cats_str = ", ".join(f"'{c}'" for c in covered)
            if not qt_alert(
                self, _("Fill from Map"),
                _("The following categories already have a matching Generate node and will be "
                  "skipped: {cats}.\n\nAppend nodes for the remaining categories?").format(
                    cats=cats_str),
                kind="askokcancel",
            ):
                return

        self._flush_node_to_model()

        added = 0
        for category, suffix in category_map.items():
            if category in covered:
                continue
            node = PipelineNode(
                name=f"Generate {category}",
                condition=CompositeCondition(
                    operator="AND",
                    sub_conditions=[
                        RelatedImageCondition(
                            edit_suffix=suffix,
                            use_configured_search_directories=False,
                        ),
                        BaseStemMatchCondition(
                            require_match=False,
                            use_working_directory=True,
                        ),
                    ],
                ),
                on_match=NodeOutcome(
                    outcome_type=OutcomeType.EXECUTE_AND_CONTINUE,
                    action_type=ClassifierActionType.GENERATE,
                    action_modifier=suffix,
                ),
                on_no_match=NodeOutcome(outcome_type=OutcomeType.CONTINUE),
            )
            self._pipeline.nodes.append(node)
            added += 1

        self._rebuild_node_list()
        self._refresh_flow_preview()

        if added:
            qt_alert(
                self, _("Fill from Map"),
                _("{count} node(s) added. Set the target directory on each "
                  "BaseStemMatch sub-condition to the per-category output folder.").format(
                    count=added),
            )

    def _load_node_to_editor(self, idx: int) -> None:
        self._suppress_refresh = True
        node = self._pipeline.nodes[idx]
        prior = [n.name for n in self._pipeline.nodes[:idx]]
        later = [n.name for n in self._pipeline.nodes[idx + 1:]]

        self._node_name_edit.setText(node.name)
        self._node_enabled_cb.setChecked(node.enabled)

        ctype = getattr(node.condition, "condition_type", "embedding")
        ct_idx = _CONDITION_TYPES.index(ctype) if ctype in _CONDITION_TYPES else 0
        self._condition_type_combo.setCurrentIndex(ct_idx)
        self._condition_stack.setCurrentIndex(ct_idx)
        self._cond_panels[ct_idx].load(node.condition)

        self._node_result_panel.set_prior_nodes(prior)
        self._composite_panel.set_prior_nodes(prior)
        self._group_panel.set_prior_nodes(prior)

        # Build group_children dict: {group_node_name: [child_names]} from prior nodes only
        group_children: dict[str, list] = {}
        for n in self._pipeline.nodes[:idx]:
            if getattr(n.condition, "condition_type", "") == "group":
                group_children[n.name] = [c.name for c in n.condition.nodes]
        self._group_child_result_panel.set_prior_group_nodes(group_children)

        self._on_match_editor.load(node.on_match, later)
        self._on_no_match_editor.load(node.on_no_match, later)

        self._suppress_refresh = False
        self._refresh_flow_preview()
        self._install_wheel_guard()

    def _install_wheel_guard(self) -> None:
        for combo in self.findChildren(QComboBox):
            if combo not in self._guarded_combos:
                combo.installEventFilter(self._wheel_guard)
                self._guarded_combos.add(combo)

    def _flush_node_to_model(self) -> None:
        idx = self._current_node_idx
        if idx is None or idx >= len(self._pipeline.nodes):
            return
        node = self._pipeline.nodes[idx]
        name = self._node_name_edit.text().strip()
        if name:
            node.name = name
        node.enabled = self._node_enabled_cb.isChecked()
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
        self._update_node_list_item(idx)
        self._refresh_flow_preview()

    def _on_node_enabled_changed(self, state: int) -> None:
        idx = self._current_node_idx
        if idx is None or idx >= len(self._pipeline.nodes):
            return
        self._pipeline.nodes[idx].enabled = bool(state)
        self._update_node_list_item(idx)
        self._refresh_flow_preview()

    def _update_node_list_item(self, idx: int) -> None:
        item = self._node_list.item(idx)
        if item is None:
            return
        node = self._pipeline.nodes[idx]
        item.setText(self._node_label(node))
        item.setForeground(QColor(_FG) if node.enabled else QColor("#888888"))

    def _on_condition_type_changed(self, idx: int) -> None:
        self._condition_stack.setCurrentIndex(idx)
        self._condition_stack.updateGeometry()
        panel = self._cond_panels[idx]
        panel.load(None)
        # Forward prior nodes context to NodeResult and Composite
        curr_idx = self._current_node_idx
        if curr_idx is not None:
            prior = [n.name for n in self._pipeline.nodes[:curr_idx]]
            if hasattr(panel, "set_prior_nodes"):
                panel.set_prior_nodes(prior)
            if hasattr(panel, "set_prior_group_nodes"):
                group_children: dict[str, list] = {}
                for n in self._pipeline.nodes[:curr_idx]:
                    if getattr(n.condition, "condition_type", "") == "group":
                        group_children[n.name] = [c.name for c in n.condition.nodes]
                panel.set_prior_group_nodes(group_children)
        self._on_field_changed()

    # ------------------------------------------------------------------
    # Flow preview
    # ------------------------------------------------------------------

    def _refresh_flow_preview(self) -> None:
        if self._suppress_refresh:
            return
        if not hasattr(self, "_flow_scene"):
            return
        self._flush_node_to_model()
        try:
            self._render_flow_graph()
        except Exception:
            self._flow_scene.clear()
            item = self._flow_scene.addText(_("(preview unavailable)"))
            item.setDefaultTextColor(QColor(_FG))

    def _render_flow_graph(self) -> None:
        """Rebuild the QGraphicsScene with a node-graph view of the pipeline."""
        scene = self._flow_scene
        scene.clear()

        nodes = self._pipeline.nodes
        if not nodes:
            item = scene.addText(_("(no nodes)"))
            item.setDefaultTextColor(QColor(_FG))
            return

        fg        = QColor(_FG)
        node_bg   = QColor(46, 46, 46)
        green     = QColor("#5cb85c")
        red       = QColor("#d9534f")
        group_col = QColor("#5b9bd5")

        small_font = QFont()
        small_font.setPointSize(8)
        bold_font = QFont()
        bold_font.setPointSize(9)
        bold_font.setBold(True)

        NW    = _NODE_W
        NH    = _NODE_H
        NX    = _NODE_X
        GAP   = _NODE_VSTEP - NH  # vertical gap between boxes

        _CHILD_LINE_H = 16  # px per child row inside a group box

        def _node_h(node) -> int:
            if isinstance(node.condition, GroupCondition):
                n = len(node.condition.nodes)
                # 38px for header rows + child rows + 40px footer for two outcome lines
                return max(NH, 38 + n * _CHILD_LINE_H + 40)
            return NH

        # Compute per-node heights and cumulative top positions
        heights = {node.name: _node_h(node) for node in nodes}
        tops: list[int] = []
        y = 0
        for node in nodes:
            tops.append(y)
            y += heights[node.name] + GAP

        pos = {node.name: (NX, tops[i]) for i, node in enumerate(nodes)}

        # --- Node boxes ---------------------------------------------------
        box_pen      = QPen(fg, 1)
        box_brush    = QBrush(node_bg)
        dim_bg       = QColor(32, 32, 32)
        dim_pen      = QPen(QColor("#666666"), 1, Qt.DashLine)
        dim_fg       = QColor("#666666")
        for node in nodes:
            nx, ny = pos[node.name]
            nh = heights[node.name]
            ctype = getattr(node.condition, "condition_type", "?")
            disabled   = not node.enabled
            node_brush = QBrush(dim_bg)   if disabled else box_brush
            node_pen   = dim_pen          if disabled else box_pen
            text_col   = dim_fg           if disabled else fg

            if isinstance(node.condition, GroupCondition):
                grp_pen = QPen(QColor("#3a5a7a") if disabled else group_col,
                               1.5, Qt.DashLine if disabled else Qt.SolidLine)
                scene.addRect(QRectF(nx, ny, NW, nh), grp_pen, node_brush)

                name_text = _trunc(node.name, 34)
                if disabled:
                    name_text = f"[{_('disabled')}]  {name_text}"
                name_item = scene.addText(name_text)
                name_item.setFont(bold_font)
                name_item.setDefaultTextColor(text_col)
                name_item.setPos(nx + 4, ny + 2)

                op = node.condition.operator
                ct_item = scene.addText(f"[group: {op}]")
                ct_item.setFont(small_font)
                ct_item.setDefaultTextColor(QColor("#3a5a7a") if disabled else group_col)
                ct_item.setPos(nx + 4, ny + 20)

                child_col = QColor("#555555") if disabled else QColor("#cccccc")
                for j, child in enumerate(node.condition.nodes):
                    child_ctype = getattr(child.condition, "condition_type", "?")
                    child_item = scene.addText(
                        f"  · {_trunc(child.name, 22)}  [{child_ctype}]"
                    )
                    child_item.setFont(small_font)
                    child_item.setDefaultTextColor(child_col)
                    child_item.setPos(nx + 4, ny + 38 + j * _CHILD_LINE_H)
            else:
                scene.addRect(QRectF(nx, ny, NW, nh), node_pen, node_brush)

                name_text = _trunc(node.name, 34)
                if disabled:
                    name_text = f"[{_('disabled')}]  {name_text}"
                name_item = scene.addText(name_text)
                name_item.setFont(bold_font)
                name_item.setDefaultTextColor(text_col)
                name_item.setPos(nx + 4, ny + 2)

                ct_item = scene.addText(f"[{ctype}]")
                ct_item.setFont(small_font)
                ct_item.setDefaultTextColor(QColor("#555555") if disabled else QColor("#aaaaaa"))
                ct_item.setPos(nx + 4, ny + 20)

            # Outcome labels stacked at the bottom of every node (full width each)
            m_col  = QColor("#3a6e3a") if disabled else green
            nm_col = QColor("#7a3a3a") if disabled else red
            m_item = scene.addText("✓ " + _trunc(node.on_match.display_summary(), 44))
            m_item.setFont(small_font)
            m_item.setDefaultTextColor(m_col)
            m_item.setPos(nx + 4, ny + nh - 36)

            nm_item = scene.addText("✗ " + _trunc(node.on_no_match.display_summary(), 44))
            nm_item.setFont(small_font)
            nm_item.setDefaultTextColor(nm_col)
            nm_item.setPos(nx + 4, ny + nh - 18)

        # --- Flow edges ---------------------------------------------------
        # CONTINUE: dashed vertical line at two x offsets.
        # GOTO: solid C-curve on the right, proportional to actual node height.
        for i, node in enumerate(nodes):
            nx, ny = pos[node.name]
            nh = heights[node.name]

            for outcome, is_match in ((node.on_match, True), (node.on_no_match, False)):
                col    = green if is_match else red
                band   = nh * 0.35 if is_match else nh * 0.65
                offset = _GOTO_OFFSET_M if is_match else _GOTO_OFFSET_NM

                if outcome.outcome_type == OutcomeType.CONTINUE:
                    if i + 1 < len(nodes):
                        xa = nx + NW * (0.40 if is_match else 0.55)
                        dash_pen = QPen(col, 1, Qt.DashLine)
                        y_bot = ny + nh
                        y_top = tops[i + 1]
                        scene.addLine(xa, y_bot, xa, y_top, dash_pen)
                        _arrowhead(scene, xa, y_top, math.pi / 2, dash_pen)

                elif outcome.outcome_type == OutcomeType.GOTO:
                    tgt = outcome.target_node
                    if tgt in pos:
                        _, ty  = pos[tgt]
                        t_nh   = heights[tgt]
                        t_band = t_nh * 0.35 if is_match else t_nh * 0.65
                        pen    = QPen(col, 1.5)
                        x_edge = nx + NW
                        y_src  = ny + band
                        y_dst  = ty + t_band
                        ctrl_x = x_edge + offset
                        path = QPainterPath()
                        path.moveTo(x_edge, y_src)
                        path.cubicTo(ctrl_x, y_src, ctrl_x, y_dst, x_edge, y_dst)
                        scene.addPath(path, pen)
                        _arrowhead(scene, x_edge, y_dst, math.pi, pen)

        scene.setSceneRect(
            scene.itemsBoundingRect().adjusted(-8, -8, _GOTO_OFFSET_NM + 16, 8)
        )

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
        final.description = self._desc_edit.toPlainText().strip()
        final.is_active = self._active_cb.isChecked()
        final.move_to_working_dir = self._move_to_working_dir_cb.isChecked()
        final.default_action = self._default_action_combo.currentData()
        final.default_reject_action = self._default_reject_combo.currentData()
        final.generation_type = self._gen_type_combo.currentData()
        final.category_map = self._category_map_editor.get_items()
        final.seed_category = self._seed_category_combo.currentData() or ""

        errors = final.validate()
        if errors:
            qt_alert(
                self,
                _("Validation Errors"),
                "\n".join(f"• {e}" for e in errors),
                kind="warning",
            )
            return

        warnings = final.validate_warnings()
        if warnings:
            qt_alert(
                self,
                _("Category Map Warnings"),
                "\n".join(f"• {w}" for w in warnings),
                kind="info",
            )

        if self._is_edit:
            all_pipelines = ClassifierPipelines.get_all_pipelines()
            original_index = next(
                (i for i, p in enumerate(all_pipelines) if p.name == self._original_name),
                None,
            )
            ClassifierPipelines.remove_pipeline(self._original_name)
            if original_index is not None:
                ClassifierPipelines.pipelines.insert(original_index, final)
                ClassifierPipelines._rebuild_type_cache()
            else:
                ClassifierPipelines.add_pipeline(final)
        else:
            ClassifierPipelines.add_pipeline(final)
        ClassifierPipelines.store()
        self.close()
        self._refresh_callback()

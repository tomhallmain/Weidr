"""
PySide6 port of compare/classifier_management_window.py.

Contains two classes:
  - ClassifierActionModifyWindow  -- base modify dialog for classifier
    actions and prevalidations (shared fields, prototype support).
  - ClassifierManagementWindow    -- tabbed management window hosting
    ClassifierActionsTab and PrevalidationsTab.

Non-UI imports:
  - ClassifierAction, Prevalidation, ImageClassifierClassificationMode
    from compare.classifier_action
  - ClassifierActionsManager from compare.classifier_actions_manager
  - image_classifier_manager
    from image.image_classifier_manager (reuse policy)
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSlider, QTabWidget, QVBoxLayout, QWidget,
)

from compare.classifier_action import (
    ClassifierAction,
    ImageClassifierClassificationMode,
    Prevalidation,
)
from compare.classifier_actions_manager import ClassifierActionsManager
from image.image_classifier_manager import image_classifier_manager
from lib.fast_directory_picker_qt import get_existing_directory
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.config import config
from utils.constants import ClassifierActionType
from utils.translations import _
from utils.logging_setup import get_logger
logger = get_logger("classifier_management_window_qt")


# ======================================================================
# ClassifierActionModifyWindow
# ======================================================================
class ClassifierActionModifyWindow(SmartDialog):
    """
    Base modify dialog for classifier actions and prevalidations.

    Contains all shared UI elements: name, positives, negatives, validation
    type checkboxes, thresholds, action dropdown, image classifier selector,
    category multi-select, prototype directory fields.

    Subclassed by PrevalidationModifyWindow for lookahead / profile fields.
    """

    _instance: Optional[ClassifierActionModifyWindow] = None

    def __init__(
        self,
        parent: QWidget,
        app_actions,
        refresh_callback: Callable,
        classifier_action: Optional[ClassifierAction],
        window_title: Optional[str] = None,
        name_label_text: Optional[str] = None,
        new_name_default: Optional[str] = None,
        dimensions: str = "600x600",
    ) -> None:
        # Defaults based on type
        if window_title is None:
            if isinstance(classifier_action, Prevalidation):
                window_title = _("Modify Prevalidation")
                name_label_text = _("Prevalidation Name")
                new_name_default = _("New Prevalidation")
            else:
                window_title = _("Modify Classifier Action")
                name_label_text = _("Classifier Action Name")
                new_name_default = _("New Classifier Action")

        if classifier_action is None:
            classifier_action = ClassifierAction()

        title_str = f"{window_title}: {classifier_action.name}"
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=title_str,
            geometry=dimensions,
        )
        ClassifierActionModifyWindow._instance = self

        self._app_actions = app_actions
        self._refresh_callback = refresh_callback
        self._classifier_action = classifier_action
        self._name_label_text = name_label_text or _("Name")
        self._new_name_default = new_name_default or _("New")

        # Ensure image classifier is loaded for display
        if hasattr(self._classifier_action, "ensure_image_classifier_loaded"):
            self._classifier_action.ensure_image_classifier_loaded(
                app_actions.title_notify if app_actions else None
            )
        elif hasattr(self._classifier_action, "_ensure_image_classifier_loaded"):
            self._classifier_action._ensure_image_classifier_loaded(
                app_actions.title_notify if app_actions else None
            )

        self._build_ui()

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {AppStyle.BG_COLOR}; }}"
        )
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(6)
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        ca = self._classifier_action
        row = 0

        # -- Name ---------------------------------------------------------
        grid.addWidget(self._lbl(self._name_label_text), row, 0, Qt.AlignLeft)
        self._name_edit = QLineEdit(ca.name)
        grid.addWidget(self._name_edit, row, 1)
        row += 1

        # -- Positives ----------------------------------------------------
        self._positives_lbl = self._lbl(_("Positives"))
        grid.addWidget(self._positives_lbl, row, 0, Qt.AlignLeft)
        self._positives_edit = QLineEdit(ca.get_positives_str())
        grid.addWidget(self._positives_edit, row, 1)
        row += 1

        # -- Negatives ----------------------------------------------------
        self._negatives_lbl = self._lbl(_("Negatives"))
        grid.addWidget(self._negatives_lbl, row, 0, Qt.AlignLeft)
        self._negatives_edit = QLineEdit(ca.get_negatives_str())
        grid.addWidget(self._negatives_edit, row, 1)
        row += 1

        # -- Validation types ---------------------------------------------
        grid.addWidget(self._lbl(_("Validation Types")), row, 0, Qt.AlignLeft)
        self._use_embedding_cb = QCheckBox(_("Use Text Embeddings"))
        self._use_embedding_cb.setChecked(ca.use_embedding)
        self._use_embedding_cb.stateChanged.connect(self._update_ui_for_validation_types)
        grid.addWidget(self._use_embedding_cb, row, 1)
        row += 1

        self._use_classifier_cb = QCheckBox(_("Use Image Classifier"))
        self._use_classifier_cb.setChecked(ca.use_image_classifier)
        self._use_classifier_cb.stateChanged.connect(self._update_ui_for_validation_types)
        grid.addWidget(self._use_classifier_cb, row, 1)
        row += 1

        self._use_prompts_cb = QCheckBox(_("Use Prompts"))
        self._use_prompts_cb.setChecked(ca.use_prompts)
        self._use_prompts_cb.stateChanged.connect(self._update_ui_for_validation_types)
        grid.addWidget(self._use_prompts_cb, row, 1)
        row += 1

        self._use_prototype_cb = QCheckBox(_("Use Embedding Prototype"))
        self._use_prototype_cb.setChecked(ca.use_prototype)
        self._use_prototype_cb.stateChanged.connect(self._update_ui_for_validation_types)
        grid.addWidget(self._use_prototype_cb, row, 1)
        row += 1

        self._use_pseudostatic_dynamic_cb = QCheckBox(
            _("Match pseudo-static dynamic media (video/GIF/PDF with no meaningful variation)")
        )
        self._use_pseudostatic_dynamic_cb.setChecked(ca.use_pseudostatic_dynamic_media)
        self._use_pseudostatic_dynamic_cb.stateChanged.connect(self._update_ui_for_validation_types)
        grid.addWidget(self._use_pseudostatic_dynamic_cb, row, 1)
        row += 1

        # -- Text Embedding Threshold -------------------------------------
        self._text_thresh_lbl = self._lbl(_("Text Embedding Threshold"))
        grid.addWidget(self._text_thresh_lbl, row, 0, Qt.AlignLeft)
        self._text_thresh_slider = QSlider(Qt.Horizontal)
        self._text_thresh_slider.setRange(0, 100)
        self._text_thresh_slider.setValue(int(ca.text_embedding_threshold * 100))
        grid.addWidget(self._text_thresh_slider, row, 1)
        row += 1

        # -- Prototype Threshold ------------------------------------------
        self._proto_thresh_lbl = self._lbl(_("Embedding Prototype Threshold"))
        grid.addWidget(self._proto_thresh_lbl, row, 0, Qt.AlignLeft)
        self._proto_thresh_slider = QSlider(Qt.Horizontal)
        self._proto_thresh_slider.setRange(0, 100)
        self._proto_thresh_slider.setValue(int(ca.prototype_threshold * 100))
        grid.addWidget(self._proto_thresh_slider, row, 1)
        row += 1

        # -- Action -------------------------------------------------------
        grid.addWidget(self._lbl(_("Action")), row, 0, Qt.AlignLeft)
        self._action_combo = QComboBox()
        action_options = [k.get_translation() for k in ClassifierActionType]
        self._action_combo.addItems(action_options)
        self._action_combo.setCurrentText(ca.action.get_translation())
        grid.addWidget(self._action_combo, row, 1)
        row += 1

        # -- Action modifier ----------------------------------------------
        grid.addWidget(self._lbl(_("Action Modifier")), row, 0, Qt.AlignLeft)
        self._action_modifier_edit = QLineEdit(ca.action_modifier)
        self._action_modifier_hint_text = _(
            "Leave blank to auto-route by image-classifier selected category."
        )
        self._action_modifier_edit.setToolTip(self._action_modifier_hint_text)
        grid.addWidget(self._action_modifier_edit, row, 1)
        row += 1

        # -- Image classifier name ----------------------------------------
        self._ic_name_lbl = self._lbl(_("Image Classifier Name"))
        grid.addWidget(self._ic_name_lbl, row, 0, Qt.AlignLeft)
        self._ic_name_combo = QComboBox()
        name_options = [""]
        name_options.extend(image_classifier_manager.get_model_names())
        self._ic_name_combo.addItems(name_options)
        self._ic_name_combo.setCurrentText(ca.image_classifier_name)
        self._ic_name_combo.currentTextChanged.connect(self._on_image_classifier_changed)
        grid.addWidget(self._ic_name_combo, row, 1)
        row += 1

        # -- Image classifier categories (multi-select list) ---------------
        self._ic_cat_lbl = self._lbl(_("Image Classifier Selected Category"))
        grid.addWidget(self._ic_cat_lbl, row, 0, Qt.AlignLeft | Qt.AlignTop)
        self._ic_cat_list = QListWidget()
        self._ic_cat_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._ic_cat_list.setMaximumHeight(100)
        self._populate_ic_categories()
        grid.addWidget(self._ic_cat_list, row, 1)
        row += 1

        # -- Image classifier mode ----------------------------------------
        self._ic_mode_lbl = self._lbl(_("Image Classifier Classification Mode"))
        grid.addWidget(self._ic_mode_lbl, row, 0, Qt.AlignLeft)
        self._ic_mode_combo = QComboBox()
        self._ic_mode_combo.addItem(_("Selected categories"), ImageClassifierClassificationMode.SELECTED_CATEGORIES.value)
        self._ic_mode_combo.addItem(_("Model strategy"), ImageClassifierClassificationMode.MODEL_STRATEGY.value)
        mode_value = ca.classification_mode.value if hasattr(ca, "classification_mode") else ImageClassifierClassificationMode.SELECTED_CATEGORIES.value
        idx = self._ic_mode_combo.findData(mode_value)
        self._ic_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._ic_mode_combo.currentIndexChanged.connect(self._update_ui_for_validation_types)
        grid.addWidget(self._ic_mode_combo, row, 1)
        row += 1

        # -- Prototype directory ------------------------------------------
        self._proto_dir_lbl = self._lbl(_("Prototype Directory"))
        grid.addWidget(self._proto_dir_lbl, row, 0, Qt.AlignLeft)
        proto_row = QHBoxLayout()
        self._proto_dir_edit = QLineEdit(ca.prototype_directory)
        proto_row.addWidget(self._proto_dir_edit, 1)
        proto_browse_btn = QPushButton(_("Browse..."))
        proto_browse_btn.clicked.connect(self._browse_prototype_directory)
        proto_row.addWidget(proto_browse_btn)
        grid.addLayout(proto_row, row, 1)
        row += 1

        # -- Force recalculate prototype button ---------------------------
        self._force_recalc_btn = QPushButton(_("Force Recalculate Prototype"))
        self._force_recalc_btn.clicked.connect(self._force_recalculate_prototype)
        grid.addWidget(self._force_recalc_btn, row, 1, Qt.AlignLeft)
        row += 1

        # -- Negative prototype directory ---------------------------------
        self._neg_proto_dir_lbl = self._lbl(_("Negative Prototype Directory (Optional)"))
        grid.addWidget(self._neg_proto_dir_lbl, row, 0, Qt.AlignLeft)
        neg_proto_row = QHBoxLayout()
        self._neg_proto_dir_edit = QLineEdit(ca.negative_prototype_directory)
        neg_proto_row.addWidget(self._neg_proto_dir_edit, 1)
        neg_proto_browse_btn = QPushButton(_("Browse..."))
        neg_proto_browse_btn.clicked.connect(self._browse_negative_prototype_directory)
        neg_proto_row.addWidget(neg_proto_browse_btn)
        grid.addLayout(neg_proto_row, row, 1)
        row += 1

        # -- Negative prototype lambda ------------------------------------
        self._neg_proto_lambda_lbl = self._lbl(_("Negative Prototype Weight (\u03bb)"))
        grid.addWidget(self._neg_proto_lambda_lbl, row, 0, Qt.AlignLeft)
        self._neg_proto_lambda_slider = QSlider(Qt.Horizontal)
        self._neg_proto_lambda_slider.setRange(0, 100)
        self._neg_proto_lambda_slider.setValue(
            int(ca.negative_prototype_lambda * 100)
        )
        grid.addWidget(self._neg_proto_lambda_slider, row, 1)
        row += 1

        # -- Subclass-specific fields hook --------------------------------
        row = self.add_specific_fields(grid, row)

        # -- Is active checkbox -------------------------------------------
        grid.addWidget(self._lbl(_("Should Run")), row, 0, Qt.AlignLeft)
        self._is_active_cb = QCheckBox(_("Enable this classifier action"))
        self._is_active_cb.setChecked(ca.is_active)
        grid.addWidget(self._is_active_cb, row, 1)
        row += 1

        # -- Done button --------------------------------------------------
        done_btn = QPushButton(_("Done"))
        done_btn.clicked.connect(self._finalize)
        grid.addWidget(done_btn, row, 0, Qt.AlignLeft)
        row += 1

        grid.setRowStretch(row, 1)

        # Initial visibility
        self._update_ui_for_validation_types()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def add_specific_fields(self, grid: QGridLayout, row: int) -> int:
        """Override in subclasses to add type-specific fields. Returns next row."""
        return row

    def _finalize_specific(self) -> None:
        """Override in subclasses for extra finalization."""
        pass

    def _update_specific_ui_for_validation_types(self) -> None:
        """Override in subclasses for extra visibility toggling."""
        pass

    # ------------------------------------------------------------------
    # Visibility toggling
    # ------------------------------------------------------------------
    def _update_ui_for_validation_types(self) -> None:
        use_emb = self._use_embedding_cb.isChecked()
        use_ic = self._use_classifier_cb.isChecked()
        use_pr = self._use_prompts_cb.isChecked()
        use_proto = self._use_prototype_cb.isChecked()

        # Positives / negatives
        show_pn = use_emb or use_pr
        self._positives_lbl.setVisible(show_pn)
        self._positives_edit.setVisible(show_pn)
        self._negatives_lbl.setVisible(show_pn)
        self._negatives_edit.setVisible(show_pn)

        # Text embedding threshold
        self._text_thresh_lbl.setVisible(use_emb)
        self._text_thresh_slider.setVisible(use_emb)

        # Image classifier fields
        self._ic_name_lbl.setVisible(use_ic)
        self._ic_name_combo.setVisible(use_ic)
        mode = ImageClassifierClassificationMode.from_value(self._ic_mode_combo.currentData())
        uses_selected_categories = mode == ImageClassifierClassificationMode.SELECTED_CATEGORIES
        if use_ic and uses_selected_categories:
            self._action_modifier_edit.setPlaceholderText(self._action_modifier_hint_text)
        else:
            self._action_modifier_edit.setPlaceholderText("")
        self._ic_mode_lbl.setVisible(use_ic)
        self._ic_mode_combo.setVisible(use_ic)
        self._ic_cat_lbl.setVisible(use_ic and uses_selected_categories)
        self._ic_cat_list.setVisible(use_ic and uses_selected_categories)

        # Prototype fields
        self._proto_dir_lbl.setVisible(use_proto)
        self._proto_dir_edit.setVisible(use_proto)
        self._force_recalc_btn.setVisible(use_proto)
        self._neg_proto_dir_lbl.setVisible(use_proto)
        self._neg_proto_dir_edit.setVisible(use_proto)
        self._neg_proto_lambda_lbl.setVisible(use_proto)
        self._neg_proto_lambda_slider.setVisible(use_proto)
        self._proto_thresh_lbl.setVisible(use_proto)
        self._proto_thresh_slider.setVisible(use_proto)

        self._update_specific_ui_for_validation_types()

    # ------------------------------------------------------------------
    # Image classifier category helpers
    # ------------------------------------------------------------------
    def _populate_ic_categories(self) -> None:
        ca = self._classifier_action
        self._ic_cat_list.clear()
        for cat in ca.image_classifier_categories:
            item = QListWidgetItem(cat)
            self._ic_cat_list.addItem(item)
            if cat in ca.image_classifier_selected_categories:
                item.setSelected(True)

    def _on_image_classifier_changed(self, name: str) -> None:
        self._classifier_action.set_image_classifier(name)
        self._populate_ic_categories()

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------
    def _browse_prototype_directory(self) -> None:
        d = get_existing_directory(
            self, _("Select Prototype Directory")
        )
        if d:
            self._proto_dir_edit.setText(d)

    def _browse_negative_prototype_directory(self) -> None:
        d = get_existing_directory(
            self, _("Select Negative Prototype Directory")
        )
        if d:
            self._neg_proto_dir_edit.setText(d)

    def _force_recalculate_prototype(self) -> None:
        from compare.embedding_prototype import EmbeddingPrototype

        notify_cb = (
            self._app_actions.title_notify if self._app_actions else None
        )
        success_count = 0

        directory = self._proto_dir_edit.text().strip()
        if directory:
            if not os.path.isdir(directory):
                logger.error(f"Prototype directory does not exist: {directory}")
            else:
                try:
                    prototype = EmbeddingPrototype.calculate_prototype_from_directory(
                        directory, force_recalculate=True, notify_callback=notify_cb
                    )
                    if prototype is not None:
                        self._classifier_action.prototype_directory = directory
                        self._classifier_action._cached_prototype = prototype
                        success_count += 1
                    else:
                        logger.error("Failed to recalculate positive prototype")
                except Exception as e:
                    logger.error(f"Error recalculating positive prototype: {e}")

        neg_dir = self._neg_proto_dir_edit.text().strip()
        if neg_dir:
            if not os.path.isdir(neg_dir):
                logger.error(f"Negative prototype directory does not exist: {neg_dir}")
            else:
                try:
                    neg_proto = EmbeddingPrototype.calculate_prototype_from_directory(
                        neg_dir, force_recalculate=True, notify_callback=notify_cb
                    )
                    if neg_proto is not None:
                        self._classifier_action.negative_prototype_directory = neg_dir
                        self._classifier_action._cached_negative_prototype = neg_proto
                        success_count += 1
                    else:
                        logger.error("Failed to recalculate negative prototype")
                except Exception as e:
                    logger.error(f"Error recalculating negative prototype: {e}")

        if notify_cb and success_count > 0:
            notify_cb(_("Prototypes recalculated successfully"))

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        ca = self._classifier_action
        ca.name = self._name_edit.text().strip()

        pos_text = self._positives_edit.text().strip()
        if pos_text != ClassifierAction.NO_POSITIVES_STR:
            ca.set_positives(pos_text)

        neg_text = self._negatives_edit.text().strip()
        if neg_text != ClassifierAction.NO_NEGATIVES_STR:
            ca.set_negatives(neg_text)

        ca.text_embedding_threshold = self._text_thresh_slider.value() / 100.0
        ca.threshold = ca.text_embedding_threshold
        ca.prototype_threshold = self._proto_thresh_slider.value() / 100.0

        ca.use_embedding = self._use_embedding_cb.isChecked()
        ca.use_image_classifier = self._use_classifier_cb.isChecked()
        ca.use_prompts = self._use_prompts_cb.isChecked()
        ca.use_prototype = self._use_prototype_cb.isChecked()
        ca.use_pseudostatic_dynamic_media = self._use_pseudostatic_dynamic_cb.isChecked()

        ca.action = ClassifierActionType.get_action(
            self._action_combo.currentText()
        )
        ca.action_modifier = self._action_modifier_edit.text().strip()

        ca.image_classifier_selected_categories = [
            item.text() for item in self._ic_cat_list.selectedItems()
        ]
        ca.classification_mode = ImageClassifierClassificationMode.from_value(
            self._ic_mode_combo.currentData()
        )

        ca.prototype_directory = self._proto_dir_edit.text().strip()
        ca.negative_prototype_directory = self._neg_proto_dir_edit.text().strip()
        ca.negative_prototype_lambda = (
            self._neg_proto_lambda_slider.value() / 100.0
        )
        ca.is_active = self._is_active_cb.isChecked()

        self._finalize_specific()
        ca.validate()
        ca.mark_runtime_valid()
        self.close()
        self._refresh_callback(ca)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        lbl.setWordWrap(True)
        return lbl

    def closeEvent(self, event) -> None:  # noqa: N802
        ClassifierActionModifyWindow._instance = None
        super().closeEvent(event)


# ======================================================================
# ClassifierManagementWindow
# ======================================================================
class ClassifierManagementWindow(SmartDialog):
    """
    Tabbed management window hosting ClassifierActionsTab and
    PrevalidationsTab via QTabWidget.
    """

    _instance: Optional[ClassifierManagementWindow] = None

    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Classifier Management"),
            geometry="1200x700",
        )
        ClassifierManagementWindow._instance = self
        self._app_actions = app_actions

        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # Lazy imports to avoid circulars
        from ui.compare.classifier_actions_tab_qt import ClassifierActionsTab
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        from ui.compare.prevalidations_tab_qt import PrevalidationsTab

        self._classifier_actions_tab = ClassifierActionsTab(
            self._tabs, app_actions
        )
        self._prevalidations_tab = PrevalidationsTab(
            self._tabs, app_actions
        )
        self._classifier_pipelines_tab = ClassifierPipelinesTab(
            self._tabs, app_actions
        )

        self._tabs.addTab(
            self._classifier_actions_tab, _("Classifier Actions")
        )
        self._tabs.addTab(self._prevalidations_tab, _("Prevalidations"))
        self._tabs.addTab(
            self._classifier_pipelines_tab, _("Pipelines")
        )

        if not config.enable_prevalidations:
            self._tabs.setTabEnabled(1, False)
            self._tabs.setTabToolTip(
                1,
                _("Prevalidations are disabled. Enable them in the Prevalidations tab settings."),
            )

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(
            self.close
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def show_window(cls, parent: QWidget, app_actions) -> None:
        if cls._instance is not None:
            try:
                if cls._instance.isVisible():
                    cls._instance.raise_()
                    cls._instance.activateWindow()
                    return
            except Exception:
                pass
        win = cls(parent, app_actions)
        win.show()

    # ------------------------------------------------------------------
    # Static persistence helpers (matching original API)
    # ------------------------------------------------------------------
    @staticmethod
    def set_prevalidations() -> None:
        ClassifierActionsManager.load_prevalidations()

    @staticmethod
    def store_prevalidations() -> None:
        ClassifierActionsManager.store_prevalidations()

    @staticmethod
    def set_classifier_actions() -> None:
        ClassifierActionsManager.load_classifier_actions()

    @staticmethod
    def store_classifier_actions() -> None:
        ClassifierActionsManager.store_classifier_actions()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        ClassifierManagementWindow._instance = None
        super().closeEvent(event)

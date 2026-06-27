from __future__ import annotations

import os
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from compare.classifier_actions_manager import ClassifierActionsManager
from extensions.hf_hub_api import HfHubApiBackend
from image.image_classifier_manager import image_classifier_manager
from image.image_classifier_model_config import ImageClassifierModelConfig
from lib.multi_display_qt import SmartDialog
from utils.config import config
from utils.constants import HfHubSortDirection, HfHubSortOption, HfHubVisualMediaTask
from utils.logging_setup import get_logger
from utils.translations import _
logger = get_logger("hf_model_manager_window_qt")


class _TextPreviewDialog(SmartDialog):
    def __init__(self, parent: QWidget, title: str, text: str):
        super().__init__(parent=parent, position_parent=parent, title=title, geometry="950x700")
        layout = QVBoxLayout(self)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlainText(text)
        layout.addWidget(self._text)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)


class _InstalledModelEditDialog(SmartDialog):
    """Popup editor for local/HF installed model definitions."""

    def __init__(
        self,
        parent: QWidget,
        title: str,
        initial_model: dict[str, Any],
        api_getter: Callable[[], HfHubApiBackend],
        save_callback: Callable[[dict[str, Any], str], None],
        model_file_extensions: set[str],
    ):
        super().__init__(parent=parent, position_parent=parent, title=title, geometry="980x600")
        self._api_getter = api_getter
        self._save_callback = save_callback
        self._model_file_extensions = model_file_extensions
        self._initial_name = str(initial_model.get("model_name", ""))
        self._repo_files_cache: dict[str, list[str]] = {}

        model_kwargs = dict(initial_model.get("model_kwargs", {}))

        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel(_("Model name")))
        self._model_name_edit = QLineEdit(str(initial_model.get("model_name", "")))
        row1.addWidget(self._model_name_edit, stretch=1)
        row1.addWidget(QLabel(_("Backend")))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["auto", "pytorch", "hdf5"])
        self._backend_combo.setCurrentText(str(initial_model.get("backend", "auto")))
        row1.addWidget(self._backend_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel(_("Model location")))
        self._model_location_edit = QLineEdit(str(initial_model.get("model_location", "")))
        row2.addWidget(self._model_location_edit, stretch=1)
        browse_btn = QPushButton(_("Browse..."))
        browse_btn.clicked.connect(self._browse_model_location)
        row2.addWidget(browse_btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel(_("Categories")))
        self._categories_edit = QLineEdit(",".join([str(c) for c in (initial_model.get("model_categories") or [])]))
        self._categories_edit.setPlaceholderText(_("Comma-separated categories"))
        row3.addWidget(self._categories_edit, stretch=1)
        layout.addLayout(row3)

        row3_shape = QHBoxLayout()
        input_shape_tip = _(
            "Optional. PyTorch: width and height for the internal resize "
            "(defaults to 224x224 when left blank). "
            "HDF5 (.h5): the loaded model defines its own input size; this "
            "value is stored in config but is not used when loading."
        )
        lbl_input_shape = QLabel(_("Input shape (WxH)"))
        lbl_input_shape.setToolTip(input_shape_tip)
        row3_shape.addWidget(lbl_input_shape)
        self._input_shape_edit = QLineEdit(
            _InstalledModelEditDialog._format_initial_input_shape(initial_model)
        )
        self._input_shape_edit.setPlaceholderText(_("224, 224"))
        self._input_shape_edit.setToolTip(input_shape_tip)
        row3_shape.addWidget(self._input_shape_edit, stretch=1)
        layout.addLayout(row3_shape)

        row3b = QHBoxLayout()
        row3b.addWidget(QLabel(_("Positive groups")))
        self._positive_groups_edit = QLineEdit(self._format_positive_groups(initial_model.get("positive_groups")))
        self._positive_groups_edit.setPlaceholderText(_("group1_catA,group1_catB; group2_catA,group2_catB"))
        row3b.addWidget(self._positive_groups_edit, stretch=1)
        layout.addLayout(row3b)

        row3c = QHBoxLayout()
        row3c.addWidget(QLabel(_("Neutral categories")))
        self._neutral_categories_edit = QLineEdit(",".join([str(c) for c in (initial_model.get("neutral_categories") or [])]))
        self._neutral_categories_edit.setPlaceholderText(_("Optional, comma-separated"))
        row3c.addWidget(self._neutral_categories_edit, stretch=1)
        row3c.addWidget(QLabel(_("Severity order")))
        self._severity_order_edit = QLineEdit(",".join([str(c) for c in (initial_model.get("severity_order") or [])]))
        self._severity_order_edit.setPlaceholderText(_("Optional, comma-separated"))
        row3c.addWidget(self._severity_order_edit, stretch=1)
        layout.addLayout(row3c)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel(_("HF repo id")))
        self._hf_repo_id_edit = QLineEdit(str(initial_model.get("hf_repo_id", "")))
        row4.addWidget(self._hf_repo_id_edit, stretch=1)
        load_files_btn = QPushButton(_("Load Repo Files"))
        load_files_btn.clicked.connect(self._load_repo_files)
        row4.addWidget(load_files_btn)
        view_card_btn = QPushButton(_("View Model Card"))
        view_card_btn.clicked.connect(self._view_model_card)
        row4.addWidget(view_card_btn)
        layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel(_("Repo file")))
        self._repo_file_combo = QComboBox()
        self._repo_file_combo.setEditable(True)
        self._repo_file_combo.addItem(str(initial_model.get("hf_selected_filename", "")))
        row5.addWidget(self._repo_file_combo, stretch=1)
        row5.addWidget(QLabel(_("HF snapshot path")))
        self._hf_pretrained_path_edit = QLineEdit(str(model_kwargs.get("hf_pretrained_path", "")))
        row5.addWidget(self._hf_pretrained_path_edit, stretch=1)
        use_file_btn = QPushButton(_("Use Repo File Path"))
        use_file_btn.clicked.connect(self._apply_repo_file_to_model_location)
        row5.addWidget(use_file_btn)
        layout.addLayout(row5)

        row6 = QHBoxLayout()
        self._use_transformers_cb = QCheckBox(_("Use Transformers AutoModel"))
        self._use_transformers_cb.setChecked(bool(model_kwargs.get("use_transformers_auto_model", False)))
        row6.addWidget(self._use_transformers_cb)
        row6.addWidget(QLabel(_("Arch module")))
        self._arch_module_edit = QLineEdit(str(model_kwargs.get("architecture_module_name", "")))
        row6.addWidget(self._arch_module_edit, stretch=1)
        row6.addWidget(QLabel(_("Arch class")))
        self._arch_class_edit = QLineEdit(str(model_kwargs.get("architecture_class_path", "")))
        row6.addWidget(self._arch_class_edit, stretch=1)
        layout.addLayout(row6)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(cancel_btn)
        save_btn = QPushButton(_("Save"))
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _format_initial_input_shape(initial_model: dict[str, Any]) -> str:
        raw = initial_model.get("input_shape")
        if raw is None:
            mk = initial_model.get("model_kwargs")
            if isinstance(mk, dict):
                raw = mk.get("input_shape")
        tup = ImageClassifierModelConfig.parse_input_shape(raw)
        if tup is None:
            return ""
        return f"{tup[0]}, {tup[1]}"

    def _browse_model_location(self) -> None:
        path, selected_filter = QFileDialog.getOpenFileName(
            self, _("Select model file")
        )
        if path:
            self._model_location_edit.setText(path)

    def _load_repo_files(self) -> None:
        repo_id = (self._hf_repo_id_edit.text() or "").strip()
        if not repo_id:
            return
        try:
            if repo_id in self._repo_files_cache:
                files = self._repo_files_cache[repo_id]
            else:
                files = self._api_getter().list_model_files(repo_id)
                self._repo_files_cache[repo_id] = files
            preferred = [f for f in files if os.path.splitext(f)[1].lower() in self._model_file_extensions]
            values = preferred if preferred else files
            current = self._repo_file_combo.currentText().strip()
            self._repo_file_combo.clear()
            for f in values:
                self._repo_file_combo.addItem(f)
            if current and current in values:
                self._repo_file_combo.setCurrentText(current)
            elif values:
                self._repo_file_combo.setCurrentText(values[0])
        except Exception:
            pass

    def _view_model_card(self) -> None:
        repo_id = (self._hf_repo_id_edit.text() or "").strip()
        if not repo_id:
            return
        try:
            card_text = self._api_getter().get_model_card_text(repo_id)
            _TextPreviewDialog(
                parent=self,
                title=_("Model Card - {0}").format(repo_id),
                text=card_text,
            ).show()
        except Exception:
            pass

    def _apply_repo_file_to_model_location(self) -> None:
        selected_file = (self._repo_file_combo.currentText() or "").strip()
        if not selected_file:
            return
        snapshot_path = (self._hf_pretrained_path_edit.text() or "").strip()
        if snapshot_path:
            candidate = os.path.join(snapshot_path, selected_file.replace("/", os.sep))
            self._model_location_edit.setText(candidate)
        else:
            self._model_location_edit.setText(selected_file)

    @staticmethod
    def _format_positive_groups(groups: Any) -> str:
        if not isinstance(groups, list):
            return ""
        out: list[str] = []
        for group in groups:
            if isinstance(group, list):
                cats = [str(c).strip() for c in group if str(c).strip()]
                if cats:
                    out.append(",".join(cats))
        return "; ".join(out)

    @staticmethod
    def _parse_positive_groups(text: str) -> list[list[str]]:
        groups: list[list[str]] = []
        for raw_group in (text or "").split(";"):
            cats = [c.strip() for c in raw_group.split(",") if c.strip()]
            if cats:
                groups.append(cats)
        return groups

    def _save(self) -> None:
        model_name = (self._model_name_edit.text() or "").strip()
        model_location = (self._model_location_edit.text() or "").strip()
        categories = [c.strip() for c in (self._categories_edit.text() or "").split(",") if c.strip()]
        backend = (self._backend_combo.currentText() or "auto").strip()
        if not model_name or not model_location or not categories:
            QMessageBox.warning(
                self,
                _("Missing fields"),
                _("Model name, model file path, and at least one category are required."),
            )
            return
        model_details: dict[str, Any] = {
            "model_name": model_name,
            "model_location": model_location,
            "model_categories": categories,
            "backend": backend,
        }
        positive_groups = self._parse_positive_groups((self._positive_groups_edit.text() or "").strip())
        neutral_categories = [c.strip() for c in (self._neutral_categories_edit.text() or "").split(",") if c.strip()]
        severity_order = [c.strip() for c in (self._severity_order_edit.text() or "").split(",") if c.strip()]
        if positive_groups:
            model_details["positive_groups"] = positive_groups
        if neutral_categories:
            model_details["neutral_categories"] = neutral_categories
        if severity_order:
            model_details["severity_order"] = severity_order
        raw_shape = (self._input_shape_edit.text() or "").strip()
        if raw_shape:
            parsed_shape = ImageClassifierModelConfig.parse_input_shape(raw_shape)
            if parsed_shape is None:
                QMessageBox.warning(
                    self,
                    _("Invalid input shape"),
                    _("Use two positive integers, for example 224, 224 or 384x384."),
                )
                return
            model_details["input_shape"] = [parsed_shape[0], parsed_shape[1]]
        repo_id = (self._hf_repo_id_edit.text() or "").strip()
        selected_file = (self._repo_file_combo.currentText() or "").strip()
        if repo_id:
            model_details["hf_repo_id"] = repo_id
        if selected_file:
            model_details["hf_selected_filename"] = selected_file

        model_kwargs: dict[str, Any] = {}
        if self._use_transformers_cb.isChecked():
            model_kwargs["use_transformers_auto_model"] = True
            hf_pretrained_path = (self._hf_pretrained_path_edit.text() or "").strip()
            if hf_pretrained_path:
                model_kwargs["hf_pretrained_path"] = hf_pretrained_path
        arch_module = (self._arch_module_edit.text() or "").strip()
        arch_class = (self._arch_class_edit.text() or "").strip()
        if arch_module:
            model_kwargs["architecture_module_name"] = arch_module
        if arch_class:
            model_kwargs["architecture_class_path"] = arch_class
        if model_kwargs:
            model_details["model_kwargs"] = model_kwargs

        try:
            normalized = ImageClassifierModelConfig.from_dict(model_details, logger=logger)
        except Exception as e:
            QMessageBox.warning(self, _("Invalid model configuration"), str(e))
            return
        self._save_callback(normalized.to_dict(), self._initial_name)
        self.close()


class _SearchResultTreeItem(QTreeWidgetItem):
    """Tree item with numeric-aware sorting for downloads/likes columns."""

    NUMERIC_COLUMNS = {2, 3}

    def __lt__(self, other):
        tree = self.treeWidget()
        if tree is None:
            return super().__lt__(other)
        col = tree.sortColumn()
        if col in self.NUMERIC_COLUMNS:
            return self._int_for_col(col) < other._int_for_col(col)
        return super().__lt__(other)

    def _int_for_col(self, col: int) -> int:
        user_data = self.data(col, Qt.ItemDataRole.UserRole)
        if user_data is not None:
            try:
                return int(user_data)
            except Exception:
                pass
        try:
            return int((self.text(col) or "0").replace(",", ""))
        except Exception:
            return 0


class _ClassifierTestWorker(QThread):
    """Run a single classifier on one image in a background thread."""

    finished = Signal(str, str, object)  # model_name, image_path, result_dict
    failed = Signal(str, str, str)       # model_name, image_path, error_message

    def __init__(self, model_name: str, image_path: str):
        super().__init__()
        self._model_name = model_name
        self._image_path = image_path

    def run(self):
        try:
            classifier = image_classifier_manager.get_classifier(self._model_name)
            if classifier is None or not classifier.can_run:
                self.failed.emit(self._model_name, self._image_path,
                                 "Model failed to initialize (can_run=False).")
                return
            # Evict cache entry so we always get a live result.
            classifier.predictions_cache.pop(self._image_path, None)
            ranked = classifier.predict_image_ranked(self._image_path)
            classification = classifier.classify_image(self._image_path)
            self.finished.emit(self._model_name, self._image_path,
                               {"classification": classification, "ranked": ranked})
        except Exception as exc:
            self.failed.emit(self._model_name, self._image_path, str(exc))


class HfModelManagerWindow(SmartDialog):
    """Manage image classifier models from HF Hub and local config."""

    _instance: Optional["HfModelManagerWindow"] = None
    _MODEL_FILE_EXTENSIONS = {
        ".safetensors",
        ".ckpt",
        ".bin",
        ".onnx",
        ".pt",
        ".pth",
        ".h5",
        ".keras",
    }

    def __init__(self, parent: QWidget, app_actions):
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Model Manager"),
            geometry="1100x700",
        )
        HfModelManagerWindow._instance = self
        self._app_actions = app_actions
        self._hf_api: Optional[HfHubApiBackend] = None
        self._repo_files_cache: dict[str, list[str]] = {}
        self._test_worker: Optional[_ClassifierTestWorker] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        search_page = QWidget()
        installed_page = QWidget()
        self._tabs.addTab(search_page, _("HF Hub Search"))
        self._tabs.addTab(installed_page, _("Installed Models"))

        self._build_search_tab(search_page)
        self._build_installed_tab(installed_page)
        self._refresh_installed_models()
        self._tabs.setCurrentIndex(1)

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    @classmethod
    def show_window(cls, parent: QWidget, app_actions):
        if cls._instance is not None:
            try:
                if cls._instance.isVisible():
                    if cls._instance.isMinimized():
                        cls._instance.showNormal()
                    cls._instance.raise_()
                    cls._instance.activateWindow()
                    return
                else:
                    cls._instance = None
            except Exception:
                cls._instance = None
        win = cls(parent, app_actions)
        win.show()

    def reject(self) -> None:  # noqa: N802
        HfModelManagerWindow._instance = None
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        HfModelManagerWindow._instance = None
        super().closeEvent(event)

    def _build_search_tab(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)

        # Query and ordering controls
        query_row = QHBoxLayout()
        query_row.addWidget(QLabel(_("Search")))
        self._query_edit = QLineEdit()
        self._query_edit.setPlaceholderText(_("e.g. nsfw, classifier, coherence"))
        self._query_edit.returnPressed.connect(self._search)
        query_row.addWidget(self._query_edit, stretch=1)

        query_row.addWidget(QLabel(_("Task")))
        self._task_combo = QComboBox()
        for task in HfHubVisualMediaTask:
            self._task_combo.addItem(task.display(), task.value)
        self._task_combo.setCurrentText(HfHubVisualMediaTask.IMAGE_CLASSIFICATION.display())
        query_row.addWidget(self._task_combo)

        query_row.addWidget(QLabel(_("Sort")))
        self._sort_combo = QComboBox()
        for sort_option in HfHubSortOption:
            self._sort_combo.addItem(sort_option.display(), sort_option.value)
        self._sort_combo.setCurrentText(HfHubSortOption.DOWNLOADS.display())
        query_row.addWidget(self._sort_combo)

        query_row.addWidget(QLabel(_("Direction")))
        self._direction_combo = QComboBox()
        for direction in HfHubSortDirection:
            self._direction_combo.addItem(direction.display(), direction.value)
        self._direction_combo.setCurrentText(HfHubSortDirection.DESCENDING.display())
        query_row.addWidget(self._direction_combo)

        query_row.addWidget(QLabel(_("Limit")))
        self._limit_combo = QComboBox()
        for value in ("25", "50", "100", "200"):
            self._limit_combo.addItem(value)
        self._limit_combo.setCurrentText("100")
        query_row.addWidget(self._limit_combo)

        self._include_gated_cb = QCheckBox(_("Include gated"))
        self._include_gated_cb.setChecked(True)
        query_row.addWidget(self._include_gated_cb)

        search_btn = QPushButton(_("Search"))
        search_btn.clicked.connect(self._search)
        query_row.addWidget(search_btn)
        layout.addLayout(query_row)

        # Search results
        self._search_tree = QTreeWidget()
        self._search_tree.setHeaderLabels(
            [_("Repo"), _("Task"), _("Downloads"), _("Likes"), _("License"), _("Gated")]
        )
        self._search_tree.setRootIsDecorated(False)
        self._search_tree.setAlternatingRowColors(True)
        self._search_tree.setSortingEnabled(True)
        self._search_tree.itemSelectionChanged.connect(self._on_repo_selection_changed)
        hdr = self._search_tree.header()
        hdr.setStretchLastSection(True)
        self._search_tree.sortByColumn(2, Qt.SortOrder.DescendingOrder)
        layout.addWidget(self._search_tree)

        # Download + install controls
        install_row_1 = QHBoxLayout()
        install_row_1.addWidget(QLabel(_("Model file")))
        self._filename_combo = QComboBox()
        self._filename_combo.setEditable(True)
        self._filename_combo.addItem("model.safetensors")
        install_row_1.addWidget(self._filename_combo, stretch=1)
        load_files_btn = QPushButton(_("Load Repo Files"))
        load_files_btn.clicked.connect(self._load_selected_repo_files)
        install_row_1.addWidget(load_files_btn)
        card_btn = QPushButton(_("View Model Card"))
        card_btn.clicked.connect(self._view_model_card)
        install_row_1.addWidget(card_btn)
        dl_btn = QPushButton(_("Download and Install"))
        dl_btn.clicked.connect(self._download_and_install_selected)
        install_row_1.addWidget(dl_btn)
        layout.addLayout(install_row_1)

        install_row_2 = QHBoxLayout()
        install_row_2.addWidget(QLabel(_("Model name")))
        self._model_name_edit = QLineEdit()
        install_row_2.addWidget(self._model_name_edit, stretch=1)
        install_row_2.addWidget(QLabel(_("Categories")))
        self._categories_edit = QLineEdit("positive,negative")
        self._categories_edit.setPlaceholderText(_("Comma-separated categories"))
        install_row_2.addWidget(self._categories_edit, stretch=1)
        install_row_2.addWidget(QLabel(_("Backend")))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["auto", "pytorch", "hdf5"])
        install_row_2.addWidget(self._backend_combo)
        layout.addLayout(install_row_2)

        install_row_3 = QHBoxLayout()
        self._use_transformers_auto_model_cb = QCheckBox(_("Use Transformers AutoModel"))
        self._use_transformers_auto_model_cb.setToolTip(
            _("Recommended for HF model repos with config.json and processor files.")
        )
        self._use_transformers_auto_model_cb.setChecked(True)
        install_row_3.addWidget(self._use_transformers_auto_model_cb)

        install_row_3.addWidget(QLabel(_("Arch module")))
        self._arch_module_edit = QLineEdit()
        self._arch_module_edit.setPlaceholderText("architecture_module_name")
        install_row_3.addWidget(self._arch_module_edit, stretch=1)

        install_row_3.addWidget(QLabel(_("Arch class")))
        self._arch_class_edit = QLineEdit()
        self._arch_class_edit.setPlaceholderText("architecture_class_path (optional)")
        install_row_3.addWidget(self._arch_class_edit, stretch=1)
        layout.addLayout(install_row_3)

    def _build_installed_tab(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)

        self._installed_notice_label = QLabel("")
        self._installed_notice_label.setWordWrap(True)
        layout.addWidget(self._installed_notice_label)

        self._installed_tree = QTreeWidget()
        self._installed_tree.setHeaderLabels(
            [
                _("Model Name"),
                _("Backend"),
                _("Categories"),
                _("Model Location"),
                _("Config"),
            ]
        )
        self._installed_tree.setRootIsDecorated(False)
        self._installed_tree.setAlternatingRowColors(True)
        self._installed_tree.setSortingEnabled(True)
        self._installed_tree.itemDoubleClicked.connect(
            lambda *_: self._edit_selected_installed_model()
        )
        hdr = self._installed_tree.header()
        hdr.setStretchLastSection(True)
        layout.addWidget(self._installed_tree)

        btn_row = QHBoxLayout()
        add_btn = QPushButton(_("Add New"))
        add_btn.clicked.connect(self._add_installed_model)
        btn_row.addWidget(add_btn)

        edit_btn = QPushButton(_("Edit Selected"))
        edit_btn.clicked.connect(self._edit_selected_installed_model)
        btn_row.addWidget(edit_btn)

        self._test_btn = QPushButton(_("Test on Current Image"))
        self._test_btn.clicked.connect(self._test_selected_model_on_current_image)
        btn_row.addWidget(self._test_btn)

        load_files_btn = QPushButton(_("Load Repo Files"))
        load_files_btn.clicked.connect(self._load_repo_files_for_installed_selection)
        btn_row.addWidget(load_files_btn)

        view_card_btn = QPushButton(_("View Model Card"))
        view_card_btn.clicked.connect(self._view_model_card_for_installed_selection)
        btn_row.addWidget(view_card_btn)

        refresh_btn = QPushButton(_("Refresh"))
        refresh_btn.clicked.connect(self._refresh_installed_models)
        btn_row.addWidget(refresh_btn)

        remove_btn = QPushButton(_("Remove Selected"))
        remove_btn.clicked.connect(self._remove_selected_installed_model)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _api(self) -> HfHubApiBackend:
        if self._hf_api is None:
            self._hf_api = HfHubApiBackend()
        return self._hf_api

    def _selected_repo(self) -> Optional[str]:
        selected = self._search_tree.selectedItems()
        if not selected:
            self._app_actions.warn(_("Please select a model repository first."))
            return None
        return selected[0].text(0)

    def _selected_repo_silent(self) -> Optional[str]:
        selected = self._search_tree.selectedItems()
        if not selected:
            return None
        return selected[0].text(0)

    def _search(self) -> None:
        try:
            query = (self._query_edit.text() or "").strip()
            task = HfHubVisualMediaTask.get(str(self._task_combo.currentData()))
            sort = HfHubSortOption.get(str(self._sort_combo.currentData()))
            direction = HfHubSortDirection.get(str(self._direction_combo.currentData()))
            limit = int(self._limit_combo.currentText())

            results = self._api().search_models(
                query=query,
                task=task,
                limit=limit,
                sort=sort,
                direction=direction,
                include_gated=self._include_gated_cb.isChecked(),
            )
            self._search_tree.setSortingEnabled(False)
            self._search_tree.clear()
            for r in results:
                item = _SearchResultTreeItem(
                    [
                        r.repo_id,
                        r.task or "",
                        str(r.downloads),
                        str(r.likes),
                        r.license,
                        _("yes") if r.gated else _("no"),
                    ]
                )
                item.setData(2, Qt.ItemDataRole.UserRole, int(r.downloads))
                item.setData(3, Qt.ItemDataRole.UserRole, int(r.likes))
                self._search_tree.addTopLevelItem(item)
            self._search_tree.setSortingEnabled(True)
            self._apply_search_tree_sort(sort, direction)
            if self._search_tree.topLevelItemCount() > 0:
                self._search_tree.setCurrentItem(self._search_tree.topLevelItem(0))
            self._app_actions.toast(_("Found {0} results").format(len(results)))
        except Exception as e:
            logger.error(f"HF Hub search failed: {e}")
            self._app_actions.alert(_("HF Hub Search Error"), str(e), kind="error", master=self)

    def _apply_search_tree_sort(
        self,
        sort: HfHubSortOption,
        direction: HfHubSortDirection,
    ) -> None:
        """Apply visible-table sorting to match selected sort controls where possible."""
        sort_col_map = {
            HfHubSortOption.DOWNLOADS: 2,
            HfHubSortOption.LIKES: 3,
        }
        col = sort_col_map.get(sort)
        if col is None:
            return
        order = (
            Qt.SortOrder.DescendingOrder
            if direction == HfHubSortDirection.DESCENDING
            else Qt.SortOrder.AscendingOrder
        )
        self._search_tree.sortByColumn(col, order)

    def _on_repo_selection_changed(self) -> None:
        repo_id = self._selected_repo_silent()
        if not repo_id:
            return
        self._model_name_edit.setText(self._default_model_name(repo_id))
        self._set_repo_file_options(repo_id)
        self._use_transformers_auto_model_cb.setChecked(
            self._guess_transformers_auto_model_default(repo_id)
        )

    def _set_repo_file_options(self, repo_id: str) -> None:
        try:
            if repo_id in self._repo_files_cache:
                files = self._repo_files_cache[repo_id]
            else:
                files = self._api().list_model_files(repo_id)
                self._repo_files_cache[repo_id] = files
            preferred = [f for f in files if os.path.splitext(f)[1].lower() in self._MODEL_FILE_EXTENSIONS]
            values = preferred if preferred else files
            current = self._filename_combo.currentText().strip()
            self._filename_combo.clear()
            for value in values:
                self._filename_combo.addItem(value)
            if current and current in values:
                self._filename_combo.setCurrentText(current)
            elif values:
                self._filename_combo.setCurrentText(values[0])
            elif current:
                self._filename_combo.setEditText(current)
            else:
                self._filename_combo.setEditText("model.safetensors")
        except Exception as e:
            logger.error(f"Failed to load repo file list for {repo_id}: {e}")

    def _guess_transformers_auto_model_default(self, repo_id: str) -> bool:
        files = self._repo_files_cache.get(repo_id, [])
        if not files:
            return False
        lower = {f.lower() for f in files}
        has_config = "config.json" in lower
        has_processor = (
            "preprocessor_config.json" in lower
            or "processor_config.json" in lower
            or "feature_extractor_config.json" in lower
        )
        has_model_weights = any(
            os.path.splitext(f)[1].lower() in self._MODEL_FILE_EXTENSIONS
            for f in files
        )
        return has_config and has_model_weights and has_processor

    def _load_selected_repo_files(self) -> None:
        repo_id = self._selected_repo()
        if repo_id is None:
            return
        self._set_repo_file_options(repo_id)
        self._app_actions.toast(_("Loaded repo files for {0}.").format(repo_id))

    def _view_model_card(self) -> None:
        repo_id = self._selected_repo()
        if repo_id is None:
            return
        try:
            card_text = self._api().get_model_card_text(repo_id)
            _TextPreviewDialog(
                parent=self,
                title=_("Model Card - {0}").format(repo_id),
                text=card_text,
            ).show()
        except Exception as e:
            logger.error(f"Failed to fetch model card for {repo_id}: {e}")
            self._app_actions.alert(_("Model Card Error"), str(e), kind="error", master=self)

    def _download_and_install_selected(self) -> None:
        repo_id = self._selected_repo()
        if repo_id is None:
            return
        filename = (self._filename_combo.currentText() or "").strip()
        if not filename:
            self._app_actions.warn(_("Please enter a filename to download."))
            return

        try:
            snapshot_dir = self._api().download_snapshot(repo_id)
            downloaded_path = self._resolve_downloaded_file_path(snapshot_dir, filename)
            if downloaded_path is None:
                raise RuntimeError(_("Unable to locate selected file in downloaded snapshot: {0}").format(filename))
        except Exception as e:
            logger.error(f"HF Hub download failed: {e}")
            self._app_actions.alert(_("HF Hub Download Error"), str(e), kind="error", master=self)
            return

        default_name = self._default_model_name(repo_id)
        model_name = (self._model_name_edit.text() or default_name).strip()
        if not model_name:
            self._app_actions.warn(_("Model name must not be empty."))
            return

        categories_text = (self._categories_edit.text() or "").strip()
        categories = [c.strip() for c in categories_text.split(",") if c.strip()]
        if not categories:
            self._app_actions.warn(_("Please enter at least one category."))
            return

        backend = str(self._backend_combo.currentText()).strip().lower()
        use_transformers_auto_model = self._use_transformers_auto_model_cb.isChecked()
        model_kwargs = {}
        if use_transformers_auto_model:
            model_kwargs["use_transformers_auto_model"] = True
            model_kwargs["hf_pretrained_path"] = snapshot_dir
        arch_module = (self._arch_module_edit.text() or "").strip()
        arch_class = (self._arch_class_edit.text() or "").strip()
        if arch_module:
            model_kwargs["architecture_module_name"] = arch_module
        if arch_class:
            model_kwargs["architecture_class_path"] = arch_class

        effective_backend = "pytorch" if use_transformers_auto_model else (backend if backend else "auto")
        model_details = {
            "model_name": model_name,
            "model_location": downloaded_path,
            "model_categories": categories,
            "backend": effective_backend,
            "hf_repo_id": repo_id,
            "hf_selected_filename": filename,
        }
        if model_kwargs:
            model_details["model_kwargs"] = model_kwargs
        try:
            model_details = ImageClassifierModelConfig.from_dict(model_details, logger=logger).to_dict()
        except Exception as e:
            self._app_actions.alert(_("Invalid Model Configuration"), str(e), kind="error", master=self)
            return

        existing_names = {m.get("model_name") for m in config.image_classifier_models}
        if model_name in existing_names:
            should_replace = self._app_actions.alert(
                _("Replace Existing Model?"),
                _("A model named '{0}' already exists. Replace it?").format(model_name),
                kind="askokcancel",
                master=self,
            )
            if not should_replace:
                return

        updated_models = []
        replaced = False
        for existing in config.image_classifier_models:
            if existing.get("model_name") == model_name:
                updated_models.append(model_details)
                replaced = True
            else:
                updated_models.append(existing)
        if not replaced:
            updated_models.append(model_details)

        try:
            config.set_image_classifier_models(updated_models)
            image_classifier_manager.set_classifier_metadata(config.image_classifier_models)
            ClassifierActionsManager.reset_prevalidation_lazy_init()
        except Exception as e:
            logger.error(f"Failed to persist model details: {e}")
            self._app_actions.alert(_("Config Update Error"), str(e), kind="error", master=self)
            return

        self._refresh_installed_models()
        self._tabs.setCurrentIndex(1)
        self._model_name_edit.setText(model_name)
        self._app_actions.success(
            _("Downloaded and installed model '{0}'.").format(model_name)
        )

    @staticmethod
    def _default_model_name(repo_id: str) -> str:
        repo = (repo_id or "").strip()
        return repo if repo else "hf_model"

    @staticmethod
    def _resolve_downloaded_file_path(snapshot_dir: str, selected_filename: str) -> Optional[str]:
        normalized = selected_filename.replace("/", os.sep)
        expected = os.path.join(snapshot_dir, normalized)
        if os.path.isfile(expected):
            return expected

        # Fallback: resolve by basename when file path structure differs.
        basename = os.path.basename(normalized)
        if not basename:
            return None
        for root, _unused, files in os.walk(snapshot_dir):
            if basename in files:
                return os.path.join(root, basename)
        return None

    def _refresh_installed_models(self) -> None:
        self._installed_tree.clear()
        valid_count = 0
        total_count = 0
        for model in list(config.image_classifier_models):
            total_count += 1
            normalized_model, parse_err = self._parse_installed_model(model)
            if normalized_model is not None and self._is_valid_installed_model(normalized_model):
                valid_count += 1
            display_model = normalized_model if normalized_model is not None else model
            categories = display_model.get("model_categories") or []
            categories_text = ", ".join(str(c) for c in categories)
            backend = str(display_model.get("backend", "auto"))
            if parse_err:
                config_col = parse_err if len(parse_err) <= 120 else parse_err[:117] + "…"
            else:
                config_col = _("OK")
            item = QTreeWidgetItem(
                self._installed_tree,
                [
                    str(display_model.get("model_name", "")),
                    backend,
                    categories_text,
                    str(display_model.get("model_location", "")),
                    config_col,
                ],
            )
            item.setData(0, Qt.ItemDataRole.UserRole, dict(display_model))
            item.setToolTip(
                4,
                parse_err if parse_err else _("Configuration is valid."),
            )
        if valid_count == 0:
            self._installed_notice_label.setText(
                _("No valid installed models found. Use the HF Hub Search tab to discover and install models, or add one manually.")
            )
        else:
            self._installed_notice_label.setText(
                _("Installed models: {0} valid of {1} total").format(valid_count, total_count)
            )

    @staticmethod
    def _is_valid_installed_model(model: dict[str, Any]) -> bool:
        model_name = str(model.get("model_name", "") or "").strip()
        model_location = str(model.get("model_location", "") or "").strip()
        categories = model.get("model_categories") or []
        return bool(model_name and model_location and os.path.exists(model_location) and isinstance(categories, list) and len(categories) > 0)

    @staticmethod
    def _parse_installed_model(model: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
        try:
            normalized = ImageClassifierModelConfig.from_dict(model, warn_unknown_keys=False).to_dict()
            return normalized, ""
        except Exception as e:
            return None, str(e)

    def _selected_installed_model_name(self) -> Optional[str]:
        selected = self._installed_tree.selectedItems()
        if not selected:
            return None
        return selected[0].text(0)

    def _selected_installed_model_details(self) -> Optional[dict[str, Any]]:
        selected = self._installed_tree.selectedItems()
        if not selected:
            return None
        model_data = selected[0].data(0, Qt.ItemDataRole.UserRole)
        if isinstance(model_data, dict):
            return dict(model_data)
        model_name = selected[0].text(0)
        for model in config.image_classifier_models:
            if model.get("model_name") == model_name:
                return dict(model)
        return None

    def _add_installed_model(self) -> None:
        _InstalledModelEditDialog(
            parent=self,
            title=_("Add Installed Model"),
            initial_model={},
            api_getter=self._api,
            save_callback=self._save_installed_model_details,
            model_file_extensions=self._MODEL_FILE_EXTENSIONS,
        ).show()

    def _edit_selected_installed_model(self) -> None:
        model = self._selected_installed_model_details()
        if model is None:
            self._app_actions.warn(_("Please select an installed model first."))
            return
        _InstalledModelEditDialog(
            parent=self,
            title=_("Edit Installed Model"),
            initial_model=model,
            api_getter=self._api,
            save_callback=self._save_installed_model_details,
            model_file_extensions=self._MODEL_FILE_EXTENSIONS,
        ).show()

    def _save_installed_model_details(self, model_details: dict[str, Any], original_name: str) -> None:
        normalized_model = ImageClassifierModelConfig.from_dict(model_details, logger=logger).to_dict()
        model_name = str(normalized_model.get("model_name", "")).strip()
        updated_models = []
        replaced = False
        for existing in config.image_classifier_models:
            existing_name = str(existing.get("model_name", "")).strip()
            if existing_name == original_name and original_name:
                updated_models.append(normalized_model)
                replaced = True
            elif existing_name == model_name and existing_name != original_name:
                updated_models.append(normalized_model)
                replaced = True
            else:
                updated_models.append(existing)
        if not replaced:
            updated_models.append(normalized_model)
        try:
            config.set_image_classifier_models(updated_models)
            image_classifier_manager.set_classifier_metadata(config.image_classifier_models)
            ClassifierActionsManager.reset_prevalidation_lazy_init()
            self._refresh_installed_models()
            self._app_actions.success(_("Saved model '{0}'.").format(model_name))
        except Exception as e:
            logger.error(f"Failed saving installed model details: {e}")
            self._app_actions.alert(_("Config Update Error"), str(e), kind="error", master=self)

    def _test_selected_model_on_current_image(self) -> None:
        model = self._selected_installed_model_details()
        if model is None:
            self._app_actions.warn(_("Please select an installed model first."))
            return
        model_name = str(model.get("model_name", "")).strip()
        if not model_name:
            self._app_actions.warn(_("Selected model has no name."))
            return
        image_path = self._app_actions.get_active_media_filepath()
        if not image_path or not os.path.isfile(image_path):
            self._app_actions.warn(_("No active image is available for testing."))
            return
        self._test_btn.setEnabled(False)
        self._test_btn.setText(_("Running…"))
        worker = _ClassifierTestWorker(model_name, image_path)
        worker.finished.connect(self._on_classifier_test_finished)
        worker.failed.connect(self._on_classifier_test_failed)
        self._test_worker = worker
        worker.start()

    @Slot(str, str, object)
    def _on_classifier_test_finished(self, model_name: str, image_path: str, result: dict) -> None:
        self._test_btn.setEnabled(True)
        self._test_btn.setText(_("Test on Current Image"))
        worker = self._test_worker
        self._test_worker = None
        if worker is not None:
            worker.deleteLater()
        classification = result.get("classification", "?")
        ranked: list = result.get("ranked", [])
        lines = [
            _("Model: {0}").format(model_name),
            _("Image: {0}").format(image_path),
            "",
            _("Classification: {0}").format(classification),
            "",
            _("Scores (ranked):"),
        ]
        for cat, score in ranked:
            lines.append(f"  {cat}: {score:.6f}  ({score * 100:.2f}%)")
        _TextPreviewDialog(
            parent=self,
            title=_("Classifier Test — {0}").format(model_name),
            text="\n".join(lines),
        ).show()

    @Slot(str, str, str)
    def _on_classifier_test_failed(self, model_name: str, image_path: str, error: str) -> None:
        self._test_btn.setEnabled(True)
        self._test_btn.setText(_("Test on Current Image"))
        worker = self._test_worker
        self._test_worker = None
        if worker is not None:
            worker.deleteLater()
        logger.error("Classifier test failed for %r on %r: %s", model_name, image_path, error)
        self._app_actions.alert(
            _("Classifier Test Failed"),
            f"{model_name}\n{image_path}\n\n{error}",
            kind="error",
            master=self,
        )

    def _infer_hf_repo_id_from_model(self, model: dict[str, Any]) -> Optional[str]:
        explicit = str(model.get("hf_repo_id", "") or "").strip()
        if explicit:
            return explicit
        model_kwargs = dict(model.get("model_kwargs", {}))
        hf_path = str(model_kwargs.get("hf_pretrained_path", "") or "")
        model_location = str(model.get("model_location", "") or "")
        for candidate in (hf_path, model_location):
            if "models--" not in candidate:
                continue
            marker = candidate.split("models--", 1)[1]
            repo_part = marker.split(os.sep + "snapshots", 1)[0]
            if not repo_part:
                continue
            repo_id = repo_part.replace("--", "/")
            return repo_id.strip("/")
        return None

    def _load_repo_files_for_installed_selection(self) -> None:
        model = self._selected_installed_model_details()
        if model is None:
            self._app_actions.warn(_("Please select an installed model first."))
            return
        repo_id = self._infer_hf_repo_id_from_model(model)
        if not repo_id:
            self._app_actions.warn(_("No HF repo id found for selected model."))
            return
        try:
            self._set_repo_file_options(repo_id)
            self._app_actions.toast(_("Loaded repo files for {0}.").format(repo_id))
        except Exception as e:
            self._app_actions.alert(_("HF Hub Error"), str(e), kind="error", master=self)

    def _view_model_card_for_installed_selection(self) -> None:
        model = self._selected_installed_model_details()
        if model is None:
            self._app_actions.warn(_("Please select an installed model first."))
            return
        repo_id = self._infer_hf_repo_id_from_model(model)
        if not repo_id:
            self._app_actions.warn(_("No HF repo id found for selected model."))
            return
        try:
            card_text = self._api().get_model_card_text(repo_id)
            _TextPreviewDialog(
                parent=self,
                title=_("Model Card - {0}").format(repo_id),
                text=card_text,
            ).show()
        except Exception as e:
            self._app_actions.alert(_("Model Card Error"), str(e), kind="error", master=self)

    def _remove_selected_installed_model(self) -> None:
        model = self._selected_installed_model_details()
        if model is None:
            self._app_actions.warn(_("Please select an installed model first."))
            return
        model_name = str(model.get("model_name", "")).strip()
        model_kwargs = dict(model.get("model_kwargs", {}))
        api_backend: Optional[HfHubApiBackend] = None
        try:
            api_backend = self._api()
        except Exception:
            api_backend = None
        fallback_cache = HfHubApiBackend.get_default_cache_dir()
        hf_cache_path = str(model_kwargs.get("hf_pretrained_path", "") or fallback_cache)
        repo_id = self._infer_hf_repo_id_from_model(model)

        # Confirmation #1 (always)
        should_remove = self._app_actions.alert(
            _("Remove Installed Model?"),
            _("Remove '{0}' from configured image classifier models?").format(model_name),
            kind="askokcancel",
            master=self,
        )
        if not should_remove:
            return

        hf_connected = bool(api_backend and api_backend.has_connection())
        repo_is_hosted = False
        repo_hosted_message = ""
        if api_backend and repo_id:
            repo_is_hosted, repo_hosted_message = api_backend.is_repo_hosted(repo_id)

        # Confirmation #2:
        # - connected, but source can't be confirmed from model metadata
        # - connected + repo id known, but repo is not currently hosted
        if hf_connected and (not repo_id or (repo_id and not repo_is_hosted)):
            if repo_id and not repo_is_hosted:
                prompt = _(
                    "Repo '{0}' is not currently available on Hugging Face.\n"
                    "Remove config entry anyway?\n\nDetails: {1}"
                ).format(repo_id, repo_hosted_message or _("Unknown error"))
            else:
                prompt = _("Unable to confirm this model came from Hugging Face. Remove config entry anyway?")
            second_confirm = self._app_actions.alert(
                _("Unable to Confirm HF Source"),
                prompt,
                kind="askokcancel",
                master=self,
            )
            if not second_confirm:
                return

        updated_models = [m for m in config.image_classifier_models if m.get("model_name") != model_name]
        if len(updated_models) == len(config.image_classifier_models):
            self._app_actions.warn(_("No matching model named '{0}' was found.").format(model_name))
            return

        try:
            config.set_image_classifier_models(updated_models)
            image_classifier_manager.set_classifier_metadata(config.image_classifier_models)
            ClassifierActionsManager.reset_prevalidation_lazy_init()
        except Exception as e:
            logger.error(f"Failed to remove model details: {e}")
            self._app_actions.alert(_("Config Update Error"), str(e), kind="error", master=self)
            return

        if hf_connected and repo_id and api_backend:
            deleted, cache_dir, message = api_backend.delete_cached_repo(repo_id)
            if deleted:
                self._app_actions.success(
                    _("Removed model '{0}' and deleted HF cache for {1}.").format(model_name, repo_id)
                )
            else:
                self._app_actions.alert(
                    _("Model Removed; Cache Retained"),
                    _("Removed '{0}', but cache was not deleted.\n{1}\nCache location: {2}").format(
                        model_name, message, cache_dir
                    ),
                    kind="warning",
                    master=self,
                )
        elif not hf_connected:
            self._app_actions.alert(
                _("Model Removed; Cache Retained"),
                _("Removed '{0}'. Hugging Face could not be reached, so cache deletion was skipped.\nCache may still exist at:\n{1}").format(
                    model_name, hf_cache_path or HfHubApiBackend.get_default_cache_dir()
                ),
                kind="warning",
                master=self,
            )
        else:
            self._app_actions.success(_("Removed model '{0}'.").format(model_name))

        self._refresh_installed_models()

"""
PySide6 port of compare/compare_settings_window.py -- CompareSettingsWindow.

Singleton dialog per CompareManager for configuring comparison modes,
filters, and composite search settings.

Non-UI imports:
  - CompareManager, CombinationLogic from compare.compare_manager
  - FilterBuilderPanel from ui.compare.filter_builder_panel_qt
  - AddInstanceDialog from ui.compare.add_instance_dialog_qt
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from compare.compare_history import CompareHistory
from compare.compare_manager import CompareManager, CombinationLogic
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from ui.compare.add_instance_dialog_qt import AddInstanceDialog, MAX_INSTANCES
from ui.compare.filter_builder_panel_qt import FilterBuilderPanel
from utils.config import config
from utils.constants import CompareMode
from utils.translations import I18N
from utils.logging_setup import get_logger

_ = I18N._
logger = get_logger("compare_settings_window_qt")


def _h_separator() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


class CompareSettingsWindow(SmartDialog):
    """Window for configuring comparison modes, filters, and composite search settings."""

    _open_windows: Dict[object, CompareSettingsWindow] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def open(cls, parent: QWidget, compare_manager: CompareManager) -> None:
        """Show or focus the settings window for *compare_manager*."""
        if compare_manager in cls._open_windows:
            win = cls._open_windows[compare_manager]
            try:
                if win.isVisible():
                    win.raise_()
                    win.activateWindow()
                    return
            except Exception:
                pass
        cls(parent, compare_manager)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, parent: QWidget, compare_manager: CompareManager) -> None:
        if compare_manager in CompareSettingsWindow._open_windows:
            existing = CompareSettingsWindow._open_windows[compare_manager]
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing.activateWindow()
                    return
            except Exception:
                pass

        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Compare Settings"),
            geometry="1020x720",
        )
        CompareSettingsWindow._open_windows[compare_manager] = self

        self._compare_manager = compare_manager
        self._weight_vars: Dict[str, QLineEdit] = {}   # instance_id -> weight edit
        self._threshold_combo: Optional[QComboBox] = None
        self._add_instance_btn: Optional[QPushButton] = None
        self._instance_list_layout: Optional[QVBoxLayout] = None

        self._build_ui()

        # Load existing filter into the panel
        self._filter_panel.set_filter(compare_manager.get_data_filter())

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        self.show()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(0)

        # Title
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_lbl = QLabel(_("Compare Settings"))
        title_lbl.setFont(title_font)
        title_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        outer.addWidget(title_lbl)
        outer.addSpacing(16)

        # Two-column body
        body = QHBoxLayout()
        body.setSpacing(20)

        # ============================================================
        # LEFT COLUMN — instances + combination logic
        # ============================================================
        left = QVBoxLayout()
        left.setSpacing(6)

        section_font = QFont()
        section_font.setPointSize(11)
        section_font.setBold(True)

        inst_title = QLabel(_("Compare Instances"))
        inst_title.setFont(section_font)
        inst_title.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        left.addWidget(inst_title)

        # Scrollable instance list
        inst_scroll = QScrollArea()
        inst_scroll.setWidgetResizable(True)
        inst_scroll.setFrameShape(QFrame.Shape.StyledPanel)
        inst_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inst_scroll.setMinimumHeight(180)
        inst_scroll.setMaximumHeight(340)

        self._inst_scroll_inner = QWidget()
        self._instance_list_layout = QVBoxLayout(self._inst_scroll_inner)
        self._instance_list_layout.setContentsMargins(4, 4, 4, 4)
        self._instance_list_layout.setSpacing(2)
        self._instance_list_layout.addStretch()
        inst_scroll.setWidget(self._inst_scroll_inner)
        left.addWidget(inst_scroll)

        # Add Instance button
        self._add_instance_btn = QPushButton(_("+ Add Instance"))
        self._add_instance_btn.clicked.connect(self._on_add_instance)
        left.addWidget(self._add_instance_btn, 0, Qt.AlignLeft)

        left.addWidget(_h_separator())

        # Combination Logic
        logic_row = QHBoxLayout()
        logic_lbl = QLabel(_("Combination Logic:"))
        logic_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        logic_row.addWidget(logic_lbl)

        self._logic_combo = QComboBox()
        self._logic_combo.addItems([lg.value for lg in CombinationLogic])
        self._logic_combo.setCurrentText(
            self._compare_manager.get_combination_logic().value
        )
        self._logic_combo.currentTextChanged.connect(self._on_logic_changed)
        logic_row.addWidget(self._logic_combo)
        logic_row.addStretch()
        left.addLayout(logic_row)

        left.addWidget(_h_separator())

        # Recent Analyses
        recent_title = QLabel(_("Recent Analyses"))
        recent_title.setFont(section_font)
        recent_title.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        left.addWidget(recent_title)

        recent_scroll = QScrollArea()
        recent_scroll.setWidgetResizable(True)
        recent_scroll.setFrameShape(QFrame.Shape.StyledPanel)
        recent_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        recent_scroll.setMinimumHeight(100)
        recent_scroll.setMaximumHeight(220)

        self._recent_inner = QWidget()
        self._recent_layout = QVBoxLayout(self._recent_inner)
        self._recent_layout.setContentsMargins(4, 4, 4, 4)
        self._recent_layout.setSpacing(2)
        self._recent_layout.addStretch()
        recent_scroll.setWidget(self._recent_inner)
        left.addWidget(recent_scroll)

        left.addStretch()
        body.addLayout(left, 1)

        # ============================================================
        # RIGHT COLUMN — settings + filter panel
        # ============================================================
        right = QVBoxLayout()
        right.setSpacing(6)

        settings_title = QLabel(_("Compare Settings"))
        settings_title.setFont(section_font)
        settings_title.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        right.addWidget(settings_title)

        current_args = self._compare_manager.get_args()
        primary_mode = self._compare_manager.compare_mode

        # Threshold
        thresh_row = QHBoxLayout()
        thresh_lbl = QLabel(_("Threshold"))
        thresh_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        thresh_row.addWidget(thresh_lbl)

        self._threshold_combo = QComboBox()
        self._threshold_combo.setEditable(True)
        self._populate_threshold_combo(primary_mode, current_args)
        thresh_row.addWidget(self._threshold_combo)
        thresh_row.addStretch()
        right.addLayout(thresh_row)

        # Counter limit
        limit_row = QHBoxLayout()
        limit_lbl = QLabel(_("Max files to compare"))
        limit_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        limit_row.addWidget(limit_lbl)

        counter_limit_value = (
            current_args.counter_limit
            if hasattr(current_args, "counter_limit")
            else config.file_counter_limit
        )
        self._counter_limit_edit = QLineEdit(
            "" if counter_limit_value is None else str(counter_limit_value)
        )
        self._counter_limit_edit.setFixedWidth(100)
        limit_row.addWidget(self._counter_limit_edit)
        limit_row.addStretch()
        right.addLayout(limit_row)

        # Option checkboxes (from last compare args when a run exists, else manager/history)
        self._compare_faces_cb = QCheckBox(_("Compare faces"))
        right.addWidget(self._compare_faces_cb)

        self._overwrite_cb = QCheckBox(_("Overwrite cache"))
        right.addWidget(self._overwrite_cb)

        self._store_checkpoints_cb = QCheckBox(_("Store checkpoints"))
        right.addWidget(self._store_checkpoints_cb)

        self._matrix_compare_cb = QCheckBox(_("Use matrix group compare"))
        self._matrix_compare_cb.setToolTip(
            _(
                "Embedding group compare only. Uses the chunked matrix path when "
                "checked; uses the legacy roll-index path when unchecked. "
                "Color matching is not affected."
            )
        )
        self._update_matrix_compare_cb_enabled(primary_mode)
        right.addWidget(self._matrix_compare_cb)

        self._refresh_global_settings_controls(current_args)

        self._search_closest_cb = QCheckBox(_("Search only return closest"))
        self._search_closest_cb.setChecked(config.search_only_return_closest)
        right.addWidget(self._search_closest_cb)

        right.addWidget(_h_separator())

        # Filter panel
        filter_title = QLabel(_("Filters (Applied Before Comparison)"))
        filter_title.setFont(section_font)
        filter_title.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        right.addWidget(filter_title)

        self._filter_panel = FilterBuilderPanel()
        right.addWidget(self._filter_panel, 1)

        body.addLayout(right, 1)

        outer.addLayout(body, 1)

        # ---- Bottom bar ---
        outer.addWidget(_h_separator())
        outer.addSpacing(10)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton(_("Apply"))
        apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(apply_btn)

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(cancel_btn)

        btn_row.addStretch()

        reset_btn = QPushButton(_("Reset to Default"))
        reset_btn.setToolTip(_("Reset to a single CLIP Embedding instance with no filters"))
        reset_btn.clicked.connect(self._on_reset_to_default)
        btn_row.addWidget(reset_btn)

        outer.addLayout(btn_row)

        # Populate instance list and recent history after all widgets exist
        self._refresh_instance_list()
        self._refresh_recent_history()

    # ------------------------------------------------------------------
    # Instance list
    # ------------------------------------------------------------------
    def _refresh_instance_list(self) -> None:
        """Rebuild the scrollable instance list from manager state."""
        _clear_layout(self._instance_list_layout)
        self._weight_vars.clear()

        instances = self._compare_manager.get_mode_instances()
        is_weighted = (
            self._compare_manager.get_combination_logic() == CombinationLogic.WEIGHTED
        )
        total = len(instances)

        for cfg in instances:
            row = QHBoxLayout()
            row.setSpacing(6)

            # Mode label
            mode_lbl = QLabel(cfg.compare_mode.get_text())
            mode_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            mode_lbl.setFixedWidth(160)
            row.addWidget(mode_lbl)

            # Search text summary
            search_summary = ""
            if cfg.search_text:
                search_summary = cfg.search_text[:30]
                if len(cfg.search_text) > 30:
                    search_summary += "…"
            if cfg.search_text_negative:
                neg = cfg.search_text_negative[:20]
                if len(cfg.search_text_negative) > 20:
                    neg += "…"
                search_summary += f"  −{neg}"
            if search_summary:
                st_lbl = QLabel(search_summary)
                st_lbl.setStyleSheet(
                    f"color: {AppStyle.FG_COLOR}; font-size: 9pt; font-style: italic;"
                )
                row.addWidget(st_lbl)

            # Threshold override indicator
            if cfg.threshold is not None:
                thresh_lbl = QLabel(f"[t={cfg.threshold}]")
                thresh_lbl.setStyleSheet(
                    f"color: {AppStyle.FG_COLOR}; font-size: 9pt;"
                )
                row.addWidget(thresh_lbl)

            row.addStretch()

            # Weight edit (WEIGHTED mode only)
            if is_weighted:
                w_edit = QLineEdit(str(cfg.weight))
                w_edit.setFixedWidth(60)
                w_edit.setToolTip(_("Weight for this instance"))
                row.addWidget(w_edit)
                self._weight_vars[cfg.instance_id] = w_edit

            # Remove button — disabled when only one instance remains
            remove_btn = QPushButton("✕")
            remove_btn.setFixedWidth(28)
            remove_btn.setToolTip(_("Remove this instance"))
            remove_btn.setEnabled(total > 1)
            remove_btn.clicked.connect(
                lambda _checked, iid=cfg.instance_id: self._on_remove_instance(iid)
            )
            row.addWidget(remove_btn)

            # Wrap in a container widget for clean insertion
            container = QWidget()
            container.setLayout(row)
            # Insert before the trailing stretch
            insert_pos = max(0, self._instance_list_layout.count() - 1)
            self._instance_list_layout.insertWidget(insert_pos, container)

        # Disable add button at capacity
        if self._add_instance_btn is not None:
            self._add_instance_btn.setEnabled(total < MAX_INSTANCES)

        self._update_matrix_compare_cb_enabled(self._compare_manager.compare_mode)

    def _on_remove_instance(self, instance_id: str) -> None:
        self._compare_manager.remove_mode_instance(instance_id)
        self._refresh_instance_list()

    # ------------------------------------------------------------------
    # Threshold combo helpers
    # ------------------------------------------------------------------
    def _refresh_global_settings_controls(self, current_args=None) -> None:
        """Sync checkboxes and global threshold/limit from manager or last compare args."""
        mgr = self._compare_manager

        def _bool_from_run(attr: str, getter_name: str, default: bool) -> bool:
            if mgr.has_compare() and current_args is not None:
                return bool(getattr(current_args, attr, default))
            return getattr(mgr, getter_name)()

        self._compare_faces_cb.setChecked(
            _bool_from_run("compare_faces", "get_compare_faces", False)
        )
        self._overwrite_cb.setChecked(
            _bool_from_run("overwrite", "get_overwrite", False)
        )
        self._store_checkpoints_cb.setChecked(
            _bool_from_run("store_checkpoints", "get_store_checkpoints", config.store_checkpoints)
        )
        self._matrix_compare_cb.setChecked(
            _bool_from_run("use_matrix_comparison", "get_use_matrix_comparison", True)
        )

        counter_limit = (
            current_args.counter_limit
            if mgr.has_compare() and current_args is not None
            and hasattr(current_args, "counter_limit")
            else mgr.get_counter_limit()
        )
        self._counter_limit_edit.setText(
            "" if counter_limit is None else str(counter_limit)
        )

        primary_mode = mgr.compare_mode
        threshold = (
            current_args.threshold
            if mgr.has_compare() and current_args is not None
            and hasattr(current_args, "threshold")
            else mgr.get_threshold()
        )
        self._populate_threshold_combo(primary_mode, current_args=None)
        if threshold is not None:
            self._threshold_combo.setCurrentText(
                str(int(threshold))
                if primary_mode == CompareMode.COLOR_MATCHING
                else str(threshold)
            )

    def _populate_threshold_combo(
        self,
        mode: Optional[CompareMode],
        current_args=None,
    ) -> None:
        self._threshold_combo.clear()
        threshold_vals = (
            mode.threshold_vals()
            if mode is not None
            else CompareMode.CLIP_EMBEDDING.threshold_vals()
        )
        if not threshold_vals:
            threshold_vals = CompareMode.CLIP_EMBEDDING.threshold_vals()

        for v in threshold_vals:
            self._threshold_combo.addItem(str(v))

        if current_args is not None and hasattr(current_args, "threshold"):
            current_val = str(current_args.threshold)
        elif mode == CompareMode.COLOR_MATCHING:
            current_val = str(config.color_diff_threshold)
        else:
            current_val = str(config.embedding_similarity_threshold)

        self._threshold_combo.setCurrentText(current_val)

    # ------------------------------------------------------------------
    # Combination logic
    # ------------------------------------------------------------------
    def _update_matrix_compare_cb_enabled(self, mode: Optional[CompareMode]) -> None:
        if self._matrix_compare_cb is None:
            return
        self._matrix_compare_cb.setEnabled(
            mode is None or mode.is_embedding()
        )

    def _on_logic_changed(self, logic_str: str) -> None:
        try:
            logic = CombinationLogic(logic_str)
            self._compare_manager.set_combination_logic(logic)
            self._refresh_instance_list()
        except ValueError:
            logger.warning(f"Invalid combination logic: {logic_str}")

    # ------------------------------------------------------------------
    # Add Instance
    # ------------------------------------------------------------------
    def _on_add_instance(self) -> None:
        instances = self._compare_manager.get_mode_instances()
        dlg = AddInstanceDialog(
            parent=self,
            combination_logic=self._compare_manager.get_combination_logic(),
            current_instance_count=len(instances),
        )
        dlg.exec()
        result = dlg.get_result()
        if result is None:
            return

        self._compare_manager.add_mode_instance(
            compare_mode=result["mode"],
            weight=result["weight"],
            threshold=result["threshold"],
            search_text=result["search_text"],
            search_text_negative=result["search_text_negative"],
        )
        self._refresh_instance_list()

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def _on_apply(self) -> None:
        # Weights (only meaningful in WEIGHTED mode, but always persist)
        for instance_id, weight_edit in self._weight_vars.items():
            try:
                self._compare_manager.set_mode_weight(instance_id, float(weight_edit.text()))
            except ValueError:
                logger.warning(f"Invalid weight for {instance_id}: {weight_edit.text()!r}")

        # Global threshold
        primary_mode = self._compare_manager.compare_mode
        if primary_mode and self._threshold_combo:
            try:
                t_str = self._threshold_combo.currentText().strip()
                threshold = (
                    int(t_str) if primary_mode == CompareMode.COLOR_MATCHING
                    else float(t_str)
                )
                self._compare_manager.set_threshold(threshold)
            except ValueError:
                logger.warning(f"Invalid threshold: {self._threshold_combo.currentText()!r}")

        # Counter limit
        try:
            cl_str = self._counter_limit_edit.text().strip()
            self._compare_manager.set_counter_limit(
                None if cl_str == "" else int(cl_str)
            )
        except ValueError:
            logger.warning(f"Invalid counter limit: {self._counter_limit_edit.text()!r}")

        # Boolean options
        self._compare_manager.set_compare_faces(self._compare_faces_cb.isChecked())
        self._compare_manager.set_overwrite(self._overwrite_cb.isChecked())
        self._compare_manager.set_store_checkpoints(self._store_checkpoints_cb.isChecked())
        self._compare_manager.set_use_matrix_comparison(
            self._matrix_compare_cb.isChecked()
        )
        config.search_only_return_closest = self._search_closest_cb.isChecked()

        # Data filter
        self._compare_manager.set_data_filter(self._filter_panel.get_filter())

        self.close()

    # ------------------------------------------------------------------
    # Recent history
    # ------------------------------------------------------------------
    def _refresh_recent_history(self) -> None:
        """Rebuild the recent-analyses list from persisted history."""
        _clear_layout(self._recent_layout)
        history = CompareHistory.load_recent()

        if not history:
            empty_lbl = QLabel(_("No recent analyses."))
            empty_lbl.setStyleSheet(
                f"color: {AppStyle.FG_COLOR}; font-size: 9pt; font-style: italic;"
            )
            insert_pos = max(0, self._recent_layout.count() - 1)
            self._recent_layout.insertWidget(insert_pos, empty_lbl)
            return

        for h in history:
            row = QHBoxLayout()
            row.setSpacing(6)

            lbl = QLabel(h.label())
            lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 9pt;")
            lbl.setWordWrap(False)
            row.addWidget(lbl, 1)

            load_btn = QPushButton(_("Load"))
            load_btn.setFixedWidth(50)
            load_btn.clicked.connect(
                lambda _checked, entry=h: self._on_load_history(entry)
            )
            row.addWidget(load_btn)

            remove_btn = QPushButton("✕")
            remove_btn.setFixedWidth(28)
            remove_btn.setToolTip(_("Remove from history"))
            remove_btn.clicked.connect(
                lambda _checked, entry=h: self._on_remove_history(entry)
            )
            row.addWidget(remove_btn)

            container = QWidget()
            container.setLayout(row)
            insert_pos = max(0, self._recent_layout.count() - 1)
            self._recent_layout.insertWidget(insert_pos, container)

    def _on_load_history(self, entry: CompareHistory) -> None:
        self._compare_manager.apply_snapshot(entry)
        self._filter_panel.set_filter(self._compare_manager.get_data_filter())
        self._logic_combo.setCurrentText(
            self._compare_manager.get_combination_logic().value
        )
        self._refresh_instance_list()
        self._refresh_global_settings_controls()

    def _on_remove_history(self, entry: CompareHistory) -> None:
        CompareHistory.remove(entry)
        self._refresh_recent_history()

    # ------------------------------------------------------------------
    # Reset to default
    # ------------------------------------------------------------------
    def _on_reset_to_default(self) -> None:
        reply = QMessageBox.question(
            self,
            _("Reset to Default"),
            _(
                "Reset all compare settings to a single CLIP Embedding instance "
                "with no filters?\n\nThis will clear all current instances and filters."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._compare_manager.reset_to_default()
        self._filter_panel.set_filter(None)
        self._logic_combo.setCurrentText(
            self._compare_manager.get_combination_logic().value
        )
        self._refresh_instance_list()
        self._refresh_global_settings_controls()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        CompareSettingsWindow._open_windows.pop(self._compare_manager, None)
        super().closeEvent(event)


# ======================================================================
# Helpers
# ======================================================================
def _clear_layout(layout) -> None:
    """Recursively remove all items from a QLayout."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        sub = item.layout()
        if sub is not None:
            _clear_layout(sub)

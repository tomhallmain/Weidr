"""
PySide6 port of the DirectoryProfileWindow from compare/directory_profile.py.

Only the UI class is ported here. The non-UI ``DirectoryProfile`` data class
is imported from the original module per the reuse policy.

Non-UI imports:
  - DirectoryProfile from compare.directory_profile (reuse policy)
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from compare.classifier_action import Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from files.directory_profile import DirectoryProfile
from lib.fast_directory_picker_qt import get_existing_directory
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.translations import I18N
from utils.logging_setup import get_logger

_ = I18N._
logger = get_logger("directory_profile_window_qt")


class DirectoryProfileWindow(SmartDialog):
    """
    Create / edit dialog for a single DirectoryProfile.

    Fields:
      - Profile name (QLineEdit)
      - Directories list (QListWidget) with Add / Edit / Remove /
        Add subdirs / Clear all buttons
      - Done button
    """

    _instance: Optional[DirectoryProfileWindow] = None

    def __init__(
        self,
        parent: QWidget,
        app_actions,
        refresh_callback: Callable,
        profile: Optional[DirectoryProfile] = None,
        copy_from_profile: Optional[DirectoryProfile] = None,
        dimensions: str = "600x500",
    ) -> None:
        self._is_copy = copy_from_profile is not None
        self._is_edit = profile is not None and not self._is_copy
        self._copy_source_profile = copy_from_profile
        if self._is_copy and copy_from_profile is not None:
            self._profile = DirectoryProfile(
                name=self._generate_profile_copy_name(copy_from_profile.name),
                directories=list(copy_from_profile.directories),
            )
        else:
            self._profile = profile if profile is not None else DirectoryProfile()
        self._original_name = self._profile.name if self._is_edit else None
        self._source_prevalidations: list[Prevalidation] = (
            self._get_associated_prevalidations(copy_from_profile.name)
            if self._is_copy and copy_from_profile is not None
            else []
        )
        self._source_prevalidations_by_name: dict[str, Prevalidation] = {
            pv.name: pv for pv in self._source_prevalidations
        }
        self._prevalidation_copy_drafts: list[dict[str, str]] = []
        if self._source_prevalidations:
            taken_names = {pv.name for pv in ClassifierActionsManager.prevalidations}
            for source_pv in self._source_prevalidations:
                draft_name = self._generate_prevalidation_copy_name(source_pv.name, taken_names)
                taken_names.add(draft_name)
                self._prevalidation_copy_drafts.append(
                    {
                        "source_name": source_pv.name,
                        "name": draft_name,
                        "action_modifier": source_pv.action_modifier or "",
                    }
                )
        self._prevalidation_modifier_inputs: dict[str, QLineEdit] = {}
        self._prevalidation_name_inputs: dict[str, QLineEdit] = {}

        super().__init__(
            parent=parent,
            position_parent=parent,
            title=(
                _("Copy Profile")
                if self._is_copy
                else (_("Edit Profile") if self._is_edit else _("Create Profile"))
            ),
            geometry=dimensions,
        )
        DirectoryProfileWindow._instance = self

        self._app_actions = app_actions
        self._refresh_callback = refresh_callback

        self._build_ui()

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_Return), self).activated.connect(self._finalize_profile)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # -- Profile name -------------------------------------------------
        name_row = QHBoxLayout()
        name_lbl = QLabel(_("Profile Name"))
        name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        name_row.addWidget(name_lbl)

        self._name_edit = QLineEdit(self._profile.name)
        name_row.addWidget(self._name_edit, 1)
        root.addLayout(name_row)

        # -- Directories --------------------------------------------------
        dir_lbl = QLabel(_("Directories"))
        dir_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(dir_lbl)

        dir_area = QHBoxLayout()

        self._dir_list = QListWidget()
        self._dir_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._dir_list.setStyleSheet(
            f"QListWidget {{ background: {AppStyle.BG_COLOR}; "
            f"color: {AppStyle.FG_COLOR}; }}"
        )
        self._refresh_directory_list()
        dir_area.addWidget(self._dir_list, 1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        add_btn = QPushButton(_("Add"))
        add_btn.clicked.connect(self._add_directory)
        btn_col.addWidget(add_btn)

        edit_btn = QPushButton(_("Edit"))
        edit_btn.clicked.connect(self._edit_directory)
        btn_col.addWidget(edit_btn)

        remove_btn = QPushButton(_("Remove"))
        remove_btn.clicked.connect(self._remove_directory)
        btn_col.addWidget(remove_btn)

        add_subdirs_btn = QPushButton(_("Add dirs from subdirs"))
        add_subdirs_btn.clicked.connect(self._add_subdirectories)
        btn_col.addWidget(add_subdirs_btn)

        clear_btn = QPushButton(_("Clear all"))
        clear_btn.clicked.connect(self._clear_all_directories)
        btn_col.addWidget(clear_btn)

        btn_col.addStretch()
        dir_area.addLayout(btn_col)
        root.addLayout(dir_area, 1)

        if self._is_copy:
            self._build_copy_prevalidations_section(root)

        # -- Done button --------------------------------------------------
        done_btn = QPushButton(_("Done"))
        done_btn.clicked.connect(self._finalize_profile)
        root.addWidget(done_btn, 0, Qt.AlignLeft)

    def _build_copy_prevalidations_section(self, root: QVBoxLayout) -> None:
        self._copy_prevalidations_cb = QCheckBox(_("Copy associated prevalidations"))
        self._copy_prevalidations_cb.setChecked(True)
        self._copy_prevalidations_cb.stateChanged.connect(
            lambda _state: self._toggle_prevalidation_modifier_inputs()
        )
        root.addWidget(self._copy_prevalidations_cb)

        if not self._source_prevalidations:
            no_assoc_lbl = QLabel(_("No prevalidations are associated with this profile."))
            no_assoc_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            root.addWidget(no_assoc_lbl)
            return

        self._pv_copy_hint_label = QLabel(
            _("Optional: adjust action target directories for copied prevalidations.")
        )
        self._pv_copy_hint_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._pv_copy_hint_label.setWordWrap(True)
        root.addWidget(self._pv_copy_hint_label)

        self._pv_modifiers_container = QWidget()
        self._pv_mod_layout = QVBoxLayout(self._pv_modifiers_container)
        self._pv_mod_layout.setContentsMargins(0, 0, 0, 0)
        self._pv_mod_layout.setSpacing(4)
        self._rebuild_prevalidation_modifier_rows()
        root.addWidget(self._pv_modifiers_container)
        self._toggle_prevalidation_modifier_inputs()

    # ------------------------------------------------------------------
    # Directory list helpers
    # ------------------------------------------------------------------
    def _refresh_directory_list(self) -> None:
        self._dir_list.clear()
        for d in self._profile.directories:
            self._dir_list.addItem(d)

    def _browse_directory(
        self, title: str = _("Select directory"), initial_dir: Optional[str] = None
    ) -> Optional[str]:
        if initial_dir is None:
            initial_dir = self._profile.directories[-1] if self._profile.directories else "."
        if not os.path.isdir(initial_dir):
            initial_dir = "."

        directory = get_existing_directory(self, title, initial_dir)
        return directory if directory and directory.strip() else None

    def _add_directory(self) -> None:
        directory = self._browse_directory(_("Add Directory"))
        if directory:
            directory = directory.strip()
            if os.path.isdir(directory):
                if directory not in self._profile.directories:
                    self._profile.directories.append(directory)
                    self._refresh_directory_list()
                else:
                    logger.warning(f"Directory {directory} already in profile")
            else:
                logger.error(f"Invalid directory: {directory}")

    def _edit_directory(self) -> None:
        current = self._dir_list.currentItem()
        if current is None:
            logger.warning("No directory selected for editing")
            return
        idx = self._dir_list.currentRow()
        if idx >= len(self._profile.directories):
            return

        current_dir = self._profile.directories[idx]
        new_directory = self._browse_directory(_("Edit Directory"), initial_dir=current_dir)
        if new_directory:
            new_directory = new_directory.strip()
            if os.path.isdir(new_directory):
                if (
                    new_directory in self._profile.directories
                    and self._profile.directories.index(new_directory) != idx
                ):
                    logger.warning(f"Directory {new_directory} already in profile")
                else:
                    self._profile.directories[idx] = new_directory
                    self._refresh_directory_list()
                    self._dir_list.setCurrentRow(idx)
            else:
                logger.error(f"Invalid directory: {new_directory}")

    def _remove_directory(self) -> None:
        rows = {self._dir_list.row(item) for item in self._dir_list.selectedItems()}
        if not rows:
            cur = self._dir_list.currentRow()
            if cur < 0 or cur >= len(self._profile.directories):
                return
            rows = {cur}
        n_dirs = len(self._profile.directories)
        valid_rows = {r for r in rows if 0 <= r < n_dirs}
        if not valid_rows:
            return
        if len(valid_rows) > 1:
            ok = self._app_actions.alert(
                _("Remove Directories"),
                _("Remove {0} selected directories from this profile?").format(len(valid_rows)),
                kind="askokcancel",
            )
            if not ok:
                return
        for idx in sorted(valid_rows, reverse=True):
            del self._profile.directories[idx]
        self._refresh_directory_list()

    def _clear_all_directories(self) -> None:
        self._profile.directories.clear()
        self._refresh_directory_list()
        logger.info("Cleared all directories from profile")

    def _add_subdirectories(self) -> None:
        parent_dir = self._browse_directory(_("Select directory to add subdirectories from"))
        if not parent_dir:
            return
        parent_dir = parent_dir.strip()
        if not os.path.isdir(parent_dir):
            logger.error(f"Invalid directory: {parent_dir}")
            return

        subdirs_added = 0
        try:
            for item in os.listdir(parent_dir):
                subdir_path = os.path.join(parent_dir, item)
                if os.path.isdir(subdir_path):
                    if subdir_path not in self._profile.directories:
                        self._profile.directories.append(subdir_path)
                        subdirs_added += 1
                    else:
                        logger.debug(f"Subdirectory {subdir_path} already in profile")
            if subdirs_added > 0:
                self._refresh_directory_list()
                logger.info(f"Added {subdirs_added} subdirectories from {parent_dir}")
            else:
                logger.info(f"No new subdirectories found in {parent_dir}")
        except Exception as e:
            logger.error(f"Error reading subdirectories from {parent_dir}: {e}")

    def _get_associated_prevalidations(self, profile_name: str) -> list[Prevalidation]:
        return [
            pv
            for pv in ClassifierActionsManager.prevalidations
            if pv.profile_name == profile_name
        ]

    def _generate_profile_copy_name(self, source_name: str) -> str:
        copy_token = _("Copy")
        copy_suffix = f" {copy_token}"
        fallback_suffix = " Copy"
        split_suffix = copy_suffix if copy_suffix in source_name else fallback_suffix
        if split_suffix in source_name:
            base_name = source_name.rsplit(split_suffix, 1)[0]
            copy_num = 2
            while True:
                candidate = f"{base_name}{copy_suffix} {copy_num}"
                if DirectoryProfile.get_profile_by_name(candidate) is None:
                    return candidate
                copy_num += 1
        candidate = f"{source_name}{copy_suffix}"
        if DirectoryProfile.get_profile_by_name(candidate) is None:
            return candidate
        copy_num = 2
        while True:
            candidate = f"{source_name}{copy_suffix} {copy_num}"
            if DirectoryProfile.get_profile_by_name(candidate) is None:
                return candidate
            copy_num += 1

    def _generate_prevalidation_copy_name(self, source_name: str, taken_names: Optional[set[str]] = None) -> str:
        existing = taken_names if taken_names is not None else {pv.name for pv in ClassifierActionsManager.prevalidations}
        copy_token = _("Copy")
        copy_suffix = f" {copy_token}"
        fallback_suffix = " Copy"
        split_suffix = copy_suffix if copy_suffix in source_name else fallback_suffix
        if split_suffix in source_name:
            base_name = source_name.rsplit(split_suffix, 1)[0]
            copy_num = 2
            while True:
                candidate = f"{base_name}{copy_suffix} {copy_num}"
                if candidate not in existing:
                    return candidate
                copy_num += 1
        candidate = f"{source_name}{copy_suffix}"
        if candidate not in existing:
            return candidate
        copy_num = 2
        while True:
            candidate = f"{source_name}{copy_suffix} {copy_num}"
            if candidate not in existing:
                return candidate
            copy_num += 1

    def _rebuild_prevalidation_modifier_rows(self) -> None:
        while self._pv_mod_layout.count():
            item = self._pv_mod_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            sub = item.layout()
            if sub is not None:
                while sub.count():
                    sub_item = sub.takeAt(0)
                    sub_widget = sub_item.widget()
                    if sub_widget is not None:
                        sub_widget.deleteLater()

        self._sync_prevalidation_copy_drafts_from_inputs()
        self._prevalidation_modifier_inputs = {}
        self._prevalidation_name_inputs = {}

        for draft in self._prevalidation_copy_drafts:
            source_name = draft["source_name"]
            row = QHBoxLayout()
            name_entry = QLineEdit(draft["name"])
            name_entry.setPlaceholderText(_("Copied prevalidation name"))
            name_entry.setMinimumWidth(220)
            name_entry.textChanged.connect(
                lambda text, sn=source_name: self._set_draft_field(sn, "name", text)
            )
            name_entry.returnPressed.connect(
                lambda sn=source_name: self._sync_single_prevalidation_draft(sn)
            )
            self._prevalidation_name_inputs[source_name] = name_entry
            row.addWidget(name_entry)

            modifier_entry = QLineEdit(draft["action_modifier"])
            modifier_entry.setPlaceholderText(_("Action target directory (action_modifier)"))
            modifier_entry.textChanged.connect(
                lambda text, sn=source_name: self._set_draft_field(sn, "action_modifier", text)
            )
            modifier_entry.returnPressed.connect(
                lambda sn=source_name: self._sync_single_prevalidation_draft(sn)
            )
            self._prevalidation_modifier_inputs[source_name] = modifier_entry
            row.addWidget(modifier_entry, 1)

            remove_btn = QPushButton(_("Remove"))
            remove_btn.clicked.connect(
                lambda _=False, sn=source_name: self._remove_associated_prevalidation(sn)
            )
            row.addWidget(remove_btn)

            self._pv_mod_layout.addLayout(row)

    def _set_draft_field(self, source_name: str, field: str, value: str) -> None:
        for draft in self._prevalidation_copy_drafts:
            if draft["source_name"] == source_name:
                draft[field] = value.strip()
                return

    def _sync_single_prevalidation_draft(self, source_name: str) -> None:
        name_entry = self._prevalidation_name_inputs.get(source_name)
        modifier_entry = self._prevalidation_modifier_inputs.get(source_name)
        if name_entry is None and modifier_entry is None:
            return
        for draft in self._prevalidation_copy_drafts:
            if draft["source_name"] == source_name:
                if name_entry is not None:
                    draft["name"] = name_entry.text().strip()
                if modifier_entry is not None:
                    draft["action_modifier"] = modifier_entry.text().strip()
                return

    def _sync_prevalidation_copy_drafts_from_inputs(self) -> None:
        for source_name in list(self._prevalidation_name_inputs.keys()):
            self._sync_single_prevalidation_draft(source_name)
        for source_name in list(self._prevalidation_modifier_inputs.keys()):
            self._sync_single_prevalidation_draft(source_name)

    def _remove_associated_prevalidation(self, source_name: str) -> None:
        ok = self._app_actions.alert(
            _("Remove Prevalidation"),
            _("Remove prevalidation '{0}' from this copy operation?").format(source_name),
            kind="askokcancel",
        )
        if not ok:
            return
        self._prevalidation_copy_drafts = [
            draft
            for draft in self._prevalidation_copy_drafts
            if draft["source_name"] != source_name
        ]
        self._rebuild_prevalidation_modifier_rows()
        self._toggle_prevalidation_modifier_inputs()

    def _toggle_prevalidation_modifier_inputs(self) -> None:
        if not hasattr(self, "_copy_prevalidations_cb"):
            return
        enabled = self._copy_prevalidations_cb.isChecked()
        if hasattr(self, "_pv_modifiers_container"):
            self._pv_modifiers_container.setVisible(enabled)
        if hasattr(self, "_pv_copy_hint_label"):
            self._pv_copy_hint_label.setVisible(enabled)
        for entry in self._prevalidation_modifier_inputs.values():
            entry.setEnabled(enabled)

    def _copy_associated_prevalidations(self, new_profile_name: str) -> None:
        if not hasattr(self, "_copy_prevalidations_cb") or not self._copy_prevalidations_cb.isChecked():
            return
        if len(self._prevalidation_copy_drafts) == 0:
            return

        self._sync_prevalidation_copy_drafts_from_inputs()
        used_names = {pv.name for pv in ClassifierActionsManager.prevalidations}
        for draft in self._prevalidation_copy_drafts:
            source_pv = self._source_prevalidations_by_name.get(draft["source_name"])
            if source_pv is None:
                continue
            source_dict = source_pv.to_dict()
            requested_name = draft["name"].strip()
            if requested_name and requested_name not in used_names:
                new_name = requested_name
            else:
                new_name = self._generate_prevalidation_copy_name(source_pv.name, used_names)
            used_names.add(new_name)
            source_dict["name"] = new_name
            source_dict["profile_name"] = new_profile_name
            source_dict["action_modifier"] = draft["action_modifier"].strip()

            copied_pv = Prevalidation.from_dict(source_dict)
            copied_pv.update_profile_instance(profile_name=new_profile_name)
            # For profile-copy workflows, preserve source order and place copied
            # prevalidations after existing entries (append behavior).
            ClassifierActionsManager.prevalidations.append(copied_pv)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def _finalize_profile(self) -> None:
        name = self._name_edit.text().strip()

        if not name:
            logger.error("Profile name is required")
            return

        # Duplicate-name check
        if not self._is_edit:
            if DirectoryProfile.get_profile_by_name(name) is not None:
                logger.error(f"Profile with name {name} already exists")
                return
        else:
            if name != self._original_name:
                if DirectoryProfile.get_profile_by_name(name) is not None:
                    logger.error(f"Profile with name {name} already exists")
                    return

        self._profile.name = name

        if not self._is_edit:
            if not DirectoryProfile.add_profile(self._profile):
                return
            if self._is_copy:
                # Create and register the copied profile before adding copied
                # prevalidations so profile_name resolution succeeds.
                self._copy_associated_prevalidations(self._profile.name)
        else:
            DirectoryProfile.update_profile(self._original_name, self._profile)

        self.close()
        self._refresh_callback()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        DirectoryProfileWindow._instance = None
        super().closeEvent(event)

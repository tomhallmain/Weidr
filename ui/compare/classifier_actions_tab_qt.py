"""
PySide6 port of compare/classifier_actions_tab.py -- ClassifierActionsTab.

Tab-page QWidget listing classifier actions with buttons for add / modify /
delete / copy / run, plus batch validation controls.

Non-UI imports:
  - ClassifierAction from compare.classifier_action
  - ClassifierActionsManager from compare.classifier_actions_manager
    from compare.classifier_actions_manager (reuse policy)
  - DirectoryProfile from compare.directory_profile (reuse policy)
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from compare.classifier_action import ClassifierAction
from compare.classifier_actions_manager import ClassifierActionsManager
from files.directory_profile import DirectoryProfile
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import I18N

_ = I18N._
logger = get_logger("classifier_actions_tab_qt")


class ClassifierActionsTab(QWidget):
    """
    Tab content widget for managing classifier actions.

    Can be embedded inside a QTabWidget (ClassifierManagementWindow)
    or used standalone.
    """

    _modify_window = None  # ClassifierActionModifyWindow or None

    BATCH_VALIDATION_MAX_IMAGES = 40000

    # ------------------------------------------------------------------
    # Static action runner (non-UI logic, kept for API compat)
    # ------------------------------------------------------------------
    @staticmethod
    def run_classifier_action(
        classifier_action: ClassifierAction,
        directory_paths: list[str],
        hide_callback: Callable,
        notify_callback: Callable,
        add_mark_callback: Optional[Callable] = None,
        profile_name_or_path: Optional[str] = None,
    ) -> None:
        """Run a classifier action across *directory_paths*."""
        classifier_action.run(
            directory_paths,
            hide_callback,
            notify_callback,
            add_mark_callback,
            profile_name_or_path,
            ClassifierActionsTab.BATCH_VALIDATION_MAX_IMAGES,
        )

    @staticmethod
    def _is_modify_window_valid() -> bool:
        win = ClassifierActionsTab._modify_window
        if win is None:
            return False
        try:
            return win.isVisible()
        except Exception:
            ClassifierActionsTab._modify_window = None
            return False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(parent)
        self._app_actions = app_actions
        self._filtered = ClassifierActionsManager.classifier_actions[:]

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # -- Title + buttons row ------------------------------------------
        title_row = QHBoxLayout()
        title_lbl = QLabel(_("Classifier Actions"))
        title_lbl.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 13pt;"
        )
        title_row.addWidget(title_lbl)

        add_btn = QPushButton(_("Add Classifier Action"))
        add_btn.clicked.connect(lambda: self._open_modify_window())
        title_row.addWidget(add_btn)

        clear_btn = QPushButton(_("Clear Classifier Actions"))
        clear_btn.clicked.connect(self._clear_all)
        title_row.addWidget(clear_btn)

        run_all_btn = QPushButton(_("Run All"))
        run_all_btn.clicked.connect(self._run_all)
        title_row.addWidget(run_all_btn)

        title_row.addStretch()
        root.addLayout(title_row)

        # -- Profile selection --------------------------------------------
        prof_row = QHBoxLayout()
        prof_lbl = QLabel(_("Run on Profile:"))
        prof_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        prof_row.addWidget(prof_lbl)

        self._profile_combo = QComboBox()
        self._refresh_profile_combo()
        prof_row.addWidget(self._profile_combo, 1)
        prof_row.addStretch()
        root.addLayout(prof_row)

        # -- Scrollable list of actions -----------------------------------
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {AppStyle.BG_COLOR}; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(2)
        self._scroll.setWidget(self._scroll_content)
        root.addWidget(self._scroll, 1)

        self._rebuild_rows()

    # ------------------------------------------------------------------
    # Profile combo
    # ------------------------------------------------------------------
    def _refresh_profile_combo(self) -> None:
        current = self._profile_combo.currentText()
        self._profile_combo.clear()
        options = [p.name for p in DirectoryProfile.directory_profiles]
        self._profile_combo.addItems(options)
        if current in options:
            self._profile_combo.setCurrentText(current)
        elif options:
            self._profile_combo.setCurrentIndex(0)
        self._profile_combo.setEnabled(bool(options))

    # ------------------------------------------------------------------
    # Rebuild action rows
    # ------------------------------------------------------------------
    def _rebuild_rows(self) -> None:
        _clear_layout(self._scroll_layout)

        # Header
        hdr = QHBoxLayout()
        for text, stretch in [
            ("", 0), (_("Active"), 0), (_("Name"), 1), (_("Action"), 0),
            ("", 0), ("", 0), ("", 0), ("", 0),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-weight: bold;")
            hdr.addWidget(lbl, stretch)
        self._scroll_layout.addLayout(hdr)

        for idx, ca in enumerate(self._filtered):
            row = QHBoxLayout()

            run_btn = QPushButton(_("Run"))
            run_btn.clicked.connect(lambda _=False, c=ca: self._run_single(c))
            row.addWidget(run_btn)

            active_cb = QCheckBox()
            active_cb.setChecked(ca.is_active)
            active_cb.stateChanged.connect(lambda state, c=ca: setattr(c, "is_active", bool(state)))
            row.addWidget(active_cb)

            name_lbl = QLabel(str(ca))
            name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            name_lbl.setWordWrap(True)
            row.addWidget(name_lbl, 1)

            action_lbl = QLabel(ca.action.get_translation())
            action_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            row.addWidget(action_lbl)

            mod_btn = QPushButton(_("Modify"))
            mod_btn.clicked.connect(lambda _=False, c=ca: self._open_modify_window(c))
            row.addWidget(mod_btn)

            copy_btn = QPushButton(_("Copy"))
            copy_btn.clicked.connect(lambda _=False, c=ca: self._open_copy_window(c))
            row.addWidget(copy_btn)

            del_btn = QPushButton(_("Delete"))
            del_btn.clicked.connect(lambda _=False, c=ca: self._delete(c))
            row.addWidget(del_btn)

            down_btn = QPushButton(_("Move down"))
            down_btn.clicked.connect(lambda _=False, i=idx, c=ca: self._move_down(i, c))
            row.addWidget(down_btn)

            self._scroll_layout.addLayout(row)

        self._scroll_layout.addStretch()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _open_modify_window(self, classifier_action=None) -> None:
        from ui.compare.classifier_management_window_qt import ClassifierActionModifyWindow

        if self._is_modify_window_valid():
            try:
                ClassifierActionsTab._modify_window.close()
            except Exception:
                ClassifierActionsTab._modify_window = None

        ClassifierActionsTab._modify_window = ClassifierActionModifyWindow(
            self.window(),
            self._app_actions,
            self.refresh_classifier_actions,
            classifier_action,
        )
        ClassifierActionsTab._modify_window.show()

    def _open_copy_window(self, classifier_action) -> None:
        from ui.compare.classifier_action_copy_window_qt import ClassifierActionCopyWindow

        ClassifierActionCopyWindow(
            self.window(),
            self._app_actions,
            classifier_action,
            source_type="classifier_action",
            refresh_classifier_actions_callback=self.refresh_classifier_actions,
            refresh_prevalidations_callback=None,
        ).show()

    def refresh_classifier_actions(self, classifier_action=None) -> None:
        if (
            classifier_action is not None
            and classifier_action not in ClassifierActionsManager.classifier_actions
        ):
            ClassifierActionsManager.classifier_actions.insert(0, classifier_action)
        self._filtered = ClassifierActionsManager.classifier_actions[:]
        self.refresh()

    def _delete(self, classifier_action) -> None:
        if (
            classifier_action is not None
            and classifier_action in ClassifierActionsManager.classifier_actions
        ):
            ClassifierActionsManager.classifier_actions.remove(classifier_action)
        self.refresh()

    def _move_down(self, idx: int, classifier_action) -> None:
        classifier_action.move_index(idx, 1)
        self.refresh()

    def _clear_all(self) -> None:
        ClassifierActionsManager.classifier_actions.clear()
        self._filtered.clear()
        self.refresh()

    def refresh(self) -> None:
        self._refresh_profile_combo()
        self._filtered = ClassifierActionsManager.classifier_actions[:]
        self._rebuild_rows()

    # ------------------------------------------------------------------
    # Run single / run all
    # ------------------------------------------------------------------
    def _get_selected_profile(self) -> Optional[DirectoryProfile]:
        name = self._profile_combo.currentText().strip()
        if not name:
            logger.error("No profile selected")
            return None
        profile = DirectoryProfile.get_profile_by_name(name)
        if profile is None:
            logger.error(f"Profile {name} not found")
        return profile

    def _format_directory_message(self, directories: list[str]) -> str:
        if len(directories) > 10:
            parent_counts: dict[str, int] = defaultdict(int)
            for d in directories:
                parent_counts[os.path.dirname(d)] += 1
            return "\n".join(f"{p} - {c} " + _("directories") for p, c in sorted(parent_counts.items()))
        return "\n".join(f"  - {d}" for d in directories)

    def _run_single(self, classifier_action: ClassifierAction) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            return
        if not classifier_action.can_run:
            self._app_actions.warn(
                _("This classifier action cannot run until configuration is fixed: {0}").format(
                    classifier_action.initialization_error or _("unknown error")
                )
            )
            return

        msg = _("Run classifier action '{0}' on the following directories?").format(
            classifier_action.name
        )
        full = f"{msg}\n\n{self._format_directory_message(profile.directories)}"

        if not qt_alert(_("Run Classifier Action"), full, kind="askokcancel", master=self):
            return

        hide_cb = getattr(self._app_actions, "hide_current_media", None)
        notify_cb = getattr(self._app_actions, "title_notify", None)
        add_mark_cb = None
        try:
            from files.marked_files import MarkedFiles
            add_mark_cb = MarkedFiles.add_mark_if_not_present
        except ImportError:
            pass

        ClassifierActionsTab.run_classifier_action(
            classifier_action,
            profile.directories,
            hide_cb,
            notify_cb,
            add_mark_cb,
            profile.name,
        )

    def _run_all(self) -> None:
        profile = self._get_selected_profile()
        if profile is None:
            return

        active = [ca for ca in self._filtered if ca.is_active and ca.can_run]
        if not active:
            self._app_actions.warn(_("No active classifier actions to run"))
            return

        action_list = "\n".join(f"  - {ca.name}" for ca in active)
        dir_list = self._format_directory_message(profile.directories)
        msg = _("Run {0} active classifier action(s) on the following directories?").format(
            len(active)
        )
        full = (
            f"{msg}\n\n{_('Active actions:')}\n{action_list}"
            f"\n\n{_('Directories:')}\n{dir_list}"
        )

        if not qt_alert(_("Run All Classifier Actions"), full, kind="askokcancel", master=self):
            return

        hide_cb = getattr(self._app_actions, "hide_current_media", None)
        notify_cb = getattr(self._app_actions, "title_notify", None)
        add_mark_cb = None
        try:
            from files.marked_files import MarkedFiles
            add_mark_cb = MarkedFiles.add_mark_if_not_present
        except ImportError:
            pass

        for ca in active:
            ClassifierActionsTab.run_classifier_action(
                ca, profile.directories, hide_cb, notify_cb,
                add_mark_cb, profile.name,
            )


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

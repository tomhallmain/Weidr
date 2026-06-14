"""
PySide6 port of compare/prevalidations_tab.py.

Contains two classes:
  - PrevalidationModifyWindow  -- subclass of ClassifierActionModifyWindow
    adding lookahead and directory-profile fields.
  - PrevalidationsTab          -- tab-page QWidget listing prevalidations
    with add / modify / delete / copy / run controls, plus lookahead and
    directory-profile management sections.

Non-UI imports:
  - Prevalidation from compare.classifier_action
  - ClassifierActionsManager from compare.classifier_actions_manager
  - Lookahead from compare.lookahead (reuse policy)
  - DirectoryProfile from compare.directory_profile (reuse policy)

Cache-invalidation policy
-------------------------
Every mutating operation must evict stale prevalidation results from both
the session cache (ClassifierActionsManager.prevalidated_cache) and the
persistent per-file bucket store (lib.file_invalidation_cache).

On bulk load from disk the full cache is always wiped via
ClassifierActionsManager.clear_prevalidation_result_cache().  For
interactive edits during a session, targeted eviction is preferred so that
expensive dynamic-media results for unaffected directories are preserved.

The helpers ``_pv_dirs`` and ``_profile_linked_dirs`` compute the set of
filesystem directories that a prevalidation or profile touches.  A return
value of ``None`` from ``_pv_dirs`` means the prevalidation is global-scoped
(no profile), which forces a full eviction via the ``_invalidate_for_dir_sets``
dispatcher.

Per-operation rules
~~~~~~~~~~~~~~~~~~~
Prevalidation *saved* (add or modify):
    Evict the union of the prevalidation's profile directories **before** the
    edit (snapshot taken in ``_open_modify_window`` before the window mutates
    the object) and **after** the edit (read back from the now-modified object
    in the callback).  Either state being ``None`` (global scope) forces a
    full eviction.  A freshly added prevalidation has no prior state, so
    ``old_dirs`` is treated as an empty set.

Prevalidation *added via copy* (``ClassifierActionCopyWindow``):
    The new prevalidation has no old state.  ``refresh_prevalidations`` falls
    back to a full eviction because there is no snapshot — conservative but
    correct, and this path is rare.

Prevalidation *deleted*:
    If the deleted prevalidation was profile-scoped, evict only its profile
    directories.  If it was global, its removal **narrows** the effective
    scope: no previously cached result becomes wrong, so no eviction is
    needed.

Prevalidation *reordered*:
    No eviction.  Reordering can in theory change which prevalidation fires
    first, but re-evaluating all dynamic media on every drag is too costly.
    Operators can force a re-run via the ``force=True`` keybind path.

All prevalidations *cleared*:
    Full eviction.

Profile *added* or *copied*:
    No eviction — a brand-new profile has no prevalidations linked to it so
    no files were ever cached under its scope.

Profile *edited* (directories changed):
    Evict the union of the profile's directory set **before** the edit
    (snapshot taken in ``_edit_profile``) and **after** the edit (read from
    the now-modified profile object in the callback), but **only** if at
    least one prevalidation currently references this profile.  If no
    prevalidations reference it, no files were cached under its scope.

Profile *removed*:
    If prevalidations referenced it their scope expands to global (the
    profile filter no longer applies), meaning cached ``None`` results in
    previously unscoped directories may now be wrong → **full eviction**.
    If no prevalidations referenced the profile, no eviction is needed.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QGridLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from files.directory_profile import DirectoryProfile
from compare.lookahead import Lookahead
from ui.compare.classifier_management_window_qt import ClassifierActionModifyWindow
from ui.app_style import AppStyle
from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils
logger = get_logger("prevalidations_tab_qt")


# ======================================================================
# PrevalidationModifyWindow
# ======================================================================
class PrevalidationModifyWindow(ClassifierActionModifyWindow):
    """Modify dialog for Prevalidation objects, adding lookahead/profile fields."""

    _pv_instance: Optional[PrevalidationModifyWindow] = None

    def __init__(
        self,
        parent: QWidget,
        app_actions,
        refresh_callback,
        prevalidation: Optional[Prevalidation] = None,
        dimensions: str = "600x600",
    ) -> None:
        prevalidation = prevalidation if prevalidation is not None else Prevalidation()
        super().__init__(
            parent,
            app_actions,
            refresh_callback,
            prevalidation,
            _("Modify Prevalidation"),
            _("Prevalidation Name"),
            _("New Prevalidation"),
            dimensions,
        )
        PrevalidationModifyWindow._pv_instance = self

    # ------------------------------------------------------------------
    # Subclass hook: add lookahead + profile fields
    # ------------------------------------------------------------------
    def add_specific_fields(self, grid: QGridLayout, row: int) -> int:
        pv = self._classifier_action  # actually a Prevalidation

        # -- Lookaheads multi-select --------------------------------------
        grid.addWidget(
            self._lbl(_("Lookaheads (select from shared list)")),
            row, 0, Qt.AlignLeft | Qt.AlignTop,
        )
        self._lookahead_list = QListWidget()
        self._lookahead_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self._lookahead_list.setMaximumHeight(80)
        self._populate_lookahead_list()
        grid.addWidget(self._lookahead_list, row, 1)

        # -- Profile dropdown ---------------------------------------------
        row += 1
        grid.addWidget(
            self._lbl(_("Directory Profile")), row, 0, Qt.AlignLeft
        )
        self._profile_combo = QComboBox()
        profile_options = [""]
        profile_options.extend(
            p.name for p in DirectoryProfile.directory_profiles
        )
        self._profile_combo.addItems(profile_options)
        current_profile = pv.profile_name if pv.profile_name else ""
        if current_profile in profile_options:
            self._profile_combo.setCurrentText(current_profile)
        else:
            self._profile_combo.setCurrentIndex(0)
        grid.addWidget(self._profile_combo, row, 1)

        return row + 1

    def _populate_lookahead_list(self) -> None:
        pv = self._classifier_action
        self._lookahead_list.clear()
        for lh in Lookahead.lookaheads:
            item = QListWidgetItem(lh.name)
            self._lookahead_list.addItem(item)
            if lh.name in pv.lookahead_names:
                item.setSelected(True)

    def refresh_lookahead_options(self) -> None:
        self._populate_lookahead_list()

    def refresh_profile_options(self) -> None:
        current = self._profile_combo.currentText()
        self._profile_combo.clear()
        options = [""]
        options.extend(p.name for p in DirectoryProfile.directory_profiles)
        self._profile_combo.addItems(options)
        if current in options:
            self._profile_combo.setCurrentText(current)
        else:
            self._profile_combo.setCurrentIndex(0)

    def _finalize_specific(self) -> None:
        pv = self._classifier_action
        pv.lookahead_names = [
            item.text() for item in self._lookahead_list.selectedItems()
        ]
        selected_profile = self._profile_combo.currentText().strip()
        profile_name = selected_profile if selected_profile else None
        pv.update_profile_instance(profile_name=profile_name)

    def closeEvent(self, event) -> None:  # noqa: N802
        PrevalidationModifyWindow._pv_instance = None
        super().closeEvent(event)


# ======================================================================
# PrevalidationsTab
# ======================================================================
class PrevalidationsTab(QWidget):
    """
    Tab content widget for managing prevalidations.

    Sections (top-to-bottom):
      1. Lookahead management (QListWidget + Add/Edit/Remove)
      2. Directory profile management (QListWidget + Add/Edit/Remove)
      3. Prevalidation list (scrollable rows with action buttons)
    """

    _modify_window: Optional[PrevalidationModifyWindow] = None
    _lookahead_window = None
    _profile_window = None

    @staticmethod
    def _is_modify_window_valid() -> bool:
        win = PrevalidationsTab._modify_window
        if win is None:
            return False
        try:
            return win.isVisible()
        except Exception:
            PrevalidationsTab._modify_window = None
            return False

    @staticmethod
    def clear_prevalidated_cache() -> None:
        """
        Clear in-memory prevalidation results for the current run.

        Note: we intentionally do not bump the prevalidation policy epoch here.
        Manual keybind runs can bypass stale results via the `force=` flag.
        """
        ClassifierActionsManager.prevalidated_cache.clear()
        ClassifierActionsManager.directories_to_exclude.clear()

    @staticmethod
    def remove_directory_from_exclusion_list(directory: str) -> bool:
        if directory in ClassifierActionsManager.directories_to_exclude:
            ClassifierActionsManager.directories_to_exclude.remove(directory)
            return True
        return False

    @staticmethod
    def add_directory_to_exclusion_list(directory: str):
        ClassifierActionsManager.directories_to_exclude.append(directory)

    @staticmethod
    def prevalidate(
        media_path,
        get_base_dir_func,
        callbacks: ActionCallbacks,
        force: bool = False,
    ):
        """Run prevalidations and return action type or None."""
        return ClassifierActionsManager.prevalidate_media(
            media_path,
            get_base_dir_func,
            callbacks,
            force=force,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(parent)
        self._app_actions = app_actions
        self._filtered = ClassifierActionsManager.prevalidations[:]

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ---- Lookahead management section --------------------------------
        root.addWidget(self._section_label(_("Lookaheads")))
        lh_area = QHBoxLayout()

        self._lh_listbox = QListWidget()
        self._lh_listbox.setMaximumHeight(100)
        self._lh_listbox.setStyleSheet(
            f"QListWidget {{ background: {AppStyle.BG_COLOR}; color: {AppStyle.FG_COLOR}; }}"
        )
        self._lh_listbox.doubleClicked.connect(self._edit_lookahead)
        lh_area.addWidget(self._lh_listbox, 1)

        lh_btns = QVBoxLayout()
        lh_btns.setSpacing(2)
        add_lh = QPushButton(_("Add Lookahead"))
        add_lh.clicked.connect(self._add_lookahead)
        lh_btns.addWidget(add_lh)
        edit_lh = QPushButton(_("Edit Lookahead"))
        edit_lh.clicked.connect(self._edit_lookahead)
        lh_btns.addWidget(edit_lh)
        rm_lh = QPushButton(_("Remove Lookahead"))
        rm_lh.clicked.connect(self._remove_lookahead)
        lh_btns.addWidget(rm_lh)
        lh_btns.addStretch()
        lh_area.addLayout(lh_btns)
        root.addLayout(lh_area)

        # ---- Directory profile management section ------------------------
        root.addWidget(self._section_label(_("Directory Profiles")))
        prof_area = QHBoxLayout()

        self._prof_listbox = QListWidget()
        self._prof_listbox.setMaximumHeight(100)
        self._prof_listbox.setStyleSheet(
            f"QListWidget {{ background: {AppStyle.BG_COLOR}; color: {AppStyle.FG_COLOR}; }}"
        )
        self._prof_listbox.doubleClicked.connect(self._edit_profile)
        prof_area.addWidget(self._prof_listbox, 1)

        prof_btns = QVBoxLayout()
        prof_btns.setSpacing(2)
        add_prof = QPushButton(_("Add Profile"))
        add_prof.clicked.connect(self._add_profile)
        prof_btns.addWidget(add_prof)
        edit_prof = QPushButton(_("Edit Profile"))
        edit_prof.clicked.connect(self._edit_profile)
        prof_btns.addWidget(edit_prof)
        copy_prof = QPushButton(_("Copy Profile"))
        copy_prof.clicked.connect(self._copy_profile)
        prof_btns.addWidget(copy_prof)
        rm_prof = QPushButton(_("Remove Profile"))
        rm_prof.clicked.connect(self._remove_profile)
        prof_btns.addWidget(rm_prof)
        prof_btns.addStretch()
        prof_area.addLayout(prof_btns)
        root.addLayout(prof_area)

        # ---- Prevalidations title + buttons ------------------------------
        pv_title_row = QHBoxLayout()
        pv_title_row.addWidget(self._section_label(_("Prevalidations")))
        add_pv = QPushButton(_("Add prevalidation"))
        add_pv.clicked.connect(lambda: self._open_modify_window())
        pv_title_row.addWidget(add_pv)
        self._clear_pv_btn = QPushButton(_("Clear prevalidations"))
        self._clear_pv_btn.clicked.connect(self._clear_all)
        pv_title_row.addWidget(self._clear_pv_btn)
        clear_pv_cache_btn = QPushButton(_("Clear prevalidation cache"))
        clear_pv_cache_btn.setToolTip(
            _("Discard cached prevalidation outcomes (this session and on disk).\n"
              "Rules, profiles, and lookaheads are not changed; media will be re-evaluated.")
        )
        clear_pv_cache_btn.clicked.connect(self._clear_prevalidation_result_cache_only)
        pv_title_row.addWidget(clear_pv_cache_btn)
        self._clear_current_dir_cb = QCheckBox("")
        self._clear_current_dir_cb.setToolTip(
            _("When checked, the clear button evicts only the cached results\n"
              "for the current base directory — prevalidation rules are kept.")
        )
        self._clear_current_dir_cb.toggled.connect(self._on_clear_scope_changed)
        pv_title_row.addWidget(self._clear_current_dir_cb)
        self._update_clear_dir_label()
        pv_title_row.addStretch()
        root.addLayout(pv_title_row)

        self._pv_cache_stats_lbl = QLabel()
        self._pv_cache_stats_lbl.setWordWrap(True)
        self._pv_cache_stats_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(self._pv_cache_stats_lbl)

        # -- Enable prevalidations checkbox --------------------------------
        self._enable_pv_cb = QCheckBox(_("Enable Prevalidations"))
        self._enable_pv_cb.setChecked(config.enable_prevalidations)
        self._enable_pv_cb.stateChanged.connect(
            lambda state: setattr(config, "enable_prevalidations", bool(state))
        )
        root.addWidget(self._enable_pv_cb)

        # -- Scrollable prevalidation rows ---------------------------------
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

        # Populate
        self._refresh_lh_listbox()
        self._refresh_prof_listbox()
        self._rebuild_pv_rows()
        self._update_prevalidation_cache_stats_label()

    # ------------------------------------------------------------------
    # Lookahead management
    # ------------------------------------------------------------------
    def _refresh_lh_listbox(self) -> None:
        self._lh_listbox.clear()
        for lh in Lookahead.lookaheads:
            text = _("{name} ({name_or_text}, threshold: {threshold:.2f})").format(
                name=lh.name,
                name_or_text=lh.name_or_text,
                threshold=lh.threshold,
            )
            self._lh_listbox.addItem(text)

    def _add_lookahead(self) -> None:
        from ui.compare.lookahead_window_qt import LookaheadWindow

        if PrevalidationsTab._lookahead_window is not None:
            try:
                PrevalidationsTab._lookahead_window.close()
            except Exception:
                pass
        PrevalidationsTab._lookahead_window = LookaheadWindow(
            self.window(), self._app_actions, self._refresh_lh_listbox
        )
        PrevalidationsTab._lookahead_window.show()

    def _edit_lookahead(self) -> None:
        from ui.compare.lookahead_window_qt import LookaheadWindow

        idx = self._lh_listbox.currentRow()
        if idx < 0 or idx >= len(Lookahead.lookaheads):
            return
        if PrevalidationsTab._lookahead_window is not None:
            try:
                PrevalidationsTab._lookahead_window.close()
            except Exception:
                pass
        PrevalidationsTab._lookahead_window = LookaheadWindow(
            self.window(),
            self._app_actions,
            self._refresh_lh_listbox,
            Lookahead.lookaheads[idx],
        )
        PrevalidationsTab._lookahead_window.show()

    def _remove_lookahead(self) -> None:
        idx = self._lh_listbox.currentRow()
        if idx < 0 or idx >= len(Lookahead.lookaheads):
            return
        lh = Lookahead.lookaheads[idx]
        used_by = [
            pv.name
            for pv in ClassifierActionsManager.prevalidations
            if lh.name in pv.lookahead_names
        ]
        if used_by:
            logger.warning(
                f"Lookahead {lh.name} is used by prevalidations: "
                f"{', '.join(used_by)}"
            )
        del Lookahead.lookaheads[idx]
        self._refresh_lh_listbox()
        if self._is_modify_window_valid():
            try:
                PrevalidationsTab._modify_window.refresh_lookahead_options()
            except Exception:
                PrevalidationsTab._modify_window = None

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------
    def _refresh_prof_listbox(self) -> None:
        self._prof_listbox.clear()
        for profile in DirectoryProfile.directory_profiles:
            n = len(profile.directories)
            word = _("directory") if n == 1 else _("directories")
            self._prof_listbox.addItem(f"{profile.name} ({n} {word})")

        if self._is_modify_window_valid():
            try:
                PrevalidationsTab._modify_window.refresh_profile_options()
            except Exception:
                PrevalidationsTab._modify_window = None

    def _add_profile(self) -> None:
        from ui.compare.directory_profile_window_qt import DirectoryProfileWindow

        if PrevalidationsTab._profile_window is not None:
            try:
                PrevalidationsTab._profile_window.close()
            except Exception:
                pass
        # New profile has no prevalidations linked → no cache eviction needed.
        PrevalidationsTab._profile_window = DirectoryProfileWindow(
            self.window(),
            self._app_actions,
            self._on_profile_added,
        )
        PrevalidationsTab._profile_window.show()

    def _edit_profile(self) -> None:
        from ui.compare.directory_profile_window_qt import DirectoryProfileWindow

        idx = self._prof_listbox.currentRow()
        if idx < 0 or idx >= len(DirectoryProfile.directory_profiles):
            return
        if PrevalidationsTab._profile_window is not None:
            try:
                PrevalidationsTab._profile_window.close()
            except Exception:
                pass
        profile = DirectoryProfile.directory_profiles[idx]
        # Snapshot the profile's linked directories *before* the window edits
        # the profile object in-place.
        old_dirs = self._profile_linked_dirs(profile)

        def _on_edited(*_args) -> None:
            # profile.directories now reflects the post-edit state.
            new_dirs = self._profile_linked_dirs(profile)
            self._invalidate_for_dir_sets(
                old_dirs, new_dirs,
                reason=f"profile '{profile.name}' edited",
            )
            self._rebuild_supporting_state()
            self._refresh_prof_listbox()

        PrevalidationsTab._profile_window = DirectoryProfileWindow(
            self.window(),
            self._app_actions,
            _on_edited,
            profile,
        )
        PrevalidationsTab._profile_window.show()

    def _remove_profile(self) -> None:
        idx = self._prof_listbox.currentRow()
        if idx < 0 or idx >= len(DirectoryProfile.directory_profiles):
            return
        profile = DirectoryProfile.directory_profiles[idx]
        # Check linkage *before* removal while prevalidations still reference it.
        linked = self._profile_linked_dirs(profile)
        DirectoryProfile.remove_profile(profile.name)
        if linked:
            # Linked prevalidations lose their profile scope and become global;
            # cached None results in previously unscoped dirs are now stale.
            logger.info(
                "Prevalidation cache: full eviction — profile '%s' removed,"
                " linked prevalidations become global-scoped",
                profile.name,
            )
            ClassifierActionsManager.clear_prevalidation_result_cache()
        else:
            logger.info(
                "Prevalidation cache: no eviction — profile '%s' removed"
                " but no prevalidations were linked to it",
                profile.name,
            )
        self._rebuild_supporting_state()
        if self._is_modify_window_valid():
            try:
                PrevalidationsTab._modify_window.refresh_profile_options()
            except Exception:
                PrevalidationsTab._modify_window = None

    def _copy_profile(self) -> None:
        from ui.compare.directory_profile_window_qt import DirectoryProfileWindow

        idx = self._prof_listbox.currentRow()
        if idx < 0 or idx >= len(DirectoryProfile.directory_profiles):
            return
        if PrevalidationsTab._profile_window is not None:
            try:
                PrevalidationsTab._profile_window.close()
            except Exception:
                pass
        # Copied profile is new → no prevalidations linked → no eviction needed.
        PrevalidationsTab._profile_window = DirectoryProfileWindow(
            self.window(),
            self._app_actions,
            self._on_profile_added,
            copy_from_profile=DirectoryProfile.directory_profiles[idx],
        )
        PrevalidationsTab._profile_window.show()

    def _on_profile_added(self, *_args) -> None:
        """Callback for profile add/copy — no eviction, just rebuild the UI."""
        self._rebuild_supporting_state()
        self._refresh_prof_listbox()

    # ------------------------------------------------------------------
    # Prevalidation rows
    # ------------------------------------------------------------------
    def _rebuild_pv_rows(self) -> None:
        _clear_layout(self._scroll_layout)

        # Header
        hdr = QHBoxLayout()
        for text, stretch in [
            (_("Name"), 1), (_("Action"), 0), (_("Profile"), 0),
            (_("Active"), 0), ("", 0), ("", 0), ("", 0), ("", 0),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-weight: bold;")
            hdr.addWidget(lbl, stretch)
        self._scroll_layout.addLayout(hdr)

        for idx, pv in enumerate(self._filtered):
            row = QHBoxLayout()

            name_lbl = QLabel(str(pv))
            name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            name_lbl.setWordWrap(True)
            row.addWidget(name_lbl, 1)

            action_lbl = QLabel(pv.action.get_translation())
            action_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            row.addWidget(action_lbl)

            # Profile column
            if pv.profile_name:
                prof_text = pv.profile_name
            elif pv.profile:
                prof_text = pv.profile.name
            else:
                prof_text = _("(Global)")
            prof_lbl = QLabel(prof_text)
            prof_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            row.addWidget(prof_lbl)

            active_cb = QCheckBox()
            active_cb.setChecked(pv.is_active)
            active_cb.stateChanged.connect(
                lambda state, p=pv: self._toggle_active(p, bool(state))
            )
            row.addWidget(active_cb)

            mod_btn = QPushButton(_("Modify"))
            mod_btn.clicked.connect(
                lambda _=False, p=pv: self._open_modify_window(p)
            )
            row.addWidget(mod_btn)

            copy_btn = QPushButton(_("Copy"))
            copy_btn.clicked.connect(
                lambda _=False, p=pv: self._open_copy_window(p)
            )
            row.addWidget(copy_btn)

            del_btn = QPushButton(_("Delete"))
            del_btn.clicked.connect(
                lambda _=False, p=pv: self._delete(p)
            )
            row.addWidget(del_btn)

            down_btn = QPushButton(_("Move down"))
            down_btn.clicked.connect(
                lambda _=False, i=idx, p=pv: self._move_down(i, p)
            )
            row.addWidget(down_btn)

            self._scroll_layout.addLayout(row)

        self._scroll_layout.addStretch()

    # ------------------------------------------------------------------
    # Prevalidation actions
    # ------------------------------------------------------------------
    def _open_modify_window(self, prevalidation=None) -> None:
        if PrevalidationsTab._modify_window is not None:
            try:
                PrevalidationsTab._modify_window.close()
            except Exception:
                pass
        # Snapshot the prevalidation's current profile dirs *before* the
        # modify window mutates the object in-place via _finalize_specific.
        # An add (prevalidation=None) has no prior state → treat as empty set.
        self._modify_old_dirs = (
            self._pv_dirs(prevalidation) if prevalidation is not None else set()
        )
        PrevalidationsTab._modify_window = PrevalidationModifyWindow(
            self.window(),
            self._app_actions,
            self.refresh_prevalidations,
            prevalidation,
        )
        PrevalidationsTab._modify_window.show()

    def _open_copy_window(self, prevalidation) -> None:
        from ui.compare.classifier_action_copy_window_qt import (
            ClassifierActionCopyWindow,
        )

        ClassifierActionCopyWindow(
            self.window(),
            self._app_actions,
            prevalidation,
            source_type="prevalidation",
            refresh_classifier_actions_callback=None,
            refresh_prevalidations_callback=self.refresh_prevalidations,
        ).show()

    # ------------------------------------------------------------------
    # Cache-eviction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pv_dirs(pv: "Prevalidation") -> "set[str] | None":
        """
        Return the set of directories this prevalidation is scoped to, or
        ``None`` if it is global (no profile).
        """
        if pv.profile is not None:
            return set(pv.profile.directories)
        if pv.profile_name:
            prof = next(
                (p for p in DirectoryProfile.directory_profiles
                 if p.name == pv.profile_name),
                None,
            )
            if prof:
                return set(prof.directories)
        return None

    @staticmethod
    def _profile_linked_dirs(profile: "DirectoryProfile") -> set[str]:
        """
        Return the profile's directory set if at least one prevalidation
        currently references it, otherwise an empty set (nothing to evict).
        """
        used = any(
            pv.profile_name == profile.name or pv.profile is profile
            for pv in ClassifierActionsManager.prevalidations
        )
        return set(profile.directories) if used else set()

    def _invalidate_for_dir_sets(
        self, *dir_sets: "set[str] | None", reason: str = ""
    ) -> None:
        """
        Evict the union of *dir_sets* from both caches, or perform a full
        eviction if any element is ``None`` (global prevalidation scope).
        """
        if any(d is None for d in dir_sets):
            logger.info(
                "Prevalidation cache: full eviction requested — %s"
                " (global-scoped prevalidation affected)",
                reason,
            )
            ClassifierActionsManager.clear_prevalidation_result_cache()
            return
        affected: set[str] = set()
        for d in dir_sets:
            affected |= d  # type: ignore[operator]
        if affected:
            logger.info(
                "Prevalidation cache: targeted eviction requested — %s", reason
            )
            ClassifierActionsManager.invalidate_for_directories(affected)
        else:
            logger.info(
                "Prevalidation cache: no eviction needed — %s"
                " (no directories affected)",
                reason,
            )

    def _rebuild_supporting_state(self) -> None:
        """Rebuild directories_to_exclude and repaint the UI rows."""
        self._filtered = ClassifierActionsManager.prevalidations[:]
        ClassifierActionsManager.directories_to_exclude.clear()
        for pv in ClassifierActionsManager.prevalidations:
            if pv.is_move_action() and pv.action_modifier:
                ClassifierActionsManager.directories_to_exclude.append(
                    pv.action_modifier
                )
        self.refresh()

    # ------------------------------------------------------------------
    # Prevalidation actions
    # ------------------------------------------------------------------
    def refresh_prevalidations(self, prevalidation=None) -> None:
        if (
            prevalidation is not None
            and prevalidation not in ClassifierActionsManager.prevalidations
        ):
            ClassifierActionsManager.prevalidations.insert(0, prevalidation)

        # Consume any targeted-eviction snapshot set by _open_modify_window.
        # If the attribute is absent the call came from an external caller
        # (e.g. ClassifierActionCopyWindow) → fall back to full eviction.
        if hasattr(self, "_modify_old_dirs"):
            old_dirs = self._modify_old_dirs
            del self._modify_old_dirs
            new_dirs = (
                self._pv_dirs(prevalidation)
                if prevalidation is not None
                else set()
            )
            pv_name = prevalidation.name if prevalidation is not None else "<new>"
            self._invalidate_for_dir_sets(
                old_dirs, new_dirs,
                reason=f"prevalidation '{pv_name}' saved",
            )
        else:
            logger.info(
                "Prevalidation cache: full eviction — prevalidation saved"
                " via external caller (no snapshot available)"
            )
            ClassifierActionsManager.clear_prevalidation_result_cache()

        self._rebuild_supporting_state()

    def _toggle_active(self, prevalidation, value: bool) -> None:
        prevalidation.is_active = value
        dirs = self._pv_dirs(prevalidation)
        label = f"prevalidation '{prevalidation.name}' {'activated' if value else 'deactivated'}"
        if dirs is None:
            # Global scope: clear all in-memory entries and memo, but leave
            # buckets intact — stale entries self-invalidate via signature mismatch.
            logger.info("Prevalidation cache: session-cache eviction — %s (global)", label)
            ClassifierActionsManager.invalidate_session_cache_only()
        else:
            ClassifierActionsManager.invalidate_for_directories(
                dirs, evict_buckets=False
            )

    def _delete(self, prevalidation) -> None:
        # Read dirs before removal; the object's profile attrs are still valid
        # after list removal since we hold a reference to the same instance.
        dirs = self._pv_dirs(prevalidation) if prevalidation is not None else set()
        if (
            prevalidation is not None
            and prevalidation in ClassifierActionsManager.prevalidations
        ):
            ClassifierActionsManager.prevalidations.remove(prevalidation)
        # Global prevalidation (dirs is None) removed → effective scope narrows,
        # no cached result becomes wrong → no eviction needed.
        # Profile-scoped removal → evict only the affected directories.
        if dirs is not None:
            pv_name = prevalidation.name if prevalidation is not None else "<unknown>"
            self._invalidate_for_dir_sets(
                dirs, reason=f"prevalidation '{pv_name}' deleted"
            )
        else:
            logger.info(
                "Prevalidation cache: no eviction — global prevalidation '%s'"
                " deleted (scope narrows, no cached results become stale)",
                prevalidation.name if prevalidation is not None else "<unknown>",
            )
        self._rebuild_supporting_state()

    def _move_down(self, idx: int, prevalidation) -> None:
        prevalidation.move_index(idx, 1)
        self.refresh()

    def _update_clear_dir_label(self) -> None:
        """Refresh the scope checkbox text to reflect the current base directory."""
        base_dir = self._app_actions.get_base_dir()
        if base_dir:
            short = Utils.get_relative_dirpath(base_dir, levels=2)
        else:
            short = _("current dir")
        self._clear_current_dir_cb.setText(_("Only: {0}").format(short))
        # Keep button text in sync if scope is already set to dir-only
        if self._clear_current_dir_cb.isChecked():
            self._clear_pv_btn.setText(_("Clear results ({0})").format(short))

    def _on_clear_scope_changed(self, current_dir_only: bool) -> None:
        base_dir = self._app_actions.get_base_dir()
        short = Utils.get_relative_dirpath(base_dir, levels=2) if base_dir else _("current dir")
        if current_dir_only:
            self._clear_pv_btn.setText(_("Clear results ({0})").format(short))
        else:
            self._clear_pv_btn.setText(_("Clear prevalidations"))

    def _clear_all(self) -> None:
        if self._clear_current_dir_cb.isChecked():
            base_dir = self._app_actions.get_base_dir()
            if not base_dir:
                return
            logger.info(
                "Prevalidation cache: evicting results for current directory: %s", base_dir
            )
            ClassifierActionsManager.invalidate_for_directories({base_dir})
            self._update_prevalidation_cache_stats_label()
            return
        if not self._app_actions.alert(
            _("Confirm Clear Prevalidations"),
            _(
                "Remove all prevalidation rules? Cached outcomes for this session "
                "will also be cleared.\n\nThis cannot be undone."
            ),
            kind="askokcancel",
            severity="high",
            master=self,
        ):
            return
        ClassifierActionsManager.prevalidations.clear()
        self._filtered.clear()
        logger.info("Prevalidation cache: full eviction — all prevalidations cleared")
        ClassifierActionsManager.clear_prevalidation_result_cache()
        self._rebuild_supporting_state()

    def _clear_prevalidation_result_cache_only(self) -> None:
        """Clear stored prevalidation outcomes; keep rules and refresh move exclusions."""
        logger.info("Prevalidation cache: full eviction — manual clear from prevalidations tab")
        ClassifierActionsManager.clear_prevalidation_result_cache()
        self._rebuild_supporting_state()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._update_clear_dir_label()
        self._update_prevalidation_cache_stats_label()

    def refresh(self) -> None:
        self._filtered = ClassifierActionsManager.prevalidations[:]
        self._refresh_lh_listbox()
        self._refresh_prof_listbox()
        self._rebuild_pv_rows()
        self._update_clear_dir_label()
        self._update_prevalidation_cache_stats_label()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_approx_bytes(num_bytes: int) -> str:
        if num_bytes < 1024:
            return f"{num_bytes} B"
        if num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.1f} KiB"
        return f"{num_bytes / (1024 * 1024):.1f} MiB"

    def _update_prevalidation_cache_stats_label(self) -> None:
        if not hasattr(self, "_pv_cache_stats_lbl"):
            return
        est_bytes, n_items, n_dirs = ClassifierActionsManager.get_prevalidation_cache_statistics()
        self._pv_cache_stats_lbl.setText(
            _("Prevalidation cache stats: ~{memory} — {items} items — {directories} directories").format(
                memory=self._format_approx_bytes(est_bytes),
                items=n_items,
                directories=n_dirs,
            )
        )

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 13pt;"
        )
        return lbl


# ======================================================================
# Layout helper
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

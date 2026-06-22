"""
DirectoryProfilesTab — standalone tab for managing DirectoryProfile objects.

Extracted from PrevalidationsTab so profiles can be managed independently.
Cache-eviction logic (invalidating prevalidation results when a profile's
directory set changes) is preserved here because profile edits/removals
directly affect cached prevalidation outcomes.

Cross-tab notification (refreshing an open PrevalidationModifyWindow or the
Prevalidations tab when a profile is added/edited/removed) is deferred; it
will be wired via the ClassifierManagementWindow once a shared refresh
protocol is in place.

Cache-invalidation policy — profile operations
----------------------------------------------
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

``_profile_linked_dirs`` returns the profile's directory set only when at
least one prevalidation references the profile, otherwise an empty set so
that unlinked profile changes never trigger unnecessary eviction.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from compare.classifier_actions_manager import ClassifierActionsManager
from files.directory_profile import DirectoryProfile
from ui.app_style import AppStyle
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("directory_profiles_tab_qt")


class DirectoryProfilesTab(QWidget):
    """Tab content widget for managing DirectoryProfile objects."""

    _profile_window = None

    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(parent)
        self._app_actions = app_actions

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        title = QLabel(_("Directory Profiles"))
        title.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 13pt;"
        )
        root.addWidget(title)

        prof_area = QHBoxLayout()

        self._prof_listbox = QListWidget()
        self._prof_listbox.setStyleSheet(
            f"QListWidget {{ background: {AppStyle.BG_COLOR};"
            f" color: {AppStyle.FG_COLOR}; }}"
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
        root.addStretch()

        self._refresh_prof_listbox()

    # ------------------------------------------------------------------
    # List population
    # ------------------------------------------------------------------

    def _refresh_prof_listbox(self) -> None:
        self._prof_listbox.clear()
        for profile in DirectoryProfile.directory_profiles:
            n = len(profile.directories)
            word = _("directory") if n == 1 else _("directories")
            self._prof_listbox.addItem(f"{profile.name} ({n} {word})")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _add_profile(self) -> None:
        from ui.compare.directory_profile_window_qt import DirectoryProfileWindow

        if DirectoryProfilesTab._profile_window is not None:
            try:
                DirectoryProfilesTab._profile_window.close()
            except Exception:
                pass
        # New profile has no prevalidations linked → no cache eviction needed.
        DirectoryProfilesTab._profile_window = DirectoryProfileWindow(
            self.window(),
            self._app_actions,
            self._on_profile_added,
        )
        DirectoryProfilesTab._profile_window.show()

    def _edit_profile(self) -> None:
        from ui.compare.directory_profile_window_qt import DirectoryProfileWindow

        idx = self._prof_listbox.currentRow()
        if idx < 0 or idx >= len(DirectoryProfile.directory_profiles):
            return
        if DirectoryProfilesTab._profile_window is not None:
            try:
                DirectoryProfilesTab._profile_window.close()
            except Exception:
                pass
        profile = DirectoryProfile.directory_profiles[idx]
        # Snapshot the profile's linked directories *before* the window edits
        # the profile object in-place.
        old_dirs = self._profile_linked_dirs(profile)

        def _on_edited(*_args) -> None:
            new_dirs = self._profile_linked_dirs(profile)
            self._invalidate_for_dir_sets(
                old_dirs, new_dirs,
                reason=f"profile '{profile.name}' edited",
            )
            self._refresh_prof_listbox()

        DirectoryProfilesTab._profile_window = DirectoryProfileWindow(
            self.window(),
            self._app_actions,
            _on_edited,
            profile,
        )
        DirectoryProfilesTab._profile_window.show()

    def _copy_profile(self) -> None:
        from ui.compare.directory_profile_window_qt import DirectoryProfileWindow

        idx = self._prof_listbox.currentRow()
        if idx < 0 or idx >= len(DirectoryProfile.directory_profiles):
            return
        if DirectoryProfilesTab._profile_window is not None:
            try:
                DirectoryProfilesTab._profile_window.close()
            except Exception:
                pass
        # Copied profile is new → no prevalidations linked → no eviction needed.
        DirectoryProfilesTab._profile_window = DirectoryProfileWindow(
            self.window(),
            self._app_actions,
            self._on_profile_added,
            copy_from_profile=DirectoryProfile.directory_profiles[idx],
        )
        DirectoryProfilesTab._profile_window.show()

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
        self._refresh_prof_listbox()

    def _on_profile_added(self, *_args) -> None:
        """Callback for profile add/copy — no eviction, just rebuild the list."""
        self._refresh_prof_listbox()

    # ------------------------------------------------------------------
    # Cache-eviction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_linked_dirs(profile: DirectoryProfile) -> set[str]:
        """
        Return the profile's directory set if at least one prevalidation
        currently references it, otherwise an empty set (nothing to evict).
        """
        used = any(
            pv.profile_name == profile.name or pv.profile is profile
            for pv in ClassifierActionsManager.prevalidations
        )
        return set(profile.directories) if used else set()

    @staticmethod
    def _invalidate_for_dir_sets(
        *dir_sets: "set[str] | None", reason: str = ""
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

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._refresh_prof_listbox()

"""
UI tests for ClassifierActionsTab — focused on profile-dropdown persistence.

Run with:
    pytest tests/ui/test_classifier_actions_tab_qt.py -v
"""

from __future__ import annotations

import pytest

from files.directory_profile import DirectoryProfile
from ui.compare.classifier_actions_tab_qt import ClassifierActionsTab, _PROFILE_CACHE_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeActions:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _add_profiles(names):
    for name in names:
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name=name, directories=[])
        )


def _make_tab(qtbot, actions=None) -> ClassifierActionsTab:
    tab = ClassifierActionsTab(None, actions or _FakeActions())
    qtbot.addWidget(tab)
    return tab


# ---------------------------------------------------------------------------
# Profile dropdown persistence
# ---------------------------------------------------------------------------

class TestProfileDropdownPersistence:
    def test_profile_combo_populated_from_profiles(self, qtbot, isolated_singletons):
        _add_profiles(["Alpha", "Beta"])
        tab = _make_tab(qtbot)
        items = [tab._profile_combo.itemText(i) for i in range(tab._profile_combo.count())]
        assert "Alpha" in items
        assert "Beta" in items

    def test_initial_selection_restored_from_cache(self, qtbot, isolated_singletons):
        _add_profiles(["X", "Y", "Z"])
        isolated_singletons.set_meta(_PROFILE_CACHE_KEY, "Y")
        tab = _make_tab(qtbot)
        assert tab._profile_combo.currentText() == "Y"

    def test_missing_cached_profile_falls_back_to_first(self, qtbot, isolated_singletons):
        _add_profiles(["P1", "P2"])
        isolated_singletons.set_meta(_PROFILE_CACHE_KEY, "DoesNotExist")
        tab = _make_tab(qtbot)
        assert tab._profile_combo.currentText() == "P1"

    def test_selection_change_updates_cache(self, qtbot, isolated_singletons):
        _add_profiles(["A", "B", "C"])
        tab = _make_tab(qtbot)
        tab._profile_combo.setCurrentText("C")
        assert isolated_singletons.get_meta(_PROFILE_CACHE_KEY) == "C"

    def test_empty_cache_does_not_change_selection(self, qtbot, isolated_singletons):
        _add_profiles(["First", "Second"])
        tab = _make_tab(qtbot)
        assert tab._profile_combo.currentIndex() == 0

    def test_no_profiles_combo_disabled(self, qtbot, isolated_singletons):
        tab = _make_tab(qtbot)
        assert not tab._profile_combo.isEnabled()

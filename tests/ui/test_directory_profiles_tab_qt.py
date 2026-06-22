"""
Tests for DirectoryProfilesTab.

Covers widget construction, list population, cache-eviction logic on
remove/edit, _profile_linked_dirs, _invalidate_for_dir_sets, and refresh.
DirectoryProfileWindow open/close paths are not exercised here.
"""

from __future__ import annotations

from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_action import Prevalidation
from files.directory_profile import DirectoryProfile
from ui.compare.directory_profiles_tab_qt import DirectoryProfilesTab

# Isolation (DirectoryProfile.directory_profiles, ClassifierActionsManager.prevalidations)
# is provided by the root conftest reset_app_globals autouse fixture.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeActions:
    pass


def _add_profile(name: str, dirs: list[str] | None = None) -> DirectoryProfile:
    p = DirectoryProfile(name=name, directories=dirs or ["/tmp/a"])
    DirectoryProfile.directory_profiles.append(p)
    return p


def _make_tab(qtbot) -> DirectoryProfilesTab:
    tab = DirectoryProfilesTab(None, _FakeActions())
    qtbot.addWidget(tab)
    return tab


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDirectoryProfilesTabConstruction:
    def test_builds_with_no_profiles(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab._prof_listbox.count() == 0

    def test_builds_with_populated_profiles(self, qtbot):
        _add_profile("p1")
        _add_profile("p2")
        tab = _make_tab(qtbot)
        assert tab._prof_listbox.count() == 2

    def test_listbox_shows_profile_name(self, qtbot):
        _add_profile("my_profile", dirs=["/a", "/b"])
        tab = _make_tab(qtbot)
        text = tab._prof_listbox.item(0).text()
        assert "my_profile" in text

    def test_listbox_shows_directory_count_singular(self, qtbot):
        _add_profile("single", dirs=["/only"])
        tab = _make_tab(qtbot)
        text = tab._prof_listbox.item(0).text()
        assert "1" in text  # count is locale-independent; translated word is not asserted

    def test_listbox_shows_directory_count_plural(self, qtbot):
        _add_profile("multi", dirs=["/a", "/b", "/c"])
        tab = _make_tab(qtbot)
        text = tab._prof_listbox.item(0).text()
        assert "3" in text  # count is locale-independent; translated word is not asserted


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

class TestDirectoryProfilesTabRefresh:
    def test_refresh_adds_new_entry(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab._prof_listbox.count() == 0
        _add_profile("late_profile")
        tab.refresh()
        assert tab._prof_listbox.count() == 1

    def test_refresh_removes_deleted_entry(self, qtbot):
        p = _add_profile("gone")
        tab = _make_tab(qtbot)
        assert tab._prof_listbox.count() == 1
        DirectoryProfile.directory_profiles.remove(p)
        tab.refresh()
        assert tab._prof_listbox.count() == 0


# ---------------------------------------------------------------------------
# _profile_linked_dirs
# ---------------------------------------------------------------------------

class TestProfileLinkedDirs:
    def test_returns_dirs_when_prevalidation_links_by_name(self):
        p = _add_profile("linked", dirs=["/x", "/y"])
        pv = Prevalidation(name="pv")
        pv.profile_name = "linked"
        ClassifierActionsManager.prevalidations.append(pv)
        result = DirectoryProfilesTab._profile_linked_dirs(p)
        assert result == {"/x", "/y"}

    def test_returns_dirs_when_prevalidation_links_by_instance(self):
        p = _add_profile("linked_inst", dirs=["/z"])
        pv = Prevalidation(name="pv2")
        pv.profile = p
        ClassifierActionsManager.prevalidations.append(pv)
        result = DirectoryProfilesTab._profile_linked_dirs(p)
        assert result == {"/z"}

    def test_returns_empty_set_when_no_prevalidation_references_profile(self):
        p = _add_profile("orphan", dirs=["/unused"])
        result = DirectoryProfilesTab._profile_linked_dirs(p)
        assert result == set()


# ---------------------------------------------------------------------------
# Remove profile — cache eviction
# ---------------------------------------------------------------------------

class TestRemoveProfile:
    def test_remove_unlinked_profile_does_not_clear_cache(self, qtbot, monkeypatch):
        _add_profile("orphan")
        tab = _make_tab(qtbot)
        tab._prof_listbox.setCurrentRow(0)

        cleared = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        tab._remove_profile()
        assert cleared == []
        assert len(DirectoryProfile.directory_profiles) == 0

    def test_remove_linked_profile_clears_full_cache(self, qtbot, monkeypatch):
        p = _add_profile("linked", dirs=["/d"])
        pv = Prevalidation(name="pv")
        pv.profile_name = "linked"
        ClassifierActionsManager.prevalidations.append(pv)

        tab = _make_tab(qtbot)
        tab._prof_listbox.setCurrentRow(0)

        cleared = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        tab._remove_profile()
        assert cleared == [1]

    def test_remove_with_no_selection_is_noop(self, qtbot):
        _add_profile("p")
        tab = _make_tab(qtbot)
        tab._prof_listbox.setCurrentRow(-1)
        tab._remove_profile()
        assert len(DirectoryProfile.directory_profiles) == 1

    def test_remove_updates_listbox(self, qtbot):
        _add_profile("gone")
        tab = _make_tab(qtbot)
        tab._prof_listbox.setCurrentRow(0)
        tab._remove_profile()
        assert tab._prof_listbox.count() == 0


# ---------------------------------------------------------------------------
# _invalidate_for_dir_sets
# ---------------------------------------------------------------------------

class TestInvalidateForDirSets:
    def test_none_in_dir_sets_triggers_full_eviction(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        DirectoryProfilesTab._invalidate_for_dir_sets(None, reason="test")
        assert cleared == [1]

    def test_empty_dir_sets_triggers_no_eviction(self, monkeypatch):
        cleared = []
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        monkeypatch.setattr(
            ClassifierActionsManager,
            "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        DirectoryProfilesTab._invalidate_for_dir_sets(set(), set(), reason="test")
        assert cleared == []
        assert targeted == []

    def test_non_empty_dir_sets_triggers_targeted_eviction(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        DirectoryProfilesTab._invalidate_for_dir_sets({"/a"}, {"/b"}, reason="test")
        assert targeted == [{"/a", "/b"}]

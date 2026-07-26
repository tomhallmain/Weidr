"""
Tests for DirectoryProfilesTab.

Covers widget construction, list population, cache-eviction logic on
remove/edit, _profile_linked_dirs, ClassifierActionsManager.invalidate_for_profile_edit,
and refresh. The real DirectoryProfileWindow dialog is not constructed;
TestEditProfileEvictionMessaging substitutes a fake to capture and invoke
the _on_edited callback directly.
"""

from __future__ import annotations

import pytest

from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_action import Prevalidation
from compare.classifier_pipeline import ClassifierPipelines, PrevalidationPipeline
from files.directory_profile import DirectoryProfile
from ui.compare.directory_profiles_tab_qt import DirectoryProfilesTab

# Isolation (DirectoryProfile.directory_profiles, ClassifierActionsManager.prevalidations)
# is provided by the root conftest reset_app_globals autouse fixture.


@pytest.fixture(autouse=True)
def _reset_profile_window():
    """DirectoryProfilesTab._profile_window is a class-level singleton
    reference, not reset by the root conftest -- clear it so a fake window
    substituted in one test can't leak into another."""
    DirectoryProfilesTab._profile_window = None
    yield
    DirectoryProfilesTab._profile_window = None


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

    def test_returns_dirs_when_pipeline_links_by_name(self):
        p = _add_profile("pipeline_linked", dirs=["/x", "/y"])
        pp = PrevalidationPipeline(name="pp")
        pp.profile_name = "pipeline_linked"
        ClassifierPipelines.add_pipeline(pp)
        result = DirectoryProfilesTab._profile_linked_dirs(p)
        assert result == {"/x", "/y"}

    def test_returns_dirs_when_pipeline_links_by_instance(self):
        p = _add_profile("pipeline_linked_inst", dirs=["/z"])
        pp = PrevalidationPipeline(name="pp2")
        pp.profile = p
        ClassifierPipelines.add_pipeline(pp)
        result = DirectoryProfilesTab._profile_linked_dirs(p)
        assert result == {"/z"}

    def test_profile_used_only_by_action_pipeline_is_not_linked(self):
        """A plain (non-prevalidation) ClassifierPipeline never gates on a
        profile, so it must not count as usage."""
        from compare.classifier_pipeline import ClassifierPipeline

        p = _add_profile("action_only", dirs=["/w"])
        cp = ClassifierPipeline(name="cp")
        ClassifierPipelines.add_pipeline(cp)
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
# _edit_profile / _on_edited -- eviction messaging
# ---------------------------------------------------------------------------

class _FakeProfileWindow:
    """Stand-in for DirectoryProfileWindow that just captures the
    refresh_callback (_on_edited) so tests can invoke it directly without
    constructing the real dialog."""

    captured_callback = None

    def __init__(self, parent, app_actions, callback, profile=None, **kwargs):
        _FakeProfileWindow.captured_callback = callback

    def show(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestEditProfileEvictionMessaging:
    def test_unused_profile_edit_skips_dirs_diff_entirely(self, qtbot, monkeypatch):
        """Regression test: a profile edit with nothing referencing it must
        not go through the symmetric-difference dirs-diff at all -- doing so
        previously produced a misleading "directory scope unchanged" log even
        when directories were actually added/removed, since
        _profile_linked_dirs returns an empty set for both snapshots
        whenever nothing is linked."""
        import ui.compare.directory_profile_window_qt as dpw_module

        p = _add_profile("unused_edit", dirs=["/a", "/b"])
        monkeypatch.setattr(dpw_module, "DirectoryProfileWindow", _FakeProfileWindow)
        tab = _make_tab(qtbot)
        tab._prof_listbox.setCurrentRow(0)

        called = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_profile_edit",
            lambda *a, **kw: called.append(1),
        )

        tab._edit_profile()
        assert _FakeProfileWindow.captured_callback is not None
        # Simulate the directory-removal edit, then the window's completion callback.
        p.directories = ["/a"]
        _FakeProfileWindow.captured_callback()

        assert called == []

    def test_linked_profile_edit_uses_dirs_diff(self, qtbot, monkeypatch):
        import ui.compare.directory_profile_window_qt as dpw_module

        p = _add_profile("linked_edit", dirs=["/a", "/b"])
        pv = Prevalidation(name="pv")
        pv.profile_name = "linked_edit"
        ClassifierActionsManager.prevalidations.append(pv)
        monkeypatch.setattr(dpw_module, "DirectoryProfileWindow", _FakeProfileWindow)
        tab = _make_tab(qtbot)
        tab._prof_listbox.setCurrentRow(0)

        diffed = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_profile_edit",
            lambda old, new, **kw: diffed.append((old, new)),
        )

        tab._edit_profile()
        p.directories = ["/a"]
        _FakeProfileWindow.captured_callback()

        assert diffed == [({"/a", "/b"}, {"/a"})]


# ---------------------------------------------------------------------------
# ClassifierActionsManager.invalidate_for_profile_edit
# ---------------------------------------------------------------------------

class TestInvalidateForProfileEdit:
    def test_old_none_triggers_full_eviction(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        ClassifierActionsManager.invalidate_for_profile_edit(None, {"/a"}, reason="test")
        assert cleared == [1]

    def test_new_none_triggers_full_eviction(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        ClassifierActionsManager.invalidate_for_profile_edit({"/a"}, None, reason="test")
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
        ClassifierActionsManager.invalidate_for_profile_edit(set(), set(), reason="test")
        assert cleared == []
        assert targeted == []

    def test_disjoint_dir_sets_triggers_targeted_eviction_of_both(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        ClassifierActionsManager.invalidate_for_profile_edit({"/a"}, {"/b"}, reason="test")
        assert targeted == [{"/a", "/b"}]

    def test_directory_added_to_profile_evicts_only_the_added_directory(self, monkeypatch):
        """Regression test for the reported bug: adding one directory to a
        profile with N existing directories must evict only the new one, not
        all N+1 -- i.e. the symmetric difference, not the union."""
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        old_dirs = {f"/dir{i}" for i in range(18)}
        new_dirs = old_dirs | {"/dir18"}
        ClassifierActionsManager.invalidate_for_profile_edit(old_dirs, new_dirs, reason="test")
        assert targeted == [{"/dir18"}]

    def test_directory_removed_from_profile_evicts_only_the_removed_directory(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        old_dirs = {"/a", "/b", "/c"}
        new_dirs = {"/a", "/b"}
        ClassifierActionsManager.invalidate_for_profile_edit(old_dirs, new_dirs, reason="test")
        assert targeted == [{"/c"}]

    def test_unchanged_dir_sets_trigger_no_eviction(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager,
            "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        same = {"/a", "/b"}
        ClassifierActionsManager.invalidate_for_profile_edit(same, set(same), reason="test")
        assert targeted == []

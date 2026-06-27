"""
Tests for ClassifierManagementWindow.

Covers tab layout (count and labels), last-selected-tab persistence via
app_info_cache, and the disabled-tab behaviour when prevalidations are off.

Tab indices expected:
  0  Classifier Actions
  1  Pipelines
  2  Prevalidations
  3  Seek to Trigger
  4  Lookaheads
  5  Directory Profiles
"""

from __future__ import annotations

from utils.config import config

# Isolation (ClassifierManagementWindow._instance, app_info_cache, Lookahead.lookaheads,
# DirectoryProfile.directory_profiles, ClassifierActionsManager.prevalidations) is
# provided by the root conftest isolated_singletons and reset_app_globals fixtures.

_CACHE_KEY = "classifier_management_tab"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeActions:
    """Returns None for every app_actions call made during tab construction."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _make_window(qtbot):
    from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
    win = ClassifierManagementWindow(None, _FakeActions())
    qtbot.addWidget(win)
    return win


# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

class TestTabLayout:
    def test_six_tabs(self, qtbot):
        win = _make_window(qtbot)
        assert win._tabs.count() == 6

    def test_tab_labels(self, qtbot):
        from utils.translations import _
        win = _make_window(qtbot)
        labels = [win._tabs.tabText(i) for i in range(win._tabs.count())]
        assert labels[0] == _("Classifier Actions")
        assert labels[1] == _("Pipelines")
        assert labels[2] == _("Prevalidations")
        assert labels[3] == _("Seek to Trigger")
        assert labels[4] == _("Lookaheads")
        assert labels[5] == _("Directory Profiles")

    def test_classifier_actions_tab_is_enabled(self, qtbot):
        win = _make_window(qtbot)
        assert win._tabs.isTabEnabled(0)

    def test_pipelines_tab_is_enabled(self, qtbot):
        win = _make_window(qtbot)
        assert win._tabs.isTabEnabled(1)

    def test_seek_to_trigger_tab_is_enabled(self, qtbot):
        win = _make_window(qtbot)
        assert win._tabs.isTabEnabled(3)


# ---------------------------------------------------------------------------
# Disabled-tab behaviour when prevalidations are off
# ---------------------------------------------------------------------------

class TestDisabledTabs:
    def test_prevalidations_off_disables_prevalidations_tab(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "enable_prevalidations", False)
        win = _make_window(qtbot)
        assert not win._tabs.isTabEnabled(2), "Prevalidations tab should be disabled"

    def test_prevalidations_off_leaves_all_other_tabs_enabled(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "enable_prevalidations", False)
        win = _make_window(qtbot)
        for i in (0, 1, 3, 4, 5):
            assert win._tabs.isTabEnabled(i), f"Tab {i} should be enabled"

    def test_prevalidations_on_all_tabs_enabled(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "enable_prevalidations", True)
        win = _make_window(qtbot)
        for i in range(win._tabs.count()):
            assert win._tabs.isTabEnabled(i), f"Tab {i} should be enabled"


# ---------------------------------------------------------------------------
# Tab persistence
# ---------------------------------------------------------------------------

class TestTabPersistence:
    def test_defaults_to_tab_0_when_no_cached_value(self, qtbot):
        win = _make_window(qtbot)
        assert win._tabs.currentIndex() == 0

    def test_restores_saved_tab_index(self, qtbot):
        import ui.compare.classifier_management_window_qt as _cmw
        _cmw.app_info_cache.set_meta(_CACHE_KEY, 4)
        win = _make_window(qtbot)
        assert win._tabs.currentIndex() == 4

    def test_ignores_out_of_range_cached_index(self, qtbot):
        import ui.compare.classifier_management_window_qt as _cmw
        _cmw.app_info_cache.set_meta(_CACHE_KEY, 99)
        win = _make_window(qtbot)
        assert win._tabs.currentIndex() == 0

    def test_ignores_non_integer_cached_value(self, qtbot):
        import ui.compare.classifier_management_window_qt as _cmw
        _cmw.app_info_cache.set_meta(_CACHE_KEY, "bad")
        win = _make_window(qtbot)
        assert win._tabs.currentIndex() == 0

    def test_tab_change_writes_to_cache(self, qtbot):
        import ui.compare.classifier_management_window_qt as _cmw
        win = _make_window(qtbot)
        win._tabs.setCurrentIndex(4)
        assert _cmw.app_info_cache.get_meta(_CACHE_KEY) == 4

    def test_tab_change_updates_cache_on_each_switch(self, qtbot):
        import ui.compare.classifier_management_window_qt as _cmw
        win = _make_window(qtbot)
        win._tabs.setCurrentIndex(4)
        win._tabs.setCurrentIndex(0)
        assert _cmw.app_info_cache.get_meta(_CACHE_KEY) == 0

    def test_new_window_picks_up_cache_written_by_previous(self, qtbot):
        import ui.compare.classifier_management_window_qt as _cmw
        win1 = _make_window(qtbot)
        win1.close()
        _cmw.app_info_cache.set_meta(_CACHE_KEY, 5)
        win2 = _make_window(qtbot)
        assert win2._tabs.currentIndex() == 5

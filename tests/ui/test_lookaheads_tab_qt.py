"""
Tests for LookaheadsTab.

Covers widget construction, list population, remove logic, and public refresh.
LookaheadWindow open/close paths are not exercised here (they require the full
app window stack); those are integration-test territory.
"""

from __future__ import annotations

from compare.classifier_actions_manager import ClassifierActionsManager
from compare.lookahead import Lookahead
from ui.compare.lookaheads_tab_qt import LookaheadsTab

# Isolation (Lookahead.lookaheads, ClassifierActionsManager.prevalidations) is
# provided by the root conftest reset_app_globals autouse fixture.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeActions:
    pass


def _make_lookahead(name: str, name_or_text: str = "apple", threshold: float = 0.5) -> Lookahead:
    lh = Lookahead(name=name, name_or_text=name_or_text, threshold=threshold)
    Lookahead.lookaheads.append(lh)
    return lh


def _make_tab(qtbot) -> LookaheadsTab:
    tab = LookaheadsTab(None, _FakeActions())
    qtbot.addWidget(tab)
    return tab


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLookaheadsTabConstruction:
    def test_builds_with_no_lookaheads(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab._lh_listbox.count() == 0

    def test_builds_with_populated_lookaheads(self, qtbot):
        _make_lookahead("la1")
        _make_lookahead("la2")
        tab = _make_tab(qtbot)
        assert tab._lh_listbox.count() == 2

    def test_listbox_item_includes_name_and_threshold(self, qtbot):
        _make_lookahead("my_lookahead", name_or_text="cat", threshold=0.75)
        tab = _make_tab(qtbot)
        item_text = tab._lh_listbox.item(0).text()
        assert "my_lookahead" in item_text
        assert "0.75" in item_text

    def test_listbox_item_includes_name_or_text(self, qtbot):
        _make_lookahead("la", name_or_text="fruit")
        tab = _make_tab(qtbot)
        item_text = tab._lh_listbox.item(0).text()
        assert "fruit" in item_text


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

class TestLookaheadsTabRefresh:
    def test_refresh_adds_new_entry(self, qtbot):
        tab = _make_tab(qtbot)
        assert tab._lh_listbox.count() == 0
        _make_lookahead("new_la")
        tab.refresh()
        assert tab._lh_listbox.count() == 1

    def test_refresh_removes_deleted_entry(self, qtbot):
        lh = _make_lookahead("to_remove")
        tab = _make_tab(qtbot)
        assert tab._lh_listbox.count() == 1
        Lookahead.lookaheads.remove(lh)
        tab.refresh()
        assert tab._lh_listbox.count() == 0


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestLookaheadsTabRemove:
    def test_remove_selected_lookahead(self, qtbot):
        _make_lookahead("la_a")
        _make_lookahead("la_b")
        tab = _make_tab(qtbot)
        tab._lh_listbox.setCurrentRow(0)
        tab._remove_lookahead()
        assert len(Lookahead.lookaheads) == 1
        assert Lookahead.lookaheads[0].name == "la_b"
        assert tab._lh_listbox.count() == 1

    def test_remove_with_no_selection_is_noop(self, qtbot):
        _make_lookahead("la")
        tab = _make_tab(qtbot)
        tab._lh_listbox.setCurrentRow(-1)
        tab._remove_lookahead()
        assert len(Lookahead.lookaheads) == 1

    def test_remove_warns_when_used_by_prevalidation(self, qtbot, caplog, monkeypatch):
        import logging
        import ui.compare.lookaheads_tab_qt as _lhm
        from compare.classifier_action import Prevalidation

        lh = _make_lookahead("shared_la")
        pv = Prevalidation(name="pv_uses_la")
        pv.lookahead_names = ["shared_la"]
        ClassifierActionsManager.prevalidations.append(pv)

        tab = _make_tab(qtbot)
        tab._lh_listbox.setCurrentRow(0)
        # weidr.* loggers set propagate=False; enable it so caplog captures the record.
        monkeypatch.setattr(_lhm.logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="weidr.lookaheads_tab_qt"):
            tab._remove_lookahead()
        assert "shared_la" in caplog.text
        # Lookahead is still removed despite the warning
        assert lh not in Lookahead.lookaheads

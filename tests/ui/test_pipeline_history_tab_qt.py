"""Tests for PipelineHistoryTab: rows from the run history, which buttons are
enabled for the selected run, and the stop / resume / remove actions.

The history is the in-memory process singleton, reset per test by the root
conftest; qt_alert is patched so no dialog is shown.
"""

from __future__ import annotations

import pytest

from compare import pipeline_run_service
from compare.pipeline_run_control import RunCheckpoint
from compare.pipeline_run_history import (
    PipelineRunRecord,
    RunOwner,
    RunStatus,
    pipeline_run_history,
)


class _FakeActions:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def alerts(monkeypatch):
    """Captures the tab's qt_alert calls; set alerts.answer for confirmations."""
    import ui.compare.pipeline_history_tab_qt as _mod

    class _Alerts(list):
        answer = True

    shown = _Alerts()

    def fake(_parent, _title, msg, kind="info", **kw):
        shown.append((msg, kind))
        return shown.answer

    monkeypatch.setattr(_mod, "qt_alert", fake)
    return shown


def _seed(name="P", dirs=("/data/a",), status=RunStatus.COMPLETED, started=None, owner=None):
    record = PipelineRunRecord.new(name, list(dirs), profile_name="Prof", source="gui",
                                   owner=owner or pipeline_run_history.owner)
    if started is not None:
        record.started_at = started
    pipeline_run_history.add_run(record)
    if status is not RunStatus.RUNNING:
        pipeline_run_history.finish_run(record.run_id, status, counters=RunCheckpoint(files_evaluated=2))
    return record.run_id


def _make_tab(qtbot):
    from ui.compare.pipeline_history_tab_qt import PipelineHistoryTab
    tab = PipelineHistoryTab(None, _FakeActions())
    qtbot.addWidget(tab)
    return tab


def _select(tab, run_id):
    tab._select_run(run_id)
    assert tab.selected_run_id() == run_id


def test_rows_newest_first_with_status_labels(qtbot):
    from utils.translations import _
    old = _seed("Old", started=1.0)
    new = _seed("New", status=RunStatus.CANCELLED, started=2.0)
    tab = _make_tab(qtbot)
    assert tab._table.rowCount() == 2
    assert tab._table.item(0, 1).text() == "New"
    assert tab._table.item(0, 0).text() == _("Cancelled")
    assert tab._table.item(1, 0).text() == _("Completed")
    assert tab._records[1].run_id == old and tab._records[0].run_id == new


def test_buttons_follow_the_selected_run(qtbot):
    active = pipeline_run_service.begin_run(
        type("P", (), {"name": "Live", "to_dict": lambda self: {}})(), ["/live"],
        profile_name=None, source="gui")
    done = _seed("Done", started=1.0)
    resumable = _seed("Half", status=RunStatus.INTERRUPTED, started=2.0)
    tab = _make_tab(qtbot)

    _select(tab, active.run_id)
    assert tab._stop_btn.isEnabled()
    assert not tab._resume_btn.isEnabled()
    assert not tab._remove_btn.isEnabled()

    _select(tab, done)
    assert not tab._stop_btn.isEnabled()
    assert not tab._resume_btn.isEnabled()
    assert tab._remove_btn.isEnabled()

    _select(tab, resumable)
    assert tab._resume_btn.isEnabled()
    assert tab._resume_last_btn.isEnabled()


def test_other_process_running_run_cannot_be_stopped_here(qtbot):
    import os
    from utils.translations import _
    foreign = _seed("Foreign", status=RunStatus.RUNNING,
                    owner=RunOwner(token="other", pid=os.getpid(), create_time=None))
    tab = _make_tab(qtbot)
    _select(tab, foreign)
    assert not tab._stop_btn.isEnabled()
    assert tab._table.item(0, 7).text() == _("Other process")


def test_stop_asks_then_requests_stop(qtbot, alerts):
    from utils.translations import _
    active = pipeline_run_service.begin_run(
        type("P", (), {"name": "Live", "to_dict": lambda self: {}})(), ["/live"],
        profile_name=None, source="gui")
    tab = _make_tab(qtbot)
    _select(tab, active.run_id)
    tab._stop_selected()
    assert alerts[0][1] == "askyesno"
    assert active.control.is_cancelled()
    assert tab._table.item(0, 0).text() == _("Stopping…")
    assert not tab._stop_btn.isEnabled()


def test_stop_declined_keeps_running(qtbot, alerts):
    alerts.answer = False
    active = pipeline_run_service.begin_run(
        type("P", (), {"name": "Live", "to_dict": lambda self: {}})(), ["/live"],
        profile_name=None, source="gui")
    tab = _make_tab(qtbot)
    _select(tab, active.run_id)
    tab._stop_selected()
    assert not active.control.is_cancelled()


def test_resume_goes_through_the_pipelines_tab(qtbot, monkeypatch):
    from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
    run_id = _seed("Half", status=RunStatus.CANCELLED)
    resumed = []
    monkeypatch.setattr(ClassifierPipelinesTab, "resume_run",
                        staticmethod(lambda parent, actions, rid: resumed.append(rid)))
    tab = _make_tab(qtbot)
    _select(tab, run_id)
    tab._resume_selected()
    tab._resume_last_incomplete()
    assert resumed == [run_id, run_id]


def test_resume_last_incomplete_without_any(qtbot, alerts):
    _seed("Done")
    tab = _make_tab(qtbot)
    tab._resume_last_incomplete()
    assert len(alerts) == 1


def test_remove_after_confirmation(qtbot, alerts):
    run_id = _seed("Done")
    tab = _make_tab(qtbot)
    _select(tab, run_id)
    tab._remove_selected()
    assert pipeline_run_history.get(run_id) is None
    assert tab._table.rowCount() == 0


def test_selection_is_kept_across_refresh(qtbot):
    a = _seed("A", started=1.0)
    _seed("B", started=2.0)
    tab = _make_tab(qtbot)
    _select(tab, a)
    _seed("C", started=3.0)
    tab.refresh()
    assert tab._table.rowCount() == 3
    assert tab.selected_run_id() == a


def test_details_show_the_directory_snapshot(qtbot):
    run_id = _seed("Snap", dirs=("/snap/one", "/snap/two"))
    tab = _make_tab(qtbot)
    _select(tab, run_id)
    assert "/snap/one" in tab._details.text()
    assert "/snap/two" in tab._details.text()

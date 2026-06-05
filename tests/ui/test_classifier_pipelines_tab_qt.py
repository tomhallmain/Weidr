"""
UI tests for ClassifierPipelinesTab (Phase 4a).

These tests cover what the unit tests in test_classifier_pipelines_manager.py
cannot: widget construction, row rendering, and the tab methods that mutate
the manager and rebuild the UI.

qt_alert calls are monkeypatched to a no-op so QMessageBox.exec() never
blocks the test runner.

Run with:
    pytest tests/ui/test_classifier_pipelines_tab_qt.py -v
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QCheckBox, QWidget

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    EmbeddingCondition,
    NodeOutcome,
    OutcomeType,
    PipelineNode,
    PrevalidationPipeline,
)
from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab


# ---------------------------------------------------------------------------
# File-level isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_editor_window():
    """Ensure the class-level editor window reference is cleared before/after
    each test regardless of whether reset_app_globals manages to import the
    module (it uses try/except, so this is belt-and-suspenders)."""
    ClassifierPipelinesTab._editor_window = None
    yield
    ClassifierPipelinesTab._editor_window = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeActions:
    current_media_path = None


def _make_pipeline(name: str, is_active: bool = True) -> ClassifierPipeline:
    p = ClassifierPipeline(name=name, is_active=is_active)
    cond = EmbeddingCondition(positives=["cat"], negatives=[], threshold=0.3)
    node = PipelineNode(
        name="n1",
        condition=cond,
        on_match=NodeOutcome(OutcomeType.ACCEPT),
        on_no_match=NodeOutcome(OutcomeType.REJECT),
    )
    p.nodes = [node]
    return p


def _make_tab(qtbot, actions=None) -> ClassifierPipelinesTab:
    parent = QWidget()
    qtbot.addWidget(parent)
    tab = ClassifierPipelinesTab(parent, actions or _FakeActions())
    qtbot.addWidget(tab)
    return tab


def _row_count(tab: ClassifierPipelinesTab) -> int:
    """Pipeline rows only (excludes the header layout and the trailing stretch)."""
    return tab._scroll_layout.count() - 2


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTabConstruction:
    def test_tab_constructs_with_no_pipelines(self, qtbot, isolated_singletons):
        tab = _make_tab(qtbot)
        assert tab is not None

    def test_tab_constructs_with_pipelines(self, qtbot, isolated_singletons):
        ClassifierPipelines.add_pipeline(_make_pipeline("a"))
        ClassifierPipelines.add_pipeline(_make_pipeline("b"))
        tab = _make_tab(qtbot)
        assert _row_count(tab) == 2

    def test_empty_tab_has_header_and_stretch_only(self, qtbot, isolated_singletons):
        tab = _make_tab(qtbot)
        # layout: header (1) + stretch (1) = 2
        assert tab._scroll_layout.count() == 2


# ---------------------------------------------------------------------------
# Row rendering
# ---------------------------------------------------------------------------

class TestRowRendering:
    def test_row_count_matches_pipeline_count(self, qtbot, isolated_singletons):
        for i in range(3):
            ClassifierPipelines.add_pipeline(_make_pipeline(f"pipe_{i}"))
        tab = _make_tab(qtbot)
        assert _row_count(tab) == 3

    def test_active_checkbox_reflects_pipeline_state(self, qtbot, isolated_singletons):
        ClassifierPipelines.add_pipeline(_make_pipeline("active_one", is_active=True))
        ClassifierPipelines.add_pipeline(_make_pipeline("inactive_one", is_active=False))
        tab = _make_tab(qtbot)
        # First pipeline row is layout index 1 (index 0 is header)
        first_row_layout = tab._scroll_layout.itemAt(1).layout()
        first_checkbox = None
        for i in range(first_row_layout.count()):
            w = first_row_layout.itemAt(i).widget()
            if isinstance(w, QCheckBox):
                first_checkbox = w
                break
        assert first_checkbox is not None
        assert first_checkbox.isChecked() is True

    def test_prevalidation_type_label_shown(self, qtbot, isolated_singletons):
        from PySide6.QtWidgets import QLabel
        from utils.translations import _
        pv = PrevalidationPipeline(profile_name="portraits")
        pv.name = "pv_pipe"
        ClassifierPipelines.add_pipeline(pv)
        tab = _make_tab(qtbot)
        # Collect all label texts in the first pipeline row
        row_layout = tab._scroll_layout.itemAt(1).layout()
        labels = []
        for i in range(row_layout.count()):
            w = row_layout.itemAt(i).widget()
            if isinstance(w, QLabel):
                labels.append(w.text())
        assert _("Prevalidation") in labels

    def test_general_type_label_shown(self, qtbot, isolated_singletons):
        from PySide6.QtWidgets import QLabel
        from utils.translations import _
        ClassifierPipelines.add_pipeline(_make_pipeline("gen_pipe"))
        tab = _make_tab(qtbot)
        row_layout = tab._scroll_layout.itemAt(1).layout()
        labels = [
            row_layout.itemAt(i).widget().text()
            for i in range(row_layout.count())
            if isinstance(row_layout.itemAt(i).widget(), QLabel)
        ]
        assert _("General") in labels


# ---------------------------------------------------------------------------
# refresh()
# ---------------------------------------------------------------------------

class TestRefresh:
    def test_refresh_adds_new_row(self, qtbot, isolated_singletons):
        tab = _make_tab(qtbot)
        assert _row_count(tab) == 0
        ClassifierPipelines.add_pipeline(_make_pipeline("new_one"))
        tab.refresh()
        assert _row_count(tab) == 1

    def test_refresh_removes_deleted_row(self, qtbot, isolated_singletons):
        p = _make_pipeline("removable")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        assert _row_count(tab) == 1
        ClassifierPipelines.remove_pipeline("removable")
        tab.refresh()
        assert _row_count(tab) == 0

    def test_refresh_reloads_from_cache(self, qtbot, isolated_singletons):
        p = _make_pipeline("cached")
        ClassifierPipelines.add_pipeline(p)
        ClassifierPipelines.store()
        # Clear in-memory list, then refresh should reload from cache
        ClassifierPipelines.pipelines = []
        tab = _make_tab(qtbot)
        assert _row_count(tab) == 0
        tab.refresh()
        assert _row_count(tab) == 1


# ---------------------------------------------------------------------------
# _toggle_active
# ---------------------------------------------------------------------------

class TestToggleActive:
    def test_toggle_active_sets_attribute(self, qtbot, isolated_singletons):
        p = _make_pipeline("toggle_me", is_active=True)
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._toggle_active(p, False)
        assert p.is_active is False

    def test_toggle_active_stores_to_cache(self, qtbot, isolated_singletons):
        p = _make_pipeline("store_test", is_active=True)
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._toggle_active(p, False)
        # Reload from cache and verify
        ClassifierPipelines.load()
        reloaded = ClassifierPipelines.get_pipeline_by_name("store_test")
        assert reloaded is not None
        assert reloaded.is_active is False

    def test_toggle_active_true(self, qtbot, isolated_singletons):
        p = _make_pipeline("activate_me", is_active=False)
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._toggle_active(p, True)
        assert p.is_active is True


# ---------------------------------------------------------------------------
# _duplicate
# ---------------------------------------------------------------------------

class TestDuplicate:
    def test_duplicate_adds_pipeline(self, qtbot, isolated_singletons):
        p = _make_pipeline("original")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._duplicate(p)
        assert len(ClassifierPipelines.get_all_pipelines()) == 2

    def test_duplicate_name_contains_copy(self, qtbot, isolated_singletons):
        p = _make_pipeline("source")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._duplicate(p)
        names = [pip.name for pip in ClassifierPipelines.get_all_pipelines()]
        copies = [n for n in names if n != "source"]
        assert len(copies) == 1
        assert "copy" in copies[0].lower() or "source" in copies[0]

    def test_duplicate_avoids_name_collision(self, qtbot, isolated_singletons):
        p = _make_pipeline("base")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._duplicate(p)
        tab._duplicate(p)
        names = [pip.name for pip in ClassifierPipelines.get_all_pipelines()]
        # All names must be unique
        assert len(names) == len(set(names))

    def test_duplicate_rebuilds_rows(self, qtbot, isolated_singletons):
        p = _make_pipeline("rebuild_test")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        assert _row_count(tab) == 1
        tab._duplicate(p)
        assert _row_count(tab) == 2

    def test_duplicate_stores_to_cache(self, qtbot, isolated_singletons):
        p = _make_pipeline("dup_store")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._duplicate(p)
        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()
        assert len(ClassifierPipelines.get_all_pipelines()) == 2


# ---------------------------------------------------------------------------
# _delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_removes_pipeline(self, qtbot, isolated_singletons):
        p = _make_pipeline("to_delete")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._delete(p)
        assert ClassifierPipelines.get_pipeline_by_name("to_delete") is None

    def test_delete_rebuilds_rows(self, qtbot, isolated_singletons):
        p = _make_pipeline("rebuild_del")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        assert _row_count(tab) == 1
        tab._delete(p)
        assert _row_count(tab) == 0

    def test_delete_stores_to_cache(self, qtbot, isolated_singletons):
        p = _make_pipeline("del_store")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._delete(p)
        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()
        assert ClassifierPipelines.get_pipeline_by_name("del_store") is None

    def test_delete_one_of_several(self, qtbot, isolated_singletons):
        a = _make_pipeline("keep_a")
        b = _make_pipeline("delete_b")
        c = _make_pipeline("keep_c")
        for pip in (a, b, c):
            ClassifierPipelines.add_pipeline(pip)
        tab = _make_tab(qtbot)
        tab._delete(b)
        remaining = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert remaining == ["keep_a", "keep_c"]


# ---------------------------------------------------------------------------
# _move_down
# ---------------------------------------------------------------------------

class TestMoveDown:
    def test_move_down_reorders_pipelines(self, qtbot, isolated_singletons):
        a = _make_pipeline("alpha")
        b = _make_pipeline("beta")
        ClassifierPipelines.add_pipeline(a)
        ClassifierPipelines.add_pipeline(b)
        tab = _make_tab(qtbot)
        tab._move_down(0, a)
        names = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert names == ["beta", "alpha"]

    def test_move_down_last_item_is_noop(self, qtbot, isolated_singletons):
        a = _make_pipeline("first")
        b = _make_pipeline("last")
        ClassifierPipelines.add_pipeline(a)
        ClassifierPipelines.add_pipeline(b)
        tab = _make_tab(qtbot)
        tab._move_down(1, b)
        names = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert names == ["first", "last"]

    def test_move_down_rebuilds_rows(self, qtbot, isolated_singletons):
        a = _make_pipeline("m1")
        b = _make_pipeline("m2")
        ClassifierPipelines.add_pipeline(a)
        ClassifierPipelines.add_pipeline(b)
        tab = _make_tab(qtbot)
        tab._move_down(0, a)
        assert _row_count(tab) == 2

    def test_move_down_stores_to_cache(self, qtbot, isolated_singletons):
        a = _make_pipeline("mv_a")
        b = _make_pipeline("mv_b")
        ClassifierPipelines.add_pipeline(a)
        ClassifierPipelines.add_pipeline(b)
        tab = _make_tab(qtbot)
        tab._move_down(0, a)
        ClassifierPipelines.pipelines = []
        ClassifierPipelines.load()
        names = [p.name for p in ClassifierPipelines.get_all_pipelines()]
        assert names == ["mv_b", "mv_a"]


# ---------------------------------------------------------------------------
# _run_on_current
# ---------------------------------------------------------------------------

class TestRunOnCurrent:
    def test_run_with_no_path_shows_alert(self, qtbot, monkeypatch, isolated_singletons):
        alerted = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.qt_alert",
            lambda *a, **kw: alerted.append(a),
        )
        p = _make_pipeline("run_test")
        ClassifierPipelines.add_pipeline(p)
        tab = _make_tab(qtbot)
        tab._run_on_current(p)
        assert len(alerted) == 1

    def test_run_with_callable_path_returning_none_shows_alert(
        self, qtbot, monkeypatch, isolated_singletons
    ):
        alerted = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.qt_alert",
            lambda *a, **kw: alerted.append(a),
        )

        class _Actions:
            def current_media_path(self):  # callable returning None
                return None

        p = _make_pipeline("callable_none")
        ClassifierPipelines.add_pipeline(p)
        parent = QWidget()
        qtbot.addWidget(parent)
        tab = ClassifierPipelinesTab(parent, _Actions())
        qtbot.addWidget(tab)
        tab._run_on_current(p)
        assert len(alerted) == 1

    def test_run_with_valid_path_calls_run_pipeline(
        self, qtbot, monkeypatch, isolated_singletons
    ):
        import compare.classifier_pipeline_runner as runner_mod

        calls = []

        def _fake_run(pipeline, image_path, **kwargs):
            calls.append((pipeline.name, image_path))
            return None

        monkeypatch.setattr(runner_mod, "run_pipeline", _fake_run)
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.qt_alert",
            lambda *a, **kw: None,
        )

        class _Actions:
            current_media_path = "/fake/image.jpg"

        p = _make_pipeline("run_with_path")
        ClassifierPipelines.add_pipeline(p)
        parent = QWidget()
        qtbot.addWidget(parent)
        tab = ClassifierPipelinesTab(parent, _Actions())
        qtbot.addWidget(tab)
        tab._run_on_current(p)
        assert len(calls) == 1
        assert calls[0] == ("run_with_path", "/fake/image.jpg")

    def test_run_pipeline_result_shown_in_alert(
        self, qtbot, monkeypatch, isolated_singletons
    ):
        import compare.classifier_pipeline_runner as runner_mod
        from utils.constants import ClassifierActionType

        monkeypatch.setattr(runner_mod, "run_pipeline",
                            lambda *a, **kw: ClassifierActionType.NOTIFY)
        alerts = []
        monkeypatch.setattr(
            "ui.compare.classifier_pipelines_tab_qt.qt_alert",
            lambda *a, **kw: alerts.append(a),
        )

        class _Actions:
            current_media_path = "/img.png"

        p = _make_pipeline("result_test")
        ClassifierPipelines.add_pipeline(p)
        parent = QWidget()
        qtbot.addWidget(parent)
        tab = ClassifierPipelinesTab(parent, _Actions())
        qtbot.addWidget(tab)
        tab._run_on_current(p)
        assert len(alerts) == 1
        # The result string should appear somewhere in the alert message
        assert "NOTIFY" in str(alerts[0])

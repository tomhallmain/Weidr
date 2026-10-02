"""UI tests for ClassifierFilterDialog and the filter panel's classifier row."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from compare.compare_filters import (
    SELECTION_MODEL_STRATEGY,
    ClassifierFilter,
    CompareFilterGroup,
    FilterOperator,
    SizeFilter,
)
from ui.compare import classifier_filter_dialog_qt, filter_builder_panel_qt
from ui.compare.classifier_filter_dialog_qt import (
    ClassifierFilterDialog,
    describe_classifier_filter,
)
from ui.compare.filter_builder_panel_qt import FilterBuilderPanel
from utils.translations import _


_IMAGE_MODELS = [
    SimpleNamespace(model_name="content", model_categories=["photo", "drawing", "nsfw"],
                    positive_groups=[["nsfw"]]),
    SimpleNamespace(model_name="plain", model_categories=["a", "b"], positive_groups=[]),
]
_AUDIO_MODELS = [
    SimpleNamespace(model_name="speech", model_categories=["speech", "music"], positive_groups=[]),
]


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch):
    monkeypatch.setattr(
        classifier_filter_dialog_qt, "_model_configs",
        lambda domain: _AUDIO_MODELS if domain == "audio" else _IMAGE_MODELS,
    )


@pytest.fixture
def parent(qtbot):
    w = QWidget()
    qtbot.addWidget(w)
    return w


def _checked(dlg):
    return dlg._checked_categories()


def _check(dlg, category):
    for i in range(dlg._category_list.count()):
        item = dlg._category_list.item(i)
        if item.text() == category:
            item.setCheckState(Qt.CheckState.Checked)


class TestClassifierFilterDialog:
    def test_new_filter_ok_disabled_until_category_checked(self, parent, qtbot):
        dlg = ClassifierFilterDialog(parent)
        qtbot.addWidget(dlg)
        assert dlg._model_combo.currentData() == "content"
        assert not dlg._ok_btn.isEnabled()
        _check(dlg, "photo")
        assert dlg._ok_btn.isEnabled()

    def test_ok_returns_edited_filter(self, parent, qtbot):
        dlg = ClassifierFilterDialog(parent)
        qtbot.addWidget(dlg)
        _check(dlg, "drawing")
        dlg._mode_combo.setCurrentIndex(dlg._mode_combo.findData("exclude"))
        dlg._confidence_spin.setValue(0.5)
        dlg._on_ok()
        assert dlg.get_result() == ClassifierFilter(
            classifier_name="content", categories=["drawing"], mode="exclude", min_confidence=0.5)

    def test_cancel_returns_none(self, parent, qtbot):
        dlg = ClassifierFilterDialog(parent, ClassifierFilter(classifier_name="content", categories=["photo"]))
        qtbot.addWidget(dlg)
        dlg.reject()
        assert dlg.get_result() is None

    def test_initial_filter_prefills_controls(self, parent, qtbot):
        initial = ClassifierFilter(classifier_name="content", categories=["photo", "nsfw"],
                                   mode="exclude", min_confidence=0.25)
        dlg = ClassifierFilterDialog(parent, initial)
        qtbot.addWidget(dlg)
        assert _checked(dlg) == ["photo", "nsfw"]
        assert dlg._mode_combo.currentData() == "exclude"
        assert dlg._confidence_spin.value() == pytest.approx(0.25)
        dlg._on_ok()
        assert dlg.get_result() == initial

    def test_unknown_categories_dropped_with_warning(self, parent, qtbot):
        initial = ClassifierFilter(classifier_name="content", categories=["photo", "landscape"])
        dlg = ClassifierFilterDialog(parent, initial)
        qtbot.addWidget(dlg)
        assert _checked(dlg) == ["photo"]
        assert not dlg._warning_lbl.isHidden()
        assert "landscape" in dlg._warning_lbl.text()

    def test_model_strategy_only_for_models_with_positive_groups(self, parent, qtbot):
        dlg = ClassifierFilterDialog(parent)
        qtbot.addWidget(dlg)
        assert dlg._strategy_radio.isEnabled()
        dlg._model_combo.setCurrentIndex(dlg._model_combo.findData("plain"))
        assert not dlg._strategy_radio.isEnabled()

    def test_model_strategy_result_has_no_categories(self, parent, qtbot):
        dlg = ClassifierFilterDialog(parent)
        qtbot.addWidget(dlg)
        dlg._strategy_radio.setChecked(True)
        assert _checked(dlg) == ["nsfw"]
        assert dlg._ok_btn.isEnabled()
        dlg._on_ok()
        result = dlg.get_result()
        assert result.selection_mode == SELECTION_MODEL_STRATEGY
        assert result.categories == []

    def test_audio_domain_lists_audio_models(self, parent, qtbot):
        initial = ClassifierFilter(classifier_name="speech", domain="audio", categories=["music"])
        dlg = ClassifierFilterDialog(parent, initial)
        qtbot.addWidget(dlg)
        assert dlg._model_combo.currentData() == "speech"
        assert _checked(dlg) == ["music"]

    def test_unregistered_model_kept_visible_and_ok_disabled(self, parent, qtbot):
        dlg = ClassifierFilterDialog(parent, ClassifierFilter(classifier_name="gone", categories=["x"]))
        qtbot.addWidget(dlg)
        assert dlg._model_combo.currentData() == "gone"
        assert not dlg._warning_lbl.isHidden()
        assert not dlg._ok_btn.isEnabled()


    def test_sampling_ratios_round_trip(self, parent, qtbot):
        initial = ClassifierFilter(classifier_name="content", categories=["photo"],
                                   sample_ratio=0.25, positive_ratio=0.5)
        dlg = ClassifierFilterDialog(parent, initial)
        qtbot.addWidget(dlg)
        dlg._on_ok()
        assert dlg.get_result() == initial

    def test_sampling_disabled_for_audio(self, parent, qtbot):
        dlg = ClassifierFilterDialog(parent)
        qtbot.addWidget(dlg)
        assert dlg._sample_ratio_spin.isEnabled()
        dlg._domain_combo.setCurrentIndex(dlg._domain_combo.findData("audio"))
        assert not dlg._sample_ratio_spin.isEnabled()
        assert not dlg._positive_ratio_spin.isEnabled()


class TestDescribeClassifierFilter:
    def test_unconfigured(self):
        assert describe_classifier_filter(None) == _("(not configured)")

    def test_include_lists_categories(self):
        f = ClassifierFilter(classifier_name="m", categories=["photo", "drawing"])
        assert describe_classifier_filter(f) == _("{0}: only {1}").format("m", "photo, drawing")


class _FakeDialog:
    """Replaces ClassifierFilterDialog in the panel; returns a preset result."""
    result = None
    opened = 0

    def __init__(self, parent, initial=None):
        type(self).opened += 1
        self.initial = initial

    def exec(self):
        return 0

    def get_result(self):
        return type(self).result


@pytest.fixture
def fake_dialog(monkeypatch):
    _FakeDialog.result = None
    _FakeDialog.opened = 0
    monkeypatch.setattr(filter_builder_panel_qt, "ClassifierFilterDialog", _FakeDialog)
    return _FakeDialog


class TestFilterPanelClassifierRow:
    def test_set_and_get_classifier_filter(self, qtbot, fake_dialog):
        panel = FilterBuilderPanel()
        qtbot.addWidget(panel)
        f = ClassifierFilter(classifier_name="content", categories=["photo"])
        panel.set_filter(f)
        assert panel.get_filter() == f
        assert fake_dialog.opened == 0

    def test_classifier_child_of_group_survives_set_filter(self, qtbot, fake_dialog):
        panel = FilterBuilderPanel()
        qtbot.addWidget(panel)
        group = CompareFilterGroup(operator=FilterOperator.OR, filters=[
            SizeFilter(min_size=(512, 512)),
            ClassifierFilter(classifier_name="content", categories=["photo"]),
        ])
        panel.set_filter(group)
        assert panel.get_filter() == group

    def test_switching_to_classifier_opens_dialog_and_keeps_result(self, qtbot, fake_dialog):
        panel = FilterBuilderPanel()
        qtbot.addWidget(panel)
        panel._add_row()
        row = panel._rows[0]
        fake_dialog.result = ClassifierFilter(classifier_name="content", categories=["nsfw"])
        row._type_combo.setCurrentText(_("Classifier"))
        assert fake_dialog.opened == 1
        assert panel.get_filter() == fake_dialog.result
        assert row._classifier_summary.text() == describe_classifier_filter(fake_dialog.result)

    def test_cancelled_dialog_reverts_row_type(self, qtbot, fake_dialog):
        panel = FilterBuilderPanel()
        qtbot.addWidget(panel)
        panel._add_row()
        row = panel._rows[0]
        row._type_combo.setCurrentText(_("Classifier"))
        assert fake_dialog.opened == 1
        assert row._type_combo.currentText() == _("Size")
        assert panel.get_filter() is None

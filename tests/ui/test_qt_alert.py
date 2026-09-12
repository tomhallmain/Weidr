"""
Behaviour of lib/qt_alert.py.

Qt supplies its own labels for standard buttons and this app installs no
QTranslator, so every message box has to relabel them through `_()` or they
render in English whatever the app locale is. That relabelling, the default
button, and each kind's return contract are what is pinned here.

exec() is never called on a real box -- it would block the runner -- so the
box is inspected directly, and the kind dispatch is driven through a stubbed
_make_box that reports what it was asked to build.
"""

import pytest
from PySide6.QtWidgets import QMessageBox, QWidget

import lib.qt_alert as qt_alert_module
from lib.qt_alert import _make_box, qt_alert
from utils.translations import _

OK_CANCEL = QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
YES_NO = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No


@pytest.fixture
def parent(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


class _StubBox:
    """Stands in for a QMessageBox so the dispatch can be driven without exec()."""

    def __init__(self, result):
        self._result = result

    def exec(self):
        return self._result


@pytest.fixture
def built(monkeypatch):
    """Capture what qt_alert asks _make_box to build, and choose the answer.

    Returns (calls, answer); set answer["value"] to the StandardButton the
    stubbed exec() should report.
    """
    calls = []
    answer = {"value": QMessageBox.StandardButton.Yes}

    def _fake_make_box(parent, icon, title, message, buttons, default, overrides=None):
        calls.append(
            {
                "icon": icon,
                "title": title,
                "message": message,
                "buttons": buttons,
                "default": default,
                "overrides": overrides or {},
            }
        )
        return _StubBox(answer["value"])

    monkeypatch.setattr(qt_alert_module, "_make_box", _fake_make_box)
    return calls, answer


class TestButtonLabels:
    def test_ok_and_cancel_are_translated(self, parent):
        box = _make_box(
            parent, QMessageBox.Icon.Question, "t", "m",
            OK_CANCEL, QMessageBox.StandardButton.Ok,
        )

        assert box.button(QMessageBox.StandardButton.Ok).text() == _("OK")
        assert box.button(QMessageBox.StandardButton.Cancel).text() == _("Cancel")

    def test_yes_and_no_are_translated(self, parent):
        box = _make_box(
            parent, QMessageBox.Icon.Question, "t", "m",
            YES_NO, QMessageBox.StandardButton.Yes,
        )

        assert box.button(QMessageBox.StandardButton.Yes).text() == _("Yes")
        assert box.button(QMessageBox.StandardButton.No).text() == _("No")

    def test_an_override_replaces_the_translated_label(self, parent):
        box = _make_box(
            parent, QMessageBox.Icon.Question, "t", "m",
            YES_NO, QMessageBox.StandardButton.Yes,
            overrides={QMessageBox.StandardButton.Yes: "Keep beginning"},
        )

        assert box.button(QMessageBox.StandardButton.Yes).text() == "Keep beginning"
        assert box.button(QMessageBox.StandardButton.No).text() == _("No")

    def test_the_requested_default_button_is_set(self, parent):
        box = _make_box(
            parent, QMessageBox.Icon.Question, "t", "m",
            YES_NO, QMessageBox.StandardButton.No,
        )

        assert box.defaultButton() is box.button(QMessageBox.StandardButton.No)


class TestDefaultButton:
    def test_a_question_defaults_to_yes(self, built):
        calls, _answer = built

        qt_alert(None, "t", "m", kind="askyesno")

        assert calls[0]["default"] == QMessageBox.StandardButton.Yes

    def test_default_no_declines_a_destructive_prompt(self, built):
        """Enter on a reset/delete confirm must not carry it out."""
        calls, _answer = built

        qt_alert(None, "t", "m", kind="askyesno", default_no=True)

        assert calls[0]["default"] == QMessageBox.StandardButton.No

    def test_default_no_also_applies_to_the_three_button_form(self, built):
        calls, _answer = built

        qt_alert(None, "t", "m", kind="askyesnocancel", default_no=True)

        assert calls[0]["default"] == QMessageBox.StandardButton.No


class TestReturnContract:
    def test_askokcancel_answers_with_a_bool(self, built):
        """Callers branch on truthiness; a StandardButton comparison against
        this would never match and would silently skip the action."""
        _calls, answer = built

        answer["value"] = QMessageBox.StandardButton.Ok
        assert qt_alert(None, "t", "m", kind="askokcancel") is True

        answer["value"] = QMessageBox.StandardButton.Cancel
        assert qt_alert(None, "t", "m", kind="askokcancel") is False

    def test_askyesno_answers_with_a_bool(self, built):
        _calls, answer = built

        answer["value"] = QMessageBox.StandardButton.Yes
        assert qt_alert(None, "t", "m", kind="askyesno") is True

        answer["value"] = QMessageBox.StandardButton.No
        assert qt_alert(None, "t", "m", kind="askyesno") is False

    def test_askyesnocancel_answers_with_the_button(self, built):
        """Three outcomes do not fit a bool, so this kind reports the button
        and its callers compare against StandardButton."""
        _calls, answer = built

        answer["value"] = QMessageBox.StandardButton.Cancel
        assert (
            qt_alert(None, "t", "m", kind="askyesnocancel")
            == QMessageBox.StandardButton.Cancel
        )

    @pytest.mark.parametrize("kind", ["info", "warning", "error"])
    def test_a_plain_message_answers_with_none(self, built, kind):
        _calls, _answer = built
        assert qt_alert(None, "t", "m", kind=kind) is None

    def test_an_unknown_kind_falls_back_to_info(self, built):
        calls, _answer = built

        qt_alert(None, "t", "m", kind="not-a-kind")

        assert calls[0]["icon"] == QMessageBox.Icon.Information
        assert calls[0]["buttons"] == QMessageBox.StandardButton.Ok


class TestCustomLabels:
    def test_yes_and_no_text_reach_the_box_as_overrides(self, built):
        calls, _answer = built

        qt_alert(
            None, "t", "m", kind="askyesno",
            yes_text="Keep beginning", no_text="Keep end",
        )

        assert calls[0]["overrides"] == {
            QMessageBox.StandardButton.Yes: "Keep beginning",
            QMessageBox.StandardButton.No: "Keep end",
        }

    def test_they_are_ignored_by_kinds_without_those_buttons(self, built):
        calls, _answer = built

        qt_alert(None, "t", "m", kind="askokcancel", yes_text="Keep beginning")

        assert calls[0]["overrides"] == {}

"""UI tests for show_high_severity_dialog's per-item checkbox mode.

The dialog is modal, so every test drives it without exec(): the checkbox
widgets and the accept/reject paths are exercised directly, which is enough to
pin the return contract that callers branch on.
"""

import pytest
from PySide6.QtWidgets import QCheckBox, QDialog

from lib.custom_dialogs_qt import show_high_severity_dialog


def _run_dialog(monkeypatch, items, accept=True, check=None):
    """Show the dialog with exec() stubbed, optionally re-checking boxes.

    *check*, when given, receives the dialog's checkboxes and sets their state
    to whatever the "user" chose before the result is read.
    """
    captured = {}

    def _fake_exec(self):
        captured["dialog"] = self
        boxes = self.findChildren(QCheckBox)
        captured["boxes"] = boxes
        if check is not None:
            check(boxes)
        return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", _fake_exec)
    result = show_high_severity_dialog(None, "Title", "Message", items=items)
    return result, captured


class TestItemsMode:
    def test_renders_one_checkbox_per_item(self, qtbot, monkeypatch):
        items = [("alpha", True), ("beta", False), ("gamma", True)]
        _result, captured = _run_dialog(monkeypatch, items)
        assert [b.text() for b in captured["boxes"]] == ["alpha", "beta", "gamma"]

    def test_default_checked_state_is_honoured(self, qtbot, monkeypatch):
        items = [("alpha", True), ("beta", False)]
        _result, captured = _run_dialog(monkeypatch, items)
        assert [b.isChecked() for b in captured["boxes"]] == [True, False]

    def test_accept_returns_the_checked_labels(self, qtbot, monkeypatch):
        items = [("alpha", True), ("beta", False), ("gamma", True)]
        result, _captured = _run_dialog(monkeypatch, items)
        assert result == ["alpha", "gamma"]

    def test_accept_reflects_a_changed_selection(self, qtbot, monkeypatch):
        items = [("alpha", True), ("beta", True)]

        def _uncheck_alpha(boxes):
            boxes[0].setChecked(False)

        result, _captured = _run_dialog(monkeypatch, items, check=_uncheck_alpha)
        assert result == ["beta"]

    def test_accept_with_nothing_checked_returns_empty_list(self, qtbot, monkeypatch):
        """Empty must stay distinguishable from cancel: 'proceed with none of
        these' is a different instruction than 'do not proceed'."""
        items = [("alpha", True)]

        def _uncheck_all(boxes):
            for box in boxes:
                box.setChecked(False)

        result, _captured = _run_dialog(monkeypatch, items, check=_uncheck_all)
        assert result == []
        assert result is not False

    def test_cancel_returns_false(self, qtbot, monkeypatch):
        items = [("alpha", True)]
        result, _captured = _run_dialog(monkeypatch, items, accept=False)
        assert result is False


class TestLegacyModesUnaffected:
    """items defaults to None, so every pre-existing caller keeps its contract."""

    def test_plain_mode_still_returns_bool(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            QDialog, "exec", lambda self: QDialog.DialogCode.Accepted
        )
        assert show_high_severity_dialog(None, "T", "M") is True

    def test_plain_mode_cancel_still_returns_false(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            QDialog, "exec", lambda self: QDialog.DialogCode.Rejected
        )
        assert show_high_severity_dialog(None, "T", "M") is False

    def test_no_checkboxes_are_added_without_items(self, qtbot, monkeypatch):
        captured = {}

        def _fake_exec(self):
            captured["boxes"] = self.findChildren(QCheckBox)
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(QDialog, "exec", _fake_exec)
        show_high_severity_dialog(None, "T", "M")
        assert captured["boxes"] == []

    def test_custom_buttons_still_return_the_clicked_label(self, qtbot, monkeypatch):
        """buttons mode takes precedence; items is ignored there rather than
        silently changing the return type a caller already depends on."""
        monkeypatch.setattr(
            QDialog, "exec", lambda self: QDialog.DialogCode.Accepted
        )
        result = show_high_severity_dialog(
            None, "T", "M", buttons=[("Proceed", "destructive"), ("Stop", "reject")]
        )
        # No button was actually clicked under a stubbed exec, so the tracked
        # label stays None -- the contract under test is that it does not raise
        # and does not return a list.
        assert not isinstance(result, list)

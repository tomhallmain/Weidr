"""
Qt message box helpers for Qt applications.
"""
from typing import Optional

from PySide6.QtWidgets import QWidget, QMessageBox

from utils.translations import _


def _make_box(
    parent: Optional[QWidget],
    icon: QMessageBox.Icon,
    title: str,
    message: str,
    buttons: QMessageBox.StandardButton,
    default: QMessageBox.StandardButton,
    overrides: Optional[dict] = None,
) -> QMessageBox:
    """Create a QMessageBox with translated button labels and the given default.

    *overrides* maps a QMessageBox.StandardButton to a custom label that
    takes precedence over the default translation (e.g. renaming Yes/No to
    describe the actual choice being made).
    """
    box = QMessageBox(icon, title, message, buttons, parent)
    box.setDefaultButton(default)

    # Translate standard button labels
    _translations = {
        QMessageBox.StandardButton.Ok: _("OK"),
        QMessageBox.StandardButton.Cancel: _("Cancel"),
        QMessageBox.StandardButton.Yes: _("Yes"),
        QMessageBox.StandardButton.No: _("No"),
    }
    if overrides:
        _translations.update(overrides)
    for btn_type, label in _translations.items():
        btn = box.button(btn_type)
        if btn is not None:
            btn.setText(label)

    return box


def qt_alert(
    parent: Optional[QWidget],
    title: str,
    message: str,
    kind: str = "info",
    yes_text: Optional[str] = None,
    no_text: Optional[str] = None,
):
    """Show a Qt message box. kind: info, warning, error, askokcancel, askyesno, askyesnocancel.

    *yes_text* and *no_text* override the Yes/No button labels for the
    askyesno and askyesnocancel kinds; they are ignored otherwise.
    """
    overrides = {}
    if yes_text:
        overrides[QMessageBox.StandardButton.Yes] = yes_text
    if no_text:
        overrides[QMessageBox.StandardButton.No] = no_text

    if kind == "askokcancel":
        box = _make_box(
            parent, QMessageBox.Icon.Question, title, message,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        return box.exec() == QMessageBox.StandardButton.Ok
    if kind == "askyesno":
        box = _make_box(
            parent, QMessageBox.Icon.Question, title, message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
            overrides=overrides,
        )
        return box.exec() == QMessageBox.StandardButton.Yes
    if kind == "askyesnocancel":
        box = _make_box(
            parent, QMessageBox.Icon.Question, title, message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
            overrides=overrides,
        )
        return box.exec()
    if kind == "error":
        box = _make_box(
            parent, QMessageBox.Icon.Critical, title, message,
            QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        box.exec()
        return None
    if kind == "warning":
        box = _make_box(
            parent, QMessageBox.Icon.Warning, title, message,
            QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
        box.exec()
        return None
    # info
    box = _make_box(
        parent, QMessageBox.Icon.Information, title, message,
        QMessageBox.StandardButton.Ok,
        QMessageBox.StandardButton.Ok,
    )
    box.exec()
    return None

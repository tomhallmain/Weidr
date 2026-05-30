"""
Custom dialog implementations for PySide6.

Provides a high-severity confirmation dialog with red warning styling.
Standard dialogs (info, error, askokcancel, etc.) are already covered
by lib/qt_alert.py.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from utils.translations import _

# Colour palette for the high-severity dialog
_BG = "#ff4444"
_BTN_OK = "#cc0000"
_BTN_CANCEL = "#666666"
_BTN_ACTION = "#2255aa"
_FG = "white"

_STYLE = f"""
    QDialog {{
        background-color: {_BG};
    }}
    QLabel {{
        color: {_FG};
        background-color: {_BG};
    }}
    QPushButton#ok_btn {{
        background-color: {_BTN_OK};
        color: {_FG};
        font-weight: bold;
        border: 2px solid #aa0000;
        padding: 6px 20px;
        border-radius: 3px;
    }}
    QPushButton#ok_btn:hover {{
        background-color: #dd0000;
    }}
    QPushButton#cancel_btn {{
        background-color: {_BTN_CANCEL};
        color: {_FG};
        border: 2px solid #555555;
        padding: 6px 20px;
        border-radius: 3px;
    }}
    QPushButton#cancel_btn:hover {{
        background-color: #777777;
    }}
    QPushButton#action_btn {{
        background-color: {_BTN_ACTION};
        color: {_FG};
        border: 2px solid #1a3d77;
        padding: 6px 20px;
        border-radius: 3px;
    }}
    QPushButton#action_btn:hover {{
        background-color: #2f66cc;
    }}
"""


def show_high_severity_dialog(
    master: Optional[QWidget],
    title: str,
    message: str,
    buttons: Optional[list[tuple[str, str]]] = None,
) -> "bool | str":
    """
    Show a custom dialog with red warning colours for high-severity operations.

    Args:
        master: Parent widget (or None).
        title: Dialog title.
        message: Dialog message.
        buttons: Optional list of (label, role) pairs to replace the default
            OK/Cancel buttons.  Supported roles: ``"destructive"``,
            ``"action"``, ``"reject"``.  When provided, reject-role buttons
            return ``False`` and all other buttons return their label string,
            so callers can uniformly use ``if not result:`` to detect
            cancellation regardless of button mode.  When *not* provided the
            legacy two-button behaviour is used and a plain ``bool`` is
            returned.

    Returns:
        * ``bool`` when *buttons* is ``None``: ``True`` for OK, ``False``
          for Cancel.
        * ``str`` (label) when *buttons* is provided and a non-reject button
          was clicked.
        * ``False`` when *buttons* is provided and a reject-role button was
          clicked.
    """
    dialog = QDialog(master)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.setStyleSheet(_STYLE)
    dialog.setMinimumWidth(400)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(10)

    # Warning icon + title row
    title_row = QHBoxLayout()
    icon_label = QLabel("\u26A0\uFE0F")  # ⚠️
    icon_font = QFont()
    icon_font.setPointSize(24)
    icon_label.setFont(icon_font)
    title_row.addWidget(icon_label)

    title_label = QLabel(title)
    title_font = QFont()
    title_font.setPointSize(14)
    title_font.setBold(True)
    title_label.setFont(title_font)
    title_row.addWidget(title_label)
    title_row.addStretch()
    layout.addLayout(title_row)

    # Message body
    msg_label = QLabel(message)
    msg_font = QFont()
    msg_font.setPointSize(10)
    msg_label.setFont(msg_font)
    msg_label.setWordWrap(True)
    msg_label.setMaximumWidth(450)
    layout.addWidget(msg_label)

    layout.addStretch()

    # Buttons
    btn_layout = QHBoxLayout()
    btn_layout.addStretch()

    if buttons is None:
        # Legacy two-button mode: Cancel (focus) then OK
        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.setObjectName("cancel_btn")
        cancel_btn.clicked.connect(dialog.reject)
        cancel_btn.setFocus()  # Default focus on Cancel for safety
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton(_("OK"))
        ok_btn.setObjectName("ok_btn")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)
        dialog.adjustSize()
        return dialog.exec() == QDialog.DialogCode.Accepted

    # Custom-button mode: track which label was clicked
    _clicked_label: list[str | None] = [None]

    _VALID_ROLES = {"destructive", "action", "reject"}
    _ROLE_OBJECT_NAMES = {
        "destructive": "ok_btn",
        "action": "action_btn",
        "reject": "cancel_btn",
    }
    _reject_roles = {"reject"}

    last_reject_btn: Optional[QPushButton] = None

    for label, role in buttons:
        if role not in _VALID_ROLES:
            raise ValueError(
                f"Invalid button role {role!r}. Must be one of: {sorted(_VALID_ROLES)}"
            )
        btn = QPushButton(label)
        btn.setObjectName(_ROLE_OBJECT_NAMES[role])

        def _make_handler(lbl: str, is_reject: bool):
            def _handler():
                # Reject-role buttons return False (consistent with legacy mode)
                # so callers can uniformly use `if not result:` to detect cancellation.
                _clicked_label[0] = False if is_reject else lbl
                if is_reject:
                    dialog.reject()
                else:
                    dialog.accept()
            return _handler

        btn.clicked.connect(_make_handler(label, role in _reject_roles))
        btn_layout.addWidget(btn)

        if role == "reject":
            last_reject_btn = btn

    # High-severity dialogs always default focus to the reject button so that
    # an accidental Enter/Space press cannot trigger a destructive action.
    if last_reject_btn is not None:
        last_reject_btn.setFocus()

    layout.addLayout(btn_layout)
    dialog.adjustSize()
    dialog.exec()
    return _clicked_label[0]

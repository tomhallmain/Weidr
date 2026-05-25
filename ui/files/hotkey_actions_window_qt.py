"""
PySide6 port of files/hotkey_actions_window.py -- HotkeyActionsWindow.

Displays the configured hotkey actions (T for permanent, 0-9 for numbered)
with their current target directories and "Set" buttons.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGridLayout, QLabel, QPushButton, QWidget,
)

from files.file_action import FileAction
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from ui.auth.password_utils import require_password
from utils.app_actions import AppActions
from utils.constants import ProtectedActions
from utils.translations import I18N

_ = I18N._


class HotkeyActionsWindow(SmartDialog):
    """
    Dialog listing each numeric hotkey (T, 1-9, 0) with its currently
    assigned target directory and a button to (re)set it.
    """

    _instance: Optional[HotkeyActionsWindow] = None
    MAX_HEIGHT = 900
    COL_1_WIDTH = 600

    @staticmethod
    def get_geometry() -> str:
        return "600x400"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        master: QWidget,
        app_actions: AppActions,
        set_permanent_action_callback: Callable,
        set_hotkey_action_callback: Callable,
    ) -> None:
        super().__init__(
            parent=master,
            position_parent=master,
            title=_("Hotkey Actions"),
            geometry=self.get_geometry(),
        )
        HotkeyActionsWindow._instance = self

        self._app_actions = app_actions
        self._set_permanent_action_callback = set_permanent_action_callback
        self._set_hotkey_action_callback = set_hotkey_action_callback

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(10, 10, 10, 10)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 8)
        self._grid.setColumnStretch(2, 1)

        # Header row
        self._add_label(_("Key name"), 0, 0, bold=True)
        self._add_label(_("Target directory"), 0, 1, bold=True)

        # T = permanent action (row 1)
        self._add_hotkey_row("T", is_index=False, grid_row=1)

        # 1-9 (rows 2-10)
        for i in range(1, 10):
            self._add_hotkey_row(i, grid_row=i + 1)

        # 0 (row 11)
        self._add_hotkey_row(0, grid_row=11)

        # Keyboard shortcuts: T / 0-9 trigger set, Escape closes
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence("Shift+T"), self).activated.connect(
            lambda: self._on_keyboard_set("T")
        )
        for digit in range(10):
            QShortcut(QKeySequence(str(digit)), self).activated.connect(
                lambda d=digit: self._on_keyboard_set(d)
            )

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------
    def _add_hotkey_row(self, key, *, is_index: bool = True, grid_row: int) -> None:
        key_index = int(key) if is_index else key
        hotkey_name = f"Shift-{key_index}"
        self._add_label(hotkey_name, grid_row, 0)

        # Resolve current target
        if is_index:
            action = FileAction.hotkey_actions.get(key_index, _("(unset)"))
        else:
            action = FileAction.permanent_action or _("(unset)")

        self._add_label(str(action), grid_row, 1, wrap_width=self.COL_1_WIDTH)

        set_btn = QPushButton(_("Set"))
        set_btn.clicked.connect(lambda _checked=False, k=key, ki=key_index: self._on_set_clicked(k, ki))
        self._grid.addWidget(set_btn, grid_row, 2)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    @require_password(ProtectedActions.SET_HOTKEY_ACTIONS)
    def _on_set_clicked(self, hotkey, key_index) -> None:
        if hotkey == "T":
            self._set_permanent_action_callback()
        else:
            self._set_hotkey_action_callback(hotkey_override=key_index)

    @require_password(ProtectedActions.SET_HOTKEY_ACTIONS)
    def _on_keyboard_set(self, key) -> None:
        if key == "T":
            self._set_permanent_action_callback()
        else:
            self._set_hotkey_action_callback(hotkey_override=int(key))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add_label(
        self,
        text: str,
        row: int,
        col: int,
        *,
        bold: bool = False,
        wrap_width: int = 200,
    ) -> None:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(wrap_width)
        style = f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        if bold:
            style += " font-weight: bold;"
        lbl.setStyleSheet(style)
        self._grid.addWidget(lbl, row, col, Qt.AlignLeft | Qt.AlignTop)

    def close_windows(self, event=None) -> None:
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        HotkeyActionsWindow._instance = None
        super().closeEvent(event)

"""
LookaheadsTab — standalone tab for managing Lookahead objects.

Extracted from PrevalidationsTab so lookaheads can be managed independently.
Cross-tab notification (refreshing an open PrevalidationModifyWindow when a
lookahead is renamed/removed) is deferred; it will be wired via the
ClassifierManagementWindow once a shared refresh protocol is in place.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from compare.classifier_actions_manager import ClassifierActionsManager
from compare.lookahead import Lookahead
from ui.app_style import AppStyle
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("lookaheads_tab_qt")


class LookaheadsTab(QWidget):
    """Tab content widget for managing Lookahead objects."""

    _lookahead_window = None

    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(parent)
        self._app_actions = app_actions

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        title = QLabel(_("Lookaheads"))
        title.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 13pt;"
        )
        root.addWidget(title)

        lh_area = QHBoxLayout()

        self._lh_listbox = QListWidget()
        self._lh_listbox.setStyleSheet(
            f"QListWidget {{ background: {AppStyle.BG_COLOR};"
            f" color: {AppStyle.FG_COLOR}; }}"
        )
        self._lh_listbox.doubleClicked.connect(self._edit_lookahead)
        lh_area.addWidget(self._lh_listbox, 1)

        lh_btns = QVBoxLayout()
        lh_btns.setSpacing(2)
        add_lh = QPushButton(_("Add Lookahead"))
        add_lh.clicked.connect(self._add_lookahead)
        lh_btns.addWidget(add_lh)
        edit_lh = QPushButton(_("Edit Lookahead"))
        edit_lh.clicked.connect(self._edit_lookahead)
        lh_btns.addWidget(edit_lh)
        rm_lh = QPushButton(_("Remove Lookahead"))
        rm_lh.clicked.connect(self._remove_lookahead)
        lh_btns.addWidget(rm_lh)
        lh_btns.addStretch()
        lh_area.addLayout(lh_btns)

        root.addLayout(lh_area)
        root.addStretch()

        self._refresh_lh_listbox()

    # ------------------------------------------------------------------
    # List population
    # ------------------------------------------------------------------

    def _refresh_lh_listbox(self) -> None:
        self._lh_listbox.clear()
        for lh in Lookahead.lookaheads:
            text = _("{name} ({name_or_text}, threshold: {threshold:.2f})").format(
                name=lh.name,
                name_or_text=lh.name_or_text,
                threshold=lh.threshold,
            )
            self._lh_listbox.addItem(text)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _add_lookahead(self) -> None:
        from ui.compare.lookahead_window_qt import LookaheadWindow

        if LookaheadsTab._lookahead_window is not None:
            try:
                LookaheadsTab._lookahead_window.close()
            except Exception:
                pass
        LookaheadsTab._lookahead_window = LookaheadWindow(
            self.window(), self._app_actions, self._refresh_lh_listbox
        )
        LookaheadsTab._lookahead_window.show()

    def _edit_lookahead(self) -> None:
        from ui.compare.lookahead_window_qt import LookaheadWindow

        idx = self._lh_listbox.currentRow()
        if idx < 0 or idx >= len(Lookahead.lookaheads):
            return
        if LookaheadsTab._lookahead_window is not None:
            try:
                LookaheadsTab._lookahead_window.close()
            except Exception:
                pass
        LookaheadsTab._lookahead_window = LookaheadWindow(
            self.window(),
            self._app_actions,
            self._refresh_lh_listbox,
            Lookahead.lookaheads[idx],
        )
        LookaheadsTab._lookahead_window.show()

    def _remove_lookahead(self) -> None:
        idx = self._lh_listbox.currentRow()
        if idx < 0 or idx >= len(Lookahead.lookaheads):
            return
        lh = Lookahead.lookaheads[idx]
        used_by = [
            pv.name
            for pv in ClassifierActionsManager.prevalidations
            if lh.name in pv.lookahead_names
        ]
        if used_by:
            logger.warning(
                "Lookahead %s is used by prevalidations: %s",
                lh.name,
                ", ".join(used_by),
            )
        del Lookahead.lookaheads[idx]
        self._refresh_lh_listbox()

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self._refresh_lh_listbox()

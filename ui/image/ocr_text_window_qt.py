"""
PySide6 OCR text viewer window.

Displays text extracted via Surya OCR in a read-only text area.
The user can select and copy a portion of the text or copy all of it
with a single button press.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.translations import _


class OCRTextWindow(SmartDialog):
    """Window that shows OCR-extracted text with copy support."""

    _instance: OCRTextWindow | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        parent: QWidget,
        app_actions,
        ocr_text: str,
        image_path: str,
        confidence: float | None = None,
        dimensions: str = "600x500",
    ) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("OCR Text") + " - " + image_path,
            geometry=dimensions,
        )
        OCRTextWindow._instance = self

        self._app_actions = app_actions
        self._ocr_text = ocr_text
        self._has_closed = False

        self._build_ui(confidence)
        self._bind_shortcuts()

        self._text_edit.setPlainText(ocr_text)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self, confidence: float | None) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        # -- Top bar -----------------------------------------------------
        top_bar = QHBoxLayout()

        self._copy_btn = QPushButton(_("Copy All Text"))
        self._copy_btn.clicked.connect(self.copy_text_to_clipboard)
        top_bar.addWidget(self._copy_btn)

        header_text = _("OCR Result")
        if confidence is not None:
            header_text += f"  ({_('avg confidence')}: {confidence:.1%})"
        header = QLabel(header_text)
        header.setWordWrap(True)
        header.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        top_bar.addWidget(header)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # -- Text area ---------------------------------------------------
        self._text_edit = QPlainTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        layout.addWidget(self._text_edit, stretch=1)

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------
    def _bind_shortcuts(self) -> None:
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self.close)

        copy_sc = QShortcut(QKeySequence("Ctrl+C"), self)
        copy_sc.activated.connect(self._copy_shortcut)

    def _copy_shortcut(self) -> None:
        """Ctrl+C: copy selection if any, otherwise copy all."""
        cursor = self._text_edit.textCursor()
        if cursor.hasSelection():
            self._text_edit.copy()
        else:
            self.copy_text_to_clipboard()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def has_closed(self) -> bool:
        return self._has_closed

    def update_text(self, ocr_text: str, image_path: str, confidence: float | None = None) -> None:
        """Refresh with new OCR results."""
        self._ocr_text = ocr_text
        self._text_edit.setPlainText(ocr_text)
        title = _("OCR Text") + " - " + image_path
        if confidence is not None:
            title += f"  ({_('avg confidence')}: {confidence:.1%})"
        self.setWindowTitle(title)

    def copy_text_to_clipboard(self) -> None:
        """Copy all OCR text to the system clipboard."""
        try:
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(self._ocr_text)
            if self._app_actions:
                self._app_actions.success(_("Copied OCR text to clipboard"))
        except Exception as e:
            if self._app_actions:
                self._app_actions.warn(
                    _("Error copying OCR text: ") + str(e)
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self._has_closed = True
        OCRTextWindow._instance = None
        super().closeEvent(event)

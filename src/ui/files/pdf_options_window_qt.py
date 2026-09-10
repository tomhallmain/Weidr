"""
PDFOptionsWindow -- PDF export options before creating from marks (PySide6).

Singleton dialog for quality and filename settings.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.app_actions import AppActions
from utils.translations import _


class PDFOptionsWindow(SmartDialog):
    """Singleton dialog for configuring PDF creation options."""

    _instance: Optional[PDFOptionsWindow] = None
    COL_0_WIDTH = 400

    # ------------------------------------------------------------------
    # Singleton show / close
    # ------------------------------------------------------------------
    @classmethod
    def show(
        cls,
        master: QWidget,
        app_actions: AppActions,
        callback: Callable,
        initial_filename: Optional[str] = None,
    ) -> None:
        """Show the PDF options dialog (singleton)."""
        if cls._instance is not None:
            try:
                if cls._instance.isVisible():
                    if cls._instance.isMinimized():
                        cls._instance.showNormal()
                    cls._instance.raise_()
                    cls._instance.activateWindow()
                    return
                else:
                    cls._instance = None
            except Exception:
                cls._instance = None

        cls._instance = cls(
            parent=master,
            callback=callback,
            initial_filename=initial_filename,
        )
        cls._instance.show_()

    def show_(self) -> None:
        """Non-static show to avoid name clash with QWidget.show."""
        super().show()

    @classmethod
    def on_closing(cls, event=None) -> None:
        if cls._instance is not None:
            cls._instance.close()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        parent: QWidget,
        callback: Callable,
        initial_filename: Optional[str] = None,
    ) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("PDF Creation Options"),
            geometry="500x250",
        )
        self._callback = callback

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        # Title
        title = QLabel(_("PDF Creation Options"))
        title.setStyleSheet(
            f"font-size: 14pt; font-weight: bold; "
            f"color: {AppStyle.FG_COLOR};"
        )
        layout.addWidget(title)

        # Filename
        layout.addSpacing(8)
        fn_label = QLabel(_("PDF Filename:"))
        fn_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        layout.addWidget(fn_label)

        filename_default = (
            str(initial_filename).strip()
            if initial_filename is not None and str(initial_filename).strip()
            else _("combined_images")
        )
        self._filename_entry = QLineEdit(filename_default)
        self._filename_entry.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_INPUT}; "
            f"border: 1px solid {AppStyle.BORDER_COLOR}; padding: 2px 4px;"
        )
        layout.addWidget(self._filename_entry)

        # Quality checkbox
        layout.addSpacing(8)
        self._preserve_quality = QCheckBox(_("Preserve original image quality and format"))
        self._preserve_quality.setChecked(True)
        self._preserve_quality.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        layout.addWidget(self._preserve_quality)

        desc = QLabel(
            _("If enabled, images will maintain their original quality and format.\n"
              "If disabled, images will be compressed to reduce PDF size.")
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 8pt;")
        layout.addWidget(desc)

        # Button row
        layout.addStretch()
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(cancel_btn)

        create_btn = QPushButton(_("Create PDF"))
        create_btn.clicked.connect(self._on_create)
        btn_row.addWidget(create_btn)

        layout.addLayout(btn_row)

        # Escape to close
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_create(self) -> None:
        options = {
            "preserve_quality": self._preserve_quality.isChecked(),
            "filename": self._filename_entry.text(),
        }
        self._callback(options)
        self.close()

    def reject(self) -> None:  # noqa: N802
        PDFOptionsWindow._instance = None
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        PDFOptionsWindow._instance = None
        super().closeEvent(event)

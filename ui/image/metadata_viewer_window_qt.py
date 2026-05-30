"""
PySide6 port of image/metadata_viewer_window.py -- MetadataViewerWindow.

Displays raw image metadata in a read-only text area with a copy-to-clipboard
button.  Singleton pattern: re-uses the existing window when already open.
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


class MetadataViewerWindow(SmartDialog):
    """Window to hold raw metadata."""

    MAX_ACTION_ROWS = 2000
    COL_0_WIDTH = 150

    # Singleton bookkeeping
    _instance: MetadataViewerWindow | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        parent: QWidget,
        app_actions,
        metadata_text: str,
        image_path: str,
        dimensions: str = "600x600",
    ) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Metadata Viewer") + " - " + image_path,
            geometry=dimensions,
        )
        MetadataViewerWindow._instance = self

        self._app_actions = app_actions
        self._metadata_text = metadata_text
        self._has_closed = False

        self._build_ui()
        self._bind_shortcuts()

        self._metadata_edit.setPlainText(metadata_text)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        # -- Top bar: Copy button + header label -----------------------
        top_bar = QHBoxLayout()

        self._copy_btn = QPushButton(_("Copy Metadata"))
        self._copy_btn.clicked.connect(self.copy_metadata_to_clipboard)
        top_bar.addWidget(self._copy_btn)

        header = QLabel(_("Raw Image Metadata"))
        header.setMaximumWidth(MetadataViewerWindow.COL_0_WIDTH)
        header.setWordWrap(True)
        header.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        top_bar.addWidget(header)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        # -- Metadata text area (read-only, scrollable) ----------------
        self._metadata_edit = QPlainTextEdit()
        self._metadata_edit.setReadOnly(True)
        self._metadata_edit.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        layout.addWidget(self._metadata_edit, stretch=1)

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------
    def _bind_shortcuts(self) -> None:
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self.close)

        copy_sc = QShortcut(QKeySequence("Ctrl+C"), self)
        copy_sc.activated.connect(self._copy_shortcut)

    def _copy_shortcut(self) -> None:
        """Ctrl+C: if text is selected copy selection, otherwise copy all."""
        cursor = self._metadata_edit.textCursor()
        if cursor.hasSelection():
            # Let the native copy handle it
            self._metadata_edit.copy()
        else:
            self.copy_metadata_to_clipboard()

    # ------------------------------------------------------------------
    # Public API (mirrors original)
    # ------------------------------------------------------------------
    @property
    def has_closed(self) -> bool:
        return self._has_closed

    def update_metadata(self, metadata_text: str, image_path: str) -> None:
        """Refresh the displayed metadata and window title."""
        self._metadata_text = metadata_text
        self._metadata_edit.setPlainText(metadata_text)
        self.setWindowTitle(_("Metadata Viewer") + " - " + image_path)

    def copy_metadata_to_clipboard(self) -> None:
        """Copy the full raw metadata text to the system clipboard."""
        try:
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(self._metadata_text)
            if self._app_actions:
                self._app_actions.success(_("Copied metadata to clipboard"))
        except Exception as e:
            if self._app_actions:
                self._app_actions.warn(
                    _("Error copying metadata to clipboard: ") + str(e)
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        self._has_closed = True
        MetadataViewerWindow._instance = None
        super().closeEvent(event)

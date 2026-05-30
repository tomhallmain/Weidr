"""
PySide6 port of files/type_configuration_window.py -- TypeConfigurationWindow.

Singleton dialog with a checkbox grid for enabling/disabling media types
in the compare pipeline.  Non-UI logic (apply_changes, pending-change
tracking, config mutations) is kept here because it's tightly coupled to
the UI state; persistence delegates to app_info_cache.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

from lib.multi_display_qt import SmartDialog
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.constants import CompareMediaType
from utils.translations import _

from image.frame_cache import (
    has_imported_pypdfium2,
    has_imported_cairosvg,
    has_imported_pyppeteer,
)


class TypeConfigurationWindow(SmartDialog):
    """Singleton dialog for configuring which media types are compared."""

    _instance: Optional[TypeConfigurationWindow] = None
    COL_0_WIDTH = 600
    _pending_changes: dict[CompareMediaType, bool] = {}
    _original_config: dict[CompareMediaType, bool] = {}

    # Media type descriptions
    MEDIA_TYPE_DESCRIPTIONS: dict[CompareMediaType, str] = {
        CompareMediaType.IMAGE: _("Basic image files (PNG, JPG, etc.)"),
        CompareMediaType.VIDEO: _("Video files (MP4, AVI, etc.) - First frame will be extracted"),
        CompareMediaType.GIF: _("Animated GIF files - First frame will be extracted"),
        CompareMediaType.PDF: _("PDF documents - First page will be extracted"),
        CompareMediaType.SVG: _("Vector graphics - Will be converted to raster image"),
        CompareMediaType.HTML: _("HTML files - Will be rendered and converted to image"),
        CompareMediaType.AUDIO: _(
            "Audio files (MP3, FLAC, etc.) - Browse and playback; not used in embedding compare"
        ),
    }

    DEPENDENCY_INFO: dict[CompareMediaType, dict] = {
        CompareMediaType.PDF: {
            "available": has_imported_pypdfium2,
            "package": "pypdfium2",
            "description": _("PDF support requires pypdfium2 package"),
        },
        CompareMediaType.SVG: {
            "available": has_imported_cairosvg,
            "package": "cairosvg",
            "description": _("SVG support requires cairosvg package"),
        },
        CompareMediaType.HTML: {
            "available": has_imported_pyppeteer,
            "package": "pyppeteer",
            "description": _("HTML support requires pyppeteer package"),
        },
    }

    # ------------------------------------------------------------------
    # Persistence (class-level, no UI needed)
    # ------------------------------------------------------------------
    @classmethod
    def load_pending_changes(cls) -> None:
        pending = app_info_cache.get_meta("file_type_configuration", default_val={})
        assert isinstance(pending, dict)
        for name, enabled in pending.items():
            cls._pending_changes[CompareMediaType[name]] = enabled

    @classmethod
    def save_pending_changes(cls) -> None:
        out: dict[str, bool] = {}
        if cls._original_config:
            for mt, enabled in cls._original_config.items():
                out[mt.name] = cls._pending_changes.get(mt, enabled)
        else:
            # Startup apply path can run without _original_config.
            for mt, enabled in cls._pending_changes.items():
                out[mt.name] = enabled
        app_info_cache.set_meta("file_type_configuration", out)
        app_info_cache.store()

    @staticmethod
    def get_geometry() -> str:
        return "700x450"

    # ------------------------------------------------------------------
    # Singleton show / close
    # ------------------------------------------------------------------
    @classmethod
    def show(cls, master: Optional[QWidget] = None, app_actions=None) -> None:  # noqa: D401
        if cls._instance is not None:
            cls._instance.raise_()
            cls._instance.activateWindow()
            return
        if master is None:
            raise ValueError("Master window must be provided")
        if app_actions is None:
            raise ValueError("AppActions instance must be provided")

        cls._original_config = {
            CompareMediaType.VIDEO: config.enable_videos,
            CompareMediaType.GIF: config.enable_gifs,
            CompareMediaType.PDF: config.enable_pdfs,
            CompareMediaType.SVG: config.enable_svgs,
            CompareMediaType.HTML: config.enable_html,
            CompareMediaType.AUDIO: config.enable_audio,
        }

        cls._instance = cls(parent=master, app_actions=app_actions)
        cls._instance.show_()

    def show_(self) -> None:
        super().show()

    @classmethod
    def on_closing(cls, event=None) -> None:
        if cls._instance is not None:
            cls._pending_changes.clear()
            cls._original_config.clear()
            cls._instance.close()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Media Type Configuration"),
            geometry=self.get_geometry(),
        )
        self._app_actions = app_actions
        self._checkboxes: dict[CompareMediaType, QCheckBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(6)

        # Title
        title = QLabel(_("Configure Media Types"))
        title.setStyleSheet(
            f"font-size: 14pt; font-weight: bold; color: {AppStyle.FG_COLOR};"
        )
        root.addWidget(title)

        desc = QLabel(
            _("Select which types of media files you want to compare. "
              "Changes will require a refresh of open comparisons but "
              "files in browsing mode should update automatically.")
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(desc)
        root.addSpacing(8)

        # Checkboxes
        for media_type in CompareMediaType:
            if media_type == CompareMediaType.UNCONFIGURED:
                continue
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(media_type.get_translation())
            cb.setChecked(self._get_initial_value(media_type))
            cb.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            self._checkboxes[media_type] = cb

            # IMAGE is always on; disable if dependency missing
            dep = self.DEPENDENCY_INFO.get(media_type)
            if media_type == CompareMediaType.IMAGE:
                cb.setEnabled(False)
                cb.setChecked(True)
            elif dep and not dep["available"]:
                cb.setEnabled(False)
            else:
                cb.stateChanged.connect(
                    lambda _state, mt=media_type: self._store_pending_change(mt)
                )

            row_layout.addWidget(cb)

            # Description
            desc_text = self.MEDIA_TYPE_DESCRIPTIONS.get(media_type, "")
            desc_lbl = QLabel(desc_text)
            desc_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 9pt;")
            row_layout.addWidget(desc_lbl, 1)

            root.addLayout(row_layout)

            # Dependency warning
            if dep and not dep["available"]:
                warn = QLabel(
                    f"\u26a0\ufe0f {dep['description']} (pip install {dep['package']})"
                )
                warn.setStyleSheet("color: #FFA500; font-size: 9pt; padding-left: 24px;")
                root.addWidget(warn)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(line)

        # Buttons
        root.addStretch()
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton(_("Cancel"))
        cancel_btn.clicked.connect(self.close)
        btn_row.addWidget(cancel_btn)

        apply_btn = QPushButton(_("Apply Changes"))
        apply_btn.clicked.connect(self._confirm_changes)
        btn_row.addWidget(apply_btn)

        root.addLayout(btn_row)

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_initial_value(media_type: CompareMediaType) -> bool:
        return {
            CompareMediaType.IMAGE: True,
            CompareMediaType.VIDEO: config.enable_videos,
            CompareMediaType.GIF: config.enable_gifs,
            CompareMediaType.PDF: config.enable_pdfs,
            CompareMediaType.SVG: config.enable_svgs,
            CompareMediaType.HTML: config.enable_html,
            CompareMediaType.AUDIO: config.enable_audio,
            CompareMediaType.UNCONFIGURED: False,
        }.get(media_type, False)

    def _store_pending_change(self, media_type: CompareMediaType) -> None:
        cb = self._checkboxes.get(media_type)
        if cb is not None:
            TypeConfigurationWindow._pending_changes[media_type] = cb.isChecked()

    @classmethod
    def _has_changes(cls) -> bool:
        for mt, new_val in cls._pending_changes.items():
            if mt not in cls._original_config or new_val != cls._original_config[mt]:
                return True
        return False

    def _confirm_changes(self) -> None:
        if not self._has_changes():
            self.close()
            return

        if self._app_actions.find_window_with_compare() is not None:
            ok = qt_alert(
                self,
                _("Confirm Changes"),
                _("This will clear all existing compares in open windows. Continue?"),
                kind="askokcancel",
            )
            if not ok:
                return

        self.apply_changes(self._app_actions)

    # ------------------------------------------------------------------
    # Apply (reused from original, kept here because it mutates config)
    # ------------------------------------------------------------------
    @classmethod
    def apply_changes(cls, app_actions=None) -> None:  # noqa: C901
        if cls._has_changes():
            cls.save_pending_changes()
        elif app_actions is None:
            return

        for media_type, enabled in cls._pending_changes.items():
            if media_type == CompareMediaType.VIDEO:
                config.enable_videos = enabled
                if enabled:
                    for ext in config.video_types:
                        if ext not in config.file_types:
                            config.file_types.append(ext)
                else:
                    config.file_types = [e for e in config.file_types if e not in config.video_types]
            elif media_type == CompareMediaType.GIF:
                config.enable_gifs = enabled
                if enabled and ".gif" not in config.file_types:
                    config.file_types.append(".gif")
                elif not enabled and ".gif" in config.file_types:
                    config.file_types.remove(".gif")
            elif media_type == CompareMediaType.PDF:
                config.enable_pdfs = enabled
                if enabled and ".pdf" not in config.file_types:
                    config.file_types.append(".pdf")
                elif not enabled and ".pdf" in config.file_types:
                    config.file_types.remove(".pdf")
            elif media_type == CompareMediaType.SVG:
                config.enable_svgs = enabled
                if enabled and ".svg" not in config.file_types:
                    config.file_types.append(".svg")
                elif not enabled and ".svg" in config.file_types:
                    config.file_types.remove(".svg")
            elif media_type == CompareMediaType.HTML:
                config.enable_html = enabled
                if enabled:
                    for ext in [".html", ".htm"]:
                        if ext not in config.file_types:
                            config.file_types.append(ext)
                else:
                    config.file_types = [e for e in config.file_types if e not in [".html", ".htm"]]
            elif media_type == CompareMediaType.AUDIO:
                config.enable_audio = enabled
                if enabled:
                    for ext in config.audio_types:
                        if ext not in config.file_types:
                            config.file_types.append(ext)
                else:
                    audio_set = set(config.audio_types)
                    config.file_types = [e for e in config.file_types if e not in audio_set]

        if app_actions is not None:
            app_actions.refresh_all_compares()
            app_actions.toast(_("Media type configuration updated"), time_in_seconds=5)
            cls.on_closing()
        else:
            cls._pending_changes.clear()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        TypeConfigurationWindow._instance = None
        super().closeEvent(event)

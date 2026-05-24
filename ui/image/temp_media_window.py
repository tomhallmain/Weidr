"""
Lightweight media viewer window (TempMediaWindow) used to display
temporary/generated files (rotated, cropped, enhanced, related, etc.).

Extracted from ui/image/media_details.py for clarity.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import QMenu, QVBoxLayout, QWidget

from lib.multi_display_qt import SmartWindow
from ui.app_window.media_frame import MediaFrame
from files.marked_files import MarkedFiles
from utils.config import config
from utils.translations import I18N
from utils.utils import Utils

_ = I18N._


class TempMediaWindow(SmartWindow):
    """Lightweight media viewer window.  Qt replacement for TempImageCanvas."""

    _instance: Optional[TempMediaWindow] = None

    def __init__(
        self,
        parent: QWidget,
        title: str,
        dimensions: str,
        app_actions,
    ) -> None:
        # Position at top of screen, offset slightly from parent
        parent_x = parent.pos().x() if parent is not None else 0
        geo = f"{dimensions}+{parent_x + 50}+0"
        super().__init__(
            persistent_parent=None,
            position_parent=parent,
            title=title,
            geometry=geo,
            auto_position=False,
            window_flags=Qt.WindowType.Window,
            respect_title_bar=True,
        )
        TempMediaWindow._instance = self
        self._app_actions = app_actions
        self._media_path: Optional[str] = None

        # -- layout --
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._media_frame = MediaFrame(self)
        self._media_frame.set_fill_canvas(False)
        layout.addWidget(self._media_frame)

        self._media_frame.seek_requested.connect(self._media_frame.video_seek_ms)
        self._media_frame.play_pause_requested.connect(self._media_frame.video_toggle_pause)
        self._media_frame.volume_requested.connect(self._media_frame.set_volume)
        self._media_frame.mute_requested.connect(self._media_frame.toggle_mute)

        self._bind_shortcuts()

    # -- shortcuts -------------------------------------------------
    def _bind_shortcuts(self) -> None:
        def sc(key: str, fn) -> None:
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)

        sc("Escape", lambda: self._app_actions.refocus())
        sc("Shift+Escape", self.close)
        sc(
            "Shift+D",
            lambda: self._app_actions.get_media_details(
                media_path=self._media_path
            ),
        )
        sc(
            "Shift+I",
            lambda: self._app_actions.run_image_generation(
                _type=None, media_path=self._media_path
            ),
        )
        sc(
            "Shift+Y",
            lambda: self._app_actions.set_marks_from_downstream_related_images(
                media_to_use=self._media_path
            ),
        )
        sc("Ctrl+M", lambda: self._open_move_marks_window())
        sc("Ctrl+K", lambda: self._open_move_marks_window(open_gui=False))
        sc("Ctrl+R", self._run_previous_marks_action)
        sc("Ctrl+E", self._run_penultimate_marks_action)
        sc("Shift+C", self._copy_file_to_base_dir)
        sc("Ctrl+C", self._copy_media_path)
        sc("Ctrl+T", self._run_permanent_marks_action)
        sc("Ctrl+W", self._new_full_window_with_media)

    # -- media display ---------------------------------------------
    def create_media(
        self, media_path: str, extra_text: str | None = None
    ) -> None:
        self._media_path = media_path
        self._media_frame.show_media(self._media_path)
        title = (
            media_path
            if extra_text is None
            else f"{media_path} - {extra_text}"
        )
        self.setWindowTitle(title)
        self.show()
        self.raise_()
        self.activateWindow()

    def clear_media(self) -> None:
        self._media_frame.clear()
        self._media_path = None
        self.setWindowTitle(
            _("Open a new related image with Shift+R on main window")
        )

    # -- guards ----------------------------------------------------
    def _require_media(self) -> bool:
        return (
            self._media_path is not None
            and os.path.isfile(self._media_path)
        )

    # -- mark actions ----------------------------------------------
    def _open_move_marks_window(self, open_gui: bool = True) -> None:
        if not self._require_media():
            return
        self._app_actions.open_move_marks_window(
            open_gui=open_gui, override_marks=[self._media_path]
        )
        self.clear_media()

    def _ensure_media_marked_for_quick_action(self) -> bool:
        """Ensure current media is in marks; return False only if transfer guard blocks."""
        if not MarkedFiles.guard_mark_mutation(self._app_actions, _("add mark")):
            return False
        MarkedFiles.add_mark_if_not_present(
            self._media_path, app_actions=self._app_actions
        )
        return True

    def _run_previous_marks_action(self) -> None:
        if not self._require_media():
            return
        if not self._ensure_media_marked_for_quick_action():
            return
        _, exceptions_present = MarkedFiles.run_previous_action(
            self._app_actions
        )
        if not exceptions_present:
            self.clear_media()

    def _run_penultimate_marks_action(self) -> None:
        if not self._require_media():
            return
        if not self._ensure_media_marked_for_quick_action():
            return
        _, exceptions_present = MarkedFiles.run_penultimate_action(
            self._app_actions
        )
        if not exceptions_present:
            self.clear_media()

    def _run_permanent_marks_action(self) -> None:
        if not self._require_media():
            return
        if not self._ensure_media_marked_for_quick_action():
            return
        _, exceptions_present = MarkedFiles.run_permanent_action(
            self._app_actions
        )
        if not exceptions_present:
            self.clear_media()

    # -- clipboard / file actions ----------------------------------
    def _copy_media_path(self) -> None:
        if not self._require_media():
            return
        filepath = str(self._media_path)
        if sys.platform == "win32":
            filepath = os.path.normpath(filepath)
            if config.escape_backslash_filepaths:
                filepath = filepath.replace("\\", "\\\\")
        QGuiApplication.clipboard().setText(filepath)
        self._app_actions.toast(_("Copied filepath to clipboard"))

    def _copy_file_to_base_dir(self) -> None:
        if not self._require_media():
            return
        base_dir = self._app_actions.get_base_dir()
        current_media_dir = os.path.dirname(self._media_path)
        if (
            base_dir
            and base_dir != ""
            and os.path.normpath(base_dir)
            != os.path.normpath(current_media_dir)
        ):
            new_file = os.path.join(
                base_dir, os.path.basename(self._media_path)
            )
            Utils.copy_file(
                self._media_path,
                new_file,
                overwrite_existing=config.move_marks_overwrite_existing_file,
            )

    def _new_full_window_with_media(self) -> None:
        if not self._require_media():
            return
        base_dir = os.path.dirname(self._media_path)
        self._app_actions.new_window(
            base_dir=base_dir, media_path=self._media_path
        )
        self.close()

    # -- right-click context menu ------------------------------------
    def contextMenuEvent(self, event) -> None:  # noqa: N802
        if not self._media_path:
            return
        menu = QMenu(self)
        menu.addAction(
            _("Open in Full Window"),
            self._new_full_window_with_media,
        )
        menu.addAction(
            _("Run Image Generation"),
            lambda: self._app_actions.run_image_generation(
                _type=None, media_path=self._media_path
            ),
        )
        menu.addSeparator()
        menu.addAction(
            _("Copy File Path"),
            self._copy_media_path,
        )
        menu.exec(event.globalPosition().toPoint())

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._media_frame.release_media()
        except Exception:
            pass
        super().closeEvent(event)

"""
RecentDirectoryWindow -- pick from recently used base directories (PySide6).

Uses ``RecentDirectories`` from ``files.recent_directories`` for history
and subclasses ``DirectoryPickerDialog`` for the scrollable list UI.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from files.recent_directories import RecentDirectories
from lib.fast_directory_picker_qt import get_existing_directory
from ui.files.directory_picker_dialog import DirectoryPickerDialog
from utils.app_actions import AppActions
from utils.translations import _
from utils.logging_setup import get_logger

logger = get_logger("recent_directory_window_qt")


class RecentDirectoryWindow(DirectoryPickerDialog):
    """
    Directory picker backed by ``RecentDirectories``.

    Supports:
      - Selecting a directory to set as base_dir in the current window
      - Opening a new compare window with a run_compare_media
      - Invoking a downstream_callback with extra args
      - Alt+Enter to select the penultimate history directory
      - Non-GUI mode (window hidden, immediate action via Return)
    """

    # Class-level history (matches original)
    last_set_directory: Optional[str] = None
    last_comparison_directory: Optional[str] = None

    MAX_HEIGHT = 900
    N_DIRECTORIES_CUTOFF = 30

    # ------------------------------------------------------------------
    # History helpers (static)
    # ------------------------------------------------------------------
    @staticmethod
    def get_history_directory(start_index: int = 0) -> Optional[str]:
        for i, d in enumerate(RecentDirectories.directory_history):
            if i >= start_index:
                return d
        return None

    @staticmethod
    def update_history(_dir: str) -> None:
        hist = RecentDirectories.directory_history
        if hist and hist[0] == _dir:
            return
        hist.insert(0, _dir)
        if len(hist) > RecentDirectories.MAX_RECENT_DIRECTORIES:
            del hist[-1]

    @staticmethod
    def get_geometry(is_gui: bool = True) -> str:
        if is_gui:
            return "800x800"
        return "300x100"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        app_master: QWidget,
        is_gui: bool,
        app_actions: AppActions,
        base_dir: str = ".",
        run_compare_media: Optional[str] = None,
        extra_callback_args: tuple = (None, None),
    ) -> None:
        self._is_gui = is_gui
        self._app_actions = app_actions
        self._base_dir = os.path.normpath(base_dir)
        self._run_compare_media = run_compare_media

        # Parse downstream callback
        if extra_callback_args is None or extra_callback_args[0] is None:
            self._downstream_callback = None
            self._callback_kwargs: dict[str, Any] = {}
            dirs_to_prepend: list[str] = []
        else:
            self._downstream_callback = extra_callback_args[0]
            dirs_to_prepend = extra_callback_args[1] if len(extra_callback_args) > 1 else []
            self._callback_kwargs = extra_callback_args[2] if len(extra_callback_args) > 2 else {}

        # Prepend requested directories
        for d in dirs_to_prepend:
            if d in RecentDirectories.directories:
                RecentDirectories.directories.remove(d)
        for d in sorted(dirs_to_prepend, reverse=True):
            RecentDirectories.directories.insert(0, d)

        super().__init__(
            parent=app_master,
            title=_("Set Image Comparison Directory"),
            geometry=self.get_geometry(is_gui=is_gui),
            position_parent=app_master,
        )

        if not is_gui:
            self.setWindowOpacity(0.3)

        # Alt+Enter → penultimate directory
        alt_ret = QShortcut(QKeySequence("Alt+Return"), self)
        alt_ret.activated.connect(self._select_penultimate)

    # ------------------------------------------------------------------
    # DirectoryPickerDialog hooks
    # ------------------------------------------------------------------
    def _get_all_directories(self) -> list[str]:
        return RecentDirectories.directories

    def _get_all_directories_copy(self) -> list[str]:
        return RecentDirectories.directories[:]

    def _on_directory_selected(self, directory: str) -> None:
        """Route the selection to the correct handler."""
        _dir = self._validate_and_add(directory)
        if _dir is None:
            return

        RecentDirectoryWindow.update_history(_dir)

        if self._downstream_callback is not None:
            self._downstream_callback(base_dir=_dir, **self._callback_kwargs)
            RecentDirectories.last_comparison_directory = _dir
        elif self._run_compare_media is None:
            self._app_actions.set_base_dir(base_dir_from_dir_window=_dir)
        elif self._run_compare_media == "":
            self._app_actions.new_window(base_dir=_dir)
        else:
            self._app_actions.new_window(
                base_dir=_dir,
                media_path=self._run_compare_media,
                do_search=True,
            )

        RecentDirectories.last_set_directory = _dir

    def _add_directory(self, directory: str) -> None:
        RecentDirectories.set_recent_directory(os.path.normpath(directory))

    def _clear_directories(self) -> None:
        RecentDirectories.directories.clear()

    def _remove_directory(self, directory: str) -> None:
        RecentDirectories.remove_directory(directory)

    def _browse_dialog_title(self) -> str:
        return _("Set image comparison directory")

    def _get_initial_browse_dir(self) -> str:
        dirs = RecentDirectories.directories
        if dirs and os.path.isdir(dirs[0]):
            return dirs[0]
        return self._base_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_and_add(self, directory: str) -> Optional[str]:
        """Validate the directory; prompt if invalid. Returns normalised path or None."""
        if directory and os.path.isdir(directory):
            RecentDirectories.set_recent_directory(directory)
            return os.path.normpath(directory)

        # Remove stale entry
        if directory in RecentDirectories.directories:
            RecentDirectories.directories.remove(directory)
        self._app_actions.toast(_("Invalid directory: {0}").format(directory))

        _dir = get_existing_directory(
            self,
            self._browse_dialog_title(),
            self._get_initial_browse_dir(),
        )
        if not _dir or not os.path.isdir(_dir):
            return None

        _dir = os.path.normpath(_dir)
        RecentDirectories.set_recent_directory(_dir)
        return _dir

    def _select_penultimate(self) -> None:
        penultimate = self.get_history_directory(start_index=1)
        if penultimate and os.path.isdir(penultimate):
            self._on_directory_selected(penultimate)
            self.close()

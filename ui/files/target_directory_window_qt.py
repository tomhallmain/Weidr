"""
TargetDirectoryWindow -- directory picker with its own recent list (PySide6).

Separate from ``RecentDirectories``.  Subclasses ``DirectoryPickerDialog``.
Non-UI data in ``files.target_directories.TargetDirectories``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtWidgets import QWidget

from files.target_directories import TargetDirectories
from ui.files.directory_picker_dialog import DirectoryPickerDialog
from utils.translations import _


class TargetDirectoryWindow(DirectoryPickerDialog):
    """
    Directory picker for related-file searches.

    Backed by ``TargetDirectories.recent_directories`` (persisted
    under a separate cache key from the main recent-directories list).
    Calls ``callback(directory)`` on selection.
    """

    # Delegate class-level state to the data module
    MAX_RECENT_DIRECTORIES = TargetDirectories.MAX_RECENT_DIRECTORIES
    RECENT_DIRECTORIES_KEY = TargetDirectories.RECENT_DIRECTORIES_KEY

    # ------------------------------------------------------------------
    # Persistence (delegate to data module)
    # ------------------------------------------------------------------
    @staticmethod
    def load_recent_directories() -> None:
        TargetDirectories.load_recent_directories()

    @staticmethod
    def save_recent_directories() -> None:
        TargetDirectories.save_recent_directories()

    @staticmethod
    def add_recent_directory(directory: str) -> None:
        TargetDirectories.add_recent_directory(directory)

    @staticmethod
    def get_geometry() -> str:
        return "800x800"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        master: QWidget,
        callback: Optional[Callable[[str], None]] = None,
        initial_dir: Optional[str] = None,
    ) -> None:
        # Ensure backing list is loaded
        TargetDirectories.load_recent_directories()

        self._callback = callback
        self._initial_dir = initial_dir

        super().__init__(
            parent=master,
            title=_("Select Target Directory"),
            geometry=self.get_geometry(),
            position_parent=master,
        )

    # ------------------------------------------------------------------
    # DirectoryPickerDialog hooks
    # ------------------------------------------------------------------
    def _get_all_directories(self) -> list[str]:
        return TargetDirectories.recent_directories

    def _get_all_directories_copy(self) -> list[str]:
        return TargetDirectories.recent_directories[:]

    def _on_directory_selected(self, directory: str) -> None:
        TargetDirectories.add_recent_directory(directory)
        if self._callback:
            self._callback(directory)

    def _add_directory(self, directory: str) -> None:
        TargetDirectories.add_recent_directory(os.path.normpath(directory))

    def _clear_directories(self) -> None:
        TargetDirectories.recent_directories.clear()
        TargetDirectories.save_recent_directories()

    def _remove_directory(self, directory: str) -> None:
        if directory in TargetDirectories.recent_directories:
            TargetDirectories.recent_directories.remove(directory)
            TargetDirectories.save_recent_directories()

    def _browse_dialog_title(self) -> str:
        return _("Select directory to search for related files")

    def _get_initial_browse_dir(self) -> str:
        if self._initial_dir and os.path.isdir(self._initial_dir):
            return self._initial_dir
        dirs = TargetDirectories.recent_directories
        if dirs and os.path.isdir(dirs[0]):
            return dirs[0]
        return os.path.expanduser("~")

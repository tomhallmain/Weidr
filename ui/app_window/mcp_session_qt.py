"""Adapts a live AppWindow to the session interface extensions/mcp_server.py
calls through.

Every method here is a thin call through the window's existing
``app_actions`` -- already wrapped onto the GUI thread by
``AppWindow._build_app_actions()`` -- so this file does no thread-marshaling
of its own. Kept out of extensions/mcp_server.py so that module stays free
of any Qt dependency; app_headless.py provides the other implementation of
the same session interface, against a Qt-free FileBrowser/CompareManager
pair instead of a live window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from compare.compare_args import CompareArgs
from utils.config import config
from utils.constants import CompareMode, Mode

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow


class QtWindowMCPSession:
    """Session backed by one open ``AppWindow``."""

    def __init__(self, window: "AppWindow"):
        self._window = window
        self._actions = window.app_actions

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def get_current_file(self) -> Optional[str]:
        return self._actions.get_active_media_filepath()

    def next_file(self) -> Optional[str]:
        self._actions.show_next_media()
        return self.get_current_file()

    def go_to_file(self, path: str) -> Optional[str]:
        found = self._actions.go_to_file(search_text=path)
        return self.get_current_file() if found else None

    def go_to_index(self, index: int) -> Optional[str]:
        found = self._actions.go_to_file_by_index(index)
        return self.get_current_file() if found else None

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------
    def get_base_dir(self) -> Optional[str]:
        return self._actions.get_base_dir()

    def set_base_dir(self, path: str) -> None:
        self._actions.set_base_dir(path)

    # ------------------------------------------------------------------
    # Marks / file operations
    # ------------------------------------------------------------------
    def list_marks(self) -> list:
        from files.marked_files import MarkedFiles
        return list(MarkedFiles.file_marks)

    def delete_file(self, path: str) -> None:
        self._actions.delete(path)

    def hide_current_file(self, path: Optional[str]) -> None:
        self._actions.hide_current_media(media_path=path)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    def run_compare(self, mode: str, find_duplicates: bool) -> None:
        try:
            compare_mode = CompareMode[mode]
        except KeyError:
            raise ValueError(f"unknown compare mode: {mode}")

        # COLOR_MATCHING reads compare_threshold as a LAB colour distance
        # rather than an embedding similarity -- see
        # scripts/agent_headless_demo.py's API notes on this same gotcha.
        threshold = (
            config.color_diff_threshold
            if compare_mode == CompareMode.COLOR_MATCHING
            else config.embedding_similarity_threshold
        )

        self._actions.run_compare(
            CompareArgs(
                base_dir=self.get_base_dir(),
                mode=Mode.GROUP,
                compare_mode=compare_mode,
                app_actions=self._actions,
                compare_threshold=threshold,
            ),
            find_duplicates=find_duplicates,
        )

    def is_compare_running(self) -> bool:
        return self._actions.is_compare_running()

    def get_compare_mode(self) -> str:
        return self._actions.get_compare_mode().name

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def run_image_generation(self, edit_suffix: Optional[str], target_dir: Optional[str]) -> None:
        self._actions.run_image_generation(
            edit_suffix=edit_suffix, suppress_toast=True, target_dir=target_dir,
        )

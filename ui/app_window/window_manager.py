"""
WindowManager -- singleton tracking all open AppWindow instances.

Extracted from the static/class-level methods and variables of the original
App class (add_secondary_window, get_open_windows, get_window,
find_window_with_compare, refresh_all_compares, cycle_windows,
open_secondary_compare_window, get_other_window_or_self_dir, etc.).

All methods are classmethods or staticmethods so that any controller
can call WindowManager.get_window() without needing an instance.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Optional

from utils.config import config
from utils.constants import Direction, Mode
from utils.logging_setup import get_logger
from utils.translations import I18N

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow

_ = I18N._
logger = get_logger("window_manager")


class WindowManager:
    """Singleton tracking all open AppWindow instances."""

    _windows: list[AppWindow] = []
    _secondary_toplevels: dict[int, object] = {}  # window_id -> SmartWindow (keep reference to avoid gc)
    _primary: Optional[AppWindow] = None
    _cycle_index: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @classmethod
    def set_primary(cls, window: AppWindow) -> None:
        """Register the primary (first) window."""
        cls._primary = window

    @classmethod
    def get_primary(cls) -> Optional[AppWindow]:
        return cls._primary

    @classmethod
    def register(cls, window: AppWindow) -> None:
        """Add a window to the tracking list."""
        if window not in cls._windows:
            cls._windows.append(window)

    @classmethod
    def unregister(cls, window: AppWindow) -> None:
        """Remove a window from the tracking list."""
        if window in cls._windows:
            cls._windows.remove(window)
        cls._secondary_toplevels.pop(window.window_id, None)
        from utils.notification_manager import notification_manager
        notification_manager.unregister_window(window.window_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @classmethod
    def get_open_windows(cls) -> list[AppWindow]:
        """Return the list of all open AppWindow instances."""
        return cls._windows[:]

    @classmethod
    def get_window_name(cls, window: AppWindow) -> str:
        """Return a display-friendly name for the given window."""
        if window.window_id == 0:
            return window.base_title + " " + _("(Main Window)")
        return window.base_title

    @classmethod
    def get_window(
        cls,
        window_id: Optional[int] = None,
        base_dir: Optional[str] = None,
        media_path: Optional[str] = None,
        refocus: bool = False,
        disallow_if_compare_state: bool = False,
        new_media: bool = False,
    ) -> Optional[AppWindow]:
        """Find an open window by id, base_dir, or image path."""
        from files.file_browser import SortBy

        for win in cls._windows:
            matched = False
            if window_id is not None and win.window_id == window_id:
                matched = True
            elif base_dir is not None and win.base_dir == base_dir:
                matched = True
            elif media_path is not None and win.media_path == media_path:
                matched = True

            if not matched:
                continue

            if media_path is not None:
                win.media_navigator.go_to_file(search_text=os.path.basename(media_path))

            if refocus:
                win.refocus()

            if disallow_if_compare_state and win.mode != Mode.BROWSE:
                logger.info(f"{cls.get_window_name(win)} has compare state, not returning")
                return None

            if new_media and win.mode == Mode.BROWSE and (
                win.file_browser.sort_by == SortBy.CREATION_TIME
                or win.file_browser.sort_by == SortBy.MODIFY_TIME
            ):
                win.direction = Direction.BACKWARD

            return win

        return None

    @classmethod
    def find_window_with_compare(
        cls, default_window: Optional[AppWindow] = None
    ) -> Optional[AppWindow]:
        """Find the first window that has an active compare."""
        for win in cls._windows:
            if win.compare_manager.has_compare():
                return win
        return default_window

    @classmethod
    def refresh_all_compares(cls) -> None:
        """Refresh all windows that have an active compare."""
        for win in cls._windows:
            if win.compare_manager.has_compare():
                win.search_ctrl.refresh_compare()

    # ------------------------------------------------------------------
    # Secondary windows
    # ------------------------------------------------------------------
    @classmethod
    def add_secondary_window(
        cls,
        base_dir: str,
        media_path: Optional[str] = None,
        do_search: bool = False,
        master: Optional[AppWindow] = None,
    ) -> None:
        """
        Open a new secondary AppWindow.

        Reuses an existing window for the same base_dir unless
        ``config.always_open_new_windows`` is set.
        """
        from ui.app_window.app_window import AppWindow

        # Reuse existing window for the same directory unless config says otherwise
        if not config.always_open_new_windows:
            for win in cls._windows:
                if win.base_dir == base_dir:
                    if media_path is not None and media_path != "":
                        if do_search:
                            win.sidebar_panel.search_media_path_box.setText(media_path)
                            win.search_ctrl.set_search()
                        else:
                            win.media_navigator.go_to_file(search_text=media_path)
                    win.raise_()
                    win.activateWindow()
                    return

        if do_search and (media_path is None or media_path == ""):
            do_search = False

        new_id = random.randint(1_000_000_000, 9_999_999_999)

        window = AppWindow(
            base_dir=base_dir,
            media_path=media_path,
            sidebar_visible=False,
            do_search=do_search,
            window_id=new_id,
        )
        # Staggered positioning: offset from the last secondary window
        cls._position_secondary(window)
        window.show()
        cls._secondary_toplevels[new_id] = window  # prevent gc
        logger.info(f"Opened secondary window id={new_id} for {base_dir}")

    @classmethod
    def _position_secondary(cls, window: AppWindow) -> None:
        """Position a new secondary window offset from the most recent one."""
        from PySide6.QtCore import QPoint

        # Find the last secondary window for staggering
        last_secondary = None
        for win in reversed(cls._windows):
            if win.is_secondary() and win.isVisible():
                last_secondary = win
                break

        if last_secondary is not None:
            pos = last_secondary.pos()
            window.move(pos + QPoint(30, 30))
        elif cls._primary is not None:
            pos = cls._primary.pos()
            window.move(pos + QPoint(50, 50))

        # Apply default secondary window size
        try:
            w, h = config.default_secondary_window_size.split("x")
            window.resize(int(w), int(h))
        except Exception:
            window.resize(600, 700)

    # ------------------------------------------------------------------
    # Window cycling
    # ------------------------------------------------------------------
    @classmethod
    def cycle_windows(cls, reverse: bool = False) -> None:
        """Cycle focus to the next (or previous) open window (round-robin)."""
        if len(cls._windows) <= 1:
            return

        n = len(cls._windows)
        step = -1 if reverse else 1

        if cls._cycle_index >= n:
            cls._cycle_index = 0

        target = cls._windows[cls._cycle_index]

        # If the target is the currently active window, advance once more
        if target.isActiveWindow() and n > 1:
            cls._cycle_index = (cls._cycle_index + step) % n
            target = cls._windows[cls._cycle_index]

        target.raise_()
        target.activateWindow()
        target.media_frame.setFocus()
        cls._cycle_index = (cls._cycle_index + step) % n

    # ------------------------------------------------------------------
    # Cross-window queries
    # ------------------------------------------------------------------
    @classmethod
    def get_other_window_or_self_dir(
        cls,
        requesting_window: AppWindow,
        allow_current_window: bool = False,
        prefer_compare_window: bool = False,
    ) -> tuple[Optional[AppWindow], list[str]]:
        """Find another open window or gather all secondary window directories."""
        from ui.files.recent_directory_window_qt import RecentDirectoryWindow

        if prefer_compare_window:
            win = cls.find_window_with_compare()
            if win is not None:
                return win, [win.base_dir]

        last_comp_dir = RecentDirectoryWindow.last_comparison_directory
        if last_comp_dir is not None and os.path.isdir(last_comp_dir):
            win = cls.get_window(base_dir=last_comp_dir)
            if win is not None:
                return win, [win.base_dir]
            else:
                RecentDirectoryWindow.last_comparison_directory = None

        if len(cls._secondary_toplevels) == 0:
            return requesting_window, [requesting_window.base_dir]

        result_window = None
        other_dirs: list[str] = []
        for win in cls._windows:
            if win is not None and (allow_current_window or win.window_id != requesting_window.window_id):
                if win.base_dir and os.path.isdir(win.base_dir):
                    result_window = win
                    other_dirs.append(win.base_dir)

        if len(other_dirs) == 1:
            return result_window, other_dirs
        return None, other_dirs

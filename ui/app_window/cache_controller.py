"""
CacheController -- persistence: loading and storing the app info cache.

Extracted from: load_info_cache, store_info_cache, apply_cached_display_position,
do_periodic_store_cache.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QTimer

from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import I18N

if TYPE_CHECKING:
    from files.file_browser import FileBrowser
    from ui.app_window.app_window import AppWindow

_ = I18N._
logger = get_logger("cache_controller")


class CacheController:
    """
    Owns persistence: loading and storing the application info cache,
    recent directories, marked file targets, and display position.
    Also owns the periodic file-check and cache-store timers.
    """

    def __init__(self, app_window: AppWindow, file_browser: FileBrowser):
        self._app = app_window
        self._fb = file_browser

        # Periodic timer (QTimer replaces start_thread + asyncio periodic)
        self._store_cache_timer: Optional[QTimer] = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_info_cache(self) -> Optional[str]:
        """
        Load cached application state. Returns the cached base directory
        if one exists, or None.

        Ported from App.load_info_cache -- calls the static loaders on
        every module that persists data through the app_info_cache.
        """
        try:
            from ui.files.marked_file_mover_qt import MarkedFiles
            from files.recent_directories import RecentDirectories
            from files.file_action import FileAction
            from ui.image.image_details_qt import ImageDetails
            from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
            from ui.files.favorites_window_qt import FavoritesWindow
            from ui.files.go_to_file_qt import GoToFile
            from ui.files.target_directory_window_qt import TargetDirectoryWindow

            MarkedFiles.load_target_dirs()
            RecentDirectories.load_recent_directories()
            FileAction.load_actions()
            ImageDetails.load_image_generation_mode()
            ClassifierManagementWindow.set_prevalidations()
            ClassifierManagementWindow.set_classifier_actions()
            from compare.classifier_actions_manager import ClassifierActionsManager

            ClassifierActionsManager.load_prevalidation_file_cache_from_disk()
            FavoritesWindow.load_favorites()
            GoToFile.load_persisted_data()
            TargetDirectoryWindow.load_recent_directories()

            return app_info_cache.get_meta("base_dir")
        except Exception as e:
            logger.error(f"Error loading info cache: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------
    def store_info_cache(self, store_window_state: bool = False) -> None:
        """
        Persist current application state to the info cache.

        Ported from App.store_info_cache.
        """
        from ui.files.marked_file_mover_qt import MarkedFiles
        from files.recent_directories import RecentDirectories
        from files.file_action import FileAction
        from ui.image.image_details_qt import ImageDetails
        from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
        from ui.files.favorites_window_qt import FavoritesWindow
        from ui.files.go_to_file_qt import GoToFile
        from ui.files.target_directory_window_qt import TargetDirectoryWindow
        from ui.app_window.window_manager import WindowManager
        from image.frame_cache import FrameCache

        base_dir = self._app.get_base_dir()
        if config.debug2:
            logger.debug(f"App info cache has_changes={app_info_cache.has_changes} and store will be attempted")

        if base_dir and base_dir != "":
            if not self._app.is_secondary():
                app_info_cache.set_meta("base_dir", base_dir)

            if self._app.img_path and self._app.img_path != "":
                cursor_media_path = self._app.img_path
                if os.path.dirname(cursor_media_path) != base_dir:
                    resolved_media_path = FrameCache.get_media_path_for_cached(cursor_media_path)
                    if resolved_media_path:
                        cursor_media_path = resolved_media_path
                app_info_cache.set(base_dir, "image_cursor", os.path.basename(cursor_media_path))

            app_info_cache.set(base_dir, "recursive", self._fb.is_recursive())
            app_info_cache.set(base_dir, "sort_by", self._fb.get_sort_by().get_text())
            app_info_cache.set(base_dir, "sort", self._fb.sort.get_text())
            app_info_cache.set(
                base_dir, "compare_mode",
                self._app.compare_manager.get_primary_mode_name(),
            )

        if store_window_state:
            secondary_base_dirs = []
            for win in WindowManager.get_open_windows():
                if win.is_secondary() and win.base_dir not in secondary_base_dirs:
                    secondary_base_dirs.append(win.base_dir)
            app_info_cache.set_meta("secondary_base_dirs", secondary_base_dirs)

            # Store main window display position and virtual screen info
            if not self._app.is_secondary():
                try:
                    app_info_cache.set_display_position(self._app)
                    app_info_cache.set_virtual_screen_info(self._app)
                except Exception as e:
                    logger.warning(f"Failed to store display position or virtual screen info: {e}")

        RecentDirectories.store_recent_directories()
        MarkedFiles.store_target_dirs()
        FileAction.store_actions()
        ImageDetails.store_image_generation_mode()
        ClassifierManagementWindow.store_prevalidations()
        ClassifierManagementWindow.store_classifier_actions()
        FavoritesWindow.store_favorites()
        GoToFile.save_persisted_data()
        TargetDirectoryWindow.save_recent_directories()
        from compare.classifier_actions_manager import ClassifierActionsManager

        ClassifierActionsManager.store_prevalidation_file_cache_to_disk()
        if app_info_cache.has_changes:
            logger.info("Storing app info cache")
            app_info_cache.store()

    # ------------------------------------------------------------------
    # Display position
    # ------------------------------------------------------------------
    def apply_cached_display_position(self) -> bool:
        """
        Restore the window geometry from the cached display position.
        Returns True if a position was applied.

        Ported from App.apply_cached_display_position.
        """
        try:
            position_data = app_info_cache.get_display_position()
            if not position_data:
                return False
            if not position_data.is_valid():
                logger.warning("Invalid cached display position data")
                return False
            virtual_info = app_info_cache.get_virtual_screen_info()
            if not position_data.is_visible_on_display(self._app, virtual_info):
                return False
            self._app.setGeometry(
                position_data.x,
                position_data.y,
                position_data.width,
                position_data.height,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to apply cached display position: {e}")
            return False

    # ------------------------------------------------------------------
    # Periodic cache store (replaces start_thread + async do_periodic_store_cache)
    # ------------------------------------------------------------------
    def start_periodic_store(self) -> None:
        """
        Start a periodic timer to store the cache at intervals.

        Replaces the async ``do_periodic_store_cache`` coroutine.
        """
        interval_ms = int(self._app.store_cache_config.interval_seconds * 1000)
        if interval_ms <= 0:
            return

        self._store_cache_timer = QTimer()
        self._store_cache_timer.timeout.connect(self._on_periodic_store)
        self._store_cache_timer.start(interval_ms)

    def stop_periodic_store(self) -> None:
        if self._store_cache_timer is not None:
            self._store_cache_timer.stop()
            self._store_cache_timer = None

    def _on_periodic_store(self) -> None:
        """
        Called on the main thread by QTimer.

        Ported from App.do_periodic_store_cache + App._store_info_cache_main_thread.
        """
        try:
            self.store_info_cache(store_window_state=True)
        except Exception as e:
            logger.debug(f"Error in periodic store info cache: {e}")

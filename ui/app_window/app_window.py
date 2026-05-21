"""
AppWindow -- main application window orchestrator (PySide6).

This is the thin shell described in APP_DECOMPOSITION.md. It owns the
top-level SmartMainWindow, instantiates all controller objects, assembles
the AppActions dict, and handles top-level lifecycle events.

All substantial logic lives in the controller modules:
    SidebarPanel, MediaNavigator, SearchController, FileMarksController,
    FileOpsController, WindowLauncher, KeyBindingManager, ContextMenuBuilder,
    NotificationController, WindowManager, CacheController.
"""

import os
import functools
import threading
import time
from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QThread, QMetaObject
from PySide6.QtWidgets import QApplication, QHBoxLayout, QSplitter, QStackedWidget, QWidget, QVBoxLayout, QFrame

from compare.compare_manager import CompareManager
from files.file_browser import FileBrowser
from files.marked_files import MarkedFiles
from lib.multi_display_qt import SmartMainWindow
from ui.app_style import AppStyle
from ui.custom_title_bar import FramelessWindowMixin, WindowResizeHandler
from ui.app_window.cache_controller import CacheController
from ui.app_window.context_menu_builder import ContextMenuBuilder
from ui.app_window.file_marks_controller import FileMarksController
from ui.app_window.file_ops_controller import FileOpsController
from ui.app_window.key_binding_manager import KeyBindingManager
from ui.app_window.masonry_browser import MasonryBrowser
from ui.app_window.media_frame import MediaFrame
from ui.app_window.media_navigator import MediaNavigator
from ui.app_window.notification_controller import NotificationController
from ui.app_window.search_controller import SearchController
from ui.app_window.sidebar_panel import SidebarPanel
from ui.app_window.window_launcher import WindowLauncher
from ui.app_window.window_manager import WindowManager
from utils.app_actions import AppActions
from utils.config import config, FileCheckConfig, SlideshowConfig, StoreCacheConfig
from utils.constants import Mode, Direction, ViewMode
from utils.logging_setup import get_logger
from utils.translations import I18N
from utils.utils import Utils

_ = I18N._
logger = get_logger("app_window")


class _MainThreadBridge(QWidget):
    """Marshals arbitrary callables from worker threads to the main/GUI thread.

    Uses ``QMetaObject.invokeMethod`` with ``BlockingQueuedConnection`` so that
    the calling (worker) thread blocks until the callable finishes on the main
    thread.  When already on the main thread the callable runs directly.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()  # invisible helper widget
        self._lock = threading.Lock()
        self._func = None
        self._args = ()
        self._kwargs = {}
        self._result = None
        self._error = None

    @Slot()
    def _execute(self):
        try:
            self._result = self._func(*self._args, **self._kwargs)
        except Exception as e:
            self._error = e

    def invoke(self, func, *args, **kwargs):
        """Call *func* on the main thread, blocking until it returns."""
        app = QApplication.instance()
        if app is None or QThread.currentThread() == app.thread():
            return func(*args, **kwargs)
        with self._lock:
            self._func = func
            self._args = args
            self._kwargs = kwargs
            self._result = None
            self._error = None
            QMetaObject.invokeMethod(
                self, "_execute", Qt.ConnectionType.BlockingQueuedConnection,
            )
            if self._error:
                raise self._error
            return self._result

    def wrap(self, func):
        """Return a wrapper that always invokes *func* on the main thread."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.invoke(func, *args, **kwargs)
        return wrapper


class AppWindow(FramelessWindowMixin, SmartMainWindow):
    """
    Main application window.

    Orchestrates controllers via composition. Each controller receives the
    dependencies it needs at construction time; the AppWindow itself keeps
    only cross-cutting state (mode, fullscreen flag, direction).

    Inherits FramelessWindowMixin for a custom draggable title bar, and
    SmartMainWindow for automatic geometry persistence.
    """

    # Signal for thread-safe title updates.
    # notification_manager fires threading.Timer callbacks that call
    # app_actions.title() from a background thread.  Using a Signal
    # ensures setWindowTitle always runs on the main / GUI thread.
    _sig_set_title = Signal(str)

    def __init__(
        self,
        base_dir: Optional[str] = None,
        media_path: Optional[str] = None,
        sidebar_visible: bool = config.sidebar_visible,
        do_search: bool = False,
        window_id: int = 0,
    ):
        super().__init__(restore_geometry=(window_id == 0))

        # Set up frameless window with custom title bar
        self.setup_frameless_window(
            title=_(" Weidr - Media Handler "), corner_radius=10
        )

        # Set icon in the custom title bar and connect context menu
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        icon_path = os.path.join(_root, "assets", "icon.png")
        if os.path.isfile(icon_path):
            title_bar = self.get_title_bar()
            if title_bar:
                title_bar.set_icon(icon_path)

        # Connect title bar right-click context menu
        title_bar = self.get_title_bar()
        if title_bar:
            title_bar.context_menu_requested.connect(self._show_title_bar_context_menu)

        # Thread-safe title update signal → slot
        self._sig_set_title.connect(self._on_set_title)

        self.window_id = window_id
        self.base_title = ""

        # ------------------------------------------------------------------
        # Core state
        # ------------------------------------------------------------------
        self.mode = Mode.BROWSE
        self.view_mode = ViewMode.FULL
        self.fullscreen = False
        self.delete_lock = False
        self.media_path: Optional[str] = None
        self.prev_media_path: Optional[str] = None
        self.search_dir: Optional[str] = None
        self.is_toggled_view_matches = True
        self.direction = Direction.FORWARD
        self.has_added_buttons_for_mode = {
            Mode.BROWSE: False,
            Mode.GROUP: False,
            Mode.SEARCH: False,
            Mode.DUPLICATES: False,
        }
        # Refresh guards:
        # - Prevent nested refresh execution.
        # - Briefly suppress timer-driven file-check refresh after a move refresh.
        self._is_refreshing = False
        self._suppress_file_check_refresh_until = 0.0
        self._incremental_status_timer: Optional[QTimer] = None
        self._base_dir_load_spinner_active = False
        self._startup_media_path: Optional[str] = media_path

        # ------------------------------------------------------------------
        # Backend (non-UI) objects -- shared across controllers
        # ------------------------------------------------------------------
        self.file_browser = FileBrowser(
            recursive=config.browse_recursive, sort_by=config.sort_by
        )
        self.compare_manager = CompareManager(master=self)
        self.file_check_config = FileCheckConfig(self.window_id)
        self.slideshow_config = SlideshowConfig(self.window_id)
        self.store_cache_config = StoreCacheConfig(self.window_id)

        # ------------------------------------------------------------------
        # Window title
        # ------------------------------------------------------------------
        self.setWindowTitle(_(" Weidr - Media Handler "))

        # ------------------------------------------------------------------
        # Central widget: frameless structure with custom title bar
        # ------------------------------------------------------------------
        grip_size = getattr(self, '_frameless_grip_size', 8)

        # Outer widget for translucent background (needed for rounded corners)
        outer_widget = QWidget()
        outer_widget.setObjectName("transparentOuter")
        outer_widget.setAttribute(Qt.WA_TranslucentBackground)
        self.setCentralWidget(outer_widget)
        outer_layout = QVBoxLayout(outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Main container frame with rounded corners
        self._main_frame = QFrame()
        self._main_frame.setObjectName("mainFrame")
        outer_layout.addWidget(self._main_frame)

        root_layout = QVBoxLayout(self._main_frame)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Custom title bar at the top
        title_bar = self.get_title_bar()
        if title_bar:
            root_layout.addWidget(title_bar)

        # Content area below title bar
        content_widget = QWidget()
        content_widget.setObjectName("contentArea")
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        root_layout.addWidget(content_widget)

        # Splitter inside content: sidebar | media frame
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        content_layout.addWidget(self.splitter)

        # Sidebar panel (left)
        self.sidebar_panel = SidebarPanel(parent=self, app_window=self)
        self.splitter.addWidget(self.sidebar_panel)

        # Right-side stack: full viewer (index 0) or masonry grid (index 1)
        self._media_stack = QStackedWidget()
        self.splitter.addWidget(self._media_stack)

        self.media_frame = MediaFrame(parent=self)
        self._media_stack.addWidget(self.media_frame)   # index 0 — ViewMode.FULL
        self.media_frame.seek_requested.connect(self.seek_media_position)
        self.media_frame.play_pause_requested.connect(self.toggle_media_play_pause)
        self.media_frame.volume_requested.connect(self.set_media_volume)
        self.media_frame.mute_requested.connect(self.toggle_media_mute)

        self.masonry_browser = MasonryBrowser(parent=self)
        self._media_stack.addWidget(self.masonry_browser)  # index 1 — ViewMode.MASONRY
        self.masonry_browser.tile_activated.connect(self._on_masonry_tile_activated)

        # Give most space to the media area
        self.splitter.setStretchFactor(0, 1)  # sidebar
        self.splitter.setStretchFactor(1, 9)  # media stack

        if not sidebar_visible:
            self.sidebar_panel.setVisible(False)

        # Install resize handler for frameless edge resizing
        self._resize_handler = WindowResizeHandler(self, grip_size)

        # Apply combined stylesheet (base + frameless)
        self._apply_theme()

        # ------------------------------------------------------------------
        # Controllers
        # ------------------------------------------------------------------
        self.notification_ctrl = NotificationController(app_window=self)
        self.notification_ctrl.set_loading_spinner(
            self.sidebar_panel.loading_spinner
        )

        self.cache_ctrl = CacheController(
            app_window=self,
            file_browser=self.file_browser,
        )

        self.media_navigator = MediaNavigator(
            app_window=self,
            file_browser=self.file_browser,
            compare_manager=self.compare_manager,
            media_frame=self.media_frame,
        )

        self.search_ctrl = SearchController(
            app_window=self,
            file_browser=self.file_browser,
            compare_manager=self.compare_manager,
            sidebar_panel=self.sidebar_panel,
        )

        self.file_marks_ctrl = FileMarksController(
            app_window=self,
            file_browser=self.file_browser,
            compare_manager=self.compare_manager,
            media_navigator=self.media_navigator,
        )

        self.file_ops_ctrl = FileOpsController(
            app_window=self,
            file_browser=self.file_browser,
            compare_manager=self.compare_manager,
            media_navigator=self.media_navigator,
        )

        self.window_launcher = WindowLauncher(
            app_window=self,
        )

        # ------------------------------------------------------------------
        # Thread-safety bridge (compare engine calls app_actions from a
        # worker thread; the bridge marshals those calls to the main thread)
        # ------------------------------------------------------------------
        self._thread_bridge = _MainThreadBridge(parent=self)

        # ------------------------------------------------------------------
        # Assemble AppActions dict
        # ------------------------------------------------------------------
        self.app_actions = self._build_app_actions()

        # CompareManager was created early (without app_actions) so
        # controllers could receive it.  Now wire in the real values.
        self.compare_manager.set_app_actions(self, self.app_actions)

        # ------------------------------------------------------------------
        # Key bindings & context menu (need app_actions / controllers ready)
        # ------------------------------------------------------------------
        self.key_binding_mgr = KeyBindingManager(app_window=self)
        self.context_menu_builder = ContextMenuBuilder(app_window=self)

        # ------------------------------------------------------------------
        # Register with WindowManager
        # ------------------------------------------------------------------
        if window_id == 0:
            WindowManager.set_primary(self)
        WindowManager.register(self)

        # ------------------------------------------------------------------
        # Notification manager callback
        # ------------------------------------------------------------------
        from utils.notification_manager import notification_manager
        notification_manager.set_app_actions(self.app_actions, self.window_id)

        # ------------------------------------------------------------------
        # Load cache, apply initial base directory
        # ------------------------------------------------------------------
        if not self.is_secondary():
            initial_base_dir = self.cache_ctrl.load_info_cache()
            if base_dir:
                initial_base_dir = base_dir
            if initial_base_dir:
                QTimer.singleShot(0, lambda bd=initial_base_dir: self.set_base_dir(bd))
        elif base_dir:
            QTimer.singleShot(0, lambda bd=base_dir: self.set_base_dir(bd))

        # Restore window geometry (SmartMainWindow feature)
        self.restore_window_geometry()
        QTimer.singleShot(0, self._apply_default_sidebar_width)

        # ------------------------------------------------------------------
        # Start periodic timers (replaces start_thread + async)
        # ------------------------------------------------------------------
        self.file_ops_ctrl.start_file_check_timer()
        if not self.is_secondary():
            self.cache_ctrl.start_periodic_store()
            # Re-open secondary windows that were open last session
            QTimer.singleShot(100, self._restore_secondary_windows)

        # Handle initial media_path / do_search
        if media_path is not None:
            if do_search:
                QTimer.singleShot(200, lambda: self.search_ctrl.set_search())
            else:
                QTimer.singleShot(200, lambda ip=media_path: self.media_navigator.go_to_file(search_text=ip))

        if self.is_secondary():
            QTimer.singleShot(300, lambda: self.cache_ctrl.store_info_cache())

        logger.info(f"AppWindow created (id={window_id})")

    def _restore_secondary_windows(self) -> None:
        """Re-open secondary windows that were open in the previous session."""
        from utils.app_info_cache import app_info_cache
        for _dir in app_info_cache.get_meta("secondary_base_dirs", default_val=[]):
            WindowManager.add_secondary_window(_dir)
        # Re-focus the primary after all secondaries have been opened
        QTimer.singleShot(50, self._refocus_primary)

    def _refocus_primary(self) -> None:
        """Raise and focus the primary window."""
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------------
    # AppActions assembly
    # ------------------------------------------------------------------
    def _build_app_actions(self) -> AppActions:
        """Wire the AppActions dict, mapping action names to controller methods.

        Actions that touch the Qt GUI are wrapped via :class:`_MainThreadBridge`
        so that the compare engine (which runs on a worker thread) can call them
        safely.  Pure-data / thread-safe getters are left unwrapped.
        """
        ts = self._thread_bridge.wrap  # shorthand

        actions = {
            # Window title -- thread-safe via Signal so that
            # notification_manager's Timer thread never touches Qt directly
            "title": self._sig_set_title.emit,
            # Window management (static)
            "new_window": ts(WindowManager.add_secondary_window),
            "get_window": WindowManager.get_window,
            "get_open_windows": WindowManager.get_open_windows,
            "refresh_all_compares": ts(WindowManager.refresh_all_compares),
            "find_window_with_compare": WindowManager.find_window_with_compare,
            # Notifications
            "toast": ts(self.notification_ctrl.toast),
            "title_notify": ts(self.notification_ctrl.title_notify),
            "_alert": ts(self.notification_ctrl.alert),
            "start_loading_spinner": self.notification_ctrl.start_loading_spinner,
            "stop_loading_spinner": self.notification_ctrl.stop_loading_spinner,
            # Navigation / display
            "refresh": ts(self.refresh),
            "refocus": ts(self.refocus),
            "set_mode": ts(self.set_mode),
            "is_fullscreen": lambda: self.fullscreen,
            "get_active_media_filepath": self.media_navigator.get_active_media_filepath,
            "create_media": ts(self.media_navigator.create_media),
            "show_next_media": ts(self.media_navigator.show_next_media),
            "play_media": ts(self.play_media),
            "pause_media": ts(self.pause_media),
            "toggle_media_play_pause": ts(self.toggle_media_play_pause),
            "seek_media": ts(self.seek_media_position),
            "stop_media": ts(self.stop_media),
            "set_media_volume": ts(self.set_media_volume),
            "get_media_volume": self.get_media_volume,
            "toggle_media_mute": ts(self.toggle_media_mute),
            "set_media_mute": ts(self.set_media_mute),
            "is_media_muted": self.is_media_muted,
            # Window launchers
            "get_media_details": ts(self.window_launcher.open_media_details),
            "open_move_marks_window": ts(self.file_marks_ctrl.open_move_marks_window),
            "open_password_admin_window": ts(self.window_launcher.open_password_admin_window),
            # Search / compare
            "run_image_generation": ts(self.search_ctrl.run_image_generation),
            "set_marks_from_downstream_related_images": ts(self.file_marks_ctrl.set_marks_from_downstream_related_images),
            # File navigation
            "go_to_file": ts(self.media_navigator.go_to_file),
            "go_to_file_by_index": ts(self.media_navigator.go_to_file_by_index),
            "set_base_dir": ts(self.set_base_dir),
            "get_base_dir": self.get_base_dir,
            # File operations
            "delete": ts(self.file_ops_ctrl.handle_delete),
            "hide_current_media": ts(self.file_ops_ctrl.hide_current_media),
            "copy_media_path": ts(self.file_ops_ctrl.copy_media_path),
            "release_media_canvas": ts(self.release_media_canvas),
            # Persistence
            "store_info_cache": self.cache_ctrl.store_info_cache,
            # Internal (prefixed with _)
            "_set_toggled_view_matches": ts(self.media_navigator.set_toggled_view_matches),
            "_set_label_state": ts(self.notification_ctrl.set_label_state),
            "_add_buttons_for_mode": ts(self.sidebar_panel.add_buttons_for_mode),
        }
        return AppActions(actions=actions, master=self)

    # ------------------------------------------------------------------
    # Title bar / theme helpers
    # ------------------------------------------------------------------
    def setWindowTitle(self, title: str) -> None:
        """Override to keep the custom title bar text in sync."""
        super().setWindowTitle(title)
        title_bar = self.get_title_bar()
        if title_bar:
            title_bar.set_title(title)

    def _on_set_title(self, title: str) -> None:
        """Slot for :pyattr:`_sig_set_title` -- always runs on the GUI thread.

        Calls processEvents() so that the title-bar repaint happens
        immediately, even when the main thread is about to block on a
        long-running operation (e.g. TensorFlow model loading).

        WARNING: this method must never be called directly from a worker thread.
        Always route title updates via ``_sig_set_title.emit(...)`` so Qt can
        marshal execution to the main thread safely.
        """
        self.setWindowTitle(title)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def _apply_theme(self) -> None:
        """Apply the combined base + frameless stylesheet and title bar theme."""
        is_dark = AppStyle.IS_DEFAULT_THEME
        stylesheet = AppStyle.get_stylesheet() + AppStyle.get_frameless_stylesheet(is_dark)
        self.setStyleSheet(stylesheet)
        self.media_frame.set_background_color(
            AppStyle.MEDIA_BG if is_dark else AppStyle.LIGHT_MEDIA_BG
        )
        self.apply_frameless_theme(is_dark)

    def _apply_default_sidebar_width(self) -> None:
        """
        Apply a smaller default sidebar width.

        Uses DPI scaling and config.font_size as a fallback user scaling
        preference so translated labels still fit at larger UI scales.
        """
        if not self.sidebar_panel.isVisible():
            return

        total_width = self.splitter.width()
        if total_width <= 1:
            total_width = self.width()
        if total_width <= 1:
            return

        dpi_scale = max(1.0, float(self.logicalDpiX()) / 96.0)
        font_scale = max(1.0, float(getattr(config, "font_size", 8)) / 8.0)
        ui_scale = max(dpi_scale, font_scale)

        preferred_sidebar_width = int(round(280 * ui_scale))
        min_sidebar_width = int(round(180 * ui_scale))
        max_sidebar_width = int(total_width * 0.45)

        sidebar_width = max(
            min_sidebar_width,
            min(preferred_sidebar_width, max_sidebar_width),
        )
        media_width = max(1, total_width - sidebar_width)
        self.splitter.setSizes([sidebar_width, media_width])

    # ------------------------------------------------------------------
    # Title bar directory color
    # ------------------------------------------------------------------
    def _apply_directory_title_bar_color(self, directory: str) -> None:
        """Apply any stored custom title bar color for the given directory."""
        from utils.app_info_cache import app_info_cache
        title_bar = self.get_title_bar()
        if not title_bar:
            return
        color = app_info_cache.get_directory_color(directory)
        if color:
            title_bar.set_custom_bg_color(color)
        else:
            title_bar.clear_custom_bg_color()

    def _show_title_bar_context_menu(self, global_pos) -> None:
        """Show a context menu specific to the title bar."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)

        menu.addAction(
            _("Set Title Bar Color..."),
            self._set_title_bar_color_for_directory,
        )

        # Only show the clear option if a custom color is currently set
        from utils.app_info_cache import app_info_cache
        current_color = app_info_cache.get_directory_color(self.base_dir) if self.base_dir else None
        if current_color:
            menu.addAction(
                _("Clear Title Bar Color"),
                self._clear_title_bar_color_for_directory,
            )

        menu.exec(global_pos)

    def _set_title_bar_color_for_directory(self) -> None:
        """Open a color picker and set the title bar color for the current directory."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        from utils.app_info_cache import app_info_cache

        if not self.base_dir:
            self.notification_ctrl.toast(_("No directory loaded."))
            return

        # Use the current custom color (or the theme background) as the initial color
        current_hex = app_info_cache.get_directory_color(self.base_dir)
        initial_color = QColor(current_hex) if current_hex else QColor(AppStyle.BG_COLOR)

        color = QColorDialog.getColor(initial_color, self, _("Choose Title Bar Color"))
        if color.isValid():
            color_hex = color.name()  # "#rrggbb"
            app_info_cache.set_directory_color(self.base_dir, color_hex)
            title_bar = self.get_title_bar()
            if title_bar:
                title_bar.set_custom_bg_color(color_hex)
            self.notification_ctrl.toast(
                _("Title bar color set for this directory.")
            )

    def _clear_title_bar_color_for_directory(self) -> None:
        """Clear the custom title bar color for the current directory."""
        from utils.app_info_cache import app_info_cache

        if not self.base_dir:
            return

        app_info_cache.set_directory_color(self.base_dir, None)
        title_bar = self.get_title_bar()
        if title_bar:
            title_bar.clear_custom_bg_color()
        self.notification_ctrl.toast(
            _("Title bar color cleared for this directory.")
        )

    # ------------------------------------------------------------------
    # Properties / simple accessors
    # ------------------------------------------------------------------
    def is_secondary(self) -> bool:
        return self.window_id != 0

    @property
    def base_dir(self) -> str:
        return self.file_browser.directory if self.file_browser else ""

    def get_base_dir(self) -> str:
        return self.base_dir

    def get_search_dir(self) -> str:
        """Return the search directory, falling back to base_dir."""
        return self.get_base_dir() if self.search_dir is None else self.search_dir

    # ------------------------------------------------------------------
    # Top-level actions (kept on AppWindow per decomposition doc)
    # ------------------------------------------------------------------
    def set_base_dir(self, base_dir_from_dir_window: Optional[str] = None) -> None:
        """
        Set the base directory and reload files.

        If *base_dir_from_dir_window* is provided it is used directly; otherwise
        the sidebar entry is read, and if it is empty the recent-directory window
        is opened.
        """
        from files.recent_directories import RecentDirectories
        from utils.app_info_cache import app_info_cache as base_cache
        from utils.constants import CompareMode

        self.cache_ctrl.store_info_cache()

        dir_from_sidebar_entry = False
        if base_dir_from_dir_window is not None:
            new_dir = base_dir_from_dir_window
        else:
            entry_text = self.sidebar_panel.set_base_dir_box.text().strip()
            if not entry_text or entry_text == _("Enter base directory...") or entry_text == self.base_dir:
                if len(RecentDirectories.directories) == 0:
                    from lib.fast_directory_picker_qt import get_existing_directory

                    chosen = get_existing_directory(
                        self, _("Set media comparison directory"), self.get_base_dir()
                    )
                    if not chosen:
                        return
                    new_dir = chosen
                    RecentDirectories.directories.append(new_dir)
                else:
                    self.window_launcher.open_recent_directory_window()
                    return
            else:
                new_dir = entry_text
                dir_from_sidebar_entry = True

        if not new_dir or not os.path.isdir(new_dir):
            if dir_from_sidebar_entry:
                self.app_actions.warn(
                    _("Path is not an existing directory:\n{0}").format(new_dir),
                    time_in_seconds=12,
                )
            return

        # Check for large directory before loading
        if self._check_large_directory_before_load(new_dir):
            return  # user cancelled

        # From here: no more early returns. Hide masonry (if active) and start the
        # spinner together — both are suppressed during dialogs and cancelled paths.
        if self.view_mode == ViewMode.MASONRY:
            self._media_stack.setCurrentIndex(0)
        self._start_base_dir_load_spinner()
        try:
            # Restore per-directory settings from cache (read before compare/directory work)
            recursive = base_cache.get(new_dir, "recursive", default_val=False)
            sort_by_text = base_cache.get(new_dir, "sort_by", default_val=None)
            sort_text = base_cache.get(new_dir, "sort", default_val=None)
            compare_mode_text = base_cache.get(new_dir, "compare_mode", default_val=None)

            # Clear stale compare if the directory changed
            if self.compare_manager.has_compare() and new_dir != self.compare_manager.compare().base_dir:
                self.compare_manager.clear_compare()
                self.notification_ctrl.set_label_state(group_number=None, size=0)
                self.sidebar_panel.remove_all_mode_buttons()

            if recursive != self.file_browser.recursive:
                self.file_browser.set_recursive(recursive)
                self.sidebar_panel.recursive_check.setChecked(recursive)

            if sort_by_text:
                try:
                    from files.file_browser import SortBy
                    sb = SortBy.get(sort_by_text) if isinstance(sort_by_text, str) else sort_by_text
                    self.file_browser.sort_by = sb
                    self.sidebar_panel.set_sort_by_value(sb.get_text())
                    self.sidebar_panel.sort_direction_choice.setEnabled(sb != SortBy.RANDOMIZE)
                except Exception as e:
                    logger.error(f"Error setting stored sort by: {e}")

            if sort_text:
                try:
                    from utils.constants import Sort
                    sort = Sort.DESC if sort_text == Sort.DESC.get_text() else Sort.ASC
                    self.file_browser.sort = sort
                    self.sidebar_panel.set_sort_value(sort.get_text())
                except Exception as e:
                    logger.error(f"Error setting stored sort direction: {e}")

            # Apply directory to file browser
            self.sidebar_panel.update_base_dir_display(new_dir)
            previous_file = base_cache.get(new_dir, "file_cursor") or base_cache.get(new_dir, "image_cursor")
            previous_file_path = (
                Utils.get_valid_file(new_dir, previous_file)
                if previous_file and previous_file != ""
                else None
            )
            preferred_initial_file = None
            if (
                self._startup_media_path
                and isinstance(self._startup_media_path, str)
                and os.path.isfile(self._startup_media_path)
                and os.path.dirname(self._startup_media_path) == new_dir
            ):
                preferred_initial_file = self._startup_media_path
            elif previous_file_path and os.path.isfile(previous_file_path):
                # Seed incremental loading with the cached media when available.
                preferred_initial_file = previous_file_path

            self.file_browser.set_directory_with_preferred_file(
                new_dir, preferred_file=preferred_initial_file
            )
            self._start_incremental_status_updates()
        except Exception:
            self._stop_base_dir_load_spinner_if_active()
            raise
        finally:
            if not self.file_browser.is_incremental_loading:
                self._stop_base_dir_load_spinner_if_active()

        if compare_mode_text:
            try:
                mode_to_set = CompareMode.get(compare_mode_text) if isinstance(compare_mode_text, str) else compare_mode_text
                if mode_to_set != self.compare_manager.compare_mode:
                    self.compare_manager.set_compare_mode(mode_to_set)
            except Exception as e:
                logger.error(f"Error setting stored compare mode: {e}")

        if self.compare_manager.compare_mode is None:
            self.compare_manager.set_compare_mode(CompareMode.CLIP_EMBEDDING)

        # Navigate to the previously-viewed file, or show the first
        if not self.compare_manager.has_compare():
            self.set_mode(Mode.BROWSE)
            if self.file_browser.is_incremental_loading:
                progress_text = self.file_browser.get_incremental_progress_text()
                if progress_text:
                    self.notification_ctrl.set_label_state(text=progress_text)
            else:
                if previous_file and previous_file != "":
                    if not self.media_navigator.go_to_file(search_text=previous_file, retry_with_delay=1):
                        self.media_navigator.show_next_media()
                else:
                    self.media_navigator.show_next_media()
                self.notification_ctrl.set_label_state()
                self._refresh_masonry_if_active()
            self._sync_media_empty_directory_message()

        self.setWindowTitle(self.get_title_from_base_dir(overwrite=True))
        self._apply_directory_title_bar_color(new_dir)
        RecentDirectories.set_recent_directory(new_dir)

        # Return focus to the main canvas so global shortcuts work without Escape
        # after navigating via the base-directory entry field.
        if QApplication.focusWidget() is self.sidebar_panel.set_base_dir_box:
            self.refocus()

    def _start_base_dir_load_spinner(self) -> None:
        """Show the sidebar loading spinner for base-directory scan/load."""
        self._base_dir_load_spinner_active = True
        self.notification_ctrl.start_loading_spinner(force=True)

    def _stop_base_dir_load_spinner_if_active(self) -> None:
        """Hide the base-dir load spinner if this window started it."""
        if self._base_dir_load_spinner_active:
            self._base_dir_load_spinner_active = False
            self.notification_ctrl.stop_loading_spinner()

    def _start_incremental_status_updates(self) -> None:
        if not self.file_browser.is_incremental_loading:
            return
        if self._incremental_status_timer is not None:
            self._incremental_status_timer.stop()
        self._incremental_status_timer = QTimer(self)
        self._incremental_status_timer.timeout.connect(self._on_incremental_status_tick)
        self._incremental_status_timer.start(1000)
        self._on_incremental_status_tick()

    def _on_incremental_status_tick(self) -> None:
        if not self.file_browser.is_incremental_loading:
            if self._incremental_status_timer is not None:
                self._incremental_status_timer.stop()
                self._incremental_status_timer = None
            self._stop_base_dir_load_spinner_if_active()
            self.notification_ctrl.set_status_title(None)
            self.notification_ctrl.set_label_state()
            if self.mode == Mode.BROWSE and self.media_path is None and self.file_browser.has_files():
                self.media_navigator.show_next_media()
            self._sync_media_empty_directory_message()
            self._refresh_masonry_if_active()
            return
        if self.mode == Mode.BROWSE and self.media_path is None and self.file_browser.has_files():
            # Show the first available item as soon as batches arrive.
            self.media_navigator.show_next_media()
        progress_text = self.file_browser.get_incremental_progress_text()
        if progress_text:
            self.notification_ctrl.set_label_state(text=progress_text)
            if not self.sidebar_panel.isVisible():
                self.notification_ctrl.set_status_title(progress_text)

    def get_title_from_base_dir(self, overwrite: bool = False) -> str:
        """Generate the window title from the current base directory."""
        if overwrite:
            relative_dirpath = Utils.get_relative_dirpath(self.base_dir, levels=2)
            self.base_title = _(" Weidr - Media Handler ") + "- " + relative_dirpath
        return self.base_title

    def _check_large_directory_before_load(self, base_dir: str, threshold: int = 5000) -> bool:
        """
        Check if a directory has many files before loading.
        Returns True if the user cancels, False to proceed.
        """
        if self.file_browser.has_confirmed_dir() and self.file_browser.directory == base_dir:
            return False

        # Snapshot all mutable state that _gather_files touches
        original_directory = self.file_browser.directory
        original_files = self.file_browser._files.copy()
        original_filepaths = self.file_browser.filepaths.copy()

        try:
            self.file_browser.directory = base_dir
            # Must match drive to *base_dir*: is_slow_total_files() uses a 5× factor for
            # "external" paths; leaving the flag from a previous directory mis-classifies
            # e.g. C:\ after browsing D:\ and triggers the large-dir prompt far below threshold.
            self.file_browser._update_drive_characteristics()
            if self.file_browser.has_confirmed_dir():
                return False
            self.file_browser._gather_files()
            if self.file_browser.is_slow_total_files(threshold=threshold):
                ok = self.notification_ctrl.alert(
                    _("Large Directory Detected: {0}").format(base_dir),
                    _("A directory with a large number of files or a directory with a fair-sized number "
                      "of files on external storage was detected, which may result in a slow load time. "
                      "Please confirm you want to load the full directory."),
                    kind="askokcancel",
                )
                if not ok:
                    logger.info(f"User canceled loading large directory: {base_dir}")
                    return True
                self.file_browser.set_dir_confirmed()
        finally:
            # Restore ALL mutated state so the file browser is unchanged
            self.file_browser.directory = original_directory
            self.file_browser._files = original_files
            self.file_browser.filepaths = original_filepaths
            self.file_browser._update_drive_characteristics()
        return False

    def set_mode(self, mode: Mode, do_update: bool = True) -> None:
        """Change the current application mode."""
        self.mode = mode
        self.sidebar_panel.set_mode_label(mode.get_text())
        if mode != Mode.SEARCH:
            self.sidebar_panel.remove_search_mode_buttons()
        if mode not in (Mode.GROUP, Mode.DUPLICATES):
            self.sidebar_panel.remove_group_mode_buttons()

    def _sync_media_empty_directory_message(self) -> None:
        """
        Show an empty-directory message only when BROWSE has zero displayable files.
        """
        if self.mode != Mode.BROWSE:
            self.media_frame.hide_empty_directory_message()
            return
        if self.file_browser.count() == 0:
            self.media_frame.show_empty_directory_message()
        else:
            self.media_frame.hide_empty_directory_message()

    def return_to_browsing_mode(self, event=None, suppress_toast: bool = False) -> None:
        """Return to browsing mode, clearing compare state."""
        self.set_mode(Mode.BROWSE)
        self.file_browser.refresh()
        self.notification_ctrl.set_label_state()
        if self.media_path is not None:
            if not self.media_navigator.go_to_file(search_text=self.media_path, retry_with_delay=1):
                self.media_navigator.home()
        self._sync_media_empty_directory_message()
        self.cache_ctrl.store_info_cache()
        if not suppress_toast:
            self.notification_ctrl.toast(_("Browsing mode set."))

    def check_many_files(
        self,
        window: "AppWindow",
        action: str = "do this action",
        threshold: int = 2000,
    ) -> bool:
        """
        Check if a window has too many files and prompt the user.

        Returns True if user cancels, False if user wants to proceed.
        """
        if not window.file_browser.has_confirmed_dir() and window.file_browser.is_slow_total_files(threshold=threshold):
            ok = self.notification_ctrl.alert(
                _("Many Files"),
                f"There are a lot of files in {window.base_dir} and it may take a while"
                f" to {action}.\n\nWould you like to proceed?",
                kind="askokcancel",
            )
            if not ok:
                return True
            window.file_browser.set_dir_confirmed()
        return False

    def refresh(
        self,
        show_new_media: bool = False,
        refresh_cursor: bool = False,
        file_check: bool = True,
        removed_files: Optional[list] = None,
        from_file_check: bool = False,
        force: bool = False,
    ) -> None:
        """
        Refresh the file list and update the display.

        When *force* is True, run a full rescan even while incremental directory
        loading is in progress (used after bulk operations such as directory-wide
        prevalidations).
        """
        if removed_files is None:
            removed_files = []

        if self.file_browser.is_incremental_loading and not force:
            progress_text = self.file_browser.get_incremental_progress_text()
            if progress_text:
                self.notification_ctrl.set_label_state(text=progress_text)
            # TODO: During incremental load we should explicitly disallow a
            # subset of mutating actions instead of only returning early here.
            return

        now = time.monotonic()
        if from_file_check and now < self._suppress_file_check_refresh_until:
            if config.debug:
                logger.debug("Skipped file-check refresh during post-move settle window")
            return
        if self._is_refreshing:
            if config.debug:
                logger.debug("Skipped re-entrant refresh call")
            return

        self._is_refreshing = True
        active_media_in_removed = (
            self.media_navigator.get_active_media_filepath() in removed_files
        )
        try:
            self.file_browser.refresh(
                refresh_cursor=refresh_cursor,
                file_check=file_check,
                removed_files=removed_files,
                direction=self.direction,
            )

            if len(removed_files) > 0:
                # Give the just-moved state time to settle before the next periodic file-check refresh.
                self._suppress_file_check_refresh_until = time.monotonic() + 1.0
                if self.mode == Mode.BROWSE:
                    self.notification_ctrl.set_label_state()
                else:
                    self.file_ops_ctrl.handle_remove_files_from_groups(removed_files)
                if self.compare_manager.has_compare():
                    self.compare_manager.compare().remove_from_groups(removed_files)

            if self.file_browser.has_files():
                if self.mode != Mode.BROWSE:
                    self._sync_media_empty_directory_message()
                    return
                has_new_media = False
                if show_new_media:
                    has_new_media = self.file_browser.update_cursor_to_new_media()
                    if has_new_media:
                        self.media_navigator.show_next_media()
                if active_media_in_removed:
                    self.media_navigator.last_chosen_direction_func()
                self.notification_ctrl.set_label_state()
                if show_new_media and has_new_media:
                    # Brief delete-lock to prevent misdeletion after automatic media change
                    self.delete_lock = True
                    time.sleep(1)
                    self.delete_lock = False
            else:
                self.media_navigator.clear_media()
                self.notification_ctrl.set_label_state()
                self.notification_ctrl.alert(
                    _("Warning"), _("No files found in directory after refresh."), kind="warning"
                )
            self._sync_media_empty_directory_message()

            if config.debug:
                logger.debug("Refreshed files")
        finally:
            self._is_refreshing = False

    def refocus(self, event=None) -> None:
        """Return keyboard focus to the main window."""
        self.activateWindow()
        self.media_frame.setFocus()

    def play_media(self, event=None) -> None:
        """Resume/start VLC playback in the media frame."""
        self.media_frame.video_play()

    def pause_media(self, event=None) -> None:
        """Pause VLC playback in the media frame."""
        self.media_frame.video_pause()

    def toggle_media_play_pause(self, event=None) -> None:
        """Toggle VLC playback state in the media frame."""
        self.media_frame.video_toggle_pause()

    def stop_media(self, event=None) -> None:
        """Stop VLC playback in the media frame."""
        self.media_frame.video_stop()

    def seek_media_position(self, position_ms: int) -> None:
        """Seek VLC playback to an absolute millisecond position."""
        self.media_frame.video_seek_ms(position_ms)

    def set_media_volume(self, volume: int) -> None:
        """Set VLC volume level (0-100)."""
        self.media_frame.set_volume(volume)

    def get_media_volume(self) -> int:
        """Return current VLC volume level (0-100)."""
        return self.media_frame.get_volume()

    def toggle_media_mute(self, event=None) -> None:
        """Toggle VLC mute state."""
        self.media_frame.toggle_mute()

    def set_media_mute(self, muted: bool) -> None:
        """Set VLC mute state explicitly."""
        self.media_frame.set_mute(muted)

    def is_media_muted(self) -> bool:
        """Return whether VLC audio is muted."""
        return self.media_frame.is_muted()

    def _get_screenshot_directory(self, active_path: Optional[str] = None) -> str:
        configured = config.screenshot_directory
        if configured and str(configured).strip():
            out_dir = str(configured).strip().replace("{HOME}", os.path.expanduser("~"))
        elif config.save_screenshot_to_same_dir and active_path:
            out_dir = os.path.dirname(active_path) or os.path.join(
                os.path.expanduser("~"), "Pictures", "Screenshots"
            )
        else:
            out_dir = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
        return os.path.normpath(out_dir)

    def release_media_canvas(self) -> None:
        """Release media resources and flush deferred Qt cleanup work."""
        self.media_frame.release_media()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def take_media_screenshot(self, event=None) -> None:
        """Save screenshot for active time-based media and alert the user."""
        if not self.media_frame.has_time_based_media():
            self.app_actions.warn(
                _("Screenshots are available for videos and animated GIF files.")
            )
            return

        active_path = self.media_navigator.get_active_media_filepath() or ""
        out_dir = self._get_screenshot_directory(active_path=active_path)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            self.notification_ctrl.alert(
                _("Screenshot Failed"),
                _("Unable to create screenshot directory:\n{0}\n\n{1}").format(out_dir, str(e)),
                kind="error",
            )
            return

        active_path = active_path or "media"
        stem = os.path.splitext(os.path.basename(active_path))[0] or "media"
        safe_stem = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in stem).strip("_") or "media"
        filename = f"{safe_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
        out_path = os.path.join(out_dir, filename)

        ok, error = self.media_frame.save_media_screenshot(out_path)
        if not ok:
            self.notification_ctrl.alert(
                _("Screenshot Failed"),
                _("Unable to save screenshot.\n\n{0}").format(error),
                kind="error",
            )
            return

        self.app_actions.success(
            _("Screenshot saved to:\n{0}").format(out_path)
        )

    # ------------------------------------------------------------------
    # View toggles (kept on AppWindow -- they touch the top-level layout)
    # ------------------------------------------------------------------
    def toggle_fullscreen(self, event=None) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.sidebar_panel.setVisible(False)
            self.set_title_bar_visible(False)
            self.showFullScreen()
        else:
            self.sidebar_panel.setVisible(True)
            self.set_title_bar_visible(True)
            self.showNormal()

    def end_fullscreen(self, event=None) -> bool:
        if self.fullscreen:
            self.toggle_fullscreen()
        return True

    def toggle_sidebar(self, event=None) -> None:
        showing_sidebar = not self.sidebar_panel.isVisible()
        self.sidebar_panel.setVisible(showing_sidebar)
        if showing_sidebar:
            # Re-apply startup sizing when restoring sidebar visibility so
            # Ctrl+H does not reopen it at an oversized width.
            QTimer.singleShot(0, self._apply_default_sidebar_width)

    def toggle_masonry_view(self, event=None) -> None:
        """Toggle between the single-media viewer and the masonry thumbnail grid."""
        if self.mode != Mode.BROWSE:
            self.notification_ctrl.toast(_("Masonry view is only available in browse mode."))
            return
        if self.view_mode == ViewMode.FULL:
            self.view_mode = ViewMode.MASONRY
            if hasattr(self.media_frame, "pause_video_if_playing"):
                self.media_frame.pause_video_if_playing()
            self._media_stack.setCurrentIndex(1)
            # Defer populate by one event-loop tick so the viewport is laid out
            # and viewport().width() returns the correct value before tile widths
            # are computed.
            files = self.file_browser.get_files()
            current = self.media_path
            QTimer.singleShot(0, lambda: self.masonry_browser.populate(files, current_file=current))
            self.notification_ctrl.toast(_("Masonry view"))
        else:
            self.view_mode = ViewMode.FULL
            self._media_stack.setCurrentIndex(0)
            self.media_frame.setFocus()
            self.notification_ctrl.toast(_("Full-window view"))

    def _refresh_masonry_if_active(self) -> None:
        """Repopulate the masonry grid if view_mode is MASONRY and reveal it.

        Called after a directory load completes (both incremental and non-incremental
        paths) so the grid reflects the new file list without forcing the user to
        re-toggle. The masonry widget retains its viewport size while hidden by the
        QStackedWidget, so populate() can run synchronously before revealing.
        """
        if self.view_mode != ViewMode.MASONRY:
            return
        self.masonry_browser.populate(
            self.file_browser.get_files(), current_file=self.media_path
        )
        self.media_frame.pause_video_if_playing()
        QTimer.singleShot(0, lambda: self._media_stack.setCurrentIndex(1))

    def _on_masonry_tile_activated(self, filepath: str) -> None:
        """Handle a tile click: navigate to the file and return to single view."""
        self.file_browser.go_to_file(filepath)
        self.media_navigator.create_media(filepath)
        self.view_mode = ViewMode.FULL
        self._media_stack.setCurrentIndex(0)
        self.media_frame.setFocus()

    def toggle_theme(self, to_theme: Optional[str] = None, do_toast: bool = True) -> None:
        """Switch between dark and light themes."""
        AppStyle.toggle_theme(to_theme)
        self._apply_theme()
        if do_toast:
            self.notification_ctrl.toast(
                _("Theme switched to {0}.").format(AppStyle.get_theme_name())
            )

    # ------------------------------------------------------------------
    # Context menu (right-click)
    # ------------------------------------------------------------------
    def contextMenuEvent(self, event):  # noqa: N802
        """Show the right-click context menu at the cursor position."""
        self.context_menu_builder.show(event.globalPos())

    # ------------------------------------------------------------------
    # Middle-click → delete media file
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):  # noqa: N802
        """Middle mouse button (wheel click) deletes the current media file."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.file_ops_ctrl.delete_media()
            event.accept()
            return
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    # Mouse scroll → navigate media
    # ------------------------------------------------------------------
    def wheelEvent(self, event):  # noqa: N802
        """Scroll up/down navigates between media files.

        In masonry view the QScrollArea consumes wheel events before they reach
        here, but guard explicitly so stray events don't navigate the file list
        while the grid is visible.  Shift is reserved for future pan/zoom.
        """
        if self.view_mode == ViewMode.MASONRY:
            super().wheelEvent(event)
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.media_navigator.show_next_media()
        elif delta < 0:
            self.media_navigator.show_prev_media()
        event.accept()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    _closing = False  # guard against recursive closeEvent

    def on_closing(self) -> None:
        """
        Clean up and close this window.

        Mirrors App.on_closing: secondary windows clean up their own
        resources; the primary window stores all caches, then destroys
        the application.
        """
        # Stop periodic timers
        self.file_ops_ctrl.stop_file_check_timer()
        self.cache_ctrl.stop_periodic_store()
        # Ensure VLC playback is fully torn down before window destruction.
        self.media_frame.dispose_vlc()

        self.cache_ctrl.store_info_cache(store_window_state=not self.is_secondary())

        if self.is_secondary():
            try:
                if self.compare_manager.has_compare():
                    self.compare_manager.cancel()
            except Exception as e:
                logger.error(f"Error signalling compare cancellation: {e}")

            MarkedFiles.remove_marks_for_base_dir(self.base_dir, self.app_actions)
            WindowManager.unregister(self)
            self.file_check_config.end_filecheck()
            self.slideshow_config.end_slideshows()
            self.media_navigator.stop_slideshow_timers()
        else:
            # Primary window: clean up global resources and store all caches
            from utils.notification_manager import notification_manager
            notification_manager.cleanup_threads()
            self.store_cache_config.end_store_cache()
            for win in list(WindowManager.get_open_windows()):
                if win is not self:
                    win.cache_ctrl.store_info_cache()

    def closeEvent(self, event):
        """Qt close event handler."""
        if self._closing:
            event.accept()
            return
        self._closing = True
        self.on_closing()
        event.accept()
        # If this is the primary window, terminate the entire application
        # so the process doesn't linger after the window is destroyed.
        if not self.is_secondary():
            from PySide6.QtWidgets import QApplication
            QApplication.instance().quit()

    def quit(self, event=None) -> None:
        """
        Prompt the user and quit the entire application.

        Mirrors App.quit: finds the primary window, calls on_closing
        on it (which stores all caches), then terminates the app.
        """
        from PySide6.QtWidgets import QApplication
        from lib.qt_alert import qt_alert

        if qt_alert(self, _("Confirm Quit"), _("Would you like to quit the application?"), kind="askokcancel"):
            logger.warning("Exiting application")
            primary = WindowManager.get_primary()
            if primary:
                primary.on_closing()
            # Kills all windows and exits the event loop.
            QApplication.instance().quit()


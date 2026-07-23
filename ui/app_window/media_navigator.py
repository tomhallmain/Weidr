"""
MediaNavigator -- media browsing logic controller.

Decides which file or page to show next (full ``MediaFrame`` viewer or
``MasonryBrowser`` grid) and coordinates slideshow timing; rendering stays
in the view widgets.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QTimer

from ui.app_window.slideshow_dynamic_policy import (
    should_advance_slideshow_poll,
    skip_classic_slideshow_primary_tick,
    slideshow_poll_should_run,
)
from utils.config import config
from utils.constants import Direction, Mode, ViewMode
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

if TYPE_CHECKING:
    from compare.compare_manager import CompareManager
    from files.file_browser import FileBrowser, SortBy
    from ui.app_window.app_window import AppWindow
    from ui.app_window.media_frame import MediaFrame
logger = get_logger("media_navigator")


class MediaNavigator:
    """
    Controls which media file is displayed and provides navigation
    (prev, next, home, page up/down, go-to-file, etc.).
    """

    def __init__(
        self,
        app_window: AppWindow,
        file_browser: FileBrowser,
        compare_manager: CompareManager,
        media_frame: MediaFrame,
    ):
        self._app = app_window
        self._fb = file_browser
        self._cm = compare_manager
        self._mf = media_frame
        self._slideshow_timer = QTimer(self._app)
        self._slideshow_timer.timeout.connect(self._on_slideshow_tick)
        self._slideshow_dynamic_poll_timer = QTimer(self._app)
        self._slideshow_dynamic_poll_timer.setInterval(250)
        self._slideshow_dynamic_poll_timer.timeout.connect(self._on_slideshow_dynamic_poll_tick)
        self._slideshow_media_started_monotonic: Optional[float] = None

    # ==================================================================
    # Navigation
    # ==================================================================
    def show_prev_media(self, event=None, show_alert: bool = True) -> bool:
        """
        Navigate to the previous media file.

        In BROWSE mode, walks backward through the file browser, skipping
        files the compare manager says to skip. In compare modes, delegates
        to ``CompareManager.show_prev_media``.
        """
        self._app.direction = Direction.BACKWARD

        if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
            start_media = self._fb.current_file()
            previous_media = self._fb.previous_file()
            if self._app.media_path == previous_media:
                return True  # already at this file (refresh case)
            while self._cm.skip_media(previous_media) and previous_media != start_media:
                previous_media = self._fb.previous_file()
            try:
                self.create_media(previous_media)
                return True
            except Exception as e:
                self._app.notification_ctrl.handle_error(str(e), title="Exception")
                return False

        return self._cm.show_prev_media(show_alert=show_alert)

    def show_next_media(self, event=None, show_alert: bool = True) -> bool:
        """Navigate to the next media file."""
        self._app.direction = Direction.FORWARD

        if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
            start_media = self._fb.current_file()
            next_media = self._fb.next_file()
            if self._app.media_path == next_media:
                return True  # already at this file (refresh case)
            while self._cm.skip_media(next_media) and next_media != start_media:
                next_media = self._fb.next_file()
            try:
                self.create_media(next_media)
                return True
            except Exception as e:
                traceback.print_exc()
                self._app.notification_ctrl.handle_error(str(e), title="Exception")
                return False

        return self._cm.show_next_media(show_alert=show_alert)

    def last_chosen_direction_func(self) -> None:
        """Repeat the last navigation direction."""
        if self._app.direction == Direction.BACKWARD:
            self.show_prev_media()
        elif self._app.direction == Direction.FORWARD:
            self.show_next_media()
        else:
            raise Exception(f"Direction was improperly set. Direction was {self._app.direction}")

    def home(self, event=None, last_file: bool = False) -> None:
        """Jump to the first or last file."""
        from ui.files.marked_file_mover_qt import MarkedFiles

        if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
            current_media = self.get_active_media_filepath()
            if not self._fb.is_incremental_loading:
                self._fb.refresh()
            elif not self._fb.has_files():
                if self._fb.is_incremental_loading:
                    self._app.notification_ctrl.toast(
                        _("Directory is still loading; no files available yet.")
                    )
                else:
                    recursive_str = "" if self._fb.recursive else _(" (try setting recursive to True)")
                    self._app.notification_ctrl.toast(
                        _("No files found for current browsing settings.") + recursive_str
                    )
                return
            if current_media is None:
                if self._fb.has_files():
                    if self._fb.is_incremental_loading:
                        # NOTE: no skip/prevalidation check here — during
                        # incremental load the file list and cursor are still
                        # in flux, so direct-display prevalidation
                        # (prevalidate_on_direct_media_display) is not applied
                        # to this bootstrap display yet.
                        self.create_media(self._fb.get_files()[0])
                        current_media = self.get_active_media_filepath()
                        if current_media is None:
                            self._app.notification_ctrl.toast(
                                _("Directory is still loading; no files available yet.")
                            )
                            return
                    else:
                        raise Exception("No active media file found.")
                else:
                    raise Exception("No active media file found.")

            try:
                if last_file:
                    target = self._fb.last_file()
                    while self._cm.skip_media(target) and target != current_media:
                        target = self._fb.previous_file()
                    self.create_media(target)
                    if (len(MarkedFiles.file_marks) == 1
                            and self._fb.has_file(MarkedFiles.file_marks[0])):
                        self._app.file_marks_ctrl.add_all_marks_from_last_or_current_group()
                    self._app.direction = Direction.BACKWARD
                else:
                    target = self._fb.next_file()
                    while self._cm.skip_media(target) and target != current_media:
                        target = self._fb.next_file()
                    self.create_media(target)
            except Exception:
                traceback.print_exc()
                if self._fb.is_incremental_loading:
                    self._app.notification_ctrl.toast(
                        _("Directory is still loading; try again in a moment.")
                    )
                    return
                if not self._fb.has_files():
                    recursive_str = "" if self._fb.recursive else _(" (try setting recursive to True)")
                    self._app.notification_ctrl.toast(
                        _("No files found for current browsing settings.") + recursive_str
                    )
                    return
                raise

        elif self._cm.has_compare():
            self._app.direction = Direction.BACKWARD if last_file else Direction.FORWARD
            self._cm.show_boundary_match(last_file=last_file)

    def page_up(self, event=None) -> None:
        """Jump backward by a page of files.

        In masonry view, steps to the previous page of thumbnails instead of
        navigating the file cursor — the grid has its own page concept.
        """
        if self._app.view_mode == ViewMode.MASONRY:
            self._app.masonry_browser.prev_page()
            return

        current_media = self.get_active_media_filepath()
        if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
            prev_media = self._fb.page_up()
        else:
            prev_media = self._cm.page_up()

        while self._cm.skip_media(prev_media) and prev_media != current_media:
            if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
                prev_media = self._fb.previous_file()
            else:
                prev_media = self._cm._get_prev_media()

        self.create_media(prev_media)
        self._app.direction = Direction.BACKWARD

    def page_down(self, event=None) -> None:
        """Jump forward by a page of files.

        In masonry view, steps to the next page of thumbnails instead of
        navigating the file cursor — the grid has its own page concept.
        """
        if self._app.view_mode == ViewMode.MASONRY:
            self._app.masonry_browser.next_page()
            return

        current_media = self.get_active_media_filepath()
        if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
            next_media = self._fb.page_down()
        else:
            next_media = self._cm.page_down()

        while self._cm.skip_media(next_media) and next_media != current_media:
            if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
                next_media = self._fb.next_file()
            else:
                next_media = self._cm._get_next_media()

        self.create_media(next_media)
        self._app.direction = Direction.FORWARD

    # ==================================================================
    # Go-to-file
    # ==================================================================
    def _suppress_direct_display(self, media_path, compare_manager=None) -> bool:
        """Run prevalidations for a directly-requested file (config-gated).

        Returns True when the file must not be displayed. A matched
        prevalidation notifies through its own normal callbacks (the real
        skip_media pipeline runs, side effects included); the generic toast
        here only covers suppressions that emit nothing of their own (e.g.
        hidden files).
        """
        if not config.prevalidate_on_direct_media_display:
            return False
        cm = compare_manager if compare_manager is not None else self._cm
        if not cm.skip_media(media_path):
            return False
        self._app.notification_ctrl.toast(
            _("Not shown (prevalidation or hidden): {0}").format(
                os.path.basename(media_path)
            )
        )
        return True

    def go_to_file(
        self,
        event=None,
        search_text: str = "",
        retry_with_delay: int = 0,
        exact_match: bool = True,
        closest_sort_by: Optional[SortBy] = None,
    ) -> bool:
        """
        Navigate to a specific file by name or path.

        Searches the current window first, then other open windows. If the
        file is not found anywhere and ``search_text`` is a valid file path,
        opens it in a temporary canvas.
        """
        from ui.image.media_details import MediaDetails
        from ui.app_window.window_manager import WindowManager

        original_search_text = search_text
        resolved_path = Utils.get_valid_file(self._app.get_base_dir(), original_search_text)
        original_search_text_is_file = resolved_path and os.path.isfile(resolved_path)
        exact_match = exact_match or original_search_text_is_file
        if not exact_match:
            search_text = os.path.basename(search_text)

        # --- Search in current window ---
        if self._app.mode == Mode.BROWSE or self._app.search_ctrl.is_compare_running():
            if not self._fb.is_incremental_loading:
                self._fb.refresh()
            if config.debug:
                logger.debug(f"Finding file in current window: {search_text}, closest sort by: {closest_sort_by}")
            media_path = self._fb.find(
                search_text=search_text,
                retry_with_delay=retry_with_delay,
                exact_match=exact_match,
                closest_sort_by=closest_sort_by,
            )
            if media_path:
                if self._suppress_direct_display(media_path):
                    return True
                self.create_media(media_path)
                return True
        else:
            media_path, group_indexes = self._cm.find_file_after_comparison(
                search_text, exact_match=exact_match
            )
            if group_indexes:
                if self._suppress_direct_display(media_path):
                    return True
                self._cm.current_group_index = group_indexes[0]
                self._cm.set_current_group(start_match_index=group_indexes[1])
                return True

        # --- Search in other open windows ---
        for window in WindowManager.get_open_windows():
            if window.window_id == self._app.window_id:
                continue

            if window.mode == Mode.BROWSE:
                if not window.file_browser.is_incremental_loading:
                    window.file_browser.refresh()
                found_path = window.file_browser.find(
                    search_text=search_text,
                    retry_with_delay=retry_with_delay,
                    exact_match=exact_match,
                    closest_sort_by=closest_sort_by,
                )
                if found_path:
                    if self._suppress_direct_display(
                        found_path, compare_manager=window.compare_manager
                    ):
                        return True
                    window.raise_()
                    window.activateWindow()
                    window.media_navigator.create_media(found_path)
                    return True
            else:
                found_path, group_indexes = window.compare_manager.find_file_after_comparison(
                    search_text, exact_match=exact_match
                )
                if found_path and group_indexes:
                    if self._suppress_direct_display(
                        found_path, compare_manager=window.compare_manager
                    ):
                        return True
                    window.compare_manager.current_group_index = group_indexes[0]
                    window.compare_manager.set_current_group(start_match_index=group_indexes[1])
                    window.raise_()
                    window.activateWindow()
                    return True
                # If not found in compare results, search the full directory
                if not window.file_browser.is_incremental_loading:
                    window.file_browser.refresh()
                found_path = window.file_browser.find(
                    search_text=search_text,
                    retry_with_delay=retry_with_delay,
                    exact_match=exact_match,
                    closest_sort_by=closest_sort_by,
                )
                if found_path:
                    MediaDetails.open_temp_media_canvas(
                        master=self._app, media_path=found_path,
                        app_actions=self._app.app_actions, skip_get_window_check=True,
                    )
                    return True

        # --- File is a valid path on disk → open in temp canvas ---
        if original_search_text_is_file:
            MediaDetails.open_temp_media_canvas(
                master=self._app, media_path=resolved_path,
                app_actions=self._app.app_actions, skip_get_window_check=True,
            )
            return True

        # --- Not found anywhere ---
        self._app.notification_ctrl.alert(
            _("File not found"),
            _('No file was found for the search text: "{0}"').format(search_text),
        )
        return False

    def go_to_file_by_index(self, index: int) -> bool:
        """Navigate to a file by its index (1-based) in the file browser."""
        if self._app.mode != Mode.BROWSE:
            self._app.notification_ctrl.alert(
                _("Index navigation not available"),
                _("Index navigation is only available in BROWSE mode."),
            )
            return False

        try:
            if not self._fb.is_incremental_loading:
                self._fb.refresh()
            file_path = self._fb.go_to_index(index)
            if file_path:
                if self._suppress_direct_display(file_path):
                    return True
                self.create_media(file_path)
                return True
        except ValueError as e:
            self._app.notification_ctrl.alert(_("Invalid index"), str(e))
            return False
        except Exception as e:
            self._app.notification_ctrl.handle_error(str(e), title="Go To Index Error")
            return False

        return False

    def go_to_previous_media(self, event=None) -> None:
        """Navigate back to the previously viewed media file."""
        if self._app.prev_media_path is not None:
            self.go_to_file(event=event, search_text=self._app.prev_media_path)

    # ==================================================================
    # Display
    # ==================================================================
    def create_media(self, media_path: str, extra_text: Optional[str] = None) -> None:
        """
        Show a media file in the main content pane of the UI.

        Updates the sidebar label, the internal path state, and refreshes
        the media-details window if open.
        """
        if not media_path:
            return
        self._mf.show_media(media_path)
        self._mf.apply_pending_blur(media_path)

        relative_filepath, basename = Utils.get_relative_dirpath_split(
            self._app.base_dir, media_path
        )
        self._app.prev_media_path = self._app.media_path
        self._app.media_path = media_path
        self._slideshow_media_started_monotonic = time.monotonic()
        self._sync_slideshow_dynamic_poll_timer()
        self._restart_classic_slideshow_primary_timer_if_running()

        text = basename if relative_filepath == "" else relative_filepath + "\n" + basename
        if extra_text is not None:
            text += "\n" + extra_text
        self._app.sidebar_panel.update_current_media_label(text)

        # Auto-refresh the media details window if it is open
        if self._app.app_actions.media_details_window() is not None:
            self._app.window_launcher.open_media_details(manually_keyed=False)

    def clear_media(self) -> None:
        """Clear the currently displayed media."""
        self._mf.clear()
        self._app.sidebar_panel.update_current_media_label("")
        self._app.media_path = None
        self._slideshow_media_started_monotonic = None
        self._slideshow_dynamic_poll_timer.stop()

    def show_searched_media(self) -> None:
        """Display the media file found by the last search."""
        search_path = self._cm.search_media_path
        if config.debug:
            logger.debug(f"Search file path: {search_path}")
        if search_path is not None and search_path.strip() != "":
            if os.path.isfile(search_path):
                self.create_media(search_path, extra_text="(search media)")
            else:
                logger.warning(search_path)
                self._app.notification_ctrl.handle_error(
                    _("Somehow, the search file is invalid")
                )

    def toggle_media_view(self) -> None:
        """While in search mode, toggle between the search media and the results."""
        if self._app.mode != Mode.SEARCH:
            return

        if self._app.is_toggled_view_matches:
            self.show_searched_media()
        else:
            self.create_media(self._cm.current_match())

        self._app.is_toggled_view_matches = not self._app.is_toggled_view_matches

    def set_toggled_view_matches(self) -> None:
        """Set the toggled view to show matches."""
        self._app.is_toggled_view_matches = True

    # ==================================================================
    # Slideshow
    # ==================================================================
    def _restart_classic_slideshow_primary_timer_if_running(self) -> None:
        """Reset the main interval so the next auto-advance is a full interval away."""
        if not self._app.slideshow_config.slideshow_running:
            return
        interval_sec = float(self._app.slideshow_config.interval_seconds)
        if interval_sec <= 0:
            interval_sec = 7.0
        interval_ms = max(1, int(interval_sec * 1000))
        self._slideshow_timer.start(interval_ms)

    def toggle_slideshow(self, event=None) -> None:
        """
        Toggle the slideshow on or off.

        Uses an internal QTimer for the classic slideshow mode and the
        file-check timer for "new media" mode.
        """
        self._app.slideshow_config.toggle_slideshow()
        if self._app.slideshow_config.show_new_media:
            message = _("Slideshow for new media started")
            self.stop_slideshow_timers()
        elif self._app.slideshow_config.slideshow_running:
            message = _("Slideshow started")
            self._slideshow_media_started_monotonic = time.monotonic()
            self._restart_classic_slideshow_primary_timer_if_running()
            self._sync_slideshow_dynamic_poll_timer()
        else:
            message = _("Slideshows ended")
            self.stop_slideshow_timers()
            self._app.clear_new_media_queue()
        self._app.notification_ctrl.toast(message)

    def stop_slideshow_timers(self) -> None:
        """Stop classic slideshow timers (e.g. when slideshow ends from sidebar or window close)."""
        self._slideshow_timer.stop()
        self._slideshow_dynamic_poll_timer.stop()

    def _sync_slideshow_dynamic_poll_timer(self) -> None:
        if not self._app.slideshow_config.slideshow_running:
            self._slideshow_dynamic_poll_timer.stop()
            return
        path = self._app.media_path
        if slideshow_poll_should_run(self._mf, path):
            if not self._slideshow_dynamic_poll_timer.isActive():
                self._slideshow_dynamic_poll_timer.start()
        else:
            self._slideshow_dynamic_poll_timer.stop()

    def _on_slideshow_dynamic_poll_tick(self) -> None:
        if not self._app.slideshow_config.slideshow_running:
            self._slideshow_dynamic_poll_timer.stop()
            return
        path = self._app.media_path
        if not path:
            return
        if should_advance_slideshow_poll(self._mf, path, self._slideshow_media_started_monotonic):
            self.show_next_media()

    def _on_slideshow_tick(self) -> None:
        """Advance slideshow when classic slideshow mode is active."""
        if not self._app.slideshow_config.slideshow_running:
            self.stop_slideshow_timers()
            return
        base_dir = self._app.get_base_dir()
        if not base_dir or base_dir == "":
            return
        path = self._app.media_path
        if skip_classic_slideshow_primary_tick(self._mf, path):
            return
        self.show_next_media()

    # ==================================================================
    # Queries
    # ==================================================================
    def get_active_media_filepath(self) -> Optional[str]:
        """Return the path of the currently displayed media file."""
        # In browse mode, prefer the file_browser cursor
        if self._app.mode == Mode.BROWSE:
            return self._fb.current_file()

        if self.is_toggled_search_media():
            filepath = self._cm.search_media_path
        else:
            filepath = self._cm.current_match()

        return Utils.get_valid_file(self._app.get_base_dir(), filepath)

    def is_toggled_search_media(self) -> bool:
        """Return True if the toggled view is showing the search media."""
        return self._app.mode == Mode.SEARCH and not self._app.is_toggled_view_matches

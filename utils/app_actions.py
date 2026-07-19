from functools import cached_property
from typing import Callable, Dict, Any, Optional

from ui.app_style import AppStyle

class AppActions:
    REQUIRED_ACTIONS = {
        "new_window", "get_window", "toast", "_alert", "title_notify", "refresh",
        "refocus", "set_mode", "get_active_media_filepath", "create_media", "show_next_media",
        "play_media", "pause_media", "toggle_media_play_pause", "seek_media", "stop_media",
        "set_media_volume", "get_media_volume", "toggle_media_mute", "set_media_mute", "is_media_muted",
        "get_media_details", "open_move_marks_window", "open_password_admin_window", "open_file_action_sets_window",
         "run_image_generation", "set_marks_from_downstream_related_images",
        "set_base_dir", "get_base_dir", "go_to_file", "go_to_file_by_index", "delete",
        "hide_current_media", "request_media_blur", "copy_media_path",
        "release_media_canvas", "store_info_cache",
        "_add_buttons_for_mode", "_set_label_state",
        "_set_toggled_view_matches", "refresh_all_compares",
        "start_loading_spinner", "stop_loading_spinner",
        "start_progress_bar", "stop_progress_bar",
        "refresh_masonry",
        "is_compare_running",
        "restore_compare_state_for_undone_move",
    }
    
    def __init__(self, actions: Dict[str, Callable[..., Any]], master: Optional[object] = None):
        missing = self.REQUIRED_ACTIONS - set(actions.keys())
        if missing:
            raise ValueError(f"Missing required actions: {missing}")
        self._actions = actions
        self._master = master
    
    def __getattr__(self, name):
        if name in self._actions:
            return self._actions[name]
        raise AttributeError(f"Action '{name}' not found")
    
    def alert(self, title: str, message: str, kind: str = "info", severity: str = "normal", master: Optional[object] = None) -> None:
        """
        Override the alert method to automatically inject the master parameter.
        If master is explicitly provided, use it; otherwise use the stored master.
        """
        # Use provided master or fall back to stored master
        parent_window = master if master is not None else self._master
        
        # Call the original alert method with the determined parent window
        return self._alert(title, message, kind=kind, severity=severity, master=parent_window)

    def warn(self, message: str, time_in_seconds: int = None) -> None:
        """
        Show a warning toast with the warning background color.
        This is a convenience method for displaying warning messages.
        """
        # Import here to avoid circular dependency
        from utils.config import config
        
        if time_in_seconds is None:
            time_in_seconds = config.toasts_persist_seconds
        
        # Call toast with the warning background color
        return self.toast(message, time_in_seconds=time_in_seconds, bg_color=AppStyle.TOAST_COLOR_WARNING)

    def success(self, message: str, time_in_seconds: int = None) -> None:
        """
        Show a success toast with the success background color.
        This is a convenience method for displaying success messages.
        """
        # Import here to avoid circular dependency
        from utils.config import config
        
        if time_in_seconds is None:
            time_in_seconds = config.toasts_persist_seconds
        
        # Call toast with the success background color
        return self.toast(message, time_in_seconds=time_in_seconds, bg_color=AppStyle.TOAST_COLOR_SUCCESS)

    def get_master(self):
        return self._master

    def media_details_window(self):
        return self._actions.get("_media_details_window")

    def set_media_details_window(self, media_details_window):
        self._actions["_media_details_window"] = media_details_window

    @cached_property
    def prevalidation_callbacks(self):
        return self._build_callbacks()

    @cached_property
    def prevalidation_callbacks_with_mark(self):
        from files.marked_files import MarkedFiles
        return self._build_callbacks(add_mark_callback=MarkedFiles.add_mark_if_not_present)

    def make_prevalidation_callbacks(self, add_mark_callback):
        """Build an ActionCallbacks bundle with a custom add_mark_callback."""
        return self._build_callbacks(add_mark_callback=add_mark_callback)

    def _build_callbacks(self, add_mark_callback=None):
        from compare.action_callbacks import ActionCallbacks
        return ActionCallbacks(
            hide_callback=self.hide_current_media,
            notify_callback=self.title_notify,
            add_mark_callback=add_mark_callback,
            blur_callback=self.request_media_blur,
            generate_callback=lambda path, edit_suffix=None, target_dir=None: self.run_image_generation(
                media_path=path, edit_suffix=edit_suffix, suppress_toast=True, target_dir=target_dir
            ),
        )

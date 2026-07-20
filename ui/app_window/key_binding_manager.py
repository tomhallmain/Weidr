"""
KeyBindingManager -- owns all QShortcut creation.

Each shortcut is guarded by a focus check so that single-key shortcuts
are suppressed while the user is typing in an AwareEntry.

Tkinter key syntax → Qt key syntax mapping:
    <Left>              → Left
    <Shift-M>           → Shift+M
    <Control-q>         → Ctrl+Q
    <Control-Shift-N>   → Ctrl+Shift+N
    <F11>               → F11
    <Home>              → Home
    <Prior>             → PgUp
    <Next>              → PgDown
    <Return>            → Return
    <Shift-BackSpace>   → Shift+Backspace
    <Shift-Delete>      → Shift+Delete
    <Control-Return>    → Ctrl+Return

Mouse bindings (Button-2, Button-3, MouseWheel) are NOT handled via
QShortcut — they are handled in the media frame's event filters or
the AppWindow's wheelEvent / contextMenuEvent overrides.

Keybinding map  (■ = bound  · = free | cols: Shift | Ctrl | Ctrl+Shift)

    A  ■ ■ ·   N  ■ ■ ■
    B  ■ ■ ·   O  ■ · ·
    C  ■ ■ ■   P  ■ ■ ■
    D  ■ ■ ■   Q  ■ ■ ·
    E  ■ ■ ■   R  ■ ■ ·
    F  ■ ■ ·   S  ■ ■ ■
    G  ■ ■ ■   T  ■ ■ ■
    H  ■ ■ ·   U  ■ · ·
    I  ■ ■ ·   V  ■ ■ ·
    J  ■ ■ ·   W  ■ ■ ·
    K  ■ ■ ■   X  · ■ ·
    L  ■ ■ ·   Y  ■ ■ ■
    M  ■ ■ ■   Z  ■ ■ ·

    Special: 0-9 ■ (bare + Shift), Left/Right ■ (bare + Shift + Ctrl+Shift),
             Home/End/PgUp/PgDown ■ (bare), Backspace ■ (Shift),
             Delete ■ (Shift + Ctrl+Shift), Escape ■ (bare + Shift),
             Return ■ (Ctrl + Ctrl+Shift + Ctrl+Alt),
             Tab ■ (Ctrl + Ctrl+Shift), F1/F11 ■, Y ■ (Alt)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PySide6.QtGui import QKeySequence, QShortcut

from lib.aware_entry_qt import AwareEntry
from utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow

logger = get_logger("key_binding_manager")


class KeyBindingManager:
    """
    Creates all keyboard shortcuts for the main AppWindow.

    Shortcuts that conflict with text entry (single-character keys, Shift+<letter>)
    are wrapped in a guard that checks ``AwareEntry.an_entry_has_focus``.
    """

    def __init__(self, app_window: AppWindow):
        self._app = app_window
        self._shortcuts: list[QShortcut] = []
        self._bind_all()

    # ------------------------------------------------------------------
    # Guard wrapper
    # ------------------------------------------------------------------
    @staticmethod
    def _guarded(func: Callable) -> Callable:
        """Wrap *func* so it only fires when no AwareEntry has focus."""
        def wrapper():
            if not AwareEntry.an_entry_has_focus:
                func()
        return wrapper

    # ------------------------------------------------------------------
    # Shortcut helpers
    # ------------------------------------------------------------------
    def _bind(self, key: str, func: Callable, guarded: bool = True) -> None:
        """Create a QShortcut, optionally guarded against text entry focus."""
        target = self._guarded(func) if guarded else func
        shortcut = QShortcut(QKeySequence(key), self._app)
        shortcut.activated.connect(target)
        self._shortcuts.append(shortcut)

    # ------------------------------------------------------------------
    # Named handlers
    # ------------------------------------------------------------------

    def _run_last_seek_to_trigger(self) -> None:
        from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab
        SeekToTriggerTab.run_last_seek_to_trigger(self._app.app_actions)

    # ------------------------------------------------------------------
    # All bindings -- ported from App.__init__ lines 405-475
    # ------------------------------------------------------------------
    def _bind_all(self) -> None:  # noqa: C901  (complexity is inherent)
        """Register all keyboard shortcuts."""
        app = self._app
        from ui.files.marked_file_mover_qt import MarkedFiles, MarkedFileMover
        from ui.image.media_details import MediaDetails
        from ui.app_window.window_manager import WindowManager
        from utils.constants import ImageGenerationType, Mode
        from files.file_action import FileAction

        # ==============================================================
        # Navigation (arrow keys, Home/End, PgUp/PgDown)
        # ==============================================================
        self._bind("Left", app.media_navigator.show_prev_media)
        self._bind("Right", app.media_navigator.show_next_media)
        self._bind("Up", lambda: app.media_frame.pdf_navigate(-1))
        self._bind("Down", lambda: app.media_frame.pdf_navigate(1))
        self._bind("Shift+Backspace", app.media_navigator.go_to_previous_media)
        self._bind(
            "Shift+Left",
            lambda: app.compare_manager.show_prev_group(
                file_browser=(app.file_browser if app.mode == Mode.BROWSE else None)
            ),
        )
        self._bind(
            "Shift+Right",
            lambda: app.compare_manager.show_next_group(
                file_browser=(app.file_browser if app.mode == Mode.BROWSE else None)
            ),
        )
        self._bind("Ctrl+Shift+Left", app.compare_manager.show_prev_supergroup)
        self._bind("Ctrl+Shift+Right", app.compare_manager.show_next_supergroup)
        self._bind("Home", lambda: app.media_navigator.home())
        self._bind("End", lambda: app.media_navigator.home(last_file=True))
        self._bind("PgUp", app.media_navigator.page_up)
        self._bind("PgDown", app.media_navigator.page_down)

        # ==============================================================
        # File operations (Shift+<key>)
        # ==============================================================
        self._bind("Shift+O", app.file_ops_ctrl.open_media_location)
        self._bind("Shift+P", app.file_ops_ctrl.open_image_in_gimp)
        self._bind("Shift+Delete", app.file_ops_ctrl.delete_media)
        self._bind("Ctrl+Shift+Delete", app.file_ops_ctrl.delete_current_base_dir)
        self._bind("Shift+V", app.file_ops_ctrl.hide_current_media)
        self._bind("Shift+B", app.file_ops_ctrl.clear_hidden_media)
        self._bind("Shift+U", app.file_ops_ctrl.run_refacdir)

        # ==============================================================
        # View / mode
        # ==============================================================
        self._bind("F11", app.toggle_fullscreen, guarded=False)
        self._bind("Shift+F", app.toggle_fullscreen)
        self._bind("Escape", lambda: app.end_fullscreen() and app.refocus(), guarded=False)
        self._bind("Ctrl+Shift+G", app.toggle_masonry_view)

        # ==============================================================
        # Info / details
        # ==============================================================
        self._bind("Shift+D", app.window_launcher.open_media_details)
        self._bind("Shift+R", app.window_launcher.show_related_media)
        self._bind(
            "Shift+T",
            lambda: app.search_ctrl.find_related_media_in_open_window()
            if hasattr(app.search_ctrl, "find_related_media_in_open_window")
            else None,
        )
        self._bind("Shift+Y", app.file_marks_ctrl.set_marks_from_downstream_related_images)
        self._bind("Ctrl+Y", app.file_marks_ctrl.mark_sources_with_downstream_in_dir, guarded=False)
        # Ctrl+Shift+Y previously ran mark_downstream_files_in_dir directly;
        # that action is now a button inside the related images window.
        self._bind("Ctrl+Shift+Y", app.window_launcher.open_related_images_window, guarded=False)
        self._bind("Alt+Y", app.file_marks_ctrl.set_marks_from_downstream_related_images_all_windows)
        self._bind("Shift+H", app.window_launcher.get_help_and_config)
        self._bind("Shift+E", app.window_launcher.copy_prompt)
        self._bind(
            "Shift+K",
            lambda: MediaDetails.open_temp_media_canvas(
                app, MarkedFiles.last_moved_image, app.app_actions
            ),
        )
        self._bind(
            "Ctrl+Shift+K",
            lambda: MediaDetails.open_temp_media_canvas(
                app, FileAction.get_last_auto_file(), app.app_actions
            ),
        )

        # ==============================================================
        # Search / compare
        # ==============================================================
        self._bind(
            "Shift+A",
            lambda: app.search_ctrl.set_current_media_run_search(base_dir=app.base_dir),
        )
        # Keep Shift+Z on the generic resolver flow. Unlike Shift+A, there is no
        # complementary Ctrl+<key> path available for explicit cross-window targeting
        # because Ctrl+Z is reserved for undo.
        # TODO: Consider adding Ctrl+Shift+Z for explicit current-window negative
        # search binding if shortcut conflicts remain acceptable.
        self._bind("Shift+Z", app.search_ctrl.add_current_media_to_negative_search)
        self._bind(
            "Shift+I",
            lambda: MediaDetails.run_image_generation_static(app.app_actions),
        )
        self._bind(
            "Shift+Q",
            lambda: MediaDetails.randomly_modify_image(
                app.media_navigator.get_active_media_filepath(), app.app_actions, app
            ),
        )
        self._bind(
            "Shift+W",
            lambda: MediaDetails.source_random_prompt(app.file_browser, app, app.app_actions),
        )
        self._bind(
            "Ctrl+Return",
            lambda: app.search_ctrl.run_image_generation(
                _type=ImageGenerationType.LAST_SETTINGS,
                media_path=MediaDetails.previous_image_generation_adapter_path,
            ),
            guarded=False,
        )
        self._bind(
            "Ctrl+Shift+Return",
            lambda: app.search_ctrl.run_image_generation(
                _type=ImageGenerationType.CANCEL,
                media_path=MediaDetails.previous_image_generation_adapter_path,
            ),
            guarded=False,
        )
        self._bind(
            "Ctrl+Alt+Return",
            lambda: app.search_ctrl.run_image_generation(
                _type=ImageGenerationType.REVERT_TO_SIMPLE_GEN,
                media_path=MediaDetails.previous_image_generation_adapter_path,
            ),
            guarded=False,
        )
        # Note: no global Return binding. The individual search QLineEdits
        # connect their returnPressed signals to the appropriate set_search_*
        # methods (see SidebarPanel), matching the Tkinter version where
        # <Return> was only bound on the entry widgets.

        # ==============================================================
        # Slideshow
        # ==============================================================
        self._bind("Shift+S", app.take_media_screenshot)
        self._bind("Ctrl+S", app.media_navigator.toggle_slideshow, guarded=False)

        # ==============================================================
        # Prevalidations / debug
        # ==============================================================
        self._bind("Shift+J", app.window_launcher.run_prevalidations_for_base_dir)
        self._bind("Shift+L", app.window_launcher.toggle_prevalidations)
        self._bind("Ctrl+Shift+D", app.window_launcher.toggle_extra_debug_logging)

        # ==============================================================
        # File marks
        # ==============================================================
        self._bind("Shift+M", app.file_marks_ctrl.add_or_remove_mark)
        self._bind("Shift+N", app.file_marks_ctrl.add_all_marks_from_last_or_current_group)
        self._bind("Shift+G", app.file_marks_ctrl.go_to_mark)
        self._bind("Shift+C", lambda: MarkedFiles.clear_file_marks(app.app_actions))
        self._bind("Ctrl+C", app.file_marks_ctrl.copy_marks_list)

        # Digit keys (0-9) for hotkey marks actions.
        # In Qt, QShortcut doesn't pass an event, so we capture the digit
        # and shift state in closures.
        for i in range(10):
            self._bind(
                str(i),
                lambda _n=i: app.file_marks_ctrl.run_hotkey_marks_action(
                    number=_n, shift_pressed=False
                ),
            )
            self._bind(
                f"Shift+{i}",
                lambda _n=i: app.file_marks_ctrl.run_hotkey_marks_action(
                    number=_n, shift_pressed=True
                ),
                guarded=False,
            )

        # ==============================================================
        # Window management
        # ==============================================================
        self._bind("Ctrl+Tab", WindowManager.cycle_windows, guarded=False)
        self._bind("Ctrl+Shift+Tab", lambda: WindowManager.cycle_windows(reverse=True), guarded=False)
        self._bind(
            "Shift+Escape",
            lambda: app.close() if app.is_secondary() else None,
            guarded=False,
        )
        self._bind("Ctrl+Q", app.quit, guarded=False)

        # ==============================================================
        # Window launchers (Ctrl+<key>)
        # ==============================================================
        self._bind("Ctrl+P", app.window_launcher.open_password_admin_window, guarded=False)
        self._bind("Ctrl+W", app.window_launcher.open_secondary_compare_window, guarded=False)
        self._bind(
            "Ctrl+A",
            lambda: app.window_launcher.open_secondary_compare_window(run_compare_media=app.media_path),
            guarded=False,
        )
        self._bind("Ctrl+G", app.window_launcher.open_go_to_file_window, guarded=False)
        self._bind("Ctrl+I", app.window_launcher.open_go_to_file_with_current_media, guarded=False)
        self._bind("Ctrl+H", app.toggle_sidebar, guarded=False)
        self._bind("Ctrl+F", app.window_launcher.open_favorites_window, guarded=False)
        self._bind("Ctrl+N", app.window_launcher.open_file_actions_window, guarded=False)
        self._bind("Ctrl+M", app.file_marks_ctrl.open_move_marks_window, guarded=False)
        self._bind("Ctrl+Shift+N", app.window_launcher.open_directory_notes_window, guarded=False)
        self._bind(
            "Ctrl+K",
            lambda: app.file_marks_ctrl.open_move_marks_window(open_gui=False),
            guarded=False,
        )
        self._bind("Ctrl+J", app.window_launcher.open_prevalidations_window, guarded=False)
        self._bind("Ctrl+Shift+M", app.window_launcher.open_hf_model_manager_window, guarded=False)
        self._bind("Ctrl+V", app.window_launcher.open_type_configuration_window)
        self._bind("Ctrl+Shift+C", app.window_launcher.open_compare_settings_window)

        # ==============================================================
        # Marks actions (Ctrl+<key>)
        # ==============================================================
        self._bind("Ctrl+R", app.file_marks_ctrl.run_previous_marks_action, guarded=False)
        self._bind("Ctrl+E", app.file_marks_ctrl.run_penultimate_marks_action, guarded=False)
        self._bind("Ctrl+Shift+E", app.file_marks_ctrl.run_antepenultimate_marks_action, guarded=False)
        self._bind("Ctrl+T", app.file_marks_ctrl.run_permanent_marks_action, guarded=False)
        self._bind("Ctrl+Shift+T", app.file_marks_ctrl.run_file_action_set, guarded=False)
        self._bind(
            "Ctrl+D",
            lambda: MarkedFiles.set_current_marks_from_previous(app.app_actions),
            guarded=False,
        )
        self._bind("Ctrl+Z", app.file_marks_ctrl.revert_last_marks_change, guarded=False)
        self._bind("Ctrl+Shift+I", app.file_marks_ctrl.create_interpolation_gif, guarded=False)
        self._bind(
            "Ctrl+X",
            lambda: MarkedFileMover.undo_move_marks(None, app.app_actions),
        )
        self._bind("Ctrl+Shift+P", app.window_launcher.interactive_crop, guarded=False)
        self._bind("Ctrl+Shift+B", app.window_launcher.interactive_box, guarded=False)

        # ==============================================================
        # Search presets / mode
        # ==============================================================
        self._bind("Ctrl+Shift+S", app.search_ctrl.next_text_embedding_preset, guarded=False)
        self._bind("Ctrl+B", app.return_to_browsing_mode, guarded=False)

        # ==============================================================
        # Seek to trigger (Ctrl+L — repeat last action)
        # ==============================================================
        self._bind("Ctrl+L", self._run_last_seek_to_trigger, guarded=False)

        # ==============================================================
        # F-keys
        # ==============================================================
        self._bind("F1", app.window_launcher.get_help_and_config, guarded=False)

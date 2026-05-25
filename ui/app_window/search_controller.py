"""
SearchController -- search and comparison execution logic.

Extracted from: set_search_for_media, set_search_for_text, set_search,
run_compare, _debounced_run_compare, _run_with_progress, _run_compare,
_validate_run, display_progress, get_search_media_path, get_compare_threshold,
get_inclusion_pattern, set_current_media_run_search, _set_media_run_search,
add_current_media_to_negative_search, negative_media_search,
next_text_embedding_preset, run_image_generation,
trigger_image_generation, run_image_generation_on_directory,
find_related_media_in_open_window.
"""

from __future__ import annotations

import os
import traceback
from typing import TYPE_CHECKING, Any, Callable, Optional

from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtWidgets import QFileDialog

from compare.compare_args import CompareArgs
from lib.debounce_qt import QtDebouncer
from ui.auth.password_utils import require_password
from utils.config import config
from utils.constants import CompareMode, ImageGenerationType, Mode, ProtectedActions, SortBy
from utils.logging_setup import get_logger
from utils.translations import I18N
from utils.utils import Utils

if TYPE_CHECKING:
    from compare.compare_manager import CompareManager
    from files.file_browser import FileBrowser
    from ui.app_window.app_window import AppWindow
    from ui.app_window.sidebar_panel import SidebarPanel

_ = I18N._
logger = get_logger("search_controller")


class ProgressListener:
    """Adapter that the compare engine calls to report progress."""

    def __init__(self, update_func: Callable[[str, Optional[int]], None]):
        self.update_func = update_func

    def update(self, context: str, percent_complete: Optional[int] = None) -> None:
        self.update_func(context, percent_complete)


class _CompareWorkerSignals(QObject):
    """Signals emitted by the background compare worker."""
    finished = Signal()
    error = Signal(str)
    progress = Signal(str, int)  # context, percent (-1 means indeterminate)


class _CompareWorker(QThread):
    """Runs the actual compare function in a background thread."""

    signals = _CompareWorkerSignals()

    def __init__(self, exec_func: Callable, args: list[Any]):
        super().__init__()
        self._exec_func = exec_func
        self._args = args
        self.signals = _CompareWorkerSignals()

    def run(self):
        from compare.base_compare import CompareCancelled
        try:
            self._exec_func(*self._args)
        except CompareCancelled:
            pass
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class SearchController:
    """
    Owns everything related to search and comparison execution.
    Reads search parameters from the SidebarPanel widgets and
    delegates compare execution to CompareManager.
    """

    def __init__(
        self,
        app_window: AppWindow,
        file_browser: FileBrowser,
        compare_manager: CompareManager,
        sidebar_panel: SidebarPanel,
    ):
        self._app = app_window
        self._fb = file_browser
        self._cm = compare_manager
        self._sidebar = sidebar_panel
        self._pending_compare: Optional[Callable] = None
        self._debouncer = QtDebouncer(
            parent=app_window,
            delay_seconds=0.3,
            callback=self._fire_pending_compare,
        )
        self._worker: Optional[_CompareWorker] = None
        self._img_gen_worker: Optional[_CompareWorker] = None

    def _fire_pending_compare(self) -> None:
        """Callback for the debouncer — invokes whatever compare was last scheduled."""
        fn = self._pending_compare
        self._pending_compare = None
        if fn is not None:
            fn()

    # ==================================================================
    # Search setup
    # ==================================================================
    @require_password(ProtectedActions.RUN_SEARCH)
    def set_search_for_media(self, event=None) -> None:
        """Set search mode to media search."""
        media_path = self.get_search_media_path()
        if media_path is None or media_path == "":
            if self._app.media_path is None:
                self._app.notification_ctrl.handle_error(
                    _("No media file selected."), title=_("Invalid Setting")
                )
            self._sidebar.search_media_path_box.clear()
            self._sidebar.search_media_path_box.setText(str(self._app.media_path))
        self.set_search()

    @require_password(ProtectedActions.RUN_SEARCH)
    def set_search_for_text(self, event=None) -> None:
        """Set search mode to text search."""
        search_text = self._sidebar.search_text_box.text()
        search_text_negative = self._sidebar.search_text_negative_box.text()
        if search_text.strip() == "" and search_text_negative.strip() == "":
            self._sidebar.search_text_box.setText("cat")
        self.set_search()

    @require_password(ProtectedActions.RUN_SEARCH)
    def set_negative_search_for_media(self, event=None) -> None:
        """
        Set search mode to include a negative search media file.

        Mirrors ``set_search_for_media`` behavior for the negative media input.
        """
        media_path = self.get_negative_search_media_path()
        if media_path is None or media_path == "":
            if self._app.media_path is None:
                self._app.notification_ctrl.handle_error(
                    _("No media file selected."), title=_("Invalid Setting")
                )
            self._sidebar.search_media_negative_path_box.clear()
            self._sidebar.search_media_negative_path_box.setText(str(self._app.media_path))
        self.set_search()

    def set_search(self, event=None) -> None:
        """
        Set the search media or text using the provided UI values.
        Set the mode based on the result.
        """
        args = CompareArgs()
        media_path = self.get_search_media_path()
        negative_media_path = self.get_negative_search_media_path()
        search_text = self._sidebar.search_text_box.text()
        search_text_negative = self._sidebar.search_text_negative_box.text()

        if search_text.strip() == "":
            search_text = None
        if search_text_negative.strip() == "":
            search_text_negative = None
        args.search_text = search_text
        args.search_text_negative = search_text_negative
        args.negative_search_media_path = negative_media_path

        if (
            args.search_text is not None
            or args.search_text_negative is not None
            or args.negative_search_media_path is not None
        ):
            self._cm.validate_compare_mode(
                CompareMode.text_search_modes(),
                _("Compare mode must be set to an embedding mode to search text embeddings"),
            )

        if media_path is not None and not os.path.isfile(media_path):
            media_path, _filter = QFileDialog.getOpenFileName(
                self._app,
                _("Select media file"),
                self._app.get_search_dir(),
                _("Media files") + " (*.jpg *.jpeg *.png *.tiff *.gif)",
            )

        if media_path is not None and media_path.strip() != "":
            if media_path.startswith(self._app.get_base_dir()):
                self._sidebar.search_media_path_box.setText(os.path.basename(media_path))
            self._app.search_dir = os.path.dirname(media_path)
            args.search_media_path = media_path
            self._cm.search_media_path = media_path
            self._app.media_navigator.show_searched_media()

        if args.not_searching():
            if self._app.mode != Mode.BROWSE:
                self._app.set_mode(Mode.GROUP)
        else:
            self._app.set_mode(Mode.SEARCH)

        self._app.media_frame.setFocus()
        self.run_compare(compare_args=args)

    # ==================================================================
    # Compare execution
    # ==================================================================
    @require_password(ProtectedActions.RUN_COMPARES)
    def run_compare(
        self,
        compare_args: CompareArgs = CompareArgs(),
        find_duplicates: bool = False,
    ) -> None:
        """Entry point for running a comparison (debounced)."""
        self._pending_compare = lambda: self._debounced_run_compare(
            compare_args, find_duplicates
        )
        self._debouncer.schedule()

    def refresh_compare(self, compare_args: CompareArgs = CompareArgs()) -> None:
        """Re-run comparison for windows with an active compare (see WindowManager)."""
        self.run_compare(compare_args=compare_args)

    def _debounced_run_compare(
        self, compare_args: CompareArgs, find_duplicates: bool
    ) -> None:
        """Actually enqueue the compare after debounce."""
        if not self._validate_run():
            return
        compare_args.find_duplicates = find_duplicates
        self._run_with_progress(self._run_compare, args=[compare_args])

    def _run_with_progress(self, exec_func: Callable, args: list[Any] = []) -> None:
        """Run *exec_func* in a background thread while showing a progress bar."""
        self._sidebar.start_progress_bar()

        worker = _CompareWorker(exec_func, args)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.progress.connect(self._on_progress)
        self._worker = worker
        worker.start()

    def is_compare_running(self) -> bool:
        """Return True while a compare worker is active in the background."""
        return self._worker is not None

    def _on_worker_finished(self) -> None:
        self._sidebar.stop_progress_bar()
        self._worker = None

    def _on_worker_error(self, error_text: str) -> None:
        self._app.notification_ctrl.alert(_("Error running compare"), error_text, kind="error")

    def _on_progress(self, context: str, percent: int) -> None:
        """Main-thread handler for progress updates from the worker."""
        if percent < 0:
            self._app.notification_ctrl.set_label_state(context)
        else:
            self._app.notification_ctrl.set_label_state(
                _("{0}: {1}% complete").format(context, percent)
            )

    def _on_img_gen_finished(self) -> None:
        """Release the image-generation worker so it can be garbage-collected."""
        self._img_gen_worker = None

    def _run_compare(self, args: CompareArgs = CompareArgs()) -> None:
        """Execute the comparison logic."""
        args.base_dir = self._app.get_base_dir()
        args.mode = self._app.mode
        args.recursive = self._fb.recursive

        # Apply all compare settings from CompareManager
        self._cm.apply_settings_to_args(args)

        # Settings still on the controller / sidebar
        args.inclusion_pattern = self.get_inclusion_pattern()
        args.include_videos = config.enable_videos
        args.include_gifs = config.enable_gifs
        args.include_pdfs = config.enable_pdfs
        args.use_matrix_comparison = False
        args.listener = ProgressListener(update_func=self.display_progress)
        args.app_actions = self._app.app_actions
        self._cm.run(args)

    def _validate_run(self) -> bool:
        """Validate that the current state allows running a compare."""
        base_dir = self._app.get_base_dir()
        if not base_dir or base_dir == "" or base_dir == ".":
            ok = self._app.notification_ctrl.alert(
                _("Confirm comparison"),
                _("No base directory has been set, will use current base directory of ")
                + f"{base_dir}\n\n" + _("Are you sure you want to proceed?"),
                kind="askokcancel",
            )
            return ok
        return True

    def display_progress(self, context: str, percent_complete: Optional[int] = None) -> None:
        """
        Thread-safe progress callback invoked by the compare engine.

        Emits the worker's progress signal so the update happens on the main thread.
        """
        if self._worker is not None:
            self._worker.signals.progress.emit(
                context, int(percent_complete) if percent_complete is not None else -1
            )
        else:
            # Fallback: called outside a worker (shouldn't happen normally)
            self._app.notification_ctrl.set_label_state(context)

    # ==================================================================
    # Search helpers
    # ==================================================================
    def get_search_media_path(self) -> Optional[str]:
        """Read the search media path from the sidebar entry."""
        media_path = self._sidebar.search_media_path_box.text().strip()
        if not media_path:
            self._cm.search_media_path = None
            return None
        search_file = Utils.get_valid_file(self._app.get_base_dir(), media_path)
        if search_file is None:
            search_file = Utils.get_valid_file(self._app.get_search_dir(), media_path)
            if search_file is None:
                self._app.notification_ctrl.handle_error(
                    "Search file is not a valid file for base dir.",
                    title="Invalid search file",
                )
                raise AssertionError("Search file is not a valid file.")
        return search_file

    def get_negative_search_media_path(self) -> Optional[str]:
        """Read the negative-search media path from the dedicated sidebar entry."""
        media_path = self._sidebar.search_media_negative_path_box.text().strip()
        if not media_path:
            return None
        search_file = Utils.get_valid_file(self._app.get_base_dir(), media_path)
        if search_file is None:
            search_file = Utils.get_valid_file(self._app.get_search_dir(), media_path)
            if search_file is None:
                self._app.notification_ctrl.handle_error(
                    "Negative search file is not a valid file for base dir.",
                    title="Invalid negative search file",
                )
                raise AssertionError("Negative search file is not a valid file.")
        return search_file

    def get_compare_threshold(self) -> float:
        """Get compare threshold from CompareManager, with fallback to config."""
        threshold = self._cm.get_threshold()
        if threshold is not None:
            return threshold

        primary_mode = self._cm.compare_mode
        if primary_mode == CompareMode.COLOR_MATCHING:
            return config.color_diff_threshold
        return config.embedding_similarity_threshold

    def get_inclusion_pattern(self) -> Optional[str]:
        """Read the inclusion pattern from the sidebar entry."""
        text = self._sidebar.inclusion_pattern.text().strip()
        return text if text else None

    @require_password(ProtectedActions.RUN_SEARCH)
    def set_current_media_run_search(self, event=None, base_dir: Optional[str] = None) -> None:
        """Use the current media file as the search target and run search."""
        from ui.app_window.window_manager import WindowManager

        if base_dir is None:
            window, dirs = WindowManager.get_other_window_or_self_dir(
                self._app, allow_current_window=True, prefer_compare_window=True
            )
            if window is None:
                self._app.window_launcher.open_recent_directory_window(
                    extra_callback_args=(self.set_current_media_run_search, dirs)
                )
                return
            base_dir = dirs[0]
        else:
            window = WindowManager.get_window(base_dir=base_dir)

        if self._app.mode == Mode.BROWSE:
            pass

        filepath = self._app.media_navigator.get_active_media_filepath()
        if filepath:
            window.search_ctrl._set_media_run_search(filepath)
        else:
            self._app.notification_ctrl.handle_error(_("Failed to get active media filepath"))

    def _set_media_run_search(self, filepath: str) -> None:
        """Set the search media path and trigger the search."""
        base_dir = self._app.get_base_dir()
        if filepath.startswith(base_dir):
            filepath = filepath[len(base_dir) + 1 :]
        self._sidebar.search_media_path_box.setText(filepath)
        self.set_search()

    @require_password(ProtectedActions.RUN_SEARCH)
    def add_current_media_to_negative_search(self, event=None, base_dir: Optional[str] = None) -> None:
        """Add the current media file to the negative search list."""
        from ui.app_window.window_manager import WindowManager

        filepath = self._app.media_navigator.get_active_media_filepath()
        if filepath:
            if base_dir is None:
                window, dirs = WindowManager.get_other_window_or_self_dir(
                    self._app, allow_current_window=True, prefer_compare_window=True
                )
                if window is None:
                    self._app.window_launcher.open_recent_directory_window(
                        extra_callback_args=(self.add_current_media_to_negative_search, dirs)
                    )
                    return
                base_dir = dirs[0]
            else:
                window = WindowManager.get_window(base_dir=base_dir)
            window.search_ctrl.negative_media_search(filepath)
        else:
            self._app.notification_ctrl.handle_error(_("Failed to get active media filepath"))

    def negative_media_search(self, filepath: str) -> None:
        """Set up a negative media search."""
        base_dir = self._app.get_base_dir()
        display_path = filepath
        if filepath.startswith(base_dir):
            display_path = filepath[len(base_dir) + 1 :]
        self._sidebar.search_media_negative_path_box.clear()
        self._sidebar.search_media_negative_path_box.setText(display_path)
        self.set_search()

    def next_text_embedding_preset(self, event=None) -> None:
        """Cycle to the next text embedding search preset."""
        preset = config.next_text_embedding_search_preset()
        if preset is None:
            self._app.notification_ctrl.alert(
                _("No Text Search Presets Found"),
                _("No text embedding search presets found. Set them in the config.json file."),
            )
            return

        self._sidebar.search_media_path_box.clear()
        self._sidebar.search_media_negative_path_box.clear()
        self._sidebar.search_text_box.clear()
        self._sidebar.search_text_negative_box.clear()

        if isinstance(preset, dict):
            if "negative" in preset:
                self._sidebar.search_text_negative_box.setText(preset["negative"])
            if "positive" in preset:
                self._sidebar.search_text_box.setText(preset["positive"])
        elif isinstance(preset, str):
            self._sidebar.search_text_box.setText(preset)

        self.set_search()

    # ==================================================================
    # Image generation
    # ==================================================================
    def _prompt_for_redo_prompt_adjustment(self, media_path: str) -> Optional[tuple[str, str]]:
        """Show the rerun prompt editor; returns ``(positive, negative)`` or ``None`` on cancel."""
        from ui.image.rerun_prompt_adjustment_window_qt import RerunPromptAdjustmentWindow

        return RerunPromptAdjustmentWindow.prompt_adjustment(self._app, media_path)

    def trigger_image_generation(self, event=None) -> None:
        """Open the image generation dialog."""
        from ui.image.media_details import MediaDetails

        # In Tkinter, shift state was checked from event; in Qt we don't
        # have the event from QShortcut, so always pass False here.
        # A separate Shift-keyed binding can be added if needed.
        MediaDetails.run_image_generation_static(self._app.app_actions, modify_call=False)

    @require_password(ProtectedActions.RUN_IMAGE_GENERATION)
    def run_image_generation(
        self,
        event=None,
        _type: Optional[str] = None,
        media_path: Optional[str] = None,
        modify_call: bool = False,
    ) -> None:
        """Trigger image generation via SD runner."""
        from extensions.sd_runner_client import SDRunnerClient
        from ui.image.media_details import MediaDetails

        if media_path is None:
            media_path = self._get_media_path()
        if media_path is None:
            return
        if _type is None:
            _type = MediaDetails.get_image_specific_generation_mode()

        prompt_overrides = None
        if _type == ImageGenerationType.REDO_PROMPT:
            prompt_overrides = self._prompt_for_redo_prompt_adjustment(media_path)
            if prompt_overrides is None:
                return

        sd_client = SDRunnerClient()

        def _do_run() -> None:
            run_kwargs = {"append": modify_call}
            if prompt_overrides is not None:
                run_kwargs["positive_prompt"] = prompt_overrides[0]
                run_kwargs["negative_prompt"] = prompt_overrides[1]
            sd_client.run(_type, media_path, **run_kwargs)
            MediaDetails.previous_image_generation_adapter_path = media_path

        worker = _CompareWorker(_do_run, [])
        worker.signals.finished.connect(
            lambda: self._app.notification_ctrl.toast(_("Running image gen: ") + str(_type))
        )
        worker.signals.error.connect(
            lambda msg: self._app.notification_ctrl.handle_error(
                _("Error running image generation:") + "\n" + msg, title=_("Warning")
            )
        )
        worker.signals.finished.connect(lambda: self._on_img_gen_finished())
        self._img_gen_worker = worker
        worker.start()

    @require_password(ProtectedActions.RUN_IMAGE_GENERATION)
    def run_image_generation_on_directory(
        self, event=None, _type: Optional[str] = None, media_path: Optional[str] = None
    ) -> None:
        """Run image generation on all files in the directory."""
        from extensions.sd_runner_client import SDRunnerClient
        from ui.image.media_details import MediaDetails

        if media_path is None:
            media_path = self._get_media_path()
        if media_path is None:
            return
        directory_path = os.path.dirname(media_path)
        if _type is None:
            _type = MediaDetails.get_image_specific_generation_mode()

        sd_client = SDRunnerClient()

        def _do_run() -> None:
            sd_client.run_on_directory(_type, directory_path)
            # Keep last-generation target path for Ctrl+Enter/Cancel/Revert flows.
            # This may be either a single file path or a directory path.
            MediaDetails.previous_image_generation_adapter_path = directory_path

        worker = _CompareWorker(_do_run, [])
        worker.signals.finished.connect(
            lambda: self._app.notification_ctrl.toast(
                _("Running image gen on directory: ") + str(_type)
            )
        )
        worker.signals.error.connect(
            lambda msg: self._app.notification_ctrl.handle_error(
                _("Error running image generation:") + "\n" + msg, title=_("Warning")
            )
        )
        worker.signals.finished.connect(lambda: self._on_img_gen_finished())
        self._img_gen_worker = worker
        worker.start()

    # ==================================================================
    # Related images (cross-window)
    # ==================================================================
    def find_related_media_in_open_window(self, event=None, base_dir: Optional[str] = None) -> None:
        """Navigate to the next downstream related media file in another window."""
        from ui.files.marked_file_mover_qt import MarkedFiles
        from ui.image.media_details import MediaDetails
        from ui.app_window.window_manager import WindowManager

        if base_dir is None:
            window, dirs = WindowManager.get_other_window_or_self_dir(self._app)
            if window is None:
                self._app.window_launcher.open_recent_directory_window(
                    extra_callback_args=(self.find_related_media_in_open_window, dirs)
                )
                return
            base_dir = dirs[0]
        else:
            window = WindowManager.get_window(base_dir=base_dir)

        media_to_use = (
            self._app.media_path
            if len(MarkedFiles.file_marks) != 1
            else MarkedFiles.file_marks[0]
        )

        if self._app.check_many_files(window, action="find related media"):
            return

        next_related_image = MediaDetails.next_downstream_related_image(
            media_to_use, base_dir, self._app.app_actions
        )
        if next_related_image is not None:
            window.media_navigator.go_to_file(search_text=next_related_image)
            window.media_frame.setFocus()
        else:
            self._app.notification_ctrl.toast(
                _("No downstream related image(s) found in {0}").format(base_dir)
            )

    # ==================================================================
    # Private helpers
    # ==================================================================
    def _get_media_path(self) -> Optional[str]:
        """Get the current media path, falling back to prev if delete-locked."""
        if self._app.delete_lock:
            media_path = self._app.prev_media_path
        else:
            media_path = self._app.media_navigator.get_active_media_filepath()
        if not media_path:
            self._app.notification_ctrl.handle_error(
                _("Failed to get active media filepath"), title=_("Warning")
            )
        return media_path

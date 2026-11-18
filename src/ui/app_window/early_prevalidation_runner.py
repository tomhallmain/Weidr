"""Runs slideshow early prevalidation (``compare.early_prevalidation``) for
one window on a background thread.

A run starts when a slide is shown during the classic slideshow (after the
configured delay) and is cancelled by the next slide, the end of the
slideshow or the window closing; cancellation takes effect between two
candidates. Each evaluation holds ``ClassifierActionsManager.evaluation_lock``,
which ``CompareWrapper.skip_media`` waits for off the GUI thread, so the
slideshow reaching a file still being evaluated waits for that result.

The worker's callbacks are posted to the GUI thread without waiting. That
keeps the worker from ever blocking on the GUI thread while it holds the
lock, so GUI-thread code may wait on the lock without deadlocking.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable, Optional

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from compare.action_callbacks import ActionCallbacks
from compare.classifier_actions_manager import ClassifierActionsManager
from compare.early_prevalidation import (
    is_costly_candidate,
    run_early_prevalidation,
    upcoming_candidates,
)
from files.marked_files import MarkedFiles
from utils.config import config
from utils.constants import Mode
from utils.logging_setup import get_logger
from utils.translations import _

if TYPE_CHECKING:
    from compare.compare_manager import CompareManager
    from files.file_browser import FileBrowser
    from ui.app_window.app_window import AppWindow

logger = get_logger("early_prevalidation_runner")

# How long closing a window waits for a candidate still being evaluated. A
# video's frame sampling can take longer; past this the daemon thread finishes
# on its own and its callbacks are dropped.
_SHUTDOWN_JOIN_SECONDS = 2.0


class EarlyPrevalidationRunner(QObject):
    _posted = Signal(object)

    def __init__(
        self,
        app_window: AppWindow,
        file_browser: FileBrowser,
        compare_manager: CompareManager,
    ) -> None:
        super().__init__(app_window)
        self._app = app_window
        self._fb = file_browser
        self._cm = compare_manager
        self._cancel_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._start_timer = QTimer(self)
        self._start_timer.setSingleShot(True)
        self._start_timer.timeout.connect(self._start)
        self._posted.connect(self._run_posted)

    # ------------------------------------------------------------------
    # GUI-thread API
    # ------------------------------------------------------------------
    def on_media_shown(self, media_path: Optional[str]) -> None:
        """Cancel any run and schedule one for the slide now on screen."""
        self.cancel()
        if not media_path or not self._should_run():
            return
        delay_seconds = max(0.0, float(config.slideshow_early_prevalidation_start_delay_seconds))
        # Even with no delay the start goes through the event loop, so the
        # slide is painted before the snapshot below is taken.
        self._start_timer.start(int(delay_seconds * 1000))

    def cancel(self) -> None:
        self._start_timer.stop()
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._cancel_event = None

    def shutdown(self) -> None:
        """Stop for good and wait briefly for a candidate still being evaluated,
        so no callback reaches this object after the window is gone."""
        self._closed = True
        self.cancel()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_SHUTDOWN_JOIN_SECONDS)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _should_run(self) -> bool:
        return (
            not self._closed
            and config.slideshow_early_prevalidation
            and self._app.slideshow_config.slideshow_running
            and self._app.mode == Mode.BROWSE
            and self._cm.prevalidations_running
            and bool(ClassifierActionsManager.prevalidations)
        )

    @Slot()
    def _start(self) -> None:
        if not self._should_run():
            return
        files = list(self._fb.get_files())
        cursor = self._fb.get_cursor()
        if not 0 <= cursor < len(files) or files[cursor] != self._app.media_path:
            return
        candidates = upcoming_candidates(
            files, cursor, int(config.slideshow_early_prevalidation_max_candidates)
        )
        if not candidates:
            return
        base_dir = self._app.get_base_dir()
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self._thread = threading.Thread(
            target=self._run,
            args=(candidates, base_dir, cancel_event),
            name="early-prevalidation",
            daemon=True,
        )
        self._thread.start()

    def _run(self, candidates, base_dir: str, cancel_event: threading.Event) -> None:
        callbacks = self._make_callbacks()

        def should_abort() -> bool:
            return (
                cancel_event.is_set()
                or not self._app.slideshow_config.slideshow_running
                or not self._cm.prevalidations_running
                or self._app.get_base_dir() != base_dir
            )

        result = run_early_prevalidation(
            candidates,
            prevalidate=lambda path: self._cm.early_prevalidate(path, callbacks),
            is_costly=lambda path: is_costly_candidate(path, base_dir),
            is_hidden=self._cm.is_hidden_media,
            should_abort=should_abort,
        )
        if config.debug2:
            logger.info(
                "Early prevalidation finished: evaluated=%s stop=%s at=%s",
                result.evaluated, result.stop_reason, result.stopped_at,
            )

    def _make_callbacks(self) -> ActionCallbacks:
        app_actions = self._app.app_actions

        def post(func: Callable, *args, **kwargs) -> None:
            if self._closed:
                return
            try:
                self._posted.emit(lambda: func(*args, **kwargs))
            except RuntimeError:
                # The window was destroyed after the join timed out.
                pass

        return ActionCallbacks(
            hide_callback=lambda path: post(app_actions.hide_media, path),
            notify_callback=lambda *args, **kwargs: post(app_actions.title_notify, *args, **kwargs),
            add_mark_callback=lambda path: post(
                MarkedFiles.add_mark_if_not_present, path, app_actions=app_actions
            ),
            blur_callback=lambda path: post(app_actions.request_media_blur, path),
            generate_callback=lambda path, edit_suffix=None, target_dir=None: post(
                app_actions.run_image_generation,
                media_path=path, edit_suffix=edit_suffix, suppress_toast=True, target_dir=target_dir,
            ),
        )

    @Slot(object)
    def _run_posted(self, func: Callable) -> None:
        if self._closed:
            return
        try:
            func()
        except Exception:
            logger.exception(_("Early prevalidation callback failed."))

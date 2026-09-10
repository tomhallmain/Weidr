"""Firing a callback on a fixed interval, without an event loop.

The Qt path uses a QTimer, which delivers on the GUI thread. That needs a
running QApplication, so periodic work -- saving the info cache, for one --
stops happening the moment there is no window. The interval itself is not a
Qt concept, so it becomes a port with two implementations.

The callback here runs on the scheduler's own thread, not a main thread; a
callback touching shared state does its own locking, as with any thread. The
Qt implementation keeps its main-thread delivery, which is the whole reason it
stays the default wherever a window exists.

A non-positive interval means "do not schedule". Both implementations treat it
that way rather than spinning or raising.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from utils.logging_setup import get_logger

logger = get_logger("periodic_scheduler")


class ThreadedScheduler:
    """Repeats a callback on a daemon thread until stopped."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()

    def start(self, interval_seconds: float, callback: Callable[[], None]) -> None:
        """Begin firing *callback* every *interval_seconds*.

        Any previous schedule is stopped first, so starting twice replaces the
        schedule instead of leaving an orphaned thread behind.
        """
        self.stop()
        if interval_seconds <= 0:
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._loop,
            args=(float(interval_seconds), callback, stop_event),
            daemon=True,
            name="periodic-scheduler",
        )
        with self._lock:
            self._stop_event = stop_event
            self._thread = thread
        thread.start()

    def _loop(
        self,
        interval_seconds: float,
        callback: Callable[[], None],
        stop_event: threading.Event,
    ) -> None:
        # wait() returning True means stop was requested, so cancellation takes
        # effect immediately instead of after the rest of the interval.
        while not stop_event.wait(interval_seconds):
            try:
                callback()
            except Exception:
                # One failing tick must not end the schedule.
                logger.exception("Periodic callback failed.")

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            self._thread = None
            self._stop_event = None
        if stop_event is not None:
            stop_event.set()

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None

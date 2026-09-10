"""Qt implementation of the periodic-scheduler port.

Counterpart to utils.periodic_scheduler.ThreadedScheduler. Kept here so the
port's callers do not import PySide6, and kept the default wherever a window
exists because QTimer delivers on the GUI thread -- which a callback that
touches widgets requires.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QTimer


class QtScheduler:
    def __init__(self) -> None:
        self._timer: Optional[QTimer] = None

    def start(self, interval_seconds: float, callback: Callable[[], None]) -> None:
        """Begin firing *callback* every *interval_seconds* on the GUI thread.

        Any previous timer is stopped first: starting twice used to leave the
        earlier QTimer running and unreferenced, firing alongside the new one.
        """
        self.stop()
        interval_ms = int(interval_seconds * 1000)
        if interval_ms <= 0:
            return
        timer = QTimer()
        timer.timeout.connect(callback)
        timer.start(interval_ms)
        self._timer = timer

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def is_running(self) -> bool:
        return self._timer is not None

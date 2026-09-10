from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional


class DebouncedGenerateQueue:
    """Serialises generate_callback calls with a minimum inter-dispatch interval.

    Prevents burst requests from reaching the cross-process generator.
    Scoped to a single pipeline batch run; call shutdown() when the batch
    is complete to drain remaining items and stop the worker thread.

    If natural spacing between GENERATE outcomes already exceeds
    dispatch_interval (e.g. due to classifier inference time), the worker
    sleeps for zero additional time and adds no latency.
    """

    _SENTINEL = object()

    def __init__(self, dispatch_interval: float = 2.0) -> None:
        self.dispatch_interval = dispatch_interval
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="generate-dispatch")
        self._thread.start()

    def submit(self, fn: Callable, *args, **kwargs) -> None:
        """Queue a generate_callback call for deferred dispatch."""
        self._queue.put((fn, args, kwargs))

    def _worker(self) -> None:
        last_dispatch: float = 0.0
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            fn, args, kwargs = item
            remaining = self.dispatch_interval - (time.monotonic() - last_dispatch)
            if remaining > 0:
                time.sleep(remaining)
            fn(*args, **kwargs)
            last_dispatch = time.monotonic()

    def shutdown(self) -> None:
        """Drain remaining queued items, dispatch them, then stop the worker."""
        self._queue.put(self._SENTINEL)
        self._thread.join()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

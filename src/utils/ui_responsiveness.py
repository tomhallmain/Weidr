"""Keeping a user interface responsive during long synchronous work.

Some operations run on the thread that also draws the window: a purge loop over
every similarity group, or frame-sampling a video for prevalidation. Left
alone they freeze the display until they finish, so the code pushes the event
loop along by hand -- pumping pending events between iterations, or running the
slow part on a worker thread and waiting on a nested loop.

Both are properties of having a window, not of the work. Expressed as a port,
the logic stops naming Qt and a caller with no event loop supplies the
implementation that does nothing, because there is nothing to keep responsive.

Two operations, matching the two situations:

- yield_to_ui() -- let the display catch up mid-loop.
- run_off_thread(func) -- run func elsewhere, block until it returns, and keep
  the interface alive meanwhile.

run_off_thread reports a failure as None rather than raising. That mirrors what
the Qt path already does: an exception inside the worker thread never reaches
the caller, leaving the result unset. Both implementations log it, which is
more than the thread did on its own.
"""

from __future__ import annotations

from typing import Any, Callable

from utils.logging_setup import get_logger

logger = get_logger("ui_responsiveness")


class NullResponsiveness:
    """For callers with no event loop: nothing to yield to, nothing to unblock.

    run_off_thread runs the callable inline. A thread would add nothing here --
    its only purpose in the Qt path is to leave the drawing thread free.
    """

    def yield_to_ui(self) -> None:
        return None

    def run_off_thread(self, func: Callable[[], Any]) -> Any:
        try:
            return func()
        except Exception:
            logger.exception("Background work failed.")
            return None

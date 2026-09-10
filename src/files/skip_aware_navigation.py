"""Advance a file-browser cursor past files a caller-supplied predicate
vetoes, and past files that no longer exist on disk.

Extracted from ``ui/app_window/media_navigator.py``'s
``show_next_media``/``show_prev_media``, which had this loop inline and
Qt-only -- so a headless session driving the same kind of file browser had
no skip-awareness at all. Takes the veto as a predicate rather than a
``CompareManager`` directly so this module never has to import ``compare/``.
"""

import os


def advance_past_skipped(file_browser, skip, backward, start, current):
    """Continue stepping *file_browser* from *current* until it finds a
    file *skip* doesn't veto and that still exists on disk, or wraps back
    around to *start*.

    *current* is a candidate the caller already produced with one step of
    its own (``file_browser.next_file()``/``previous_file()``) -- this
    function only continues from there, it does not take that first step
    itself. *start* is the file the caller began stepping from, used to
    detect a full wrap with nothing acceptable found.
    """
    step = file_browser.previous_file if backward else file_browser.next_file
    while current != start and (skip(current) or not os.path.isfile(current)):
        current = step()
    return current

"""Closing the primary window via the title-bar X asks for the same
confirmation as Ctrl+Q.

Window-level patches live in a MonkeyPatch.context() inside the test body so
they are undone before the fixture teardown calls the real on_closing().
"""

import pytest
from PySide6.QtWidgets import QApplication


def _patch_close_path(mp, win, answer, prompts, closings, quits):
    # Instance attribute: takes precedence over conftest's class-level override.
    mp.setattr(win, "_confirm_quit", lambda: prompts.append(True) or answer)
    mp.setattr(win, "on_closing", lambda: closings.append(True))
    mp.setattr(QApplication.instance(), "quit", lambda: quits.append(True))
    mp.setattr(win, "is_secondary", lambda: False)


def test_declining_keeps_primary_window_open(window):
    win = window
    prompts, closings, quits = [], [], []
    with pytest.MonkeyPatch.context() as mp:
        _patch_close_path(mp, win, False, prompts, closings, quits)
        win.close()

    assert len(prompts) == 1
    assert win.isVisible()
    assert not win._closing
    assert closings == [] and quits == []


def test_confirming_closes_primary_window_and_quits(window):
    win = window
    prompts, closings, quits = [], [], []
    with pytest.MonkeyPatch.context() as mp:
        _patch_close_path(mp, win, True, prompts, closings, quits)
        win.close()

    assert len(prompts) == 1
    assert win._closing
    assert closings == [True]
    assert quits == [True]

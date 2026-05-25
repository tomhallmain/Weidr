"""
Qt media widget fixtures.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication


def _teardown_media_frame(frame) -> None:
    """Stop timers, tear down VLC without blocking on hung libvlc stop threads, destroy UI."""
    try:
        frame._invalidate_pending_image_promotion()
    except Exception:
        pass
    try:
        frame._mouse_poll_timer.stop()
    except Exception:
        pass
    try:
        frame._playback_timer.stop()
    except Exception:
        pass
    try:
        overlay = frame._controls_overlay
        overlay.hide()
        overlay.close()
    except Exception:
        pass
    try:
        dispose = getattr(frame, "dispose_vlc_for_test_teardown", None)
        if callable(dispose):
            dispose()
        else:
            frame.dispose_vlc()
    except Exception:
        pass
    try:
        frame.close()
        frame.deleteLater()
    except Exception:
        pass


def sweep_qt_media_widgets() -> None:
    """Close any leaked MediaFrame instances and process pending Qt events."""
    app = QApplication.instance()
    if app is None:
        return

    try:
        from ui.app_window.media_frame import MediaFrame
    except Exception:
        MediaFrame = None  # type: ignore[misc, assignment]

    if MediaFrame is not None:
        for widget in list(app.allWidgets()):
            if isinstance(widget, MediaFrame):
                _teardown_media_frame(widget)

    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass

    app.processEvents()


@pytest.fixture
def media_frame(qtbot):
    """A visible MediaFrame, fully torn down after each test (avoids Qt/VLC hang)."""
    from ui.app_window.media_frame import MediaFrame

    frame = MediaFrame()
    qtbot.addWidget(frame)
    frame.show()
    qtbot.waitExposed(frame)
    # The mouse-poll timer fires every 100 ms and calls show_overlay() on the
    # controls overlay, which calls show() on a top-level positioned widget.
    # In a headless offscreen environment that segfaults accessing screen
    # geometry, so stop the timer for all media frame tests.
    frame._mouse_poll_timer.stop()
    yield frame
    _teardown_media_frame(frame)

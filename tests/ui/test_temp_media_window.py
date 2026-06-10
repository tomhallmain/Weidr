"""UI smoke tests for TempMediaWindow (temp canvas viewer)."""

import os

import pytest
from PySide6.QtWidgets import QApplication

from ui.image.media_details import MediaDetails
from ui.image.temp_media_window import TempMediaWindow


def _close_temp_canvas() -> None:
    canvas = MediaDetails.temp_media_canvas
    if canvas is not None:
        try:
            canvas.close()
            canvas.deleteLater()
        except RuntimeError:
            pass
    MediaDetails.temp_media_canvas = None
    TempMediaWindow._instance = None
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture
def temp_media_window(window_with_dir, qtbot, monkeypatch):
    """Create TempMediaWindow and load a small PNG."""
    app_win, media_dir = window_with_dir
    media_path = app_win.file_browser.get_files()[0]

    monkeypatch.setattr(
        "ui.app_window.media_frame.MediaFrame.show_video",
        lambda self, path: None,
    )

    MediaDetails.temp_media_canvas = None
    MediaDetails.set_temp_media_canvas(app_win, media_path, app_win.app_actions)
    canvas = MediaDetails.temp_media_canvas
    assert canvas is not None
    canvas.create_media(media_path)
    qtbot.addWidget(canvas)
    qtbot.waitExposed(canvas, timeout=3000)

    yield canvas, app_win, media_path

    _close_temp_canvas()


@pytest.fixture(autouse=True)
def _cleanup_temp_media_singleton():
    yield
    _close_temp_canvas()


class TestTempMediaWindow:
    def test_create_media_sets_path_and_title(self, temp_media_window, qtbot):
        canvas, _app_win, media_path = temp_media_window

        assert canvas._media_path == media_path
        assert media_path in canvas.windowTitle()
        qtbot.waitUntil(
            lambda: canvas._media_frame.path == media_path,
            timeout=5000,
        )

    def test_open_temp_media_canvas_reuses_canvas(
        self, window_with_dir, qtbot, monkeypatch
    ):
        app_win, media_dir = window_with_dir
        first = os.path.join(media_dir, "img01.png")
        second = os.path.join(media_dir, "img02.png")

        monkeypatch.setattr(
            "ui.app_window.media_frame.MediaFrame.show_video",
            lambda self, path: None,
        )
        monkeypatch.setattr(
            app_win.app_actions,
            "get_window",
            lambda **kwargs: None,
        )

        MediaDetails.temp_media_canvas = None
        MediaDetails.open_temp_media_canvas(
            master=app_win,
            media_path=first,
            app_actions=app_win.app_actions,
            skip_get_window_check=True,
        )
        canvas = MediaDetails.temp_media_canvas
        assert canvas is not None
        assert canvas._media_path == first

        MediaDetails.open_temp_media_canvas(
            master=app_win,
            media_path=second,
            app_actions=app_win.app_actions,
            skip_get_window_check=True,
        )
        assert MediaDetails.temp_media_canvas is canvas
        assert canvas._media_path == second

        _close_temp_canvas()

    def test_clear_media_resets_title(self, temp_media_window):
        canvas, _app_win, media_path = temp_media_window

        canvas.clear_media()

        assert canvas._media_path is None
        assert not canvas._media_frame.media_displayed
        assert canvas.windowTitle() != media_path


class TestGoToFileRelatedImage:
    def test_go_to_file_uses_full_path_for_temp_canvas(
        self, window_with_dir, monkeypatch
    ):
        """go_to_file must pass the resolved full path to open_temp_media_canvas,
        not the bare filename, when the file exists on disk but isn't found by
        any browser.  This is the Qt equivalent of the old Tkinter
        FileNotFoundError triggered when get_window called go_to_file with
        os.path.basename(media_path) and the fallback branch forwarded only
        the basename to the temp canvas."""
        app_win, media_dir = window_with_dir
        basename = "img01.png"
        full_path = os.path.join(media_dir, basename)

        # Simulate the browser not finding the file (e.g. filtered out), so
        # go_to_file falls through to the on-disk fallback branch.
        monkeypatch.setattr(
            app_win.file_browser, "find", lambda *args, **kwargs: None
        )

        opened_paths = []
        monkeypatch.setattr(
            "ui.image.media_details.MediaDetails.open_temp_media_canvas",
            lambda **kwargs: opened_paths.append(kwargs.get("media_path")),
        )

        app_win.media_navigator.go_to_file(search_text=basename)

        assert len(opened_paths) == 1, "open_temp_media_canvas was not called"
        assert os.path.normpath(opened_paths[0]) == os.path.normpath(full_path)

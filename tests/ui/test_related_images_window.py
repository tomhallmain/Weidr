"""UI tests for RelatedImagesWindow and the all-windows downstream search.

Covers docs/downstream-search-all-windows.md: window construction with all
six actions, result reporting into the persistent result area (including
stopping after close), and the FileMarksController all-windows aggregation
(de-duplication, oversized-window skip, no-result path).

IMPORTANT — patch scoping: the controller tests patch
WindowManager.get_open_windows with fakes. The shared `monkeypatch` fixture
is instantiated before window_with_dir (isolated_singletons requests it), so
it undoes AFTER the AppWindow teardown — leaving the fakes visible to
AppWindow.closeEvent, which iterates open windows and accesses attributes
the fakes don't have (historically: a teardown AttributeError wrapped in a
recursive Qt eventFilter traceback whose formatting crashed the GC). All
window-level patches therefore live in a pytest.MonkeyPatch.context() inside
the test body, restored before teardown runs.
"""

import os
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from files.marked_files import MarkedFiles
from ui.app_window.window_manager import WindowManager
from ui.files.related_images_window_qt import RelatedImagesWindow


def _find_related_images_window() -> RelatedImagesWindow | None:
    for w in QApplication.topLevelWidgets():
        if isinstance(w, RelatedImagesWindow) and w.isVisible():
            return w
    return None


def _close_all_related_images_windows() -> None:
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        if isinstance(widget, RelatedImagesWindow):
            widget.close()
            widget.deleteLater()
    app.processEvents()


@pytest.fixture(autouse=True)
def _related_images_window_cleanup():
    """Delete dialogs while the parent AppWindow is still alive (same pattern
    as test_favorites_window)."""
    yield
    _close_all_related_images_windows()


def _open_window(win, qtbot) -> RelatedImagesWindow:
    win.window_launcher.open_related_images_window()
    qtbot.waitUntil(lambda: _find_related_images_window() is not None, timeout=5000)
    # Deliberately NOT qtbot.addWidget: the window is WA_DeleteOnClose, so
    # the in-test close() destroys it and qtbot's finalizer would then call
    # close() on a dead wrapper. The autouse cleanup fixture handles strays.
    return _find_related_images_window()


class TestRelatedImagesWindow:
    def test_open_lists_all_actions(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        rw = _open_window(win, qtbot)
        texts = [b.text() for b in rw.findChildren(QPushButton)]
        rw.close()
        assert len(texts) == 6
        assert any("all open windows" in t.lower() for t in texts)

    def test_result_area_shows_reported_results(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        rw = _open_window(win, qtbot)
        win.app_actions.notify_related_images_result(
            "downstream test result 123", action_label="Test Action"
        )
        text = rw._result_label.text()
        rw.close()
        assert "downstream test result 123" in text
        assert "Test Action" in text

    def test_result_area_stops_after_close(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        rw = _open_window(win, qtbot)
        results = rw._results  # plain list, survives widget destruction
        win.app_actions.notify_related_images_result("before close message")
        assert len(results) == 1
        rw.close()  # WA_DeleteOnClose → destroyed; Qt auto-disconnects
        QApplication.processEvents()
        win.app_actions.notify_related_images_result("after close message")
        QApplication.processEvents()
        assert len(results) == 1


class TestAllWindowsDownstreamSearch:
    @staticmethod
    def _fake_secondary_window(base_dir: str, slow: bool = False):
        return SimpleNamespace(
            base_dir=base_dir,
            is_secondary=True,  # closeEvent iterates windows and reads this
            file_browser=SimpleNamespace(
                has_confirmed_dir=lambda: not slow,
                is_slow_total_files=lambda threshold=2000: slow,
            ),
            file_marks_ctrl=SimpleNamespace(go_to_mark=lambda: None),
            media_frame=SimpleNamespace(setFocus=lambda: None),
        )

    @staticmethod
    def _prepare(win, mp, results, windows, toasts):
        """Apply the window-level patches to *mp* — a MonkeyPatch.context()
        instance scoped to the test body, so everything is undone before the
        AppWindow teardown (see module docstring)."""
        mp.setattr(
            "ui.app_window.file_marks_controller.get_downstream_related_images",
            lambda media, base_dir, aa, force_refresh=False, quiet=False:
                results.get(base_dir),
        )
        mp.setattr(
            WindowManager, "get_open_windows", classmethod(lambda cls: windows)
        )
        mp.setattr(
            win.notification_ctrl, "toast", lambda msg, **kw: toasts.append(msg)
        )
        win.media_path = win.file_browser.get_files()[0]
        MarkedFiles.file_marks = []

    def test_aggregates_and_dedupes_across_windows(
        self, window_with_dir, qtbot, tmp_path
    ):
        win, media_dir = window_with_dir
        other_dir = str(tmp_path / "other")
        os.makedirs(other_dir, exist_ok=True)
        fake2 = self._fake_secondary_window(other_dir)
        toasts: list = []
        goto_calls: list = []
        reports: list = []

        def _capture(message, action_label, data):
            reports.append((message, data))

        with pytest.MonkeyPatch.context() as mp:
            self._prepare(
                win, mp,
                results={
                    media_dir: ["/x/a.png", "/x/b.png"],
                    other_dir: ["/x/b.png", "/x/c.png"],
                },
                windows=[win, fake2],
                toasts=toasts,
            )
            mp.setattr(
                win.file_marks_ctrl, "go_to_mark", lambda: goto_calls.append(True)
            )
            win.app_actions.related_images_signals().result.connect(_capture)
            try:
                win.file_marks_ctrl.set_marks_from_downstream_related_images_all_windows()
            finally:
                win.app_actions.related_images_signals().result.disconnect(_capture)

        marks = MarkedFiles.file_marks[:]
        MarkedFiles.file_marks = []  # don't leak fake paths into teardown
        assert marks == ["/x/a.png", "/x/b.png", "/x/c.png"]
        assert goto_calls  # first-result owner is the real window
        assert any("2 window(s)" in t for t in toasts)
        # The structured report reaches the listener with the payload data.
        assert len(reports) == 1
        message, data = reports[0]
        assert "2 window(s)" in message
        assert data["found"] == 3
        assert data["skipped_dirs"] == []

    def test_oversized_window_skipped_not_fatal(
        self, window_with_dir, qtbot, tmp_path
    ):
        win, media_dir = window_with_dir
        other_dir = str(tmp_path / "other")
        os.makedirs(other_dir, exist_ok=True)
        fake2 = self._fake_secondary_window(other_dir, slow=True)
        toasts: list = []
        queried: list = []

        def _fake_get(media, base_dir, aa, force_refresh=False, quiet=False):
            queried.append(base_dir)
            return {media_dir: ["/x/a.png"]}.get(base_dir)

        with pytest.MonkeyPatch.context() as mp:
            self._prepare(
                win, mp, results={}, windows=[win, fake2], toasts=toasts
            )
            mp.setattr(
                "ui.app_window.file_marks_controller.get_downstream_related_images",
                _fake_get,
            )
            mp.setattr(win.file_marks_ctrl, "go_to_mark", lambda: None)

            win.file_marks_ctrl.set_marks_from_downstream_related_images_all_windows()

        marks = MarkedFiles.file_marks[:]
        MarkedFiles.file_marks = []  # don't leak fake paths into teardown
        assert other_dir not in queried
        assert marks == ["/x/a.png"]
        assert any("Skipped" in t for t in toasts)

    def test_no_results_leaves_marks_unchanged(self, window_with_dir, qtbot):
        win, media_dir = window_with_dir
        toasts: list = []

        with pytest.MonkeyPatch.context() as mp:
            self._prepare(win, mp, results={}, windows=[win], toasts=toasts)

            win.file_marks_ctrl.set_marks_from_downstream_related_images_all_windows()

        assert MarkedFiles.file_marks == []
        assert any("No downstream related images found" in t for t in toasts)

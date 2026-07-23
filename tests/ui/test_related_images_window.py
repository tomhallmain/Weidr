"""UI tests for RelatedImagesWindow and the all-windows downstream search.

Covers docs/downstream-search-all-windows.md: window construction with all
seven actions, result reporting into the persistent result area (including
stopping after close), the FileMarksController all-windows aggregation
(de-duplication, oversized-window skip, no-result path), and marking files
in the current directory that have no related image in either direction.

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
from utils.translations import _


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
        win, _media_dir = window_with_dir
        rw = _open_window(win, qtbot)
        texts = [b.text() for b in rw.findChildren(QPushButton)]
        rw.close()
        assert len(texts) == 7
        assert _("Search all open windows for downstream images") in texts
        assert _("Mark files without a related image in current directory") in texts

    def test_shows_source_window_directory(self, window_with_dir, qtbot):
        """The opener's directory is displayed — that window's current media
        is what every action sources from."""
        win, media_dir = window_with_dir
        rw = _open_window(win, qtbot)
        text = rw._source_label.text()
        rw.close()
        assert media_dir in text

    def test_result_area_shows_reported_results(self, window_with_dir, qtbot):
        win, _media_dir = window_with_dir
        rw = _open_window(win, qtbot)
        win.app_actions.notify_related_images_result(
            "downstream test result 123", action_label="Test Action"
        )
        text = rw._result_label.text()
        rw.close()
        assert "downstream test result 123" in text
        assert "Test Action" in text

    def test_result_area_stops_after_close(self, window_with_dir, qtbot):
        win, _media_dir = window_with_dir
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
        expected_window_frag = _("{0} file marks set from {1} window(s)").format(3, 2)
        assert any(expected_window_frag in t for t in toasts)
        # The structured report reaches the listener with the payload data.
        assert len(reports) == 1
        message, data = reports[0]
        assert expected_window_frag in message
        assert data["found"] == 3
        assert data["skipped_dirs"] == []
        # Per-directory breakdown: post-dedup contributions sum to the total
        # ("/x/b.png" counts for the first directory that yielded it).
        assert data["found_by_dir"] == {media_dir: 2, other_dir: 1}
        assert ": 2" in message and ": 1" in message
        # The source image that started the search is reported.
        assert data["source"] == win.media_path
        assert os.path.basename(win.media_path) in message

    def test_oversized_window_skipped_not_fatal(
        self, window_with_dir, qtbot, tmp_path
    ):
        win, media_dir = window_with_dir
        other_dir = str(tmp_path / "other")
        os.makedirs(other_dir, exist_ok=True)
        fake2 = self._fake_secondary_window(other_dir, slow=True)
        toasts: list = []
        queried: list = []
        reports: list = []

        def _fake_get(media, base_dir, aa, force_refresh=False, quiet=False):
            queried.append(base_dir)
            return {media_dir: ["/x/a.png"]}.get(base_dir)

        def _capture(message, action_label, data):
            reports.append(data)

        with pytest.MonkeyPatch.context() as mp:
            self._prepare(
                win, mp, results={}, windows=[win, fake2], toasts=toasts
            )
            mp.setattr(
                "ui.app_window.file_marks_controller.get_downstream_related_images",
                _fake_get,
            )
            mp.setattr(win.file_marks_ctrl, "go_to_mark", lambda: None)
            win.app_actions.related_images_signals().result.connect(_capture)
            try:
                win.file_marks_ctrl.set_marks_from_downstream_related_images_all_windows()
            finally:
                win.app_actions.related_images_signals().result.disconnect(_capture)

        marks = MarkedFiles.file_marks[:]
        MarkedFiles.file_marks = []  # don't leak fake paths into teardown
        assert other_dir not in queried
        assert marks == ["/x/a.png"]
        expected_skip_frag = _("Skipped {0} large unconfirmed director(ies).").format(1)
        assert any(expected_skip_frag in t for t in toasts)
        # Breakdown: a skipped directory is absent from found_by_dir entirely
        # (0 would mean "searched, nothing found"), and appears in skipped_dirs.
        assert len(reports) == 1
        assert reports[0]["found_by_dir"] == {media_dir: 1}
        assert reports[0]["skipped_dirs"] == [other_dir]
        assert reports[0]["source"] == win.media_path

    def test_no_results_leaves_marks_unchanged(self, window_with_dir, qtbot):
        win, media_dir = window_with_dir
        toasts: list = []

        with pytest.MonkeyPatch.context() as mp:
            self._prepare(win, mp, results={}, windows=[win], toasts=toasts)

            win.file_marks_ctrl.set_marks_from_downstream_related_images_all_windows()

        assert MarkedFiles.file_marks == []
        expected = _("No downstream related images found across {0} open window(s).").format(1)
        assert any(expected in t for t in toasts)


class TestMarkFilesWithoutRelatedImages:
    """mark_files_without_related_images_in_dir: union of
    get_sources_with_downstream_in_dir and get_downstream_files_for_sources
    (both directions of "related", searched against the current directory
    itself) gives every connected file; the complement is what gets marked.
    The two helpers are exercised against real files elsewhere
    (test_downstream_image_matching.py) -- these tests pin the controller's
    own logic: complement computation, empty-result messaging, marks
    mutation, and result reporting.
    """

    def test_marks_files_with_no_related_image_in_either_direction(
        self, window_with_dir, qtbot
    ):
        win, media_dir = window_with_dir
        files = win.file_browser.get_files()
        assert len(files) == 3
        source_related, downstream_related, isolated = files
        goto_calls: list = []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "ui.app_window.file_marks_controller.get_sources_with_downstream_in_dir",
                lambda paths, base_dir: [source_related],
            )
            mp.setattr(
                "ui.app_window.file_marks_controller.get_downstream_files_for_sources",
                lambda paths, base_dir: [downstream_related],
            )
            mp.setattr(
                win.file_marks_ctrl, "go_to_mark", lambda: goto_calls.append(True)
            )
            MarkedFiles.file_marks = []
            win.file_marks_ctrl.mark_files_without_related_images_in_dir()
            marks = MarkedFiles.file_marks[:]
            MarkedFiles.file_marks = []

        assert marks == [isolated]
        assert goto_calls

    def test_no_unrelated_files_leaves_marks_unchanged_and_toasts(
        self, window_with_dir, qtbot
    ):
        win, media_dir = window_with_dir
        files = win.file_browser.get_files()
        toasts: list = []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "ui.app_window.file_marks_controller.get_sources_with_downstream_in_dir",
                lambda paths, base_dir: files,
            )
            mp.setattr(
                "ui.app_window.file_marks_controller.get_downstream_files_for_sources",
                lambda paths, base_dir: [],
            )
            mp.setattr(win.notification_ctrl, "toast", lambda msg, **kw: toasts.append(msg))
            MarkedFiles.file_marks = []
            win.file_marks_ctrl.mark_files_without_related_images_in_dir()
            marks = MarkedFiles.file_marks[:]

        assert marks == []
        assert toasts == [_("Every file in the current directory has a related image.")]

    def test_reports_result_with_base_dir_and_found_count(
        self, window_with_dir, qtbot
    ):
        win, media_dir = window_with_dir
        files = win.file_browser.get_files()
        reports: list = []

        def _capture(message, action_label, data):
            reports.append((message, action_label, data))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "ui.app_window.file_marks_controller.get_sources_with_downstream_in_dir",
                lambda paths, base_dir: [],
            )
            mp.setattr(
                "ui.app_window.file_marks_controller.get_downstream_files_for_sources",
                lambda paths, base_dir: [],
            )
            mp.setattr(win.file_marks_ctrl, "go_to_mark", lambda: None)
            MarkedFiles.file_marks = []
            win.app_actions.related_images_signals().result.connect(_capture)
            try:
                win.file_marks_ctrl.mark_files_without_related_images_in_dir()
            finally:
                win.app_actions.related_images_signals().result.disconnect(_capture)
            MarkedFiles.file_marks = []

        assert len(reports) == 1
        message, action_label, data = reports[0]
        assert action_label == _("Mark files without related images")
        assert data["found"] == len(files)
        assert data["base_dir"] == media_dir
        assert str(len(files)) in message

    def test_no_files_in_directory_toasts_and_returns(self, window_with_dir, qtbot):
        win, _media_dir = window_with_dir
        toasts: list = []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(win.file_browser, "filepaths", [])
            mp.setattr(win.notification_ctrl, "toast", lambda msg, **kw: toasts.append(msg))
            win.file_marks_ctrl.mark_files_without_related_images_in_dir()

        assert toasts == [_("No files in current directory.")]

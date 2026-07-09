"""
UI tests for direct-display prevalidation (docs/skip-handling-direct-media-display.md):

- go_to_file on a skippable file is suppressed (config on) or displayed
  (config off); the real skip_media pipeline is what runs.
- open_temp_media_canvas gates external-context files behind an advisory
  confirm dialog (dry-run advise_media): cancel keeps the canvas closed,
  accept shows the file.
"""
import os

import pytest

from compare.compare_manager import CompareManager
from utils.config import config
from utils.constants import ClassifierActionType

from tests.ui.app_window_fixtures import make_png


def _set_direct_display_config(monkeypatch, value: bool) -> None:
    """Patch the flag on every Config reference the consumers may hold.

    The autouse isolated_singletons fixture rebinds utils.config.config to a
    fresh Config() per test, so a lazily-imported module (media_details is
    imported inside functions throughout the app) can capture a *different*
    Config instance than this module did at collection time. Patching only
    this module's `config` is then invisible to the gate under test — e.g.
    when running this file on its own rather than in the full suite.
    """
    import ui.app_window.media_navigator as mn_module
    import ui.image.media_details as md_module

    seen_ids = set()
    for cfg_obj in (config, mn_module.config, md_module.config):
        if id(cfg_obj) in seen_ids:
            continue
        seen_ids.add(id(cfg_obj))
        monkeypatch.setattr(cfg_obj, "prevalidate_on_direct_media_display", value)


class TestGoToFileSuppression:
    def test_skippable_file_is_suppressed(self, window_with_dir, monkeypatch):
        win, media_dir = window_with_dir
        _set_direct_display_config(monkeypatch, True)
        skip_calls = []
        monkeypatch.setattr(
            CompareManager, "skip_media",
            lambda self, p: skip_calls.append(p) or True,
        )

        result = win.media_navigator.go_to_file(search_text="img02.png")

        assert result is True
        assert any(p.endswith("img02.png") for p in skip_calls)
        assert "img02" not in os.path.basename(win.media_path or "")

    def test_config_off_displays_without_skip_check(self, window_with_dir, monkeypatch, qtbot):
        win, media_dir = window_with_dir
        _set_direct_display_config(monkeypatch, False)
        skip_calls = []
        monkeypatch.setattr(
            CompareManager, "skip_media",
            lambda self, p: skip_calls.append(p) or True,
        )

        result = win.media_navigator.go_to_file(search_text="img02.png")

        assert result is True
        assert skip_calls == []
        qtbot.waitUntil(
            lambda: "img02" in os.path.basename(win.media_path or ""), timeout=3000
        )

    def test_unskipped_file_is_displayed(self, window_with_dir, monkeypatch, qtbot):
        win, media_dir = window_with_dir
        _set_direct_display_config(monkeypatch, True)
        monkeypatch.setattr(CompareManager, "skip_media", lambda self, p: False)

        result = win.media_navigator.go_to_file(search_text="img02.png")

        assert result is True
        qtbot.waitUntil(
            lambda: "img02" in os.path.basename(win.media_path or ""), timeout=3000
        )


@pytest.fixture
def external_png(tmp_path_factory):
    """A PNG outside any window's base directory (external context)."""
    ext_dir = tmp_path_factory.mktemp("external_media")
    path = str(ext_dir / "external.png")
    make_png(path, (120, 90, 200))
    return path


class TestTempCanvasAdvisoryGate:
    def _patch_advise(self, monkeypatch, action, name="TestPreval"):
        from compare.classifier_actions_manager import ClassifierActionsManager
        monkeypatch.setattr(
            ClassifierActionsManager, "advise_media",
            staticmethod(lambda p, base_dir=None: (action, name)),
        )

    def _close_temp_canvas(self):
        from ui.image.media_details import MediaDetails
        canvas = MediaDetails.temp_media_canvas
        if canvas is not None:
            try:
                canvas.close()
            except Exception:
                pass
            MediaDetails.temp_media_canvas = None

    def test_cancel_keeps_canvas_closed(self, window_with_dir, external_png, monkeypatch):
        from ui.image.media_details import MediaDetails
        import ui.app_window.notification_controller as _nc

        win, _ = window_with_dir
        self._close_temp_canvas()  # isolate from any canvas leaked by other tests
        _set_direct_display_config(monkeypatch, True)
        self._patch_advise(monkeypatch, ClassifierActionType.MOVE)
        monkeypatch.setattr(_nc, "qt_alert", lambda *a, **kw: False)

        MediaDetails.open_temp_media_canvas(
            master=win, media_path=external_png,
            app_actions=win.app_actions, skip_get_window_check=True,
        )

        assert MediaDetails.temp_media_canvas is None

    def test_accept_shows_canvas(self, window_with_dir, external_png, monkeypatch):
        from ui.image.media_details import MediaDetails
        import ui.app_window.notification_controller as _nc

        win, _ = window_with_dir
        _set_direct_display_config(monkeypatch, True)
        self._patch_advise(monkeypatch, ClassifierActionType.MOVE)
        monkeypatch.setattr(_nc, "qt_alert", lambda *a, **kw: True)

        try:
            MediaDetails.open_temp_media_canvas(
                master=win, media_path=external_png,
                app_actions=win.app_actions, skip_get_window_check=True,
            )
            assert MediaDetails.temp_media_canvas is not None
        finally:
            self._close_temp_canvas()

    def test_no_advisory_match_opens_without_dialog(
        self, window_with_dir, external_png, monkeypatch
    ):
        from ui.image.media_details import MediaDetails
        import ui.app_window.notification_controller as _nc

        win, _ = window_with_dir
        _set_direct_display_config(monkeypatch, True)
        self._patch_advise(monkeypatch, None, name=None)
        alert_calls = []
        monkeypatch.setattr(
            _nc, "qt_alert", lambda *a, **kw: alert_calls.append(a) or True
        )

        try:
            MediaDetails.open_temp_media_canvas(
                master=win, media_path=external_png,
                app_actions=win.app_actions, skip_get_window_check=True,
            )
            assert MediaDetails.temp_media_canvas is not None
            assert alert_calls == []
        finally:
            self._close_temp_canvas()

    def test_config_off_bypasses_advisory(
        self, window_with_dir, external_png, monkeypatch
    ):
        from ui.image.media_details import MediaDetails
        from compare.classifier_actions_manager import ClassifierActionsManager

        win, _ = window_with_dir
        _set_direct_display_config(monkeypatch, False)
        advise_calls = []
        monkeypatch.setattr(
            ClassifierActionsManager, "advise_media",
            staticmethod(lambda p, base_dir=None: advise_calls.append(p) or (None, None)),
        )

        try:
            MediaDetails.open_temp_media_canvas(
                master=win, media_path=external_png,
                app_actions=win.app_actions, skip_get_window_check=True,
            )
            assert advise_calls == []
        finally:
            self._close_temp_canvas()

    def test_recently_actioned_file_bypasses_advisory(
        self, window_with_dir, external_png, monkeypatch
    ):
        """A file matching a recent move action must display without any
        advisory dialog or classifier run (Appendix A.2 / A.3)."""
        from ui.image.media_details import MediaDetails
        from compare.classifier_actions_manager import ClassifierActionsManager
        from files.file_action import FileAction
        from utils.utils import Utils
        import ui.app_window.notification_controller as _nc

        win, _ = window_with_dir
        _set_direct_display_config(monkeypatch, True)
        monkeypatch.setattr(
            FileAction, "action_history",
            [FileAction(Utils.move_file, os.path.dirname(external_png),
                        new_files=[external_png], auto=True)],
        )
        advise_calls, alert_calls = [], []
        monkeypatch.setattr(
            ClassifierActionsManager, "advise_media",
            staticmethod(lambda p, base_dir=None: advise_calls.append(p) or (ClassifierActionType.MOVE, "X")),
        )
        monkeypatch.setattr(
            _nc, "qt_alert", lambda *a, **kw: alert_calls.append(a) or True
        )

        try:
            MediaDetails.open_temp_media_canvas(
                master=win, media_path=external_png,
                app_actions=win.app_actions, skip_get_window_check=True,
            )
            assert MediaDetails.temp_media_canvas is not None
            assert advise_calls == []
            assert alert_calls == []
        finally:
            self._close_temp_canvas()


class TestOriginatingWindowContext:
    """Appendix A.1 — requests resolve the last-active AppWindow's context."""

    def test_window_manager_tracks_last_active(self, window):
        from ui.app_window.window_manager import WindowManager
        from types import SimpleNamespace

        WindowManager.notify_window_activated(window)
        assert WindowManager.get_active_window() is window

        # A tracked window that is no longer open falls back to primary.
        WindowManager._last_active = SimpleNamespace()
        assert WindowManager.get_active_window() is WindowManager.get_primary()

    def test_file_actions_window_view_uses_active_context(
        self, window_with_dir, qtbot, monkeypatch
    ):
        from unittest.mock import MagicMock
        from ui.app_window.window_manager import WindowManager
        from ui.files.file_actions_window_qt import FileActionsWindow

        win, media_dir = window_with_dir
        stale_actions = MagicMock()
        received = {}
        dialog = FileActionsWindow(
            win,
            stale_actions,
            lambda **kwargs: received.update(kwargs),
            lambda *a, **k: None,
        )
        qtbot.addWidget(dialog)
        try:
            WindowManager.notify_window_activated(win)
            existing_file = os.path.join(media_dir, "img01.png")

            dialog._view(existing_file)

            assert received["app_actions"] is win.app_actions
            assert received["master"] is win
        finally:
            dialog.close()
            FileActionsWindow._instance = None

    def test_go_to_file_dialog_resolves_active_context(self, window, qtbot):
        from unittest.mock import MagicMock
        from ui.app_window.window_manager import WindowManager
        from ui.files.go_to_file_qt import GoToFile

        stale_actions = MagicMock()
        dialog = GoToFile(window, stale_actions)
        qtbot.addWidget(dialog)
        try:
            WindowManager.notify_window_activated(window)
            assert dialog._app_actions is window.app_actions

            # With no live AppWindow tracked, the stored context is the fallback.
            WindowManager._last_active = None
            WindowManager._primary = None
            assert dialog._app_actions is stale_actions
        finally:
            dialog.close()
            GoToFile._instance = None
            WindowManager.set_primary(window)

    def test_temp_canvas_rebinds_context_on_reuse(
        self, window_with_dir, external_png, monkeypatch
    ):
        from unittest.mock import MagicMock
        from ui.image.media_details import MediaDetails

        win, _ = window_with_dir
        _set_direct_display_config(monkeypatch, False)
        stale_actions = MagicMock()
        stale_actions.get_window.return_value = None

        try:
            MediaDetails.open_temp_media_canvas(
                master=win, media_path=external_png,
                app_actions=stale_actions, skip_get_window_check=True,
            )
            assert MediaDetails.temp_media_canvas._app_actions is stale_actions

            MediaDetails.open_temp_media_canvas(
                master=win, media_path=external_png,
                app_actions=win.app_actions, skip_get_window_check=True,
            )
            assert MediaDetails.temp_media_canvas._app_actions is win.app_actions
        finally:
            canvas = MediaDetails.temp_media_canvas
            if canvas is not None:
                canvas.close()
                MediaDetails.temp_media_canvas = None

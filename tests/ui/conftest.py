import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if "WEIDR_CACHE_DIR" not in os.environ:
    import atexit, shutil, tempfile
    _tmp = tempfile.mkdtemp(prefix="weidr_ui_")
    os.environ["WEIDR_CACHE_DIR"] = os.path.join(_tmp, "cache")
    os.environ["WEIDR_CONFIGS_DIR"] = os.path.join(_tmp, "configs")
    os.makedirs(os.environ["WEIDR_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["WEIDR_CONFIGS_DIR"], exist_ok=True)
    _src = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config_example.json")
    shutil.copy(_src, os.path.join(os.environ["WEIDR_CONFIGS_DIR"], "config.json"))
    atexit.register(shutil.rmtree, _tmp, True)

# Importing a fixture into conftest.py makes it available to all tests in this
# directory — the pytest-idiomatic way to share fixtures from a fixtures/ module.
from tests.fixtures.compare_image_fixtures import compare_colors_dir  # noqa: F401
from tests.fixtures.media_fixtures import media_frame, sweep_qt_media_widgets  # noqa: F401
from tests.fixtures.show_media_assets import show_media_files  # noqa: F401
from tests.ui.app_window_fixtures import media_dir, window, window_with_dir  # noqa: F401


@pytest.fixture
def bypass_password(monkeypatch):
    """Run password-gated controller methods without prompting."""

    def _noop_decorator(*_actions, **_kwargs):
        def decorator(func):
            return func

        return decorator

    monkeypatch.setattr("ui.auth.password_utils.require_password", _noop_decorator)


@pytest.fixture
def immediate_compare_debounce(monkeypatch):
    """Invoke compare immediately instead of waiting on QtDebouncer."""

    def _apply(search_ctrl):
        monkeypatch.setattr(
            search_ctrl._debouncer, "schedule", search_ctrl._fire_pending_compare
        )

    return _apply


@pytest.fixture(autouse=True)
def ui_qt_media_cleanup_after_test():
    """Sweep MediaFrame/VLC/Qt widgets after each UI test so pytest exits cleanly."""
    yield
    sweep_qt_media_widgets()


def pytest_sessionfinish(session, exitstatus):
    """Final Qt sweep so VLC/MediaFrame tests do not leave the process hanging at 100%."""
    sweep_qt_media_widgets()


@pytest.fixture(autouse=True)
def ui_app_window_test_hygiene(monkeypatch):
    """
    Keep UI tests from leaking AppWindow instances and cache-store work.

    Primary windows were not removed from WindowManager on teardown, so each
    test's on_closing() stored the info cache once per stale window (slowing the
    suite over time). Session restore and periodic cache timers are disabled here.
    """
    from ui.app_window.app_window import AppWindow
    from ui.app_window.cache_controller import CacheController
    from ui.app_window.window_manager import WindowManager

    def _reset_window_manager() -> None:
        WindowManager._windows.clear()
        WindowManager._primary = None
        WindowManager._secondary_toplevels.clear()
        WindowManager._cycle_index = 0

    _reset_window_manager()
    monkeypatch.setattr(AppWindow, "_refocus_primary", lambda self: None)
    monkeypatch.setattr(AppWindow, "_restore_secondary_windows", lambda self: None)
    monkeypatch.setattr(CacheController, "start_periodic_store", lambda self: None)
    yield
    _reset_window_manager()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()

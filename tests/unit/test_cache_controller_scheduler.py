"""CacheController drives its periodic store through the injected scheduler.

Deliberately not in tests/ui/: that package's conftest replaces
CacheController.start_periodic_store with a no-op for every test, so the real
method is not reachable there.

These assert on behaviour rather than on source text. An earlier version of
this check read the method's source and passed only by accident -- once the
method was monkeypatched, inspect.getsource returned the patch line instead.
"""

from ui.app_window.cache_controller import CacheController
from ui.app_window.qt_scheduler import QtScheduler


class _FakeScheduler:
    def __init__(self):
        self.started = []
        self.stop_calls = 0

    def start(self, interval_seconds, callback):
        self.started.append((interval_seconds, callback))

    def stop(self):
        self.stop_calls += 1

    def is_running(self):
        return bool(self.started)


class _StubStoreCacheConfig:
    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds


class _StubAppWindow:
    def __init__(self, interval_seconds=42):
        self.store_cache_config = _StubStoreCacheConfig(interval_seconds)


def _controller(scheduler, interval_seconds=42):
    return CacheController(
        _StubAppWindow(interval_seconds), file_browser=None, scheduler=scheduler
    )


class TestDelegation:
    def test_start_passes_the_configured_interval(self):
        fake = _FakeScheduler()
        controller = _controller(fake, interval_seconds=42)
        controller.start_periodic_store()
        assert len(fake.started) == 1
        assert fake.started[0][0] == 42

    def test_start_passes_the_store_callback(self):
        fake = _FakeScheduler()
        controller = _controller(fake)
        controller.start_periodic_store()
        assert fake.started[0][1] == controller._on_periodic_store

    def test_stop_delegates(self):
        fake = _FakeScheduler()
        controller = _controller(fake)
        controller.stop_periodic_store()
        assert fake.stop_calls == 1

    def test_the_interval_is_read_at_start_not_at_construction(self):
        # The configured interval can change while the window is open.
        fake = _FakeScheduler()
        app = _StubAppWindow(10)
        controller = CacheController(app, file_browser=None, scheduler=fake)
        app.store_cache_config.interval_seconds = 99
        controller.start_periodic_store()
        assert fake.started[0][0] == 99


class TestDefault:
    def test_defaults_to_the_qt_scheduler(self):
        # A window needs main-thread delivery, so the Qt one has to be the
        # default; nothing else wires it in.
        controller = CacheController(_StubAppWindow(), file_browser=None)
        assert isinstance(controller._scheduler, QtScheduler)

    def test_default_scheduler_starts_idle(self):
        controller = CacheController(_StubAppWindow(), file_browser=None)
        assert controller._scheduler.is_running() is False

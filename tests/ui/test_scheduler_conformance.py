"""Both scheduler implementations must present the same interface.

CacheController drives the periodic store through whichever it is given, so
the Qt one and the headless one have to agree on the contract.

Lives in tests/ui/ because importing the Qt scheduler needs a QApplication.
"""

import inspect

import pytest

from ui.app_window.cache_controller import CacheController
from ui.app_window.qt_scheduler import QtScheduler
from utils.periodic_scheduler import ThreadedScheduler

SCHEDULERS = [QtScheduler, ThreadedScheduler]


@pytest.mark.parametrize("cls", SCHEDULERS, ids=lambda c: c.__name__)
class TestInterfaceParity:
    @pytest.mark.parametrize("method", ["start", "stop", "is_running"])
    def test_method_exists(self, cls, method):
        assert callable(getattr(cls, method, None))

    def test_start_takes_interval_and_callback(self, cls):
        params = list(inspect.signature(cls.start).parameters)
        assert params == ["self", "interval_seconds", "callback"]

    def test_idle_scheduler_is_not_running(self, cls):
        assert cls().is_running() is False

    def test_stop_before_start_is_safe(self, cls):
        cls().stop()

    @pytest.mark.parametrize("interval", [0, -1])
    def test_non_positive_interval_does_not_schedule(self, cls, interval):
        scheduler = cls()
        scheduler.start(interval, lambda: None)
        try:
            assert scheduler.is_running() is False
        finally:
            scheduler.stop()

    def test_start_then_stop_clears_running(self, cls):
        scheduler = cls()
        scheduler.start(30, lambda: None)
        assert scheduler.is_running() is True
        scheduler.stop()
        assert scheduler.is_running() is False


class TestControllerAcceptsInjection:
    """Delegation itself is covered behaviourally in
    tests/unit/test_cache_controller_scheduler.py -- this package's conftest
    replaces start_periodic_store, so the real method cannot be exercised here.
    """

    def test_constructor_accepts_an_injected_scheduler(self):
        params = inspect.signature(CacheController.__init__).parameters
        assert "scheduler" in params
        assert params["scheduler"].default is None

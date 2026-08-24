"""Tests for the Qt-free periodic scheduler.

Intervals here are deliberately tiny and the assertions wait on Events rather
than sleeping for a fixed span, so the suite does not pay for wall-clock time
it does not need.
"""

import threading
import time

from utils.periodic_scheduler import ThreadedScheduler


class TestFiring:
    def test_fires_repeatedly(self):
        fired = threading.Event()
        count = []

        def _tick():
            count.append(1)
            if len(count) >= 3:
                fired.set()

        scheduler = ThreadedScheduler()
        scheduler.start(0.01, _tick)
        try:
            assert fired.wait(5), "callback did not fire three times"
        finally:
            scheduler.stop()
        assert len(count) >= 3

    def test_does_not_fire_before_the_first_interval(self):
        # The first call comes after one interval, not immediately.
        calls = []
        scheduler = ThreadedScheduler()
        scheduler.start(5, lambda: calls.append(1))
        try:
            time.sleep(0.05)
            assert calls == []
        finally:
            scheduler.stop()


class TestScheduling:
    def test_non_positive_interval_does_not_schedule(self):
        for interval in (0, -1):
            scheduler = ThreadedScheduler()
            scheduler.start(interval, lambda: None)
            assert not scheduler.is_running(), interval

    def test_reports_running_between_start_and_stop(self):
        scheduler = ThreadedScheduler()
        assert not scheduler.is_running()
        scheduler.start(5, lambda: None)
        assert scheduler.is_running()
        scheduler.stop()
        assert not scheduler.is_running()

    def test_stop_is_safe_when_never_started(self):
        ThreadedScheduler().stop()

    def test_stop_is_idempotent(self):
        scheduler = ThreadedScheduler()
        scheduler.start(5, lambda: None)
        scheduler.stop()
        scheduler.stop()
        assert not scheduler.is_running()

    def test_restart_replaces_the_previous_schedule(self):
        # Starting twice must not leave the first schedule firing alongside
        # the second.
        first, second = [], []
        scheduler = ThreadedScheduler()
        scheduler.start(0.01, lambda: first.append(1))
        scheduler.start(0.01, lambda: second.append(1))
        try:
            deadline = time.monotonic() + 5
            while len(second) < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(second) >= 3
            settled = len(first)
            time.sleep(0.05)
            assert len(first) == settled, "the replaced schedule is still firing"
        finally:
            scheduler.stop()


class TestResilience:
    def test_a_failing_tick_does_not_end_the_schedule(self):
        survived = threading.Event()
        calls = []

        def _tick():
            calls.append(1)
            if len(calls) >= 3:
                survived.set()
            raise ValueError("tick failed")

        scheduler = ThreadedScheduler()
        scheduler.start(0.01, _tick)
        try:
            assert survived.wait(5), "schedule stopped after a failing tick"
        finally:
            scheduler.stop()

    def test_stop_takes_effect_promptly(self):
        # Cancellation must not wait out the remaining interval.
        scheduler = ThreadedScheduler()
        scheduler.start(30, lambda: None)
        started = time.monotonic()
        scheduler.stop()
        assert time.monotonic() - started < 1
        assert not scheduler.is_running()

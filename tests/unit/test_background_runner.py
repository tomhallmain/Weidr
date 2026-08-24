"""Tests for the Qt-free background execution strategy.

Synchronisation uses Events rather than sleeps so the assertions are
deterministic: a sleep long enough to be reliable on a loaded machine is long
enough to slow the suite noticeably.
"""

import threading

import pytest

from utils.background_runner import ThreadedTaskRunner


class _Cancelled(Exception):
    """Stands in for the compare engine's cancellation signal."""


def _run_and_wait(runner, func, args=(), timeout=5, **kwargs):
    done = threading.Event()
    supplied = kwargs.pop("on_finished", None)

    def _finished():
        if supplied is not None:
            supplied()
        done.set()

    runner.start(func, args, on_finished=_finished, **kwargs)
    assert done.wait(timeout), "task did not finish in time"


class TestExecution:
    def test_runs_the_function_with_its_arguments(self):
        seen = []
        _run_and_wait(ThreadedTaskRunner(), lambda a, b: seen.append((a, b)), ["x", 2])
        assert seen == [("x", 2)]

    def test_on_finished_runs_on_success(self):
        calls = []
        _run_and_wait(ThreadedTaskRunner(), lambda: None, on_finished=lambda: calls.append("done"))
        assert calls == ["done"]

    def test_no_callbacks_is_allowed(self):
        runner = ThreadedTaskRunner()
        seen = []
        runner.start(lambda: seen.append(1))
        assert runner.wait(5)
        assert seen == [1]


class TestFailure:
    def test_error_is_reported_as_its_string_form(self):
        errors = []

        def _boom():
            raise ValueError("it broke")

        _run_and_wait(ThreadedTaskRunner(), _boom, on_error=errors.append)
        assert errors == ["it broke"]

    def test_on_finished_still_runs_after_a_failure(self):
        calls = []

        def _boom():
            raise ValueError("nope")

        _run_and_wait(
            ThreadedTaskRunner(), _boom,
            on_error=lambda _e: calls.append("error"),
            on_finished=lambda: calls.append("finished"),
        )
        assert calls == ["error", "finished"]

    def test_a_failed_task_leaves_the_runner_reusable(self):
        runner = ThreadedTaskRunner()

        def _boom():
            raise ValueError("first")

        _run_and_wait(runner, _boom, on_error=lambda _e: None)
        assert not runner.is_running()
        seen = []
        _run_and_wait(runner, lambda: seen.append("second"))
        assert seen == ["second"]


class TestCancellation:
    def test_named_cancellation_is_not_an_error(self):
        errors = []

        def _cancel():
            raise _Cancelled()

        _run_and_wait(
            ThreadedTaskRunner(), _cancel,
            on_error=errors.append,
            cancelled_exceptions=(_Cancelled,),
        )
        assert errors == []

    def test_cancellation_still_finishes(self):
        calls = []

        def _cancel():
            raise _Cancelled()

        _run_and_wait(
            ThreadedTaskRunner(), _cancel,
            on_finished=lambda: calls.append("finished"),
            cancelled_exceptions=(_Cancelled,),
        )
        assert calls == ["finished"]

    def test_an_unnamed_exception_is_still_an_error(self):
        # Without cancelled_exceptions the same type must surface normally.
        errors = []

        def _cancel():
            raise _Cancelled("not declared")

        _run_and_wait(ThreadedTaskRunner(), _cancel, on_error=errors.append)
        assert errors == ["not declared"]


class TestRunningState:
    def test_reports_running_while_the_task_is_in_flight(self):
        runner = ThreadedTaskRunner()
        started, release = threading.Event(), threading.Event()

        def _block():
            started.set()
            release.wait(5)

        runner.start(_block)
        assert started.wait(5)
        assert runner.is_running()
        release.set()
        assert runner.wait(5)
        assert not runner.is_running()

    def test_not_running_inside_on_finished(self):
        # The state is cleared first so a caller may start new work from the
        # callback without tripping the already-running guard.
        runner = ThreadedTaskRunner()
        observed = []
        _run_and_wait(runner, lambda: None, on_finished=lambda: observed.append(runner.is_running()))
        assert observed == [False]

    def test_starting_twice_raises(self):
        runner = ThreadedTaskRunner()
        started, release = threading.Event(), threading.Event()

        def _block():
            started.set()
            release.wait(5)

        runner.start(_block)
        assert started.wait(5)
        with pytest.raises(RuntimeError):
            runner.start(lambda: None)
        release.set()
        assert runner.wait(5)

    def test_wait_returns_true_when_idle(self):
        assert ThreadedTaskRunner().wait(0) is True


class TestProgress:
    def test_forwards_reports_from_the_task(self):
        reports = []
        runner = ThreadedTaskRunner()
        _run_and_wait(
            runner,
            lambda: runner.report_progress("scanning", 40),
            on_progress=lambda ctx, pct: reports.append((ctx, pct)),
        )
        assert reports == [("scanning", 40)]

    def test_missing_percent_becomes_the_indeterminate_sentinel(self):
        # -1 is what the Qt progress signal carries for indeterminate, so both
        # strategies hand the caller the same value.
        reports = []
        runner = ThreadedTaskRunner()
        _run_and_wait(
            runner,
            lambda: runner.report_progress("loading"),
            on_progress=lambda ctx, pct: reports.append((ctx, pct)),
        )
        assert reports == [("loading", -1)]

    def test_reporting_after_the_task_ended_is_ignored(self):
        reports = []
        runner = ThreadedTaskRunner()
        _run_and_wait(runner, lambda: None, on_progress=lambda c, p: reports.append((c, p)))
        runner.report_progress("late", 10)
        assert reports == []

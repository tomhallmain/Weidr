"""Both execution strategies must present the same interface.

SearchController now drives the compare run through whichever strategy it is
given. That only holds up if the Qt one and the headless one agree on the
contract, so this checks them side by side rather than trusting that two
separately written classes stayed in step.

Lives in tests/ui/ because importing the Qt strategy needs a QApplication.
"""

import ast
import inspect
import sys

import pytest

from ui.app_window.search_controller import QtTaskRunner, SearchController
from utils.background_runner import ThreadedTaskRunner


def _method_source(cls, method_name):
    """Source of a method as written in its module file.

    Read from the file rather than via inspect.getsource(cls.method) because a
    conftest may have monkeypatched the attribute, in which case getsource
    returns the patch line and the assertion silently checks the wrong text.
    """
    module = sys.modules[cls.__module__]
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.unparse(item)
    raise AssertionError(f"{cls.__name__}.{method_name} not found in its module")

RUNNERS = [QtTaskRunner, ThreadedTaskRunner]
INTERFACE = ["start", "is_running", "wait", "report_progress"]


@pytest.mark.parametrize("runner_cls", RUNNERS, ids=lambda c: c.__name__)
class TestInterfaceParity:
    @pytest.mark.parametrize("method", INTERFACE)
    def test_method_exists(self, runner_cls, method):
        assert callable(getattr(runner_cls, method, None))

    def test_start_accepts_the_same_keywords(self, runner_cls):
        params = inspect.signature(runner_cls.start).parameters
        for name in ("on_finished", "on_error", "on_progress", "cancelled_exceptions"):
            assert name in params, name
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY

    def test_idle_runner_reports_not_running(self, runner_cls):
        assert runner_cls().is_running() is False

    def test_report_progress_on_an_idle_runner_is_ignored(self, runner_cls):
        # Neither strategy may raise when nothing is in flight; the controller's
        # display_progress falls back to a label update in that case.
        runner_cls().report_progress("nothing", 10)

    def test_wait_on_an_idle_runner_reports_done(self, runner_cls):
        assert runner_cls().wait(0) is True


class TestControllerUsesTheStrategy:
    """The controller must go through the injected runner, not a QThread."""

    def test_constructor_accepts_an_injected_runner(self):
        params = inspect.signature(SearchController.__init__).parameters
        assert "task_runner" in params
        assert params["task_runner"].default is None

    def test_is_compare_running_delegates(self):
        source = _method_source(SearchController, "is_compare_running")
        assert "_runner.is_running()" in source

    def test_run_with_progress_names_cancellation_explicitly(self):
        # The historical behaviour is that a cancelled compare is not an error.
        # Passing the type explicitly is what keeps that true once the worker
        # no longer hardcodes it.
        source = _method_source(SearchController, "_run_with_progress")
        assert "cancelled_exceptions=" in source
        assert "CompareCancelled" in source

    def test_run_with_progress_goes_through_the_runner(self):
        source = _method_source(SearchController, "_run_with_progress")
        assert "_runner.start(" in source

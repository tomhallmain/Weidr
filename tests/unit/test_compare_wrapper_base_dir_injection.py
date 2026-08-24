"""CompareWrapper resolves the working directory through an injectable seam.

Prevalidation needs a base directory. It used to be read straight off
app_actions, which a caller without a window cannot supply. The wrapper now
exposes get_base_dir() instead, backed by an injected callable when one is
given and by app_actions otherwise, so existing callers are unaffected.
"""

import pytest

from compare.compare_wrapper import CompareWrapper


class _AppActionsStub:
    def __init__(self, base_dir="/from/app_actions"):
        self._base_dir = base_dir
        self.get_base_dir_calls = 0

    def get_base_dir(self):
        self.get_base_dir_calls += 1
        return self._base_dir


def _wrapper(app_actions=None, get_base_dir=None):
    return CompareWrapper(
        master=None,
        compare_mode=None,
        app_actions=app_actions,
        get_base_dir=get_base_dir,
    )


class TestInjectedCallable:
    def test_injected_callable_supplies_the_directory(self):
        wrapper = _wrapper(get_base_dir=lambda: "/injected")
        assert wrapper.get_base_dir() == "/injected"

    def test_injected_callable_wins_over_app_actions(self):
        stub = _AppActionsStub()
        wrapper = _wrapper(app_actions=stub, get_base_dir=lambda: "/injected")
        assert wrapper.get_base_dir() == "/injected"
        assert stub.get_base_dir_calls == 0

    def test_injected_callable_is_re_read_each_call(self):
        # The working directory changes while the app runs, so the seam must
        # not capture a value at construction time.
        current = {"dir": "/first"}
        wrapper = _wrapper(get_base_dir=lambda: current["dir"])
        assert wrapper.get_base_dir() == "/first"
        current["dir"] = "/second"
        assert wrapper.get_base_dir() == "/second"


class TestFallback:
    def test_falls_back_to_app_actions(self):
        stub = _AppActionsStub("/from/app_actions")
        wrapper = _wrapper(app_actions=stub)
        assert wrapper.get_base_dir() == "/from/app_actions"
        assert stub.get_base_dir_calls == 1

    def test_default_keeps_existing_callers_unchanged(self):
        # Constructing without the new keyword must behave exactly as before.
        stub = _AppActionsStub("/legacy")
        wrapper = CompareWrapper(master=None, compare_mode=None, app_actions=stub)
        assert wrapper.get_base_dir() == "/legacy"


class TestLazyResolution:
    def test_referencing_is_safe_without_app_actions(self):
        # Several existing tests build a wrapper with app_actions=None, and
        # the prevalidation call site passes this method as a callable rather
        # than calling it, so merely reaching it must not raise.
        wrapper = _wrapper()
        assert callable(wrapper.get_base_dir)

    def test_calling_without_any_source_raises(self):
        wrapper = _wrapper()
        with pytest.raises(AttributeError):
            wrapper.get_base_dir()

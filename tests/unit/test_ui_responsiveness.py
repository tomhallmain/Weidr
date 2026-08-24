"""The UI-responsiveness port, and the Qt-freedom it was introduced to achieve."""

import ast
import os

import pytest

from utils.ui_responsiveness import NullResponsiveness

PACKAGES = ["compare", "files", "image"]


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _import_statements(path):
    """Names imported by real import statements, ignoring prose mentions.

    A plain text search is not enough: files/file_browser.py describes a
    "pre-PySide6-port feature" in a docstring and would trip it.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _module_level_import_statements(path):
    """Like _import_statements, but only imports executed on module load.

    An import inside a function is a deliberate deferral -- it costs nothing
    until called -- so a layering check has to distinguish the two.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class TestNullResponsiveness:
    def test_yield_is_a_no_op(self):
        assert NullResponsiveness().yield_to_ui() is None

    def test_runs_the_callable_and_returns_its_result(self):
        assert NullResponsiveness().run_off_thread(lambda: "value") == "value"

    def test_runs_inline(self):
        # No thread is warranted without a drawing thread to keep free, and
        # running inline is what makes the result immediately available.
        import threading
        seen = []
        NullResponsiveness().run_off_thread(lambda: seen.append(threading.current_thread().ident))
        assert seen == [threading.current_thread().ident]

    def test_failure_becomes_none_rather_than_raising(self):
        # Matches the Qt path, where an exception inside the worker thread
        # never reached the caller and left the result unset.
        def _boom():
            raise ValueError("failed")

        assert NullResponsiveness().run_off_thread(_boom) is None


class TestQtFreedom:
    """The packages the port was extracted for must import no Qt."""

    @pytest.mark.parametrize("package", PACKAGES)
    def test_package_imports_no_pyside6(self, package):
        offenders = []
        root = os.path.join(_project_root(), package)
        for dirpath, _dirnames, filenames in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                if any(n.split(".")[0] == "PySide6" for n in _import_statements(path)):
                    offenders.append(os.path.relpath(path, _project_root()))
        assert offenders == []

    def test_the_check_would_catch_a_regression(self):
        # Guards the guard: a test that can never fail is worse than none.
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("from PySide6.QtWidgets import QApplication\n")
            probe = fh.name
        try:
            assert any(n.split(".")[0] == "PySide6" for n in _import_statements(probe))
        finally:
            os.unlink(probe)

    def test_prose_mentions_are_not_flagged(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write('"""A pre-PySide6-port feature."""\nimport os\n')
            probe = fh.name
        try:
            assert not any(n.split(".")[0] == "PySide6" for n in _import_statements(probe))
        finally:
            os.unlink(probe)


class TestLayeringDirection:
    """utils/ must not reach up into ui/ at import time.

    The dependency runs ui -> utils: AppActions exists so a UI component can
    call back into AppWindow without importing it. An import the other way
    inverts that, and drags the UI layer into every headless import of the
    facade. warn()/success() used to do it for two toast colours; they name a
    kind now and the display layer picks the colour.
    """

    def test_app_actions_has_no_module_level_ui_import(self):
        path = os.path.join(_project_root(), "utils", "app_actions.py")
        offenders = [
            n for n in _module_level_import_statements(path)
            if n.split(".")[0] == "ui"
        ]
        assert offenders == []

    def test_the_lazy_ui_import_is_still_permitted(self):
        # related_images_signals() imports a Qt bridge on demand. That is
        # deliberate -- headless callers seed the slot so it is never reached --
        # so the check must look at module level only, not every import node.
        path = os.path.join(_project_root(), "utils", "app_actions.py")
        assert any(
            n.split(".")[0] == "ui" for n in _import_statements(path)
        ), "expected the lazy ui import to still be present"

    def test_the_check_would_catch_a_regression(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("from ui.app_style import AppStyle\n")
            probe = fh.name
        try:
            assert any(
                n.split(".")[0] == "ui"
                for n in _module_level_import_statements(probe)
            )
        finally:
            os.unlink(probe)

    def test_a_function_local_import_is_not_flagged(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("def f():\n    from ui.app_style import AppStyle\n    return AppStyle\n")
            probe = fh.name
        try:
            assert _module_level_import_statements(probe) == []
        finally:
            os.unlink(probe)

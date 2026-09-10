"""Proves app_headless.py's full import chain -- not just
utils/headless_app_actions.py -- never imports PySide6.

Lives as its own file rather than folded into test_headless_app_actions.py:
that file's subprocess test only imports utils.headless_app_actions and
exercises AppActions; this one imports app_headless itself and constructs a
real HeadlessMCPSession, which additionally pulls in
compare.compare_manager, files.file_browser, files.marked_files,
files.skip_aware_navigation, and extensions.mcp_server -- a materially
bigger import graph that deserves its own guard rather than being assumed
clean because the smaller one is.

A no-Qt-imported assertion can only be shown from a subprocess: once any
other test in the same pytest session has imported PySide6 (most have),
sys.modules already has it, and no import inside this process would ever
raise or prove anything either way.
"""

import os
import subprocess
import sys
import textwrap

_NO_QT_PROBE = textwrap.dedent(
    """
    import sys

    class _BlockPySide6:
        def find_spec(self, name, path=None, target=None):
            if name == "PySide6" or name.startswith("PySide6."):
                raise AssertionError("PySide6 was imported: " + name)
            return None

    sys.meta_path.insert(0, _BlockPySide6())

    import app_headless

    session = app_headless.HeadlessMCPSession(sys.argv[1])
    session.get_current_file()
    session.list_marks()
    session.get_base_dir()
    session.is_compare_running()
    session.get_compare_mode()
    print("NO_QT_OK")
    """
)


def test_headless_mcp_session_usable_without_importing_pyside6(tmp_path):
    """Runs in a fresh interpreter that raises if anything imports PySide6."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_dir = os.path.join(project_root, "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        project_root + os.pathsep + src_dir + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.pop("QT_QPA_PLATFORM", None)

    base_dir = tmp_path / "media"
    base_dir.mkdir()

    result = subprocess.run(
        [sys.executable, "-c", _NO_QT_PROBE, str(base_dir)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"probe failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "NO_QT_OK" in result.stdout

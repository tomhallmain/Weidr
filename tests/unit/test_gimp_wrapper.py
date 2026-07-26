"""
Unit tests for GimpWrapper's GIMP process-launch environment handling.
All subprocess calls are mocked -- no real GIMP required.
"""

from unittest.mock import MagicMock, patch

import pytest

import extensions.gimp.gimp_wrapper as gimp_wrapper_module
from extensions.gimp.gimp_wrapper import GimpWrapper
from utils.config import config


@pytest.fixture(autouse=True)
def _reset_gimp_process_globals():
    """GimpWrapper's process-tracking state is module-global (process-wide,
    per the same singleton-state gotcha as MarkedFiles), not per-instance --
    reset it around each test so tests can't see each other's state."""
    def _reset():
        gimp_wrapper_module._gimp_process = None
        gimp_wrapper_module._current_filepath = None
        gimp_wrapper_module._is_gimp_running = False
        gimp_wrapper_module._current_wrapper = None

    _reset()
    yield
    _reset()


@pytest.fixture()
def app_actions():
    actions = MagicMock()
    return actions


@pytest.fixture()
def wrapper(app_actions):
    return GimpWrapper(files_threshold_reached=lambda: False, app_actions=app_actions)


def _fake_terminated_process():
    proc = MagicMock()
    proc.poll.return_value = 0
    return proc


class TestBuildGimpLaunchEnv:
    def test_defaults_to_resolved_locale(self):
        config.gimp_locale = None
        config.locale = "de"
        env = GimpWrapper._build_gimp_launch_env()
        assert env["LANG"] == "de"

    def test_explicit_gimp_locale_overrides_app_locale(self):
        config.gimp_locale = "en"
        config.locale = "de"
        env = GimpWrapper._build_gimp_launch_env()
        assert env["LANG"] == "en"

    def test_inherits_rest_of_os_environ(self, monkeypatch):
        config.gimp_locale = None
        config.locale = "en"
        monkeypatch.setenv("SOME_UNRELATED_VAR", "keep-me")
        env = GimpWrapper._build_gimp_launch_env()
        assert env.get("SOME_UNRELATED_VAR") == "keep-me"


class TestUnloadCurrentFileFromGimp:
    @patch("subprocess.call")
    def test_uses_plain_argv_and_explicit_env_not_shell(self, mock_call, wrapper):
        mock_call.return_value = 0

        wrapper.unload_current_file_from_gimp("gimp-3.0")

        args, kwargs = mock_call.call_args
        assert args[0] == ["gimp-3.0"]
        assert kwargs.get("shell") is not True
        assert kwargs.get("env")


class TestOpenDirectly:
    @patch("extensions.gimp.gimp_wrapper.start_thread", side_effect=lambda fn, *a, **kw: fn())
    @patch("subprocess.Popen")
    def test_uses_plain_argv_and_explicit_env_not_shell(
        self, mock_popen, _start_thread, wrapper, tmp_path
    ):
        mock_popen.return_value = _fake_terminated_process()
        filepath = str(tmp_path / "image.png")

        wrapper._open_directly(filepath, "gimp-3.0")

        args, kwargs = mock_popen.call_args
        assert args[0] == ["gimp-3.0", filepath]
        assert kwargs.get("shell") is not True
        assert kwargs.get("env")


class TestOpenWithTempDirectory:
    @patch("extensions.gimp.gimp_wrapper.start_thread", side_effect=lambda fn, *a, **kw: fn())
    @patch.object(GimpWrapper, "_monitor_gimp_process")
    @patch("subprocess.Popen")
    def test_uses_plain_argv_and_explicit_env_not_shell(
        self, mock_popen, _monitor, _start_thread, wrapper, tmp_path
    ):
        mock_popen.return_value = _fake_terminated_process()
        source = tmp_path / "image.png"
        source.write_bytes(b"fake-image-bytes")

        try:
            wrapper._open_with_temp_directory(str(source), "gimp-3.0")

            args, kwargs = mock_popen.call_args
            assert args[0][0] == "gimp-3.0"
            assert args[0][1].endswith("image.png")
            assert kwargs.get("shell") is not True
            assert kwargs.get("env")
        finally:
            # _monitor_gimp_process is mocked out, so it never reaches the
            # normal completion cleanup -- clean up the real temp dir here.
            wrapper._cleanup_temp_directory()

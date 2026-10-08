"""
Unit tests for GimpWrapper: GIMP process-launch environment handling, and the
hand-off when another file is opened while a GIMP session is finishing (one
claim on each session's completion, process tracking left to the newest
wrapper, no unload call for an exited process).
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


class TestCompletionClaim:
    def test_only_the_first_claim_succeeds(self, wrapper):
        assert wrapper._claim_completion() is True
        assert wrapper._claim_completion() is False

    def test_previous_wrapper_completion_skips_when_its_monitor_claimed_it(self, wrapper, app_actions):
        previous = GimpWrapper(files_threshold_reached=lambda: True, app_actions=app_actions)
        previous._claim_completion()
        with patch.object(previous, "_handle_gimp_completion") as handle, \
                patch.object(previous, "_wait_for_file_release") as wait:
            wrapper._handle_previous_wrapper_completion(previous)
        handle.assert_not_called()
        wait.assert_not_called()

    def test_previous_wrapper_completion_runs_when_unclaimed(self, wrapper, app_actions, tmp_path):
        previous = GimpWrapper(files_threshold_reached=lambda: True, app_actions=app_actions)
        previous._temp_dir = str(tmp_path)
        with patch.object(previous, "_handle_gimp_completion") as handle:
            wrapper._handle_previous_wrapper_completion(previous)
        handle.assert_called_once()
        assert previous._claim_completion() is False

    def test_monitor_skips_completion_already_claimed(self, wrapper):
        gimp_wrapper_module._current_wrapper = wrapper
        wrapper._claim_completion()
        with patch.object(wrapper, "_handle_gimp_completion") as handle:
            wrapper._monitor_gimp_process(_fake_terminated_process())
        handle.assert_not_called()

    def test_monitor_handles_unclaimed_completion(self, wrapper):
        gimp_wrapper_module._current_wrapper = wrapper
        with patch.object(wrapper, "_handle_gimp_completion") as handle:
            wrapper._monitor_gimp_process(_fake_terminated_process())
        handle.assert_called_once()


class TestWaitForFileRelease:
    def test_missing_file_counts_as_released_without_waiting(self, wrapper, tmp_path):
        with patch("extensions.gimp.gimp_wrapper.time.sleep") as sleep:
            assert wrapper._wait_for_file_release(str(tmp_path / "moved_away.png")) is True
        sleep.assert_not_called()

    def test_no_path_counts_as_released(self, wrapper):
        assert wrapper._wait_for_file_release(None) is True


class TestProcessTrackingOwnership:
    @patch("extensions.gimp.gimp_wrapper.start_thread", side_effect=lambda fn, *a, **kw: fn())
    @patch("subprocess.Popen")
    def test_finished_wrapper_leaves_a_newer_wrappers_tracking(
        self, mock_popen, _start_thread, wrapper, app_actions
    ):
        newer = GimpWrapper(files_threshold_reached=lambda: False, app_actions=app_actions)

        def launch(*args, **kwargs):
            # Another file is opened while this wrapper's GIMP runs.
            gimp_wrapper_module._current_wrapper = newer
            gimp_wrapper_module._current_filepath = "/newer.png"
            return _fake_terminated_process()

        mock_popen.side_effect = launch
        wrapper._open_directly("/older.png", "gimp-3.0")
        assert gimp_wrapper_module._current_wrapper is newer
        assert gimp_wrapper_module._current_filepath == "/newer.png"

    @patch("extensions.gimp.gimp_wrapper.start_thread", side_effect=lambda fn, *a, **kw: fn())
    @patch("subprocess.Popen")
    def test_finished_current_wrapper_clears_tracking(self, mock_popen, _start_thread, wrapper):
        mock_popen.return_value = _fake_terminated_process()
        wrapper._open_directly("/image.png", "gimp-3.0")
        assert gimp_wrapper_module._current_wrapper is None
        assert gimp_wrapper_module._is_gimp_running is False


class TestOpenWhilePreviousProcessExited:
    def _set_previous(self, poll_result):
        proc = MagicMock()
        proc.poll.return_value = poll_result
        gimp_wrapper_module._gimp_process = proc
        gimp_wrapper_module._is_gimp_running = True
        gimp_wrapper_module._current_filepath = "/previous.png"

    def test_exited_previous_process_skips_unload(self, wrapper):
        self._set_previous(poll_result=0)
        with patch.object(wrapper, "unload_current_file_from_gimp") as unload, \
                patch.object(wrapper, "_open_directly"):
            wrapper.open_image_in_gimp("/next.png", "gimp-3.0")
        unload.assert_not_called()

    def test_running_previous_process_is_unloaded(self, wrapper):
        self._set_previous(poll_result=None)
        with patch.object(wrapper, "unload_current_file_from_gimp") as unload, \
                patch.object(wrapper, "_open_directly"):
            wrapper.open_image_in_gimp("/next.png", "gimp-3.0")
        unload.assert_called_once_with("gimp-3.0")

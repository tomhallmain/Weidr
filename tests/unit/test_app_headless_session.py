"""Tests for HeadlessMCPSession's app_actions wiring -- specifically the
hide_current_media/run_image_generation domain actions that
prevalidation/pipeline HIDE and GENERATE rules call through
ActionCallbacks. Both were previously unsupplied, so a rule firing either
action during headless navigation raised HeadlessActionUnavailable instead
of running (see docs/mcp-server-pipelines-and-prevalidation-spec.md, Part A).

Constructs a real HeadlessMCPSession against a small temp directory of
generated PNGs -- no real media, no real network calls (SDRunnerClient is
mocked).
"""

from __future__ import annotations

from unittest.mock import patch

from PIL import Image

import app_headless


def _png(path, color=(120, 120, 120)):
    Image.new("RGB", (8, 8), color).save(str(path), format="PNG")


def _session(tmp_path):
    _png(tmp_path / "a.png", (200, 30, 30))
    _png(tmp_path / "b.png", (30, 200, 30))
    return app_headless.HeadlessMCPSession(str(tmp_path))


def test_hide_current_media_domain_action_is_supplied(tmp_path):
    """Calling it the way classifier_action.py actually does --
    hide_callback(image_path), one positional arg -- must not raise
    HeadlessActionUnavailable."""
    session = _session(tmp_path)
    target = str(tmp_path / "a.png")
    session._actions.hide_current_media(target)
    assert target in session._compare_manager.hidden_media


def test_run_image_generation_domain_action_is_supplied(tmp_path):
    """Calling it the way classifier_action.py actually does --
    generate_callback(image_path, edit_suffix, target_dir=...) -- must not
    raise HeadlessActionUnavailable, and must generate from the given path,
    not whatever the session's current file happens to be."""
    session = _session(tmp_path)
    current = session.get_current_file()
    other = str(tmp_path / "b.png")
    assert other != current

    with patch("extensions.sd_runner_client.SDRunnerClient") as mock_client_cls:
        session._actions.run_image_generation(other, "suffix", target_dir="/out")

    mock_client_cls.return_value.run.assert_called_once()
    call_args = mock_client_cls.return_value.run.call_args
    assert call_args.args[1] == other
    assert call_args.kwargs["edit_suffix"] == "suffix"
    assert call_args.kwargs["target_dir"] == "/out"

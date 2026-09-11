"""Tests for HeadlessMCPSession against a real FileBrowser/CompareManager
over a small temp directory of generated PNGs.

Covers the hide_current_media/run_image_generation domain actions that
prevalidation/pipeline HIDE and GENERATE rules call through ActionCallbacks
(unsupplied, a rule firing either would raise HeadlessActionUnavailable
mid-navigation), navigation, marks, and compare-run orchestration. No model
is loaded and no network is used: CompareManager.run and SDRunnerClient are
patched where a test reaches them.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

import app_headless
from compare.compare_args import CompareArgs
from files.marked_files import MarkedFiles
from utils.constants import CompareMode, Mode


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


def test_get_index_reports_position_and_count(tmp_path):
    session = _session(tmp_path)
    assert session.get_index() == (1, 2)


def test_previous_file_from_a_fresh_session_wraps_to_the_last_file(tmp_path):
    session = _session(tmp_path)
    session._compare_manager.set_prevalidations_running(False)
    assert os.path.basename(session.get_current_file()) == "a.png"
    assert os.path.basename(session.previous_file()) == "b.png"
    assert session.get_index() == (2, 2)


def test_clear_marks_returns_how_many_were_cleared(tmp_path):
    session = _session(tmp_path)
    MarkedFiles.file_marks = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    assert session.clear_marks() == 2
    assert session.list_marks() == []


def test_clear_marks_raises_during_a_transfer(tmp_path):
    session = _session(tmp_path)
    MarkedFiles.file_marks = [str(tmp_path / "a.png")]
    MarkedFiles.is_performing_action = True
    try:
        with pytest.raises(ValueError):
            session.clear_marks()
        assert MarkedFiles.file_marks == [str(tmp_path / "a.png")]
    finally:
        MarkedFiles.is_performing_action = False


def _session_abc(tmp_path):
    for name, color in (("a.png", (200, 30, 30)), ("b.png", (30, 200, 30)), ("c.png", (30, 30, 200))):
        _png(tmp_path / name, color)
    return app_headless.HeadlessMCPSession(str(tmp_path))


def _fake_group_run(session, groups, scanned):
    """Stands in for CompareManager.run: leaves the primary wrapper as a
    GROUP run over *scanned* would, without loading a model."""
    def run(args):
        wrapper = session._compare_manager._primary_wrapper()
        wrapper._compare = SimpleNamespace(compare_data=SimpleNamespace(files_found=list(scanned)))
        wrapper.file_groups = groups
        session._actions.set_mode(Mode.GROUP, do_update=False)
    return run


def test_group_complement_lists_ungrouped_files_in_listing_order(tmp_path):
    session = _session_abc(tmp_path)
    a, b, c = session._file_browser.get_files()
    cm = session._compare_manager
    fake = _fake_group_run(session, groups={0: {b: 0.0}}, scanned=[c, b, a])
    with patch.object(cm, "run", fake):
        session._run_compare_task(cm.compare_mode, CompareArgs(), complement=True)

    results = session.compare_results()
    assert results["run_mode"] == "GROUP_COMPLEMENT"
    assert results["files_matched"] == [a, c]  # listing order, not scan order
    assert results["file_groups"] == {0: {b: 0.0}}


def test_group_complement_with_nothing_ungrouped_stays_in_group_mode(tmp_path):
    session = _session_abc(tmp_path)
    a, b, c = session._file_browser.get_files()
    cm = session._compare_manager
    fake = _fake_group_run(session, groups={0: {a: 0.0, b: 0.1, c: 0.2}}, scanned=[a, b, c])
    with patch.object(cm, "run", fake):
        session._run_compare_task(cm.compare_mode, CompareArgs(), complement=True)

    assert session.compare_results()["run_mode"] == "GROUP"


def test_plain_group_run_does_not_enter_complement_mode(tmp_path):
    session = _session_abc(tmp_path)
    a, b, c = session._file_browser.get_files()
    cm = session._compare_manager
    fake = _fake_group_run(session, groups={0: {b: 0.0}}, scanned=[a, b, c])
    with patch.object(cm, "run", fake):
        session._run_compare_task(cm.compare_mode, CompareArgs(), complement=False)

    assert session.compare_results()["run_mode"] == "GROUP"


def test_run_compare_task_switches_to_the_requested_compare_mode(tmp_path):
    session = _session(tmp_path)
    cm = session._compare_manager
    target = CompareMode.SIZE if cm.compare_mode != CompareMode.SIZE else CompareMode.COLOR_MATCHING
    with patch.object(cm, "run") as run:
        session._run_compare_task(target, CompareArgs(), complement=False)
    run.assert_called_once()
    assert cm.compare_mode == target


def test_run_compare_rejects_group_complement_with_find_duplicates(tmp_path):
    session = _session(tmp_path)
    with patch.object(session._runner, "start") as start:
        with pytest.raises(ValueError):
            session.run_compare("CLIP_EMBEDDING", True, run_mode="GROUP_COMPLEMENT")
    start.assert_not_called()


def test_run_compare_rejects_a_run_mode_other_than_group_or_complement(tmp_path):
    session = _session(tmp_path)
    with patch.object(session._runner, "start") as start:
        with pytest.raises(ValueError):
            session.run_compare("CLIP_EMBEDDING", False, run_mode="SEARCH")
    start.assert_not_called()


def test_run_compare_while_one_is_running_is_a_value_error(tmp_path):
    session = _session(tmp_path)
    with patch.object(session._runner, "start", side_effect=RuntimeError("busy")):
        with pytest.raises(ValueError):
            session.run_compare("CLIP_EMBEDDING", False)

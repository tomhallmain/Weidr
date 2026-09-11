"""Tests for HeadlessMCPSession against a real FileBrowser/CompareManager
over a small temp directory of generated PNGs.

Covers the hide_media/run_image_generation domain actions that
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
from compare.classifier_actions_manager import ClassifierActionsManager
from compare.compare_args import CompareArgs
from files.marked_files import MarkedFiles
from utils.constants import ClassifierActionType, CompareMode, Mode


def _png(path, color=(120, 120, 120)):
    Image.new("RGB", (8, 8), color).save(str(path), format="PNG")


def _session(tmp_path):
    _png(tmp_path / "a.png", (200, 30, 30))
    _png(tmp_path / "b.png", (30, 200, 30))
    return app_headless.HeadlessMCPSession(str(tmp_path))


def test_hide_callback_hides_the_matched_file_without_navigating(tmp_path):
    """hide_callback(image_path), as classifier_action.py calls it, through
    the real ActionCallbacks bundle."""
    session = _session(tmp_path)
    session.go_to_index(1)
    target = str(tmp_path / "b.png")
    session._actions.prevalidation_callbacks.hide_callback(target)
    assert target in session._compare_manager.hidden_media
    assert session.get_index() == (1, 2)


def test_generate_callback_generates_from_the_matched_file(tmp_path):
    """generate_callback(image_path, edit_suffix, target_dir=...), as
    classifier_action.py calls it, through the real ActionCallbacks bundle
    (whose wrapper adds suppress_toast=True). Must generate from the given
    path, not the session's current file."""
    session = _session(tmp_path)
    current = session.get_current_file()
    other = str(tmp_path / "b.png")
    assert other != current

    with patch("extensions.sd_runner_client.SDRunnerClient") as mock_client_cls:
        session._actions.prevalidation_callbacks.generate_callback(other, "suffix", target_dir="/out")

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


def _session_abcd(tmp_path):
    for i, name in enumerate(("a.png", "b.png", "c.png", "d.png")):
        _png(tmp_path / name, (40 * i, 100, 100))
    return app_headless.HeadlessMCPSession(str(tmp_path))


def test_hide_rule_during_next_file_stops_on_the_file_after_it(tmp_path):
    """A HIDE match on b while stepping from a must land on c: the skip loop
    steps past b once, and the HIDE callback itself must not step again."""
    session = _session_abcd(tmp_path)
    a, b, c, d = session._file_browser.get_files()
    session.go_to_index(1)

    def prevalidate(media_path, get_base_dir, callbacks, force=False):
        if media_path == b:
            callbacks.hide_callback(media_path)
            return ClassifierActionType.HIDE
        return None

    with patch.object(ClassifierActionsManager, "prevalidate_media", prevalidate):
        assert session.next_file() == c
    assert b in session._compare_manager.hidden_media


def test_compare_results_leave_out_files_that_no_longer_exist(tmp_path):
    session = _session_abcd(tmp_path)
    a, b, c, d = session._file_browser.get_files()
    cm = session._compare_manager
    groups = {0: {a: 0.0, b: 0.1}, 1: {c: 0.0, d: 0.2}, 2: {a: 0.0}}
    with patch.object(cm, "run", _fake_group_run(session, groups=groups, scanned=[a, b, c, d])):
        session._run_compare_task(cm.compare_mode, CompareArgs(), complement=False)
    cm._primary_wrapper().files_matched = [a, b]

    os.remove(b)
    results = session.compare_results()
    # Group 0 fell to one member through the removal and is dropped; group 2
    # was a one-member group as the engine returned it and is kept.
    assert results["file_groups"] == {1: {c: 0.0, d: 0.2}, 2: {a: 0.0}}
    assert results["files_matched"] == [a]


def test_load_persisted_classifier_state_restores_saved_rules():
    from compare.classifier_action import ClassifierAction, Prevalidation

    ClassifierActionsManager.prevalidations = [Prevalidation(name="HeadlessPv", positives=["cat"])]
    ClassifierActionsManager.classifier_actions = [ClassifierAction(name="HeadlessCa", positives=["dog"])]
    ClassifierActionsManager.store_prevalidations()
    ClassifierActionsManager.store_classifier_actions()
    ClassifierActionsManager.prevalidations = []
    ClassifierActionsManager.classifier_actions = []
    try:
        app_headless.load_persisted_classifier_state()
        assert [p.name for p in ClassifierActionsManager.prevalidations] == ["HeadlessPv"]
        assert [a.name for a in ClassifierActionsManager.classifier_actions] == ["HeadlessCa"]
    finally:
        ClassifierActionsManager.classifier_actions = []


def test_prevalidations_running_follows_the_saved_directory_setting(tmp_path):
    from utils.app_info_cache import app_info_cache

    on_dir, off_dir = tmp_path / "on", tmp_path / "off"
    for directory in (on_dir, off_dir):
        directory.mkdir()
        _png(directory / "a.png")
    app_info_cache.set(str(on_dir), "prevalidations_running", True)
    app_info_cache.set(str(off_dir), "prevalidations_running", False)

    session = app_headless.HeadlessMCPSession(str(off_dir))
    assert session.get_prevalidations_running() is False
    session.set_base_dir(str(on_dir))
    assert session.get_prevalidations_running() is True


def test_go_to_file_does_not_land_on_a_vetoed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(app_headless.config, "prevalidate_on_direct_media_display", True)
    session = _session(tmp_path)
    session._compare_manager.set_prevalidations_running(False)  # the hidden list alone vetoes
    a, b = session._file_browser.get_files()
    session.go_to_index(1)
    session._compare_manager.hidden_media.append(b)

    assert session.go_to_file(b) == a
    assert session.go_to_index(2) == a
    assert session.get_current_file() == a


def test_go_to_file_lands_on_a_vetoed_file_when_direct_display_checks_are_off(tmp_path, monkeypatch):
    monkeypatch.setattr(app_headless.config, "prevalidate_on_direct_media_display", False)
    session = _session(tmp_path)
    a, b = session._file_browser.get_files()
    session._compare_manager.hidden_media.append(b)

    assert session.go_to_file(b) == b
    assert session.get_current_file() == b


def test_set_prevalidations_running_is_remembered_for_the_directory(tmp_path):
    from utils.app_info_cache import app_info_cache

    session = _session(tmp_path)
    session.set_prevalidations_running(False)
    assert session.get_prevalidations_running() is False
    assert app_info_cache.get(str(tmp_path), "prevalidations_running") is False
    session.set_base_dir(str(tmp_path))  # restores the directory's saved setting
    assert session.get_prevalidations_running() is False


def test_password_blocked_names_the_blocking_action(tmp_path, monkeypatch):
    from ui.auth import password_core
    from utils.constants import ProtectedActions

    session = _session(tmp_path)
    monkeypatch.setattr(password_core, "first_password_protected",
                        lambda actions: ProtectedActions.DELETE_MEDIA)
    assert session.password_blocked(["delete_media"]) == "delete_media"
    monkeypatch.setattr(password_core, "first_password_protected", lambda actions: None)
    assert session.password_blocked(["delete_media"]) is None


def test_run_pipeline_uses_the_non_navigating_hide_and_the_persisted_generation_type(tmp_path, monkeypatch):
    from compare import pipeline_profile_run
    from utils.app_info_cache import app_info_cache
    from utils.constants import ImageGenerationType

    app_info_cache.set_meta("image_generation_mode", ImageGenerationType.IP_ADAPTER.value)
    session = _session(tmp_path)
    captured = {}
    monkeypatch.setattr(pipeline_profile_run.pipeline_runs, "start",
                        lambda name, profile, **kwargs: captured.update(kwargs, name=name, profile=profile))

    session.run_pipeline("Sorter", "Photos", continue_without_sd_runner=True)

    assert captured["name"] == "Sorter" and captured["profile"] == "Photos"
    assert captured["continue_without_sd_runner"] is True
    assert captured["fallback_generation_type"] == ImageGenerationType.IP_ADAPTER
    assert captured["hide_callback"] == session._hide_media


def test_persisted_generation_type_falls_back_to_control_net():
    from utils.app_info_cache import app_info_cache
    from utils.constants import ImageGenerationType

    app_info_cache.set_meta("image_generation_mode", ImageGenerationType.RENOISER.value)
    assert app_headless._persisted_image_generation_type() == ImageGenerationType.CONTROL_NET
    app_info_cache.set_meta("image_generation_mode", "not a mode")
    assert app_headless._persisted_image_generation_type() == ImageGenerationType.CONTROL_NET

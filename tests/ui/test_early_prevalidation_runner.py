"""UI tests for EarlyPrevalidationRunner: when a slideshow look-ahead run
starts, what it is given, and how it stops."""

import os
import threading

import pytest

import ui.app_window.early_prevalidation_runner as runner_module
from compare.classifier_action import Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from compare.early_prevalidation import EarlyRunResult
from utils.config import config
from utils.constants import ClassifierActionType


@pytest.fixture
def slideshow_window(window_with_dir, monkeypatch):
    win, media_dir = window_with_dir
    monkeypatch.setattr(
        ClassifierActionsManager, "prevalidations",
        [Prevalidation(name="pv", action=ClassifierActionType.SKIP, use_embedding=False)],
    )
    monkeypatch.setattr(config, "slideshow_early_prevalidation", True)
    monkeypatch.setattr(config, "slideshow_early_prevalidation_start_delay_seconds", 0.0)
    win.compare_manager.set_prevalidations_running(True)
    return win, media_dir, win.media_navigator._early_prevalidation


@pytest.fixture
def recorded_runs(monkeypatch):
    runs = []
    done = threading.Event()

    def fake_run(candidates, **_kwargs):
        runs.append(list(candidates))
        done.set()
        return EarlyRunResult()

    monkeypatch.setattr(runner_module, "run_early_prevalidation", fake_run)
    return runs, done


def _names(paths):
    return [os.path.basename(p) for p in paths]


class TestEarlyPrevalidationRunner:
    def test_no_run_without_slideshow(self, slideshow_window, recorded_runs):
        win, _, runner = slideshow_window
        runner.on_media_shown(win.media_path)
        assert runner._start_timer.isActive() is False

    def test_slideshow_start_runs_on_following_files(self, slideshow_window, recorded_runs, qtbot):
        win, _, runner = slideshow_window
        runs, done = recorded_runs
        current = os.path.basename(win.media_path)
        win.media_navigator.toggle_slideshow()
        qtbot.waitUntil(done.is_set, timeout=2000)
        files = _names(win.file_browser.get_files())
        start = files.index(current)
        expected = [files[(start + step) % len(files)] for step in range(1, len(files))]
        assert _names(runs[0]) == expected
        win.media_navigator.stop_slideshow_timers()
        win.slideshow_config.end_slideshows()

    def test_delayed_start_is_cancelled_by_slideshow_end(self, slideshow_window, recorded_runs, monkeypatch):
        win, _, runner = slideshow_window
        monkeypatch.setattr(config, "slideshow_early_prevalidation_start_delay_seconds", 60.0)
        win.media_navigator.toggle_slideshow()
        assert runner._start_timer.isActive() is True
        win.media_navigator.stop_slideshow_timers()
        win.slideshow_config.end_slideshows()
        assert runner._start_timer.isActive() is False
        assert recorded_runs[0] == []

    def test_cancel_signals_running_worker(self, slideshow_window):
        _, _, runner = slideshow_window
        event = threading.Event()
        runner._cancel_event = event
        runner.cancel()
        assert event.is_set()

    def test_callbacks_dropped_after_shutdown(self, slideshow_window):
        _, _, runner = slideshow_window
        calls = []
        runner.shutdown()
        runner._run_posted(lambda: calls.append(1))
        assert calls == []

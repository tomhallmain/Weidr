"""Tests for slideshow early prevalidation: prevalidate_media(early=True), cost
classification and the candidate walk in compare.early_prevalidation."""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction, Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_pipeline import ClassifierPipelines, PrevalidationPipeline
from compare.early_prevalidation import (
    StopReason,
    is_costly_candidate,
    run_early_prevalidation,
    upcoming_candidates,
)
from utils.constants import ClassifierActionType


def _noop(*_args, **_kwargs):
    pass


@contextmanager
def _isolated(prevalidations, pipelines=()):
    saved_prevalidations = ClassifierActionsManager.prevalidations[:]
    saved_cache = dict(ClassifierActionsManager.prevalidated_cache)
    saved_initialized = ClassifierActionsManager._prevalidations_initialized
    saved_exclude = list(ClassifierActionsManager.directories_to_exclude)
    saved_overrides = set(ClassifierActionsManager.user_prevalidation_overrides)
    saved_pipelines = ClassifierPipelines.pipelines[:]
    saved_pv = ClassifierPipelines._prevalidation_pipelines[:]
    saved_ac = ClassifierPipelines._action_pipelines[:]
    try:
        ClassifierActionsManager.prevalidations = list(prevalidations)
        ClassifierActionsManager.prevalidated_cache.clear()
        ClassifierActionsManager._prevalidations_initialized = True
        ClassifierActionsManager.directories_to_exclude.clear()
        ClassifierActionsManager.user_prevalidation_overrides.clear()
        ClassifierPipelines.pipelines = list(pipelines)
        ClassifierPipelines._rebuild_type_cache()
        yield
    finally:
        ClassifierActionsManager.prevalidations = saved_prevalidations
        ClassifierActionsManager.prevalidated_cache.clear()
        ClassifierActionsManager.prevalidated_cache.update(saved_cache)
        ClassifierActionsManager._prevalidations_initialized = saved_initialized
        ClassifierActionsManager.directories_to_exclude.clear()
        ClassifierActionsManager.directories_to_exclude.extend(saved_exclude)
        ClassifierActionsManager.user_prevalidation_overrides.clear()
        ClassifierActionsManager.user_prevalidation_overrides.update(saved_overrides)
        ClassifierPipelines.pipelines = saved_pipelines
        ClassifierPipelines._prevalidation_pipelines = saved_pv
        ClassifierPipelines._action_pipelines = saved_ac


def _prevalidation(name, action, **flags):
    values = dict(
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_blacklist=False,
        use_prototype=False,
    )
    values.update(flags)
    return Prevalidation(name=name, action=action, **values)


def _early(media_path, base_dir, notify=_noop):
    return ClassifierActionsManager.prevalidate_media(
        media_path,
        lambda: base_dir,
        ActionCallbacks(hide_callback=_noop, notify_callback=notify, add_mark_callback=_noop),
        early=True,
    )


def _media(root, name="a.mp4"):
    path = Path(root) / name
    path.write_bytes(b"x")
    return str(path)


class TestPrevalidateMediaEarly:
    def test_deferred_actions_are_not_executed_or_cached(self):
        for action in (ClassifierActionType.COPY, ClassifierActionType.NOTIFY):
            with tempfile.TemporaryDirectory() as root:
                media = _media(root)
                pv = _prevalidation("deferred", action)
                with _isolated([pv]):
                    with patch.object(ClassifierAction, "_evaluate_image_path_match",
                                      return_value=(True, None)), \
                         patch.object(ClassifierAction, "run_action") as run_action:
                        result = _early(media, root)
                    assert result == action
                    run_action.assert_not_called()
                    assert media not in ClassifierActionsManager.prevalidated_cache

    def test_other_actions_execute_early(self):
        with tempfile.TemporaryDirectory() as root:
            media = _media(root)
            pv = _prevalidation("mover", ClassifierActionType.MOVE, action_modifier=str(Path(root) / "out"))
            with _isolated([pv]):
                with patch.object(ClassifierAction, "_evaluate_image_path_match",
                                  return_value=(True, None)), \
                     patch.object(ClassifierAction, "run_action",
                                  return_value=ClassifierActionType.MOVE) as run_action:
                    result = _early(media, root)
            assert result == ClassifierActionType.MOVE
            run_action.assert_called_once()

    def test_pipelines_are_left_for_arrival(self):
        with tempfile.TemporaryDirectory() as root:
            media = _media(root)
            pipeline = PrevalidationPipeline(name="pipe", is_active=True)
            with _isolated([], pipelines=[pipeline]):
                with patch("compare.classifier_pipeline_runner.run_pipeline") as run_pipeline:
                    result = _early(media, root)
                assert result is None
                run_pipeline.assert_not_called()
                assert media not in ClassifierActionsManager.prevalidated_cache

    def test_no_match_without_pipelines_is_cached(self):
        with tempfile.TemporaryDirectory() as root:
            media = _media(root)
            pv = _prevalidation("never", ClassifierActionType.SKIP)
            with _isolated([pv]):
                with patch.object(ClassifierAction, "_evaluate_image_path_match",
                                  return_value=(False, None)):
                    result = _early(media, root)
                assert result is None
                assert ClassifierActionsManager.prevalidated_cache[media] is None


class TestCost:
    def test_static_image_is_never_a_candidate(self):
        cheap = _prevalidation("cheap", ClassifierActionType.SKIP, use_filename_contains=True)
        costly = _prevalidation("costly", ClassifierActionType.SKIP, use_image_classifier=True)
        with tempfile.TemporaryDirectory() as root:
            image = _media(root, "a.jpg")
            with _isolated([cheap]):
                assert not is_costly_candidate(image, root)
            with _isolated([cheap, costly]):
                assert not is_costly_candidate(image, root)

    def test_dynamic_media_with_applicable_rule_is_a_candidate(self):
        import compare.classifier_actions_manager as cam_module

        cheap = _prevalidation("cheap", ClassifierActionType.SKIP, use_filename_contains=True)
        with tempfile.TemporaryDirectory() as root:
            video = _media(root, "a.mp4")
            with patch.object(cam_module.config, "enable_videos", True), _isolated([cheap]):
                assert is_costly_candidate(video, root)

    def test_nothing_applicable_is_cheap(self):
        with tempfile.TemporaryDirectory() as root:
            with _isolated([]):
                assert not is_costly_candidate(_media(root, "a.mp4"), root)


class TestUpcomingCandidates:
    def test_wraps_without_returning_current(self):
        assert upcoming_candidates(["a", "b", "c"], 1, 10) == ["c", "a"]

    def test_limit(self):
        assert upcoming_candidates(["a", "b", "c", "d"], 0, 2) == ["b", "c"]

    def test_degenerate(self):
        assert upcoming_candidates(["a"], 0, 5) == []
        assert upcoming_candidates(["a", "b"], 5, 5) == []


class TestRunEarlyPrevalidation:
    def _run(self, candidates, actions, costly=lambda p: True, hidden=(), abort=lambda: False):
        calls = []

        def prevalidate(path):
            calls.append(path)
            return actions.get(path)

        result = run_early_prevalidation(
            candidates,
            prevalidate=prevalidate,
            is_costly=costly,
            is_hidden=lambda p: p in hidden,
            should_abort=abort,
        )
        return result, calls

    def test_walks_past_skipped_until_shown(self):
        with tempfile.TemporaryDirectory() as root, _isolated([]):
            a, b, c = (_media(root, n) for n in ("a.mp4", "b.mp4", "c.mp4"))
            result, calls = self._run([a, b, c], {a: ClassifierActionType.MOVE,
                                                  b: ClassifierActionType.COPY})
        assert calls == [a, b, c]
        assert result.stop_reason == StopReason.SHOWN and result.stopped_at == c

    def test_notify_ends_walk(self):
        with tempfile.TemporaryDirectory() as root, _isolated([]):
            a, b = _media(root, "a.mp4"), _media(root, "b.mp4")
            result, calls = self._run([a, b], {a: ClassifierActionType.NOTIFY})
        assert calls == [a]
        assert result.stop_reason == StopReason.SHOWN

    def test_cheap_candidate_ends_walk_unevaluated(self):
        with tempfile.TemporaryDirectory() as root, _isolated([]):
            a, b = _media(root, "a.mp4"), _media(root, "b.jpg")
            result, calls = self._run([a, b], {a: ClassifierActionType.SKIP},
                                      costly=lambda p: p.endswith(".mp4"))
        assert calls == [a]
        assert result.stop_reason == StopReason.CHEAP and result.stopped_at == b

    def test_missing_and_hidden_files_are_passed_over(self):
        with tempfile.TemporaryDirectory() as root, _isolated([]):
            gone = str(Path(root) / "gone.mp4")
            hidden = _media(root, "h.mp4")
            shown = _media(root, "s.mp4")
            result, calls = self._run([gone, hidden, shown], {}, hidden={hidden})
        assert calls == [shown]
        assert result.stopped_at == shown

    def test_session_cache_hit_is_not_reevaluated(self):
        with tempfile.TemporaryDirectory() as root, _isolated([]):
            a, b = _media(root, "a.mp4"), _media(root, "b.mp4")
            ClassifierActionsManager.prevalidated_cache[a] = ClassifierActionType.SKIP
            result, calls = self._run([a, b], {})
        assert calls == [b]
        assert result.stopped_at == b

    def test_abort_before_next_candidate(self):
        with tempfile.TemporaryDirectory() as root, _isolated([]):
            a, b = _media(root, "a.mp4"), _media(root, "b.mp4")
            state = {"n": 0}

            def abort():
                state["n"] += 1
                return state["n"] > 1

            result, calls = self._run([a, b], {a: ClassifierActionType.SKIP}, abort=abort)
        assert calls == [a]
        assert result.stop_reason == StopReason.ABORTED

    def test_error_ends_walk(self):
        with tempfile.TemporaryDirectory() as root, _isolated([]):
            a = _media(root, "a.mp4")

            def failing(_path):
                raise RuntimeError("boom")

            result = run_early_prevalidation(
                [a], prevalidate=failing, is_costly=lambda p: True,
                is_hidden=lambda p: False, should_abort=lambda: False)
        assert result.stop_reason == StopReason.ERROR


class TestSkipMediaWaitsForLock:
    """skip_media while another thread (an early prevalidation) holds the
    evaluation lock: it waits off the calling thread, and a file moved away
    meanwhile is skipped without being evaluated. With the lock free, a file
    already moved away is skipped the same way."""

    def _wrapper(self):
        from types import SimpleNamespace

        from compare.compare_wrapper import CompareWrapper
        from utils.constants import CompareMode

        app_actions = SimpleNamespace(
            prevalidation_callbacks_with_mark=ActionCallbacks(),
            make_prevalidation_callbacks=lambda add_mark: ActionCallbacks(add_mark_callback=add_mark),
            start_loading_spinner=lambda **_k: None,
            stop_loading_spinner=lambda: None,
            request_media_blur=lambda _p: None,
        )
        wrapper = CompareWrapper(
            master=SimpleNamespace(update=lambda: None),
            compare_mode=CompareMode.CLIP_EMBEDDING,
            app_actions=app_actions,
            get_base_dir=lambda: "",
        )
        wrapper.prevalidations_running = True
        return wrapper

    def _hold_lock_then(self, release_action):
        import threading
        import time

        acquired = threading.Event()

        def holder():
            with ClassifierActionsManager.evaluation_lock:
                acquired.set()
                time.sleep(0.2)
                release_action()

        thread = threading.Thread(target=holder)
        thread.start()
        acquired.wait()
        return thread

    def test_waits_then_evaluates(self):
        with tempfile.TemporaryDirectory() as root:
            media = _media(root, "a.jpg")
            wrapper = self._wrapper()
            with patch.object(ClassifierActionsManager, "prevalidate_media",
                              return_value=None) as prevalidate:
                thread = self._hold_lock_then(lambda: None)
                skipped = wrapper.skip_media(media)
                thread.join()
            assert skipped is False
            prevalidate.assert_called_once()

    def test_file_moved_while_waiting_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            media = _media(root, "a.jpg")
            wrapper = self._wrapper()
            with patch.object(ClassifierActionsManager, "prevalidate_media") as prevalidate:
                thread = self._hold_lock_then(lambda: Path(media).unlink())
                skipped = wrapper.skip_media(media)
                thread.join()
            assert skipped is True
            prevalidate.assert_not_called()

    def test_file_gone_with_free_lock_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            media = _media(root, "a.jpg")
            Path(media).unlink()
            wrapper = self._wrapper()
            with patch.object(ClassifierActionsManager, "prevalidate_media") as prevalidate:
                skipped = wrapper.skip_media(media)
            assert skipped is True
            prevalidate.assert_not_called()

    def test_existing_file_with_free_lock_is_evaluated(self):
        with tempfile.TemporaryDirectory() as root:
            media = _media(root, "a.jpg")
            wrapper = self._wrapper()
            with patch.object(ClassifierActionsManager, "prevalidate_media",
                              return_value=None) as prevalidate:
                skipped = wrapper.skip_media(media)
            assert skipped is False
            prevalidate.assert_called_once()

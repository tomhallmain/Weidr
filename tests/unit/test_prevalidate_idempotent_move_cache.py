"""Unit tests for idempotent MOVE prevalidation caching."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from compare.classifier_action import ClassifierAction, Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from utils.constants import ClassifierActionType


def _noop(*_args, **_kwargs):
    pass


def _run_prevalidate(media_path: str, base_dir: str):
    from compare.action_callbacks import ActionCallbacks
    return ClassifierActionsManager.prevalidate_media(
        media_path,
        lambda: base_dir,
        ActionCallbacks(hide_callback=_noop, notify_callback=_noop, add_mark_callback=_noop),
    )


def test_idempotent_move_result_is_cached():
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / "out"
        target.mkdir()
        media = target / "a.jpg"
        media.write_bytes(b"x")

        pv = Prevalidation(
            name="move_to_out",
            action=ClassifierActionType.MOVE,
            action_modifier=str(target),
            use_embedding=False,
            use_image_classifier=False,
            use_prompts=False,
            use_blacklist=False,
            use_prototype=False,
        )
        saved_prevalidations = ClassifierActionsManager.prevalidations[:]
        saved_cache = dict(ClassifierActionsManager.prevalidated_cache)
        saved_initialized = ClassifierActionsManager._prevalidations_initialized
        try:
            ClassifierActionsManager.prevalidations = [pv]
            ClassifierActionsManager.prevalidated_cache.clear()
            ClassifierActionsManager._prevalidations_initialized = True

            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with patch.object(
                ClassifierAction, "_evaluate_image_path_match", counting_eval
            ):
                first = _run_prevalidate(str(media), root)
                second = _run_prevalidate(str(media), root)

            assert first == ClassifierActionType.MOVE
            assert second == ClassifierActionType.MOVE
            assert eval_calls == 1
            assert ClassifierActionsManager.prevalidated_cache[str(media)] == (
                ClassifierActionType.MOVE
            )
        finally:
            ClassifierActionsManager.prevalidations = saved_prevalidations
            ClassifierActionsManager.prevalidated_cache.clear()
            ClassifierActionsManager.prevalidated_cache.update(saved_cache)
            ClassifierActionsManager._prevalidations_initialized = saved_initialized


def test_non_idempotent_move_is_not_cached():
    with tempfile.TemporaryDirectory() as root:
        target = Path(root) / "out"
        target.mkdir()
        media = Path(root) / "a.jpg"
        media.write_bytes(b"x")

        pv = Prevalidation(
            name="move_to_out",
            action=ClassifierActionType.MOVE,
            action_modifier=str(target),
            use_embedding=False,
            use_image_classifier=False,
            use_prompts=False,
            use_blacklist=False,
            use_prototype=False,
        )
        saved_prevalidations = ClassifierActionsManager.prevalidations[:]
        saved_cache = dict(ClassifierActionsManager.prevalidated_cache)
        saved_initialized = ClassifierActionsManager._prevalidations_initialized
        try:
            ClassifierActionsManager.prevalidations = [pv]
            ClassifierActionsManager.prevalidated_cache.clear()
            ClassifierActionsManager._prevalidations_initialized = True

            with patch.object(
                ClassifierAction,
                "_evaluate_image_path_match",
                return_value=(True, None),
            ):
                _run_prevalidate(str(media), root)

            assert str(media) not in ClassifierActionsManager.prevalidated_cache
        finally:
            ClassifierActionsManager.prevalidations = saved_prevalidations
            ClassifierActionsManager.prevalidated_cache.clear()
            ClassifierActionsManager.prevalidated_cache.update(saved_cache)
            ClassifierActionsManager._prevalidations_initialized = saved_initialized

"""Unit tests for prevalidate_media: HIDE, BLUR, profile gating, and skip paths."""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from compare.classifier_action import ClassifierAction, Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from files.directory_profile import DirectoryProfile
from utils.constants import ClassifierActionType


def _noop(*_args, **_kwargs):
    pass


def _run_prevalidate(
    media_path: str,
    base_dir: str,
    *,
    hide_callback=_noop,
    notify_callback=_noop,
    add_mark_callback=_noop,
    blur_callback=None,
    force: bool = False,
):
    return ClassifierActionsManager.prevalidate_media(
        media_path,
        lambda: base_dir,
        hide_callback,
        notify_callback,
        add_mark_callback,
        blur_callback=blur_callback,
        force=force,
    )


@contextmanager
def _isolated_prevalidations(prevalidations):
    """Swap prevalidation list and restore manager state after the test."""
    saved_prevalidations = ClassifierActionsManager.prevalidations[:]
    saved_cache = dict(ClassifierActionsManager.prevalidated_cache)
    saved_initialized = ClassifierActionsManager._prevalidations_initialized
    saved_exclude = list(ClassifierActionsManager.directories_to_exclude)
    saved_overrides = set(ClassifierActionsManager.user_prevalidation_overrides)
    try:
        ClassifierActionsManager.prevalidations = list(prevalidations)
        ClassifierActionsManager.prevalidated_cache.clear()
        ClassifierActionsManager._prevalidations_initialized = True
        ClassifierActionsManager.directories_to_exclude.clear()
        ClassifierActionsManager.user_prevalidation_overrides.clear()
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


def _always_match_prevalidation(name: str, action: ClassifierActionType) -> Prevalidation:
    return Prevalidation(
        name=name,
        action=action,
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_blacklist=False,
        use_prototype=False,
    )


class TestPrevalidateHideAndBlur:
    def test_hide_invokes_callback_and_caches_result(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("hide_match", ClassifierActionType.HIDE)
            hidden = []

            with _isolated_prevalidations([pv]):
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    first = _run_prevalidate(
                        str(media),
                        root,
                        hide_callback=lambda p: hidden.append(p),
                    )
                    second = _run_prevalidate(
                        str(media),
                        root,
                        hide_callback=lambda p: hidden.append(p),
                    )

                assert first == ClassifierActionType.HIDE
                assert second == ClassifierActionType.HIDE
                assert hidden == [str(media)]
                assert ClassifierActionsManager.prevalidated_cache[str(media)] == (
                    ClassifierActionType.HIDE
                )

    def test_blur_invokes_callback_and_caches_result(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("blur_match", ClassifierActionType.BLUR)
            blurred = []

            with _isolated_prevalidations([pv]):
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    result = _run_prevalidate(
                        str(media),
                        root,
                        blur_callback=lambda p: blurred.append(p),
                    )
                    _run_prevalidate(
                        str(media),
                        root,
                        blur_callback=lambda p: blurred.append(p),
                    )

                assert result == ClassifierActionType.BLUR
                assert blurred == [str(media)]
                assert ClassifierActionsManager.prevalidated_cache[str(media)] == (
                    ClassifierActionType.BLUR
                )

    def test_blur_without_callback_still_returns_blur_action(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("blur_no_cb", ClassifierActionType.BLUR)

            with _isolated_prevalidations([pv]):
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    result = _run_prevalidate(str(media), root, blur_callback=None)

            assert result == ClassifierActionType.BLUR


class TestPrevalidateProfileGating:
    def test_skips_prevalidation_when_base_dir_not_in_profile(self):
        with tempfile.TemporaryDirectory() as root:
            other = Path(root) / "other"
            other.mkdir()
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")

            profile = DirectoryProfile(name="gate_prof", directories=[str(root)])
            pv = _always_match_prevalidation("profiled_hide", ClassifierActionType.HIDE)
            pv.profile_name = "gate_prof"
            pv.profile = profile
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_prevalidations([pv]):
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    counting_eval,
                ):
                    result = _run_prevalidate(str(media), str(other))

            assert result is None
            assert eval_calls == 0

    def test_runs_prevalidation_when_base_dir_in_profile(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")

            profile = DirectoryProfile(name="gate_prof2", directories=[str(root)])
            pv = _always_match_prevalidation("profiled_hide2", ClassifierActionType.HIDE)
            pv.profile_name = "gate_prof2"
            pv.profile = profile
            hidden = []

            with _isolated_prevalidations([pv]):
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    result = _run_prevalidate(
                        str(media),
                        str(root),
                        hide_callback=lambda p: hidden.append(p),
                    )

            assert result == ClassifierActionType.HIDE
            assert hidden == [str(media)]

    def test_global_prevalidation_runs_for_any_base_dir(self):
        with tempfile.TemporaryDirectory() as root:
            other = Path(root) / "nested"
            other.mkdir()
            media = Path(other) / "a.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("global_hide", ClassifierActionType.HIDE)
            assert pv.profile is None

            with _isolated_prevalidations([pv]):
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    result = _run_prevalidate(str(media), str(other))

            assert result == ClassifierActionType.HIDE


class TestPrevalidateSkipPaths:
    def test_user_override_skips_without_evaluating(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("hide_override", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_prevalidations([pv]):
                ClassifierActionsManager.user_prevalidation_overrides.add(str(media))
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    counting_eval,
                ):
                    result = _run_prevalidate(str(media), root)

            assert result is None
            assert eval_calls == 0

    def test_force_bypasses_user_override(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("hide_force", ClassifierActionType.HIDE)

            with _isolated_prevalidations([pv]):
                ClassifierActionsManager.user_prevalidation_overrides.add(str(media))
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    result = _run_prevalidate(str(media), root, force=True)

            assert result == ClassifierActionType.HIDE

    def test_excluded_base_dir_skips_all_prevalidations(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("hide_excluded", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_prevalidations([pv]):
                ClassifierActionsManager.directories_to_exclude.append(root)
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    counting_eval,
                ):
                    result = _run_prevalidate(str(media), root)

            assert result is None
            assert eval_calls == 0

    def test_move_into_current_base_dir_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "out"
            target.mkdir()
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = Prevalidation(
                name="move_into_browse_dir",
                action=ClassifierActionType.MOVE,
                action_modifier=str(target),
                use_embedding=False,
                use_image_classifier=False,
                use_prompts=False,
                use_blacklist=False,
                use_prototype=False,
            )
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_prevalidations([pv]):
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    counting_eval,
                ):
                    # Browsing the move target directory skips MOVE prevalidations
                    # whose action_modifier equals base_dir.
                    result = _run_prevalidate(str(media), str(target))

            assert result is None
            assert eval_calls == 0

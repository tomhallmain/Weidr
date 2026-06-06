"""Integration tests for Phase 5: ClassifierPipeline execution in prevalidate_media.

Covers the pipeline execution block that runs when no Prevalidation fires,
including general pipeline execution, profile gating on PrevalidationPipeline,
inactive-pipeline skipping, and priority ordering vs prevalidations.
"""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    PrevalidationPipeline,
)
from files.directory_profile import DirectoryProfile
from utils.constants import ClassifierActionType


def _noop(*_args, **_kwargs):
    pass


def _run_prevalidate(media_path: str, base_dir: str, *, force: bool = False):
    return ClassifierActionsManager.prevalidate_media(
        media_path,
        lambda: base_dir,
        _noop,
        _noop,
        _noop,
        force=force,
    )


@contextmanager
def _isolated_manager(pipelines: list):
    """Isolate ClassifierActionsManager and ClassifierPipelines state for a test."""
    saved_prevalidations = ClassifierActionsManager.prevalidations[:]
    saved_cache = dict(ClassifierActionsManager.prevalidated_cache)
    saved_initialized = ClassifierActionsManager._prevalidations_initialized
    saved_exclude = list(ClassifierActionsManager.directories_to_exclude)
    saved_overrides = set(ClassifierActionsManager.user_prevalidation_overrides)
    saved_pipelines = ClassifierPipelines.pipelines[:]
    saved_pv = ClassifierPipelines._prevalidation_pipelines[:]
    saved_ac = ClassifierPipelines._action_pipelines[:]
    try:
        ClassifierActionsManager.prevalidations = []
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


def _make_pipeline(name: str, is_active: bool = True) -> ClassifierPipeline:
    """Base (action) pipeline — runs via classifier actions tab, not prevalidation."""
    return ClassifierPipeline(name=name, is_active=is_active)


def _make_pv_pipeline(name: str, is_active: bool = True) -> PrevalidationPipeline:
    """Prevalidation pipeline without a profile — runs for every directory."""
    return PrevalidationPipeline(name=name, is_active=is_active)


class TestPrevalidationPipelineRunsWhenNoPrevalidationFires:
    def test_active_pipeline_result_returned(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pipeline = _make_pv_pipeline("notify_pipe")

            with _isolated_manager([pipeline]):
                with patch(
                    "compare.classifier_pipeline_runner.run_pipeline",
                    return_value=ClassifierActionType.NOTIFY,
                ) as mock_run:
                    result = _run_prevalidate(str(media), root)

            assert result == ClassifierActionType.NOTIFY
            mock_run.assert_called_once()

    def test_inactive_pipeline_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pipeline = _make_pv_pipeline("inactive_pipe", is_active=False)

            with _isolated_manager([pipeline]):
                with patch(
                    "compare.classifier_pipeline_runner.run_pipeline",
                ) as mock_run:
                    result = _run_prevalidate(str(media), root)

            assert result is None
            mock_run.assert_not_called()

    def test_pipeline_returning_none_passes_through_as_none(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pipeline = _make_pv_pipeline("no_match_pipe")

            with _isolated_manager([pipeline]):
                with patch(
                    "compare.classifier_pipeline_runner.run_pipeline",
                    return_value=None,
                ):
                    result = _run_prevalidate(str(media), root)

            assert result is None

    def test_first_matching_pipeline_wins_second_not_called(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            first = _make_pv_pipeline("first")
            second = _make_pv_pipeline("second")
            calls = []

            def _fake_run(pipeline, image_path, **kwargs):
                calls.append(pipeline.name)
                return ClassifierActionType.HIDE if pipeline.name == "first" else ClassifierActionType.NOTIFY

            with _isolated_manager([first, second]):
                with patch("compare.classifier_pipeline_runner.run_pipeline", _fake_run):
                    result = _run_prevalidate(str(media), root)

            assert result == ClassifierActionType.HIDE
            assert calls == ["first"]

    def test_base_pipeline_not_run_during_prevalidation(self):
        """Base ClassifierPipeline (action type) must not run in prevalidate_media."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pipeline = _make_pipeline("action_pipe")

            with _isolated_manager([pipeline]):
                with patch(
                    "compare.classifier_pipeline_runner.run_pipeline",
                ) as mock_run:
                    result = _run_prevalidate(str(media), root)

            assert result is None
            mock_run.assert_not_called()


class TestPipelinePriorityVsPrevalidation:
    def test_pipeline_not_called_when_prevalidation_fires(self):
        """Prevalidation takes priority; run_pipeline must never be called."""
        from compare.classifier_action import ClassifierAction, Prevalidation

        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")
            pv = Prevalidation(
                name="hide_pv",
                action=ClassifierActionType.HIDE,
                use_embedding=False,
                use_image_classifier=False,
                use_prompts=False,
                use_blacklist=False,
                use_prototype=False,
            )
            pipeline = _make_pipeline("should_not_run")

            with _isolated_manager([pipeline]):
                ClassifierActionsManager.prevalidations = [pv]
                with patch.object(
                    ClassifierAction,
                    "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    with patch(
                        "compare.classifier_pipeline_runner.run_pipeline",
                    ) as mock_run:
                        result = _run_prevalidate(str(media), root)

            assert result == ClassifierActionType.HIDE
            mock_run.assert_not_called()


class TestPrevalidationPipelineProfileGating:
    def test_skips_when_base_dir_not_in_profile(self):
        with tempfile.TemporaryDirectory() as root:
            other = Path(root) / "other"
            other.mkdir()
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")

            profile = DirectoryProfile(name="portraits", directories=[str(root)])
            pv_pipe = PrevalidationPipeline(name="pv_pipe", profile_name="portraits")
            pv_pipe.profile = profile

            with _isolated_manager([pv_pipe]):
                with patch(
                    "compare.classifier_pipeline_runner.run_pipeline",
                ) as mock_run:
                    result = _run_prevalidate(str(media), str(other))

            assert result is None
            mock_run.assert_not_called()

    def test_runs_when_base_dir_in_profile(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "a.jpg"
            media.write_bytes(b"x")

            profile = DirectoryProfile(name="portraits2", directories=[str(root)])
            pv_pipe = PrevalidationPipeline(name="pv_pipe2", profile_name="portraits2")
            pv_pipe.profile = profile

            with _isolated_manager([pv_pipe]):
                with patch(
                    "compare.classifier_pipeline_runner.run_pipeline",
                    return_value=ClassifierActionType.NOTIFY,
                ) as mock_run:
                    result = _run_prevalidate(str(media), str(root))

            assert result == ClassifierActionType.NOTIFY
            mock_run.assert_called_once()

    def test_prevalidation_pipeline_without_profile_runs_for_any_directory(self):
        """PrevalidationPipeline with no profile is not gated — runs for every directory."""
        with tempfile.TemporaryDirectory() as root:
            other = Path(root) / "other"
            other.mkdir()
            media = Path(other) / "a.jpg"
            media.write_bytes(b"x")
            pipeline = _make_pv_pipeline("global_pv_pipe")

            with _isolated_manager([pipeline]):
                with patch(
                    "compare.classifier_pipeline_runner.run_pipeline",
                    return_value=ClassifierActionType.HIDE,
                ) as mock_run:
                    result = _run_prevalidate(str(media), str(other))

            assert result == ClassifierActionType.HIDE
            mock_run.assert_called_once()

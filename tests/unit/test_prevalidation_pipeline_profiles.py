"""Tests that prevalidation pipelines keep their directory-profile scope
across every change to ClassifierPipelines' list.

An unresolved ``profile`` makes prevalidate_media treat a profile-scoped
pipeline as global. The Pipelines tab reloads the list on open and refresh,
and editor saves swap pipeline objects, so resolution has to happen on
each of those, not once per process.
"""

from __future__ import annotations

import copy

import pytest

from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_pipeline import ClassifierPipelines, PrevalidationPipeline
from files.directory_profile import DirectoryProfile


@pytest.fixture
def photos(monkeypatch):
    profile = DirectoryProfile("Photos", ["/photos"])
    monkeypatch.setattr(DirectoryProfile, "directory_profiles", [profile, DirectoryProfile("Videos", ["/videos"])])
    return profile


def _gate(**kwargs):
    return PrevalidationPipeline(name="Gate", profile_name="Photos", **kwargs)


def test_load_resolves_profiles(photos):
    ClassifierPipelines.pipelines = [_gate()]
    ClassifierPipelines.store()

    ClassifierPipelines.load()  # what the Pipelines tab does on open

    (loaded,) = ClassifierPipelines.get_prevalidation_pipelines()
    assert loaded.profile is photos
    assert ClassifierActionsManager.get_profile_scope_dirs(loaded) == {"/photos"}


def test_added_pipeline_is_resolved(photos):
    ClassifierPipelines.pipelines = []
    pipeline = _gate()
    ClassifierPipelines.add_pipeline(pipeline)
    assert pipeline.profile is photos


def test_copied_pipeline_uses_the_registered_profile(photos):
    """The tab's Copy deep-copies the pipeline, profile object included."""
    original = _gate()
    ClassifierPipelines.pipelines = [original]
    ClassifierPipelines._rebuild_type_cache()
    duplicate = copy.deepcopy(original)
    duplicate.name = "Gate copy"
    ClassifierPipelines.add_pipeline(duplicate)
    assert duplicate.profile is photos


def test_profile_renamed_in_place_stays_resolved(photos):
    """update_profile renames the held object; profile_name stays stale until
    an explicit save, and the held link must survive a rebuild meanwhile."""
    pipeline = _gate()
    ClassifierPipelines.pipelines = [pipeline]
    ClassifierPipelines._rebuild_type_cache()

    DirectoryProfile.update_profile("Photos", DirectoryProfile("Pictures", ["/photos"]))
    ClassifierPipelines._rebuild_type_cache()

    assert pipeline.profile is photos
    assert pipeline.profile_name == "Photos"


def test_editor_change_to_another_profile_follows_profile_name(photos):
    pipeline = _gate()
    ClassifierPipelines.pipelines = [pipeline]
    ClassifierPipelines._rebuild_type_cache()

    pipeline.profile_name = "Videos"  # what the editor's save sets
    ClassifierPipelines._rebuild_type_cache()

    assert pipeline.profile.name == "Videos"


def test_instance_link_without_a_profile_name_is_kept(photos):
    pipeline = PrevalidationPipeline(name="Gate")
    pipeline.profile = photos
    ClassifierPipelines.add_pipeline(pipeline)
    assert pipeline.profile is photos
    assert pipeline.profile_name is None


def test_unresolvable_name_keeps_the_held_profile(photos, monkeypatch):
    """With nothing registered under profile_name (e.g. the profile was
    removed), the held reference stays -- rebuilding never widens a
    pipeline's scope to global."""
    pipeline = _gate()
    ClassifierPipelines.pipelines = [pipeline]
    ClassifierPipelines._rebuild_type_cache()

    monkeypatch.setattr(DirectoryProfile, "directory_profiles", [DirectoryProfile("Videos", ["/videos"])])
    ClassifierPipelines._rebuild_type_cache()

    assert pipeline.profile is photos


def test_unresolvable_name_with_nothing_held_stays_unresolved(photos):
    pipeline = PrevalidationPipeline(name="Gate", profile_name="Missing")
    ClassifierPipelines.add_pipeline(pipeline)
    assert pipeline.profile is None

"""
Unit tests for the shared ClassifierActionsManager cache-eviction helpers:
get_profile_scope_dirs, invalidate_for_profile_edit, invalidate_for_policy_save,
invalidate_for_removal, get_profile_usage/can_remove_profile pipeline-awareness,
and pipeline inclusion in the policy signature.

Covers the fixes for the reported over-eviction bug (union used where symmetric
difference was needed) and the pipeline eviction/usage blind spots found while
reexamining the wider policy.
"""

from __future__ import annotations

from compare.classifier_action import Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from compare.classifier_pipeline import ClassifierPipeline, ClassifierPipelines, PrevalidationPipeline
from files.directory_profile import DirectoryProfile


# ---------------------------------------------------------------------------
# get_profile_scope_dirs
# ---------------------------------------------------------------------------

class TestGetProfileScopeDirs:
    def test_prevalidation_with_resolved_profile_object(self):
        p = DirectoryProfile(name="prof", directories=["/a", "/b"])
        pv = Prevalidation(name="pv")
        pv.profile = p
        assert ClassifierActionsManager.get_profile_scope_dirs(pv) == {"/a", "/b"}

    def test_prevalidation_falls_back_to_profile_name_lookup(self):
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name="by_name", directories=["/c"])
        )
        pv = Prevalidation(name="pv")
        pv.profile_name = "by_name"
        assert ClassifierActionsManager.get_profile_scope_dirs(pv) == {"/c"}

    def test_prevalidation_with_no_profile_is_global(self):
        pv = Prevalidation(name="pv")
        assert ClassifierActionsManager.get_profile_scope_dirs(pv) is None

    def test_pipeline_with_resolved_profile_object(self):
        p = DirectoryProfile(name="prof2", directories=["/d"])
        pp = PrevalidationPipeline(name="pp")
        pp.profile = p
        assert ClassifierActionsManager.get_profile_scope_dirs(pp) == {"/d"}

    def test_pipeline_falls_back_to_profile_name_lookup(self):
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name="pp_by_name", directories=["/e"])
        )
        pp = PrevalidationPipeline(name="pp")
        pp.profile_name = "pp_by_name"
        assert ClassifierActionsManager.get_profile_scope_dirs(pp) == {"/e"}

    def test_pipeline_with_no_profile_is_global(self):
        pp = PrevalidationPipeline(name="pp")
        assert ClassifierActionsManager.get_profile_scope_dirs(pp) is None


# ---------------------------------------------------------------------------
# invalidate_for_profile_edit (symmetric difference) -- the reported bug's fix
# ---------------------------------------------------------------------------

class TestInvalidateForProfileEdit:
    def test_adding_one_directory_evicts_only_that_directory(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        old_dirs = {f"/d{i}" for i in range(18)}
        new_dirs = old_dirs | {"/d18"}
        ClassifierActionsManager.invalidate_for_profile_edit(old_dirs, new_dirs, reason="test")
        assert targeted == [{"/d18"}]

    def test_unchanged_scope_evicts_nothing(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        same = {"/a", "/b"}
        ClassifierActionsManager.invalidate_for_profile_edit(same, set(same), reason="test")
        assert targeted == []

    def test_global_scope_forces_full_eviction(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(
            ClassifierActionsManager, "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        ClassifierActionsManager.invalidate_for_profile_edit(None, {"/a"}, reason="test")
        assert cleared == [1]


# ---------------------------------------------------------------------------
# invalidate_for_policy_save (union) -- prevalidation/pipeline content saves
# ---------------------------------------------------------------------------

class TestInvalidateForPolicySave:
    def test_reassignment_to_disjoint_profile_evicts_both(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        ClassifierActionsManager.invalidate_for_policy_save({"/a"}, {"/b"}, reason="test")
        assert targeted == [{"/a", "/b"}]

    def test_unchanged_profile_still_evicts_full_scope(self, monkeypatch):
        """Editing a prevalidation/pipeline's own rule content (not its
        profile) must still evict its whole current scope -- the content may
        have changed even though the profile assignment didn't."""
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        same = {"/a", "/b"}
        ClassifierActionsManager.invalidate_for_policy_save(same, set(same), reason="test")
        assert targeted == [{"/a", "/b"}]

    def test_new_object_with_no_prior_state_evicts_new_scope(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        ClassifierActionsManager.invalidate_for_policy_save(set(), {"/new"}, reason="test")
        assert targeted == [{"/new"}]

    def test_global_scope_forces_full_eviction(self, monkeypatch):
        cleared = []
        monkeypatch.setattr(
            ClassifierActionsManager, "clear_prevalidation_result_cache",
            lambda: cleared.append(1),
        )
        ClassifierActionsManager.invalidate_for_policy_save({"/a"}, None, reason="test")
        assert cleared == [1]


# ---------------------------------------------------------------------------
# invalidate_for_removal
# ---------------------------------------------------------------------------

class TestInvalidateForRemoval:
    def test_profile_scoped_removal_evicts_its_own_dirs(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        ClassifierActionsManager.invalidate_for_removal({"/a", "/b"}, reason="test")
        assert targeted == [{"/a", "/b"}]

    def test_global_removal_evicts_nothing(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        ClassifierActionsManager.invalidate_for_removal(None, reason="test")
        assert targeted == []

    def test_empty_scope_evicts_nothing(self, monkeypatch):
        targeted = []
        monkeypatch.setattr(
            ClassifierActionsManager, "invalidate_for_directories",
            lambda dirs, **kw: targeted.append(dirs),
        )
        ClassifierActionsManager.invalidate_for_removal(set(), reason="test")
        assert targeted == []


# ---------------------------------------------------------------------------
# get_profile_usage / can_remove_profile -- pipeline-awareness (Defect 3)
# ---------------------------------------------------------------------------

class TestGetProfileUsage:
    def test_profile_used_only_by_pipeline_is_reported(self):
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name="pipeline_only", directories=["/a"])
        )
        pp = PrevalidationPipeline(name="pp")
        pp.profile_name = "pipeline_only"
        ClassifierPipelines.add_pipeline(pp)

        usage = ClassifierActionsManager.get_profile_usage("pipeline_only")
        assert usage["pipelines"] == ["pp"]
        assert usage["prevalidations"] == []

    def test_profile_used_by_both_prevalidation_and_pipeline(self):
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name="shared", directories=["/a"])
        )
        pv = Prevalidation(name="pv")
        pv.profile_name = "shared"
        ClassifierActionsManager.prevalidations.append(pv)
        pp = PrevalidationPipeline(name="pp")
        pp.profile_name = "shared"
        ClassifierPipelines.add_pipeline(pp)

        usage = ClassifierActionsManager.get_profile_usage("shared")
        assert usage["prevalidations"] == ["pv"]
        assert usage["pipelines"] == ["pp"]

    def test_action_pipeline_never_counts_as_usage(self):
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name="unused", directories=["/a"])
        )
        ClassifierPipelines.add_pipeline(ClassifierPipeline(name="cp"))
        usage = ClassifierActionsManager.get_profile_usage("unused")
        assert usage["pipelines"] == []
        assert usage["prevalidations"] == []

    def test_can_remove_profile_warns_for_pipeline_only_usage(self):
        DirectoryProfile.directory_profiles.append(
            DirectoryProfile(name="pipeline_guard", directories=["/a"])
        )
        pp = PrevalidationPipeline(name="pp")
        pp.profile_name = "pipeline_guard"
        ClassifierPipelines.add_pipeline(pp)

        can_remove, warnings = ClassifierActionsManager.can_remove_profile("pipeline_guard")
        assert can_remove is False
        assert any("pipelines" in w for w in warnings)


# ---------------------------------------------------------------------------
# _compute_prevalidation_signature -- pipelines included (Defect 2 safety net)
# ---------------------------------------------------------------------------

class TestPrevalidationSignatureIncludesPipelines:
    def test_signature_changes_when_pipeline_added(self):
        sig_before = ClassifierActionsManager._compute_prevalidation_signature()
        ClassifierPipelines.add_pipeline(PrevalidationPipeline(name="new_pp"))
        sig_after = ClassifierActionsManager._compute_prevalidation_signature()
        assert sig_before != sig_after

    def test_signature_changes_when_pipeline_profile_changes(self):
        pp = PrevalidationPipeline(name="pp", profile_name="a")
        ClassifierPipelines.add_pipeline(pp)
        sig_before = ClassifierActionsManager._compute_prevalidation_signature()
        pp.profile_name = "b"
        sig_after = ClassifierActionsManager._compute_prevalidation_signature()
        assert sig_before != sig_after

    def test_action_pipelines_do_not_affect_signature(self):
        """Only prevalidation pipelines participate in prevalidate_media's
        cache, so a plain ClassifierPipeline must not perturb the signature."""
        sig_before = ClassifierActionsManager._compute_prevalidation_signature()
        ClassifierPipelines.add_pipeline(ClassifierPipeline(name="cp"))
        sig_after = ClassifierActionsManager._compute_prevalidation_signature()
        assert sig_before == sig_after

"""
Tests for prevalidation active-toggle cache invalidation.

Key behaviors documented:
1. invalidate_session_cache_only: clears in-memory cache + memo, leaves buckets intact.
2. invalidate_for_directories(dirs, evict_buckets=False): targets only session-cache
   entries for the given directories; buckets are preserved.
3. Toggle-off: new files (not in bucket) skip the inactive pv; old files (already in
   bucket) still return the bucket result because the epoch is unchanged.
4. Accident case (toggle-off → immediate toggle-on, no browsing in between): the
   pre-toggle bucket result is reused on the next access without re-evaluation.
5. Profile-scoped toggles invalidate only the profile's directories in the session cache.
6. Signature changes when is_active changes, and is restored after a double toggle.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import lib.file_invalidation_cache as fic
from compare.classifier_action import ClassifierAction, Prevalidation
from compare.classifier_actions_manager import ClassifierActionsManager
from files.directory_profile import DirectoryProfile
from utils.constants import ClassifierActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop(*_args, **_kwargs):
    pass


def _run_prevalidate(media_path: str, base_dir: str, *, force: bool = False):
    return ClassifierActionsManager.prevalidate_media(
        media_path,
        lambda: base_dir,
        _noop,
        _noop,
        _noop,
        blur_callback=None,
        force=force,
    )


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


@contextmanager
def _isolated_cache(prevalidations):
    """
    Save and restore ALL prevalidation + file-invalidation-cache state around a test.

    Covers:
    - ClassifierActionsManager: prevalidations, prevalidated_cache, _prevalidations_initialized,
      directories_to_exclude, user_prevalidation_overrides
    - lib.file_invalidation_cache: _signature_memo, _policy_epoch, _file_buckets
    """
    # Manager state
    saved_pvs = ClassifierActionsManager.prevalidations[:]
    saved_session = dict(ClassifierActionsManager.prevalidated_cache)
    saved_init = ClassifierActionsManager._prevalidations_initialized
    saved_exclude = list(ClassifierActionsManager.directories_to_exclude)
    saved_overrides = set(ClassifierActionsManager.user_prevalidation_overrides)
    # File-invalidation-cache state
    saved_memo = fic.get_signature_memo()
    saved_epoch = fic._policy_epoch
    saved_buckets = dict(fic._file_buckets)

    try:
        ClassifierActionsManager.prevalidations = list(prevalidations)
        ClassifierActionsManager.prevalidated_cache.clear()
        ClassifierActionsManager._prevalidations_initialized = True
        ClassifierActionsManager.directories_to_exclude.clear()
        ClassifierActionsManager.user_prevalidation_overrides.clear()
        fic.set_signature_memo(None)
        fic._file_buckets.clear()
        yield
    finally:
        ClassifierActionsManager.prevalidations = saved_pvs
        ClassifierActionsManager.prevalidated_cache.clear()
        ClassifierActionsManager.prevalidated_cache.update(saved_session)
        ClassifierActionsManager._prevalidations_initialized = saved_init
        ClassifierActionsManager.directories_to_exclude.clear()
        ClassifierActionsManager.directories_to_exclude.extend(saved_exclude)
        ClassifierActionsManager.user_prevalidation_overrides.clear()
        ClassifierActionsManager.user_prevalidation_overrides.update(saved_overrides)
        fic.set_signature_memo(saved_memo)
        fic._policy_epoch = saved_epoch
        fic._file_buckets.clear()
        fic._file_buckets.update(saved_buckets)


# ---------------------------------------------------------------------------
# invalidate_session_cache_only
# ---------------------------------------------------------------------------

class TestInvalidateSessionCacheOnly:
    def test_clears_entire_session_cache(self):
        pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
        with _isolated_cache([pv]):
            ClassifierActionsManager.prevalidated_cache["a.jpg"] = ClassifierActionType.HIDE
            ClassifierActionsManager.prevalidated_cache["b.jpg"] = ClassifierActionType.BLUR
            ClassifierActionsManager.invalidate_session_cache_only()
            assert ClassifierActionsManager.prevalidated_cache == {}

    def test_clears_signature_memo(self):
        with _isolated_cache([]):
            fic.set_signature_memo("stale_hash")
            ClassifierActionsManager.invalidate_session_cache_only()
            assert fic.get_signature_memo() is None

    def test_bucket_is_not_cleared(self):
        """Session-cache-only invalidation leaves bucket entries intact."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    _run_prevalidate(str(media), root)

                bucket = fic.get_file_bucket_for_media(str(media))
                assert bucket._has_entry

                ClassifierActionsManager.invalidate_session_cache_only()

                ok, _ = bucket.try_get((str(media),))
                assert ok  # bucket entry survives

    def test_next_access_hits_bucket_without_reeval(self):
        """After session-cache clear, the next browse hits the bucket — no re-evaluation."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    _run_prevalidate(str(media), root)          # eval_calls → 1
                    ClassifierActionsManager.invalidate_session_cache_only()
                    result = _run_prevalidate(str(media), root)  # bucket hit

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1  # no re-evaluation

    def test_does_not_affect_empty_cache(self):
        """Calling on an already-empty cache is a safe no-op."""
        with _isolated_cache([]):
            ClassifierActionsManager.invalidate_session_cache_only()
            assert ClassifierActionsManager.prevalidated_cache == {}
            assert fic.get_signature_memo() is None


# ---------------------------------------------------------------------------
# invalidate_for_directories(evict_buckets=False)
# ---------------------------------------------------------------------------

class TestInvalidateForDirectoriesNoBucketEviction:
    def test_clears_only_entries_for_target_dir(self):
        with tempfile.TemporaryDirectory() as root:
            dir_a = Path(root) / "a"
            dir_b = Path(root) / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            file_a = str(dir_a / "img.jpg")
            file_b = str(dir_b / "img.jpg")
            with _isolated_cache([]):
                ClassifierActionsManager.prevalidated_cache[file_a] = ClassifierActionType.HIDE
                ClassifierActionsManager.prevalidated_cache[file_b] = ClassifierActionType.HIDE
                ClassifierActionsManager.invalidate_for_directories(
                    {str(dir_a)}, evict_buckets=False
                )
                assert file_a not in ClassifierActionsManager.prevalidated_cache
                assert file_b in ClassifierActionsManager.prevalidated_cache

    def test_clears_multiple_files_in_target_dir(self):
        with tempfile.TemporaryDirectory() as root:
            dir_a = Path(root) / "a"
            dir_a.mkdir()
            file1 = str(dir_a / "one.jpg")
            file2 = str(dir_a / "two.jpg")
            with _isolated_cache([]):
                ClassifierActionsManager.prevalidated_cache[file1] = ClassifierActionType.HIDE
                ClassifierActionsManager.prevalidated_cache[file2] = ClassifierActionType.BLUR
                ClassifierActionsManager.invalidate_for_directories(
                    {str(dir_a)}, evict_buckets=False
                )
                assert file1 not in ClassifierActionsManager.prevalidated_cache
                assert file2 not in ClassifierActionsManager.prevalidated_cache

    def test_clears_signature_memo(self):
        with _isolated_cache([]):
            fic.set_signature_memo("stale")
            ClassifierActionsManager.invalidate_for_directories(
                {"/some/dir"}, evict_buckets=False
            )
            assert fic.get_signature_memo() is None

    def test_bucket_survives_for_affected_dir(self):
        """Bucket entries for the target dir are NOT evicted when evict_buckets=False."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    _run_prevalidate(str(media), root)    # eval_calls → 1
                    ClassifierActionsManager.invalidate_for_directories(
                        {root}, evict_buckets=False
                    )
                    result = _run_prevalidate(str(media), root)  # bucket hit

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1

    def test_evict_buckets_true_forces_reeval(self):
        """Contrast: evict_buckets=True (default) removes the bucket and forces re-eval."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    _run_prevalidate(str(media), root)    # eval_calls → 1
                    ClassifierActionsManager.invalidate_for_directories(
                        {root}, evict_buckets=True
                    )
                    result = _run_prevalidate(str(media), root)  # bucket evicted → re-eval

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 2

    def test_unrelated_dir_session_entry_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            dir_a = Path(root) / "a"
            dir_b = Path(root) / "b"
            dir_a.mkdir()
            dir_b.mkdir()
            file_b = str(dir_b / "img.jpg")
            with _isolated_cache([]):
                ClassifierActionsManager.prevalidated_cache[file_b] = ClassifierActionType.HIDE
                ClassifierActionsManager.invalidate_for_directories(
                    {str(dir_a)}, evict_buckets=False
                )
                assert file_b in ClassifierActionsManager.prevalidated_cache


# ---------------------------------------------------------------------------
# Signature changes on is_active toggle
# ---------------------------------------------------------------------------

class TestSignatureOnToggle:
    def test_signature_changes_when_pv_deactivated(self):
        pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
        with _isolated_cache([pv]):
            sig_before = ClassifierActionsManager.get_prevalidation_signature()
            pv.is_active = False
            ClassifierActionsManager.invalidate_session_cache_only()  # clears memo
            sig_after = ClassifierActionsManager.get_prevalidation_signature()
            assert sig_before != sig_after

    def test_signature_restored_after_double_toggle(self):
        pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
        with _isolated_cache([pv]):
            sig_active = ClassifierActionsManager.get_prevalidation_signature()
            pv.is_active = False
            ClassifierActionsManager.invalidate_session_cache_only()
            pv.is_active = True
            ClassifierActionsManager.invalidate_session_cache_only()
            sig_reactivated = ClassifierActionsManager.get_prevalidation_signature()
            assert sig_active == sig_reactivated

    def test_toggle_clears_memo_so_signature_is_recomputed(self):
        pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
        with _isolated_cache([pv]):
            ClassifierActionsManager.get_prevalidation_signature()  # populates memo
            assert fic.get_signature_memo() is not None
            pv.is_active = False
            ClassifierActionsManager.invalidate_session_cache_only()
            assert fic.get_signature_memo() is None

    def test_signature_differs_for_each_active_state(self):
        """Three distinct signatures: active, inactive, active again (same as first)."""
        pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
        with _isolated_cache([pv]):
            sig_on = ClassifierActionsManager.get_prevalidation_signature()
            pv.is_active = False
            ClassifierActionsManager.invalidate_session_cache_only()
            sig_off = ClassifierActionsManager.get_prevalidation_signature()
            pv.is_active = True
            ClassifierActionsManager.invalidate_session_cache_only()
            sig_back_on = ClassifierActionsManager.get_prevalidation_signature()

            assert sig_on != sig_off
            assert sig_on == sig_back_on


# ---------------------------------------------------------------------------
# Toggle-active behaviour: deactivation and reactivation
# ---------------------------------------------------------------------------

class TestToggleActiveBehaviour:
    def test_deactivation_skips_pv_for_new_files(self):
        """After toggle-off, new files (no bucket entry) skip the inactive pv."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                pv.is_active = False
                ClassifierActionsManager.invalidate_session_cache_only()

                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    result = _run_prevalidate(str(media), root)

                assert result is None
                assert eval_calls == 0  # pv was inactive — loop skipped it entirely

    def test_reactivation_evaluates_fresh_files(self):
        """After toggle-off → toggle-on with no browsing in between, fresh files are evaluated."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                # Toggle off then immediately back on — no browsing during inactive
                pv.is_active = False
                ClassifierActionsManager.invalidate_session_cache_only()
                pv.is_active = True
                ClassifierActionsManager.invalidate_session_cache_only()

                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    result = _run_prevalidate(str(media), root)

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1  # new file, fully evaluated

    def test_old_bucket_entry_returned_after_deactivation(self):
        """
        Files that were in the bucket BEFORE deactivation still return the bucket
        result even after toggle-off — the epoch is unchanged so the bucket is valid.

        This is the expected trade-off: files cached before the toggle survive
        in the bucket so the accident case (toggle-off → immediate toggle-on) is fast.
        """
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    # Evaluate while pv is active → HIDE cached in session + bucket
                    _run_prevalidate(str(media), root)
                    assert eval_calls == 1

                    # Toggle off
                    pv.is_active = False
                    ClassifierActionsManager.invalidate_session_cache_only()

                    # Next access: session miss, bucket hit → returns pre-deactivation result
                    result = _run_prevalidate(str(media), root)

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1  # bucket returned the old result

    def test_session_cache_cleared_on_toggle_off(self):
        """Toggle-off clears the session-cache entry for the affected file."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            with _isolated_cache([pv]):
                ClassifierActionsManager.prevalidated_cache[str(media)] = (
                    ClassifierActionType.HIDE
                )
                pv.is_active = False
                ClassifierActionsManager.invalidate_session_cache_only()
                assert str(media) not in ClassifierActionsManager.prevalidated_cache

    def test_session_cache_cleared_on_toggle_on(self):
        """Toggle-on also clears the session cache so stale entries don't linger."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            with _isolated_cache([pv]):
                # Simulate: file was seen during inactive period → None in cache
                pv.is_active = False
                ClassifierActionsManager.prevalidated_cache[str(media)] = None
                pv.is_active = True
                ClassifierActionsManager.invalidate_session_cache_only()
                assert str(media) not in ClassifierActionsManager.prevalidated_cache


# ---------------------------------------------------------------------------
# Accident case
# ---------------------------------------------------------------------------

class TestAccidentCase:
    """
    Toggle-off → immediate toggle-on, no browsing between the two toggles.
    The pre-toggle bucket result should be reused without re-evaluation.
    """

    def test_global_pv_bucket_reused_after_double_toggle(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    # Initial evaluation
                    _run_prevalidate(str(media), root)
                    assert eval_calls == 1

                    # Accident: toggle off then immediately back on
                    pv.is_active = False
                    ClassifierActionsManager.invalidate_session_cache_only()
                    pv.is_active = True
                    ClassifierActionsManager.invalidate_session_cache_only()

                    # Should hit bucket — no re-evaluation
                    result = _run_prevalidate(str(media), root)

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1

    def test_profile_scoped_pv_bucket_reused_after_double_toggle(self):
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            profile = DirectoryProfile(name="prof", directories=[root])
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            pv.profile_name = "prof"
            pv.profile = profile
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    _run_prevalidate(str(media), root)
                    assert eval_calls == 1

                    pv.is_active = False
                    ClassifierActionsManager.invalidate_for_directories(
                        {root}, evict_buckets=False
                    )
                    pv.is_active = True
                    ClassifierActionsManager.invalidate_for_directories(
                        {root}, evict_buckets=False
                    )

                    result = _run_prevalidate(str(media), root)

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1

    def test_files_browsed_during_inactive_return_stale_bucket_after_reactivation(self):
        """
        Files browsed while the pv was inactive are cached as None in the bucket.
        After reactivation, the bucket still returns None for those files — this is
        the known trade-off for keeping the accident case cheap.  Use force=True
        to bypass the bucket and force re-evaluation if needed.
        """
        with tempfile.TemporaryDirectory() as root:
            media_during = Path(root) / "during.jpg"
            media_during.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                # Toggle off — pv inactive
                pv.is_active = False
                ClassifierActionsManager.invalidate_session_cache_only()

                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    # Browse new file while inactive → None stored in bucket
                    result_inactive = _run_prevalidate(str(media_during), root)
                    assert result_inactive is None
                    assert eval_calls == 0  # pv inactive; loop skipped it

                    # Toggle back on
                    pv.is_active = True
                    ClassifierActionsManager.invalidate_session_cache_only()

                    # Bucket has None for this file (cached during inactive period)
                    result_after = _run_prevalidate(str(media_during), root)

                # Bucket returns stale None — this is the documented trade-off
                assert result_after is None
                assert eval_calls == 0  # still came from bucket, no eval

    def test_force_bypasses_stale_bucket_after_reactivation(self):
        """force=True skips both caches and re-evaluates even if bucket has stale None."""
        with tempfile.TemporaryDirectory() as root:
            media = Path(root) / "img.jpg"
            media.write_bytes(b"x")
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            eval_calls = 0

            def counting_eval(*_args, **_kwargs):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                # Deactivate, browse file (None in bucket), reactivate
                pv.is_active = False
                ClassifierActionsManager.invalidate_session_cache_only()
                _run_prevalidate(str(media), root)  # None cached in bucket

                pv.is_active = True
                ClassifierActionsManager.invalidate_session_cache_only()

                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    result = _run_prevalidate(str(media), root, force=True)

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1  # force=True bypassed the stale bucket


# ---------------------------------------------------------------------------
# Profile-scoped toggle selectivity
# ---------------------------------------------------------------------------

class TestProfileScopedToggleSelectivity:
    def test_only_profile_dirs_cleared_from_session_cache(self):
        with tempfile.TemporaryDirectory() as root:
            profile_dir = Path(root) / "pdir"
            other_dir = Path(root) / "other"
            profile_dir.mkdir()
            other_dir.mkdir()
            file_prof = str(profile_dir / "a.jpg")
            file_other = str(other_dir / "b.jpg")
            with _isolated_cache([]):
                ClassifierActionsManager.prevalidated_cache[file_prof] = (
                    ClassifierActionType.HIDE
                )
                ClassifierActionsManager.prevalidated_cache[file_other] = (
                    ClassifierActionType.HIDE
                )
                ClassifierActionsManager.invalidate_for_directories(
                    {str(profile_dir)}, evict_buckets=False
                )
                assert file_prof not in ClassifierActionsManager.prevalidated_cache
                assert file_other in ClassifierActionsManager.prevalidated_cache

    def test_profile_scoped_pv_runs_for_profile_dir(self):
        with tempfile.TemporaryDirectory() as root:
            profile_dir = Path(root) / "pdir"
            profile_dir.mkdir()
            media = Path(profile_dir) / "img.jpg"
            media.write_bytes(b"x")
            profile = DirectoryProfile(name="p", directories=[str(profile_dir)])
            pv = _always_match_prevalidation("pv", ClassifierActionType.HIDE)
            pv.profile_name = "p"
            pv.profile = profile

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    return_value=(True, None),
                ):
                    result = _run_prevalidate(str(media), str(profile_dir))

                assert result == ClassifierActionType.HIDE

    def test_profile_scoped_pv_skips_unrelated_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            profile_dir = Path(root) / "pdir"
            other_dir = Path(root) / "other"
            profile_dir.mkdir()
            other_dir.mkdir()
            media = Path(other_dir) / "img.jpg"
            media.write_bytes(b"x")
            profile = DirectoryProfile(name="p2", directories=[str(profile_dir)])
            pv = _always_match_prevalidation("pv2", ClassifierActionType.HIDE)
            pv.profile_name = "p2"
            pv.profile = profile
            eval_calls = 0

            def counting_eval(*_a, **_kw):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    result = _run_prevalidate(str(media), str(other_dir))

                assert result is None
                assert eval_calls == 0

    def test_profile_dir_bucket_preserved_on_accident_toggle(self):
        with tempfile.TemporaryDirectory() as root:
            profile_dir = Path(root) / "pdir"
            profile_dir.mkdir()
            media = Path(profile_dir) / "img.jpg"
            media.write_bytes(b"x")
            profile = DirectoryProfile(name="p3", directories=[str(profile_dir)])
            pv = _always_match_prevalidation("pv3", ClassifierActionType.HIDE)
            pv.profile_name = "p3"
            pv.profile = profile
            eval_calls = 0

            def counting_eval(*_a, **_kw):
                nonlocal eval_calls
                eval_calls += 1
                return True, None

            with _isolated_cache([pv]):
                with patch.object(
                    ClassifierAction, "_evaluate_image_path_match",
                    counting_eval,
                ):
                    _run_prevalidate(str(media), str(profile_dir))
                    assert eval_calls == 1

                    # Accident toggle
                    pv.is_active = False
                    ClassifierActionsManager.invalidate_for_directories(
                        {str(profile_dir)}, evict_buckets=False
                    )
                    pv.is_active = True
                    ClassifierActionsManager.invalidate_for_directories(
                        {str(profile_dir)}, evict_buckets=False
                    )

                    result = _run_prevalidate(str(media), str(profile_dir))

                assert result == ClassifierActionType.HIDE
                assert eval_calls == 1  # bucket reused

    def test_multiple_profile_dirs_all_cleared(self):
        """All directories in the profile are cleared in one invalidation call."""
        with tempfile.TemporaryDirectory() as root:
            dir1 = Path(root) / "d1"
            dir2 = Path(root) / "d2"
            dir_out = Path(root) / "out"
            for d in (dir1, dir2, dir_out):
                d.mkdir()
            f1 = str(dir1 / "a.jpg")
            f2 = str(dir2 / "b.jpg")
            f_out = str(dir_out / "c.jpg")
            with _isolated_cache([]):
                for p in (f1, f2, f_out):
                    ClassifierActionsManager.prevalidated_cache[p] = (
                        ClassifierActionType.HIDE
                    )
                ClassifierActionsManager.invalidate_for_directories(
                    {str(dir1), str(dir2)}, evict_buckets=False
                )
                assert f1 not in ClassifierActionsManager.prevalidated_cache
                assert f2 not in ClassifierActionsManager.prevalidated_cache
                assert f_out in ClassifierActionsManager.prevalidated_cache

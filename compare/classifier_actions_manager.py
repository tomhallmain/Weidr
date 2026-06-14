"""
Classification Actions Manager module for managing prevalidations, classifier actions, and directory profiles.

This module centralizes the management of:
- Prevalidations list
- Classifier actions list  
- Directory profiles and their usage tracking
"""

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction, Prevalidation
from compare.lookahead import Lookahead
from files.directory_profile import DirectoryProfile
from lib.file_invalidation_cache import (
    DEFAULT_STALE_ENTRY_MAX_AGE_SECONDS,
    FileKeyedInvalidationCache,
    estimate_file_buckets_map_overhead_bytes,
    evacuate_buckets_for_directories,
    evacuate_stale_file_buckets,
    get_file_bucket_for_media,
    get_signature_memo,
    install_file_bucket,
    invalidate_policy_caches,
    iter_file_buckets,
    set_signature_memo,
)
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.media_utils import is_video_path_by_extension
from utils.constants import ClassifierActionType
from utils.logging_setup import get_logger
from utils.translations import _



logger = get_logger("classifier_actions_manager")





class ClassifierActionsManager:
    """Manages prevalidations, classifier actions, and directory profiles."""

    PREVALIDATION_FILE_CACHE_META_KEY = "prevalidation_file_invalidation_cache_v2"

    # Lists managed by this module
    prevalidations: List['Prevalidation'] = []
    classifier_actions: List['ClassifierAction'] = []
    prevalidated_cache: Dict[str, Optional[ClassifierActionType]] = {}
    user_prevalidation_overrides: set = set()
    directories_to_exclude: list[str] = []
    _prevalidations_initialized: bool = False

    @staticmethod
    def _parse_cached_action_name(name: Optional[str]) -> Optional[ClassifierActionType]:
        if name is None:
            return None
        try:
            return ClassifierActionType[name]
        except KeyError:
            return None

    @staticmethod
    def _compute_prevalidation_signature() -> str:
        def _strip_last_used(d):
            out = dict(d)
            out.pop("_last_used_profile", None)
            return out

        payload = {
            "prevalidations": [_strip_last_used(pv.to_dict()) for pv in ClassifierActionsManager.prevalidations],
            "classifier_actions": [_strip_last_used(a.to_dict()) for a in ClassifierActionsManager.classifier_actions],
            "enable_prevalidations": config.enable_prevalidations,
            "dynamic_media_min_sample_count": config.dynamic_media_min_sample_count,
            "dynamic_media_max_sample_frames": config.dynamic_media_max_sample_frames,
            "dynamic_media_max_sample_pages": config.dynamic_media_max_sample_pages,
            "profiles": [p.to_dict() for p in DirectoryProfile.directory_profiles],
            "lookaheads": [la.to_dict() for la in Lookahead.lookaheads],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def get_prevalidation_signature() -> str:
        """Policy fingerprint; memo lives in ``lib.file_invalidation_cache``."""
        m = get_signature_memo()
        if m is None:
            s = ClassifierActionsManager._compute_prevalidation_signature()
            set_signature_memo(s)
            return s
        return m

    @staticmethod
    def record_prevalidation_override(media_path: str) -> None:
        """Mark *media_path* as user-overridden so prevalidation is skipped for it."""
        ClassifierActionsManager.user_prevalidation_overrides.add(media_path)

    @staticmethod
    def _invalidate_after_prevalidation_policy_change() -> None:
        """Call when rules or models that affect prevalidation outcomes change."""
        logger.info(
            "Prevalidation cache: full eviction — policy change affects all directories"
        )
        invalidate_policy_caches()
        ClassifierActionsManager.prevalidated_cache.clear()
        ClassifierActionsManager.user_prevalidation_overrides.clear()

    @staticmethod
    def load_prevalidation_file_cache_from_disk() -> None:
        raw = app_info_cache.get_meta(ClassifierActionsManager.PREVALIDATION_FILE_CACHE_META_KEY, default_val=None)
        if not isinstance(raw, dict):
            return
        disk_sig = raw.get("signature")
        entries = raw.get("entries")
        if not isinstance(disk_sig, str) or not isinstance(entries, list):
            return
        if disk_sig != ClassifierActionsManager.get_prevalidation_signature():
            return
        overrides = raw.get("user_overrides")
        if isinstance(overrides, list):
            ClassifierActionsManager.user_prevalidation_overrides.update(
                p for p in overrides if isinstance(p, str)
            )
        for item in entries:
            if not isinstance(item, dict):
                continue
            try:
                media_key = item["media_key"]
                path_mtimes = item["path_mtimes"]
                val = ClassifierActionsManager._parse_cached_action_name(item.get("value"))
                bucket_sig = item.get("bucket_signature", disk_sig)
                epoch_at_set = item.get("epoch_at_set")
            except (KeyError, TypeError):
                continue
            cached_at = item.get("cached_at_unix")
            if isinstance(cached_at, (int, float)):
                if time.time() - float(cached_at) > DEFAULT_STALE_ENTRY_MAX_AGE_SECONDS:
                    continue
            b: FileKeyedInvalidationCache[Optional[ClassifierActionType]] = FileKeyedInvalidationCache()
            b.load_from_snapshot(
                path_mtimes,
                val,
                bucket_sig if isinstance(bucket_sig, str) else disk_sig,
                epoch_at_set if isinstance(epoch_at_set, int) else None,
                float(cached_at) if isinstance(cached_at, (int, float)) else None,
            )
            install_file_bucket(media_key, b)

    @staticmethod
    def store_prevalidation_file_cache_to_disk() -> None:
        evacuate_stale_file_buckets()
        entries = []
        for media_key, bucket in iter_file_buckets():
            snap = bucket.snapshot_for_persistence()
            if snap is None:
                continue
            v = bucket.peek_value()
            entries.append(
                {
                    "media_key": media_key,
                    "path_mtimes": snap["path_mtimes"],
                    "bucket_signature": snap.get("signature"),
                    "epoch_at_set": snap.get("epoch_at_set"),
                    "cached_at_unix": snap.get("cached_at_unix"),
                    "value": v.name if v is not None else None,
                }
            )
        payload = {
            "signature": ClassifierActionsManager.get_prevalidation_signature(),
            "entries": entries,
            "user_overrides": list(ClassifierActionsManager.user_prevalidation_overrides),
        }
        app_info_cache.set_meta(ClassifierActionsManager.PREVALIDATION_FILE_CACHE_META_KEY, payload)

    @staticmethod
    def clear_prevalidation_result_cache() -> None:
        ClassifierActionsManager._invalidate_after_prevalidation_policy_change()

    @staticmethod
    def get_prevalidation_cache_statistics() -> tuple[int, int, int]:
        """
        Approximate in-memory footprint of the session dict plus file buckets,
        and counts of unique cached media paths and their parent directories.

        Returns:
            ``(estimated_bytes, n_cached_media_paths, n_parent_directories)``
        """
        total = sys.getsizeof(ClassifierActionsManager.prevalidated_cache)
        for path, val in ClassifierActionsManager.prevalidated_cache.items():
            total += sys.getsizeof(path) + sys.getsizeof(val)

        total += estimate_file_buckets_map_overhead_bytes()
        for media_key, bucket in iter_file_buckets():
            total += sys.getsizeof(media_key) + bucket.estimated_footprint_bytes()

        norm_paths: set[str] = set()
        for path in ClassifierActionsManager.prevalidated_cache:
            norm_paths.add(FileKeyedInvalidationCache._path_key(path))
        for media_key, bucket in iter_file_buckets():
            if bucket.has_cached_entry():
                norm_paths.add(media_key)

        n_items = len(norm_paths)
        parent_dirs = {os.path.dirname(p) for p in norm_paths if p}
        n_dirs = len(parent_dirs)
        return total, n_items, n_dirs

    @staticmethod
    def invalidate_session_cache_only() -> None:
        """Clear the in-memory prevalidated_cache and signature memo without
        touching the on-disk buckets.

        Use for is_active toggles on global-scoped prevalidations: the bucket
        entries self-invalidate via signature mismatch on next access, so
        explicit bucket eviction is unnecessary and would discard still-valid
        entries in the accident (toggle-off then immediately toggle-on) case.
        """
        logger.info("Prevalidation cache: session-cache + memo cleared (buckets preserved)")
        ClassifierActionsManager.prevalidated_cache.clear()
        set_signature_memo(None)

    @staticmethod
    def invalidate_for_directories(
        directories: set[str], *, evict_buckets: bool = True
    ) -> None:
        """
        Targeted eviction: drop caches only for files whose immediate parent
        directory is in *directories*.

        Does NOT bump the policy epoch, so unaffected buckets remain valid
        in-memory.  Clears the signature memo so subsequent bucket.set()
        calls record the updated policy fingerprint.

        Use this when a profile-scoped prevalidation changes and only the
        affected directories need re-evaluation.  Fall back to
        clear_prevalidation_result_cache() for global-scope changes.

        Pass ``evict_buckets=False`` when only the signature changes (e.g. an
        is_active toggle): stale bucket entries self-invalidate via signature
        mismatch on next access, so explicit eviction is unnecessary and would
        discard still-valid entries for the no-op (accident) case.
        """
        logger.info(
            "Prevalidation cache: targeted eviction for %d dir(s)%s: %s",
            len(directories),
            "" if evict_buckets else " (session cache only)",
            sorted(directories),
        )
        if evict_buckets:
            evacuate_buckets_for_directories(directories)
        norm_dirs = {os.path.normcase(os.path.normpath(d)) for d in directories}
        for path in list(ClassifierActionsManager.prevalidated_cache.keys()):
            if os.path.normcase(os.path.normpath(os.path.dirname(path))) in norm_dirs:
                del ClassifierActionsManager.prevalidated_cache[path]
        set_signature_memo(None)

    @staticmethod
    def _prevalidations_post_init():
        """Lazy initialization of prevalidations - called just before first use."""
        if ClassifierActionsManager._prevalidations_initialized:
            return
        temp_prevalidations = ClassifierActionsManager.prevalidations[:]
        for prevalidation in temp_prevalidations:
            try:
                prevalidation.update_profile_instance()
                prevalidation.validate_dirs()
                prevalidation.ensure_prototype_loaded(None)
                prevalidation.validate()
                prevalidation.can_run = True
                prevalidation.initialization_error = None
                prevalidation._blocked_run_logged = False
            except Exception as e:
                prevalidation.can_run = False
                prevalidation.initialization_error = str(e)
                logger.error(
                    "Prevalidation %r cannot run until fixed (kept in list): %s",
                    prevalidation.name,
                    e,
                )
        # Load ClassifierPipelines and resolve profile instances so they are
        # ready for the pipeline execution loop in prevalidate_media().
        try:
            from compare.classifier_pipeline import ClassifierPipelines
            ClassifierPipelines.load()
            for _pipeline in ClassifierPipelines.get_prevalidation_pipelines():
                try:
                    _pipeline.update_profile_instance()
                except Exception:
                    pass
        except Exception:
            logger.exception("Failed to load ClassifierPipelines during post-init")

        ClassifierActionsManager._prevalidations_initialized = True

    @staticmethod
    def reset_prevalidation_lazy_init() -> None:
        """Call when image classifier registry changes so lazy init re-runs on next prevalidate."""
        ClassifierActionsManager._prevalidations_initialized = False
        ClassifierActionsManager._invalidate_after_prevalidation_policy_change()
    
    @staticmethod
    def is_dynamic_prevalidation_media(path: str) -> bool:
        """Return True if *path* requires dynamic (frame-sampling) prevalidation."""
        path_lower = path.lower()
        return (
            (config.enable_videos and is_video_path_by_extension(path))
            or (config.enable_gifs and path_lower.endswith(".gif"))
            or (config.enable_pdfs and path_lower.endswith(".pdf"))
        )

    @staticmethod
    def prevalidate_media(
        media_path,
        get_base_dir_func,
        callbacks: ActionCallbacks,
        force: bool = False,
    ) -> Optional[ClassifierActionType]:
        """Run active prevalidations for *media_path* while browsing *base_dir*.

        Gating uses the app's current base directory (from *get_base_dir_func*), not
        the parent folder of *media_path*. Recursive subdirectory listing only
        changes which files appear in the browser; it does not treat each file's
        directory as the effective base for profile or exclusion checks.

        Relevant behavior today:

        - **Directory profile:** If a prevalidation has a profile, it runs only when
          ``base_dir`` is listed in ``profile.directories`` (exact path match).
          Files under other folders are not skipped on that basis alone.

        - **MOVE/COPY into current dir:** MOVE and COPY prevalidations whose target
          (``action_modifier``) equals ``base_dir`` are skipped so files are not
          moved or copied into the directory being browsed.

        - **MOVE/COPY target exclusion:** Each MOVE and COPY prevalidation's target
          directory is recorded in ``directories_to_exclude``. When the user sets
          ``base_dir`` to one of those paths (e.g. a subdirectory that received files
          from an earlier bulk or per-file run on the parent), this function returns
          immediately and no prevalidations run for any media there. That is why
          browsing inside a former move/copy target can appear to skip prevalidation
          entirely.

        - **Idempotent MOVE:** When a match fires but the file is already in the MOVE
          target (``dirname(media_path) == target``), no file I/O runs. That MOVE
          result is stored in ``prevalidated_cache`` and the file bucket (same as
          cache-type actions), so later navigations skip re-running the classifier.

        - **User override:** If *media_path* is present in
          ``user_prevalidation_overrides``, the function returns ``None`` immediately
          and no prevalidation runs. ``force=True`` bypasses this check (and the
          cache) to unconditionally re-evaluate the file.
        """
        # Lazy initialization - ensure prevalidations are initialized before first use
        if not ClassifierActionsManager._prevalidations_initialized:
            ClassifierActionsManager._prevalidations_post_init()

        # Reset lookahead cache for this prevalidate call
        for lookahead in Lookahead.lookaheads:
            lookahead.run_result = None

        base_dir = get_base_dir_func()
        if len(ClassifierActionsManager.directories_to_exclude) > 0 and base_dir in ClassifierActionsManager.directories_to_exclude:
            return None

        if not force and media_path in ClassifierActionsManager.user_prevalidation_overrides:
            return None

        bucket = get_file_bucket_for_media(media_path)
        if not force:
            if media_path in ClassifierActionsManager.prevalidated_cache:
                return ClassifierActionsManager.prevalidated_cache[media_path]

            ok_hit, cached_action = bucket.try_get((media_path,))
            if ok_hit:
                ClassifierActionsManager.prevalidated_cache[media_path] = cached_action
                return cached_action

        prevalidation_action = None
        matched_prevalidation: Optional[Prevalidation] = None
        for prevalidation in ClassifierActionsManager.prevalidations:
            if prevalidation.is_active and prevalidation.can_run:
                if prevalidation.is_move_action() and prevalidation.action_modifier == base_dir:
                    continue
                if prevalidation.profile is not None and base_dir not in prevalidation.profile.directories:
                    continue
                if not prevalidation.media_type_allowed(media_path):
                    continue
                prevalidation_action = prevalidation.run_on_media_path(
                    media_path,
                    callbacks,
                    base_directory=base_dir,
                )
                if prevalidation_action is not None:
                    matched_prevalidation = prevalidation
                    break

        if prevalidation_action is None:
            try:
                from compare.classifier_pipeline import ClassifierPipelines
                from compare.classifier_pipeline_runner import run_pipeline
                for pipeline in ClassifierPipelines.get_prevalidation_pipelines():
                    if not pipeline.is_active:
                        continue
                    if pipeline.profile is not None and base_dir not in pipeline.profile.directories:
                        continue
                    if not pipeline.media_type_allowed(media_path):
                        continue
                    pipeline_action = run_pipeline(
                        pipeline,
                        media_path,
                        callbacks,
                        base_directory=base_dir,
                    )
                    if pipeline_action is not None:
                        prevalidation_action = pipeline_action
                        break
            except Exception:
                logger.exception("Error running classifier pipelines for %s", media_path)

        if ClassifierActionsManager._should_persist_prevalidation_cache(
            prevalidation_action, matched_prevalidation, media_path
        ):
            ClassifierActionsManager.prevalidated_cache[media_path] = prevalidation_action
            pv_sig = ClassifierActionsManager.get_prevalidation_signature()
            bucket.set((media_path,), prevalidation_action, pv_sig)
        else:
            # Non-cacheable action (MOVE, COPY, DELETE) was triggered — evict both
            # caches so a stale session-cache entry cannot shadow the cleared bucket
            # on the next evaluation of this path.
            ClassifierActionsManager.prevalidated_cache.pop(media_path, None)
            bucket.clear()
        return prevalidation_action

    @staticmethod
    def _should_persist_prevalidation_cache(
        action: Optional[ClassifierActionType],
        matched_prevalidation: Optional[Prevalidation],
        media_path: str,
    ) -> bool:
        if action is None or action.is_cache_type():
            return True
        if (
            action == ClassifierActionType.MOVE
            and matched_prevalidation is not None
            and matched_prevalidation.action_modifier
            and os.path.normpath(os.path.dirname(media_path))
                == os.path.normpath(matched_prevalidation.action_modifier)
        ):
            return True
        return False
    
    @staticmethod
    def get_profile_usage(profile_name: str) -> dict:
        """
        Get information about what's using a profile by checking the actual lists.
        
        Returns:
            Dictionary with keys:
            - 'prevalidations': List of prevalidation names using this profile
            - 'classifier_actions': List of classifier action names that have this profile as their last used profile
        """
        # Check prevalidations list directly
        prevalidation_names = [
            pv.name for pv in ClassifierActionsManager.prevalidations 
            if pv.profile_name == profile_name
        ]
        
        # Check classifier actions for last used profile matching profile name
        classifier_action_names = [
            ca.name for ca in ClassifierActionsManager.classifier_actions
            if ca._last_used_profile and ca._last_used_profile == profile_name
        ]
        
        return {
            'prevalidations': prevalidation_names,
            'classifier_actions': classifier_action_names
        }
    
    @staticmethod
    def can_remove_profile(profile_name: str) -> tuple[bool, List[str]]:
        """
        Check if a profile can be safely removed.
        
        Returns:
            Tuple of (can_remove, warnings)
            - can_remove: True if profile can be removed
            - warnings: List of warning messages
        """
        usage = ClassifierActionsManager.get_profile_usage(profile_name)
        warnings = []
        
        if usage['prevalidations']:
            warnings.append(f"prevalidations: {', '.join(usage['prevalidations'])}")
        
        if usage['classifier_actions']:
            warnings.append(f"classifier actions (last used profile): {', '.join(usage['classifier_actions'])}")
        
        return (len(warnings) == 0, warnings)
    
    @staticmethod
    def remove_profile(profile_name: str) -> bool:
        """
        Remove a profile after checking usage.
        
        Args:
            profile_name: Name of the profile to remove
            
        Returns:
            True if profile was removed, False if removal was prevented
        """
        # Find the profile
        profile = DirectoryProfile.get_profile_by_name(profile_name)
        if profile is None:
            logger.error(f"Profile {profile_name} not found")
            return False
        
        # Check if it can be removed
        can_remove, warnings = ClassifierActionsManager.can_remove_profile(profile_name)
        
        if warnings:
            logger.warning(f"Profile {profile_name} is used by: {', '.join(warnings)}")
            # Still allow removal, but warn the user
        
        # Remove from list
        if profile in DirectoryProfile.directory_profiles:
            DirectoryProfile.directory_profiles.remove(profile)
        
        logger.info(f"Removed profile: {profile_name}")
        return True
    
    @staticmethod
    def get_prevalidation_by_name(name: str) -> 'Prevalidation':
        """Get a prevalidation by name. Returns None if not found."""
        for prevalidation in ClassifierActionsManager.prevalidations:
            if name == prevalidation.name:
                return prevalidation
        return None

    @staticmethod
    def get_classifier_action_by_name(name: str) -> 'ClassifierAction':
        """Get a classifier action by name. Returns None if not found."""
        for classifier_action in ClassifierActionsManager.classifier_actions:
            if name == classifier_action.name:
                return classifier_action
        return None
    
    @staticmethod
    def load_prevalidations():
        """Load prevalidations from app cache (disk persistence).

        Does **not** evict prevalidation result caches: restoring the same session
        from disk is not a policy change. Call ``clear_prevalidation_result_cache``
        (or targeted eviction) after user edits that change outcomes.
        """
        # Load lookaheads first
        for lookahead_dict in list(app_info_cache.get_meta("recent_lookaheads", default_val=[])):
            lookahead = Lookahead.from_dict(lookahead_dict)
            # Check if lookahead already exists - if so, use existing one; otherwise add it
            existing = Lookahead.get_lookahead_by_name(lookahead.name)
            if existing is None:
                Lookahead.lookaheads.append(lookahead)
            # If it exists, we silently use the existing lookahead (no error)

        # Load profiles
        for profile_dict in list(app_info_cache.get_meta("recent_profiles", default_val=[])):
            profile = DirectoryProfile.from_dict(profile_dict)
            # Check if profile already exists - if so, use existing one; otherwise add it
            existing = DirectoryProfile.get_profile_by_name(profile.name)
            if existing is None:
                DirectoryProfile.add_profile(profile)
            # If it exists, we silently use the existing profile (no error)

        for prevalidation_dict in list(app_info_cache.get_meta("recent_prevalidations", default_val=[])):
            prevalidation: Prevalidation = Prevalidation.from_dict(prevalidation_dict)
            # Post-init methods (update_profile_instance, validate_dirs, ensure_prototype_loaded)
            # are now called lazily in _ensure_prevalidations_initialized() just before first use
            if prevalidation not in ClassifierActionsManager.prevalidations:
                ClassifierActionsManager.prevalidations.append(prevalidation)

                # Build directories_to_exclude from loaded prevalidations
                if (
                    prevalidation.is_move_action()
                    and prevalidation.action_modifier
                    and prevalidation.action_modifier not in ClassifierActionsManager.directories_to_exclude
                ):
                    ClassifierActionsManager.directories_to_exclude.append(prevalidation.action_modifier)

        set_signature_memo(None)

    @staticmethod
    def store_prevalidations():
        """Store prevalidations to cache."""
        # Store lookaheads
        lookahead_dicts = []
        for lookahead in Lookahead.lookaheads:
            lookahead_dicts.append(lookahead.to_dict())
        app_info_cache.set_meta("recent_lookaheads", lookahead_dicts)

        # Store profiles
        profile_dicts = []
        for profile in DirectoryProfile.directory_profiles:
            profile_dicts.append(profile.to_dict())
        app_info_cache.set_meta("recent_profiles", profile_dicts)

        prevalidation_dicts = []
        for prevalidation in ClassifierActionsManager.prevalidations:
            prevalidation_dicts.append(prevalidation.to_dict())
        app_info_cache.set_meta("recent_prevalidations", prevalidation_dicts)

    @staticmethod
    def load_classifier_actions():
        """Load classifier actions from app cache (disk persistence).

        Does **not** evict prevalidation result caches; see :meth:`load_prevalidations`.
        """
        for classifier_action_dict in list(app_info_cache.get_meta("recent_classifier_actions", default_val=[])):
            classifier_action: ClassifierAction = ClassifierAction.from_dict(classifier_action_dict)
            try:
                classifier_action.validate_dirs()
                classifier_action.validate()
                classifier_action.can_run = True
                classifier_action.initialization_error = None
                classifier_action._blocked_run_logged = False
            except Exception as e:
                classifier_action.can_run = False
                classifier_action.initialization_error = str(e)
                logger.error(
                    "Classifier action %r cannot run until fixed (kept in list): %s",
                    classifier_action.name,
                    e,
                )
            if classifier_action not in ClassifierActionsManager.classifier_actions:
                ClassifierActionsManager.classifier_actions.append(classifier_action)

        set_signature_memo(None)

    @staticmethod
    def store_classifier_actions():
        """Store classifier actions to cache."""
        classifier_action_dicts = []
        for classifier_action in ClassifierActionsManager.classifier_actions:
            classifier_action_dicts.append(classifier_action.to_dict())
        app_info_cache.set_meta("recent_classifier_actions", classifier_action_dicts)


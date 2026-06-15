from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from compare.compare_wrapper import CompareWrapper
from compare.compare_args import CompareArgs
from compare.base_compare import CompareCancelled
from utils.config import config
from utils.constants import CompareMode, Mode
from utils.logging_setup import get_logger

logger = get_logger("compare_manager")


class CombinationLogic(Enum):
    """How to combine results from multiple comparison modes"""
    AND = "AND"  # File must match ALL criteria
    OR = "OR"    # File must match ANY criterion
    WEIGHTED = "WEIGHTED"  # Weighted combination of scores


@dataclass
class CompareConfig:
    """Configuration for a single compare instance in composite search"""
    instance_id: str  # Unique identifier for this instance
    compare_mode: CompareMode
    weight: float = 1.0  # For weighted combination
    threshold: Optional[float] = None  # Override default threshold
    enabled: bool = True  # Can disable without removing
    search_text: Optional[str] = None  # Positive search text for this instance
    search_text_negative: Optional[str] = None  # Negative search text for this instance


from compare.compare_filters import CompareFilter, SizeFilter, ModelFilter  # noqa: F401 (re-exported for UI imports)


class CompareManager:
    """
    Manages one or more CompareWrapper instances to support both single
    and composite comparison modes. Acts as the interface between App
    and CompareWrapper.
    """
    
    def __init__(self, master=None, app_actions=None):
        self._master = master
        self._app_actions = app_actions
        
        # Active compare mode configurations (keyed by instance_id)
        self._mode_configs: Dict[str, CompareConfig] = {}
        
        # CompareWrapper instances (one per instance_id, not shared across same-mode instances)
        self._wrappers: Dict[str, CompareWrapper] = {}
        
        # Instance counter for generating unique IDs
        self._instance_counter: int = 0
        
        # Current primary mode (for backward compatibility)
        # Set default to CLIP_EMBEDDING (matches CompareArgs default and config default)
        self._primary_mode: Optional[CompareMode] = None
        # Initialize default mode - use config.compare_mode if available, otherwise CLIP_EMBEDDING
        default_mode = getattr(config, 'compare_mode', CompareMode.CLIP_EMBEDDING)
        if isinstance(default_mode, CompareMode):
            self.set_primary_mode(default_mode)
        else:
            # If config has a string, convert it
            try:
                self.set_primary_mode(CompareMode.get(default_mode))
            except (ValueError, AttributeError):
                # Fallback to CLIP_EMBEDDING if conversion fails
                self.set_primary_mode(CompareMode.CLIP_EMBEDDING)
        
        # Combination logic for composite mode
        self._combination_logic: CombinationLogic = CombinationLogic.AND
        
        # Pre-filter applied to file list before any scoring
        self._data_filter: Optional[CompareFilter] = None
        
        # Compare settings (migrated from sidebar)
        self._threshold: Optional[float] = None
        self._counter_limit: Optional[int] = None
        self._overwrite: bool = False
        self._store_checkpoints: bool = config.store_checkpoints
        self._use_matrix_comparison: bool = True
        
        # Results from last run
        self._last_results: Optional[Dict[CompareMode, Dict[str, float]]] = None
        self._combined_results: Optional[Dict[str, float]] = None
        
        # State management (delegated to primary wrapper for single-mode)
        self._is_composite_mode: bool = False

        # Restore the most recent compare configuration from persistent history.
        # apply_snapshot restores instances, logic, filter, and run_settings; safe
        # before set_app_actions wires in the real app_actions.
        try:
            from compare.compare_history import CompareHistory
            recent = CompareHistory.load_recent()
            if recent:
                self.apply_snapshot(recent[0])
        except Exception as _e:
            logger.debug(f"Could not restore compare history on init: {_e}")

    def set_app_actions(self, master, app_actions):
        """Set master and app_actions after construction.

        Also back-fills any CompareWrapper instances that were created
        during __init__ (via set_primary_mode → _ensure_wrapper).
        """
        self._master = master
        self._app_actions = app_actions
        for wrapper in self._wrappers.values():
            wrapper._master = master
            wrapper._app_actions = app_actions

    # ========== Mode Management ==========
    
    def _generate_instance_id(self, compare_mode: CompareMode) -> str:
        """Generate a unique instance ID for a compare mode."""
        self._instance_counter += 1
        return f"{compare_mode.name}_{self._instance_counter}"

    @property
    def _primary_instance_id(self) -> Optional[str]:
        """Return the instance_id of the first enabled instance for the primary mode."""
        if self._primary_mode is None:
            return None
        for instance_id, cfg in self._mode_configs.items():
            if cfg.compare_mode == self._primary_mode and cfg.enabled:
                return instance_id
        return None

    def _primary_wrapper(self) -> Optional[CompareWrapper]:
        """Return the CompareWrapper for the primary instance, or None."""
        iid = self._primary_instance_id
        return self._wrappers.get(iid) if iid else None

    def set_primary_mode(self, compare_mode: CompareMode):
        """
        Set the primary comparison mode. For single-mode operations,
        this is the only active mode. For composite mode, this is the
        mode used for result presentation and navigation.
        """
        self._primary_mode = compare_mode
        # Find or create an instance for this mode
        existing = next((iid for iid, cfg in self._mode_configs.items()
                         if cfg.compare_mode == compare_mode), None)
        if existing is None:
            instance_id = self._generate_instance_id(compare_mode)
            self._mode_configs[instance_id] = CompareConfig(
                instance_id=instance_id,
                compare_mode=compare_mode,
                enabled=True
            )
        else:
            instance_id = existing
        self._ensure_wrapper(instance_id, compare_mode)
        self._is_composite_mode = len(self._mode_configs) > 1
        logger.info(f"Primary compare mode set to: {compare_mode.name} (composite mode: {self._is_composite_mode})")

    def get_primary_mode_text(self) -> Optional[str]:
        """Get the primary comparison mode."""
        if self._primary_mode:
            return self._primary_mode.get_text()
        return None
    
    def get_primary_mode_name(self) -> Optional[str]:
        """Get the primary comparison mode name (locale-independent)."""
        if self._primary_mode:
            return self._primary_mode.name
        return None

    def add_mode_instance(self, compare_mode: CompareMode, weight: float = 1.0, 
                          threshold: Optional[float] = None, 
                          search_text: Optional[str] = None,
                          search_text_negative: Optional[str] = None,
                          instance_id: Optional[str] = None) -> str:
        """
        Add a comparison mode instance for composite search.
        Returns the instance_id of the created instance.
        If primary mode not set, this becomes the primary mode.
        """
        if self._primary_mode is None:
            self._primary_mode = compare_mode
        
        if instance_id is None:
            instance_id = self._generate_instance_id(compare_mode)
        
        self._mode_configs[instance_id] = CompareConfig(
            instance_id=instance_id,
            compare_mode=compare_mode,
            weight=weight,
            threshold=threshold,
            search_text=search_text,
            search_text_negative=search_text_negative,
            enabled=True
        )
        self._ensure_wrapper(instance_id, compare_mode)
        self._is_composite_mode = len(self._mode_configs) > 1
        logger.info(f"Added compare mode instance: {instance_id} ({compare_mode.name}, weight={weight}, threshold={threshold}, composite={self._is_composite_mode})")
        return instance_id
    
    def add_mode(self, compare_mode: CompareMode, weight: float = 1.0, 
                  threshold: Optional[float] = None) -> None:
        """
        Add a comparison mode for composite search (backward compatibility wrapper).
        """
        self.add_mode_instance(compare_mode, weight, threshold)
    
    def remove_mode_instance(self, instance_id: str) -> None:
        """Remove a comparison mode instance from composite search."""
        if instance_id not in self._mode_configs:
            return
        
        config = self._mode_configs[instance_id]
        compare_mode = config.compare_mode
        del self._mode_configs[instance_id]
        self._wrappers.pop(instance_id, None)  # release wrapper for this instance
        
        # If this was the only instance of the primary mode, update primary
        if compare_mode == self._primary_mode:
            remaining_instances = [c for c in self._mode_configs.values() if c.compare_mode == compare_mode]
            if not remaining_instances:
                # Set new primary from remaining modes
                if self._mode_configs:
                    self._primary_mode = next(iter(self._mode_configs.values())).compare_mode
                else:
                    self._primary_mode = None
        
        self._is_composite_mode = len(self._mode_configs) > 1
        logger.info(f"Removed compare mode instance: {instance_id} (composite={self._is_composite_mode}, primary={self._primary_mode.name if self._primary_mode else None})")
    
    def remove_mode(self, compare_mode: CompareMode) -> None:
        """Remove all instances of a comparison mode (backward compatibility wrapper)."""
        instances_to_remove = [instance_id for instance_id, config in self._mode_configs.items() 
                              if config.compare_mode == compare_mode]
        for instance_id in instances_to_remove:
            self.remove_mode_instance(instance_id)
    
    def set_combination_logic(self, logic: CombinationLogic):
        """Set how to combine results from multiple modes."""
        self._combination_logic = logic
        logger.info(f"Combination logic set to: {logic.value}")
    
    def set_mode_weight(self, instance_id: str, weight: float):
        """Set weight for a comparison mode instance (for weighted combination)."""
        if instance_id in self._mode_configs:
            self._mode_configs[instance_id].weight = weight
            logger.info(f"Set weight for instance {instance_id} to {weight}")
        else:
            logger.warning(f"Cannot set weight for instance {instance_id} - instance not found")
    
    def get_combination_logic(self) -> CombinationLogic:
        """Get current combination logic."""
        return self._combination_logic
    
    def is_composite_mode(self) -> bool:
        """Check if currently in composite mode."""
        return self._is_composite_mode
    
    def get_active_modes(self) -> List[CompareMode]:
        """Get list of unique active compare modes."""
        active_modes = set()
        for config in self._mode_configs.values():
            if config.enabled:
                active_modes.add(config.compare_mode)
        return list(active_modes)
    
    def get_mode_instances(self) -> List[CompareConfig]:
        """Get list of all mode instances."""
        return list(self._mode_configs.values())
    
    def get_mode_instances_by_mode(self, compare_mode: CompareMode) -> List[CompareConfig]:
        """Get all instances of a specific compare mode."""
        return [config for config in self._mode_configs.values() 
                if config.compare_mode == compare_mode and config.enabled]
    
    def _ensure_wrapper(self, instance_id: str, compare_mode: CompareMode) -> CompareWrapper:
        """Get or create a CompareWrapper for a specific instance."""
        if instance_id not in self._wrappers:
            self._wrappers[instance_id] = CompareWrapper(
                self._master, compare_mode, self._app_actions
            )
        return self._wrappers[instance_id]
    
    # ========== Filtering ==========

    def set_data_filter(self, f: Optional[CompareFilter]) -> None:
        self._data_filter = f
        if f and f.is_active():
            logger.info(f"Data filter set: {type(f).__name__}")
        else:
            logger.info("Data filter cleared")

    def get_data_filter(self) -> Optional[CompareFilter]:
        return self._data_filter
    
    # ========== Compare Settings ==========
    
    def set_threshold(self, threshold: float):
        """Set comparison threshold."""
        self._threshold = threshold
        logger.info(f"Compare threshold set to: {threshold}")
    
    def get_threshold(self) -> Optional[float]:
        """Get comparison threshold."""
        return self._threshold
    
    def set_counter_limit(self, counter_limit: Optional[int]):
        """Set counter limit option."""
        self._counter_limit = counter_limit
        logger.info(f"Counter limit set to: {counter_limit if counter_limit is not None else 'None (unlimited)'}")
    
    def get_counter_limit(self) -> Optional[int]:
        """Get counter limit option."""
        return self._counter_limit
    
    def apply_settings_to_args(self, args: CompareArgs) -> None:
        """
        Apply all compare settings from this manager to a CompareArgs object.
        Handles fallbacks to config defaults when settings are not explicitly set.
        """
        # Apply threshold with fallback to config defaults
        threshold = self.get_threshold()
        if threshold is not None:
            args.threshold = threshold
        else:
            # Fallback to config defaults based on primary mode
            primary_mode = self.compare_mode
            if primary_mode == CompareMode.COLOR_MATCHING:
                args.threshold = config.color_diff_threshold
            else:
                args.threshold = config.embedding_similarity_threshold
        
        # Apply counter_limit with fallback to config default
        counter_limit = self.get_counter_limit()
        if counter_limit is not None:
            args.counter_limit = counter_limit
        else:
            args.counter_limit = config.file_counter_limit
        
        # Apply boolean settings (these always have values, no fallback needed)
        args.overwrite = self.get_overwrite()
        args.store_checkpoints = self.get_store_checkpoints()
        args.use_matrix_comparison = self.get_use_matrix_comparison()
        
        # Log applied settings
        self._log_settings(args)
    
    def _log_settings(self, args: CompareArgs) -> None:
        """
        Log all comparison settings in a format similar to base_compare_embedding.print_settings().
        """
        logger.info("|--------------------------------------------------------------------|")
        logger.info(" COMPARE MANAGER SETTINGS:")
        logger.info(f" primary compare mode: {self._primary_mode.name if self._primary_mode else 'None'}")
        logger.info(f" composite mode: {self._is_composite_mode}")
        
        if self._is_composite_mode:
            active_modes = self.get_active_modes()
            logger.info(f" active modes: {[mode.name for mode in active_modes]}")
            logger.info(f" combination logic: {self._combination_logic.value}")
            for instance_id, config in self._mode_configs.items():
                if config.enabled:
                    threshold_str = f"{config.threshold}" if config.threshold else "default"
                    weight_str = f", weight={config.weight}" if self._combination_logic == CombinationLogic.WEIGHTED else ""
                    search_text_str = f", search_text='{config.search_text}'" if config.search_text else ""
                    search_neg_str = f", search_text_negative='{config.search_text_negative}'" if config.search_text_negative else ""
                    logger.info(f"   {instance_id} ({config.compare_mode.name}): threshold={threshold_str}{weight_str}{search_text_str}{search_neg_str}")
        
        logger.info(f" comparison files base directory: {args.base_dir}")
        # Threshold display depends on mode
        if self._primary_mode == CompareMode.COLOR_MATCHING:
            logger.info(f" color diff threshold: {args.threshold}")
        else:
            logger.info(f" embedding similarity threshold: {args.threshold}")
        
        logger.info(f" max file process limit: {args.counter_limit}")
        logger.info(f" recursive: {args.recursive}")
        logger.info(f" file glob pattern: {args.file_filter}")
        logger.info(f" include videos: {args.include_videos}")
        logger.info(f" overwrite media data: {args.overwrite}")
        logger.info(f" store checkpoints: {args.store_checkpoints}")
        logger.info(f" use matrix comparison: {args.use_matrix_comparison}")
        
        # Filter settings
        if self._data_filter and self._data_filter.is_active():
            logger.info(f" data filter: {type(self._data_filter).__name__}")
        else:
            logger.info(" data filter: None")
        
        logger.info("|--------------------------------------------------------------------|\n")
    
    def set_overwrite(self, overwrite: bool):
        """Set overwrite cache option."""
        self._overwrite = overwrite
        logger.info(f"Overwrite cache set to: {overwrite}")
    
    def get_overwrite(self) -> bool:
        """Get overwrite cache option."""
        return self._overwrite
    
    def set_store_checkpoints(self, store_checkpoints: bool):
        """Set store checkpoints option."""
        self._store_checkpoints = store_checkpoints
        logger.info(f"Store checkpoints set to: {store_checkpoints}")
    
    def get_store_checkpoints(self) -> bool:
        """Get store checkpoints option."""
        return self._store_checkpoints

    def set_use_matrix_comparison(self, use_matrix: bool) -> None:
        """Use chunked matrix compare for embedding group runs (vs roll-index iterative)."""
        self._use_matrix_comparison = use_matrix
        logger.info(f"Use matrix comparison set to: {use_matrix}")

    def get_use_matrix_comparison(self) -> bool:
        """Whether embedding group compare uses the matrix path."""
        return self._use_matrix_comparison

    def _sync_settings_from_last_args(self) -> None:
        """Align manager booleans with the last compare's args (for settings autoload)."""
        if not self.has_compare():
            return
        last = self.get_args()
        self.set_overwrite(last.overwrite)
        self.set_store_checkpoints(last.store_checkpoints)
        self.set_use_matrix_comparison(last.use_matrix_comparison)
    
    def toggle_search_only_return_closest(self):
        """Toggle search only return closest option (for backward compatibility)."""
        config.search_only_return_closest = not config.search_only_return_closest
    
    # ========== Backward Compatibility Properties ==========
    
    @property
    def compare_mode(self) -> Optional[CompareMode]:
        """Get primary compare mode (for backward compatibility)."""
        return self._primary_mode
    
    @compare_mode.setter
    def compare_mode(self, mode: CompareMode):
        """
        Set primary compare mode (for backward compatibility).
        Note: This clears other modes for single-mode operation.
        TODO: In the future, we should support persisting multiple modes per directory.
              When implementing this, we'll need to:
              - Store all active modes in cache (not just primary)
              - Load all modes when restoring (use set_primary_mode() + add_mode() instead of set_compare_mode())
              - Consider adding a parameter or separate method to set primary without clearing others
        """
        self.set_primary_mode(mode)
        # Clear other modes for single-mode operation (backward compatibility)
        if len(self._mode_configs) > 1:
            # Find all instance IDs for this mode
            instances_to_keep = {
                instance_id: config 
                for instance_id, config in self._mode_configs.items() 
                if config.compare_mode == mode
            }
            # If we have instances of this mode, keep only those; otherwise keep the first one created
            if instances_to_keep:
                self._mode_configs = instances_to_keep
            else:
                # Fallback: keep only the first instance (shouldn't happen, but be safe)
                first_instance_id = next(iter(self._mode_configs.keys()))
                self._mode_configs = {first_instance_id: self._mode_configs[first_instance_id]}
            self._is_composite_mode = False
    
    def set_compare_mode(self, mode: CompareMode):
        """
        Set the compare mode (alias for compare_mode property setter).
        This method provides an explicit way to set the compare mode.
        """
        self.compare_mode = mode
    
    @property
    def files_matched(self) -> List[str]:
        """Get matched files from primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().files_matched
        return []
    
    @property
    def file_groups(self) -> Dict:
        """Get file groups from primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().file_groups
        return {}
    
    @property
    def files_grouped(self) -> Dict:
        """Get grouped files from primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().files_grouped
        return {}
    
    @property
    def match_index(self) -> int:
        """Get match index from primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().match_index
        return 0
    
    @match_index.setter
    def match_index(self, value: int):
        """Set match index on primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            self._primary_wrapper().match_index = value
    
    @property
    def current_group_index(self) -> int:
        """Get current group index from primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().current_group_index
        return 0
    
    @current_group_index.setter
    def current_group_index(self, value: int):
        """Set current group index on primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            self._primary_wrapper().current_group_index = value
    
    @property
    def search_media_path(self) -> Optional[str]:
        """Get search file path from primary wrapper."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().search_media_path
        return None
    
    @search_media_path.setter
    def search_media_path(self, value: Optional[str]):
        """Set search file path on primary wrapper."""
        if self._primary_wrapper() is not None:
            self._primary_wrapper().search_media_path = value
    
    @property
    def hidden_media(self) -> List[str]:
        """Get hidden media from primary wrapper."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().hidden_media
        return []
    
    @property
    def group_indexes(self) -> List[int]:
        """Get group indexes from primary wrapper."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().group_indexes
        return []
    
    @property
    def max_group_index(self) -> int:
        """Get max group index from primary wrapper."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().max_group_index
        return 0
    
    @property
    def has_media_matches(self) -> bool:
        """Get has_media_matches from primary wrapper."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().has_media_matches
        return False
    
    # ========== Delegation Methods (for backward compatibility) ==========
    
    def has_compare(self) -> bool:
        """Check if primary wrapper has a compare instance."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().has_compare()
        return False
    
    def cancel(self):
        """Cancel all running compare operations."""
        for wrapper in self._wrappers.values():
            wrapper.cancel()
    
    def clear_compare(self):
        """Clear compare instances from all wrappers."""
        for wrapper in self._wrappers.values():
            wrapper.clear_compare()
    
    def get_args(self) -> CompareArgs:
        """Get args from primary wrapper."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().get_args()
        return CompareArgs()
    
    def validate_compare_mode(self, required_compare_mode, error_text):
        """Validate compare mode (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            self._primary_wrapper().validate_compare_mode(
                required_compare_mode, error_text
            )
    
    def current_match(self) -> Optional[str]:
        """Get current match from primary wrapper."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().current_match()
        return None
    
    def show_prev_media(self, show_alert=True) -> bool:
        """Show previous media (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().show_prev_media(show_alert)
        return False
    
    def show_next_media(self, show_alert=True) -> bool:
        """Show next media (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().show_next_media(show_alert)
        return False
    
    def skip_media(self, media_path: str) -> bool:
        """Check if media should be skipped (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().skip_media(media_path)
        return False

    def show_prev_group(self, event=None, file_browser=None):
        """Show previous group (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().show_prev_group(event, file_browser)
    
    def show_next_group(self, event=None, file_browser=None):
        """Show next group (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().show_next_group(event, file_browser)

    def random_purge_groups(self, event=None):
        """Delete all but one random file per group (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().random_purge_groups()

    def set_current_group(self, start_match_index=0):
        """Set current group (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().set_current_group(start_match_index)

    def get_grouped_filepaths(self, app_mode) -> list:
        """Return all comparison files ordered group-by-group for masonry display."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().get_grouped_filepaths(app_mode)
        return []

    def get_file_group_for_filepath(self, filepath: str, app_mode) -> Optional[tuple]:
        """Return (group_display_idx, file_idx_within_group) for a filepath, or None."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().get_file_group_for_filepath(filepath, app_mode)
        return None

    def seek_to_file(self, filepath: str) -> None:
        """Seek the match cursor to filepath in the existing files_matched list.

        Used by SEARCH mode tile activation: files_matched is already sorted
        correctly by run_search(), so we must not rebuild it via set_current_group.
        """
        wrapper = self._primary_wrapper()
        if wrapper is None:
            return
        try:
            wrapper.match_index = wrapper.files_matched.index(filepath)
        except ValueError:
            pass
    
    def page_down(self, half_length=False) -> Optional[str]:
        """Page down (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().page_down(half_length)
        return None
    
    def page_up(self, half_length=False) -> Optional[str]:
        """Page up (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().page_up(half_length)
        return None
    
    def select_series(self, start_file: str, end_file: str) -> List[str]:
        """Select series (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().select_series(start_file, end_file)
        return []
    
    def find_file_after_comparison(self, app_mode, search_text="", exact_match=False):
        """Find file after comparison (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().find_file_after_comparison(
                app_mode, search_text, exact_match
            )
        return None, None
    
    def _update_groups_for_removed_file(self, app_mode, group_index, match_index, 
                                       set_group=True, show_next_media=None):
        """Update groups for removed file (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper()._update_groups_for_removed_file(
                app_mode, group_index, match_index, set_group, show_next_media
            )
    
    def update_compare_for_readded_file(self, readded_file: str):
        """Update compare for readded file (delegated to all wrappers)."""
        for wrapper in self._wrappers.values():
            wrapper.update_compare_for_readded_file(readded_file)
    
    def _get_file_group_map(self, app_mode):
        """Get file group map (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper()._get_file_group_map(app_mode)
        return {}
    
    def find_next_unrelated_media(self, file_browser, forward=True):
        """Find next unrelated file (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().find_next_unrelated_media(
                file_browser, forward
            )
    
    def _get_prev_media(self):
        """Get previous file (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper()._get_prev_media()
        return None
    
    def _get_next_media(self):
        """Get next file (delegated to primary wrapper)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper()._get_next_media()
        return None
    
    def compare(self):
        """Get compare instance from primary wrapper (for backward compatibility)."""
        if self._primary_wrapper() is not None:
            return self._primary_wrapper().compare()
        raise Exception("No compare object created")
    
    # ========== Main Execution ==========
    
    def run(self, args: CompareArgs = CompareArgs()):
        """
        Execute comparison. For single-mode, delegates to wrapper.
        For composite mode, runs multiple comparisons and combines results.
        """
        if not self._primary_mode:
            raise ValueError("No compare mode set")

        # Record this run in the persistent recent-history list.
        try:
            from compare.compare_history import CompareHistory
            CompareHistory.record(self.snapshot(args))
        except Exception as _e:
            logger.debug(f"Failed to record compare history: {_e}")

        # Attach the pre-filter so BaseCompare.get_files() can apply it
        args.data_filter = self._data_filter
        
        # Log comparison mode configuration
        if self._is_composite_mode:
            active_modes = self.get_active_modes()
            logger.info(f"Running composite comparison with {len(self._mode_configs)} instances across {len(active_modes)} modes: {[mode.name for mode in active_modes]}")
            logger.info(f"Combination logic: {self._combination_logic.value}")
            for instance_id, config in self._mode_configs.items():
                if config.enabled:
                    threshold_str = f"threshold={config.threshold}" if config.threshold else "threshold=default"
                    weight_str = f", weight={config.weight}" if self._combination_logic == CombinationLogic.WEIGHTED else ""
                    search_text_str = f", search_text='{config.search_text}'" if config.search_text else ""
                    search_neg_str = f", search_text_negative='{config.search_text_negative}'" if config.search_text_negative else ""
                    logger.info(f"  Instance {instance_id} ({config.compare_mode.name}): {threshold_str}{weight_str}{search_text_str}{search_neg_str}")
        else:
            logger.info(f"Running single-mode comparison: {self._primary_mode.name}")
        
        # Single-mode operation (backward compatible)
        if not self._is_composite_mode:
            primary_iid = self._primary_instance_id
            if primary_iid is None:
                raise ValueError("No enabled instance for primary mode")
            wrapper = self._ensure_wrapper(primary_iid, self._primary_mode)
            wrapper.run(args)
            self._sync_settings_from_last_args()
            return
        
        # Composite mode operation
        self._run_composite(args)
        self._sync_settings_from_last_args()

    @staticmethod
    def _normalize_score(mode: CompareMode, score: float) -> float:
        """Convert any mode's raw score to a [0, 1] similarity scale."""
        if mode == CompareMode.COLOR_MATCHING:
            from compare.compare_colors import CompareColors
            return max(0.0, min(1.0, 1.0 - score / CompareColors.THRESHHOLD_GROUP_CUTOFF))
        return score

    def _run_composite(self, args: CompareArgs):
        """
        Run composite comparison across multiple mode instances and combine results.

        Instances run sequentially on the worker thread (not in parallel) because
        progress/UI signals are main-thread-bound.
        """
        if not self._mode_configs:
            raise ValueError("No compare modes configured for composite search")
        
        self._app_actions._set_label_state(
            f"Running composite comparison with {len(self._mode_configs)} instances..."
        )
        
        # Run each enabled instance
        instance_results: Dict[str, Dict[str, float]] = {}  # instance_id -> {file_path: score}
        enabled_configs = [(iid, cfg) for iid, cfg in self._mode_configs.items() if cfg.enabled]
        total_instances = len(enabled_configs)

        # Enforce exclusive run-mode paths: all instances must run GROUP or all must run SEARCH.
        # An instance is SEARCH if global args carry search criteria, or the instance itself has
        # search_text / search_text_negative configured.
        def _is_search_instance(cfg: CompareConfig) -> bool:
            return not args.not_searching() or bool(cfg.search_text or cfg.search_text_negative)

        run_mode_flags = {_is_search_instance(cfg) for _unused, cfg in enabled_configs}
        if len(run_mode_flags) > 1:
            raise ValueError(
                "All composite instances must run in the same mode. "
                "Mix of GROUP and SEARCH instances is not supported — "
                "ensure either all instances have search criteria, or none do."
            )

        skipped_instance_ids: set = set()

        for run_index, (instance_id, config) in enumerate(enabled_configs, start=1):
            self._app_actions._set_label_state(
                f"Running composite ({run_index}/{total_instances}): {config.compare_mode.name}…"
            )

            wrapper = self._ensure_wrapper(instance_id, config.compare_mode)

            # Share already-loaded data from a prior same-mode instance to avoid
            # redundant disk I/O and GPU inference.
            for prior_iid, prior_cfg in enabled_configs[:run_index - 1]:
                if (prior_cfg.compare_mode == config.compare_mode
                        and prior_iid in self._wrappers
                        and self._wrappers[prior_iid]._compare is not None):
                    wrapper.share_data_from(self._wrappers[prior_iid])
                    logger.debug(f"Instance {instance_id} sharing loaded data from {prior_iid}")
                    break

            # Create instance-specific args
            instance_args = args.clone()
            instance_args.compare_mode = config.compare_mode
            if config.threshold is not None:
                instance_args.threshold = config.threshold

            # Apply instance-specific search text
            if config.search_text:
                instance_args.search_text = config.search_text
            if config.search_text_negative:
                instance_args.search_text_negative = config.search_text_negative

            # Non-embedding modes do not support semantic text search.
            # Strip any text that was inherited from global args so these modes
            # don't attempt (and fail) a text-based search. Instance-specific
            # text (explicitly configured per-instance) is left untouched above.
            if not config.compare_mode.is_embedding():
                if not config.search_text:
                    instance_args.search_text = None
                if not config.search_text_negative:
                    instance_args.search_text_negative = None

            # If the overall run is a search and this instance has no remaining
            # search criteria (no image path, no text), skip it rather than
            # silently switching it to a full GROUP compare.
            if not args.not_searching() and instance_args.not_searching():
                logger.warning(
                    f"Skipping instance {instance_id} ({config.compare_mode.name}) in search run: "
                    f"mode does not support text search and no image search path is set."
                )
                instance_results[instance_id] = {}
                skipped_instance_ids.add(instance_id)
                continue

            # Run comparison
            try:
                wrapper.run(instance_args)

                # Extract results into a flat {file_path: score} dict.
                #
                # SEARCH mode: wrapper.files_grouped = {0: {file_path: score}}
                # GROUP mode:  wrapper.file_groups   = {group_idx: {file_path: score}}
                #              wrapper.files_grouped = {file_idx: (group_idx, score)}
                #              — we must flatten file_groups, not files_grouped.
                if instance_args.not_searching():
                    flat: Dict[str, float] = {}
                    for group in wrapper.file_groups.values():
                        flat.update(group)
                    instance_results[instance_id] = flat
                else:
                    instance_results[instance_id] = wrapper.files_grouped.get(0, {})

            except CompareCancelled:
                instance_results[instance_id] = {}
                raise  # propagate — abort the whole composite run
            except Exception as e:
                logger.error(f"Error running instance {instance_id} ({config.compare_mode.name}): {e}")
                instance_results[instance_id] = {}

        # If every enabled instance was skipped because none support text search,
        # raise a user-visible error instead of silently returning empty results.
        if not args.not_searching() and skipped_instance_ids == {iid for iid, _unused in enabled_configs}:
            raise ValueError(
                "Text search is not supported by any mode in this composite bundle. "
                "Add an embedding mode (e.g. CLIP, SigLIP) to use text search, "
                "or remove the search text."
            )

        # Store individual results (convert to mode-based for backward compatibility)
        self._last_results = {}
        for instance_id, results in instance_results.items():
            config = self._mode_configs[instance_id]
            if config.compare_mode not in self._last_results:
                self._last_results[config.compare_mode] = {}
            # Merge results from multiple instances of same mode (use max score)
            for file_path, score in results.items():
                if file_path not in self._last_results[config.compare_mode]:
                    self._last_results[config.compare_mode][file_path] = score
                else:
                    self._last_results[config.compare_mode][file_path] = max(
                        self._last_results[config.compare_mode][file_path], score
                    )
        
        # Log individual instance results
        for instance_id, results in instance_results.items():
            config = self._mode_configs[instance_id]
            logger.info(f"Instance {instance_id} ({config.compare_mode.name}) found {len(results)} matches")
        
        # Normalize all instance results to [0, 1] similarity scale
        instance_results_normalized: Dict[str, Dict[str, float]] = {}
        for instance_id, results in instance_results.items():
            cfg = self._mode_configs[instance_id]
            instance_results_normalized[instance_id] = {
                fp: self._normalize_score(cfg.compare_mode, score)
                for fp, score in results.items()
            }

        # For AND/OR: aggregate per mode (merge same-mode instances via max score)
        mode_results_for_combine: Dict[CompareMode, Dict[str, float]] = {}
        for instance_id, results in instance_results_normalized.items():
            cfg = self._mode_configs[instance_id]
            if cfg.compare_mode not in mode_results_for_combine:
                mode_results_for_combine[cfg.compare_mode] = {}
            for file_path, normalized in results.items():
                if file_path not in mode_results_for_combine[cfg.compare_mode]:
                    mode_results_for_combine[cfg.compare_mode][file_path] = normalized
                else:
                    mode_results_for_combine[cfg.compare_mode][file_path] = max(
                        mode_results_for_combine[cfg.compare_mode][file_path], normalized
                    )

        self._combined_results = self._combine_results(mode_results_for_combine, instance_results_normalized)
        logger.info(f"Combined results using {self._combination_logic.value} logic: {len(self._combined_results)} files")

        # is_group_mode is consistent across all instances (enforced by pre-flight validation)
        is_group_mode = args.not_searching()

        # Update primary wrapper with combined results
        self._apply_combined_results_to_primary(is_group_mode=is_group_mode)
    
    def _combine_results(self, mode_results: Dict[CompareMode, Dict[str, float]],
                         instance_results: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
        """
        Combine results from multiple comparison modes.
        Returns dict mapping file_path -> combined_score
        """
        if self._combination_logic == CombinationLogic.WEIGHTED:
            # WEIGHTED operates at instance level to preserve per-instance weights
            return self._combine_weighted(instance_results or {})

        if not mode_results:
            return {}

        if self._combination_logic == CombinationLogic.AND:
            return self._combine_and(mode_results)
        else:  # OR
            return self._combine_or(mode_results)
    
    def _combine_and(self, mode_results: Dict[CompareMode, Dict[str, float]]) -> Dict[str, float]:
        """AND logic: file must appear in ALL mode results."""
        if not mode_results:
            return {}

        # Start with files from first mode
        result_sets = [set(mode_results[mode].keys()) for mode in mode_results]
        common_files = set.intersection(*result_sets) if result_sets else set()

        # For common files, use geometric mean — preserves "must pass all modes" semantics
        # without over-penalising strong matches the way min() does
        combined = {}
        for file_path in common_files:
            scores = [mode_results[mode][file_path] for mode in mode_results if file_path in mode_results[mode]]
            product = 1.0
            for s in scores:
                product *= s
            combined[file_path] = product ** (1.0 / len(scores))

        return combined
    
    def _combine_or(self, mode_results: Dict[CompareMode, Dict[str, float]]) -> Dict[str, float]:
        """OR logic: file must appear in ANY mode result."""
        all_files = set()
        for mode_results_dict in mode_results.values():
            all_files.update(mode_results_dict.keys())
        
        # For files in multiple modes, use maximum score (most optimistic)
        combined = {}
        for file_path in all_files:
            scores = [mode_results[mode][file_path] 
                     for mode in mode_results 
                     if file_path in mode_results[mode]]
            combined[file_path] = max(scores)  # Most optimistic
        
        return combined
    
    def _combine_weighted(self, instance_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Weighted combination: weighted average of per-instance scores.

        Each instance contributes score * weight independently. A file absent from
        an instance contributes 0 * weight (that instance's weight is still in the
        denominator), so cross-instance comparisons remain meaningful.
        """
        # Collect all files that appear in any instance
        all_files: set = set()
        for results in instance_results.values():
            all_files.update(results.keys())

        combined = {}
        for file_path in all_files:
            weighted_sum = 0.0
            weight_sum = 0.0
            for instance_id, results in instance_results.items():
                cfg = self._mode_configs.get(instance_id)
                if cfg is None or not cfg.enabled:
                    continue
                score = results.get(file_path, 0.0)
                weighted_sum += score * cfg.weight
                weight_sum += cfg.weight
            combined[file_path] = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        return combined
    
    def _apply_combined_results_to_primary(self, is_group_mode: bool = False):
        """Apply combined results to the primary wrapper for UI navigation.

        GROUP mode:  rebuilds file_groups by filtering the primary instance's existing
                     groups down to files that survived the composite combine, then
                     delegates to set_current_group() so the label and navigation
                     behave identically to a single-compare group run.
        SEARCH mode: stores a flat group-0 result and drives create_media() directly.
        """
        if not self._primary_mode or not self._combined_results:
            return

        primary_iid = self._primary_instance_id
        if not primary_iid:
            return
        wrapper = self._ensure_wrapper(primary_iid, self._primary_mode)

        if is_group_mode:
            # Filter the primary instance's groups to only files in combined results.
            combined_file_set = set(self._combined_results.keys())
            original_groups = dict(wrapper.file_groups)  # snapshot before overwriting

            new_idx = 0
            filtered_groups: Dict[int, Dict[str, float]] = {}
            for _unused, group_files in sorted(original_groups.items()):
                kept = {fp: s for fp, s in group_files.items() if fp in combined_file_set}
                if len(kept) > 1:  # discard groups left with only one member
                    filtered_groups[new_idx] = kept
                    new_idx += 1

            if not filtered_groups:
                wrapper.has_media_matches = False
                wrapper.file_groups = {}
                wrapper.files_grouped = {}
                wrapper.group_indexes = []
                self._app_actions._set_label_state("No matching groups found")
                self._app_actions.alert(
                    "No Match Found",
                    "No groups survived the composite filter criteria."
                )
                return

            wrapper.file_groups = filtered_groups
            wrapper.files_grouped = deepcopy(filtered_groups)
            wrapper.group_indexes = list(range(len(filtered_groups)))
            wrapper.max_group_index = max(filtered_groups.keys())
            wrapper.current_group_index = 0
            wrapper.has_media_matches = True
            wrapper.label_suffix = " (composite)"

            logger.info(
                f"Composite GROUP result: {len(filtered_groups)} group(s) "
                f"({sum(len(g) for g in filtered_groups.values())} files)"
            )
            self._app_actions._add_buttons_for_mode()
            # set_current_group sets files_matched, calls _set_label_state(group_number, size)
            # and create_media — identical behaviour to a single-compare group run.
            wrapper.set_current_group()
            self._app_actions.refresh_masonry()

        else:
            # SEARCH mode — flat results in a single pseudo-group
            wrapper.files_grouped = {0: self._combined_results}
            wrapper.file_groups = deepcopy(wrapper.files_grouped)

            reverse = self._primary_mode.is_embedding()
            wrapper.files_matched = []
            for f in sorted(self._combined_results.keys(),
                           key=lambda f: self._combined_results[f],
                           reverse=reverse):
                wrapper.files_matched.append(f)

            wrapper.group_indexes = [0]
            wrapper.current_group_index = 0
            wrapper.max_group_index = 0
            wrapper.match_index = 0
            wrapper.has_media_matches = len(wrapper.files_matched) > 0

            if wrapper.has_media_matches:
                self._app_actions._set_label_state(
                    f"{len(wrapper.files_matched)} matches found (composite search)"
                )
                self._app_actions._add_buttons_for_mode()
                self._app_actions.create_media(wrapper.files_matched[0])
            else:
                self._app_actions._set_label_state("No matches found")
                self._app_actions.alert(
                    "No Match Found",
                    "None of the files match the composite search criteria."
                )

    # ========== History / Snapshot ==========

    def _current_run_settings(self) -> "CompareRunSettings":
        from compare.compare_history import CompareRunSettings
        return CompareRunSettings(
            overwrite=self.get_overwrite(),
            store_checkpoints=self.get_store_checkpoints(),
            use_matrix_comparison=self.get_use_matrix_comparison(),
            threshold=self.get_threshold(),
            counter_limit=self.get_counter_limit(),
        )

    def _apply_run_settings(self, settings: "CompareRunSettings") -> None:
        self.set_overwrite(settings.overwrite)
        self.set_store_checkpoints(settings.store_checkpoints)
        self.set_use_matrix_comparison(settings.use_matrix_comparison)
        self.set_threshold(settings.threshold)
        self.set_counter_limit(settings.counter_limit)

    def snapshot(self, args: "CompareArgs") -> "CompareHistory":
        """Capture current configuration as a CompareHistory entry."""
        from compare.compare_history import CompareHistory
        from compare.compare_filters import filter_to_dict
        from datetime import datetime
        instances = [
            {
                "instance_id": cfg.instance_id,
                "compare_mode": cfg.compare_mode.value,
                "weight": cfg.weight,
                "threshold": cfg.threshold,
                "enabled": cfg.enabled,
                "search_text": cfg.search_text,
                "search_text_negative": cfg.search_text_negative,
            }
            for cfg in self._mode_configs.values()
        ]
        return CompareHistory(
            directory=args.base_dir,
            timestamp=datetime.now().isoformat(),
            instances=instances,
            combination_logic=self._combination_logic.value,
            filter_dict=filter_to_dict(self._data_filter),
            run_settings=self._current_run_settings(),
            file_filter=args.file_filter or None,
        )

    def apply_snapshot(self, h: "CompareHistory") -> None:
        """Restore configuration from a CompareHistory snapshot."""
        from compare.compare_filters import filter_from_dict
        # Remove all current instances
        for iid in list(self._mode_configs.keys()):
            self.remove_mode_instance(iid)
        # Re-add from snapshot
        for i, inst in enumerate(h.instances):
            mode_str = inst.get("compare_mode", CompareMode.CLIP_EMBEDDING.value)
            try:
                mode = CompareMode(mode_str)
            except ValueError:
                try:
                    mode = CompareMode[mode_str]
                except KeyError:
                    logger.warning(f"Unknown compare mode in snapshot: {mode_str!r}")
                    continue
            if i == 0:
                self.set_primary_mode(mode)
                primary_iid = self._primary_instance_id
                if primary_iid and primary_iid in self._mode_configs:
                    cfg = self._mode_configs[primary_iid]
                    cfg.weight = inst.get("weight", 1.0)
                    cfg.threshold = inst.get("threshold")
                    cfg.search_text = inst.get("search_text")
                    cfg.search_text_negative = inst.get("search_text_negative")
            else:
                self.add_mode_instance(
                    compare_mode=mode,
                    weight=inst.get("weight", 1.0),
                    threshold=inst.get("threshold"),
                    search_text=inst.get("search_text"),
                    search_text_negative=inst.get("search_text_negative"),
                )
        # Restore combination logic
        try:
            self.set_combination_logic(CombinationLogic(h.combination_logic))
        except ValueError:
            logger.warning(f"Unknown combination logic in snapshot: {h.combination_logic!r}")
        # Restore filter and global run settings
        self.set_data_filter(filter_from_dict(h.filter_dict))
        self._apply_run_settings(h.run_settings)

    def reset_to_default(self) -> None:
        """Clear all instances and restore a single CLIP_EMBEDDING with no filter."""
        from compare.compare_history import CompareRunSettings
        for iid in list(self._mode_configs.keys()):
            self.remove_mode_instance(iid)
        self.set_primary_mode(CompareMode.CLIP_EMBEDDING)
        self.set_combination_logic(CombinationLogic.AND)
        self.set_data_filter(None)
        self._apply_run_settings(CompareRunSettings(
            store_checkpoints=config.store_checkpoints,
        ))


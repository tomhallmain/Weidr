from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from compare.compare_wrapper import CompareWrapper
from compare.compare_args import CompareArgs
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


# Filter classes (will be moved to compare_args.py when filtering is integrated)
@dataclass
class SizeFilter:
    """Filter criteria for media dimensions."""
    min_size: Optional[Tuple[int, int]] = None  # Minimum (width, height)
    max_size: Optional[Tuple[int, int]] = None  # Maximum (width, height)
    exact_size: Optional[Tuple[int, int]] = None  # Exact size match
    tolerance: int = 0  # Pixel tolerance for exact match
    
    def is_active(self) -> bool:
        """Check if any size filtering criteria is set."""
        return (self.min_size is not None or 
                self.max_size is not None or 
                self.exact_size is not None)


@dataclass
class ModelFilter:
    """Filter criteria for media models/loras."""
    models: Optional[List[str]] = None  # Model names to filter by
    mode: str = 'include'  # 'include' or 'exclude' - include/exclude matching models
    match_any: bool = False  # If True, match any model; if False, match all models
    include_loras: bool = True  # Include loras in model matching
    
    def is_active(self) -> bool:
        """Check if model filtering criteria is set."""
        return self.models is not None and len(self.models) > 0


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
        
        # CompareWrapper instances (one per mode, shared across instances of same mode)
        self._wrappers: Dict[CompareMode, CompareWrapper] = {}
        
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
        
        # Filtering (from FILTERING_PROPOSAL.md)
        self._size_filter: Optional[SizeFilter] = None
        self._model_filter: Optional[ModelFilter] = None
        
        # Compare settings (migrated from sidebar)
        self._threshold: Optional[float] = None
        self._counter_limit: Optional[int] = None
        self._compare_faces: bool = False
        self._overwrite: bool = False
        self._store_checkpoints: bool = config.store_checkpoints
        
        # Results from last run
        self._last_results: Optional[Dict[CompareMode, Dict[str, float]]] = None
        self._combined_results: Optional[Dict[str, float]] = None
        
        # State management (delegated to primary wrapper for single-mode)
        self._is_composite_mode: bool = False
    
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
    
    def set_primary_mode(self, compare_mode: CompareMode):
        """
        Set the primary comparison mode. For single-mode operations,
        this is the only active mode. For composite mode, this is the
        mode used for result presentation and navigation.
        """
        self._primary_mode = compare_mode
        # If no instances exist, create one for this mode
        if not any(config.compare_mode == compare_mode for config in self._mode_configs.values()):
            instance_id = self._generate_instance_id(compare_mode)
            self._mode_configs[instance_id] = CompareConfig(
                instance_id=instance_id,
                compare_mode=compare_mode,
                enabled=True
            )
        self._ensure_wrapper(compare_mode)
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
        self._ensure_wrapper(compare_mode)
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
        # Don't delete wrapper - keep for potential reuse
        
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
    
    def _ensure_wrapper(self, compare_mode: CompareMode) -> CompareWrapper:
        """Get or create CompareWrapper for a mode."""
        if compare_mode not in self._wrappers:
            self._wrappers[compare_mode] = CompareWrapper(
                self._master, compare_mode, self._app_actions
            )
        return self._wrappers[compare_mode]
    
    # ========== Filtering ==========
    
    def set_size_filter(self, size_filter: Optional[SizeFilter]):
        """Set size filtering criteria."""
        self._size_filter = size_filter
        if size_filter and size_filter.is_active():
            logger.info(f"Size filter set: min={size_filter.min_size}, max={size_filter.max_size}, exact={size_filter.exact_size}")
        else:
            logger.info("Size filter cleared")
    
    def get_size_filter(self) -> Optional[SizeFilter]:
        """Get current size filter."""
        return self._size_filter
    
    def set_model_filter(self, model_filter: Optional[ModelFilter]):
        """Set model filtering criteria."""
        self._model_filter = model_filter
        if model_filter and model_filter.is_active():
            logger.info(f"Model filter set: models={model_filter.models}, mode={model_filter.mode}, match_any={model_filter.match_any}, include_loras={model_filter.include_loras}")
        else:
            logger.info("Model filter cleared")
    
    def get_model_filter(self) -> Optional[ModelFilter]:
        """Get current model filter."""
        return self._model_filter
    
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
        args.compare_faces = self.get_compare_faces()
        args.overwrite = self.get_overwrite()
        args.store_checkpoints = self.get_store_checkpoints()
        
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
        logger.info(f" compare faces: {args.compare_faces}")
        
        # Threshold display depends on mode
        if self._primary_mode == CompareMode.COLOR_MATCHING:
            logger.info(f" color diff threshold: {args.threshold}")
        else:
            logger.info(f" embedding similarity threshold: {args.threshold}")
        
        logger.info(f" max file process limit: {args.counter_limit}")
        logger.info(f" recursive: {args.recursive}")
        logger.info(f" file glob pattern: {args.inclusion_pattern}")
        logger.info(f" include videos: {args.include_videos}")
        logger.info(f" overwrite media data: {args.overwrite}")
        logger.info(f" store checkpoints: {args.store_checkpoints}")
        
        # Filter settings
        if self._size_filter and self._size_filter.is_active():
            logger.info(f" size filter: min={self._size_filter.min_size}, max={self._size_filter.max_size}, exact={self._size_filter.exact_size}")
        else:
            logger.info(" size filter: None")
        
        if self._model_filter and self._model_filter.is_active():
            logger.info(f" model filter: models={self._model_filter.models}, mode={self._model_filter.mode}, match_any={self._model_filter.match_any}, include_loras={self._model_filter.include_loras}")
        else:
            logger.info(" model filter: None")
        
        logger.info("|--------------------------------------------------------------------|\n")
    
    def set_compare_faces(self, compare_faces: bool):
        """Set compare faces option."""
        self._compare_faces = compare_faces
        logger.info(f"Compare faces set to: {compare_faces}")
    
    def get_compare_faces(self) -> bool:
        """Get compare faces option."""
        return self._compare_faces
    
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
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].files_matched
        return []
    
    @property
    def file_groups(self) -> Dict:
        """Get file groups from primary wrapper (for backward compatibility)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].file_groups
        return {}
    
    @property
    def files_grouped(self) -> Dict:
        """Get grouped files from primary wrapper (for backward compatibility)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].files_grouped
        return {}
    
    @property
    def match_index(self) -> int:
        """Get match index from primary wrapper (for backward compatibility)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].match_index
        return 0
    
    @match_index.setter
    def match_index(self, value: int):
        """Set match index on primary wrapper (for backward compatibility)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            self._wrappers[self._primary_mode].match_index = value
    
    @property
    def current_group_index(self) -> int:
        """Get current group index from primary wrapper (for backward compatibility)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].current_group_index
        return 0
    
    @current_group_index.setter
    def current_group_index(self, value: int):
        """Set current group index on primary wrapper (for backward compatibility)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            self._wrappers[self._primary_mode].current_group_index = value
    
    @property
    def search_file_path(self) -> Optional[str]:
        """Get search image path from primary wrapper."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].search_file_path
        return None
    
    @search_file_path.setter
    def search_file_path(self, value: Optional[str]):
        """Set search image path on primary wrapper."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            self._wrappers[self._primary_mode].search_file_path = value
    
    @property
    def hidden_media(self) -> List[str]:
        """Get hidden images from primary wrapper."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].hidden_media
        return []
    
    @property
    def group_indexes(self) -> List[int]:
        """Get group indexes from primary wrapper."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].group_indexes
        return []
    
    @property
    def max_group_index(self) -> int:
        """Get max group index from primary wrapper."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].max_group_index
        return 0
    
    @property
    def has_media_matches(self) -> bool:
        """Get has_media_matches from primary wrapper."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].has_media_matches
        return False
    
    # ========== Delegation Methods (for backward compatibility) ==========
    
    def has_compare(self) -> bool:
        """Check if primary wrapper has a compare instance."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].has_compare()
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
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].get_args()
        return CompareArgs()
    
    def validate_compare_mode(self, required_compare_mode, error_text):
        """Validate compare mode (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            self._wrappers[self._primary_mode].validate_compare_mode(
                required_compare_mode, error_text
            )
    
    def current_match(self) -> Optional[str]:
        """Get current match from primary wrapper."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].current_match()
        return None
    
    def show_prev_media(self, show_alert=True) -> bool:
        """Show previous media (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].show_prev_media(show_alert)
        return False
    
    def show_next_media(self, show_alert=True) -> bool:
        """Show next media (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].show_next_media(show_alert)
        return False
    
    def skip_file(self, image_path: str) -> bool:
        """Check if image should be skipped (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].skip_file(image_path)
        return False
    
    def show_prev_group(self, event=None, file_browser=None):
        """Show previous group (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].show_prev_group(event, file_browser)
    
    def show_next_group(self, event=None, file_browser=None):
        """Show next group (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].show_next_group(event, file_browser)
    
    def set_current_group(self, start_match_index=0):
        """Set current group (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].set_current_group(start_match_index)
    
    def page_down(self, half_length=False) -> Optional[str]:
        """Page down (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].page_down(half_length)
        return None
    
    def page_up(self, half_length=False) -> Optional[str]:
        """Page up (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].page_up(half_length)
        return None
    
    def select_series(self, start_file: str, end_file: str) -> List[str]:
        """Select series (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].select_series(start_file, end_file)
        return []
    
    def find_file_after_comparison(self, app_mode, search_text="", exact_match=False):
        """Find file after comparison (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].find_file_after_comparison(
                app_mode, search_text, exact_match
            )
        return None, None
    
    def _update_groups_for_removed_file(self, app_mode, group_index, match_index, 
                                       set_group=True, show_next_media=None):
        """Update groups for removed file (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode]._update_groups_for_removed_file(
                app_mode, group_index, match_index, set_group, show_next_media
            )
    
    def update_compare_for_readded_file(self, readded_file: str):
        """Update compare for readded file (delegated to all wrappers)."""
        for wrapper in self._wrappers.values():
            wrapper.update_compare_for_readded_file(readded_file)
    
    def _get_file_group_map(self, app_mode):
        """Get file group map (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode]._get_file_group_map(app_mode)
        return {}
    
    def find_next_unrelated_file(self, file_browser, forward=True):
        """Find next unrelated image (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].find_next_unrelated_file(
                file_browser, forward
            )
    
    def _get_prev_file(self):
        """Get previous image (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode]._get_prev_file()
        return None
    
    def _get_next_file(self):
        """Get next image (delegated to primary wrapper)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode]._get_next_file()
        return None
    
    def compare(self):
        """Get compare instance from primary wrapper (for backward compatibility)."""
        if self._primary_mode and self._primary_mode in self._wrappers:
            return self._wrappers[self._primary_mode].compare()
        raise Exception("No compare object created")
    
    # ========== Main Execution ==========
    
    def run(self, args: CompareArgs = CompareArgs()):
        """
        Execute comparison. For single-mode, delegates to wrapper.
        For composite mode, runs multiple comparisons and combines results.
        """
        if not self._primary_mode:
            raise ValueError("No compare mode set")
        
        # Apply filters to args (from FILTERING_PROPOSAL.md)
        # Note: This will be integrated into CompareArgs when filtering is implemented
        # For now, filters are stored in manager but not yet applied to args
        # TODO: Add size_filter and model_filter properties to CompareArgs
        
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
            wrapper = self._ensure_wrapper(self._primary_mode)
            wrapper.run(args)
            return
        
        # Composite mode operation
        self._run_composite(args)
    
    def _run_composite(self, args: CompareArgs):
        """
        Run composite comparison across multiple mode instances and combine results.
        """
        if not self._mode_configs:
            raise ValueError("No compare modes configured for composite search")
        
        self._app_actions._set_label_state(
            f"Running composite comparison with {len(self._mode_configs)} instances..."
        )
        
        # Run each enabled instance
        instance_results: Dict[str, Dict[str, float]] = {}  # instance_id -> {file_path: score}
        
        for instance_id, config in self._mode_configs.items():
            if not config.enabled:
                continue
            
            wrapper = self._ensure_wrapper(config.compare_mode)
            
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
            
            # Run comparison
            try:
                wrapper.run(instance_args)
                
                # Extract results
                # run_search() returns {0: {file_path: score}}
                files_grouped = wrapper.files_grouped
                if 0 in files_grouped:
                    instance_results[instance_id] = files_grouped[0]
                else:
                    instance_results[instance_id] = {}
                    
            except Exception as e:
                logger.error(f"Error running instance {instance_id} ({config.compare_mode.name}): {e}")
                instance_results[instance_id] = {}
        
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
        
        # Combine results based on logic (convert instance_results to mode-based for combination)
        mode_results_for_combine = {}
        for instance_id, results in instance_results.items():
            config = self._mode_configs[instance_id]
            if config.compare_mode not in mode_results_for_combine:
                mode_results_for_combine[config.compare_mode] = {}
            # Merge results from multiple instances of same mode
            for file_path, score in results.items():
                if file_path not in mode_results_for_combine[config.compare_mode]:
                    mode_results_for_combine[config.compare_mode][file_path] = score
                else:
                    mode_results_for_combine[config.compare_mode][file_path] = max(
                        mode_results_for_combine[config.compare_mode][file_path], score
                    )
        
        self._combined_results = self._combine_results(mode_results_for_combine)
        logger.info(f"Combined results using {self._combination_logic.value} logic: {len(self._combined_results)} matches")
        
        # Update primary wrapper with combined results
        self._apply_combined_results_to_primary()
    
    def _combine_results(self, mode_results: Dict[CompareMode, Dict[str, float]]) -> Dict[str, float]:
        """
        Combine results from multiple comparison modes.
        Returns dict mapping file_path -> combined_score
        """
        if not mode_results:
            return {}
        
        if self._combination_logic == CombinationLogic.AND:
            return self._combine_and(mode_results)
        elif self._combination_logic == CombinationLogic.OR:
            return self._combine_or(mode_results)
        else:  # WEIGHTED
            return self._combine_weighted(mode_results)
    
    def _combine_and(self, mode_results: Dict[CompareMode, Dict[str, float]]) -> Dict[str, float]:
        """AND logic: file must appear in ALL mode results."""
        if not mode_results:
            return {}
        
        # Start with files from first mode
        result_sets = [set(mode_results[mode].keys()) for mode in mode_results]
        common_files = set.intersection(*result_sets) if result_sets else set()
        
        # For common files, use minimum score (most conservative)
        combined = {}
        for file_path in common_files:
            scores = [mode_results[mode][file_path] for mode in mode_results if file_path in mode_results[mode]]
            combined[file_path] = min(scores)  # Most conservative
        
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
    
    def _combine_weighted(self, mode_results: Dict[CompareMode, Dict[str, float]]) -> Dict[str, float]:
        """Weighted combination: weighted average of scores."""
        # Collect all files
        all_files = set()
        for mode_results_dict in mode_results.values():
            all_files.update(mode_results_dict.keys())
        
        # Calculate total weight for normalization (sum weights of enabled instances)
        total_weight = sum(config.weight for config in self._mode_configs.values() if config.enabled)
        if total_weight == 0:
            total_weight = 1.0
        
        combined = {}
        for file_path in all_files:
            weighted_sum = 0.0
            weight_sum = 0.0
            
            # For each mode, use the average weight of its instances
            for compare_mode in mode_results:
                if file_path in mode_results[compare_mode]:
                    # Get average weight of enabled instances for this mode
                    mode_instances = [config for config in self._mode_configs.values() 
                                    if config.compare_mode == compare_mode and config.enabled]
                    if mode_instances:
                        avg_weight = sum(inst.weight for inst in mode_instances) / len(mode_instances)
                        score = mode_results[compare_mode][file_path]
                        weighted_sum += score * avg_weight
                        weight_sum += avg_weight
            
            if weight_sum > 0:
                combined[file_path] = weighted_sum / weight_sum
            else:
                combined[file_path] = 0.0
        
        return combined
    
    def _apply_combined_results_to_primary(self):
        """Apply combined results to primary wrapper for navigation."""
        if not self._primary_mode or not self._combined_results:
            return
        
        wrapper = self._ensure_wrapper(self._primary_mode)
        
        # Update wrapper's results structure
        wrapper.files_grouped = {0: self._combined_results}
        wrapper.file_groups = deepcopy(wrapper.files_grouped)
        
        # Sort files by combined score
        reverse = self._primary_mode.is_embedding()
        wrapper.files_matched = []
        for f in sorted(self._combined_results.keys(), 
                       key=lambda f: self._combined_results[f], 
                       reverse=reverse):
            wrapper.files_matched.append(f)
        
        # Set up navigation state
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


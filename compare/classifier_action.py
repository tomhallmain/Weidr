"""
Classifier action and prevalidation models (serialization, validation, run logic).

List storage and directory/prevalidation cache live in
``compare.classifier_actions_manager.ClassifierActionsManager``.
"""

from enum import Enum
import math
import os
from typing import Optional

from compare.compare_embeddings_clip import CompareEmbeddingClip
from compare.embedding_prototype import EmbeddingPrototype
from compare.lookahead import Lookahead
from files.directory_profile import DirectoryProfile
from files.file_action import FileAction
from image.image_classifier_manager import image_classifier_manager
from image.image_data_extractor import image_data_extractor
from image.frame_cache import FrameCache
from image.image_ops import ImageOps
from utils.config import config
from utils.media_utils import is_classifier_dynamic_media_path
from utils.constants import ActionType, ClassifierActionType
from utils.logging_setup import get_logger
from utils.running_tasks_registry import start_thread
from utils.translations import _
from utils.utils import Utils


logger = get_logger("classifier_action")


class ImageClassifierClassificationMode(Enum):
    SELECTED_CATEGORIES = "selected_categories"
    MODEL_STRATEGY = "model_strategy"

    @staticmethod
    def from_value(value):
        if isinstance(value, ImageClassifierClassificationMode):
            return value
        value_str = str(value or "").strip().lower()
        if value_str == ImageClassifierClassificationMode.MODEL_STRATEGY.value:
            return ImageClassifierClassificationMode.MODEL_STRATEGY
        return ImageClassifierClassificationMode.SELECTED_CATEGORIES





class ClassifierAction:
    NO_POSITIVES_STR = _("(no positives set)")
    NO_NEGATIVES_STR = _("(no negatives set)")

    def __init__(self, name=_("New Classifier Action"), positives=[], negatives=[], threshold=0.23,
                 text_embedding_threshold=None, prototype_threshold=0.23,
                 action=ClassifierActionType.NOTIFY, action_modifier="",
                 image_classifier_name="", image_classifier_selected_categories=[], 
                 classification_mode=ImageClassifierClassificationMode.SELECTED_CATEGORIES,
                 use_embedding=True, use_image_classifier=False, use_prompts=False, use_blacklist=False,
                 use_pseudostatic_dynamic_media=False,
                 is_active=True, use_prototype=False, prototype_directory="",
                 negative_prototype_directory="", negative_prototype_lambda=0.5,
                 dynamic_content_sample_ratio=0.1, dynamic_content_positive_ratio=0.1,
                 _last_used_profile=None, lookahead_names=[],
                 can_run: bool = True, initialization_error: Optional[str] = None):
        self.name = name
        self.positives = positives
        self.negatives = negatives
        # Backward compatibility: if text_embedding_threshold is None, use threshold
        self.text_embedding_threshold = text_embedding_threshold if text_embedding_threshold is not None else threshold
        self.prototype_threshold = prototype_threshold
        # Keep threshold for backward compatibility (maps to text_embedding_threshold)
        self.threshold = self.text_embedding_threshold
        self.action = action if isinstance(action, Enum) else ClassifierActionType[action]
        self.action_modifier = action_modifier  # Target directory for MOVE/COPY actions
        self.is_active = is_active  # Whether this action is enabled/active
        self.image_classifier_name = image_classifier_name
        self.image_classifier = None
        self._missing_image_classifier_logged = False
        self.image_classifier_categories = []
        self.image_classifier_selected_categories = image_classifier_selected_categories
        self.classification_mode = ImageClassifierClassificationMode.from_value(classification_mode)
        self.lookahead_names = lookahead_names if lookahead_names else []  # List of lookahead names (strings)
        self.use_embedding = use_embedding
        self.use_image_classifier = use_image_classifier
        self.use_prompts = use_prompts
        self.use_blacklist = use_blacklist
        self.use_pseudostatic_dynamic_media = bool(use_pseudostatic_dynamic_media)
        self.use_prototype = use_prototype  # Whether to use embedding prototype
        self.prototype_directory = prototype_directory  # Directory containing sample images for positive prototype
        self.negative_prototype_directory = negative_prototype_directory  # Directory containing sample images for negative prototype
        self.negative_prototype_lambda = negative_prototype_lambda  # Weight for negative prototype (λ)
        self._cached_prototype = None  # Cached positive prototype embedding
        self._cached_negative_prototype = None  # Cached negative prototype embedding
        self._last_used_profile = _last_used_profile  # Last used profile name or directory path (None for new actions)
        # For dynamic media (video/GIF, etc.), sample this proportion of frames
        # and require this proportion of positives before firing the action.
        self.dynamic_content_sample_ratio = self._normalize_ratio(dynamic_content_sample_ratio, default_val=0.1)
        self.dynamic_content_positive_ratio = self._normalize_ratio(dynamic_content_positive_ratio, default_val=0.1)
        self.can_run = bool(can_run)
        self.initialization_error = initialization_error
        self._blocked_run_logged = False
        self._model_strategy_sig: Optional[tuple] = None
        self._model_strategy_positives: Optional[frozenset[str]] = None

    def mark_runtime_valid(self) -> None:
        """Call after validate() succeeds (e.g. user saved in the editor) to clear load-time failure state."""
        self.can_run = True
        self.initialization_error = None
        self._blocked_run_logged = False
        self._invalidate_model_strategy_cache()

    def _invalidate_model_strategy_cache(self) -> None:
        self._model_strategy_sig = None
        self._model_strategy_positives = None

    def __eq__(self, other):
        """Check equality based on name (classifier actions are uniquely identified by name)."""
        if not isinstance(other, ClassifierAction):
            return False
        return self.name == other.name

    def __hash__(self):
        """Hash based on name (classifier actions are uniquely identified by name)."""
        return hash(self.name)

    def set_positives(self, text):
        self.positives = [x.strip() for x in Utils.split(text, ",") if len(x) > 0]

    def set_negatives(self, text):
        self.negatives = [x.strip() for x in Utils.split(text, ",") if len(x) > 0]

    @staticmethod
    def _normalize_ratio(value, default_val: float = 0.1) -> float:
        try:
            normalized = float(value)
        except Exception:
            normalized = default_val
        return max(0.0, min(1.0, normalized))

    def get_positives_str(self):
        if len(self.positives) > 0:
            out = ""
            for i in range(len(self.positives)):
                out += self.positives[i].replace(",", "\\,") + ", " if i < len(self.positives)-1 else self.positives[i]
            return out
        else:
            return ClassifierAction.NO_POSITIVES_STR

    def set_image_classifier(self, classifier_name):
        self.image_classifier_name = classifier_name
        self.image_classifier = image_classifier_manager.get_classifier(classifier_name)
        self._missing_image_classifier_logged = False
        self.image_classifier_categories = []
        self._invalidate_model_strategy_cache()
        if self.image_classifier is not None:
            self.image_classifier_categories.extend(list(self.image_classifier.model_categories))

    def ensure_image_classifier_loaded(self, notify_callback):
        """Lazy load the image classifier if it hasn't been loaded yet."""
        if self.image_classifier is None and self.image_classifier_name:
            try:
                if notify_callback is not None:
                    notify_callback(_("Loading image classifier <{0}> ...").format(self.image_classifier_name))
                self.set_image_classifier(self.image_classifier_name)
            except Exception as e:
                logger.error(f"Error loading image classifier <{self.image_classifier_name}>: {e}")

    def is_selected_category_unset(self):
        # TODO - this may be incorrect, would make more sense to be the opposite logic, need to check
        return len(self.image_classifier_selected_categories) > 0

    def _check_prompt_validation(self, image_path):
        """Check if image prompts match the positive or negative criteria."""
        try:
            positive_prompt, negative_prompt = image_data_extractor.extract_prompts_all_strategies(image_path)

            # Skip if no prompts found (None indicates failure to extract prompts)
            if positive_prompt is None:
                return False
            
            # Check positive prompts
            positive_match = True
            if self.positives:
                positive_match = any(pos.lower() in positive_prompt.lower() for pos in self.positives)
            
            # Check negative prompts
            negative_match = True
            if self.negatives:
                negative_match = any(neg.lower() in negative_prompt.lower() for neg in self.negatives)
            
            return positive_match or negative_match
            
        except Exception as e:
            logger.error(f"Error checking prompt validation for {image_path}: {e}")
            return False

    def _check_lookaheads(self, image_path, lookahead_eval_cache=None):
        """Check if any lookahead prevalidations are triggered. Returns True if any lookahead passes."""
        from compare.classifier_actions_manager import ClassifierActionsManager

        if not self.lookahead_names:
            return False
        
        for lookahead_name in self.lookahead_names:
            # Look up the lookahead from the shared list
            lookahead = Lookahead.get_lookahead_by_name(lookahead_name)
            if lookahead is None:
                continue

            if lookahead_eval_cache is not None:
                eval_cache_key = Lookahead.eval_cache_key(lookahead_name, image_path)
                if eval_cache_key in lookahead_eval_cache:
                    if lookahead_eval_cache[eval_cache_key]:
                        return True
                    continue
            else:
                # Check if this lookahead has already been evaluated in this prevalidate call
                if lookahead.run_result is not None:
                    # Use cached result
                    if lookahead.run_result:
                        logger.info(f"Lookahead {lookahead_name} triggered for prevalidation {self.name} (cached)")
                        return True
                    continue
            
            name_or_text = lookahead.name_or_text
            threshold = lookahead.threshold
            
            # Check if it's a prevalidation name or custom text
            if lookahead.is_prevalidation_name:
                # It's a prevalidation name - get the referenced prevalidation
                lookahead_prevalidation = ClassifierActionsManager.get_prevalidation_by_name(name_or_text)
                if lookahead_prevalidation is None:
                    # Prevalidation not found, skip this lookahead
                    if lookahead_eval_cache is not None:
                        lookahead_eval_cache[eval_cache_key] = False
                    else:
                        lookahead.run_result = False  # Cache the result
                    continue
                # Use the lookahead prevalidation's positives/negatives
                positives = lookahead_prevalidation.positives
                negatives = lookahead_prevalidation.negatives
                # Skip if the referenced prevalidation has no positives or negatives
                if not positives and not negatives:
                    if lookahead_eval_cache is not None:
                        lookahead_eval_cache[eval_cache_key] = False
                    else:
                        lookahead.run_result = False  # Cache the result
                    continue
            else:
                # It's custom text, treat as a positive
                positives = [name_or_text]
                negatives = []
            
            # Check if this lookahead passes
            result = CompareEmbeddingClip.multi_text_compare(image_path, positives, negatives, threshold)
            if lookahead_eval_cache is not None:
                lookahead_eval_cache[eval_cache_key] = result
            else:
                lookahead.run_result = result  # Cache the result
            
            if result:
                # logger.info(f"Lookahead {lookahead_name} triggered for prevalidation {self.name}")
                return True
        
        return False

    def ensure_prototype_loaded(self, notify_callback, force_recalculate=False):
        """Lazy load the prototype embeddings if needed."""
        if not self.use_prototype or not self.prototype_directory:
            return
        
        # Load positive prototype
        if self._cached_prototype is None or force_recalculate:
            try:
                if notify_callback is not None:
                    notify_callback(_("Loading embedding prototype from {0}...").format(self.prototype_directory))
                self._cached_prototype = EmbeddingPrototype.calculate_prototype_from_directory(
                    self.prototype_directory,
                    force_recalculate=force_recalculate,
                    notify_callback=notify_callback
                )
                if self._cached_prototype is None:
                    logger.error(f"Failed to load prototype from {self.prototype_directory}")
            except Exception as e:
                logger.error(f"Error loading prototype from {self.prototype_directory}: {e}")
                self._cached_prototype = None
        
        # Load negative prototype if specified
        if self.negative_prototype_directory and (self._cached_negative_prototype is None or force_recalculate):
            try:
                if notify_callback is not None:
                    notify_callback(_("Loading negative embedding prototype from {0}...").format(self.negative_prototype_directory))
                self._cached_negative_prototype = EmbeddingPrototype.calculate_prototype_from_directory(
                    self.negative_prototype_directory,
                    force_recalculate=force_recalculate,
                    notify_callback=notify_callback
                )
                if self._cached_negative_prototype is None:
                    logger.error(f"Failed to load negative prototype from {self.negative_prototype_directory}")
            except Exception as e:
                logger.error(f"Error loading negative prototype from {self.negative_prototype_directory}: {e}")
                self._cached_negative_prototype = None
    
    def _check_prototype_validation(self, image_path):
        """Check if image matches the prototype embedding.
        
        Uses formula: Final Score = sim(query, positive_proto) - λ * sim(query, negative_proto)
        If negative prototype is not set, uses only positive similarity.
        """
        if not self.use_prototype or self._cached_prototype is None:
            return False
        
        try:
            # Use ClassifierAction name as session_cache_key for efficient result caching
            # Calculate similarity to positive prototype (prototype_type=0)
            positive_similarity = EmbeddingPrototype.compare_with_prototype(
                image_path, self._cached_prototype, session_cache_key=self.name
            )
            if config.debug2:
                logger.info(self.name + " Positive similarity: " + str(positive_similarity))
            # If negative prototype is set, subtract weighted negative similarity (prototype_type=1)
            if self._cached_negative_prototype is not None:
                negative_similarity = EmbeddingPrototype.compare_with_prototype(
                    image_path, self._cached_negative_prototype, session_cache_key=self.name, negative_prototype=1
                )
                if config.debug2:
                    logger.info(self.name + " Negative similarity: " + str(negative_similarity))
                final_score = positive_similarity - self.negative_prototype_lambda * negative_similarity
                if config.debug2:
                    logger.info(self.name + " Final score: " + str(final_score))
            else:
                final_score = positive_similarity
                if config.debug2:
                    logger.info(self.name + " Final score: " + str(final_score))
            return final_score >= self.prototype_threshold
        except Exception as e:
            logger.error(f"Error checking prototype validation for {image_path}: {e}")
            return False
    
    def _run_with_batch_prototype_validation(self, directory_paths: list[str], hide_callback, notify_callback, add_mark_callback=None, max_images_per_batch: Optional[int] = None):
        """
        Run classifier action with batch prototype validation for efficiency.
        
        Delegates batch processing to EmbeddingPrototype, then runs actions on matching images.
        Runs the entire process (batch validation + action execution) in a separate thread.
        
        Args:
            directory_paths: List of directory paths to process
            hide_callback: Callback for hiding images
            notify_callback: Callback for notifications
            add_mark_callback: Optional callback for marking images
            max_images_per_batch: Optional maximum number of images to process per batch
        """
        if not self.use_prototype or self._cached_prototype is None:
            return
        
        def batch_validation_worker():
            """Worker function to run batch validation and actions in a separate thread."""
            try:
                # Use EmbeddingPrototype to batch validate images from directories
                matching_paths = EmbeddingPrototype.batch_validate_with_prototypes(
                    directories=directory_paths,
                    positive_prototype=self._cached_prototype,
                    threshold=self.prototype_threshold,
                    negative_prototype=self._cached_negative_prototype,
                    negative_lambda=self.negative_prototype_lambda,
                    notify_callback=notify_callback,
                    max_images_per_batch=max_images_per_batch
                )
                
                # Run actions on matching images
                for image_path in matching_paths:
                    try:
                        self.run_action(image_path, hide_callback, notify_callback, add_mark_callback)
                    except Exception as e:
                        logger.error(f"Error running action on {image_path}: {e}")
            except Exception as e:
                logger.error(f"Error in batch prototype validation: {e}")
        
        # Start batch validation and action execution in a separate thread
        start_thread(batch_validation_worker, use_asyncio=False)

    def run(self, directory_paths: list[str], hide_callback, notify_callback, add_mark_callback=None, profile_name_or_path: Optional[str] = None, max_images_per_batch: Optional[int] = None):
        """Run the classifier action on the given directory paths.
        
        Args:
            directory_paths: List of directory paths to process
            hide_callback: Callback for hiding images
            notify_callback: Callback for notifications
            add_mark_callback: Optional callback for marking images
            profile_name_or_path: Optional profile name or directory path to store as last used
            max_images_per_batch: Optional maximum number of images to process per batch
        """
        if not self.is_active:
            logger.info(f"Classifier action {self.name} is disabled, skipping")
            return
        if not self.can_run:
            if not self._blocked_run_logged:
                logger.warning(
                    "Classifier action %r is skipped (runtime init failed): %s",
                    self.name,
                    self.initialization_error or _("unknown error"),
                )
                self._blocked_run_logged = True
            return

        # Store the last used profile or directory path
        if profile_name_or_path:
            self._last_used_profile = profile_name_or_path
        elif directory_paths:
            # If no profile name provided, use the first directory path
            self._last_used_profile = directory_paths[0]

        logger.info(f"Running classifier action {self.name} on {len(directory_paths)} directories")
        
        # Pre-load image classifier and prototype before processing images
        self.ensure_image_classifier_loaded(notify_callback)
        self.ensure_prototype_loaded(notify_callback)
        
        # Use batch prototype validation when prototype validation is enabled
        if self.use_prototype:
            self._run_with_batch_prototype_validation(directory_paths, hide_callback, notify_callback, add_mark_callback, max_images_per_batch)

    def _evaluate_image_path_match(
        self, image_path: str, lookahead_eval_cache=None
    ) -> tuple[bool, Optional[str]]:
        # Note: Image classifier and prototype should be loaded before calling this method
        # (see ClassifierActionsWindow.run_classifier_action for pre-loading)
        if not self.can_run:
            return False, None

        # Check each enabled validation type with short-circuit OR logic        
        if self.use_prototype:
            if self._check_prototype_validation(image_path):
                return True, None

        # Check lookaheads first - if any pass, skip this prevalidation
        if self._check_lookaheads(image_path, lookahead_eval_cache=lookahead_eval_cache):
            return False, None

        if self.use_embedding:
            if CompareEmbeddingClip.multi_text_compare(image_path, self.positives, self.negatives, self.text_embedding_threshold):
                return True, None
        
        if self.use_image_classifier:
            if self.image_classifier is None and self.image_classifier_name:
                # Lazy attempt; no notify callback here.
                self.ensure_image_classifier_loaded(None)
            if self.image_classifier is not None:
                if self.classification_mode == ImageClassifierClassificationMode.MODEL_STRATEGY:
                    predicted_category = self.image_classifier.classify_image(image_path)
                    positive_categories = self._resolve_model_strategy_positive_categories()
                    if predicted_category in positive_categories:
                        return True, predicted_category
                else:
                    if self.image_classifier.test_image_for_categories(
                        image_path, self.image_classifier_selected_categories
                    ):
                        try:
                            predicted_category = self.image_classifier.classify_image(
                                image_path
                            )
                        except Exception:
                            predicted_category = None
                        if predicted_category in self.image_classifier_selected_categories:
                            return True, predicted_category
                        return True, None
            else:
                if not self._missing_image_classifier_logged:
                    logger.error(f"Image classifier {self.image_classifier_name} not found for classifier action {self.name}")
                    self._missing_image_classifier_logged = True
        
        if self.use_prompts:
            if self._check_prompt_validation(image_path):
                return True, None
        
        # No validation type passed
        return False, None

    def matches_image_path(self, image_path, lookahead_eval_cache=None) -> bool:
        is_match, _unused = self._evaluate_image_path_match(
            image_path, lookahead_eval_cache=lookahead_eval_cache
        )
        return is_match

    def run_on_media_path(
        self,
        media_path,
        hide_callback,
        notify_callback,
        add_mark_callback=None,
        blur_callback=None,
        base_directory: Optional[str] = None,
    ) -> Optional[ClassifierActionType]:
        if not self.can_run:
            return None
        if is_classifier_dynamic_media_path(media_path):
            planned_slots, sample_iter = FrameCache.stream_frame_samples(
                media_path,
                sample_ratio=self.dynamic_content_sample_ratio,
                detect_pseudostatic=self.use_pseudostatic_dynamic_media,
            )
            if planned_slots > 0:
                stats = FrameCache.get_dynamic_media_stats(media_path) if config.debug else None
                lookahead_eval_cache = {}
                positive_count = 0
                required_positive_count = math.ceil(
                    planned_slots * self.dynamic_content_positive_ratio
                )
                processed_samples = 0
                last_processed_index = -1
                threshold_met = False
                resolved_match_category: Optional[str] = None
                reached_last_sample = False
                try:
                    for idx, sampled_path in enumerate(sample_iter):
                        try:
                            processed_samples += 1
                            last_processed_index = idx
                            is_match, matched_category = self._evaluate_image_path_match(
                                sampled_path, lookahead_eval_cache=lookahead_eval_cache
                            )
                            if is_match:
                                positive_count += 1
                                if matched_category:
                                    resolved_match_category = matched_category
                                # Early success once the positive threshold is met.
                                if positive_count >= required_positive_count:
                                    threshold_met = True
                                    break
                            # Early failure if even all remaining planned slots cannot meet threshold.
                            remaining_samples = planned_slots - (idx + 1)
                            if positive_count + remaining_samples < required_positive_count:
                                break
                        except Exception as e:
                            logger.debug(
                                f"Sample frame prevalidation failed for {sampled_path}: {e}"
                            )
                    else:
                        # No break: consumed all yielded samples (may be fewer than planned_slots).
                        reached_last_sample = True
                finally:
                    close_m = getattr(sample_iter, "close", None)
                    if callable(close_m):
                        close_m()
                if config.debug:
                    media_type = stats.media_type if stats else "dynamic"
                    total_items = stats.total_items if stats else None
                    duration_seconds = stats.duration_seconds if stats else None
                    logger.debug(
                        "Dynamic prevalidation summary | action=%s media=%s type=%s "
                        "sample_idx=%s/%s tested=%s positives=%s required=%s met=%s reached_last=%s "
                        "total_items=%s duration_s=%s",
                        self.name,
                        media_path,
                        media_type,
                        (last_processed_index + 1) if last_processed_index >= 0 else 0,
                        planned_slots,
                        processed_samples,
                        positive_count,
                        required_positive_count,
                        threshold_met,
                        reached_last_sample,
                        total_items,
                        f"{duration_seconds:.2f}" if isinstance(duration_seconds, (int, float)) else "n/a",
                    )
                pseudostatic_match = (
                    self.use_pseudostatic_dynamic_media
                    and FrameCache.is_pseudostatic_dynamic_media(media_path)
                )
                if threshold_met or pseudostatic_match:
                    return self.run_action(
                        media_path,
                        hide_callback,
                        notify_callback,
                        add_mark_callback,
                        blur_callback=blur_callback,
                        base_directory=base_directory or os.path.dirname(media_path),
                        resolved_category=resolved_match_category,
                    )
                return None

        try:
            return self.run_on_image_path(
                media_path,
                hide_callback,
                notify_callback,
                add_mark_callback,
                blur_callback=blur_callback,
                base_directory=base_directory or os.path.dirname(media_path),
            )
        except Exception as e:
            # For non-image media, fall back to first extracted frame.
            # TODO: Extend dynamic multi-sample handling to additional media
            # types (HTML/audio and others) instead of first-frame fallback.
            actual_image_path = FrameCache.get_image_path(media_path)
            if actual_image_path == media_path:
                raise e
            return self.run_on_image_path(
                actual_image_path,
                hide_callback,
                notify_callback,
                add_mark_callback,
                blur_callback=blur_callback,
                base_directory=base_directory or os.path.dirname(media_path),
            )

    def run_on_image_path(
        self,
        image_path,
        hide_callback,
        notify_callback,
        add_mark_callback=None,
        blur_callback=None,
        base_directory: Optional[str] = None,
    ) -> Optional[ClassifierActionType]:
        if not self.can_run:
            return None
        is_match, matched_category = self._evaluate_image_path_match(image_path)
        if is_match:
            return self.run_action(
                image_path,
                hide_callback,
                notify_callback,
                add_mark_callback,
                blur_callback=blur_callback,
                base_directory=base_directory,
                resolved_category=matched_category,
            )
        return None

    def _resolve_classifier_target_category(self, image_path: str) -> Optional[str]:
        if not self.use_image_classifier:
            return None
        if self.image_classifier is None and self.image_classifier_name:
            self.ensure_image_classifier_loaded(None)
        if self.image_classifier is None:
            return None
        predicted_category = self.image_classifier.classify_image(image_path)
        if predicted_category is None:
            return None
        if self.classification_mode == ImageClassifierClassificationMode.MODEL_STRATEGY:
            positive_categories = self._resolve_model_strategy_positive_categories()
            return predicted_category if predicted_category in positive_categories else None
        if self.image_classifier_selected_categories:
            return (
                predicted_category
                if predicted_category in self.image_classifier_selected_categories
                else None
            )
        return None

    def _resolve_action_target_directory(
        self,
        image_path: str,
        base_directory: Optional[str] = None,
        resolved_category: Optional[str] = None,
    ) -> Optional[str]:
        if self.action_modifier:
            return self.action_modifier
        if not self.is_move_action():
            return None
        category = resolved_category or self._resolve_classifier_target_category(image_path)
        if not category:
            return None
        resolved_base_dir = (base_directory or os.path.dirname(image_path) or "").strip()
        if not resolved_base_dir:
            return None
        return os.path.join(resolved_base_dir, category)

    def run_action(
        self,
        image_path,
        hide_callback,
        notify_callback,
        add_mark_callback=None,
        blur_callback=None,
        base_directory: Optional[str] = None,
        resolved_category: Optional[str] = None,
    ):
        base_message = self.name + _(" detected")
        if self.action == ClassifierActionType.SKIP:
            notify_callback("\n" + base_message + _(" - skipped"), base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
        elif self.action == ClassifierActionType.HIDE:
            hide_callback(image_path)
            notify_callback("\n" + base_message + _(" - hidden"), base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
        elif self.action == ClassifierActionType.NOTIFY:
            notify_callback("\n" + base_message, base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
        elif self.action == ClassifierActionType.ADD_MARK:
            add_mark_callback(image_path)
            notify_callback("\n" + base_message + _(" - marked"), base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
        elif self.action == ClassifierActionType.MOVE or self.action == ClassifierActionType.COPY:
            target_directory = self._resolve_action_target_directory(
                image_path,
                base_directory=base_directory,
                resolved_category=resolved_category,
            )
            if target_directory is not None and len(target_directory) > 0:
                if not os.path.exists(target_directory):
                    try:
                        os.makedirs(target_directory, exist_ok=True)
                        logger.info(
                            f"Created missing action target directory for classifier action "
                            f"{self.name}: {target_directory}"
                        )
                    except Exception as e:
                        raise Exception(
                            "Invalid move target directory for classifier action "
                            + self.name + ": " + target_directory + f" ({e})"
                        )
                already_at_target = (
                    os.path.normpath(os.path.dirname(image_path))
                    == os.path.normpath(target_directory)
                )
                if not already_at_target:
                    action_modifier_name = Utils.get_relative_dirpath(target_directory, levels=2)
                    action_type = ActionType.MOVE_FILE if self.action == ClassifierActionType.MOVE else ActionType.COPY_FILE
                    specific_message = _("Moving file: ") + os.path.basename(image_path) + " -> " + action_modifier_name
                    notify_callback("\n" + specific_message, base_message=base_message,
                                    action_type=action_type, is_manual=False)
                    try:
                        FileAction.add_file_action(
                            Utils.move_file if self.action == ClassifierActionType.MOVE else Utils.copy_file,
                            image_path, target_directory
                        )
                    except Exception as e:
                        if (self.action == ClassifierActionType.MOVE and
                            "File already exists:" in str(e) and
                            os.path.exists(image_path)):
                            target_path = os.path.join(target_directory, os.path.basename(image_path))
                            if Utils.calculate_hash(image_path) == Utils.calculate_hash(target_path):
                                # The file already exists in target, so we need to remove it from the source
                                # NOTE: this is a hack to avoid an error that sometimes happens where a file gets stranded
                                # possibly due to the sd-runner application re-saving it after the move, but it could 
                                # technically happen for other more valid reasons. Ideally need to identify why this error
                                # occurs and fix it.
                                try:
                                    with Utils.file_operation_lock:
                                        os.remove(image_path)
                                        logger.info("Removed file from source: " + image_path)
                                except Exception as e:
                                    logger.error("Error removing file from source: " + image_path + ": " + str(e))
                            elif ImageOps.compare_image_content_without_exif(image_path, target_path):
                                # Hash comparison failed, but image content is identical
                                # (different EXIF data but same visual content)
                                logger.info(f"File hashes differ but image content matches: {image_path} <> {target_path}")
                                logger.info("Replacing target file with source file (source has more EXIF data)")
                                try:
                                    # Replace target with source file (source has more information)
                                    FileAction.add_file_action(
                                        Utils.move_file if self.action == ClassifierActionType.MOVE else Utils.copy_file,
                                        image_path, target_directory, auto=True, overwrite_existing=True
                                    )
                                    logger.info("Replaced target file with source: " + image_path)
                                except Exception as e:
                                    logger.error("Error replacing target file with source: " + image_path + ": " + str(e))
                            else:
                                logger.error(e)
                        else:
                            logger.error(e)
            else:
                raise Exception("Target directory not defined on classifier action "  + self.name)
        elif self.action == ClassifierActionType.DELETE:
            notify_callback("\n" + _("Deleting file: ") + os.path.basename(image_path), base_message=base_message,
                            action_type=ActionType.REMOVE_FILE, is_manual=False)
            try:
                with Utils.file_operation_lock:
                    os.remove(image_path)
                    logger.info("Deleted file at " + image_path)
            except Exception as e:
                logger.error("Error deleting file at " + image_path + ": " + str(e))
        elif self.action == ClassifierActionType.BLUR:
            notify_callback("\n" + base_message + _(" - blurred"), base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
            if blur_callback is not None:
                blur_callback(image_path)
        return self.action

    def get_negatives_str(self):
        if len(self.negatives) > 0:
            out = ""
            for i in range(len(self.negatives)):
                out += self.negatives[i].replace(",", "\\,") + ", " if i < len(self.negatives)-1 else self.negatives[i]
            return out
        else:
            return ClassifierAction.NO_NEGATIVES_STR

    def validate(self):
        if self.name is None or len(self.name) == 0:
            raise Exception('Classifier action name is None or empty')
        
        # Check if at least one validation type is enabled
        if not (
            self.use_embedding
            or self.use_image_classifier
            or self.use_prompts
            or self.use_blacklist
            or self.use_prototype
            or self.use_pseudostatic_dynamic_media
        ):
            raise Exception(
                "At least one validation type (embedding, image classifier, prompts, "
                "prompts blacklist, prototype, or pseudo-static dynamic media) must be enabled."
            )
        
        # Validate prototype settings if enabled
        if self.use_prototype:
            if not self.prototype_directory or not self.prototype_directory.strip():
                raise Exception("Prototype directory must be set when using prototype validation.")
            if not os.path.isdir(self.prototype_directory):
                raise Exception(f"Prototype directory does not exist: {self.prototype_directory}")
            # Validate negative prototype directory if set
            if self.negative_prototype_directory and self.negative_prototype_directory.strip():
                if not os.path.isdir(self.negative_prototype_directory):
                    raise Exception(f"Negative prototype directory does not exist: {self.negative_prototype_directory}")
        
        # Check if positives/negatives are set when needed
        if (self.use_embedding or self.use_prompts) and (self.positives is None or len(self.positives) == 0) and \
                (self.negatives is None or len(self.negatives) == 0):
            raise Exception("At least one of positive or negative texts must be set when using embedding or prompt validation.")
        
        # Validate image classifier settings if enabled (registry only — no lazy load here;
        # wrappers load during prevalidation / matches_image_path via ensure_image_classifier_loaded).
        if self.use_image_classifier:
            name = (self.image_classifier_name or "").strip()
            if name:
                resolved = image_classifier_manager.resolve_registered_model_name(self.image_classifier_name)
                if resolved is None:
                    keys = list(image_classifier_manager.classifier_metadata.keys())
                    raise Exception(
                        f"The image classifier \"{self.image_classifier_name}\" was not found in the available image classifiers. "
                        f"Registered model_name keys: {keys}. "
                        f"If this is a Hugging Face repo id, ensure the model config sets hf_repo_id to match."
                    )
                model_cfg = image_classifier_manager.classifier_metadata[resolved]
                allowed = set(model_cfg.model_categories)
                if self.image_classifier_selected_categories:
                    bad = [c for c in self.image_classifier_selected_categories if c not in allowed]
                    if bad:
                        raise Exception(
                            f"One or more selected categories {bad} were not found in the image classifier's "
                            f"category options (model \"{resolved}\": {list(model_cfg.model_categories)})"
                        )
            if self.classification_mode == ImageClassifierClassificationMode.MODEL_STRATEGY:
                self._resolve_model_strategy_positive_categories()
        
        if self.is_move_action():
            if self.action_modifier and not os.path.isdir(self.action_modifier):
                raise Exception('Action modifier must be a valid directory')
            if not self.action_modifier and not self.use_image_classifier:
                raise Exception('Action modifier must be set when image classifier is disabled')
            if (
                not self.action_modifier
                and self.use_image_classifier
                and self.classification_mode == ImageClassifierClassificationMode.SELECTED_CATEGORIES
                and not self.image_classifier_selected_categories
            ):
                raise Exception(
                    "Action modifier may be empty only when image classifier selected categories are provided."
                )

    def validate_dirs(self):
        errors = []
        if self.action_modifier and self.action_modifier != "" and not os.path.isdir(self.action_modifier):
            errors.append(_("Action modifier is not a valid directory: ") + self.action_modifier)
        if len(errors) > 0:
            logger.error(_("Invalid classifier action {0}, may cause errors or be unable to run!").format(self.name))
            for error in errors:
                logger.warning(error)

    def is_move_action(self):
        return self.action == ClassifierActionType.MOVE or self.action == ClassifierActionType.COPY

    def move_index(self, idx, direction_count=1):
        """Move a classifier action in the list by the specified number of positions.
        
        Args:
            idx: Current index of the classifier action to move
            direction_count: Positive to move down (higher index), negative to move up (lower index)
        """
        from compare.classifier_actions_manager import ClassifierActionsManager

        classifier_actions = ClassifierActionsManager.classifier_actions
        ClassifierAction.do_move_index(idx, classifier_actions, direction_count)
    
    @staticmethod
    def do_move_index(idx, classifier_actions, direction_count=1):
        list_len = len(classifier_actions)
        if list_len <= 1:
            return  # Nothing to move
        
        # Calculate target index with wrapping
        target_idx = (idx + direction_count) % list_len
        if target_idx < 0:
            target_idx += list_len
        
        # If target is the same as current, no move needed
        if target_idx == idx:
            return
        
        # Simple approach: remove the item and insert it at the target position
        move_item = classifier_actions.pop(idx)
        classifier_actions.insert(target_idx, move_item)

    def to_dict(self):
        return {
            "name": self.name,
            "positives": self.positives,
            "negatives": self.negatives,
            "threshold": self.text_embedding_threshold,  # Keep for backward compatibility
            "text_embedding_threshold": self.text_embedding_threshold,
            "prototype_threshold": self.prototype_threshold,
            "action": self.action.value,
            "action_modifier": self.action_modifier,
            "is_active": self.is_active,
            "image_classifier_name": self.image_classifier_name,
            "image_classifier_selected_categories": self.image_classifier_selected_categories,
            "classification_mode": self.classification_mode.value,
            "use_embedding": self.use_embedding,
            "use_image_classifier": self.use_image_classifier,
            "use_prompts": self.use_prompts,
            "use_blacklist": self.use_blacklist,
            "use_pseudostatic_dynamic_media": self.use_pseudostatic_dynamic_media,
            "use_prototype": self.use_prototype,
            "prototype_directory": self.prototype_directory,
            "negative_prototype_directory": self.negative_prototype_directory,
            "negative_prototype_lambda": self.negative_prototype_lambda,
            "dynamic_content_sample_ratio": self.dynamic_content_sample_ratio,
            "dynamic_content_positive_ratio": self.dynamic_content_positive_ratio,
            "_last_used_profile": self._last_used_profile,
            "lookahead_names": self.lookahead_names,
            }

    @staticmethod
    def from_dict(d):
        # Handle backward compatibility - detect original type based on data presence
        if 'use_embedding' not in d:
            # If image_classifier_name is set, it was an image classifier action
            if 'image_classifier_name' in d and d['image_classifier_name'] and d['image_classifier_name'].strip():
                d['use_embedding'] = False
                d['use_image_classifier'] = True
            else:
                # Otherwise it was an embedding action
                d['use_embedding'] = True
                d['use_image_classifier'] = False
        if 'use_image_classifier' not in d:
            d['use_image_classifier'] = False
        if 'classification_mode' not in d:
            d['classification_mode'] = ImageClassifierClassificationMode.SELECTED_CATEGORIES.value
        if 'lookahead_names' not in d:
            d['lookahead_names'] = []
        if 'use_prompts' not in d:
            d['use_prompts'] = False
        if 'use_blacklist' not in d:
            d['use_blacklist'] = False
        if 'use_pseudostatic_dynamic_media' not in d:
            d['use_pseudostatic_dynamic_media'] = False
        if 'is_active' not in d:
            d['is_active'] = True
        if 'use_prototype' not in d:
            d['use_prototype'] = False
        if 'prototype_directory' not in d:
            d['prototype_directory'] = ""
        if 'negative_prototype_directory' not in d:
            d['negative_prototype_directory'] = ""
        if 'negative_prototype_lambda' not in d:
            d['negative_prototype_lambda'] = 0.5
        if 'dynamic_content_sample_ratio' not in d:
            d['dynamic_content_sample_ratio'] = 0.1
        if 'dynamic_content_positive_ratio' not in d:
            d['dynamic_content_positive_ratio'] = 0.1
        if '_last_used_profile' not in d:
            d['_last_used_profile'] = None
        # Handle threshold backward compatibility
        if 'text_embedding_threshold' not in d:
            # Use existing threshold as text_embedding_threshold
            d['text_embedding_threshold'] = d.get('threshold', 0.23)
        if 'prototype_threshold' not in d:
            # Use existing threshold as prototype_threshold for backward compatibility
            d['prototype_threshold'] = d.get('threshold', 0.23)
        d.pop("can_run", None)
        d.pop("initialization_error", None)

        return ClassifierAction(**d)

    def _resolve_model_strategy_positive_categories(self) -> frozenset[str]:
        name = (self.image_classifier_name or "").strip()
        if not name:
            raise Exception(
                "Image classifier name must be set when using model strategy classification mode."
            )
        resolved = image_classifier_manager.resolve_registered_model_name(name)
        cfg = (
            image_classifier_manager.classifier_metadata.get(resolved)
            if resolved
            else None
        )
        groups_sig = tuple(
            tuple(g)
            for g in ((cfg.positive_groups if cfg else None) or [])
            if isinstance(g, list) and g
        )
        sig = (resolved, groups_sig)
        if sig == self._model_strategy_sig and self._model_strategy_positives is not None:
            return self._model_strategy_positives

        if cfg is None:
            raise Exception(
                f'Image classifier model config "{name}" is invalid or unavailable; '
                "objects depending on this model strategy are invalid."
            )
        if not groups_sig:
            raise Exception(
                f'Image classifier model config "{name}" must define positive_groups '
                "for model strategy classification mode."
            )
        positives = frozenset(c for grp in groups_sig for c in grp)
        if not positives:
            raise Exception(
                f'Image classifier model config "{name}" has no usable positive categories.'
            )
        self._model_strategy_sig = sig
        self._model_strategy_positives = positives
        return positives

    def __str__(self) -> str:
        out = self.name
        validation_types = []
        if self.use_embedding:
            validation_types.append(_("embedding"))
        if self.use_image_classifier and self.image_classifier_name and self.image_classifier_name.strip():
            validation_types.append(_("classifier {0}").format(self.image_classifier_name))
        if self.use_prompts:
            validation_types.append(_("prompts"))
        if self.use_prototype:
            validation_types.append(_("prototype"))
        if self.use_pseudostatic_dynamic_media:
            validation_types.append(_("pseudo-static dynamic media"))

        if validation_types:
            # Build the description parts
            description_parts = []
            
            # Add categories if image classifier is enabled and has categories
            if self.use_image_classifier and self.image_classifier_selected_categories:
                description_parts.append(_("categories: {0}").format(", ".join(self.image_classifier_selected_categories)))
            
            # Add positives/negatives if any are set
            if self.positives or self.negatives:
                description_parts.append(_("{0} positives, {1} negatives").format(len(self.positives), len(self.negatives)))
            
            # Combine all parts
            if description_parts:
                out += _(" using {0} ({1})").format(", ".join(validation_types), "; ".join(description_parts))
            else:
                out += _(" using {0}").format(", ".join(validation_types))
        else:
            out += _(" ({0} positives, {1} negatives)").format(len(self.positives), len(self.negatives))

        if not self.is_active:
            out += " [" + _("disabled") + "]"
        if not self.can_run:
            out += " [" + _("cannot run") + "]"

        return out



class Prevalidation(ClassifierAction):
    def __init__(self, name=_("New Prevalidation"), positives=[], negatives=[], threshold=0.23,
                 text_embedding_threshold=None, prototype_threshold=0.23,
                 action=ClassifierActionType.NOTIFY, action_modifier="", run_on_folder=None, is_active=True,
                 image_classifier_name="", image_classifier_selected_categories=[], 
                 classification_mode=ImageClassifierClassificationMode.SELECTED_CATEGORIES,
                 use_embedding=True, use_image_classifier=False, use_prompts=False, use_blacklist=False,
                 use_pseudostatic_dynamic_media=False,
                 lookahead_names=[], profile_name=None, use_prototype=False, prototype_directory="",
                 negative_prototype_directory="", negative_prototype_lambda=0.5,
                 dynamic_content_sample_ratio=0.1, dynamic_content_positive_ratio=0.1,
                 _last_used_profile=None, can_run: bool = True, initialization_error: Optional[str] = None):
        # Pass all parameters including prototype settings to parent ClassifierAction
        super().__init__(name, positives, negatives, threshold,
                        text_embedding_threshold, prototype_threshold, action, action_modifier,
                        image_classifier_name, image_classifier_selected_categories, classification_mode,
                        use_embedding, use_image_classifier, use_prompts, use_blacklist,
                        use_pseudostatic_dynamic_media,
                        is_active, use_prototype, prototype_directory,
                        negative_prototype_directory, negative_prototype_lambda,
                        dynamic_content_sample_ratio, dynamic_content_positive_ratio,
                        _last_used_profile, lookahead_names=lookahead_names,
                        can_run=can_run, initialization_error=initialization_error)
        self.profile_name = profile_name  # Name of DirectoryProfile to use (None = global)
        self.profile = None  # Cached DirectoryProfile instance (set after loading, or temporary for backward compatibility)
        # Note: run_on_folder parameter is kept for backward compatibility in from_dict but not stored as instance variable

    def __eq__(self, other):
        """Check equality based on name (prevalidations are uniquely identified by name)."""
        if not isinstance(other, Prevalidation):
            return False
        return self.name == other.name

    def __hash__(self):
        """Hash based on name (prevalidations are uniquely identified by name)."""
        return hash(self.name)

    def update_profile_instance(self, profile_name=None, directory_path=None):
        """
        Update the cached profile instance based on profile_name.
        
        Args:
            profile_name: Optional profile name to set. If provided, updates self.profile_name first,
                         then updates the cached profile instance. If None, uses existing self.profile_name.
            auto_create: If True and profile doesn't exist, create it (for backward compatibility).
            directory_path: Directory path to use when creating profile (required if auto_create=True).
        """
        if profile_name is not None:
            self.profile_name = profile_name
        
        if self.profile_name:
            # Profile may be cached
            self.profile = DirectoryProfile.get_profile_by_name(self.profile_name)
            
            if self.profile is None:
                if directory_path:
                    # Create DirectoryProfile for backward compatibility
                    if not os.path.isdir(directory_path):
                        logger.warning(f"Invalid directory in run_on_folder for prevalidation '{self.name}': {directory_path}. Skipping profile creation.")
                        self.profile = None
                        self.profile_name = None
                        return
                    
                    # Use the directory path as the name to ensure uniqueness
                    temp_profile = DirectoryProfile(name=self.profile_name, directories=[directory_path])
                    # Add to profiles list so it can be reused by other prevalidations
                    DirectoryProfile.directory_profiles.append(temp_profile)
                    logger.info(f"Created temporary DirectoryProfile for backward compatibility: name='{self.profile_name}', prevalidation='{self.name}'")
                    self.profile = temp_profile
                else:
                    # Profile not found and not creating
                    logger.warning(f"Profile {self.profile_name} not found for prevalidation {self.name}")
                    self.profile = None
        else:
            self.profile = None

    def run_on_image_path(
        self,
        image_path,
        hide_callback,
        notify_callback,
        add_mark_callback=None,
        blur_callback=None,
        base_directory: Optional[str] = None,
    ) -> Optional[ClassifierActionType]:
        # Lazy load the image classifier if needed
        super().ensure_image_classifier_loaded(notify_callback)
        return super().run_on_image_path(
            image_path,
            hide_callback,
            notify_callback,
            add_mark_callback,
            blur_callback=blur_callback,
            base_directory=base_directory,
        )

    def run_on_media_path(
        self,
        media_path,
        hide_callback,
        notify_callback,
        add_mark_callback=None,
        blur_callback=None,
        base_directory: Optional[str] = None,
    ) -> Optional[ClassifierActionType]:
        # Keep lazy image-classifier loading behavior for sampled media paths too.
        super().ensure_image_classifier_loaded(notify_callback)
        return super().run_on_media_path(
            media_path,
            hide_callback,
            notify_callback,
            add_mark_callback,
            blur_callback=blur_callback,
            base_directory=base_directory,
        )

    def validate_dirs(self):
        super().validate_dirs()
        # Add prevalidation-specific profile directory validation
        errors = []
        if self.profile is not None:
            for directory in self.profile.directories:
                if not os.path.isdir(directory):
                    errors.append(_("Profile directory is not a valid directory: ") + directory)
        if len(errors) > 0:
            logger.error(_("Invalid prevalidation {0}, may cause errors or be unable to run!").format(self.name))
            for error in errors:
                logger.warning(error)

    def move_index(self, idx, direction_count=1):
        """Move a prevalidation in the list by the specified number of positions.
        
        Args:
            idx: Current index of the prevalidation to move
            direction_count: Positive to move down (higher index), negative to move up (lower index)
        """
        from compare.classifier_actions_manager import ClassifierActionsManager

        prevalidations = ClassifierActionsManager.prevalidations
        ClassifierAction.do_move_index(idx, prevalidations, direction_count)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "profile_name": self.profile_name,
            # is_active is already in parent's dict, no need to duplicate
            # Prototype properties (use_prototype, prototype_directory, negative_prototype_directory, 
            # negative_prototype_lambda) are already in parent's dict, no need to duplicate
            "lookahead_names": self.lookahead_names,
        })
        return d

    @staticmethod
    def from_dict(d):
        # Handle backward compatibility - detect original type based on data presence
        if 'use_embedding' not in d:
            # If image_classifier_name is set, it was an image classifier prevalidation
            if 'image_classifier_name' in d and d['image_classifier_name'] and d['image_classifier_name'].strip():
                d['use_embedding'] = False
                d['use_image_classifier'] = True
            else:
                # Otherwise it was an embedding prevalidation
                d['use_embedding'] = True
                d['use_image_classifier'] = False
        if 'use_image_classifier' not in d:
            d['use_image_classifier'] = False
        if 'classification_mode' not in d:
            d['classification_mode'] = ImageClassifierClassificationMode.SELECTED_CATEGORIES.value
        if 'use_prompts' not in d:
            d['use_prompts'] = False
        if 'use_blacklist' not in d:
            d['use_blacklist'] = False
        if 'use_pseudostatic_dynamic_media' not in d:
            d['use_pseudostatic_dynamic_media'] = False
        if 'use_prototype' not in d:
            d['use_prototype'] = False
        if 'prototype_directory' not in d:
            d['prototype_directory'] = ""
        if 'negative_prototype_directory' not in d:
            d['negative_prototype_directory'] = ""
        if 'negative_prototype_lambda' not in d:
            d['negative_prototype_lambda'] = 0.5
        if 'dynamic_content_sample_ratio' not in d:
            d['dynamic_content_sample_ratio'] = 0.1
        if 'dynamic_content_positive_ratio' not in d:
            d['dynamic_content_positive_ratio'] = 0.1
        if 'lookahead_names' not in d:
            d['lookahead_names'] = []
        if 'profile_name' not in d:
            d['profile_name'] = None
        if 'is_active' not in d:
            d['is_active'] = True  # Default to active for prevalidations
        # Handle threshold backward compatibility
        if 'text_embedding_threshold' not in d:
            # Use existing threshold as text_embedding_threshold
            d['text_embedding_threshold'] = d.get('threshold', 0.23)
        if 'prototype_threshold' not in d:
            # Use existing threshold as prototype_threshold for backward compatibility
            d['prototype_threshold'] = d.get('threshold', 0.23)
        d.pop("can_run", None)
        d.pop("initialization_error", None)

        # Handle backward compatibility: if run_on_folder exists but no profile_name, create temporary profile
        run_on_folder = d.get('run_on_folder')
        if run_on_folder and not d.get('profile_name'):
            pv = Prevalidation(**d)
            # Use update_profile_instance to handle profile lookup/creation
            pv.update_profile_instance(profile_name=run_on_folder, directory_path=run_on_folder)
            return pv
        
        return Prevalidation(**d)

    def __str__(self) -> str:
        # Use parent's __str__ implementation and append lookahead info
        out = super().__str__()
        
        if self.lookahead_names:
            out += " <" + _("lookaheads: {0}").format(", ".join(self.lookahead_names)) + ">"
        
        return out


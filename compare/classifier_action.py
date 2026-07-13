"""
Classifier action and prevalidation models (serialization, validation, run logic).

List storage and directory/prevalidation cache live in
``compare.classifier_actions_manager.ClassifierActionsManager``.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import os
from typing import Callable, ClassVar, Optional

from compare.action_callbacks import ActionCallbacks
from compare.compare_embeddings_clip import CompareEmbeddingClip
from compare.embedding_prototype import EmbeddingPrototype
from compare.lookahead import Lookahead
from files.directory_profile import DirectoryProfile
from files.file_action import FileAction
from image.image_classifier_manager import image_classifier_manager
from image.image_data_extractor import image_data_extractor
from image.frame_cache import FrameCache
from image.image_ops import ImageOps
from image.video_ops import VideoOps
from files.related_image import extract_filename_base_stem, find_files_by_base_stem
from utils.config import config
from utils.media_utils import (
    get_media_type_for_path,
    is_classifier_dynamic_media_path,
    parse_rotation_label,
    resolve_rendered_frame_source,
    rotated_sibling_output_path,
)
from utils.constants import ActionType, ClassifierActionType, CompareMediaType
from utils.logging_setup import get_logger
from utils.running_tasks_registry import start_thread
from utils.translations import _
from utils.utils import Utils


logger = get_logger("classifier_action")


@dataclass
class TriggerDetail:
    """What caused find_first_trigger_slot() to return a positive result."""
    trigger_type: str                              # "image_classifier", "embedding", "prompt", "prototype", "filename"
    category: Optional[str] = None                # matched category (image classifier only)
    top_predictions: Optional[list] = None        # [(category, score), ...] ranked highest first


@dataclass
class TriggerFrameResult:
    """Position of the first matching frame found by find_first_trigger_slot()."""
    slot_index: int               # 0-based index into the planned sample slots
    total_planned_slots: int      # planned_slots returned by stream_frame_samples
    frame_path: str               # absolute path to the matching frame file
    detail: Optional["TriggerDetail"] = None      # what triggered (populated for manual seeks)


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





@dataclass(eq=False, repr=False)
class ClassifierAction:
    NO_POSITIVES_STR: ClassVar[str] = _("(no positives set)")
    NO_NEGATIVES_STR: ClassVar[str] = _("(no negatives set)")

    # ------------------------------------------------------------------
    # Serializable fields — all appear in the generated __init__
    # ------------------------------------------------------------------
    name: str = field(default_factory=lambda: _("New Classifier Action"))
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)
    threshold: float = 0.23                    # backward-compat alias; resolved in __post_init__
    text_embedding_threshold: Optional[float] = None
    prototype_threshold: float = 0.23
    action: ClassifierActionType = field(default=ClassifierActionType.NOTIFY)
    action_modifier: str = ""
    related_image_edit_suffix: str = ""
    image_classifier_name: str = ""
    image_classifier_selected_categories: list = field(default_factory=list)
    classification_mode: ImageClassifierClassificationMode = field(
        default=ImageClassifierClassificationMode.SELECTED_CATEGORIES
    )
    use_embedding: bool = True
    use_image_classifier: bool = False
    use_prompts: bool = False
    use_blacklist: bool = False
    use_pseudostatic_dynamic_media: bool = False
    move_to_same_dir: bool = False
    is_active: bool = True
    use_prototype: bool = False
    prototype_directory: str = ""
    negative_prototype_directory: str = ""
    negative_prototype_lambda: float = 0.5
    dynamic_content_sample_ratio: float = 0.1
    dynamic_content_positive_ratio: float = 0.1
    _last_used_profile: Optional[str] = None
    lookahead_names: list = field(default_factory=list)
    use_filename_contains: bool = False
    filename_contains_patterns: Optional[list] = None
    filename_contains_case_sensitive: bool = False
    use_base_stem_match: bool = False
    base_stem_match_require_match: bool = True
    applies_to_media_types: Optional[list] = None
    can_run: bool = True
    initialization_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Runtime-only state — excluded from __init__, never serialized
    # ------------------------------------------------------------------
    image_classifier: object = field(init=False, default=None)
    _missing_image_classifier_logged: bool = field(init=False, default=False)
    image_classifier_categories: list = field(init=False, default_factory=list)
    _cached_prototype: object = field(init=False, default=None)
    _cached_negative_prototype: object = field(init=False, default=None)
    _blocked_run_logged: bool = field(init=False, default=False)
    _model_strategy_sig: Optional[tuple] = field(init=False, default=None)
    _model_strategy_positives: Optional[frozenset] = field(init=False, default=None)

    def __post_init__(self):
        # Backward compat: if text_embedding_threshold is None, fall back to threshold
        if self.text_embedding_threshold is None:
            self.text_embedding_threshold = self.threshold
        self.threshold = self.text_embedding_threshold
        # Normalize action enum
        if not isinstance(self.action, Enum):
            self.action = ClassifierActionType[self.action]
        # Normalize classification mode
        self.classification_mode = ImageClassifierClassificationMode.from_value(self.classification_mode)
        # Normalize list fields that callers may pass as None
        self.lookahead_names = list(self.lookahead_names) if self.lookahead_names else []
        self.filename_contains_patterns = list(self.filename_contains_patterns) if self.filename_contains_patterns else []
        # Normalize ratios
        self.dynamic_content_sample_ratio = self._normalize_ratio(self.dynamic_content_sample_ratio, default_val=0.1)
        self.dynamic_content_positive_ratio = self._normalize_ratio(self.dynamic_content_positive_ratio, default_val=0.1)
        # Coerce bool fields that may arrive as ints/strings from JSON
        self.use_pseudostatic_dynamic_media = bool(self.use_pseudostatic_dynamic_media)
        self.can_run = bool(self.can_run)
        # Normalize applies_to_media_types: None means all types; otherwise coerce each
        # element to CompareMediaType (accepting enum instances or their string values).
        if self.applies_to_media_types is not None:
            coerced = [
                mt if isinstance(mt, CompareMediaType) else CompareMediaType(mt)
                for mt in self.applies_to_media_types
            ]
            self.applies_to_media_types = coerced if coerced else None

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

    def _check_filename_contains(self, image_path: str) -> bool:
        if not self.filename_contains_patterns:
            return False
        filename = os.path.basename(image_path)
        if not self.filename_contains_case_sensitive:
            filename = filename.lower()
        for pattern in self.filename_contains_patterns:
            p = pattern if self.filename_contains_case_sensitive else pattern.lower()
            if p and p in filename:
                return True
        return False

    def _check_base_stem_match(self, image_path: str) -> bool:
        base_stem = extract_filename_base_stem(image_path)
        if not base_stem:
            return False
        dirs = config.directories_to_search_for_related_images
        if not dirs:
            return False
        found = bool(find_files_by_base_stem(dirs, base_stem, use_cache=True))
        return found if self.base_stem_match_require_match else not found

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
    
    def _run_with_batch_prototype_validation(
        self,
        directory_paths: list[str],
        callbacks: ActionCallbacks,
        max_images_per_batch: Optional[int] = None,
        on_complete: Optional[Callable[[dict], None]] = None,
    ):
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
            on_complete: Optional callback invoked once with a summary dict when the
                job finishes (or fails outright), e.g. to show a completion toast.
        """
        if not self.use_prototype or self._cached_prototype is None:
            return

        def batch_validation_worker():
            """Worker function to run batch validation and actions in a separate thread."""
            outcomes = 0
            errors = 0
            matching_paths = []
            try:
                # Use EmbeddingPrototype to batch validate images from directories
                matching_paths = EmbeddingPrototype.batch_validate_with_prototypes(
                    directories=directory_paths,
                    positive_prototype=self._cached_prototype,
                    threshold=self.prototype_threshold,
                    negative_prototype=self._cached_negative_prototype,
                    negative_lambda=self.negative_prototype_lambda,
                    notify_callback=callbacks.notify_callback,
                    max_images_per_batch=max_images_per_batch
                )

                # Run actions on matching images
                for image_path in matching_paths:
                    try:
                        self.run_action(image_path, callbacks)
                        outcomes += 1
                    except Exception as e:
                        errors += 1
                        logger.error(f"Error running action on {image_path}: {e}")
            except Exception as e:
                logger.error(f"Error in batch prototype validation: {e}")
            finally:
                self._invoke_on_complete(
                    on_complete,
                    files_checked=len(matching_paths),
                    outcomes=outcomes,
                    errors=errors,
                )

        # Start batch validation and action execution in a separate thread
        start_thread(batch_validation_worker, use_asyncio=False)

    def _invoke_on_complete(
        self,
        on_complete,
        *,
        files_checked: int,
        outcomes: int,
        errors: int,
    ) -> None:
        """Build and dispatch the job-summary dict every run path finishes with.
        moves/copies/deletes are derived from self.action, since a single
        ClassifierAction always executes the same action type for every match —
        only the target category subdirectory varies per file."""
        if on_complete is None:
            return
        moves = outcomes if self.action == ClassifierActionType.MOVE else 0
        copies = outcomes if self.action == ClassifierActionType.COPY else 0
        deletes = outcomes if self.action == ClassifierActionType.DELETE else 0
        try:
            on_complete({
                "action_name": self.name,
                "files_checked": files_checked,
                "outcomes": outcomes,
                "moves": moves,
                "copies": copies,
                "deletes": deletes,
                "errors": errors,
            })
        except Exception:
            logger.exception("Error in classifier action completion callback for %r", self.name)

    def run(
        self,
        directory_paths: list[str],
        callbacks: ActionCallbacks,
        profile_name_or_path: Optional[str] = None,
        max_images_per_batch: Optional[int] = None,
        media_paths: Optional[list[tuple[str, str]]] = None,
        on_complete: Optional[Callable[[dict], None]] = None,
    ):
        """Run the classifier action on the given directory paths.

        Args:
            directory_paths: List of directory paths to process (used for the
                vectorized prototype-only fast path, and to record the last-used
                profile/directory)
            hide_callback: Callback for hiding images
            notify_callback: Callback for notifications
            add_mark_callback: Optional callback for marking images
            profile_name_or_path: Optional profile name or directory path to store as last used
            max_images_per_batch: Optional maximum number of images to process per batch
            media_paths: Pre-gathered, pre-sorted (media_path, base_directory) pairs,
                required for any validation type other than prototype-only matching.
                base_directory must be the top-level directory the file was
                discovered under (not os.path.dirname(media_path)) — otherwise a
                MOVE/COPY action re-evaluating a file already sitting in its
                category subdirectory from a prior run will nest it again
                (target/category/category) instead of recognizing it's already in
                place. This class does no directory walking itself — callers must
                resolve directory_paths via
                ClassifierActionsManager.gather_sorted_media_paths() first, since
                only that layer knows a directory's cached sort preference.
            on_complete: Optional callback invoked once from the background thread
                when the job finishes, with a summary dict (action_name,
                files_checked, outcomes, moves, copies, deletes, errors) — e.g. to
                show a completion toast.
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
        self.ensure_image_classifier_loaded(callbacks.notify_callback)
        self.ensure_prototype_loaded(callbacks.notify_callback)

        # The vectorized batch path only checks prototype similarity, bypassing
        # _evaluate_image_path_match entirely — correct only when prototype is
        # the sole enabled validation type. Any other enabled type (image
        # classifier, embedding, prompts, filename-contains, base-stem) needs
        # per-file OR-combined evaluation via the media_paths sweep below.
        if self.use_prototype and not self.has_non_prototype_validation():
            self._run_with_batch_prototype_validation(
                directory_paths, callbacks, max_images_per_batch, on_complete=on_complete
            )
        elif media_paths is not None:
            self._run_media_paths_sweep(media_paths, callbacks, on_complete=on_complete)
        else:
            logger.error(
                "Classifier action %r needs media_paths for its enabled validation "
                "type(s); caller must resolve directory_paths via "
                "ClassifierActionsManager.gather_sorted_media_paths() before calling run().",
                self.name,
            )

    def has_non_prototype_validation(self) -> bool:
        return bool(
            self.use_embedding
            or self.use_image_classifier
            or self.use_prompts
            or self.use_filename_contains
            or self.use_base_stem_match
        )

    def _run_media_paths_sweep(
        self,
        media_paths: list[tuple[str, str]],
        callbacks: ActionCallbacks,
        on_complete: Optional[Callable[[dict], None]] = None,
    ):
        """Evaluate/act on each (media_path, base_directory) pair via the full
        OR-combined validation logic in run_on_media_path. Runs in a background
        thread, matching _run_with_batch_prototype_validation's behavior.

        base_directory must be the top-level directory the file was gathered
        under, not its own immediate parent — see run()'s docstring."""
        def sweep_worker():
            outcomes = 0
            errors = 0
            files_checked = 0
            try:
                total = len(media_paths)
                if callbacks.notify_callback:
                    callbacks.notify_callback(_("Evaluating {0} files...").format(total))
                for media_path, base_directory in media_paths:
                    files_checked += 1
                    try:
                        result = self.run_on_media_path(
                            media_path, callbacks, base_directory=base_directory
                        )
                        if result is not None:
                            outcomes += 1
                    except Exception as e:
                        errors += 1
                        logger.error(f"Error running action on {media_path}: {e}")
            except Exception as e:
                logger.error(f"Error in classifier action media sweep: {e}")
            finally:
                self._invoke_on_complete(
                    on_complete,
                    files_checked=files_checked,
                    outcomes=outcomes,
                    errors=errors,
                )

        start_thread(sweep_worker, use_asyncio=False)

    def _evaluate_image_path_match(
        self, image_path: str, lookahead_eval_cache=None, _detail_out: Optional[list] = None
    ) -> tuple[bool, Optional[str]]:
        # Note: Image classifier and prototype should be loaded before calling this method
        # (see ClassifierActionsWindow.run_classifier_action for pre-loading)
        #
        # _detail_out: optional single-element list. When provided, _detail_out[0] is set
        # to a TriggerDetail describing what triggered. Used only for the manual seek flow;
        # all normal callers leave it None.
        if not self.can_run:
            return False, None

        # Check each enabled validation type with short-circuit OR logic
        if self.use_prototype:
            if self._check_prototype_validation(image_path):
                if _detail_out is not None:
                    _detail_out[0] = TriggerDetail(trigger_type="prototype")
                return True, None

        # Check lookaheads first - if any pass, skip this prevalidation
        if self._check_lookaheads(image_path, lookahead_eval_cache=lookahead_eval_cache):
            return False, None

        if self.use_embedding:
            if CompareEmbeddingClip.multi_text_compare(image_path, self.positives, self.negatives, self.text_embedding_threshold):
                if _detail_out is not None:
                    _detail_out[0] = TriggerDetail(trigger_type="embedding")
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
                        if _detail_out is not None:
                            _detail_out[0] = self._build_classifier_detail(image_path, predicted_category)
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
                        if _detail_out is not None:
                            _detail_out[0] = self._build_classifier_detail(image_path, predicted_category)
                        if predicted_category in self.image_classifier_selected_categories:
                            return True, predicted_category
                        return True, None
            else:
                if not self._missing_image_classifier_logged:
                    logger.error(f"Image classifier {self.image_classifier_name} not found for classifier action {self.name}")
                    self._missing_image_classifier_logged = True

        if self.use_prompts:
            if self._check_prompt_validation(image_path):
                if _detail_out is not None:
                    _detail_out[0] = TriggerDetail(trigger_type="prompt")
                return True, None

        if self.use_filename_contains:
            if self._check_filename_contains(image_path):
                if _detail_out is not None:
                    _detail_out[0] = TriggerDetail(trigger_type="filename")
                return True, None

        if self.use_base_stem_match:
            if self._check_base_stem_match(image_path):
                if _detail_out is not None:
                    _detail_out[0] = TriggerDetail(trigger_type="base_stem_match")
                return True, None

        # No validation type passed
        return False, None

    def _build_classifier_detail(self, image_path: str, category: Optional[str]) -> "TriggerDetail":
        """Build a TriggerDetail for an image-classifier match, including ranked predictions.
        predict_image caches internally so this is effectively free after classify_image ran."""
        top_preds = None
        try:
            if self.image_classifier:
                top_preds = self.image_classifier.predict_image_ranked(image_path)
        except Exception:
            pass
        return TriggerDetail(trigger_type="image_classifier", category=category, top_predictions=top_preds)

    def media_type_allowed(self, path: str) -> bool:
        """Return False when applies_to_media_types is set and path's type is not in it."""
        if self.applies_to_media_types is None:
            return True
        return get_media_type_for_path(path) in self.applies_to_media_types

    def find_first_trigger_slot(
        self, media_path: str, start_slot: int = 0, sample_ratio: Optional[float] = None
    ) -> Optional["TriggerFrameResult"]:
        """Scan sampled frames of a dynamic media file and return the position of the
        first matching frame, provided the action's positive-frame threshold is met.

        Mirrors run_on_media_path() sampling and threshold logic but returns seek
        info instead of dispatching. Does not touch any prevalidation cache.
        Returns None if the threshold is not met or the file is not dynamic media.

        start_slot: skip all slots before this index, enabling "next trigger" cycling.
            Call with start_slot = last_result.slot_index + 1 and wrap to 0 on None.

        sample_ratio: when provided, overrides self.dynamic_content_sample_ratio and
            disables the configured max-samples cap, sampling up to every available
            frame. Use 1.0 for interactive seeks where precision matters more than speed.
        """
        if not is_classifier_dynamic_media_path(media_path):
            return None
        if not self.media_type_allowed(media_path):
            return None

        if sample_ratio is not None:
            # Interactive/manual seek: sample densely with no artificial frame-count cap.
            # _compute_sample_indices will still clamp to total_items, preventing runaway.
            planned_slots, sample_iter = FrameCache.stream_frame_samples(
                media_path,
                sample_ratio=sample_ratio,
                detect_pseudostatic=False,
                max_samples=2 ** 31 - 1,
            )
        else:
            planned_slots, sample_iter = FrameCache.stream_frame_samples(
                media_path,
                sample_ratio=self.dynamic_content_sample_ratio,
                detect_pseudostatic=False,
            )
        if planned_slots <= 0:
            return None

        if start_slot > 0:
            # Materialize (almost certainly already cached) and take the sub-range.
            all_frames = list(sample_iter)
            planned_slots = len(all_frames)
            if start_slot >= planned_slots:
                return None
            scan_frames = all_frames[start_slot:]
            slot_offset = start_slot
        else:
            scan_frames = sample_iter   # stay lazy when scanning from the beginning
            slot_offset = 0

        n = len(scan_frames) if hasattr(scan_frames, "__len__") else planned_slots
        required = max(1, math.ceil(n * self.dynamic_content_positive_ratio))
        positive_count = 0
        first_positive: Optional[tuple] = None  # (absolute_slot_index, frame_path)

        for local_idx, frame_path in enumerate(scan_frames):
            try:
                is_match, _unused = self._evaluate_image_path_match(frame_path)
            except Exception:
                is_match = False
            if is_match:
                positive_count += 1
                if first_positive is None:
                    first_positive = (slot_offset + local_idx, frame_path)
                if positive_count >= required:
                    # Re-evaluate the trigger frame with detail capture. predict_image
                    # caches results internally so this extra call is effectively free.
                    detail_out: list = [None]
                    try:
                        self._evaluate_image_path_match(first_positive[1], _detail_out=detail_out)
                    except Exception:
                        pass
                    return TriggerFrameResult(
                        slot_index=first_positive[0],
                        total_planned_slots=planned_slots,
                        frame_path=first_positive[1],
                        detail=detail_out[0],
                    )
            remaining = n - local_idx - 1
            if positive_count + remaining < required:
                break

        return None

    def run_on_media_path(
        self,
        media_path,
        callbacks: ActionCallbacks,
        base_directory: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[ClassifierActionType]:
        """Evaluate this action against *media_path* and (unless *dry_run*)
        execute it. With dry_run=True, matching returns the would-be action
        type without invoking run_action — no file I/O, callbacks, or
        FileAction history."""
        if not self.can_run:
            return None
        if not self.media_type_allowed(media_path):
            return None
        hide_callback = callbacks.hide_callback
        notify_callback = callbacks.notify_callback
        add_mark_callback = callbacks.add_mark_callback
        blur_callback = callbacks.blur_callback
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
                    if dry_run:
                        return self.action
                    return self.run_action(
                        media_path,
                        callbacks,
                        base_directory=base_directory or os.path.dirname(media_path),
                        resolved_category=resolved_match_category,
                    )
                return None

        try:
            return self.run_on_image_path(
                media_path,
                callbacks,
                base_directory=base_directory or os.path.dirname(media_path),
                dry_run=dry_run,
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
                callbacks,
                base_directory=base_directory or os.path.dirname(media_path),
                dry_run=dry_run,
            )

    def run_on_image_path(
        self,
        image_path,
        callbacks: ActionCallbacks,
        base_directory: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[ClassifierActionType]:
        if not self.can_run:
            return None
        is_match, matched_category = self._evaluate_image_path_match(image_path)
        if is_match:
            if dry_run:
                return self.action
            return self.run_action(
                image_path,
                callbacks,
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

    def _resolve_rotation_degrees(
        self,
        image_path: str,
        resolved_category: Optional[str] = None,
    ) -> Optional[int]:
        """Resolve how many degrees clockwise to rotate for a ROTATE action.

        A static ``action_modifier`` (e.g. "90") wins if set, the same way a
        static action_modifier overrides the classifier-derived MOVE target
        directory in :meth:`_resolve_action_target_directory`. Otherwise the
        degrees are parsed straight from the classifier's own predicted
        category label -- the orientation-detection model's categories already
        *are* the rotation amount needed to correct the image.
        """
        if self.action_modifier:
            return parse_rotation_label(self.action_modifier)
        category = resolved_category or self._resolve_classifier_target_category(image_path)
        if not category:
            return None
        return parse_rotation_label(category)

    def run_action(
        self,
        image_path,
        callbacks: ActionCallbacks,
        base_directory: Optional[str] = None,
        resolved_category: Optional[str] = None,
    ):
        hide_callback = callbacks.hide_callback
        notify_callback = callbacks.notify_callback
        add_mark_callback = callbacks.add_mark_callback
        blur_callback = callbacks.blur_callback
        generate_callback = callbacks.generate_callback
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
        elif self.action == ClassifierActionType.ROTATE:
            degrees = self._resolve_rotation_degrees(image_path, resolved_category=resolved_category)
            if degrees is None:
                notify_callback("\n" + base_message + _(" - could not determine rotation angle, skipped"),
                                base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
            elif degrees == 0:
                notify_callback("\n" + base_message + _(" - already correctly oriented"),
                                base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
            else:
                media_type = get_media_type_for_path(image_path)
                result = None
                if media_type == CompareMediaType.IMAGE:
                    notify_callback("\n" + base_message + _(" - rotating {0} degrees").format(degrees),
                                    base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
                    result = ImageOps.rotate_image_to_degrees(image_path, degrees)
                elif media_type == CompareMediaType.GIF:
                    notify_callback("\n" + base_message + _(" - rotating {0} degrees (new file)").format(degrees),
                                    base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
                    result = ImageOps.rotate_gif_to_degrees(image_path, degrees)
                elif media_type == CompareMediaType.VIDEO:
                    notify_callback("\n" + base_message + _(" - rotating {0} degrees (new file)").format(degrees),
                                    base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
                    try:
                        result = VideoOps.rotate_video(image_path, degrees)
                    except Exception as e:
                        logger.error(f"Error rotating video at {image_path} for classifier action {self.name}: {e}")
                else:
                    # SVG/HTML/PDF can't be rotated in their own format -- render a
                    # raster stand-in (reusing the existing FrameCache extraction
                    # used for classification) and write the rotated result as a
                    # new sibling file next to the real source. Never touches the
                    # source file or overwrites the FrameCache temp render itself.
                    true_source_path, frame_path = resolve_rendered_frame_source(image_path)
                    if true_source_path is None:
                        notify_callback("\n" + base_message + _(" - rotation not supported for this file type, skipped"),
                                        base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
                    else:
                        out_path = rotated_sibling_output_path(true_source_path, frame_path)
                        notify_callback(
                            "\n" + base_message + _(" - rotating {0} degrees, saving rendered preview: {1}").format(
                                degrees, os.path.basename(out_path)
                            ),
                            base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False,
                        )
                        result = ImageOps.rotate_image_to_degrees(frame_path, degrees, output_path=out_path)
                if result is None:
                    logger.error(f"Error rotating file at {image_path} for classifier action {self.name}")
        elif self.action.requires_target_directory():
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
        elif self.action == ClassifierActionType.GENERATE:
            if self.related_image_edit_suffix:
                from files.related_image import should_run_generate_action
                search_dir = self.action_modifier or base_directory or os.path.dirname(image_path)
                if not should_run_generate_action(image_path, self.related_image_edit_suffix, search_dir):
                    return None
            notify_callback("\n" + base_message + _(" - generating"), base_message=base_message, action_type=ActionType.GENERATE_IMAGE, is_manual=False)
            if generate_callback is not None:
                target_dir = os.path.dirname(image_path) if self.move_to_same_dir else None
                generate_callback(image_path, self.related_image_edit_suffix or None, target_dir=target_dir)
        elif self.action == ClassifierActionType.SCRAMBLE:
            notify_callback("\n" + base_message + _(" - scrambling"), base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False)
            scramble_callback = callbacks.scramble_callback
            if scramble_callback is not None:
                scramble_callback(image_path, self.related_image_edit_suffix or None)
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
            or self.use_filename_contains
            or self.use_base_stem_match
        ):
            raise Exception(
                "At least one validation type (embedding, image classifier, prompts, "
                "prompts blacklist, prototype, pseudo-static dynamic media, filename contains, "
                "or base stem match) must be enabled."
            )

        if self.use_filename_contains and not self.filename_contains_patterns:
            raise Exception("At least one filename pattern must be set when using filename contains validation.")
        
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
        # wrappers load lazily during prevalidation via ensure_image_classifier_loaded).
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
        return self.action is not None and self.action.requires_target_directory()

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
            "related_image_edit_suffix": self.related_image_edit_suffix,
            "is_active": self.is_active,
            "image_classifier_name": self.image_classifier_name,
            "image_classifier_selected_categories": self.image_classifier_selected_categories,
            "classification_mode": self.classification_mode.value,
            "use_embedding": self.use_embedding,
            "use_image_classifier": self.use_image_classifier,
            "use_prompts": self.use_prompts,
            "use_blacklist": self.use_blacklist,
            "use_pseudostatic_dynamic_media": self.use_pseudostatic_dynamic_media,
            "move_to_same_dir": self.move_to_same_dir,
            "use_prototype": self.use_prototype,
            "prototype_directory": self.prototype_directory,
            "negative_prototype_directory": self.negative_prototype_directory,
            "negative_prototype_lambda": self.negative_prototype_lambda,
            "dynamic_content_sample_ratio": self.dynamic_content_sample_ratio,
            "dynamic_content_positive_ratio": self.dynamic_content_positive_ratio,
            "_last_used_profile": self._last_used_profile,
            "lookahead_names": self.lookahead_names,
            "use_filename_contains": self.use_filename_contains,
            "filename_contains_patterns": self.filename_contains_patterns,
            "filename_contains_case_sensitive": self.filename_contains_case_sensitive,
            "use_base_stem_match": self.use_base_stem_match,
            "base_stem_match_require_match": self.base_stem_match_require_match,
            "applies_to_media_types": (
                [mt.value for mt in self.applies_to_media_types]
                if self.applies_to_media_types is not None
                else None
            ),
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
        if 'move_to_same_dir' not in d:
            d['move_to_same_dir'] = False
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
        if 'use_filename_contains' not in d:
            d['use_filename_contains'] = False
        if 'filename_contains_patterns' not in d:
            d['filename_contains_patterns'] = None
        if 'filename_contains_case_sensitive' not in d:
            d['filename_contains_case_sensitive'] = False
        if 'use_base_stem_match' not in d:
            d['use_base_stem_match'] = False
        if 'base_stem_match_require_match' not in d:
            d['base_stem_match_require_match'] = True
        if 'applies_to_media_types' not in d:
            d['applies_to_media_types'] = None
        if 'related_image_edit_suffix' not in d:
            d['related_image_edit_suffix'] = ""
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



@dataclass(eq=False, repr=False)
class Prevalidation(ClassifierAction):
    # Prevalidation-specific serializable field
    profile_name: Optional[str] = None

    # Runtime-only: populated by update_profile_instance(), never serialized
    profile: Optional[DirectoryProfile] = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        # profile starts as None; update_profile_instance() populates it once
        # the DirectoryProfile registry is available.

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
        callbacks: ActionCallbacks,
        base_directory: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[ClassifierActionType]:
        # Lazy load the image classifier if needed
        super().ensure_image_classifier_loaded(callbacks.notify_callback)
        return super().run_on_image_path(
            image_path, callbacks, base_directory=base_directory, dry_run=dry_run
        )

    def run_on_media_path(
        self,
        media_path,
        callbacks: ActionCallbacks,
        base_directory: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[ClassifierActionType]:
        # Keep lazy image-classifier loading behavior for sampled media paths too.
        super().ensure_image_classifier_loaded(callbacks.notify_callback)
        return super().run_on_media_path(
            media_path, callbacks, base_directory=base_directory, dry_run=dry_run
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

        # Handle backward compatibility: if run_on_folder exists but no profile_name, create temporary profile.
        # Pop run_on_folder so it is not forwarded to the dataclass __init__.
        run_on_folder = d.pop('run_on_folder', None)
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


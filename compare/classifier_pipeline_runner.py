"""
Execution engine for ClassifierPipeline (Phase 2).

run_pipeline() is the sole public entry point.  All imports of heavy
dependencies (CLIP, classifiers, prototype embeddings) are deferred to the
point of use so the module can be imported without loading ML models.

No existing modules are modified.
"""

from __future__ import annotations

import os
from typing import Optional

from compare.action_callbacks import ActionCallbacks
from compare.classifier_pipeline import (
    ClassifierPipeline,
    CompositeCondition,
    ClassifierRankCondition,
    EmbeddingCondition,
    FilenameContainsCondition,
    LookaheadCondition,
    MediaTypeCondition,
    NodeOutcome,
    NodeResultCondition,
    OutcomeType,
    PromptCondition,
    PrototypeCondition,
)
from utils.constants import ActionType, ClassifierActionType
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("classifier_pipeline_runner")

# ---------------------------------------------------------------------------
# Module-level prototype cache — avoids re-loading from disk on every image.
# Keyed by directory path; value is a numpy array or None (load failed).
# ---------------------------------------------------------------------------
_pos_prototype_cache: dict[str, object] = {}
_neg_prototype_cache: dict[str, object] = {}


def _load_prototype(directory: str, cache: dict) -> object:
    if directory in cache:
        return cache[directory]
    try:
        from compare.embedding_prototype import EmbeddingPrototype
        proto = EmbeddingPrototype.calculate_prototype_from_directory(directory)
        cache[directory] = proto
        return proto
    except Exception:
        logger.exception("Failed to load prototype from %s", directory)
        cache[directory] = None
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    pipeline: ClassifierPipeline,
    image_path: str,
    callbacks: ActionCallbacks,
    *,
    base_directory: Optional[str] = None,
) -> Optional[ClassifierActionType]:
    """
    Walk the pipeline nodes and return the ClassifierActionType to fire, or
    None for no action.

    Execution halts at the first EXECUTE / ACCEPT / REJECT outcome, or when
    all nodes have been visited (in which case default_action applies).
    GOTO jumps forward; CONTINUE advances sequentially.
    """
    if not pipeline.is_active or not pipeline.nodes:
        return None
    if not pipeline.media_type_allowed(image_path):
        return None

    node_results: dict[str, bool] = {}
    node_scores: dict[str, object] = {}
    node_order = [n.name for n in pipeline.nodes]
    nodes_by_name = {n.name: n for n in pipeline.nodes}

    current_name: Optional[str] = node_order[0]

    while current_name is not None:
        node = nodes_by_name[current_name]

        try:
            result, score = _evaluate_condition(
                node.condition, image_path, node_results, node_scores
            )
        except Exception:
            logger.exception(
                "Pipeline %r: error in node %r on %s — treating as no-match",
                pipeline.name, node.name, image_path,
            )
            result, score = False, None

        node_results[node.name] = result
        node_scores[node.name] = score

        outcome: NodeOutcome = node.on_match if result else node.on_no_match

        if outcome.outcome_type == OutcomeType.EXECUTE:
            _dispatch_action(
                outcome.action_type,
                outcome.action_modifier or None,
                pipeline.name,
                image_path,
                callbacks,
                base_directory,
            )
            return outcome.action_type

        if outcome.outcome_type == OutcomeType.ACCEPT:
            return None

        if outcome.outcome_type == OutcomeType.REJECT:
            _dispatch_action(
                pipeline.default_reject_action,
                None,
                pipeline.name,
                image_path,
                callbacks,
                base_directory,
            )
            return pipeline.default_reject_action

        if outcome.outcome_type == OutcomeType.GOTO:
            current_name = outcome.target_node
        else:  # CONTINUE
            idx = node_order.index(current_name)
            current_name = node_order[idx + 1] if idx + 1 < len(node_order) else None

    # All nodes exhausted without a terminal outcome.
    _dispatch_action(
        pipeline.default_action,
        None,
        pipeline.name,
        image_path,
        callbacks,
        base_directory,
    )
    return pipeline.default_action


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _evaluate_condition(
    condition,
    image_path: str,
    node_results: dict[str, bool],
    node_scores: dict[str, object],
) -> tuple[bool, object]:
    """Return (matched, score). Score is a raw float where available, else None."""

    if isinstance(condition, EmbeddingCondition):
        from compare.compare_embeddings_clip import CompareEmbeddingClip
        result = CompareEmbeddingClip.multi_text_compare(
            image_path, condition.positives, condition.negatives, condition.threshold
        )
        return bool(result), None

    if isinstance(condition, ClassifierRankCondition):
        return _eval_classifier_rank(condition, image_path)

    if isinstance(condition, PrototypeCondition):
        return _eval_prototype(condition, image_path)

    if isinstance(condition, PromptCondition):
        return _eval_prompt(condition, image_path)

    if isinstance(condition, FilenameContainsCondition):
        return _eval_filename_contains(condition, image_path)

    if isinstance(condition, MediaTypeCondition):
        return _eval_media_type(condition, image_path)

    if isinstance(condition, LookaheadCondition):
        return _eval_lookahead(condition, image_path)

    if isinstance(condition, NodeResultCondition):
        prior = node_results.get(condition.node_name)
        if prior is None:
            logger.warning(
                "NodeResultCondition references %r which has no result yet",
                condition.node_name,
            )
            return False, None
        return (prior == condition.expected_result), float(prior)

    if isinstance(condition, CompositeCondition):
        return _eval_composite(condition, image_path, node_results, node_scores)

    raise ValueError(f"Unknown condition type: {type(condition).__name__}")


def _eval_classifier_rank(
    condition: ClassifierRankCondition, image_path: str
) -> tuple[bool, object]:
    from image.image_classifier_manager import image_classifier_manager
    classifier = image_classifier_manager.get_classifier(condition.classifier_name)
    if classifier is None:
        logger.error(
            "ClassifierRankCondition: classifier %r not found", condition.classifier_name
        )
        return False, None

    ranked = classifier.predict_image_ranked(image_path)

    for rank_idx, (category, score) in enumerate(ranked, start=1):
        if rank_idx > condition.max_rank:
            break
        if rank_idx < condition.min_rank:
            continue
        if category in condition.categories and score >= condition.min_confidence:
            return True, score

    return False, None


def _eval_prototype(
    condition: PrototypeCondition, image_path: str
) -> tuple[bool, object]:
    from compare.embedding_prototype import EmbeddingPrototype

    if not condition.prototype_directory:
        return False, None

    pos_proto = _load_prototype(condition.prototype_directory, _pos_prototype_cache)
    if pos_proto is None:
        return False, None

    pos_sim = EmbeddingPrototype.compare_with_prototype(image_path, pos_proto)

    if condition.negative_prototype_directory:
        neg_proto = _load_prototype(
            condition.negative_prototype_directory, _neg_prototype_cache
        )
        if neg_proto is not None:
            neg_sim = EmbeddingPrototype.compare_with_prototype(
                image_path, neg_proto, negative_prototype=1
            )
            final = pos_sim - condition.negative_lambda * neg_sim
        else:
            final = pos_sim
    else:
        final = pos_sim

    return final >= condition.threshold, final


def _eval_prompt(
    condition: PromptCondition, image_path: str
) -> tuple[bool, object]:
    from image.image_data_extractor import image_data_extractor
    try:
        positive_prompt, negative_prompt = (
            image_data_extractor.extract_prompts_all_strategies(image_path)
        )
    except Exception:
        return False, None

    if positive_prompt is None:
        return False, None

    if condition.use_blacklist:
        from utils.config import config
        blacklist = getattr(config, "prompts_blacklist", []) or []
        return any(term.lower() in positive_prompt.lower() for term in blacklist), None

    if not condition.prompts:
        return False, None

    matched = any(p.lower() in positive_prompt.lower() for p in condition.prompts)
    return matched, None


def _eval_media_type(
    condition: MediaTypeCondition, image_path: str
) -> tuple[bool, object]:
    from utils.media_utils import get_media_type_for_path
    media_type = get_media_type_for_path(image_path)
    return media_type in condition.media_types, media_type.value


def _eval_filename_contains(
    condition: FilenameContainsCondition, image_path: str
) -> tuple[bool, object]:
    if not condition.patterns:
        return False, None
    filename = os.path.basename(image_path)
    if not condition.case_sensitive:
        filename = filename.lower()
    for pattern in condition.patterns:
        p = pattern if condition.case_sensitive else pattern.lower()
        if p and p in filename:
            return True, pattern
    return False, None


def _eval_lookahead(
    condition: LookaheadCondition, image_path: str
) -> tuple[bool, object]:
    from compare.compare_embeddings_clip import CompareEmbeddingClip
    from compare.lookahead import Lookahead

    lookahead = Lookahead.get_lookahead_by_name(condition.lookahead_name)
    if lookahead is None:
        logger.warning(
            "LookaheadCondition: lookahead %r not found", condition.lookahead_name
        )
        return False, None

    if lookahead.is_prevalidation_name:
        from compare.classifier_actions_manager import ClassifierActionsManager
        prevalidation = ClassifierActionsManager.get_prevalidation_by_name(
            lookahead.name_or_text
        )
        if prevalidation is None:
            return False, None
        positives = prevalidation.positives
        negatives = prevalidation.negatives
    else:
        positives = [lookahead.name_or_text] if lookahead.name_or_text else []
        negatives = []

    if not positives and not negatives:
        return False, None

    result = CompareEmbeddingClip.multi_text_compare(
        image_path, positives, negatives, lookahead.threshold
    )
    return bool(result), None


def _eval_composite(
    condition: CompositeCondition,
    image_path: str,
    node_results: dict[str, bool],
    node_scores: dict[str, object],
) -> tuple[bool, object]:
    sub_results = [
        _evaluate_condition(sub, image_path, node_results, node_scores)[0]
        for sub in condition.sub_conditions
    ]
    op = condition.operator
    if op == "AND":
        return all(sub_results), None
    if op == "OR":
        return any(sub_results), None
    if op == "NOT":
        return not sub_results[0], None
    if op == "XOR":
        return (sub_results[0] != sub_results[1]), None
    logger.error("Unknown composite operator %r", op)
    return False, None


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------

def _dispatch_action(
    action_type: Optional[ClassifierActionType],
    action_modifier: Optional[str],
    pipeline_name: str,
    image_path: str,
    callbacks: ActionCallbacks,
    base_directory: Optional[str],
) -> None:
    if action_type is None:
        return

    hide_callback = callbacks.hide_callback
    notify_callback = callbacks.notify_callback
    add_mark_callback = callbacks.add_mark_callback
    blur_callback = callbacks.blur_callback
    generate_callback = callbacks.generate_callback
    _notify = notify_callback or (lambda *a, **kw: None)
    base_message = pipeline_name + _(" detected")

    if action_type == ClassifierActionType.SKIP:
        _notify(
            "\n" + base_message + _(" - skipped"),
            base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False,
        )

    elif action_type == ClassifierActionType.HIDE:
        if hide_callback:
            hide_callback(image_path)
        _notify(
            "\n" + base_message + _(" - hidden"),
            base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False,
        )

    elif action_type == ClassifierActionType.NOTIFY:
        _notify(
            "\n" + base_message,
            base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False,
        )

    elif action_type == ClassifierActionType.ADD_MARK:
        if add_mark_callback:
            add_mark_callback(image_path)
        _notify(
            "\n" + base_message + _(" - marked"),
            base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False,
        )

    elif action_type in (ClassifierActionType.MOVE, ClassifierActionType.COPY):
        target_directory = action_modifier or base_directory
        if not target_directory:
            logger.error(
                "Pipeline %r: MOVE/COPY has no target directory", pipeline_name
            )
            return
        if not os.path.exists(target_directory):
            try:
                os.makedirs(target_directory, exist_ok=True)
            except Exception as e:
                logger.error(
                    "Pipeline %r: could not create %s: %s", pipeline_name, target_directory, e
                )
                return
        already_at_target = (
            os.path.normpath(os.path.dirname(image_path))
            == os.path.normpath(target_directory)
        )
        if not already_at_target:
            from files.file_action import FileAction
            from utils.utils import Utils
            move_fn = (
                Utils.move_file if action_type == ClassifierActionType.MOVE
                else Utils.copy_file
            )
            action_kind = (
                ActionType.MOVE_FILE if action_type == ClassifierActionType.MOVE
                else ActionType.COPY_FILE
            )
            short_target = Utils.get_relative_dirpath(target_directory, levels=2)
            verb = _("Moving") if action_type == ClassifierActionType.MOVE else _("Copying")
            _notify(
                f"\n{verb} file: {os.path.basename(image_path)} -> {short_target}",
                base_message=base_message, action_type=action_kind, is_manual=False,
            )
            try:
                FileAction.add_file_action(move_fn, image_path, target_directory)
            except Exception as e:
                logger.error(
                    "Pipeline %r: file action failed for %s -> %s: %s",
                    pipeline_name, image_path, target_directory, e,
                )

    elif action_type == ClassifierActionType.DELETE:
        _notify(
            "\n" + _("Deleting file: ") + os.path.basename(image_path),
            base_message=base_message, action_type=ActionType.REMOVE_FILE, is_manual=False,
        )
        try:
            from utils.utils import Utils
            with Utils.file_operation_lock:
                os.remove(image_path)
        except Exception as e:
            logger.error(
                "Pipeline %r: delete failed for %s: %s", pipeline_name, image_path, e
            )

    elif action_type == ClassifierActionType.BLUR:
        _notify(
            "\n" + base_message + _(" - blurred"),
            base_message=base_message, action_type=ActionType.SYSTEM, is_manual=False,
        )
        if blur_callback:
            blur_callback(image_path)

    elif action_type == ClassifierActionType.GENERATE:
        _notify(
            "\n" + base_message + _(" - generating"),
            base_message=base_message, action_type=ActionType.GENERATE_IMAGE, is_manual=False,
        )
        if generate_callback:
            generate_callback(image_path, action_modifier or None)

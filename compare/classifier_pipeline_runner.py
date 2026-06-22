"""
Execution engine for ClassifierPipeline (Phase 2).

run_pipeline() is the sole public entry point.  All imports of heavy
dependencies (CLIP, classifiers, prototype embeddings) are deferred to the
point of use so the module can be imported without loading ML models.

No existing modules are modified.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from compare.action_callbacks import ActionCallbacks
from compare.debounced_generate_queue import DebouncedGenerateQueue
from compare.pipeline_run_report import PipelineRunReport
from compare.classifier_pipeline import (
    ClassifierPipeline,
    CompositeCondition,
    ClassifierRankCondition,
    EmbeddingCondition,
    FilenameContainsCondition,
    GroupCondition,
    GroupChildResultCondition,
    LookaheadCondition,
    MediaTypeCondition,
    NodeOutcome,
    NodeResultCondition,
    OutcomeType,
    PromptCondition,
    PrototypeCondition,
    BaseStemMatchCondition,
    UnknownSuffixCondition,
    RelatedImageCondition,
)


from files.related_image import (
    _stem_matches_any_suffix,
    extract_filename_base_stem,
    find_files_by_base_stem,
    get_related_image_path,
)
from utils.config import config
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
# Stem-group validity helpers (§5.13)
# ---------------------------------------------------------------------------

def _seed_exists_in_dirs(base_stem: str, dirs: list) -> bool:
    """Return True if a file whose stem equals base_stem exactly (no suffix) exists in dirs."""
    for f in find_files_by_base_stem(dirs, base_stem, use_cache=True):
        if os.path.splitext(os.path.basename(f))[0].lower() == base_stem.lower():
            return True
    return False


def _resolve_stem_group(
    image_path: str,
    base_stem: str,
    is_seed: bool,
) -> tuple:
    """
    Classify this image's role in its stem group and decide whether to evaluate.

    Returns (should_evaluate: bool, should_mark_done: bool).

    The six cases (see §5.13 in the design document):

    Type 1 — valid
        Image stem == base_stem (this IS the seed) AND the seed file is found
        in at least one configured target directory.
        → (True, True): evaluate; seed marks the group done after its terminal outcome.

    Type 1 — derivative
        Stem ≠ base_stem AND the seed file IS present in the working directory.
        The RelatedImageCondition type-3 guard will block generation anyway.
        → (True, False): evaluate normally; do NOT mark done (seed will do it).

    Type 2 — valid
        Stem ≠ base_stem, seed NOT in working directory, related-image metadata
        exists, AND the exact basename named in that metadata is found in a target
        directory. The first derivative encountered acts as the generation source.
        → (True, True): evaluate; derivative marks the group done.

    Malformed A
        Stem ≠ base_stem, seed not in working dir, and no related-image metadata.
        Cannot determine seed identity.
        → (False, True): skip; mark done to suppress remaining derivatives.

    Malformed B
        Related-image metadata is present but the exact-named file is not found
        in any target directory. Seed is not filed.
        → (False, True): skip; mark done.

    Malformed C
        Image IS the seed (stem == base_stem) but no copy of the seed is found
        in any target directory. Seed is unfiled; generating variants is premature.
        → (False, True): skip; mark done; derivatives that arrive later hit the
        early-exit in run_pipeline.

    Note: if config.directories_to_search_for_related_images is empty, the target
    checks (Type 1 valid, Type 2, Malformed B/C) cannot be performed. In that case
    the function falls back to (True, is_seed) — seeds mark the group done, derivatives
    do not — without enforcing the target-presence requirement.
    """
    target_dirs = config.directories_to_search_for_related_images

    if not target_dirs:
        # No target dirs configured: skip validity checks, use seeds-only marking.
        return True, is_seed

    if is_seed:
        # Type 1 valid or Malformed C.
        if _seed_exists_in_dirs(base_stem, target_dirs):
            return True, True   # Type 1 valid
        logger.debug(
            "stem-group %r: Malformed C — seed in working dir but not in any target dir (%s)",
            base_stem, image_path,
        )
        return False, True  # Malformed C

    # Derivative: check if seed is present in the working directory.
    working_dir = os.path.dirname(os.path.abspath(image_path))
    if _seed_exists_in_dirs(base_stem, [working_dir]):
        return True, False  # Type 1 derivative; type-3 guard handles; seed marks done

    # Seed not in working dir: inspect related-image metadata.
    related_path, found_on_disk = get_related_image_path(image_path)
    if related_path is None:
        logger.debug(
            "stem-group %r: Malformed A — no related-image metadata (%s)",
            base_stem, image_path,
        )
        return False, True  # Malformed A

    if not found_on_disk:
        logger.debug(
            "stem-group %r: Malformed B — metadata names %r but file not found in target dirs (%s)",
            base_stem, os.path.basename(related_path), image_path,
        )
        return False, True  # Malformed B

    return True, True  # Type 2 valid; this derivative is the generation source


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    pipeline: ClassifierPipeline,
    image_path: str,
    callbacks: ActionCallbacks,
    *,
    base_directory: Optional[str] = None,
    report: Optional[PipelineRunReport] = None,
    generate_queue: Optional[DebouncedGenerateQueue] = None,
    processed_stems: Optional[set] = None,
) -> Optional[ClassifierActionType]:
    """
    Walk the pipeline nodes and return the ClassifierActionType to fire, or
    None for no action.

    Execution halts at the first EXECUTE / ACCEPT / REJECT outcome, or when
    all nodes have been visited (in which case default_action applies).
    GOTO jumps forward; CONTINUE advances sequentially.

    processed_stems
        Optional batch-level set of base stems already handled. When provided:
        - Images whose base_stem is already in the set are skipped immediately.
        - Before entering the evaluation loop, _resolve_stem_group() classifies
          the image's stem-group role. Malformed cases (A, B, C — see §5.13)
          are skipped without evaluation and marked done to suppress later
          derivatives of the same group.
        - After any terminal outcome, images flagged should_mark_done add
          their base_stem to the set so subsequent group members are skipped.
        Pass the same set for every call in a batch run. The parameter is
        optional so callers that need independent per-variant evaluation can
        omit it. RELATED_IMAGE ascending sort is strongly recommended when
        passing processed_stems; it ensures seeds are evaluated before their
        derivatives for maximum skip efficiency.
    """
    if not pipeline.is_active or not pipeline.nodes:
        logger.debug("Pipeline %r: skipped (inactive or no nodes) for %s", pipeline.name, image_path)
        return None
    if not pipeline.media_type_allowed(image_path):
        logger.debug("Pipeline %r: media type not allowed for %s", pipeline.name, image_path)
        return None

    base_stem: Optional[str] = extract_filename_base_stem(image_path)
    image_file_stem = os.path.splitext(os.path.basename(image_path))[0]
    is_seed = base_stem is not None and image_file_stem.lower() == base_stem.lower()
    logger.debug("Pipeline %r: evaluating %s (base_stem=%r, is_seed=%s)", pipeline.name, image_path, base_stem, is_seed)

    # Stem-group skip and validity gate.
    should_mark_done = False
    if processed_stems is not None and base_stem:
        if base_stem in processed_stems:
            logger.debug("Pipeline %r: stem %r already processed, skipping %s", pipeline.name, base_stem, image_path)
            return None

        should_evaluate, should_mark_done = _resolve_stem_group(image_path, base_stem, is_seed)
        if not should_evaluate:
            logger.debug("Pipeline %r: stem group resolve says skip %s", pipeline.name, image_path)
            processed_stems.add(base_stem)
            return None

    node_results: dict[str, bool] = {}
    node_scores: dict[str, object] = {}
    node_order = [n.name for n in pipeline.nodes]
    nodes_by_name = {n.name: n for n in pipeline.nodes}

    # Suffix that seeds cover, derived from pipeline.seed_category + category_map.
    # Empty string means the seed-category guard is disabled for this run.
    seed_suffix = (
        pipeline.category_map.get(pipeline.seed_category, "")
        if pipeline.seed_category else ""
    )

    current_name: Optional[str] = node_order[0]
    last_etc_action: Optional[ClassifierActionType] = None  # last EXECUTE_AND_CONTINUE action

    while current_name is not None:
        node = nodes_by_name[current_name]

        if not node.enabled:
            logger.debug("Pipeline %r: node %r disabled — skipped", pipeline.name, node.name)
            if report:
                report.add(
                    "INFO",
                    node.name,
                    image_path,
                    _("Node {0} is disabled — skipped").format(node.name),
                )
            idx = node_order.index(current_name)
            current_name = node_order[idx + 1] if idx + 1 < len(node_order) else None
            continue

        # Seed-category guard: when the image is a seed and the pipeline declares
        # which category seeds belong to, skip GENERATE for that category without
        # running the node condition.
        if (
            is_seed and seed_suffix
            and node.on_match.action_type == ClassifierActionType.GENERATE
            and node.on_match.action_modifier == seed_suffix
        ):
            result, score = False, None
            logger.debug(
                "Pipeline %r: node %r — seed is category %r, GENERATE skipped",
                pipeline.name, node.name, pipeline.seed_category,
            )
            if report:
                report.add(
                    "INFO",
                    node.name,
                    image_path,
                    _("Seed image is assigned to category '{0}'; GENERATE skipped.").format(
                        pipeline.seed_category
                    ),
                )
        else:
            try:
                result, score = _evaluate_condition(
                    node.condition, image_path, node_results, node_scores, base_directory,
                    node_name=node.name, report=report,
                    pipeline_categories=list(pipeline.category_map.values()),
                )
            except Exception:
                logger.exception(
                    "Pipeline %r: error in node %r on %s — treating as no-match",
                    pipeline.name, node.name, image_path,
                )
                result, score = False, None

        node_results[node.name] = result
        node_scores[node.name] = score
        logger.debug(
            "Pipeline %r: node %r → %s (score=%s)",
            pipeline.name, node.name, "MATCH" if result else "NO-MATCH", score,
        )

        outcome: NodeOutcome = node.on_match if result else node.on_no_match
        logger.debug(
            "Pipeline %r: node %r outcome → %s",
            pipeline.name, node.name, outcome.outcome_type.value,
        )

        if outcome.outcome_type == OutcomeType.EXECUTE:
            _dispatch_action(
                outcome.action_type,
                outcome.action_modifier or None,
                pipeline.name,
                image_path,
                callbacks,
                base_directory,
                generate_queue=generate_queue,
            )
            if should_mark_done and base_stem:
                processed_stems.add(base_stem)
            return outcome.action_type

        if outcome.outcome_type == OutcomeType.EXECUTE_AND_CONTINUE:
            _dispatch_action(
                outcome.action_type,
                outcome.action_modifier or None,
                pipeline.name,
                image_path,
                callbacks,
                base_directory,
                generate_queue=generate_queue,
            )
            last_etc_action = outcome.action_type
            # Fall through to CONTINUE — advance to the next node.
            idx = node_order.index(current_name)
            current_name = node_order[idx + 1] if idx + 1 < len(node_order) else None
            continue

        if outcome.outcome_type == OutcomeType.ACCEPT:
            if should_mark_done and base_stem:
                processed_stems.add(base_stem)
            return None

        if outcome.outcome_type == OutcomeType.REJECT:
            _dispatch_action(
                pipeline.default_reject_action,
                None,
                pipeline.name,
                image_path,
                callbacks,
                base_directory,
                generate_queue=generate_queue,
            )
            if should_mark_done and base_stem:
                processed_stems.add(base_stem)
            return pipeline.default_reject_action

        if outcome.outcome_type == OutcomeType.GOTO:
            current_name = outcome.target_node
        else:  # CONTINUE
            idx = node_order.index(current_name)
            current_name = node_order[idx + 1] if idx + 1 < len(node_order) else None

    # All nodes exhausted without a terminal outcome.
    logger.debug(
        "Pipeline %r: all nodes exhausted for %s — default_action=%s",
        pipeline.name, image_path, pipeline.default_action,
    )
    _dispatch_action(
        pipeline.default_action,
        None,
        pipeline.name,
        image_path,
        callbacks,
        base_directory,
        generate_queue=generate_queue,
    )
    if should_mark_done and base_stem:
        processed_stems.add(base_stem)
    # If no explicit default action was set but EXECUTE_AND_CONTINUE actions fired,
    # return the last such action so callers can account for the work done.
    return pipeline.default_action if pipeline.default_action is not None else last_etc_action


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def _evaluate_condition(
    condition,
    image_path: str,
    node_results: dict[str, bool],
    node_scores: dict[str, object],
    base_directory: Optional[str] = None,
    *,
    node_name: str = "",
    report: Optional[PipelineRunReport] = None,
    pipeline_categories: list = [],
) -> tuple[bool, object]:
    """Return (matched, score). Score is a raw float where available, else None."""

    if isinstance(condition, EmbeddingCondition):
        from compare.compare_embeddings_clip import CompareEmbeddingClip
        result = CompareEmbeddingClip.multi_text_compare(
            image_path, condition.positives, condition.negatives, condition.threshold
        )
        return bool(result), None

    if isinstance(condition, ClassifierRankCondition):
        return _eval_classifier_rank(condition, image_path, pipeline_categories=pipeline_categories)

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

    if isinstance(condition, BaseStemMatchCondition):
        return _eval_base_stem_match(condition, image_path, base_directory=base_directory,
                                     node_name=node_name, report=report,
                                     pipeline_categories=pipeline_categories)

    if isinstance(condition, UnknownSuffixCondition):
        return _eval_unknown_suffix(condition, image_path, base_directory=base_directory,
                                    node_name=node_name, report=report)

    if isinstance(condition, RelatedImageCondition):
        return _eval_related_image(condition, image_path, base_directory)

    if isinstance(condition, CompositeCondition):
        return _eval_composite(condition, image_path, node_results, node_scores, base_directory,
                               report=report, pipeline_categories=pipeline_categories)

    if isinstance(condition, GroupCondition):
        return _eval_group(condition, node_name, image_path, node_results, node_scores,
                           base_directory, report=report, pipeline_categories=pipeline_categories)

    if isinstance(condition, GroupChildResultCondition):
        key = f"{condition.group_node_name}/{condition.child_node_name}"
        prior = node_results.get(key)
        if prior is None:
            logger.warning(
                "GroupChildResultCondition references %r which has no result yet", key
            )
            return False, None
        return (prior == condition.expected_result), float(prior)

    raise ValueError(f"Unknown condition type: {type(condition).__name__}")


def _eval_classifier_rank(
    condition: ClassifierRankCondition,
    image_path: str,
    pipeline_categories: list = [],
) -> tuple[bool, object]:
    from image.image_classifier_manager import image_classifier_manager
    classifier = image_classifier_manager.get_classifier(condition.classifier_name)
    if classifier is None:
        logger.error(
            "ClassifierRankCondition: classifier %r not found", condition.classifier_name
        )
        return False, None

    categories = (
        pipeline_categories
        if condition.inherit_categories and not condition.categories
        else condition.categories
    )
    if not categories:
        return False, None

    ranked = classifier.predict_image_ranked(image_path)

    for rank_idx, (category, score) in enumerate(ranked, start=1):
        if rank_idx > condition.max_rank:
            break
        if rank_idx < condition.min_rank:
            continue
        if category in categories and score >= condition.min_confidence:
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



def _eval_base_stem_match(
    condition: BaseStemMatchCondition,
    image_path: str,
    *,
    base_directory: Optional[str] = None,
    node_name: str = "",
    report: Optional[PipelineRunReport] = None,
    pipeline_categories: list = [],
) -> tuple[bool, object]:
    base_stem = extract_filename_base_stem(image_path)
    if not base_stem:
        logger.debug("BaseStemMatch[%s]: no base stem extractable from %s", node_name, image_path)
        return False, None
    if condition.search_directory:
        dirs = [condition.search_directory]
    elif condition.use_working_directory:
        dirs = [base_directory or os.path.dirname(os.path.abspath(image_path))]
    else:
        dirs = config.directories_to_search_for_related_images
    if not dirs:
        logger.debug("BaseStemMatch[%s]: no search directories configured", node_name)
        return False, None
    matches = find_files_by_base_stem(dirs, base_stem, use_cache=True)
    logger.debug(
        "BaseStemMatch[%s]: base_stem=%r, dirs=%s, raw_matches=%d",
        node_name, base_stem, dirs, len(matches),
    )
    if condition.suffix_filter:
        matches = [
            f for f in matches
            if _stem_matches_any_suffix(
                os.path.splitext(os.path.basename(f))[0],
                condition.suffix_filter,
            )
        ]
        logger.debug("BaseStemMatch[%s]: after suffix filter %s → %d matches", node_name, condition.suffix_filter, len(matches))

    # Overflow-detection mode: active when max_stem_group_size > 0, or when it is 0
    # and the pipeline declares categories (effective limit = len(categories) + 1).
    # The inference only applies when search_directory is unset — nodes scoped to a
    # specific directory are doing targeted presence/absence checks, not broad scans
    # where stem uniqueness matters.
    effective_limit = condition.max_stem_group_size
    if effective_limit == 0 and pipeline_categories and not condition.search_directory:
        effective_limit = len(pipeline_categories) + 1

    if effective_limit > 0:
        overflow = len(matches) > effective_limit
        if overflow:
            logger.debug(
                "BaseStemMatchCondition: stem %r has %d matches (limit %d) — not unique",
                base_stem, len(matches), effective_limit,
            )
            if report:
                report.add(
                    "WARNING", node_name, image_path,
                    _(
                        "Stem {0} matches {1} files "
                        "(limit {2}) — base stem is not unique enough"
                    ).format(
                        base_stem,
                        len(matches),
                        effective_limit,
                    ),
                    data={"base_stem": base_stem, "match_count": len(matches)},
                )
        return overflow, None

    if report and len(matches) > 1:
        detail = _(
            "{0} files share base stem {1} in the search location"
        ).format(len(matches), base_stem)
        if condition.suffix_filter:
            detail += _(" (suffix filter: {0})").format(
                condition.suffix_filter
            )
        report.add(
            "NOTABLE",
            node_name,
            image_path,
            detail,
            data={"matches": matches, "base_stem": base_stem},
        )
    found = bool(matches)
    result = found if condition.require_match else not found
    logger.debug(
        "BaseStemMatch[%s]: found=%s, require_match=%s → result=%s",
        node_name, found, condition.require_match, result,
    )
    return result, None


def _eval_unknown_suffix(
    condition: UnknownSuffixCondition,
    image_path: str,
    *,
    base_directory: Optional[str] = None,
    node_name: str = "",
    report: Optional[PipelineRunReport] = None,
) -> tuple[bool, object]:
    """Return (True, None) if an unresolvable unknown-suffix file exists for this stem.

    The caller is expected to wrap this in CompositeCondition(NOT): the guard blocks
    generation when the stem group is ambiguous and classifier inference cannot resolve it.

    Search scope (see UnknownSuffixCondition docstring):
      search_directory set → use it exclusively.
      use_base_directory=True → use base_directory (or image's own dir as fallback).
      Otherwise → use config.directories_to_search_for_related_images.
    """
    base_stem = extract_filename_base_stem(image_path)
    if not base_stem:
        return False, None
    if condition.search_directory:
        dirs = [condition.search_directory]
    elif condition.use_base_directory:
        dirs = [base_directory or os.path.dirname(os.path.abspath(image_path))]
    else:
        dirs = config.directories_to_search_for_related_images
    if not dirs:
        return False, None

    all_matches = find_files_by_base_stem(dirs, base_stem, use_cache=True)

    has_unresolvable = False

    for f in all_matches:
        if os.path.normpath(f) == os.path.normpath(image_path):
            continue  # skip the image being evaluated
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem.lower() == base_stem.lower():
            continue  # seed image — no suffix, always excluded from the guard
        if _stem_matches_any_suffix(stem, condition.expected_suffixes):
            continue  # known suffix — fine

        # Unknown suffix found.
        if report:
            report.add(
                "NOTABLE", node_name, image_path,
                _("Unrecognised suffix in stem group: {0}").format(
                    os.path.basename(f)
                ),
                data={"unknown_file": f, "base_stem": base_stem},
            )

        resolved = False
        if condition.classifier_name:
            resolved = _infer_category_from_file(
                f, condition.classifier_name, condition.inference_threshold,
                node_name=node_name, image_path=image_path, report=report,
            )

        if not resolved:
            if report:
                suffix = (
                    _(" — classifier inconclusive")
                    if condition.classifier_name
                    else ""
                )
                report.add(
                    "WARNING", node_name, image_path,
                    _("Cannot determine category of {0}{1}").format(
                        os.path.basename(f),
                        suffix,
                    ),
                    data={"unknown_file": f, "base_stem": base_stem},
                )
            has_unresolvable = True

    return has_unresolvable, None


def _infer_category_from_file(
    file_path: str,
    classifier_name: str,
    threshold: float,
    *,
    node_name: str = "",
    image_path: str = "",
    report: Optional[PipelineRunReport] = None,
) -> bool:
    """Run the named classifier on file_path and return True if the top result
    meets the confidence threshold (i.e. the file's category can be inferred)."""
    try:
        from image.image_classifier_manager import image_classifier_manager
        classifier = image_classifier_manager.get_classifier(classifier_name)
        if classifier is None:
            logger.warning(
                "UnknownSuffixCondition: classifier %r not found", classifier_name
            )
            return False
        ranked = classifier.predict_image_ranked(file_path)
        if ranked and ranked[0][1] >= threshold:
            category, score = ranked[0]
            if report:
                report.add(
                    "INFO", node_name, image_path,
                    _(
                        "{0} inferred as {1} "
                        "(confidence {2}) — treated as resolved"
                    ).format(
                        os.path.basename(file_path),
                        category,
                        f"{score:.0%}",
                    ),
                    data={"unknown_file": file_path, "inferred_category": category, "score": score},
                )
            return True
        return False
    except Exception:
        logger.exception(
            "UnknownSuffixCondition: classifier inference failed for %s", file_path
        )
        return False


def _eval_related_image(
    condition: RelatedImageCondition,
    image_path: str,
    base_directory: Optional[str],
) -> tuple[bool, object]:
    from files.related_image import should_run_generate_action
    if condition.search_directory:
        dirs = [condition.search_directory]
    elif condition.use_configured_search_directories:
        dirs = list(config.directories_to_search_for_related_images or [])
    else:
        dirs = [base_directory or os.path.dirname(image_path)]
    if not dirs:
        return False, None
    for search_dir in dirs:
        if not should_run_generate_action(
            image_path, condition.edit_suffix, search_dir, condition.count_threshold
        ):
            return False, None
    return True, None


def _eval_group(
    condition: GroupCondition,
    outer_node_name: str,
    image_path: str,
    node_results: dict[str, bool],
    node_scores: dict[str, object],
    base_directory: Optional[str],
    *,
    report: Optional[PipelineRunReport] = None,
    pipeline_categories: list = [],
) -> tuple[bool, object]:
    """Evaluate every child node and store results under '<outer>/<child>' keys."""
    for child in condition.nodes:
        key = f"{outer_node_name}/{child.name}"
        try:
            child_result, child_score = _evaluate_condition(
                child.condition, image_path, node_results, node_scores, base_directory,
                node_name=key, report=report, pipeline_categories=pipeline_categories,
            )
        except Exception:
            logger.exception(
                "Group %r: error evaluating child %r — treating as no-match",
                outer_node_name, child.name,
            )
            child_result, child_score = False, None
        node_results[key] = child_result
        node_scores[key] = child_score

    child_results = [
        node_results.get(f"{outer_node_name}/{c.name}", False)
        for c in condition.nodes
    ]
    if condition.operator == "AND":
        return all(child_results), None
    return any(child_results), None   # OR (default)


def _eval_composite(
    condition: CompositeCondition,
    image_path: str,
    node_results: dict[str, bool],
    node_scores: dict[str, object],
    base_directory: Optional[str] = None,
    *,
    report: Optional[PipelineRunReport] = None,
    pipeline_categories: list = [],
) -> tuple[bool, object]:
    sub_results = [
        _evaluate_condition(sub, image_path, node_results, node_scores, base_directory,
                            report=report, pipeline_categories=pipeline_categories)[0]
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
    *,
    generate_queue: Optional[DebouncedGenerateQueue] = None,
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

    elif action_type.requires_target_directory():
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
            if generate_queue is not None:
                generate_queue.submit(generate_callback, image_path, action_modifier or None)
            else:
                generate_callback(image_path, action_modifier or None)

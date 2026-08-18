"""
ClassifierPipeline: multi-node decision-tree classifier.

A ClassifierPipeline is an ordered list of PipelineNodes.  Each node holds a
single NodeCondition and two NodeOutcomes (on_match / on_no_match).  Execution
walks the list from the first node, branching or halting according to the
outcomes, and eventually returns a ClassifierActionType (or None for no action).

Storage lives in app_info_cache under "classifier_pipelines".  No existing
ClassifierAction or Prevalidation data is touched.

The condition and node data models live in classifier_pipeline_conditions and
classifier_pipeline_nodes; execution lives in classifier_pipeline_runner.  This
module holds the pipeline object itself, its profile-scoped subclass, and the
global store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from files.related_image import suffix_is_numeric
from utils.app_info_cache import app_info_cache
from utils.constants import ClassifierActionType, CompareMediaType, ImageGenerationType, SortBy
from utils.logging_setup import get_logger
from utils.translations import _

# Re-exported for backward compatibility: these types moved to the sibling
# modules above, but call sites across the app and test suite still import them
# from here, and this module remains the single import point for the whole
# pipeline model.  Keep both lists in sync when adding a condition type.
from compare.classifier_pipeline_conditions import (  # noqa: F401
    AlwaysCondition,
    AudioClassifierRankCondition,
    BaseStemMatchCondition,
    ClassifierRankCondition,
    CompositeCondition,
    EmbeddingCondition,
    FilenameContainsCondition,
    GroupChildResultCondition,
    GroupCondition,
    LookaheadCondition,
    MediaTypeCondition,
    NodeCondition,
    NodeResultCondition,
    PromptCondition,
    PrototypeCondition,
    RelatedImageCondition,
    UnknownSuffixCondition,
)
from compare.classifier_pipeline_nodes import (  # noqa: F401
    NodeOutcome,
    OutcomeType,
    PipelineNode,
    _condition_from_dict,
)

logger = get_logger("classifier_pipeline")


# ---------------------------------------------------------------------------
# ClassifierPipeline
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class ClassifierPipeline:
    name: str = field(default_factory=lambda: _("New Pipeline"))
    description: str = ""
    nodes: Optional[list] = None      # list[PipelineNode]; defaulted in __post_init__
    default_action: Optional[ClassifierActionType] = None
    default_reject_action: Optional[ClassifierActionType] = None
    is_active: bool = True
    applies_to_media_types: Optional[list] = None   # list[CompareMediaType]; None = all types
    # ImageGenerationType to use for GENERATE actions; None inherits the application's
    # current global generation mode at run time.
    generation_type: Optional[ImageGenerationType] = None
    # When True (default), GENERATE actions include the current working directory as
    # the target directory in the sd-runner request so generated files land there.
    move_to_working_dir: bool = True
    # Optional mapping of human-readable category name → filesystem suffix.
    # e.g. {"Apple": "_apple", "Banana": "_banana"}
    # The suffix values are the identifiers used by BaseStemMatchCondition / UnknownSuffixCondition.
    # BaseStemMatchCondition infers its overflow limit as len(category_map) + 1 when
    # max_stem_group_size is left at 0.
    category_map: dict = field(default_factory=dict)
    # The category that seed images (images whose file stem equals the base stem,
    # i.e. no suffix) belong to by default.  When set, the runner skips GENERATE
    # for this category when the image under evaluation is a seed, without needing
    # a ClassifierRankCondition seed guard in the node.  Must be a key present in
    # category_map (validated by validate()).  Empty string = feature disabled.
    seed_category: str = ""
    # File order for batch runs (see utils.constants.SortBy).  None = no sort --
    # gather_files' raw, unordered glob result is used as-is, which is the
    # fastest option for a large directory since it skips wrapping/sorting
    # entirely.
    run_sort_by: Optional[SortBy] = None

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.applies_to_media_types is not None:
            coerced = [
                mt if isinstance(mt, CompareMediaType) else CompareMediaType(mt)
                for mt in self.applies_to_media_types
            ]
            self.applies_to_media_types = coerced if coerced else None

    def media_type_allowed(self, path: str) -> bool:
        """Return False when applies_to_media_types is set and path's type is not in it."""
        if self.applies_to_media_types is None:
            return True
        from utils.media_utils import get_media_type_for_path
        return get_media_type_for_path(path) in self.applies_to_media_types

    def __eq__(self, other):
        return isinstance(other, ClassifierPipeline) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def has_generate_action(self) -> bool:
        """True if any enabled node can produce a GENERATE action on match or no-match."""
        for node in self.nodes:
            if not node.enabled:
                continue
            for outcome in (node.on_match, node.on_no_match):
                if outcome is not None and outcome.action_type == ClassifierActionType.GENERATE:
                    return True
        return False

    def sort_files_for_run(self, files: list) -> list:
        """
        Order *files* (already gathered, e.g. by compare.base_compare.gather_files)
        per run_sort_by before a batch run.

        Returns *files* unchanged when run_sort_by is unset (the default) --
        the fastest option for a large directory, since it skips wrapping
        every path in a SortableFile and sorting entirely, keeping
        gather_files' raw, unordered glob result as-is.
        """
        if self.run_sort_by is None or not files:
            return files
        from files.file_browser import FileBrowser
        from files.sortable_file import SortableFile
        file_browser = FileBrowser(directory="", recursive=False, sort_by=self.run_sort_by)
        sortable_files = [SortableFile(f) for f in files]
        return file_browser.get_sorted_files(sortable_files)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Return a list of error strings (empty list = valid).

        Checks:
        - No duplicate node names
        - GOTO targets exist and are forward references
        - NodeResultCondition references name a node earlier in the list
        - ClassifierRankCondition classifier names are registered (lazy — skipped if
          image_classifier_manager is not yet loaded)
        - LookaheadCondition names resolve
        - seed_category, when set, must be a key in category_map
        """
        errors: list[str] = []
        if not self.name.strip():
            errors.append(_("Pipeline name is empty."))

        if self.seed_category:
            if not self.category_map:
                errors.append(
                    _("seed_category '{0}' is set but category_map is empty.").format(
                        self.seed_category
                    )
                )
            elif self.seed_category not in self.category_map:
                errors.append(
                    _("seed_category '{0}' is not a key in category_map.").format(
                        self.seed_category
                    )
                )

        for category, suffix in self.category_map.items():
            if suffix_is_numeric(suffix):
                errors.append(
                    _("category_map entry '{0}': suffix '{1}' is purely numeric and cannot be used as a suffix.").format(
                        category, suffix
                    )
                )

        node_names: list[str] = []
        seen_names: set[str] = set()
        for node in self.nodes:
            if not node.name.strip():
                errors.append(_("A node has an empty name."))
            elif node.name in seen_names:
                errors.append(_("Duplicate node name: {0}.").format(node.name))
            else:
                seen_names.add(node.name)
            node_names.append(node.name)

        all_names = set(node_names)

        for i, node in enumerate(self.nodes):
            defined_before = set(node_names[:i])

            for outcome in (node.on_match, node.on_no_match):
                if outcome.outcome_type == OutcomeType.GOTO:
                    if not outcome.target_node:
                        errors.append(_("Node {0}: GOTO has no target.").format(node.name))
                    elif outcome.target_node not in all_names:
                        errors.append(
                            _("Node {0}: GOTO target {1} does not exist.").format(
                                node.name, outcome.target_node
                            )
                        )
                    else:
                        target_idx = node_names.index(outcome.target_node)
                        if target_idx <= i:
                            errors.append(
                                _("Node {0}: GOTO target {1} must be a later node (cycle prevention).").format(
                                    node.name, outcome.target_node
                                )
                            )
                if outcome.outcome_type.requires_action and outcome.action_type is None:
                    errors.append(
                        _("Node {0}: {1} outcome has no action_type.").format(
                            node.name, outcome.outcome_type.display()
                        )
                    )

            errors.extend(
                self._validate_condition(node.condition, node.name, defined_before)
            )

            # RelatedImageCondition/GENERATE outcome consistency
            if isinstance(node.condition, RelatedImageCondition):
                for outcome in (node.on_match, node.on_no_match):
                    if (outcome.outcome_type.requires_action
                            and outcome.action_type == ClassifierActionType.GENERATE
                            and outcome.action_modifier != node.condition.edit_suffix):
                        errors.append(
                            _("Node {0}: RelatedImageCondition.edit_suffix ({1}) must match the GENERATE outcome's action_modifier ({2}).").format(
                                node.name, node.condition.edit_suffix, outcome.action_modifier
                            )
                        )

        return errors

    def _validate_condition(self, condition, node_name: str,
                            defined_before: set[str]) -> list[str]:
        errors: list[str] = []

        if isinstance(condition, NodeResultCondition):
            if not condition.node_name:
                errors.append(_("Node {0}: NodeResultCondition has no node_name.").format(node_name))
            elif condition.node_name not in defined_before:
                errors.append(
                    _("Node {0}: NodeResultCondition references {1} which is not a prior node.").format(
                        node_name, condition.node_name
                    )
                )

        elif isinstance(condition, ClassifierRankCondition):
            if not condition.classifier_name:
                errors.append(
                    _("Node {0}: ClassifierRankCondition has no classifier_name.").format(node_name)
                )
            else:
                try:
                    from image.image_classifier_manager import image_classifier_manager
                    names = image_classifier_manager.get_model_names()
                    if names and condition.classifier_name not in names:
                        errors.append(
                            _("Node {0}: classifier {1} is not registered.").format(
                                node_name, condition.classifier_name
                            )
                        )
                except Exception:
                    pass  # manager not available during unit tests — skip
            if condition.min_rank < 1:
                errors.append(_("Node {0}: min_rank must be ≥ 1.").format(node_name))
            if condition.max_rank < condition.min_rank:
                errors.append(_("Node {0}: max_rank must be ≥ min_rank.").format(node_name))

        elif isinstance(condition, AudioClassifierRankCondition):
            if not condition.classifier_name:
                errors.append(
                    _("Node {0}: AudioClassifierRankCondition has no classifier_name.").format(node_name)
                )
            else:
                try:
                    from image.audio_classifier_manager import audio_classifier_manager
                    names = audio_classifier_manager.get_model_names()
                    if names and condition.classifier_name not in names:
                        errors.append(
                            _("Node {0}: audio classifier {1} is not registered.").format(
                                node_name, condition.classifier_name
                            )
                        )
                except Exception:
                    pass  # manager not available during unit tests — skip
            if condition.min_rank < 1:
                errors.append(_("Node {0}: min_rank must be ≥ 1.").format(node_name))
            if condition.max_rank < condition.min_rank:
                errors.append(_("Node {0}: max_rank must be ≥ min_rank.").format(node_name))

        elif isinstance(condition, FilenameContainsCondition):
            if not condition.patterns:
                errors.append(
                    _("Node {0}: FilenameContainsCondition has no patterns.").format(node_name)
                )

        elif isinstance(condition, MediaTypeCondition):
            if not condition.media_types:
                errors.append(
                    _("Node {0}: MediaTypeCondition has no media_types.").format(node_name)
                )

        elif isinstance(condition, LookaheadCondition):
            if not condition.lookahead_name:
                errors.append(_("Node {0}: LookaheadCondition has no lookahead_name.").format(node_name))
            else:
                try:
                    from compare.lookahead import Lookahead
                    if Lookahead.get_lookahead_by_name(condition.lookahead_name) is None:
                        errors.append(
                            _("Node {0}: lookahead {1} is not defined.").format(
                                node_name, condition.lookahead_name
                            )
                        )
                except Exception:
                    pass

        elif isinstance(condition, BaseStemMatchCondition):
            if condition.search_directory and not os.path.isdir(condition.search_directory):
                errors.append(
                    _("Node {0}: BaseStemMatchCondition.search_directory ({1}) is not a valid directory.").format(
                        node_name, condition.search_directory
                    )
                )

        elif isinstance(condition, UnknownSuffixCondition):
            if condition.search_directory and not os.path.isdir(condition.search_directory):
                errors.append(
                    _("Node {0}: UnknownSuffixCondition.search_directory ({1}) is not a valid directory.").format(
                        node_name, condition.search_directory
                    )
                )

        elif isinstance(condition, RelatedImageCondition):
            if not condition.edit_suffix:
                errors.append(_("Node {0}: RelatedImageCondition has no edit_suffix.").format(node_name))
            if condition.search_directory and not os.path.isdir(condition.search_directory):
                errors.append(
                    _("Node {0}: RelatedImageCondition.search_directory ({1}) is not a valid directory.").format(
                        node_name, condition.search_directory
                    )
                )

        elif isinstance(condition, CompositeCondition):
            if condition.operator not in CompositeCondition.VALID_OPERATORS:
                errors.append(
                    _("Node {0}: unknown composite operator {1}.").format(node_name, condition.operator)
                )
            n = len(condition.sub_conditions)
            if condition.operator == "NOT" and n != 1:
                errors.append(
                    _("Node {0}: NOT requires exactly 1 sub-condition, got {1}.").format(node_name, n)
                )
            if condition.operator == "XOR" and n != 2:
                errors.append(
                    _("Node {0}: XOR requires exactly 2 sub-conditions, got {1}.").format(node_name, n)
                )
            if condition.operator in ("AND", "OR") and n < 1:
                # Unlike NOT (fixed arity 1) and XOR (fixed arity 2), AND/OR
                # generalize to any operand count: all([x]) == x and
                # any([x]) == x, so a single sub-condition is well-defined
                # (just redundant) rather than invalid. Only an empty list
                # has no sub-condition to evaluate at all.
                errors.append(
                    _("Node {0}: {1} requires at least 1 sub-condition, got {2}.").format(
                        node_name, condition.operator, n
                    )
                )
            for sub in condition.sub_conditions:
                errors.extend(self._validate_condition(sub, node_name, defined_before))

        elif isinstance(condition, GroupCondition):
            if condition.operator not in GroupCondition.VALID_OPERATORS:
                errors.append(
                    _("Node {0}: GroupCondition has unknown operator {1}.").format(
                        node_name, condition.operator
                    )
                )
            if not condition.nodes:
                errors.append(_("Node {0}: GroupCondition has no child nodes.").format(node_name))
            else:
                seen_children: set[str] = set()
                for child in condition.nodes:
                    if not child.name.strip():
                        errors.append(
                            _("Node {0}: GroupCondition child has an empty name.").format(node_name)
                        )
                    elif child.name in seen_children:
                        errors.append(
                            _("Node {0}: GroupCondition duplicate child name {1}.").format(
                                node_name, child.name
                            )
                        )
                    else:
                        seen_children.add(child.name)
                    errors.extend(
                        self._validate_condition(child.condition, f"{node_name}/{child.name}", defined_before)
                    )

        elif isinstance(condition, GroupChildResultCondition):
            if not condition.group_node_name:
                errors.append(
                    _("Node {0}: GroupChildResultCondition has no group_node_name.").format(node_name)
                )
            if not condition.child_node_name:
                errors.append(
                    _("Node {0}: GroupChildResultCondition has no child_node_name.").format(node_name)
                )
            if condition.group_node_name and condition.group_node_name not in defined_before:
                errors.append(
                    _("Node {0}: GroupChildResultCondition references group {1} which is not a prior node.").format(
                        node_name, condition.group_node_name
                    )
                )

        return errors

    # ------------------------------------------------------------------
    # Category-map suffix warnings (non-blocking)
    # ------------------------------------------------------------------

    def validate_warnings(self) -> list[str]:
        """Return non-blocking warning strings for category/suffix mismatches.

        Checks:
        - BaseStemMatchCondition.suffix_filter and UnknownSuffixCondition.expected_suffixes
          values that are not present in category_map (only when category_map is non-empty).
        - ClassifierRankCondition nodes that have inherit_categories=True but the pipeline
          has no category_map to inherit from.
        """
        known = set(self.category_map.values())
        warnings: list[str] = []
        for node in self.nodes:
            self._collect_suffix_warnings(node.condition, node.name, known, warnings)
        return warnings

    def _collect_suffix_warnings(
        self, condition, node_name: str, known_suffixes: set, warnings: list
    ) -> None:
        if isinstance(condition, ClassifierRankCondition):
            if condition.inherit_categories and not known_suffixes:
                warnings.append(
                    _("Node {0}: ClassifierRankCondition has inherit_categories=True but the pipeline has no category map — condition will match nothing.").format(node_name)
                )
        elif isinstance(condition, AudioClassifierRankCondition):
            if condition.inherit_categories and not known_suffixes:
                warnings.append(
                    _("Node {0}: AudioClassifierRankCondition has inherit_categories=True but the pipeline has no category map — condition will match nothing.").format(node_name)
                )
        elif isinstance(condition, BaseStemMatchCondition):
            if known_suffixes:
                for sf in condition.suffix_filter:
                    if sf not in known_suffixes:
                        warnings.append(
                            _("Node {0}: suffix filter {1} is not in the pipeline's category map.").format(node_name, sf)
                        )
        elif isinstance(condition, UnknownSuffixCondition):
            if known_suffixes:
                for sf in condition.expected_suffixes:
                    if sf not in known_suffixes:
                        warnings.append(
                            _("Node {0}: expected suffix {1} is not in the pipeline's category map.").format(node_name, sf)
                        )
        elif isinstance(condition, CompositeCondition):
            for sub in condition.sub_conditions:
                self._collect_suffix_warnings(sub, node_name, known_suffixes, warnings)
        elif isinstance(condition, GroupCondition):
            for child in condition.nodes:
                self._collect_suffix_warnings(
                    child.condition, f"{node_name}/{child.name}", known_suffixes, warnings
                )

    # ------------------------------------------------------------------
    # Flow preview (plain text, no Qt dependency)
    # ------------------------------------------------------------------

    def flow_summary(self) -> str:
        """Multi-line summary: one node per two lines, suitable for a scrollable list cell."""
        if not self.nodes:
            return _("(empty)")
        _LABELS = {
            "embedding": _("Embedding"),
            "classifier_rank": _("ClsRank"),
            "audio_classifier_rank": _("AudioClsRank"),
            "prototype": _("Prototype"),
            "prompt": _("Prompt"),
            "always": _("Always"),
            "lookahead": _("Lookahead"),
            "node_result": _("NodeResult"),
            "composite": _("Composite"),
            "group": _("Group"),
            "group_child_result": _("GroupChild"),
        }
        lines = []
        for node in self.nodes:
            cond_type = getattr(node.condition, "condition_type", "")
            if cond_type == "group":
                op = getattr(node.condition, "operator", "OR")
                n = len(getattr(node.condition, "nodes", []))
                cond_label = _("Group ({0}, {1} children)").format(op, n)
            else:
                cond_label = _LABELS.get(cond_type, cond_type)
            lines.append(f"{node.name} [{cond_label}]")
            lines.append(
                f"  ✓ {node.on_match.display_summary()}  ✗ {node.on_no_match.display_summary()}"
            )
        if self.default_action:
            lines.append(_("(end) → {0}").format(self.default_action.get_translation()))
        return "\n".join(lines)

    def flow_preview(self) -> str:
        if not self.nodes:
            return _("(no nodes)")
        lines: list[str] = []
        for node in self.nodes:
            lines.append(f"[{node.name}: {node.condition_summary()}]")
            if isinstance(node.condition, GroupCondition):
                op = node.condition.operator
                for child in node.condition.nodes:
                    lines.append(f"  {'·'} {child.name}: {child.condition_summary()}")
                lines.append(_("  ({0} of {1} children)").format(op, len(node.condition.nodes)))
            lines.append(f"  ✓ → {node.on_match.display_summary()}")
            lines.append(f"  ✗ → {node.on_no_match.display_summary()}")
            lines.append("")
        if self.default_action:
            lines.append(_("(end) → {0}").format(self.default_action.get_translation()))
        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "default_action": self.default_action.value if self.default_action else None,
            "default_reject_action": (
                self.default_reject_action.value if self.default_reject_action else None
            ),
            "is_active": self.is_active,
            "applies_to_media_types": (
                [mt.value for mt in self.applies_to_media_types]
                if self.applies_to_media_types is not None else None
            ),
            "generation_type": self.generation_type.value if self.generation_type else None,
            "move_to_working_dir": self.move_to_working_dir,
        }
        if self.category_map:
            d["category_map"] = dict(self.category_map)
        if self.seed_category:
            d["seed_category"] = self.seed_category
        if self.run_sort_by:
            d["run_sort_by"] = self.run_sort_by.get_text()
        return d

    @staticmethod
    def from_dict(d: dict) -> "ClassifierPipeline":
        def _opt_action(val):
            if not val:
                return None
            return ClassifierActionType[val] if isinstance(val, str) else val

        def _opt_sort_by(val):
            if not val:
                return None
            return SortBy.get(val) if isinstance(val, str) else val

        raw_map = d.get("category_map")
        if raw_map is None:
            # Backward compat: old "categories" list → identity map (name == suffix)
            raw_map = {s: s for s in d.get("categories", [])}

        return ClassifierPipeline(
            name=d.get("name", _("New Pipeline")),
            description=d.get("description", ""),
            nodes=[PipelineNode.from_dict(n) for n in d.get("nodes", [])],
            default_action=_opt_action(d.get("default_action")),
            default_reject_action=_opt_action(d.get("default_reject_action")),
            is_active=d.get("is_active", True),
            applies_to_media_types=d.get("applies_to_media_types"),
            generation_type=(
                ImageGenerationType.get(d["generation_type"])
                if d.get("generation_type") else None
            ),
            category_map=raw_map,
            seed_category=d.get("seed_category", ""),
            run_sort_by=_opt_sort_by(d.get("run_sort_by")),
            move_to_working_dir=d.get("move_to_working_dir", True),
        )


# ---------------------------------------------------------------------------
# PrevalidationPipeline — profile-scoped subclass (mirrors Prevalidation)
# ---------------------------------------------------------------------------

@dataclass(eq=False, repr=False)
class PrevalidationPipeline(ClassifierPipeline):
    """
    A ClassifierPipeline that is scoped to a DirectoryProfile, allowing it to
    participate in the prevalidation pass (Phase 5 integration).
    """

    profile_name: Optional[str] = None

    # Runtime-only: populated by update_profile_instance(), never serialized
    profile: object = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()

    def update_profile_instance(self, profile_name: Optional[str] = None) -> None:
        from files.directory_profile import DirectoryProfile
        name = profile_name or self.profile_name
        self.profile_name = name
        self.profile = None
        if name:
            for p in DirectoryProfile.directory_profiles:
                if p.name == name:
                    self.profile = p
                    break

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["profile_name"] = self.profile_name
        d["pipeline_class"] = "prevalidation"
        return d

    @staticmethod
    def from_dict(d: dict) -> "PrevalidationPipeline":
        def _opt_action(val):
            if not val:
                return None
            return ClassifierActionType[val] if isinstance(val, str) else val

        raw_map = d.get("category_map")
        if raw_map is None:
            raw_map = {s: s for s in d.get("categories", [])}

        return PrevalidationPipeline(
            profile_name=d.get("profile_name"),
            name=d.get("name", _("New Pipeline")),
            description=d.get("description", ""),
            nodes=[PipelineNode.from_dict(n) for n in d.get("nodes", [])],
            default_action=_opt_action(d.get("default_action")),
            default_reject_action=_opt_action(d.get("default_reject_action")),
            is_active=d.get("is_active", True),
            applies_to_media_types=d.get("applies_to_media_types"),
            category_map=raw_map,
        )


# ---------------------------------------------------------------------------
# ClassifierPipelines — global storage manager
# ---------------------------------------------------------------------------

_CACHE_KEY = "classifier_pipelines"


class ClassifierPipelines:
    pipelines: list[ClassifierPipeline] = []
    _prevalidation_pipelines: list["PrevalidationPipeline"] = []
    _action_pipelines: list[ClassifierPipeline] = []

    @staticmethod
    def _rebuild_type_cache() -> None:
        pv: list["PrevalidationPipeline"] = []
        ac: list[ClassifierPipeline] = []
        for p in ClassifierPipelines.pipelines:
            if isinstance(p, PrevalidationPipeline):
                pv.append(p)
            else:
                ac.append(p)
        ClassifierPipelines._prevalidation_pipelines = pv
        ClassifierPipelines._action_pipelines = ac

    @staticmethod
    def load() -> None:
        raw = app_info_cache.get_meta(_CACHE_KEY, default_val=None)
        result: list[ClassifierPipeline] = []
        for d in (raw or []):
            try:
                if d.get("pipeline_class") == "prevalidation":
                    result.append(PrevalidationPipeline.from_dict(d))
                else:
                    result.append(ClassifierPipeline.from_dict(d))
            except Exception:
                logger.exception("Failed to load pipeline from cache entry")
        ClassifierPipelines.pipelines = result
        ClassifierPipelines._rebuild_type_cache()

    @staticmethod
    def build_demo_pipeline() -> "ClassifierPipeline":
        """
        Demo pipeline that exercises every available condition type.

        Flow summary
        ────────────
        1  Media type check       MediaTypeCondition      — non-images are accepted immediately
        2  Filename hints         GroupCondition (OR)     — 3 FilenameContains children
        3  Filename says NSFW?    GroupChildResultCondition  — hide if "nsfw_" prefix matched
        4  Filename says safe?    GroupChildResultCondition  — accept if "safe_" prefix matched
        5  Person visible?        EmbeddingCondition      — skip person checks if no person
        6  Sensitive content?     CompositeCondition (OR) — embedding + prompt sub-conditions
        7  Safe prototype match?  PrototypeCondition      — jump past rank check if prototype matches
        8  Had filename hint?     NodeResultCondition     — mark for review if hinted but not resolved
        9  Classifier rank check  ClassifierRankCondition — accept if safe category ranks highly
        10 Lookahead safety       LookaheadCondition      — accept if lookahead says safe
        11 Related image exists?  RelatedImageCondition   — generate or reject
        """

        # ------------------------------------------------------------------
        # Node 1 — MediaTypeCondition
        # ------------------------------------------------------------------
        node_media_type = PipelineNode(
            name="Media type check",
            condition=MediaTypeCondition(
                media_types=[CompareMediaType.IMAGE, CompareMediaType.GIF],
            ),
            on_match=NodeOutcome(OutcomeType.CONTINUE),
            on_no_match=NodeOutcome(OutcomeType.ACCEPT),
        )

        # ------------------------------------------------------------------
        # Node 2 — GroupCondition (OR) with three FilenameContains children
        # ------------------------------------------------------------------
        node_filename_hints = PipelineNode(
            name="Filename category hints",
            condition=GroupCondition(
                operator="OR",
                nodes=[
                    PipelineNode(
                        name="is_marked_nsfw",
                        condition=FilenameContainsCondition(
                            patterns=["nsfw_", "_nsfw", "explicit_"],
                            case_sensitive=False,
                        ),
                    ),
                    PipelineNode(
                        name="is_marked_safe",
                        condition=FilenameContainsCondition(
                            patterns=["safe_", "_safe", "sfw_"],
                            case_sensitive=False,
                        ),
                    ),
                    PipelineNode(
                        name="is_a_draft",
                        condition=FilenameContainsCondition(
                            patterns=["draft_", "_draft", "_wip"],
                            case_sensitive=False,
                        ),
                    ),
                ],
            ),
            on_match=NodeOutcome(OutcomeType.CONTINUE),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        # ------------------------------------------------------------------
        # Node 3 — GroupChildResultCondition (NSFW child)
        # ------------------------------------------------------------------
        node_nsfw_hint = PipelineNode(
            name="Filename says NSFW?",
            condition=GroupChildResultCondition(
                group_node_name="Filename category hints",
                child_node_name="is_marked_nsfw",
                expected_result=True,
            ),
            on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.HIDE),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        # ------------------------------------------------------------------
        # Node 4 — GroupChildResultCondition (safe child)
        # ------------------------------------------------------------------
        node_safe_hint = PipelineNode(
            name="Filename says safe?",
            condition=GroupChildResultCondition(
                group_node_name="Filename category hints",
                child_node_name="is_marked_safe",
                expected_result=True,
            ),
            on_match=NodeOutcome(OutcomeType.ACCEPT),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        # ------------------------------------------------------------------
        # Node 5 — EmbeddingCondition
        # ------------------------------------------------------------------
        node_person = PipelineNode(
            name="Person visible?",
            condition=EmbeddingCondition(
                positives=["person", "human face", "portrait", "people"],
                negatives=["landscape", "architecture", "food", "object without people"],
                threshold=0.25,
            ),
            on_match=NodeOutcome(OutcomeType.CONTINUE),
            on_no_match=NodeOutcome(OutcomeType.GOTO, target_node="Related image exists?"),
        )

        # ------------------------------------------------------------------
        # Node 6 — CompositeCondition (OR) containing Embedding + Prompt
        # ------------------------------------------------------------------
        node_sensitive = PipelineNode(
            name="Sensitive content?",
            condition=CompositeCondition(
                operator="OR",
                sub_conditions=[
                    EmbeddingCondition(
                        positives=["explicit content", "nudity", "adult material"],
                        negatives=["clothed", "safe for work", "family friendly"],
                        threshold=0.28,
                    ),
                    PromptCondition(
                        prompts=["nsfw", "nude", "explicit", "adult content"],
                        use_blacklist=False,
                    ),
                ],
            ),
            on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.HIDE),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        # ------------------------------------------------------------------
        # Node 7 — PrototypeCondition
        # ------------------------------------------------------------------
        node_prototype = PipelineNode(
            name="Safe prototype match?",
            condition=PrototypeCondition(
                prototype_directory="prototypes/safe_content",
                negative_prototype_directory="prototypes/unsafe_content",
                threshold=0.23,
            ),
            on_match=NodeOutcome(OutcomeType.GOTO, target_node="Related image exists?"),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        # ------------------------------------------------------------------
        # Node 8 — NodeResultCondition
        # ------------------------------------------------------------------
        node_hint_review = PipelineNode(
            name="Had filename hint?",
            condition=NodeResultCondition(
                node_name="Filename category hints",
                expected_result=True,
            ),
            on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.ADD_MARK),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        # ------------------------------------------------------------------
        # Node 9 — RelatedImageCondition  (GOTO target from nodes 5 and 7)
        # ------------------------------------------------------------------
        node_related = PipelineNode(
            name="Related image exists?",
            condition=RelatedImageCondition(
                edit_suffix="_reviewed",
                search_directory="",
                count_threshold=1,
            ),
            on_match=NodeOutcome(
                OutcomeType.EXECUTE,
                action_type=ClassifierActionType.GENERATE,
                action_modifier="_reviewed",
            ),
            on_no_match=NodeOutcome(OutcomeType.REJECT),
        )

        return ClassifierPipeline(
            name="Example: Full Feature Demo",
            description=(
                "Demo pipeline (inactive). Exercises condition types that do not "
                "require runtime-registered resources: "
                "MediaType → Group(FilenameContains×3) → GroupChildResult×2 → "
                "Embedding → Composite(Embedding+Prompt) → Prototype → "
                "NodeResult → RelatedImage."
            ),
            nodes=[
                node_media_type,
                node_filename_hints,
                node_nsfw_hint,
                node_safe_hint,
                node_person,
                node_sensitive,
                node_prototype,
                node_hint_review,
                node_related,
            ],
            is_active=False,
        )

    @staticmethod
    def build_category_fill_pipeline(
        target_dir_apple: str = "target/A/",
        target_dir_banana: str = "target/B/",
        target_dir_cherry: str = "target/C/",
    ) -> "ClassifierPipeline":
        """
        Demo pipeline for filling a per-category target directory set.

        Illustrates the recommended category-fill pipeline architecture:
        guard → uniqueness check → per-category GENERATE nodes.

        Categories and suffixes
        ───────────────────────
          apple   _apple   → target/A/
          banana  _banana  → target/B/
          cherry  _cherry  → target/C/

        Pipeline flow
        ─────────────
        1  Unknown-suffix guard   CompositeCondition(NOT, UnknownSuffixCondition)
             Passes when every file in the stem group has a recognised suffix.
             Rejects when an unrecognised file is found that cannot be resolved
             by classifier inference, preventing ambiguous generation.

        2  Generate apple         CompositeCondition(AND)
        3  Generate banana          [0] RelatedImageCondition — not a local derivative
        4  Generate cherry              AND no existing variant in working dir
                                    [1] BaseStemMatchCondition(require_match=False,
                                            search_directory="target/X/")
                                        — no file with this base stem in target dir
             on_match  → EXECUTE_AND_CONTINUE GENERATE (dispatch and advance to next node)
             on_no_match → CONTINUE (check next category)

        Behaviour table
        ─────────────────────────────────────────────────────────────────────────
        Image state                         Guard   Cond[0]  Cond[1]  Action
        ─────────────────────────────────────────────────────────────────────────
        Seed, no apple in target            pass    True     True     GENERATE _apple
        Seed, apple in target               pass    True     False    skip → check banana
        Seed, apple variant in working dir  pass    False    —        skip → check banana
        Type-3 derivative (all categories)  pass    False    —        skip all → default
        Unknown suffix, unresolvable        fail    —        —        REJECT
        ─────────────────────────────────────────────────────────────────────────

        The pipeline is inactive by default. Replace the placeholder target
        directory paths with absolute paths before activating.

        Seed guard note
        ───────────────────────
        Each category node could optionally include a ClassifierRankCondition
        (categories=[the other categories], negate=True) as a third AND
        sub-condition, to skip generation when the seed image is already
        classified as one of the other categories. negate=True flips the
        condition's match so True means "none of the listed categories rank
        highly" -- matching the True-means-clear-to-generate polarity of the
        other two sub-conditions -- without wrapping it in a separate NOT
        composite (unsupported by the editor's sub-condition picker) or paying
        for a second, separate classifier evaluation. This is omitted here
        pending classifier validation.

        processed_stems note
        ─────────────────────────────
        Pass a shared set() as processed_stems to run_pipeline() to skip
        derivative images whose stem group has already been evaluated.
        Use RELATED_IMAGE ascending sort so seeds are evaluated first.
        """
        CATEGORY_MAP = {"Apple": "_apple", "Banana": "_banana", "Cherry": "_cherry"}
        ALL_SUFFIXES = list(CATEGORY_MAP.values())

        # ------------------------------------------------------------------
        # Node 1 — Unknown-suffix guard
        # Guards against stem groups that contain files with unrecognised
        # suffixes, which could indicate a miscategorised or manually renamed
        # file that would corrupt the representative set.
        # ------------------------------------------------------------------
        node_guard = PipelineNode(
            name="Unknown-suffix guard",
            condition=CompositeCondition(
                operator="NOT",
                sub_conditions=[
                    UnknownSuffixCondition(
                        expected_suffixes=ALL_SUFFIXES,
                        use_base_directory=True,
                        # classifier_name: set to a seed classifier to attempt inference
                        # on unrecognised files before rejecting.
                    ),
                ],
            ),
            on_match=NodeOutcome(OutcomeType.CONTINUE),   # clean → proceed to category nodes
            on_no_match=NodeOutcome(OutcomeType.REJECT),  # anomaly → reject without generating
        )

        # ------------------------------------------------------------------
        # Category node factory
        # Each category node is a two-condition AND gate:
        #   [0] RelatedImageCondition — True when no local variant exists AND
        #       the current image is not a working-dir derivative (type-3 guard).
        #   [1] BaseStemMatchCondition(require_match=False) — True when no file
        #       with this base stem exists in the category's target directory,
        #       meaning generation is still needed.
        # Both must be True to dispatch GENERATE. Either being False means the
        # category is already covered; the node falls through via on_no_match=CONTINUE.
        # ------------------------------------------------------------------
        def _make_category_node(name: str, suffix: str, target_dir: str) -> PipelineNode:
            return PipelineNode(
                name=name,
                condition=CompositeCondition(
                    operator="AND",
                    sub_conditions=[
                        # [0] Not a local derivative; no variant in working dir.
                        # use_configured_search_directories=False → checks base_directory
                        # (the working dir passed to run_pipeline) only.
                        RelatedImageCondition(
                            edit_suffix=suffix,
                            use_configured_search_directories=False,
                            count_threshold=1,
                        ),
                        # [1] No file with this base stem exists in the target category dir.
                        # search_directory pins the check to the specific category directory,
                        # so a file with the wrong suffix in the right directory still
                        # correctly signals that the category is covered.
                        BaseStemMatchCondition(
                            require_match=False,
                            search_directory=target_dir,
                        ),
                    ],
                ),
                on_match=NodeOutcome(
                    OutcomeType.EXECUTE_AND_CONTINUE,
                    action_type=ClassifierActionType.GENERATE,
                    action_modifier=suffix,
                ),
                on_no_match=NodeOutcome(OutcomeType.CONTINUE),
            )

        node_apple  = _make_category_node("Generate apple",  "_apple",  target_dir_apple)
        node_banana = _make_category_node("Generate banana", "_banana", target_dir_banana)
        node_cherry = _make_category_node("Generate cherry", "_cherry", target_dir_cherry)

        # ------------------------------------------------------------------
        # Node 2 — Stem uniqueness check: rejects when the base stem matches
        # more than max_stem_group_size files across the target dirs (e.g. a
        # bare label like "photo" colliding with thousands of files) --
        # search_directory empty uses directories_to_search_for_related_images
        # (all target dirs). on_match=REJECT (not unique); on_no_match=CONTINUE
        # (within limit, proceed to category nodes).
        # ------------------------------------------------------------------
        node_uniqueness = PipelineNode(
            name="Stem uniqueness check",
            condition=BaseStemMatchCondition(
                max_stem_group_size=-1,
            ),
            on_match=NodeOutcome(OutcomeType.REJECT),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        return ClassifierPipeline(
            name="Example: Representation Set Generator (apple / banana / cherry)",
            description=(
                "Demo category-fill pipeline (inactive). Fills per-category target "
                "subdirectories from a working directory of seed images. "
                "Categories: apple → target/A/, banana → target/B/, cherry → target/C/. "
                "Guard node rejects stem groups with unrecognised suffixes. "
                "Uniqueness node rejects stems with too many existing matches in the "
                "target dirs (non-unique base stem). "
                "Each category node generates if and only if (a) the image is not a "
                "local derivative and (b) the target directory does not already contain "
                "a file with this base stem."
            ),
            nodes=[node_guard, node_uniqueness, node_apple, node_banana, node_cherry],
            is_active=False,
            category_map=CATEGORY_MAP,
        )

    @staticmethod
    def build_scramble_coherence_pipeline(
        target_dir_coherent: str = "target/coherent/",
        target_dir_semiinco: str = "target/semi_incoherent/",
        target_dir_inco: str = "target/incoherent/",
    ) -> "ClassifierPipeline":
        """
        Demo pipeline for building a three-bucket coherence training set.

        Categories and suffixes
        ───────────────────────
          coherent         _coherent  → GENERATE (image-generation pass) → target/coherent/
          semi-incoherent  _semiinco  → SCRAMBLE (semi_scramble_image)   → target/semi_incoherent/
          incoherent       _inco      → SCRAMBLE (scramble_image)         → target/incoherent/

        Each category node checks two AND conditions:
          [0] RelatedImageCondition — no existing variant with this suffix in
              the working dir (use_configured_search_directories=False)
          [1] BaseStemMatchCondition(require_match=False, search_directory=target_dir)
              — no file for this base stem already exists in the category target directory

        The pipeline is inactive by default. Set target directories, configure
        directories_to_search_for_related_images to include the target dirs,
        and attach a coherence classifier before activating.
        """
        CATEGORY_MAP = {
            "Coherent":        "_coherent",
            "Semi-incoherent": "_semiinco",
            "Incoherent":      "_inco",
        }
        ALL_SUFFIXES = list(CATEGORY_MAP.values())

        node_guard = PipelineNode(
            name="Unknown-suffix guard",
            condition=CompositeCondition(
                operator="NOT",
                sub_conditions=[
                    UnknownSuffixCondition(
                        expected_suffixes=ALL_SUFFIXES,
                        use_base_directory=True,
                    ),
                ],
            ),
            on_match=NodeOutcome(OutcomeType.CONTINUE),
            on_no_match=NodeOutcome(OutcomeType.REJECT),
        )

        node_uniqueness = PipelineNode(
            name="Stem uniqueness check",
            condition=BaseStemMatchCondition(max_stem_group_size=-1),
            on_match=NodeOutcome(OutcomeType.REJECT),
            on_no_match=NodeOutcome(OutcomeType.CONTINUE),
        )

        def _make_node(
            name: str,
            suffix: str,
            action: ClassifierActionType,
            target_dir: str = "",
        ) -> PipelineNode:
            return PipelineNode(
                name=name,
                condition=CompositeCondition(
                    operator="AND",
                    sub_conditions=[
                        RelatedImageCondition(
                            edit_suffix=suffix,
                            use_configured_search_directories=False,
                            count_threshold=1,
                        ),
                        BaseStemMatchCondition(
                            require_match=False,
                            search_directory=target_dir,
                        ),
                    ],
                ),
                on_match=NodeOutcome(
                    OutcomeType.EXECUTE_AND_CONTINUE,
                    action_type=action,
                    action_modifier=suffix,
                ),
                on_no_match=NodeOutcome(OutcomeType.CONTINUE),
            )

        node_coherent = _make_node(
            "Generate coherent variant",
            "_coherent",
            ClassifierActionType.GENERATE,
            target_dir_coherent,
        )
        node_semi = _make_node(
            "Scramble semi-incoherent variant",
            "_semiinco",
            ClassifierActionType.SCRAMBLE,
            target_dir_semiinco,
        )
        node_incoherent = _make_node(
            "Scramble incoherent variant",
            "_inco",
            ClassifierActionType.SCRAMBLE,
            target_dir_inco,
        )

        return ClassifierPipeline(
            name="Example: Scramble Coherence Set Builder",
            description=(
                "Demo pipeline (inactive). Builds a three-bucket coherence training set "
                "from a directory of seed images. "
                "coherent → GENERATE (_coherent suffix, target/coherent/); "
                "semi-incoherent → SCRAMBLE via semi_scramble_image (_semiinco suffix, target/semi_incoherent/); "
                "incoherent → SCRAMBLE via scramble_image (_inco suffix, target/incoherent/). "
                "Set target directories, configure search dirs, and add a coherence classifier before activating."
            ),
            nodes=[node_guard, node_uniqueness, node_coherent, node_semi, node_incoherent],
            is_active=False,
            category_map=CATEGORY_MAP,
            seed_category="Coherent",
        )

    @staticmethod
    def store() -> None:
        app_info_cache.set_meta(
            _CACHE_KEY,
            [p.to_dict() for p in ClassifierPipelines.pipelines],
        )

    @staticmethod
    def get_pipeline_by_name(name: str) -> Optional[ClassifierPipeline]:
        for p in ClassifierPipelines.pipelines:
            if p.name == name:
                return p
        return None

    @staticmethod
    def get_all_pipelines() -> list[ClassifierPipeline]:
        return ClassifierPipelines.pipelines

    @staticmethod
    def get_prevalidation_pipelines() -> list["PrevalidationPipeline"]:
        return ClassifierPipelines._prevalidation_pipelines

    @staticmethod
    def get_action_pipelines() -> list[ClassifierPipeline]:
        return ClassifierPipelines._action_pipelines

    @staticmethod
    def add_pipeline(pipeline: ClassifierPipeline) -> None:
        ClassifierPipelines.pipelines.append(pipeline)
        ClassifierPipelines._rebuild_type_cache()

    @staticmethod
    def remove_pipeline(name: str) -> None:
        ClassifierPipelines.pipelines = [
            p for p in ClassifierPipelines.pipelines if p.name != name
        ]
        ClassifierPipelines._rebuild_type_cache()

    @staticmethod
    def get_active_pipelines_for_profile(
        profile_name: Optional[str],
    ) -> list["PrevalidationPipeline"]:
        """Return active PrevalidationPipelines whose profile matches."""
        return [
            p for p in ClassifierPipelines._prevalidation_pipelines
            if p.is_active and p.profile_name == profile_name
        ]

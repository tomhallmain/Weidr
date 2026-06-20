"""
ClassifierPipeline: multi-node decision-tree classifier.

A ClassifierPipeline is an ordered list of PipelineNodes.  Each node holds a
single NodeCondition and two NodeOutcomes (on_match / on_no_match).  Execution
walks the list from the first node, branching or halting according to the
outcomes, and eventually returns a ClassifierActionType (or None for no action).

Storage lives in app_info_cache under "classifier_pipelines".  No existing
ClassifierAction or Prevalidation data is touched.

Phase 1 — core model only (serialization + validation).
Phase 2 — execution engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

from utils.app_info_cache import app_info_cache
from utils.constants import ClassifierActionType, CompareMediaType
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("classifier_pipeline")


# ---------------------------------------------------------------------------
# Condition types
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingCondition:
    """CLIP text-embedding similarity check."""
    condition_type: ClassVar[str] = "embedding"

    positives: Optional[list] = None
    negatives: Optional[list] = None
    threshold: float = 0.23

    def __post_init__(self):
        self.positives = list(self.positives) if self.positives else []
        self.negatives = list(self.negatives) if self.negatives else []

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "positives": self.positives,
            "negatives": self.negatives,
            "threshold": self.threshold,
        }

    def summary(self) -> str:
        pos = ", ".join(self.positives) if self.positives else "(none)"
        return f"Embedding(+[{pos}], thresh={self.threshold})"


@dataclass
class ClassifierRankCondition:
    """Checks where specific categories appear in a model's ranked output."""
    condition_type: ClassVar[str] = "classifier_rank"

    classifier_name: str = ""
    categories: Optional[list] = None
    min_rank: int = 1
    max_rank: int = 1
    min_confidence: float = 0.0

    def __post_init__(self):
        self.categories = list(self.categories) if self.categories else []

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "classifier_name": self.classifier_name,
            "categories": self.categories,
            "min_rank": self.min_rank,
            "max_rank": self.max_rank,
            "min_confidence": self.min_confidence,
        }

    def summary(self) -> str:
        cats = ", ".join(self.categories) if self.categories else "(none)"
        rank = f"rank {self.min_rank}" if self.min_rank == self.max_rank else f"rank {self.min_rank}-{self.max_rank}"
        return f"ClassifierRank({self.classifier_name}, [{cats}], {rank})"


@dataclass
class PrototypeCondition:
    """Embedding prototype similarity check."""
    condition_type: ClassVar[str] = "prototype"

    prototype_directory: str = ""
    negative_prototype_directory: str = ""
    threshold: float = 0.23
    negative_lambda: float = 0.5

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "prototype_directory": self.prototype_directory,
            "negative_prototype_directory": self.negative_prototype_directory,
            "threshold": self.threshold,
            "negative_lambda": self.negative_lambda,
        }

    def summary(self) -> str:
        return f"Prototype(thresh={self.threshold})"


@dataclass
class PromptCondition:
    """Prompt / blacklist text-detection check."""
    condition_type: ClassVar[str] = "prompt"

    prompts: Optional[list] = None
    use_blacklist: bool = False

    def __post_init__(self):
        self.prompts = list(self.prompts) if self.prompts else []

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "prompts": self.prompts,
            "use_blacklist": self.use_blacklist,
        }

    def summary(self) -> str:
        if self.use_blacklist:
            return "Blacklist"
        terms = ", ".join(self.prompts) if self.prompts else "(none)"
        return f"Prompts([{terms}])"


@dataclass
class FilenameContainsCondition:
    """Checks whether the media filename contains any of the given substrings."""
    condition_type: ClassVar[str] = "filename_contains"

    patterns: Optional[list] = None
    case_sensitive: bool = False

    def __post_init__(self):
        self.patterns = list(self.patterns) if self.patterns else []

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "patterns": self.patterns,
            "case_sensitive": self.case_sensitive,
        }

    def summary(self) -> str:
        terms = ", ".join(self.patterns) if self.patterns else "(none)"
        cs = "cs" if self.case_sensitive else "ci"
        return f"FilenameContains([{terms}], {cs})"


@dataclass
class MediaTypeCondition:
    """Tests whether the file's resolved media type is one of the listed types."""
    condition_type: ClassVar[str] = "media_type"

    media_types: Optional[list] = None   # list[CompareMediaType]

    def __post_init__(self):
        raw = list(self.media_types) if self.media_types else []
        self.media_types = [
            mt if isinstance(mt, CompareMediaType) else CompareMediaType(mt)
            for mt in raw
        ]

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "media_types": [mt.value for mt in self.media_types],
        }

    def summary(self) -> str:
        names = ", ".join(mt.value for mt in self.media_types) if self.media_types else "(none)"
        return f"MediaType([{names}])"


@dataclass
class LookaheadCondition:
    """References a named Lookahead check."""
    condition_type: ClassVar[str] = "lookahead"

    lookahead_name: str = ""

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "lookahead_name": self.lookahead_name,
        }

    def summary(self) -> str:
        return f"Lookahead({self.lookahead_name})"


@dataclass
class NodeResultCondition:
    """References the boolean result of an earlier pipeline node."""
    condition_type: ClassVar[str] = "node_result"

    node_name: str = ""
    expected_result: bool = True

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "node_name": self.node_name,
            "expected_result": self.expected_result,
        }

    def summary(self) -> str:
        val = "True" if self.expected_result else "False"
        return f"NodeResult({self.node_name}={val})"


@dataclass
class CompositeCondition:
    """AND / OR / NOT / XOR composition of other conditions."""
    condition_type: ClassVar[str] = "composite"
    VALID_OPERATORS: ClassVar[set] = {"AND", "OR", "NOT", "XOR"}

    operator: str = "AND"
    sub_conditions: Optional[list] = None

    def __post_init__(self):
        self.sub_conditions = list(self.sub_conditions) if self.sub_conditions else []

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "operator": self.operator,
            "sub_conditions": [c.to_dict() for c in self.sub_conditions],
        }

    def summary(self) -> str:
        parts = " | ".join(c.summary() for c in self.sub_conditions)
        return f"Composite({self.operator}: {parts})"


@dataclass
class BaseStemMatchCondition:
    """Matches when a file sharing the same filename base stem exists in the configured search directories."""
    condition_type: ClassVar[str] = "base_stem_match"

    require_match: bool = True  # False = pass when NO related file is found

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "require_match": self.require_match,
        }

    def summary(self) -> str:
        mode = "found" if self.require_match else "not found"
        return f"BaseStemMatch(require={mode})"


@dataclass
class RelatedImageCondition:
    """Checks whether a generate action should run based on downstream image state."""
    condition_type: ClassVar[str] = "related_image"

    edit_suffix: str = ""
    search_directory: str = ""  # empty = use base_directory from pipeline run
    count_threshold: int = 1

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "edit_suffix": self.edit_suffix,
            "search_directory": self.search_directory,
            "count_threshold": self.count_threshold,
        }

    def summary(self) -> str:
        return f"RelatedImage(suffix={self.edit_suffix!r}, threshold={self.count_threshold})"


@dataclass
class GroupCondition:
    """
    An ordered group of child PipelineNodes evaluated as a unit.

    Each child's condition is evaluated independently and its result is stored
    in the shared node_results dict under the key ``"<outer_node>/<child_name>"``.
    The group's own boolean result is OR (any child matched) or AND (all matched).

    Child node on_match / on_no_match outcomes are intentionally ignored — routing
    is controlled by the outer pipeline node that holds this condition.
    """
    condition_type: ClassVar[str] = "group"
    VALID_OPERATORS: ClassVar[set] = {"OR", "AND"}

    operator: str = "OR"
    nodes: Optional[list] = None   # list[PipelineNode]

    def __post_init__(self):
        self.nodes = list(self.nodes) if self.nodes else []

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "operator": self.operator,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    def summary(self) -> str:
        n = len(self.nodes)
        names = ", ".join(c.name for c in self.nodes[:3])
        suffix = f", +{n - 3}" if n > 3 else ""
        return f"Group({self.operator}: {names}{suffix})"


@dataclass
class GroupChildResultCondition:
    """
    Checks the stored result of a specific child node inside a prior group node.

    The runner stores child results under ``"<group_node_name>/<child_name>"``
    so this condition can look them up without any extra runtime state.
    """
    condition_type: ClassVar[str] = "group_child_result"

    group_node_name: str = ""
    child_node_name: str = ""
    expected_result: bool = True

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "group_node_name": self.group_node_name,
            "child_node_name": self.child_node_name,
            "expected_result": self.expected_result,
        }

    def summary(self) -> str:
        val = "True" if self.expected_result else "False"
        return f"GroupChild({self.group_node_name}/{self.child_node_name}={val})"


# Union type alias (informational only — Python does not enforce it at runtime)
NodeCondition = (
    EmbeddingCondition
    | ClassifierRankCondition
    | PrototypeCondition
    | PromptCondition
    | FilenameContainsCondition
    | MediaTypeCondition
    | LookaheadCondition
    | NodeResultCondition
    | CompositeCondition
    | BaseStemMatchCondition
    | RelatedImageCondition
    | GroupCondition
    | GroupChildResultCondition
)


def _condition_from_dict(d: dict):
    """Deserialize any NodeCondition from a dict."""
    ct = d.get("condition_type", "")
    if ct == "embedding":
        return EmbeddingCondition(
            positives=d.get("positives", []),
            negatives=d.get("negatives", []),
            threshold=d.get("threshold", 0.23),
        )
    if ct == "classifier_rank":
        return ClassifierRankCondition(
            classifier_name=d.get("classifier_name", ""),
            categories=d.get("categories", []),
            min_rank=d.get("min_rank", 1),
            max_rank=d.get("max_rank", 1),
            min_confidence=d.get("min_confidence", 0.0),
        )
    if ct == "prototype":
        return PrototypeCondition(
            prototype_directory=d.get("prototype_directory", ""),
            negative_prototype_directory=d.get("negative_prototype_directory", ""),
            threshold=d.get("threshold", 0.23),
            negative_lambda=d.get("negative_lambda", 0.5),
        )
    if ct == "prompt":
        return PromptCondition(
            prompts=d.get("prompts", []),
            use_blacklist=d.get("use_blacklist", False),
        )
    if ct == "filename_contains":
        return FilenameContainsCondition(
            patterns=d.get("patterns", []),
            case_sensitive=d.get("case_sensitive", False),
        )
    if ct == "media_type":
        return MediaTypeCondition(media_types=d.get("media_types", []))
    if ct == "lookahead":
        return LookaheadCondition(lookahead_name=d.get("lookahead_name", ""))
    if ct == "node_result":
        return NodeResultCondition(
            node_name=d.get("node_name", ""),
            expected_result=d.get("expected_result", True),
        )
    if ct == "composite":
        return CompositeCondition(
            operator=d.get("operator", "AND"),
            sub_conditions=[_condition_from_dict(c) for c in d.get("sub_conditions", [])],
        )
    if ct == "base_stem_match":
        return BaseStemMatchCondition(
            require_match=d.get("require_match", True),
        )
    if ct == "related_image":
        return RelatedImageCondition(
            edit_suffix=d.get("edit_suffix", ""),
            search_directory=d.get("search_directory", ""),
            count_threshold=d.get("count_threshold", 1),
        )
    if ct == "group":
        return GroupCondition(
            operator=d.get("operator", "OR"),
            nodes=[PipelineNode.from_dict(n) for n in d.get("nodes", [])],
        )
    if ct == "group_child_result":
        return GroupChildResultCondition(
            group_node_name=d.get("group_node_name", ""),
            child_node_name=d.get("child_node_name", ""),
            expected_result=d.get("expected_result", True),
        )
    raise ValueError(f"Unknown condition_type: {ct!r}")


# ---------------------------------------------------------------------------
# NodeOutcome
# ---------------------------------------------------------------------------

class OutcomeType(str, Enum):
    CONTINUE = "CONTINUE"   # advance to next node in order
    GOTO     = "GOTO"       # jump to named node (forward only)
    EXECUTE  = "EXECUTE"    # fire action and halt
    ACCEPT   = "ACCEPT"     # halt with no action (explicit pass)
    REJECT   = "REJECT"     # halt with pipeline's default_reject_action


@dataclass
class NodeOutcome:
    outcome_type: OutcomeType = field(default=OutcomeType.CONTINUE)
    target_node: Optional[str] = None
    action_type: Optional[ClassifierActionType] = None
    action_modifier: str = ""

    def __post_init__(self):
        if not isinstance(self.outcome_type, OutcomeType):
            self.outcome_type = OutcomeType(self.outcome_type)
        if self.action_type is not None and not isinstance(self.action_type, ClassifierActionType):
            self.action_type = ClassifierActionType[self.action_type]

    def to_dict(self) -> dict:
        return {
            "outcome_type": self.outcome_type.value,
            "target_node": self.target_node,
            "action_type": self.action_type.value if self.action_type else None,
            "action_modifier": self.action_modifier,
        }

    def summary(self) -> str:
        if self.outcome_type == OutcomeType.EXECUTE:
            base = f"EXECUTE: {self.action_type.value if self.action_type else '?'}"
            if self.action_modifier:
                base += f" → {self.action_modifier}"
            return base
        if self.outcome_type == OutcomeType.GOTO:
            return f"GOTO: {self.target_node}"
        return self.outcome_type.value

    @staticmethod
    def from_dict(d: dict) -> "NodeOutcome":
        return NodeOutcome(
            outcome_type=d.get("outcome_type", OutcomeType.CONTINUE.value),
            target_node=d.get("target_node"),
            action_type=d.get("action_type"),
            action_modifier=d.get("action_modifier", ""),
        )

    @staticmethod
    def continue_() -> "NodeOutcome":
        return NodeOutcome(OutcomeType.CONTINUE)

    @staticmethod
    def accept() -> "NodeOutcome":
        return NodeOutcome(OutcomeType.ACCEPT)


# ---------------------------------------------------------------------------
# PipelineNode
# ---------------------------------------------------------------------------

@dataclass
class PipelineNode:
    name: str = ""
    condition: object = None          # NodeCondition; defaulted in __post_init__
    on_match: Optional[NodeOutcome] = None
    on_no_match: Optional[NodeOutcome] = None

    def __post_init__(self):
        if self.condition is None:
            self.condition = EmbeddingCondition()
        if self.on_match is None:
            self.on_match = NodeOutcome.continue_()
        if self.on_no_match is None:
            self.on_no_match = NodeOutcome.accept()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "condition": self.condition.to_dict(),
            "on_match": self.on_match.to_dict(),
            "on_no_match": self.on_no_match.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "PipelineNode":
        return PipelineNode(
            name=d.get("name", ""),
            condition=_condition_from_dict(d.get("condition", {"condition_type": "embedding"})),
            on_match=NodeOutcome.from_dict(d.get("on_match", {})),
            on_no_match=NodeOutcome.from_dict(d.get("on_no_match", {})),
        )

    def condition_summary(self) -> str:
        return self.condition.summary() if self.condition else "(no condition)"


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
        """
        errors: list[str] = []
        if not self.name.strip():
            errors.append("Pipeline name is empty.")

        node_names: list[str] = []
        seen_names: set[str] = set()
        for node in self.nodes:
            if not node.name.strip():
                errors.append("A node has an empty name.")
            elif node.name in seen_names:
                errors.append(f"Duplicate node name: {node.name!r}.")
            else:
                seen_names.add(node.name)
            node_names.append(node.name)

        all_names = set(node_names)

        for i, node in enumerate(self.nodes):
            defined_before = set(node_names[:i])

            for outcome in (node.on_match, node.on_no_match):
                if outcome.outcome_type == OutcomeType.GOTO:
                    if not outcome.target_node:
                        errors.append(f"Node {node.name!r}: GOTO has no target.")
                    elif outcome.target_node not in all_names:
                        errors.append(
                            f"Node {node.name!r}: GOTO target {outcome.target_node!r} does not exist."
                        )
                    else:
                        target_idx = node_names.index(outcome.target_node)
                        if target_idx <= i:
                            errors.append(
                                f"Node {node.name!r}: GOTO target {outcome.target_node!r} "
                                f"must be a later node (cycle prevention)."
                            )
                if outcome.outcome_type == OutcomeType.EXECUTE and outcome.action_type is None:
                    errors.append(f"Node {node.name!r}: EXECUTE outcome has no action_type.")

            errors.extend(
                self._validate_condition(node.condition, node.name, defined_before)
            )

            # RelatedImageCondition/GENERATE outcome consistency
            if isinstance(node.condition, RelatedImageCondition):
                for outcome in (node.on_match, node.on_no_match):
                    if (outcome.outcome_type == OutcomeType.EXECUTE
                            and outcome.action_type == ClassifierActionType.GENERATE
                            and outcome.action_modifier != node.condition.edit_suffix):
                        errors.append(
                            f"Node {node.name!r}: RelatedImageCondition.edit_suffix "
                            f"({node.condition.edit_suffix!r}) must match the GENERATE "
                            f"outcome's action_modifier ({outcome.action_modifier!r})."
                        )

        return errors

    def _validate_condition(self, condition, node_name: str,
                            defined_before: set[str]) -> list[str]:
        errors: list[str] = []

        if isinstance(condition, NodeResultCondition):
            if not condition.node_name:
                errors.append(f"Node {node_name!r}: NodeResultCondition has no node_name.")
            elif condition.node_name not in defined_before:
                errors.append(
                    f"Node {node_name!r}: NodeResultCondition references "
                    f"{condition.node_name!r} which is not a prior node."
                )

        elif isinstance(condition, ClassifierRankCondition):
            if not condition.classifier_name:
                errors.append(
                    f"Node {node_name!r}: ClassifierRankCondition has no classifier_name."
                )
            else:
                try:
                    from image.image_classifier_manager import image_classifier_manager
                    names = image_classifier_manager.get_model_names()
                    if names and condition.classifier_name not in names:
                        errors.append(
                            f"Node {node_name!r}: classifier {condition.classifier_name!r} "
                            f"is not registered."
                        )
                except Exception:
                    pass  # manager not available during unit tests — skip
            if condition.min_rank < 1:
                errors.append(f"Node {node_name!r}: min_rank must be ≥ 1.")
            if condition.max_rank < condition.min_rank:
                errors.append(f"Node {node_name!r}: max_rank must be ≥ min_rank.")

        elif isinstance(condition, FilenameContainsCondition):
            if not condition.patterns:
                errors.append(
                    f"Node {node_name!r}: FilenameContainsCondition has no patterns."
                )

        elif isinstance(condition, MediaTypeCondition):
            if not condition.media_types:
                errors.append(
                    f"Node {node_name!r}: MediaTypeCondition has no media_types."
                )

        elif isinstance(condition, LookaheadCondition):
            if not condition.lookahead_name:
                errors.append(f"Node {node_name!r}: LookaheadCondition has no lookahead_name.")
            else:
                try:
                    from compare.lookahead import Lookahead
                    if Lookahead.get_lookahead_by_name(condition.lookahead_name) is None:
                        errors.append(
                            f"Node {node_name!r}: lookahead {condition.lookahead_name!r} "
                            f"is not defined."
                        )
                except Exception:
                    pass

        elif isinstance(condition, RelatedImageCondition):
            if not condition.edit_suffix:
                errors.append(f"Node {node_name!r}: RelatedImageCondition has no edit_suffix.")
            if condition.search_directory and not os.path.isdir(condition.search_directory):
                errors.append(
                    f"Node {node_name!r}: RelatedImageCondition.search_directory "
                    f"({condition.search_directory!r}) is not a valid directory."
                )

        elif isinstance(condition, CompositeCondition):
            if condition.operator not in CompositeCondition.VALID_OPERATORS:
                errors.append(
                    f"Node {node_name!r}: unknown composite operator {condition.operator!r}."
                )
            n = len(condition.sub_conditions)
            if condition.operator == "NOT" and n != 1:
                errors.append(
                    f"Node {node_name!r}: NOT requires exactly 1 sub-condition, got {n}."
                )
            if condition.operator == "XOR" and n != 2:
                errors.append(
                    f"Node {node_name!r}: XOR requires exactly 2 sub-conditions, got {n}."
                )
            if condition.operator in ("AND", "OR") and n < 2:
                errors.append(
                    f"Node {node_name!r}: {condition.operator} requires ≥ 2 sub-conditions, got {n}."
                )
            for sub in condition.sub_conditions:
                errors.extend(self._validate_condition(sub, node_name, defined_before))

        elif isinstance(condition, GroupCondition):
            if condition.operator not in GroupCondition.VALID_OPERATORS:
                errors.append(
                    f"Node {node_name!r}: GroupCondition has unknown operator {condition.operator!r}."
                )
            if not condition.nodes:
                errors.append(f"Node {node_name!r}: GroupCondition has no child nodes.")
            else:
                seen_children: set[str] = set()
                for child in condition.nodes:
                    if not child.name.strip():
                        errors.append(
                            f"Node {node_name!r}: GroupCondition child has an empty name."
                        )
                    elif child.name in seen_children:
                        errors.append(
                            f"Node {node_name!r}: GroupCondition duplicate child name {child.name!r}."
                        )
                    else:
                        seen_children.add(child.name)
                    errors.extend(
                        self._validate_condition(child.condition, f"{node_name}/{child.name}", defined_before)
                    )

        elif isinstance(condition, GroupChildResultCondition):
            if not condition.group_node_name:
                errors.append(
                    f"Node {node_name!r}: GroupChildResultCondition has no group_node_name."
                )
            if not condition.child_node_name:
                errors.append(
                    f"Node {node_name!r}: GroupChildResultCondition has no child_node_name."
                )
            if condition.group_node_name and condition.group_node_name not in defined_before:
                errors.append(
                    f"Node {node_name!r}: GroupChildResultCondition references group "
                    f"{condition.group_node_name!r} which is not a prior node."
                )

        return errors

    # ------------------------------------------------------------------
    # Flow preview (plain text, no Qt dependency)
    # ------------------------------------------------------------------

    def flow_summary(self) -> str:
        """Multi-line summary: one node per two lines, suitable for a scrollable list cell."""
        if not self.nodes:
            return _("(empty)")
        _ABBREV = {
            "embedding": "Embedding",
            "classifier_rank": "ClsRank",
            "prototype": "Prototype",
            "prompt": "Prompt",
            "lookahead": "Lookahead",
            "node_result": "NodeResult",
            "composite": "Composite",
            "group": "Group",
            "group_child_result": "GroupChild",
        }
        lines = []
        for node in self.nodes:
            cond_type = getattr(node.condition, "condition_type", "")
            if cond_type == "group":
                op = getattr(node.condition, "operator", "OR")
                n = len(getattr(node.condition, "nodes", []))
                cond_label = f"Group({op},{n})"
            else:
                cond_label = _ABBREV.get(cond_type, cond_type)
            lines.append(f"{node.name} [{cond_label}]")
            lines.append(f"  ✓ {node.on_match.summary()}  ✗ {node.on_no_match.summary()}")
        if self.default_action:
            lines.append(f"(end) → {self.default_action.value}")
        return "\n".join(lines)

    def flow_preview(self) -> str:
        if not self.nodes:
            return "(no nodes)"
        lines: list[str] = []
        for node in self.nodes:
            lines.append(f"[{node.name}: {node.condition_summary()}]")
            if isinstance(node.condition, GroupCondition):
                op = node.condition.operator
                for child in node.condition.nodes:
                    lines.append(f"  {'·'} {child.name}: {child.condition_summary()}")
                lines.append(f"  ({op} of {len(node.condition.nodes)} children)")
            lines.append(f"  ✓ → {node.on_match.summary()}")
            lines.append(f"  ✗ → {node.on_no_match.summary()}")
            lines.append("")
        if self.default_action:
            lines.append(f"(end) → {self.default_action.value}")
        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
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
        }

    @staticmethod
    def from_dict(d: dict) -> "ClassifierPipeline":
        def _opt_action(val):
            if not val:
                return None
            return ClassifierActionType[val] if isinstance(val, str) else val

        return ClassifierPipeline(
            name=d.get("name", _("New Pipeline")),
            description=d.get("description", ""),
            nodes=[PipelineNode.from_dict(n) for n in d.get("nodes", [])],
            default_action=_opt_action(d.get("default_action")),
            default_reject_action=_opt_action(d.get("default_reject_action")),
            is_active=d.get("is_active", True),
            applies_to_media_types=d.get("applies_to_media_types"),
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

        return PrevalidationPipeline(
            profile_name=d.get("profile_name"),
            name=d.get("name", _("New Pipeline")),
            description=d.get("description", ""),
            nodes=[PipelineNode.from_dict(n) for n in d.get("nodes", [])],
            default_action=_opt_action(d.get("default_action")),
            default_reject_action=_opt_action(d.get("default_reject_action")),
            is_active=d.get("is_active", True),
            applies_to_media_types=d.get("applies_to_media_types"),
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

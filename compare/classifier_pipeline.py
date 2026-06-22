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
from utils.constants import ClassifierActionType, CompareMediaType, ImageGenerationType
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

    def display_summary(self) -> str:
        pos = ", ".join(self.positives) if self.positives else _("(none)")
        return _("Embedding") + f"(+[{pos}], thresh={self.threshold})"


@dataclass
class ClassifierRankCondition:
    """Checks where specific categories appear in a model's ranked output."""
    condition_type: ClassVar[str] = "classifier_rank"

    classifier_name: str = ""
    categories: Optional[list] = None
    min_rank: int = 1
    max_rank: int = 1
    min_confidence: float = 0.0
    # When True and categories is empty, the runner substitutes the pipeline's
    # category_map values at evaluation time.  Set this instead of listing every
    # category explicitly when the condition should always match the pipeline's
    # own category set.
    inherit_categories: bool = False

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
            "inherit_categories": self.inherit_categories,
        }

    def summary(self) -> str:
        if self.inherit_categories and not self.categories:
            cats = "(pipeline categories)"
        else:
            cats = ", ".join(self.categories) if self.categories else "(none)"
        rank = f"rank {self.min_rank}" if self.min_rank == self.max_rank else f"rank {self.min_rank}-{self.max_rank}"
        return f"ClassifierRank({self.classifier_name}, [{cats}], {rank})"

    def display_summary(self) -> str:
        if self.inherit_categories and not self.categories:
            cats = _("(pipeline categories)")
        else:
            cats = ", ".join(self.categories) if self.categories else _("(none)")
        rank = (
            _("rank {0}").format(self.min_rank)
            if self.min_rank == self.max_rank
            else _("rank {0}-{1}").format(self.min_rank, self.max_rank)
        )
        return _("ClsRank") + f"({self.classifier_name}, [{cats}], {rank})"


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

    def display_summary(self) -> str:
        return _("Prototype") + f"(thresh={self.threshold})"


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

    def display_summary(self) -> str:
        if self.use_blacklist:
            return _("Blacklist")
        terms = ", ".join(self.prompts) if self.prompts else _("(none)")
        return _("Prompts") + f"([{terms}])"


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

    def display_summary(self) -> str:
        terms = ", ".join(self.patterns) if self.patterns else _("(none)")
        cs = _("case-sensitive") if self.case_sensitive else _("case-insensitive")
        return _("FilenameContains") + f"([{terms}], {cs})"


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

    def display_summary(self) -> str:
        names = ", ".join(mt.get_translation() for mt in self.media_types) if self.media_types else _("(none)")
        return _("MediaType") + f"([{names}])"


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

    def display_summary(self) -> str:
        return _("Lookahead: {0}").format(self.lookahead_name)


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

    def display_summary(self) -> str:
        val = _("yes") if self.expected_result else _("no")
        return _("NodeResult") + f"({self.node_name}={val})"


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

    def display_summary(self) -> str:
        parts = " | ".join(c.display_summary() for c in self.sub_conditions)
        return _("Composite") + f"({self.operator}: {parts})"


@dataclass
class BaseStemMatchCondition:
    """Matches when a file sharing the same filename base stem exists in the configured search directories."""
    condition_type: ClassVar[str] = "base_stem_match"

    require_match: bool = True
    # One or more accepted suffixes for the category (case-insensitive; trailing digits accepted).
    # Empty list = match any file with the base stem.
    suffix_filter: list = field(default_factory=list)
    # If set, search only this directory instead of config.directories_to_search_for_related_images.
    search_directory: str = ""
    # When > 0, the condition switches to overflow-detection mode: returns True when the
    # total number of files found for this base stem exceeds this limit (non-unique stem),
    # False otherwise. require_match is ignored in this mode. Wire on_match=REJECT and
    # on_no_match=CONTINUE on the node to reject non-unique stems.
    # When 0 and suffix_filter is non-empty, the runner auto-computes the limit as
    # len(suffix_filter) + 1, providing a healthy default without an explicit value.
    # When 0 and suffix_filter is empty, overflow detection is disabled entirely.
    max_stem_group_size: int = 0

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "require_match": self.require_match,
            "suffix_filter": self.suffix_filter,
            "search_directory": self.search_directory,
            "max_stem_group_size": self.max_stem_group_size,
        }

    def summary(self) -> str:
        mode = "found" if self.require_match else "not found"
        if self.max_stem_group_size > 0:
            scope = f"dir={self.search_directory!r}, " if self.search_directory else ""
            return f"BaseStemMatch({scope}max={self.max_stem_group_size})"
        if self.suffix_filter:
            joined = ", ".join(self.suffix_filter)
            return f"BaseStemMatch(suffix=[{joined}], require={mode})"
        if self.search_directory:
            return f"BaseStemMatch(dir={self.search_directory!r}, require={mode})"
        return f"BaseStemMatch(require={mode})"

    def display_summary(self) -> str:
        if self.max_stem_group_size > 0:
            scope = f"dir={self.search_directory}, " if self.search_directory else ""
            return _("BaseStemMatch") + f"({scope}max={self.max_stem_group_size})"
        mode = _("found") if self.require_match else _("not found")
        if self.suffix_filter:
            joined = ", ".join(self.suffix_filter)
            return _("BaseStemMatch") + f"(suffix=[{joined}], require={mode})"
        if self.search_directory:
            return _("BaseStemMatch") + f"(dir={self.search_directory}, require={mode})"
        return _("BaseStemMatch") + f"(require={mode})"


@dataclass
class UnknownSuffixCondition:
    """Returns True when the stem group contains a file with an unrecognised suffix that
    cannot be resolved by classifier inference.

    Intended to be wrapped in CompositeCondition(NOT) as a guard node: the NOT passes
    (CONTINUE) when the stem group is clean, and blocks (REJECT) when an ambiguous file
    is present and the classifier cannot determine its category.

    The seed image (file whose stem equals the base stem exactly, i.e. no suffix) is
    always excluded from the unknown-suffix check.

    Search scope (in priority order):
      1. search_directory if non-empty → scan exactly that directory.
      2. use_base_directory=True → scan the base_directory passed to run_pipeline
         (i.e. the working directory), falling back to the image's own directory.
      3. Otherwise → scan config.directories_to_search_for_related_images.

    For the category-fill guard node, use_base_directory=True is the correct choice:
    the guard only needs to detect ambiguous files in the working directory pool, not
    in the target directories (files there are categorised by their location, not suffix).
    """
    condition_type: ClassVar[str] = "unknown_suffix"

    # Valid suffixes for all expected categories combined (same format as BaseStemMatchCondition.suffix_filter).
    expected_suffixes: list = field(default_factory=list)
    # If set, search only this directory; otherwise uses config.directories_to_search_for_related_images.
    search_directory: str = ""
    # Classifier to run on unrecognised files to attempt category inference.
    # Empty = no inference; unrecognised files always trigger the block.
    classifier_name: str = ""
    # Minimum top-1 confidence for classifier inference to count as deterministic.
    inference_threshold: float = 0.85
    # When True and search_directory is empty, scan base_directory (the working dir
    # passed to run_pipeline) instead of config.directories_to_search_for_related_images.
    use_base_directory: bool = False

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "expected_suffixes": self.expected_suffixes,
            "search_directory": self.search_directory,
            "classifier_name": self.classifier_name,
            "inference_threshold": self.inference_threshold,
            "use_base_directory": self.use_base_directory,
        }

    def summary(self) -> str:
        suffixes = ", ".join(self.expected_suffixes) if self.expected_suffixes else "(none)"
        infer = f", infer={self.classifier_name!r}" if self.classifier_name else ""
        if self.search_directory:
            scope = f", dir={self.search_directory!r}"
        elif self.use_base_directory:
            scope = ", dir=base"
        else:
            scope = ""
        return f"UnknownSuffix(expected=[{suffixes}]{infer}{scope})"

    def display_summary(self) -> str:
        suffixes = ", ".join(self.expected_suffixes) if self.expected_suffixes else _("(none)")
        parts = [f"expected=[{suffixes}]"]
        if self.classifier_name:
            parts.append(f"infer={self.classifier_name}")
        if self.search_directory:
            parts.append(f"dir={self.search_directory}")
        elif self.use_base_directory:
            parts.append("dir=base")
        return _("UnknownSuffix") + "(" + ", ".join(parts) + ")"


@dataclass
class RelatedImageCondition:
    """Checks whether a generate action should run based on downstream image state."""
    condition_type: ClassVar[str] = "related_image"

    edit_suffix: str = ""
    search_directory: str = ""  # empty = see use_configured_search_directories
    count_threshold: int = 1
    # When True and search_directory is empty, search all directories from
    # config.directories_to_search_for_related_images instead of base_directory.
    use_configured_search_directories: bool = True

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "edit_suffix": self.edit_suffix,
            "search_directory": self.search_directory,
            "count_threshold": self.count_threshold,
            "use_configured_search_directories": self.use_configured_search_directories,
        }

    def summary(self) -> str:
        suffix_part = f"RelatedImage(suffix={self.edit_suffix!r}, threshold={self.count_threshold}"
        if self.search_directory:
            return suffix_part + f", dir={self.search_directory!r})"
        if not self.use_configured_search_directories:
            return suffix_part + ", dir=base)"
        return suffix_part + ")"

    def display_summary(self) -> str:
        parts = [f"suffix={self.edit_suffix}", f"threshold={self.count_threshold}"]
        if self.search_directory:
            parts.append(f"dir={self.search_directory}")
        elif not self.use_configured_search_directories:
            parts.append("dir=base")
        return _("RelatedImage") + "(" + ", ".join(parts) + ")"


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

    def display_summary(self) -> str:
        n = len(self.nodes)
        names = ", ".join(c.name for c in self.nodes[:3])
        extra = _(" +{0}").format(n - 3) if n > 3 else ""
        return _("Group({0}: {1}{2})").format(self.operator, names, extra)


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

    def display_summary(self) -> str:
        val = _("yes") if self.expected_result else _("no")
        return _("GroupChild") + f"({self.group_node_name}/{self.child_node_name}={val})"


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
    | UnknownSuffixCondition
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
            inherit_categories=d.get("inherit_categories", False),
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
        raw_sf = d.get("suffix_filter", [])
        # Backward compat: old configs serialised suffix_filter as a plain string.
        if isinstance(raw_sf, str):
            raw_sf = [raw_sf] if raw_sf else []
        return BaseStemMatchCondition(
            require_match=d.get("require_match", True),
            suffix_filter=raw_sf,
            search_directory=d.get("search_directory", ""),
            max_stem_group_size=d.get("max_stem_group_size", 0),
        )
    if ct == "unknown_suffix":
        return UnknownSuffixCondition(
            expected_suffixes=d.get("expected_suffixes", []),
            search_directory=d.get("search_directory", ""),
            classifier_name=d.get("classifier_name", ""),
            inference_threshold=d.get("inference_threshold", 0.85),
            use_base_directory=d.get("use_base_directory", False),
        )
    if ct == "related_image":
        return RelatedImageCondition(
            edit_suffix=d.get("edit_suffix", ""),
            search_directory=d.get("search_directory", ""),
            count_threshold=d.get("count_threshold", 1),
            use_configured_search_directories=d.get("use_configured_search_directories", True),
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
    CONTINUE             = "CONTINUE"              # advance to next node in order
    GOTO                 = "GOTO"                  # jump to named node (forward only)
    EXECUTE              = "EXECUTE"               # fire action and halt
    EXECUTE_AND_CONTINUE = "EXECUTE_AND_CONTINUE"  # fire action then advance to next node
    ACCEPT               = "ACCEPT"                # halt with no action (explicit pass)
    REJECT               = "REJECT"                # halt with pipeline's default_reject_action

    def display(self) -> str:
        return {
            OutcomeType.CONTINUE: _("Continue"),
            OutcomeType.GOTO: _("Go to node"),
            OutcomeType.EXECUTE: _("Execute action"),
            OutcomeType.EXECUTE_AND_CONTINUE: _("Execute and continue"),
            OutcomeType.ACCEPT: _("Accept"),
            OutcomeType.REJECT: _("Reject"),
        }[self]

    @classmethod
    def display_values(cls) -> list[str]:
        return [item.display() for item in cls]

    @staticmethod
    def get(name) -> "OutcomeType":
        if isinstance(name, OutcomeType):
            return name
        for member in OutcomeType:
            if (
                name == member.name
                or name == member.value
                or name == member.display()
            ):
                return member
        raise ValueError(f"Not a valid outcome type: {name!r}")

    @property
    def requires_action(self) -> bool:
        """True for outcome types that require an action_type to be specified."""
        return self in (OutcomeType.EXECUTE, OutcomeType.EXECUTE_AND_CONTINUE)

    @property
    def requires_target_node(self) -> bool:
        """True for outcome types that require a target_node to be specified."""
        return self is OutcomeType.GOTO


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
        if self.outcome_type.requires_action:
            base = f"{self.outcome_type.value}: {self.action_type.value if self.action_type else '?'}"
            if self.action_modifier:
                base += f" → {self.action_modifier}"
            return base
        if self.outcome_type.requires_target_node:
            return f"GOTO: {self.target_node}"
        return self.outcome_type.value

    def display_summary(self) -> str:
        """User-facing outcome label for UI previews and list cells."""
        if self.outcome_type.requires_action:
            action = self.action_type.get_translation() if self.action_type else _("(unknown)")
            base = _("{0}: {1}").format(self.outcome_type.display(), action)
            if self.action_modifier:
                base += _(" → {0}").format(self.action_modifier)
            return base
        if self.outcome_type.requires_target_node:
            return _("{0}: {1}").format(self.outcome_type.display(), self.target_node)
        return self.outcome_type.display()

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
    enabled: bool = True              # False → runner skips this node (no-op CONTINUE)

    def __post_init__(self):
        if self.condition is None:
            self.condition = EmbeddingCondition()
        if self.on_match is None:
            self.on_match = NodeOutcome.continue_()
        if self.on_no_match is None:
            self.on_no_match = NodeOutcome.accept()

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "condition": self.condition.to_dict(),
            "on_match": self.on_match.to_dict(),
            "on_no_match": self.on_no_match.to_dict(),
        }
        if not self.enabled:
            d["enabled"] = False
        return d

    @staticmethod
    def from_dict(d: dict) -> "PipelineNode":
        return PipelineNode(
            name=d.get("name", ""),
            condition=_condition_from_dict(d.get("condition", {"condition_type": "embedding"})),
            on_match=NodeOutcome.from_dict(d.get("on_match", {})),
            on_no_match=NodeOutcome.from_dict(d.get("on_no_match", {})),
            enabled=d.get("enabled", True),
        )

    def condition_summary(self) -> str:
        if not self.condition:
            return _("(no condition)")
        return self.condition.display_summary()


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
            if condition.operator in ("AND", "OR") and n < 2:
                errors.append(
                    _("Node {0}: {1} requires ≥ 2 sub-conditions, got {2}.").format(
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
            "prototype": _("Prototype"),
            "prompt": _("Prompt"),
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
        }
        if self.category_map:
            d["category_map"] = dict(self.category_map)
        if self.seed_category:
            d["seed_category"] = self.seed_category
        return d

    @staticmethod
    def from_dict(d: dict) -> "ClassifierPipeline":
        def _opt_action(val):
            if not val:
                return None
            return ClassifierActionType[val] if isinstance(val, str) else val

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

        Illustrates the §4 architecture from docs/generation-pipeline-category-fill.md.

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

        Behaviour table (per §4.5)
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

        Seed guard note (§5.8)
        ───────────────────────
        Each category node could optionally include a CompositeCondition(NOT,
        ClassifierRankCondition(...)) as a third AND sub-condition to skip
        generation when the seed image is already classified as that category.
        This is omitted here pending classifier validation.

        processed_stems note (§5.13)
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
                        # correctly signals that the category is covered (§5.1).
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
        # Node 2 — Stem uniqueness check
        # Rejects when the base stem matches more than max_stem_group_size files
        # across the configured target directories, indicating the stem is not
        # unique enough for reliable per-image generation (e.g. a bare label
        # like "photo" that collides with thousands of existing files).
        # search_directory: empty → uses config.directories_to_search_for_related_images
        # (all target dirs combined).
        # on_match=REJECT: overflow detected → stem not unique.
        # on_no_match=CONTINUE: within limit → proceed to category nodes.
        # ------------------------------------------------------------------
        node_uniqueness = PipelineNode(
            name="Stem uniqueness check",
            condition=BaseStemMatchCondition(),
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
                "a file with this base stem. See §4 of "
                "docs/generation-pipeline-category-fill.md."
            ),
            nodes=[node_guard, node_uniqueness, node_apple, node_banana, node_cherry],
            is_active=False,
            category_map=CATEGORY_MAP,
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

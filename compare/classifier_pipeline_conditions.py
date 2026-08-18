"""
Condition data model for ClassifierPipeline.

Each condition is a plain dataclass describing one check a PipelineNode can
make.  Conditions carry no evaluation logic -- that lives in
compare.classifier_pipeline_runner -- and no validation logic, which lives in
ClassifierPipeline._validate_condition.

Deserialization (_condition_from_dict) lives in
compare.classifier_pipeline_nodes rather than here, because GroupCondition
holds child PipelineNodes and PipelineNode holds a condition: putting the
deserializer next to PipelineNode is what keeps the three pipeline modules
acyclic.

This module must not import from any other compare/ module.  Keeping it
dependency-free is what allows the node and pipeline modules to import it
without a cycle, so a new condition type needing a runtime dependency should
resolve that dependency at evaluation time in the runner instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional

from utils.constants import CompareMediaType
from utils.translations import _


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
    # When True, flips the rank/category/confidence result. Lets this condition
    # express "none of these categories rank highly" directly -- e.g. as a third
    # AND sub-condition on a category-fill generate node, alongside conditions
    # like RelatedImageCondition/BaseStemMatchCondition where True already means
    # "clear to generate" -- without wrapping it in a CompositeCondition(NOT, ...)
    # (which the pipeline editor's sub-condition picker doesn't support nesting,
    # and which would cost a second, separate classifier evaluation; the
    # classifier call itself is not cached and is not necessarily cheap).
    # Classifier-not-found and no-categories-configured are error states and are
    # NOT negated -- they always evaluate to no-match, regardless of this flag.
    negate: bool = False

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
            "negate": self.negate,
        }

    def summary(self) -> str:
        if self.inherit_categories and not self.categories:
            cats = "(pipeline categories)"
        else:
            cats = ", ".join(self.categories) if self.categories else "(none)"
        rank = f"rank {self.min_rank}" if self.min_rank == self.max_rank else f"rank {self.min_rank}-{self.max_rank}"
        prefix = "NOT " if self.negate else ""
        return f"{prefix}ClassifierRank({self.classifier_name}, [{cats}], {rank})"

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
        prefix = _("NOT ") if self.negate else ""
        return prefix + _("ClsRank") + f"({self.classifier_name}, [{cats}], {rank})"


@dataclass
class AudioClassifierRankCondition:
    """Audio-domain sibling of :class:`ClassifierRankCondition`.

    Same shape and semantics (rank window, categories, min confidence, negate,
    inherit_categories), evaluated against ``image.audio_classifier_manager``'s
    registry instead of the image classifier registry. Kept as a separate
    condition type rather than a ``domain`` field on ``ClassifierRankCondition``,
    mirroring how distinct concerns already get distinct condition types in
    this module (e.g. ``PrototypeCondition`` vs ``LookaheadCondition``) rather
    than retrofitting a field onto a condition type that already just shipped
    a ``negate`` flag.
    Typically paired with a ``MediaTypeCondition([CompareMediaType.AUDIO])``
    gate earlier in the pipeline, since nothing here restricts evaluation to
    audio files on its own.
    """
    condition_type: ClassVar[str] = "audio_classifier_rank"

    classifier_name: str = ""
    categories: Optional[list] = None
    min_rank: int = 1
    max_rank: int = 1
    min_confidence: float = 0.0
    inherit_categories: bool = False
    negate: bool = False

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
            "negate": self.negate,
        }

    def summary(self) -> str:
        if self.inherit_categories and not self.categories:
            cats = "(pipeline categories)"
        else:
            cats = ", ".join(self.categories) if self.categories else "(none)"
        rank = f"rank {self.min_rank}" if self.min_rank == self.max_rank else f"rank {self.min_rank}-{self.max_rank}"
        prefix = "NOT " if self.negate else ""
        return f"{prefix}AudioClassifierRank({self.classifier_name}, [{cats}], {rank})"

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
        prefix = _("NOT ") if self.negate else ""
        return prefix + _("AudioClsRank") + f"({self.classifier_name}, [{cats}], {rank})"


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
class AlwaysCondition:
    """No check required — always matches, so the node's on_match outcome
    simply executes. Useful for unconditional action nodes."""
    condition_type: ClassVar[str] = "always"

    def to_dict(self) -> dict:
        return {"condition_type": self.condition_type}

    def summary(self) -> str:
        return "Always()"

    def display_summary(self) -> str:
        return _("Always (no check)")


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
    # When True and search_directory is empty, scan base_directory (the directory currently
    # being processed by the pipeline runner) instead of config.directories_to_search_for_related_images.
    use_working_directory: bool = False
    # When > 0, the condition switches to overflow-detection mode: returns True when the
    # total number of files found for this base stem exceeds this limit (non-unique stem),
    # False otherwise. require_match is ignored in this mode. Wire on_match=REJECT and
    # on_no_match=CONTINUE on the node to reject non-unique stems.
    # When < 0 (sentinel), auto-computes the limit as len(pipeline_categories) + 1 at
    # runtime regardless of search_directory. Useful when the search covers a directory
    # that contains all category subdirs and the exact count is not known at edit time.
    # When 0 and search_directory is unset, also auto-computes from pipeline_categories.
    # When 0 and search_directory is set, overflow detection is disabled entirely.
    max_stem_group_size: int = 0

    def to_dict(self) -> dict:
        return {
            "condition_type": self.condition_type,
            "require_match": self.require_match,
            "suffix_filter": self.suffix_filter,
            "search_directory": self.search_directory,
            "use_working_directory": self.use_working_directory,
            "max_stem_group_size": self.max_stem_group_size,
        }

    def summary(self) -> str:
        mode = "found" if self.require_match else "not found"
        if self.max_stem_group_size > 0:
            scope = f"dir={self.search_directory!r}, " if self.search_directory else ("working_dir, " if self.use_working_directory else "")
            return f"BaseStemMatch({scope}max={self.max_stem_group_size})"
        if self.suffix_filter:
            joined = ", ".join(self.suffix_filter)
            return f"BaseStemMatch(suffix=[{joined}], require={mode})"
        if self.search_directory:
            return f"BaseStemMatch(dir={self.search_directory!r}, require={mode})"
        if self.use_working_directory:
            return f"BaseStemMatch(working_dir, require={mode})"
        return f"BaseStemMatch(require={mode})"

    def display_summary(self) -> str:
        if self.max_stem_group_size > 0:
            scope = f"dir={self.search_directory}, " if self.search_directory else (_("working_dir, ") if self.use_working_directory else "")
            return _("BaseStemMatch") + f"({scope}max={self.max_stem_group_size})"
        mode = _("found") if self.require_match else _("not found")
        if self.suffix_filter:
            joined = ", ".join(self.suffix_filter)
            return _("BaseStemMatch") + f"(suffix=[{joined}], require={mode})"
        if self.search_directory:
            return _("BaseStemMatch") + f"(dir={self.search_directory}, require={mode})"
        if self.use_working_directory:
            return _("BaseStemMatch") + f"({_('working_dir')}, require={mode})"
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
    | AudioClassifierRankCondition
    | PrototypeCondition
    | PromptCondition
    | FilenameContainsCondition
    | AlwaysCondition
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

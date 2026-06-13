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

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

from utils.app_info_cache import app_info_cache
from utils.constants import ClassifierActionType
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


# Union type alias (informational only — Python does not enforce it at runtime)
NodeCondition = (
    EmbeddingCondition
    | ClassifierRankCondition
    | PrototypeCondition
    | PromptCondition
    | FilenameContainsCondition
    | LookaheadCondition
    | NodeResultCondition
    | CompositeCondition
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

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []

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
        }
        lines = []
        for node in self.nodes:
            cond_type = getattr(node.condition, "condition_type", "")
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
        # Use None sentinel to distinguish "key never written" from "explicitly empty list".
        raw = app_info_cache.get_meta(_CACHE_KEY, default_val=None)
        if raw is None:
            ClassifierPipelines.pipelines = [ClassifierPipelines._build_demo_pipeline()]
            ClassifierPipelines._rebuild_type_cache()
            return
        result: list[ClassifierPipeline] = []
        for d in raw:
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
    def _build_demo_pipeline() -> "ClassifierPipeline":
        node_portrait = PipelineNode(
            name="Is a person?",
            condition=EmbeddingCondition(
                positives=["person", "human face", "portrait"],
                negatives=["landscape", "animal", "object"],
                threshold=0.25,
            ),
            on_match=NodeOutcome(OutcomeType.GOTO, target_node="Check if explicit"),
            on_no_match=NodeOutcome(OutcomeType.ACCEPT),
        )
        node_explicit = PipelineNode(
            name="Check if explicit",
            condition=EmbeddingCondition(
                positives=["explicit content", "nudity", "nsfw"],
                negatives=["clothed", "safe for work"],
                threshold=0.28,
            ),
            on_match=NodeOutcome(OutcomeType.EXECUTE, action_type=ClassifierActionType.HIDE),
            on_no_match=NodeOutcome(OutcomeType.ACCEPT),
        )
        return ClassifierPipeline(
            name="Example: Explicit Content Filter",
            description="Demo pipeline (inactive). Two-node example: checks if a person appears, then hides if explicit.",
            nodes=[node_portrait, node_explicit],
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

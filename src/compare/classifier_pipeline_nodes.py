"""
Node model for ClassifierPipeline: outcomes, nodes, and node-graph deserialization.

A PipelineNode pairs one condition (see compare.classifier_pipeline_conditions)
with two NodeOutcomes -- what to do when the condition matches, and when it
doesn't.  ClassifierPipeline is an ordered list of these; the runner walks them.

_condition_from_dict lives here rather than beside the condition dataclasses
because the two halves of the model are mutually recursive: GroupCondition
holds child PipelineNodes, and PipelineNode holds a condition.  Deserializing
either one needs both, so keeping the single function that bridges them in this
module is what lets the conditions module stay dependency-free and the import
graph stay acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from compare.classifier_pipeline_conditions import (
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
    NodeResultCondition,
    PromptCondition,
    PrototypeCondition,
    RelatedImageCondition,
    UnknownSuffixCondition,
    VarianceFromOriginalCondition,
)
from utils.constants import ClassifierActionType
from utils.translations import _


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
            negate=d.get("negate", False),
        )
    if ct == "audio_classifier_rank":
        return AudioClassifierRankCondition(
            classifier_name=d.get("classifier_name", ""),
            categories=d.get("categories", []),
            min_rank=d.get("min_rank", 1),
            max_rank=d.get("max_rank", 1),
            min_confidence=d.get("min_confidence", 0.0),
            inherit_categories=d.get("inherit_categories", False),
            negate=d.get("negate", False),
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
    if ct == "always":
        return AlwaysCondition()
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
            use_working_directory=d.get("use_working_directory", False),
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
    if ct == "variance_from_original":
        return VarianceFromOriginalCondition(
            min_similarity=d.get("min_similarity", 0.55),
            max_similarity=d.get("max_similarity", 0.95),
            compare_mode=d.get("compare_mode", "CLIP_EMBEDDING"),
            invert=d.get("invert", False),
            match_on_unresolved=d.get("match_on_unresolved", False),
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

    def is_category_generate_node(self, suffix: str) -> bool:
        """True when this node matches the form generated by Fill from Map for *suffix*.

        Checks: GENERATE action with matching modifier, CompositeCondition(AND) containing
        both a RelatedImageCondition and a BaseStemMatchCondition sub-condition.
        """
        if self.on_match.action_type != ClassifierActionType.GENERATE:
            return False
        if self.on_match.action_modifier != suffix:
            return False
        if not isinstance(self.condition, CompositeCondition) or self.condition.operator != "AND":
            return False
        sub_types = {type(sc) for sc in self.condition.sub_conditions}
        return RelatedImageCondition in sub_types and BaseStemMatchCondition in sub_types

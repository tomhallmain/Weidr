"""
Data model for file-handling interceptor rules.

An interceptor rule matches a file that the manual marks flow is about to
move or copy to a target directory, and either blocks that transfer or
substitutes a transformed file as the payload.

This module holds shapes and serialization only -- no matching logic, no file
I/O, no Qt. Keeping it free of those dependencies is what lets the rule list,
the evaluator, and the editor window all import it without a cycle; the list,
evaluation and transform execution live in
``files.file_interceptor_rules_manager``, which is what other callers import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from utils.constants import CompareMediaType
from utils.translations import _


class InterceptorAppliesTo(Enum):
    """Which transfer kinds a rule participates in."""
    MOVE_AND_COPY = "MOVE_AND_COPY"
    MOVE_ONLY = "MOVE_ONLY"
    COPY_ONLY = "COPY_ONLY"

    def allows(self, is_moving: bool) -> bool:
        if self == InterceptorAppliesTo.MOVE_ONLY:
            return is_moving
        if self == InterceptorAppliesTo.COPY_ONLY:
            return not is_moving
        return True

    def get_translation(self) -> str:
        if self == InterceptorAppliesTo.MOVE_ONLY:
            return _("Move only")
        if self == InterceptorAppliesTo.COPY_ONLY:
            return _("Copy only")
        return _("Move and copy")

    @staticmethod
    def from_value(value) -> "InterceptorAppliesTo":
        if isinstance(value, InterceptorAppliesTo):
            return value
        try:
            return InterceptorAppliesTo(str(value or "").strip().upper())
        except ValueError:
            return InterceptorAppliesTo.MOVE_AND_COPY


class InterceptorBehavior(Enum):
    """What a matching rule does to the transfer."""
    #: Do not move/copy this file; warn and leave it marked.
    BLOCK = "BLOCK"
    #: Run a transform op first and transfer the op's output instead.
    TRANSFORM = "TRANSFORM"

    def get_translation(self) -> str:
        if self == InterceptorBehavior.TRANSFORM:
            return _("Transform")
        return _("Block")

    @staticmethod
    def from_value(value) -> "InterceptorBehavior":
        if isinstance(value, InterceptorBehavior):
            return value
        try:
            return InterceptorBehavior(str(value or "").strip().upper())
        except ValueError:
            return InterceptorBehavior.BLOCK


class InterceptorTransformOp(Enum):
    """Image operation a TRANSFORM rule runs before the transfer."""
    CONVERT_TO_JPG = "CONVERT_TO_JPG"
    ENHANCE = "ENHANCE"

    def get_translation(self) -> str:
        if self == InterceptorTransformOp.ENHANCE:
            return _("Improve image")
        return _("Convert to JPG")

    @staticmethod
    def from_value(value) -> Optional["InterceptorTransformOp"]:
        if isinstance(value, InterceptorTransformOp):
            return value
        if value is None or str(value).strip() == "":
            return None
        try:
            return InterceptorTransformOp(str(value).strip().upper())
        except ValueError:
            return None


@dataclass
class FileInterceptorRule:
    """One user-configured interception of the manual move/copy flow.

    Every *set* match condition must hold for the rule to fire (AND); an empty
    or unset condition matches anything. A rule with no conditions set at all
    therefore matches every file, which is intentional -- that is how the
    "regardless of target directory" case is expressed.
    """

    name: str = field(default_factory=lambda: _("New Interceptor Rule"))
    is_active: bool = True
    applies_to: InterceptorAppliesTo = InterceptorAppliesTo.MOVE_AND_COPY

    # --- match conditions ---
    match_target_dirs: list = field(default_factory=list)
    # Whether match_target_dirs also matches subdirectories of the listed paths.
    # Off by default: a rule written for one directory should not silently
    # capture category subdirectories created underneath it later.
    include_subdirectories: bool = False
    match_filename_patterns: list = field(default_factory=list)
    filename_case_sensitive: bool = False
    #: None means any media type.
    match_media_types: Optional[list] = None

    # --- behavior ---
    behavior: InterceptorBehavior = InterceptorBehavior.BLOCK
    #: Shown to the user when a BLOCK rule fires; falls back to a generic message.
    block_message: str = ""
    transform_op: Optional[InterceptorTransformOp] = None
    #: Delete the untransformed original after a successful transformed move.
    #: Never honored on copy -- deleting a copy's source would contradict copy.
    delete_original_after_transform: bool = True

    def __post_init__(self):
        self.applies_to = InterceptorAppliesTo.from_value(self.applies_to)
        self.behavior = InterceptorBehavior.from_value(self.behavior)
        self.transform_op = InterceptorTransformOp.from_value(self.transform_op)
        self.match_target_dirs = list(self.match_target_dirs) if self.match_target_dirs else []
        self.match_filename_patterns = (
            list(self.match_filename_patterns) if self.match_filename_patterns else []
        )
        self.is_active = bool(self.is_active)
        self.include_subdirectories = bool(self.include_subdirectories)
        self.filename_case_sensitive = bool(self.filename_case_sensitive)
        self.delete_original_after_transform = bool(self.delete_original_after_transform)
        if self.match_media_types is not None:
            coerced = []
            for mt in self.match_media_types:
                try:
                    coerced.append(mt if isinstance(mt, CompareMediaType) else CompareMediaType(mt))
                except ValueError:
                    continue
            self.match_media_types = coerced if coerced else None

    def is_transform(self) -> bool:
        return self.behavior == InterceptorBehavior.TRANSFORM and self.transform_op is not None

    def summary(self) -> str:
        """One-line description of the rule for the editor list."""
        conditions = []
        if self.match_target_dirs:
            dirs = ", ".join(self.match_target_dirs)
            conditions.append(
                _("into {0} (and subdirectories)").format(dirs)
                if self.include_subdirectories
                else _("into {0}").format(dirs)
            )
        if self.match_filename_patterns:
            conditions.append(
                _("name contains {0}").format(", ".join(self.match_filename_patterns))
            )
        if self.match_media_types:
            conditions.append(
                _("type is {0}").format(
                    ", ".join(mt.get_translation() for mt in self.match_media_types)
                )
            )
        condition_text = " + ".join(conditions) if conditions else _("any file")
        if self.is_transform():
            effect = self.transform_op.get_translation()
            if self.delete_original_after_transform:
                effect += _(", removing the original")
        else:
            effect = _("Block")
        return f"{condition_text} → {effect} ({self.applies_to.get_translation()})"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_active": self.is_active,
            "applies_to": self.applies_to.value,
            "match_target_dirs": list(self.match_target_dirs),
            "include_subdirectories": self.include_subdirectories,
            "match_filename_patterns": list(self.match_filename_patterns),
            "filename_case_sensitive": self.filename_case_sensitive,
            "match_media_types": (
                [mt.value for mt in self.match_media_types]
                if self.match_media_types is not None
                else None
            ),
            "behavior": self.behavior.value,
            "block_message": self.block_message,
            "transform_op": self.transform_op.value if self.transform_op else None,
            "delete_original_after_transform": self.delete_original_after_transform,
        }

    @staticmethod
    def from_dict(dct: dict) -> "FileInterceptorRule":
        return FileInterceptorRule(
            name=dct.get("name", ""),
            is_active=dct.get("is_active", True),
            applies_to=dct.get("applies_to", InterceptorAppliesTo.MOVE_AND_COPY),
            match_target_dirs=dct.get("match_target_dirs", []),
            include_subdirectories=dct.get("include_subdirectories", False),
            match_filename_patterns=dct.get("match_filename_patterns", []),
            filename_case_sensitive=dct.get("filename_case_sensitive", False),
            match_media_types=dct.get("match_media_types", None),
            behavior=dct.get("behavior", InterceptorBehavior.BLOCK),
            block_message=dct.get("block_message", ""),
            transform_op=dct.get("transform_op", None),
            delete_original_after_transform=dct.get("delete_original_after_transform", True),
        )

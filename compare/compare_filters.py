"""
Composable pre-filter tree for compare operations.

Filters are applied in BaseCompare.get_files() *before* any expensive
embedding/color computation, so only cheap I/O is done here (PIL header
reads for size, JSON/EXIF reads for model metadata).

Tree shape
----------
CompareFilter (abstract)
├── SizeFilter          — leaf: dimension constraints
├── ModelFilter         — leaf: model/lora name constraints
└── CompareFilterGroup  — node: AND / OR / NOT of child filters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from utils.logging_setup import get_logger

logger = get_logger("compare_filters")


# ---------------------------------------------------------------------------
# Operator enum
# ---------------------------------------------------------------------------

class FilterOperator(Enum):
    AND = "and"   # file must pass every child filter
    OR  = "or"    # file must pass at least one child filter
    NOT = "not"   # exclude files that pass any child filter


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CompareFilter(ABC):
    @abstractmethod
    def is_active(self) -> bool: ...


# ---------------------------------------------------------------------------
# Leaf: size constraints
# ---------------------------------------------------------------------------

@dataclass
class SizeFilter(CompareFilter):
    """
    Dimension-based pre-filter.  All active constraints must be satisfied.

    min_size / max_size apply per-axis (width AND height must be in range).
    exact_size requires both axes within size_tolerance pixels.
    """
    min_size:      Optional[Tuple[int, int]] = None   # (min_w, min_h)
    max_size:      Optional[Tuple[int, int]] = None   # (max_w, max_h)
    exact_size:    Optional[Tuple[int, int]] = None   # (exact_w, exact_h)
    size_tolerance: int = 0                           # pixel tolerance for exact_size

    def is_active(self) -> bool:
        return (self.min_size is not None
                or self.max_size is not None
                or self.exact_size is not None)


# ---------------------------------------------------------------------------
# Leaf: model/lora constraints
# ---------------------------------------------------------------------------

@dataclass
class ModelFilter(CompareFilter):
    """
    Model/LoRA-based pre-filter.

    models     — names to match against (substring match, case-insensitive)
    mode       — 'include': only files with matching model(s) pass;
                 'exclude': files with matching model(s) are blocked
    match_any  — True  → any listed name matching is sufficient;
                 False → all listed names must be present (AND logic)
    include_loras — whether LoRA names count during matching
    """
    models:        Optional[List[str]] = None
    mode:          str  = 'include'   # 'include' | 'exclude'
    match_any:     bool = False
    include_loras: bool = True

    def is_active(self) -> bool:
        return bool(self.models)


# ---------------------------------------------------------------------------
# Composite node
# ---------------------------------------------------------------------------

@dataclass
class CompareFilterGroup(CompareFilter):
    """Recursive node: combine N child filters with AND / OR / NOT logic."""
    operator: FilterOperator         = FilterOperator.AND
    filters:  List[CompareFilter]    = field(default_factory=list)

    def is_active(self) -> bool:
        return any(f.is_active() for f in self.filters)

    def add(self, f: CompareFilter) -> CompareFilterGroup:
        self.filters.append(f)
        return self


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_filter(
    files: list,
    f: CompareFilter,
    metadata_reader=None,
) -> list:
    """
    Return the subset of *files* that passes filter *f*.

    metadata_reader — optional override for model metadata reading (for tests);
                      defaults to the real image_data_extractor singleton.
    """
    if not f.is_active():
        return files

    if isinstance(f, CompareFilterGroup):
        return _apply_group(files, f, metadata_reader)
    if isinstance(f, SizeFilter):
        return _apply_size(files, f)
    if isinstance(f, ModelFilter):
        return _apply_model(files, f, metadata_reader)

    logger.warning(f"Unknown filter type {type(f).__name__} — skipping")
    return files


# ---------------------------------------------------------------------------
# Group dispatch
# ---------------------------------------------------------------------------

def _apply_group(
    files: list,
    group: CompareFilterGroup,
    metadata_reader,
) -> list:
    op = group.operator
    active = [c for c in group.filters if c.is_active()]

    if not active:
        return files

    logger.debug(
        f"CompareFilterGroup ({op.value.upper()}) with {len(active)} active "
        f"child filter(s) applied to {len(files)} file(s)"
    )

    if op == FilterOperator.AND:
        result = files
        for child in active:
            result = apply_filter(result, child, metadata_reader)
        return result

    if op == FilterOperator.OR:
        passed: set = set()
        for child in active:
            passed.update(apply_filter(files, child, metadata_reader))
        return [fp for fp in files if fp in passed]

    if op == FilterOperator.NOT:
        excluded: set = set()
        for child in active:
            excluded.update(apply_filter(files, child, metadata_reader))
        result = [fp for fp in files if fp not in excluded]
        logger.debug(
            f"NOT group excluded {len(excluded)} file(s), "
            f"{len(result)} remain"
        )
        return result

    logger.warning(f"Unhandled FilterOperator {op} — returning files unchanged")
    return files


# ---------------------------------------------------------------------------
# Size leaf
# ---------------------------------------------------------------------------

def _apply_size(files: list, f: SizeFilter) -> list:
    from compare.compare_size import extract_size_from_image  # local to avoid circular

    total = len(files)
    passed = []
    unreadable = 0

    for fp in files:
        dims = extract_size_from_image(fp)
        if dims is None:
            unreadable += 1
            logger.debug(f"SizeFilter: could not read dimensions of '{fp}' — excluded")
            continue

        w, h = dims

        if f.exact_size is not None:
            ew, eh = f.exact_size
            if abs(w - ew) > f.size_tolerance or abs(h - eh) > f.size_tolerance:
                logger.debug(
                    f"SizeFilter: excluded '{fp}' "
                    f"({w}×{h} not within {f.size_tolerance}px of {ew}×{eh})"
                )
                continue

        if f.min_size is not None:
            mw, mh = f.min_size
            if w < mw or h < mh:
                logger.debug(
                    f"SizeFilter: excluded '{fp}' ({w}×{h} < min {mw}×{mh})"
                )
                continue

        if f.max_size is not None:
            mw, mh = f.max_size
            if w > mw or h > mh:
                logger.debug(
                    f"SizeFilter: excluded '{fp}' ({w}×{h} > max {mw}×{mh})"
                )
                continue

        passed.append(fp)

    excluded = total - len(passed)
    logger.info(
        f"SizeFilter: {excluded}/{total} file(s) excluded "
        f"({len(passed)} remain"
        + (f", {unreadable} unreadable)" if unreadable else ")")
    )
    return passed


# ---------------------------------------------------------------------------
# Model leaf
# ---------------------------------------------------------------------------

def _apply_model(files: list, f: ModelFilter, metadata_reader) -> list:
    if metadata_reader is None:
        from image.image_data_extractor import image_data_extractor as _extractor
        metadata_reader = _extractor

    names = [m.lower() for m in (f.models or [])]
    if not names:
        return files

    total = len(files)
    passed = []

    for fp in files:
        try:
            models_raw, loras_raw = metadata_reader.get_models(fp)
        except Exception as e:
            logger.debug(f"ModelFilter: could not read metadata from '{fp}': {e}")
            models_raw, loras_raw = [], []

        # Collect candidate names for this file
        candidates = [m.lower() for m in models_raw]
        if f.include_loras:
            candidates += [lr.lower() for lr in loras_raw]

        # For each filter name, check whether any candidate contains it
        def _matches_name(filter_name: str) -> bool:
            return any(filter_name in candidate for candidate in candidates)

        if f.match_any:
            file_matches = any(_matches_name(n) for n in names)
        else:
            file_matches = all(_matches_name(n) for n in names)

        include = (f.mode == 'include') == file_matches  # XOR-free logic

        if include:
            passed.append(fp)
        else:
            logger.debug(
                f"ModelFilter ({f.mode}): excluded '{fp}' "
                f"(candidates: {candidates or ['none']})"
            )

    excluded = total - len(passed)
    logger.info(
        f"ModelFilter ({f.mode}, match_{'any' if f.match_any else 'all'}): "
        f"{excluded}/{total} file(s) excluded ({len(passed)} remain)"
    )
    return passed

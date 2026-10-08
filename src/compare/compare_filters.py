"""
Composable pre-filter tree for compare operations.

Filters are applied in BaseCompare.get_files() *before* any expensive
embedding/color computation, so only cheap I/O is done here (PIL header
reads for size, JSON/EXIF reads for model metadata).

ClassifierFilter is the exception: it runs a classifier model on every file
that reaches it, so AND groups evaluate it after the cheap leaves.

Tree shape
----------
CompareFilter (abstract)
├── SizeFilter          — leaf: dimension constraints
├── ModelFilter         — leaf: model/lora name constraints
├── ClassifierFilter    — leaf: classifier category constraints
└── CompareFilterGroup  — node: AND / OR / NOT of child filters
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

from compare.classifier_categories import model_strategy_positive_categories
from utils.logging_setup import get_logger
from utils.translations import _

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
# Leaf: classifier category constraints
# ---------------------------------------------------------------------------

CLASSIFIER_DOMAIN_IMAGE = "image"
CLASSIFIER_DOMAIN_AUDIO = "audio"

# Same values as ClassifierClassificationMode in compare/classifier_action.py.
SELECTION_SELECTED_CATEGORIES = "selected_categories"
SELECTION_MODEL_STRATEGY = "model_strategy"


class ClassifierFilterError(Exception):
    """A classifier filter cannot run (model missing, not loadable, or
    misconfigured). The message is user-facing."""


@dataclass
class ClassifierFilter(CompareFilter):
    """
    Classifier-based pre-filter.

    A file matches when the model's predicted category is in the selected
    set: *categories* for 'selected_categories', the union of the model
    config's positive_groups for 'model_strategy'. With min_confidence > 0 the
    predicted category's score must also reach it. Image classifiers judge
    video/GIF/PDF/ePub the way classifier actions do (dynamic_media_sampling):
    sample_ratio of the frames/pages are sampled, and the file matches once
    positive_ratio of the sampled ones match.

    mode — 'include': only matching files pass; 'exclude': matching files are
           blocked. Files the classifier cannot handle (wrong media domain,
           read/prediction error) count as not matching, so 'include' drops
           them and 'exclude' keeps them.
    """
    classifier_name: str = ""
    domain:          str = CLASSIFIER_DOMAIN_IMAGE
    selection_mode:  str = SELECTION_SELECTED_CATEGORIES
    categories:      List[str] = field(default_factory=list)
    mode:            str = 'include'   # 'include' | 'exclude'
    min_confidence:  float = 0.0
    # Same defaults as ClassifierAction.dynamic_content_sample_ratio/_positive_ratio.
    sample_ratio:    float = 0.1
    positive_ratio:  float = 0.1

    def is_active(self) -> bool:
        if not (self.classifier_name or "").strip():
            return False
        if self.selection_mode == SELECTION_MODEL_STRATEGY:
            return True
        return bool(self.categories)


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
# Serialization helpers
# ---------------------------------------------------------------------------

def filter_to_dict(f: Optional[CompareFilter]) -> Optional[dict]:
    """Serialize a filter tree to a JSON-safe dict, or return None."""
    if f is None:
        return None
    if isinstance(f, SizeFilter):
        return {
            "type": "size",
            "min_size": list(f.min_size) if f.min_size else None,
            "max_size": list(f.max_size) if f.max_size else None,
            "exact_size": list(f.exact_size) if f.exact_size else None,
            "size_tolerance": f.size_tolerance,
        }
    if isinstance(f, ModelFilter):
        return {
            "type": "model",
            "models": f.models,
            "mode": f.mode,
            "match_any": f.match_any,
            "include_loras": f.include_loras,
        }
    if isinstance(f, ClassifierFilter):
        return {
            "type": "classifier",
            "classifier_name": f.classifier_name,
            "domain": f.domain,
            "selection_mode": f.selection_mode,
            "categories": list(f.categories) if f.categories else [],
            "mode": f.mode,
            "min_confidence": f.min_confidence,
            "sample_ratio": f.sample_ratio,
            "positive_ratio": f.positive_ratio,
        }
    if isinstance(f, CompareFilterGroup):
        return {
            "type": "group",
            "operator": f.operator.value,
            "filters": [filter_to_dict(c) for c in f.filters],
        }
    return None


def filter_signature(f: Optional[CompareFilter]) -> Optional[str]:
    """Short stable hash of an active filter tree, or None for no/inactive
    filter. Distinguishes runs whose candidate file lists differ only by filter."""
    if f is None or not f.is_active():
        return None
    payload = json.dumps(filter_to_dict(f), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def filter_from_dict(d: Optional[dict]) -> Optional[CompareFilter]:
    """Deserialize a filter tree from a dict produced by filter_to_dict."""
    if not d:
        return None
    t = d.get("type")
    if t == "size":
        return SizeFilter(
            min_size=tuple(d["min_size"]) if d.get("min_size") else None,
            max_size=tuple(d["max_size"]) if d.get("max_size") else None,
            exact_size=tuple(d["exact_size"]) if d.get("exact_size") else None,
            size_tolerance=d.get("size_tolerance", 0),
        )
    if t == "model":
        return ModelFilter(
            models=d.get("models"),
            mode=d.get("mode", "include"),
            match_any=d.get("match_any", False),
            include_loras=d.get("include_loras", True),
        )
    if t == "classifier":
        domain = d.get("domain", CLASSIFIER_DOMAIN_IMAGE)
        if domain not in (CLASSIFIER_DOMAIN_IMAGE, CLASSIFIER_DOMAIN_AUDIO):
            domain = CLASSIFIER_DOMAIN_IMAGE
        selection_mode = d.get("selection_mode", SELECTION_SELECTED_CATEGORIES)
        if selection_mode not in (SELECTION_SELECTED_CATEGORIES, SELECTION_MODEL_STRATEGY):
            selection_mode = SELECTION_SELECTED_CATEGORIES
        def _ratio(key: str, default: float) -> float:
            from compare.dynamic_media_sampling import normalize_ratio
            return normalize_ratio(d.get(key, default), default)

        try:
            min_confidence = float(d.get("min_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_confidence = 0.0
        return ClassifierFilter(
            classifier_name=str(d.get("classifier_name", "") or ""),
            domain=domain,
            selection_mode=selection_mode,
            categories=[str(c) for c in (d.get("categories") or [])],
            mode="exclude" if d.get("mode") == "exclude" else "include",
            min_confidence=min_confidence,
            sample_ratio=_ratio("sample_ratio", 0.1),
            positive_ratio=_ratio("positive_ratio", 0.1),
        )
    if t == "group":
        children = [filter_from_dict(c) for c in d.get("filters", [])]
        children = [c for c in children if c is not None]
        try:
            op = FilterOperator(d.get("operator", "and"))
        except ValueError:
            op = FilterOperator.AND
        return CompareFilterGroup(operator=op, filters=children)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_filter(
    files: list,
    f: CompareFilter,
    metadata_reader=None,
    classifier_resolver=None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list:
    """
    Return the subset of *files* that passes filter *f*.

    metadata_reader     — optional override for model metadata reading (for tests);
                          defaults to the real image_data_extractor singleton.
    classifier_resolver — optional override mapping a ClassifierFilter to a
                          classifier wrapper (for tests); defaults to
                          resolve_classifier().
    progress            — called as progress(done, total) per file a
                          ClassifierFilter classifies; may raise to cancel.
    """
    if not f.is_active():
        return files

    if isinstance(f, CompareFilterGroup):
        return _apply_group(files, f, metadata_reader, classifier_resolver, progress)
    if isinstance(f, SizeFilter):
        return _apply_size(files, f)
    if isinstance(f, ModelFilter):
        return _apply_model(files, f, metadata_reader)
    if isinstance(f, ClassifierFilter):
        return _apply_classifier(files, f, classifier_resolver, progress)

    logger.warning(f"Unknown filter type {type(f).__name__} — skipping")
    return files


# ---------------------------------------------------------------------------
# Group dispatch
# ---------------------------------------------------------------------------

def contains_classifier_filter(f: Optional[CompareFilter]) -> bool:
    """True if *f* has an active ClassifierFilter anywhere in its tree."""
    if f is None or not f.is_active():
        return False
    if isinstance(f, ClassifierFilter):
        return True
    if isinstance(f, CompareFilterGroup):
        return any(contains_classifier_filter(c) for c in f.filters)
    return False


def _apply_group(
    files: list,
    group: CompareFilterGroup,
    metadata_reader,
    classifier_resolver=None,
    progress=None,
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
        # Cheap leaves first so classifiers only see files that survived them.
        result = files
        for child in sorted(active, key=contains_classifier_filter):
            result = apply_filter(result, child, metadata_reader, classifier_resolver, progress)
        return result

    if op == FilterOperator.OR:
        passed: set = set()
        for child in active:
            passed.update(apply_filter(files, child, metadata_reader, classifier_resolver, progress))
        return [fp for fp in files if fp in passed]

    if op == FilterOperator.NOT:
        excluded: set = set()
        for child in active:
            excluded.update(apply_filter(files, child, metadata_reader, classifier_resolver, progress))
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
    from compare.compare_size import extract_size_from_media  # local to avoid circular

    total = len(files)
    passed = []
    unreadable = 0

    for fp in files:
        dims = extract_size_from_media(fp)
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


# ---------------------------------------------------------------------------
# Classifier leaf
# ---------------------------------------------------------------------------

def _classifier_manager(domain: str):
    if domain == CLASSIFIER_DOMAIN_AUDIO:
        from image.audio_classifier_manager import audio_classifier_manager
        return audio_classifier_manager
    from image.image_classifier_manager import image_classifier_manager
    return image_classifier_manager


def resolve_classifier(f: ClassifierFilter):
    """Return the loaded, runnable classifier wrapper for *f*, or raise
    ClassifierFilterError."""
    name = (f.classifier_name or "").strip()
    manager = _classifier_manager(f.domain)
    key = manager.resolve_registered_model_name(name)
    if key is None:
        raise ClassifierFilterError(
            _("The classifier \"{0}\" used in the compare filters is not registered.").format(name)
        )
    try:
        wrapper = manager.get_classifier(key)
    except Exception as e:
        logger.error(f"ClassifierFilter: failed to load classifier {key!r}: {e}")
        wrapper = None
    if wrapper is None or not getattr(wrapper, "can_run", False):
        raise ClassifierFilterError(
            _("The classifier \"{0}\" used in the compare filters could not be loaded. "
              "See the log for details.").format(name)
        )
    return wrapper


def _selected_categories(f: ClassifierFilter, wrapper) -> frozenset:
    """The category set *f* matches against, validated against *wrapper*.
    Raises ClassifierFilterError if it is empty or names unknown categories."""
    name = f.classifier_name
    if f.selection_mode == SELECTION_MODEL_STRATEGY:
        selected = model_strategy_positive_categories(getattr(wrapper, "positive_groups", None))
        if not selected:
            raise ClassifierFilterError(
                _("The classifier \"{0}\" defines no positive groups, so its model "
                  "strategy cannot be used as a compare filter.").format(name)
            )
        return selected
    selected = frozenset(f.categories or [])
    unknown = sorted(selected - set(getattr(wrapper, "model_categories", []) or []))
    if unknown:
        raise ClassifierFilterError(
            _("The compare filter for classifier \"{0}\" uses categories the model "
              "does not have: {1}").format(name, ", ".join(unknown))
        )
    if not selected:
        raise ClassifierFilterError(
            _("The compare filter for classifier \"{0}\" has no categories selected.").format(name)
        )
    return selected


def _iter_classifier_filters(f: Optional[CompareFilter]):
    if f is None or not f.is_active():
        return
    if isinstance(f, ClassifierFilter):
        yield f
    elif isinstance(f, CompareFilterGroup):
        for child in f.filters:
            yield from _iter_classifier_filters(child)


def check_classifier_configs(f: Optional[CompareFilter]) -> List[str]:
    """Like validate_filter, but against the registered model configs only,
    so no model is loaded: unknown models, unknown categories, a model
    strategy without positive groups. Configs share the attribute names
    _selected_categories reads from a loaded wrapper."""
    errors: List[str] = []
    for cf in _iter_classifier_filters(f):
        name = (cf.classifier_name or "").strip()
        manager = _classifier_manager(cf.domain)
        key = manager.resolve_registered_model_name(name)
        if key is None:
            errors.append(
                _("The classifier \"{0}\" used in the compare filters is not registered.").format(name)
            )
            continue
        try:
            _selected_categories(cf, manager.classifier_metadata[key])
        except ClassifierFilterError as e:
            errors.append(str(e))
    return errors


def validate_filter(f: Optional[CompareFilter], classifier_resolver=None) -> List[str]:
    """User-facing problems that would stop *f* from running; empty if none.
    Loads every classifier the tree uses."""
    resolver = classifier_resolver or resolve_classifier
    errors: List[str] = []
    for cf in _iter_classifier_filters(f):
        try:
            _selected_categories(cf, resolver(cf))
        except ClassifierFilterError as e:
            errors.append(str(e))
    return errors


def _classifier_applies(fp: str, domain: str) -> bool:
    """Whether a *domain* classifier can judge *fp*: audio classifiers take audio
    files, image classifiers everything else."""
    from utils.audio_media import is_audio_path_by_extension
    return is_audio_path_by_extension(fp) == (domain == CLASSIFIER_DOMAIN_AUDIO)


def _classify(wrapper, domain: str, path: str) -> Tuple[str, float]:
    if domain == CLASSIFIER_DOMAIN_AUDIO:
        category = wrapper.classify_audio(path)
        scores = wrapper.predict_audio(path)
    else:
        category = wrapper.classify_image(path)
        scores = wrapper.predict_image(path)
    return category, float(scores.get(category, 0.0))


def _file_matches(fp: str, f: ClassifierFilter, wrapper, selected: frozenset) -> Optional[bool]:
    """Whether *fp* is in the selected categories; None if *f*'s domain
    doesn't apply to it. Raises if it can't be classified."""
    def frame_matches(path: str) -> Tuple[bool, Optional[str]]:
        category, score = _classify(wrapper, f.domain, path)
        return category in selected and score >= f.min_confidence, category

    if not _classifier_applies(fp, f.domain):
        return None
    if f.domain == CLASSIFIER_DOMAIN_AUDIO:
        return frame_matches(fp)[0]
    from compare.dynamic_media_sampling import FrameSampling, match_media
    return match_media(fp, frame_matches, FrameSampling(f.sample_ratio, f.positive_ratio)).matched


def _apply_classifier(files: list, f: ClassifierFilter, classifier_resolver, progress) -> list:
    wrapper = (classifier_resolver or resolve_classifier)(f)
    selected = _selected_categories(f, wrapper)

    total = len(files)
    passed = []
    not_applicable = 0
    errors = 0

    for i, fp in enumerate(files):
        if progress is not None:
            progress(i, total)
        matches = False
        try:
            result = _file_matches(fp, f, wrapper, selected)
            if result is None:
                not_applicable += 1
            else:
                matches = result
        except Exception as e:
            errors += 1
            logger.debug(f"ClassifierFilter: could not classify '{fp}': {e}")

        if (f.mode == 'include') == matches:
            passed.append(fp)

    if progress is not None:
        progress(total, total)

    excluded = total - len(passed)
    logger.info(
        f"ClassifierFilter ({f.classifier_name}, {f.mode}, {f.selection_mode}): "
        f"{excluded}/{total} file(s) excluded ({len(passed)} remain"
        + (f", {not_applicable} not applicable" if not_applicable else "")
        + (f", {errors} unclassifiable)" if errors else ")")
    )
    return passed

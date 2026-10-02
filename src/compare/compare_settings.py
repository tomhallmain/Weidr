"""
A CompareManager's compare settings as plain data, for callers without the
compare settings window (the MCP sessions): what describe_compare_settings
returns, apply_compare_settings accepts back, key for key.

Covers the window's global settings, data filter, compare instances and
their combination logic.

Errors are ValueErrors in English, like the rest of the MCP session surface,
except classifier problems, which reuse compare_filters' translated messages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from compare.compare_filters import (
    CLASSIFIER_DOMAIN_AUDIO,
    CLASSIFIER_DOMAIN_IMAGE,
    SELECTION_MODEL_STRATEGY,
    SELECTION_SELECTED_CATEGORIES,
    CompareFilter,
    check_classifier_configs,
    filter_from_dict,
    filter_to_dict,
)
from compare.compare_manager import MAX_INSTANCES, CombinationLogic
from utils.config import config
from utils.constants import CompareMode, Sort

SETTING_KEYS = (
    "instances",
    "combination_logic",
    "threshold",
    "threshold_mode",
    "counter_limit",
    "overwrite",
    "store_checkpoints",
    "use_matrix_comparison",
    "search_only_return_closest",
    "group_sort",
    "data_filter",
)


_INSTANCE_KEYS = ("compare_mode", "enabled", "threshold", "weight", "search_text", "search_text_negative")


def describe_compare_settings(cm) -> Dict[str, Any]:
    """Current settings of *cm*. ``threshold``/``counter_limit`` are None when
    unset (the mode's / config's default applies); ``threshold`` applies only
    while ``threshold_mode`` is the compare mode. The ``effective_*`` keys say
    what a run would use."""
    threshold, threshold_mode = cm.get_threshold_setting()
    return {
        "compare_mode": cm.compare_mode.name if cm.compare_mode else None,
        "composite": cm.is_composite_mode(),
        "combination_logic": cm.get_combination_logic().value,
        "instances": [
            {
                "compare_mode": inst.compare_mode.name,
                "enabled": inst.enabled,
                "threshold": inst.threshold,
                "weight": inst.weight,
                "search_text": inst.search_text,
                "search_text_negative": inst.search_text_negative,
            }
            for inst in cm.get_mode_instances()
        ],
        "threshold": threshold,
        "threshold_mode": threshold_mode.name if threshold_mode else None,
        "effective_threshold": cm.effective_threshold(),
        "counter_limit": cm.get_counter_limit(),
        "effective_counter_limit": cm.effective_counter_limit(),
        "overwrite": cm.get_overwrite(),
        "store_checkpoints": cm.get_store_checkpoints(),
        "use_matrix_comparison": cm.get_use_matrix_comparison(),
        "search_only_return_closest": bool(config.search_only_return_closest),
        "group_sort": config.compare_group_sort.name,
        "data_filter": filter_to_dict(cm.get_data_filter()),
    }


def apply_compare_settings(cm, changes: Dict[str, Any]) -> None:
    """Apply the keys present in *changes*; absent keys are left alone, and
    null resets ``threshold``/``counter_limit`` to their defaults or clears
    ``data_filter``. ``instances`` replaces the whole instance list; a
    ``threshold`` without ``threshold_mode`` is for the compare mode after
    that. Everything is validated before anything is applied."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError(f"settings must be a non-empty object with keys from: {', '.join(SETTING_KEYS)}")
    unknown = sorted(set(changes) - set(SETTING_KEYS))
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(unknown)} (allowed: {', '.join(SETTING_KEYS)})")

    parsed: Dict[str, Any] = {}
    if "instances" in changes:
        parsed["instances"] = _parse_instances(changes["instances"])
    if "combination_logic" in changes:
        names = [lg.value for lg in CombinationLogic]
        if changes["combination_logic"] not in names:
            raise ValueError(f"combination_logic must be one of {', '.join(names)}")
        parsed["combination_logic"] = CombinationLogic(changes["combination_logic"])
    if "threshold" in changes:
        parsed["threshold"] = _optional_number(changes["threshold"], "threshold", integer=False, minimum=0)
    if "threshold_mode" in changes:
        if "threshold" not in changes:
            raise ValueError("threshold_mode needs threshold")
        if changes["threshold_mode"] is not None:
            parsed["threshold_mode"] = _compare_mode(changes["threshold_mode"], "threshold_mode")
    if "counter_limit" in changes:
        parsed["counter_limit"] = _optional_number(changes["counter_limit"], "counter_limit", integer=True, minimum=1)
    for key in ("overwrite", "store_checkpoints", "use_matrix_comparison", "search_only_return_closest"):
        if key in changes:
            if not isinstance(changes[key], bool):
                raise ValueError(f"{key} must be true or false")
            parsed[key] = changes[key]
    if "group_sort" in changes:
        names = [s.name for s in Sort.non_random_options()]
        if changes["group_sort"] not in names:
            raise ValueError(f"group_sort must be one of {', '.join(names)}")
        parsed["group_sort"] = Sort[changes["group_sort"]]
    if "data_filter" in changes:
        parsed["data_filter"] = (
            None if changes["data_filter"] is None else parse_filter_strict(changes["data_filter"])
        )
        problems = check_classifier_configs(parsed["data_filter"])
        if problems:
            raise ValueError("; ".join(problems))

    if "instances" in parsed:
        cm.replace_mode_instances(parsed["instances"])
    if "combination_logic" in parsed:
        cm.set_combination_logic(parsed["combination_logic"])
    if "threshold" in parsed:
        cm.set_threshold(parsed["threshold"], parsed.get("threshold_mode"))
    if "counter_limit" in parsed:
        cm.set_counter_limit(parsed["counter_limit"])
    if "overwrite" in parsed:
        cm.set_overwrite(parsed["overwrite"])
    if "store_checkpoints" in parsed:
        cm.set_store_checkpoints(parsed["store_checkpoints"])
    if "use_matrix_comparison" in parsed:
        cm.set_use_matrix_comparison(parsed["use_matrix_comparison"])
    if "search_only_return_closest" in parsed:
        config.search_only_return_closest = parsed["search_only_return_closest"]
    if "group_sort" in parsed:
        # Persisted, as the compare settings window's Apply persists it.
        config.compare_group_sort = parsed["group_sort"]
        config.persist()
    if "data_filter" in parsed:
        cm.set_data_filter(parsed["data_filter"])


def _compare_mode(value, key: str) -> CompareMode:
    if not isinstance(value, str) or value not in CompareMode.__members__:
        raise ValueError(f"{key} must be a compare mode name, e.g. {CompareMode.CLIP_EMBEDDING.name}")
    return CompareMode[value]


def _parse_instances(value) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_INSTANCES:
        raise ValueError(f"instances must be a list of 1 to {MAX_INSTANCES} instance objects")
    specs = []
    for i, d in enumerate(value):
        where = f"instances[{i}]"
        if not isinstance(d, dict):
            raise ValueError(f"{where} must be an object")
        unknown = sorted(set(d) - set(_INSTANCE_KEYS))
        if unknown:
            raise ValueError(f"{where} has unknown keys: {', '.join(unknown)} (allowed: {', '.join(_INSTANCE_KEYS)})")
        spec: Dict[str, Any] = {"compare_mode": _compare_mode(d.get("compare_mode"), f"{where}.compare_mode")}
        if "enabled" in d:
            if not isinstance(d["enabled"], bool):
                raise ValueError(f"{where}.enabled must be true or false")
            spec["enabled"] = d["enabled"]
        if "threshold" in d:
            spec["threshold"] = _optional_number(d["threshold"], f"{where}.threshold", integer=False, minimum=0)
        if "weight" in d:
            if d["weight"] is None:
                raise ValueError(f"{where}.weight must be a number")
            spec["weight"] = _optional_number(d["weight"], f"{where}.weight", integer=False, minimum=0)
        for key in ("search_text", "search_text_negative"):
            text = d.get(key)
            if text is not None and not isinstance(text, str):
                raise ValueError(f"{where}.{key} must be a string or null")
            if text is not None and text.strip():
                # A non-embedding instance would fail its run on a text search.
                if not spec["compare_mode"].is_embedding():
                    raise ValueError(f"{where}.{key} needs an embedding compare mode")
                spec[key] = text.strip()
            elif key in d:
                spec[key] = None
        specs.append(spec)
    # The first instance sets the primary mode, which results are shown in.
    if not specs[0].get("enabled", True):
        raise ValueError("instances[0] must be enabled")
    return specs


def resolve_run_compare_mode(cm, mode: Optional[str], searching: bool) -> CompareMode:
    """The compare mode a run_compare (*searching* False) or run_search uses:
    *mode*, which the caller switches a single-mode setup to, or with *mode*
    None the configured setup as it stands. A composite setup runs only with
    *mode* None, since switching modes collapses it to one; run_compare also
    refuses one whose instances have search texts, as they make it a search."""
    if mode is None:
        if cm.compare_mode is None:
            raise ValueError("no compare mode is configured; pass mode")
        compare_mode = cm.compare_mode
    elif mode in CompareMode.__members__:
        compare_mode = CompareMode[mode]
    else:
        raise ValueError(f"unknown compare mode: {mode}")
    if cm.is_composite_mode():
        if mode is not None:
            modes = ", ".join(sorted(m.name for m in cm.get_active_modes()))
            raise ValueError(
                f"the compare setup is composite ({modes}); leave out mode to run it, "
                "or change its instances with set_compare_settings"
            )
        if not searching and any(
                cfg.enabled and (cfg.search_text or cfg.search_text_negative)
                for cfg in cm.get_mode_instances()):
            raise ValueError("the composite setup's instances have search texts; run it with run_search")
    return compare_mode


def _optional_number(value, key: str, integer: bool, minimum: float):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    if integer and int(value) != value:
        raise ValueError(f"{key} must be a whole number or null")
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return int(value) if integer else float(value)


# ---------------------------------------------------------------------------
# Data filter
# ---------------------------------------------------------------------------

_FILTER_KEYS = {
    "size": {"type", "min_size", "max_size", "exact_size", "size_tolerance"},
    "model": {"type", "models", "mode", "match_any", "include_loras"},
    "classifier": {"type", "classifier_name", "domain", "selection_mode", "categories",
                   "mode", "min_confidence", "sample_ratio", "positive_ratio"},
    "group": {"type", "operator", "filters"},
}


def parse_filter_strict(d: Any, where: str = "data_filter") -> CompareFilter:
    """filter_from_dict for untrusted input: rejects unknown types, keys and
    values instead of falling back to defaults, and an inactive filter (e.g.
    a classifier filter without categories) instead of returning one that
    does nothing."""
    _check_filter_dict(d, where)
    f = filter_from_dict(d)
    if f is None or not f.is_active():
        raise ValueError(f"{where} has no active condition")
    return f


def _check_filter_dict(d: Any, where: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be an object")
    t = d.get("type")
    if t not in _FILTER_KEYS:
        raise ValueError(f"{where}.type must be one of {', '.join(_FILTER_KEYS)}")
    unknown = sorted(set(d) - _FILTER_KEYS[t])
    if unknown:
        raise ValueError(f"{where} ({t}) has unknown keys: {', '.join(unknown)}")

    def one_of(key, allowed, default):
        if d.get(key, default) not in allowed:
            raise ValueError(f"{where}.{key} must be one of {', '.join(allowed)}")

    def str_list(key):
        value = d.get(key)
        if value is not None and (not isinstance(value, list) or not all(isinstance(v, str) for v in value)):
            raise ValueError(f"{where}.{key} must be a list of strings")

    def ratio(key):
        value = d.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not 0 <= value <= 1):
            raise ValueError(f"{where}.{key} must be a number from 0 to 1")

    def flag(key):
        if key in d and not isinstance(d[key], bool):
            raise ValueError(f"{where}.{key} must be true or false")

    if t == "group":
        one_of("operator", ("and", "or", "not"), "and")
        children = d.get("filters")
        if not isinstance(children, list) or not children:
            raise ValueError(f"{where}.filters must be a non-empty list")
        for i, child in enumerate(children):
            _check_filter_dict(child, f"{where}.filters[{i}]")
    elif t == "size":
        for key in ("min_size", "max_size", "exact_size"):
            value = d.get(key)
            if value is not None and not (
                    isinstance(value, list) and len(value) == 2
                    and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)):
                raise ValueError(f"{where}.{key} must be [width, height] or null")
        tolerance = d.get("size_tolerance", 0)
        if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
            raise ValueError(f"{where}.size_tolerance must be a whole number >= 0")
    elif t == "model":
        str_list("models")
        one_of("mode", ("include", "exclude"), "include")
        flag("match_any")
        flag("include_loras")
    else:
        if not isinstance(d.get("classifier_name"), str) or not d["classifier_name"].strip():
            raise ValueError(f"{where}.classifier_name must be a model name")
        one_of("domain", (CLASSIFIER_DOMAIN_IMAGE, CLASSIFIER_DOMAIN_AUDIO), CLASSIFIER_DOMAIN_IMAGE)
        one_of("selection_mode", (SELECTION_SELECTED_CATEGORIES, SELECTION_MODEL_STRATEGY),
               SELECTION_SELECTED_CATEGORIES)
        one_of("mode", ("include", "exclude"), "include")
        str_list("categories")
        for key in ("min_confidence", "sample_ratio", "positive_ratio"):
            ratio(key)


# ---------------------------------------------------------------------------
# Classifier discovery
# ---------------------------------------------------------------------------

def list_classifier_models() -> Dict[str, List[Dict[str, Any]]]:
    """Registered classifier models a classifier data filter can name, with
    the categories it can select."""
    from compare.classifier_categories import model_strategy_positive_categories
    from image.audio_classifier_manager import audio_classifier_manager
    from image.image_classifier_manager import image_classifier_manager

    def describe(domain: str, manager) -> List[Dict[str, Any]]:
        return [
            {
                "domain": domain,
                "name": cfg.model_name,
                "categories": list(cfg.model_categories),
                # Empty when the model has no positive groups, so
                # selection_mode "model_strategy" is unavailable for it.
                "model_strategy_categories": sorted(model_strategy_positive_categories(cfg.positive_groups)),
            }
            for cfg in manager.get_model_configs()
        ]

    return {
        "classifiers": describe(CLASSIFIER_DOMAIN_IMAGE, image_classifier_manager)
        + describe(CLASSIFIER_DOMAIN_AUDIO, audio_classifier_manager),
    }

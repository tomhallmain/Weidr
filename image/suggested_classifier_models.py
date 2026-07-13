"""Curated, known-good image classifier models.

Definitions live in configs/suggested_classifier_models.json (edit that file to
add/adjust entries, not this module) and are surfaced in the HF Model Manager
window's "Suggested Models" tab so users can install them with one click,
instead of hand-editing configs/config.json or guessing at model_kwargs
(architecture_module_name, input_shape, etc.).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logging_setup import get_logger

logger = get_logger("suggested_classifier_models")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SUGGESTED_MODELS_JSON_PATH = os.path.join(_REPO_ROOT, "configs", "suggested_classifier_models.json")


@dataclass(slots=True, frozen=True)
class SuggestedClassifierModel:
    display_name: str
    model_name: str
    description: str
    hf_repo_id: str
    hf_selected_filename: str
    model_categories: tuple[str, ...]
    backend: str = "pytorch"
    input_shape: Optional[tuple[int, int]] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    # (url, filename) pairs downloaded into the same directory as the HF
    # snapshot, for models whose architecture code isn't published alongside
    # their weights on HF (fetched fresh at install time, never bundled here).
    extra_source_files: tuple[tuple[str, str], ...] = ()
    positive_groups: tuple[tuple[str, ...], ...] = ()
    neutral_categories: tuple[str, ...] = ()
    severity_order: tuple[str, ...] = ()

    def to_model_details(self, downloaded_path: str) -> dict[str, Any]:
        """Build an ``ImageClassifierModelConfig``-shaped dict for a downloaded file."""
        details: dict[str, Any] = {
            "model_name": self.model_name,
            "model_location": downloaded_path,
            "model_categories": list(self.model_categories),
            "backend": self.backend,
            "hf_repo_id": self.hf_repo_id,
            "hf_selected_filename": self.hf_selected_filename,
        }
        if self.input_shape is not None:
            details["input_shape"] = list(self.input_shape)
        if self.model_kwargs:
            details["model_kwargs"] = dict(self.model_kwargs)
        if self.positive_groups:
            details["positive_groups"] = [list(g) for g in self.positive_groups]
        if self.neutral_categories:
            details["neutral_categories"] = list(self.neutral_categories)
        if self.severity_order:
            details["severity_order"] = list(self.severity_order)
        return details


def _parse_extra_source_files(raw: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        return ()
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("url") and item.get("filename"):
            out.append((str(item["url"]), str(item["filename"])))
    return tuple(out)


def _parse_entry(raw: dict[str, Any]) -> SuggestedClassifierModel:
    categories = raw.get("model_categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("model_categories must be a non-empty list")

    input_shape_raw = raw.get("input_shape")
    input_shape: Optional[tuple[int, int]] = None
    if isinstance(input_shape_raw, list) and len(input_shape_raw) == 2:
        input_shape = (int(input_shape_raw[0]), int(input_shape_raw[1]))

    positive_groups_raw = raw.get("positive_groups", [])
    positive_groups = tuple(
        tuple(str(c) for c in group) for group in positive_groups_raw if isinstance(group, list)
    )

    return SuggestedClassifierModel(
        display_name=str(raw["display_name"]),
        model_name=str(raw["model_name"]),
        description=str(raw.get("description", "")),
        hf_repo_id=str(raw["hf_repo_id"]),
        hf_selected_filename=str(raw["hf_selected_filename"]),
        model_categories=tuple(str(c) for c in categories),
        backend=str(raw.get("backend", "pytorch")),
        input_shape=input_shape,
        model_kwargs=dict(raw.get("model_kwargs", {})),
        extra_source_files=_parse_extra_source_files(raw.get("extra_source_files")),
        positive_groups=positive_groups,
        neutral_categories=tuple(str(c) for c in raw.get("neutral_categories", [])),
        severity_order=tuple(str(c) for c in raw.get("severity_order", [])),
    )


def _load_suggested_classifier_models() -> tuple[SuggestedClassifierModel, ...]:
    try:
        with open(_SUGGESTED_MODELS_JSON_PATH, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)
    except FileNotFoundError:
        logger.warning("Suggested classifier models file not found: %s", _SUGGESTED_MODELS_JSON_PATH)
        return ()
    except Exception:
        logger.exception("Failed to read suggested classifier models file: %s", _SUGGESTED_MODELS_JSON_PATH)
        return ()

    if not isinstance(raw_entries, list):
        logger.error("Suggested classifier models file must contain a JSON array: %s", _SUGGESTED_MODELS_JSON_PATH)
        return ()

    parsed: list[SuggestedClassifierModel] = []
    for raw in raw_entries:
        try:
            parsed.append(_parse_entry(raw))
        except Exception:
            logger.exception(
                "Skipping invalid suggested classifier model entry (model_name=%r)",
                raw.get("model_name") if isinstance(raw, dict) else raw,
            )
    return tuple(parsed)


SUGGESTED_CLASSIFIER_MODELS: tuple[SuggestedClassifierModel, ...] = _load_suggested_classifier_models()

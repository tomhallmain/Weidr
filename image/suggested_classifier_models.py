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
    model_categories: tuple[str, ...]
    # "image" (default, backward compatible with entries predating this field) or
    # "audio". Determines which ModelConfig shape to_model_details() builds and
    # which config list / manager the Model Manager window installs into.
    classifier_type: str = "image"
    # Required for classifier_type="image" (the specific file to download from the
    # repo); unused for "audio" -- AutoModelForAudioClassification.from_pretrained
    # resolves the whole repo snapshot itself, no single-file selection needed.
    hf_selected_filename: str = ""
    backend: str = "pytorch"
    input_shape: Optional[tuple[int, int]] = None
    # Audio-only. None means fall through to AudioClassifierModelConfig's own
    # defaults (16000 / 10.0) rather than redundantly restating them here.
    sample_rate: Optional[int] = None
    max_duration_seconds: Optional[float] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    # (url, filename) pairs downloaded into the same directory as the HF
    # snapshot, for models whose architecture code isn't published alongside
    # their weights on HF (fetched fresh at install time, never bundled here).
    # Image-only -- audio installs never download a snapshot locally (see above).
    extra_source_files: tuple[tuple[str, str], ...] = ()
    positive_groups: tuple[tuple[str, ...], ...] = ()
    neutral_categories: tuple[str, ...] = ()
    severity_order: tuple[str, ...] = ()

    def to_model_details(self, downloaded_path: str) -> dict[str, Any]:
        """Build a ModelConfig-shaped dict -- ``ImageClassifierModelConfig`` or
        ``AudioClassifierModelConfig`` depending on ``classifier_type``.

        For images, *downloaded_path* is the local path to the downloaded model
        file. For audio, callers pass ``hf_repo_id`` itself as *downloaded_path*
        (see ``_SuggestedModelInstallWorker`` / the audio fast path in
        ``HfModelManagerWindow`` -- no local download step) since ``model_location``
        for an audio classifier is the repo id/directory handed straight to
        ``from_pretrained``, not a specific file.
        """
        if self.classifier_type == "audio":
            details: dict[str, Any] = {
                "model_name": self.model_name,
                "model_location": downloaded_path,
                "model_categories": list(self.model_categories),
                "hf_repo_id": self.hf_repo_id,
            }
            if self.sample_rate is not None:
                details["sample_rate"] = self.sample_rate
            if self.max_duration_seconds is not None:
                details["max_duration_seconds"] = self.max_duration_seconds
            if self.model_kwargs:
                details["model_kwargs"] = dict(self.model_kwargs)
            if self.positive_groups:
                details["positive_groups"] = [list(g) for g in self.positive_groups]
            if self.neutral_categories:
                details["neutral_categories"] = list(self.neutral_categories)
            if self.severity_order:
                details["severity_order"] = list(self.severity_order)
            return details

        details = {
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


_VALID_CLASSIFIER_TYPES = {"image", "audio"}


def _parse_entry(raw: dict[str, Any]) -> SuggestedClassifierModel:
    categories = raw.get("model_categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("model_categories must be a non-empty list")

    classifier_type = str(raw.get("classifier_type", "image")).strip().lower()
    if classifier_type not in _VALID_CLASSIFIER_TYPES:
        raise ValueError(
            f"classifier_type must be one of {sorted(_VALID_CLASSIFIER_TYPES)}, got {classifier_type!r}"
        )
    if classifier_type == "image" and not raw.get("hf_selected_filename"):
        raise ValueError("hf_selected_filename is required for classifier_type='image'")

    input_shape_raw = raw.get("input_shape")
    input_shape: Optional[tuple[int, int]] = None
    if isinstance(input_shape_raw, list) and len(input_shape_raw) == 2:
        input_shape = (int(input_shape_raw[0]), int(input_shape_raw[1]))

    sample_rate_raw = raw.get("sample_rate")
    sample_rate = int(sample_rate_raw) if sample_rate_raw is not None else None
    max_duration_raw = raw.get("max_duration_seconds")
    max_duration_seconds = float(max_duration_raw) if max_duration_raw is not None else None

    positive_groups_raw = raw.get("positive_groups", [])
    positive_groups = tuple(
        tuple(str(c) for c in group) for group in positive_groups_raw if isinstance(group, list)
    )

    return SuggestedClassifierModel(
        display_name=str(raw["display_name"]),
        model_name=str(raw["model_name"]),
        description=str(raw.get("description", "")),
        hf_repo_id=str(raw["hf_repo_id"]),
        classifier_type=classifier_type,
        hf_selected_filename=str(raw.get("hf_selected_filename", "")),
        model_categories=tuple(str(c) for c in categories),
        backend=str(raw.get("backend", "pytorch")),
        input_shape=input_shape,
        sample_rate=sample_rate,
        max_duration_seconds=max_duration_seconds,
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

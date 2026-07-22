from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class AudioClassifierModelConfig:
    """Config for a Hugging Face ``AutoModelForAudioClassification``-compatible
    audio classifier, registered via ``config.audio_classifier_models``.

    Deliberately narrower than :class:`image.image_classifier_model_config.ImageClassifierModelConfig`
    (no backend selection, no custom architecture loading, no safetensors/H5
    paths) -- Track B's first pass only supports the general Hugging Face
    ``transformers`` audio-classification path (see
    ``docs/audio-embeddings-and-classification-design.md``). ``model_location``
    is passed directly to ``from_pretrained`` and may be either a local
    directory or a bare Hugging Face repo id (``transformers`` resolves and
    caches remote repo ids itself; unlike the image classifier's HF path, no
    pre-download step is required).
    """

    model_name: str
    model_location: str
    model_categories: list[str]
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    hf_repo_id: Optional[str] = None
    hf_selected_filename: Optional[str] = None
    # Target sample rate for waveform decoding. Overridden at load time by the
    # feature extractor's own ``sampling_rate`` when available -- decoding at
    # the wrong rate silently degrades every prediction, so the model's own
    # stated rate always wins over this config value.
    sample_rate: int = 16000
    # Fixed-length analysis window in seconds (Track B's first pass processes
    # one window per file rather than sampling multiple segments -- see the
    # design doc's "segment sampling for long audio" open question).
    max_duration_seconds: float = 10.0
    positive_groups: list[list[str]] = field(default_factory=list)
    neutral_categories: list[str] = field(default_factory=list)
    severity_order: list[str] = field(default_factory=list)

    REQUIRED_KEYS = {"model_name", "model_location", "model_categories"}
    WRAPPER_ALLOWED_KEYS = {
        "model_name",
        "model_categories",
        "model_location",
        "model_kwargs",
        "sample_rate",
        "max_duration_seconds",
        "positive_groups",
        "neutral_categories",
        "severity_order",
    }
    AUXILIARY_KEYS = {"hf_repo_id", "hf_selected_filename"}
    KNOWN_KEYS = WRAPPER_ALLOWED_KEYS.union(AUXILIARY_KEYS)

    @staticmethod
    def _validate_category_references(
        model_categories: list[str],
        positive_groups: list[list[str]],
        neutral_categories: list[str],
        severity_order: list[str],
    ) -> None:
        """Every name in split fields must match a ``model_categories`` entry exactly (no typos)."""
        allowed = set(model_categories)
        hint = f"allowed names: {sorted(allowed)}"
        for gi, grp in enumerate(positive_groups):
            for c in grp:
                if c not in allowed:
                    raise ValueError(
                        f"positive_groups[{gi}] references unknown category {c!r}; {hint}"
                    )
        for c in neutral_categories:
            if c not in allowed:
                raise ValueError(
                    f"neutral_categories references unknown category {c!r}; {hint}"
                )
        for si, c in enumerate(severity_order):
            if c not in allowed:
                raise ValueError(
                    f"severity_order[{si}] references unknown category {c!r}; {hint}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any], logger=None, warn_unknown_keys: bool = True) -> "AudioClassifierModelConfig":
        if not isinstance(data, dict):
            raise ValueError(f"Expected model config dict, got {type(data)}")

        missing_required = [key for key in cls.REQUIRED_KEYS if key not in data]
        if missing_required:
            raise ValueError(f"Missing required model config keys: {missing_required}")

        unknown_keys = [k for k in data.keys() if k not in cls.KNOWN_KEYS]
        if unknown_keys and logger is not None and warn_unknown_keys:
            suggestions = []
            for unknown in unknown_keys:
                close = difflib.get_close_matches(unknown, list(cls.KNOWN_KEYS), n=1)
                if close:
                    suggestions.append(f"{unknown}->{close[0]}")
            suggestion_text = f" (did you mean: {', '.join(suggestions)})" if suggestions else ""
            logger.warning(f"Unsupported audio model config keys ignored: {unknown_keys}{suggestion_text}")

        model_name = str(data.get("model_name", "") or "").strip()
        model_location = str(data.get("model_location", "") or "").strip()
        if not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not model_location:
            raise ValueError("model_location must be a non-empty string")

        categories = data.get("model_categories")
        if not isinstance(categories, list) or len(categories) == 0:
            raise ValueError("model_categories must be a non-empty list")
        model_categories = [str(c).strip() for c in categories if str(c).strip()]
        if len(model_categories) == 0:
            raise ValueError("model_categories must contain at least one non-empty category")

        model_kwargs = data.get("model_kwargs", {})
        if model_kwargs is None:
            model_kwargs = {}
        if not isinstance(model_kwargs, dict):
            raise ValueError("model_kwargs must be a dict when provided")

        sample_rate_raw = data.get("sample_rate", 16000)
        try:
            sample_rate = int(sample_rate_raw)
        except (TypeError, ValueError):
            raise ValueError(f"sample_rate must be an int, got {sample_rate_raw!r}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        max_duration_raw = data.get("max_duration_seconds", 10.0)
        try:
            max_duration_seconds = float(max_duration_raw)
        except (TypeError, ValueError):
            raise ValueError(f"max_duration_seconds must be a number, got {max_duration_raw!r}")
        if max_duration_seconds <= 0:
            raise ValueError(f"max_duration_seconds must be positive, got {max_duration_seconds}")

        positive_groups_raw = data.get("positive_groups", [])
        if positive_groups_raw is None:
            positive_groups_raw = []
        if not isinstance(positive_groups_raw, list):
            raise ValueError("positive_groups must be a list of category lists when provided")
        positive_groups: list[list[str]] = []
        for item in positive_groups_raw:
            if not isinstance(item, list):
                raise ValueError("positive_groups must be a list of lists of category names")
            positive_groups.append([str(c).strip() for c in item if str(c).strip()])

        neutral_categories_raw = data.get("neutral_categories", [])
        if neutral_categories_raw is None:
            neutral_categories_raw = []
        if not isinstance(neutral_categories_raw, list):
            raise ValueError("neutral_categories must be a list when provided")
        neutral_categories = [str(c).strip() for c in neutral_categories_raw if str(c).strip()]

        severity_order_raw = data.get("severity_order", [])
        if severity_order_raw is None:
            severity_order_raw = []
        if not isinstance(severity_order_raw, list):
            raise ValueError("severity_order must be a list when provided")
        severity_order = [str(c).strip() for c in severity_order_raw if str(c).strip()]

        cat_counts = Counter(model_categories)
        dupes = sorted(c for c, n in cat_counts.items() if n > 1)
        if dupes:
            raise ValueError(f"model_categories must not contain duplicates: {dupes}")

        cls._validate_category_references(model_categories, positive_groups, neutral_categories, severity_order)

        hf_repo_id_raw = str(data.get("hf_repo_id", "") or "").strip()
        hf_selected_filename_raw = str(data.get("hf_selected_filename", "") or "").strip()

        return cls(
            model_name=model_name,
            model_location=model_location,
            model_categories=model_categories,
            model_kwargs=dict(model_kwargs),
            hf_repo_id=hf_repo_id_raw if hf_repo_id_raw else None,
            hf_selected_filename=hf_selected_filename_raw if hf_selected_filename_raw else None,
            sample_rate=sample_rate,
            max_duration_seconds=max_duration_seconds,
            positive_groups=positive_groups,
            neutral_categories=neutral_categories,
            severity_order=severity_order,
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "model_name": self.model_name,
            "model_location": self.model_location,
            "model_categories": list(self.model_categories),
            "sample_rate": int(self.sample_rate),
            "max_duration_seconds": float(self.max_duration_seconds),
        }
        if self.model_kwargs:
            out["model_kwargs"] = dict(self.model_kwargs)
        if self.hf_repo_id:
            out["hf_repo_id"] = self.hf_repo_id
        if self.hf_selected_filename:
            out["hf_selected_filename"] = self.hf_selected_filename
        if self.positive_groups:
            out["positive_groups"] = [list(g) for g in self.positive_groups]
        if self.neutral_categories:
            out["neutral_categories"] = list(self.neutral_categories)
        if self.severity_order:
            out["severity_order"] = list(self.severity_order)
        return out

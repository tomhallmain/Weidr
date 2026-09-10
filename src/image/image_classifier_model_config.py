from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass(slots=True)
class ImageClassifierModelConfig:
    model_name: str
    model_location: str
    model_categories: list[str]
    use_hub_keras_layers: bool = False
    backend: str = "auto"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    hf_repo_id: Optional[str] = None
    hf_selected_filename: Optional[str] = None
    # Optional (width, height) in pixels for PyTorch resize path; overrides inference when set.
    input_shape: Optional[Tuple[int, int]] = None
    positive_groups: list[list[str]] = field(default_factory=list)
    neutral_categories: list[str] = field(default_factory=list)
    severity_order: list[str] = field(default_factory=list)

    REQUIRED_KEYS = {"model_name", "model_location", "model_categories"}
    WRAPPER_ALLOWED_KEYS = {
        "model_name",
        "model_categories",
        "model_location",
        "use_hub_keras_layers",
        "backend",
        "model_kwargs",
        "input_shape",
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
    def from_dict(cls, data: dict[str, Any], logger=None, warn_unknown_keys: bool = True) -> "ImageClassifierModelConfig":
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
            logger.warning(f"Unsupported image model config keys ignored: {unknown_keys}{suggestion_text}")

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

        use_hub_keras_layers = bool(data.get("use_hub_keras_layers", False))
        backend = str(data.get("backend", "auto") or "auto").strip().lower()
        if backend == "":
            backend = "auto"

        model_kwargs = data.get("model_kwargs", {})
        if model_kwargs is None:
            model_kwargs = {}
        if not isinstance(model_kwargs, dict):
            raise ValueError("model_kwargs must be a dict when provided")

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

        input_shape_top = cls.parse_input_shape(data.get("input_shape"))
        input_shape_kw = cls.parse_input_shape(model_kwargs.get("input_shape"))
        input_shape = input_shape_top or input_shape_kw
        model_kwargs_out = dict(model_kwargs)
        if "input_shape" in model_kwargs_out:
            model_kwargs_out.pop("input_shape", None)

        return cls(
            model_name=model_name,
            model_location=model_location,
            model_categories=model_categories,
            use_hub_keras_layers=use_hub_keras_layers,
            backend=backend,
            model_kwargs=model_kwargs_out,
            hf_repo_id=hf_repo_id_raw if hf_repo_id_raw else None,
            hf_selected_filename=hf_selected_filename_raw if hf_selected_filename_raw else None,
            input_shape=input_shape,
            positive_groups=positive_groups,
            neutral_categories=neutral_categories,
            severity_order=severity_order,
        )

    @staticmethod
    def parse_input_shape(raw: Any) -> Optional[Tuple[int, int]]:
        """Parse user-specified input size as (width, height). Returns None if unset/invalid."""
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().lower().replace(" ", "")
            if not s:
                return None
            for sep in ("x", ",", "*"):
                if sep in s:
                    left, right = s.split(sep, 1)
                    return ImageClassifierModelConfig.parse_input_shape([left, right])
            return None
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                w = int(raw[0])
                h = int(raw[1])
            except (TypeError, ValueError):
                return None
            if w <= 0 or h <= 0:
                return None
            return (w, h)
        if isinstance(raw, int):
            n = int(raw)
            if n <= 0:
                return None
            return (n, n)
        if isinstance(raw, dict):
            w = raw.get("width", raw.get("w"))
            h = raw.get("height", raw.get("h"))
            if w is None or h is None:
                return None
            try:
                wi = int(w)
                hi = int(h)
            except (TypeError, ValueError):
                return None
            if wi <= 0 or hi <= 0:
                return None
            return (wi, hi)
        return None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "model_name": self.model_name,
            "model_location": self.model_location,
            "model_categories": list(self.model_categories),
            "use_hub_keras_layers": bool(self.use_hub_keras_layers),
            "backend": self.backend,
        }
        if self.input_shape is not None:
            out["input_shape"] = [int(self.input_shape[0]), int(self.input_shape[1])]
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

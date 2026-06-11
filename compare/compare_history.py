"""
Recent compare-run history.

Each CompareHistory entry snapshots CompareManager state for a directory:

- ``instances``: serialized :class:`compare.compare_manager.CompareConfig` rows
  (per-mode instance: mode, weight, threshold override, search text, …).
- ``run_settings``: global options applied to :class:`compare.compare_args.CompareArgs`
  on each run (overwrite, checkpoints, matrix path, threshold, file limit).
- ``combination_logic`` / ``filter_dict``: composite logic and pre-filter tree.

CompareArgs itself is not stored in history; it is built at run time (base_dir,
search media, listener, app_actions, …) and receives ``run_settings`` via
``CompareManager.apply_settings_to_args()``.

Persisted via app_info_cache metadata across sessions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger

logger = get_logger("compare_history")

MAX_HISTORY = 20
_CACHE_KEY = "recent_compare_history"


@dataclass
class CompareRunSettings:
    """Global compare options (CompareManager → CompareArgs), not per-instance."""

    overwrite: bool = False
    store_checkpoints: bool = False
    use_matrix_comparison: bool = True
    threshold: Optional[float] = None
    counter_limit: Optional[int] = None

    def to_json(self) -> dict:
        return {
            "overwrite": self.overwrite,
            "store_checkpoints": self.store_checkpoints,
            "use_matrix_comparison": self.use_matrix_comparison,
            "threshold": self.threshold,
            "counter_limit": self.counter_limit,
        }

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "CompareRunSettings":
        if not data:
            return cls()
        return cls(
            overwrite=bool(data.get("overwrite", False)),
            store_checkpoints=bool(data.get("store_checkpoints", False)),
            use_matrix_comparison=bool(data.get("use_matrix_comparison", True)),
            threshold=data.get("threshold"),
            counter_limit=data.get("counter_limit"),
        )


@dataclass
class CompareHistory:
    """Snapshot of a CompareManager configuration tied to an analysis directory."""

    directory: str
    timestamp: str  # ISO-8601
    instances: List[dict]  # CompareConfig-equivalent plain dicts
    combination_logic: str  # CombinationLogic.value e.g. "AND"
    filter_dict: Optional[dict]  # serialized filter tree, or None
    run_settings: CompareRunSettings = field(default_factory=CompareRunSettings)
    file_filter: Optional[str] = None  # sidebar file-filter pattern string

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "directory": self.directory,
            "timestamp": self.timestamp,
            "instances": self.instances,
            "combination_logic": self.combination_logic,
            "filter_dict": self.filter_dict,
            "run_settings": self.run_settings.to_json(),
            "file_filter": self.file_filter,
        }

    @classmethod
    def from_json(cls, d: dict) -> Optional["CompareHistory"]:
        try:
            run_settings = CompareRunSettings.from_json(d.get("run_settings"))
            # Legacy: lone top-level use_matrix_comparison before run_settings existed
            if "run_settings" not in d and "use_matrix_comparison" in d:
                run_settings = CompareRunSettings(
                    use_matrix_comparison=bool(d["use_matrix_comparison"]),
                )
            return cls(
                directory=d["directory"],
                timestamp=d["timestamp"],
                instances=d.get("instances", []),
                combination_logic=d.get("combination_logic", "AND"),
                filter_dict=d.get("filter_dict"),
                run_settings=run_settings,
                file_filter=d.get("file_filter") or d.get("inclusion_pattern") or None,
            )
        except (KeyError, TypeError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def label(self) -> str:
        """Short human-readable label for the history list."""
        from utils.utils import Utils
        dir_raw = Utils.get_relative_dirpath(self.directory, levels=2) or self.directory
        dir_display = Utils.get_centrally_truncated_string(dir_raw, 34)
        modes = ", ".join(i.get("compare_mode", "?") for i in self.instances)
        logic = self.combination_logic
        if len(self.instances) > 1:
            base = f"{dir_display}  [{modes}] ({logic})"
        else:
            base = f"{dir_display}  [{modes}]"
        if self.file_filter:
            base += f"  | {self.file_filter}"
        return base

    def _identity_key(self) -> str:
        """JSON key for deduplication — everything except the timestamp."""
        return json.dumps({
            "directory": self.directory,
            "instances": self.instances,
            "combination_logic": self.combination_logic,
            "filter_dict": self.filter_dict,
            "run_settings": self.run_settings.to_json(),
            "file_filter": self.file_filter,
        }, sort_keys=True)

    # ------------------------------------------------------------------
    # Class-level persistence
    # ------------------------------------------------------------------
    @classmethod
    def load_recent(cls) -> List["CompareHistory"]:
        """Load the persisted history list from app_info_cache."""
        raw = app_info_cache.get_meta(_CACHE_KEY, []) or []
        result = []
        for d in raw:
            h = cls.from_json(d)
            if h is not None:
                result.append(h)
        return result

    @classmethod
    def store_recent(cls, history: List["CompareHistory"]) -> None:
        """Persist the history list to app_info_cache."""
        app_info_cache.set_meta(_CACHE_KEY, [h.to_json() for h in history])

    @classmethod
    def record(cls, new_entry: "CompareHistory") -> None:
        """Insert new_entry at the head, deduplicate by identity key, cap at MAX_HISTORY."""
        existing = cls.load_recent()
        new_key = new_entry._identity_key()
        existing = [h for h in existing if h._identity_key() != new_key]
        existing.insert(0, new_entry)
        if len(existing) > MAX_HISTORY:
            existing = existing[:MAX_HISTORY]
        cls.store_recent(existing)

    @classmethod
    def remove(cls, entry: "CompareHistory") -> None:
        """Remove a specific entry from the persisted history."""
        key = entry._identity_key()
        existing = [h for h in cls.load_recent() if h._identity_key() != key]
        cls.store_recent(existing)

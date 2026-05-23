"""
Recent compare-run history.

Each CompareHistory entry is a snapshot of a CompareManager configuration
tied to the directory that was analysed. Entries are persisted via
app_info_cache metadata so they survive between sessions.

Usage (from CompareManager):
    from compare.compare_history import CompareHistory
    CompareHistory.record(self.snapshot(args.base_dir))

Usage (from UI to restore):
    for h in CompareHistory.load_recent():
        ...  # display h.label()
    manager.apply_snapshot(h)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger

logger = get_logger("compare_history")

MAX_HISTORY = 20
_CACHE_KEY = "recent_compare_history"


@dataclass
class CompareHistory:
    """Snapshot of a CompareManager configuration tied to an analysis directory."""
    directory: str
    timestamp: str       # ISO-8601
    instances: List[dict]       # list of CompareConfig-equivalent plain dicts
    combination_logic: str      # CombinationLogic.value e.g. "AND"
    filter_dict: Optional[dict] # serialized filter tree, or None

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
        }

    @classmethod
    def from_json(cls, d: dict) -> Optional[CompareHistory]:
        try:
            return cls(
                directory=d["directory"],
                timestamp=d["timestamp"],
                instances=d.get("instances", []),
                combination_logic=d.get("combination_logic", "AND"),
                filter_dict=d.get("filter_dict"),
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
            return f"{dir_display}  [{modes}] ({logic})"
        return f"{dir_display}  [{modes}]"

    def _identity_key(self) -> str:
        """JSON key for deduplication — everything except the timestamp."""
        return json.dumps({
            "directory": self.directory,
            "instances": self.instances,
            "combination_logic": self.combination_logic,
            "filter_dict": self.filter_dict,
        }, sort_keys=True)

    # ------------------------------------------------------------------
    # Class-level persistence
    # ------------------------------------------------------------------
    @classmethod
    def load_recent(cls) -> List[CompareHistory]:
        """Load the persisted history list from app_info_cache."""
        raw = app_info_cache.get_meta(_CACHE_KEY, []) or []
        result = []
        for d in raw:
            h = cls.from_json(d)
            if h is not None:
                result.append(h)
        return result

    @classmethod
    def store_recent(cls, history: List[CompareHistory]) -> None:
        """Persist the history list to app_info_cache."""
        app_info_cache.set_meta(_CACHE_KEY, [h.to_json() for h in history])

    @classmethod
    def record(cls, new_entry: CompareHistory) -> None:
        """Insert new_entry at the head, deduplicate by identity key, cap at MAX_HISTORY."""
        existing = cls.load_recent()
        new_key = new_entry._identity_key()
        existing = [h for h in existing if h._identity_key() != new_key]
        existing.insert(0, new_entry)
        if len(existing) > MAX_HISTORY:
            existing = existing[:MAX_HISTORY]
        cls.store_recent(existing)

    @classmethod
    def remove(cls, entry: CompareHistory) -> None:
        """Remove a specific entry from the persisted history."""
        key = entry._identity_key()
        existing = [h for h in cls.load_recent() if h._identity_key() != key]
        cls.store_recent(existing)

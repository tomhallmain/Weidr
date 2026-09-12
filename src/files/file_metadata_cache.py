"""
Persistent per-file metadata, keyed by path.

A SortableFile's stat, image dimensions and related-image lookup each cost I/O,
and together they dominate a directory load on a slow or external drive.
Holding those results across sessions removes the cost for files already seen,
at the price of serving stale values for a file changed outside the app -- the
trade this cache deliberately makes. Which files exist still comes from the
directory scan, so the file list itself is never stale, and an entry for a
deleted file is simply never consulted.

Keyed by path with no mtime check: a validity check would need the stat call
the cache exists to avoid.
"""

import threading
from typing import Dict, Optional

from utils.config import config
from utils.logging_setup import get_logger

logger = get_logger("file_metadata_cache")

CACHE_KEY = "file_metadata_cache"


def is_enabled() -> bool:
    return bool(getattr(config, "enable_file_metadata_cache", False))


class FileMetadataCache:
    # Entries are ~180 bytes serialized, and the whole app_info_cache blob is
    # read and written at once, so this ceiling is a size budget rather than a
    # correctness limit.
    MAX_ENTRIES = 100_000

    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}
        self._dirty = False
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Populate from app_info_cache. Idempotent: the first call wins.

        Every AppWindow calls this, so without the guard opening a second
        window would replace the in-memory dict with the on-disk one and drop
        whatever the first window had accumulated.
        """
        if not is_enabled():
            return
        from utils.app_info_cache import app_info_cache
        with self._lock:
            if self._loaded:
                return
            stored = app_info_cache.get_meta(CACHE_KEY, default_val=None)
            # Copy rather than alias. get_meta hands back the live nested dict,
            # and set_meta only marks the cache changed when the value differs
            # from the one already held -- sharing the object would make every
            # store() a comparison against itself and nothing would persist.
            self._data = (
                {k: dict(v) for k, v in stored.items()} if isinstance(stored, dict) else {}
            )
            self._loaded = True
            logger.info(f"Loaded file metadata cache with {len(self._data)} entries")

    def store(self) -> None:
        """Hand a snapshot to app_info_cache; its own store() flushes to disk."""
        if not is_enabled():
            return
        from utils.app_info_cache import app_info_cache
        with self._lock:
            if not self._loaded:
                # Persisting now would write a partial dict over the full one.
                logger.warning("Skipping file metadata cache store before load")
                return
            if not self._dirty:
                return
            if len(self._data) > self.MAX_ENTRIES:
                keys = list(self._data.keys())
                self._data = {k: self._data[k] for k in keys[-self.MAX_ENTRIES:]}
            snapshot = {k: dict(v) for k, v in self._data.items()}
            self._dirty = False
        app_info_cache.set_meta(CACHE_KEY, snapshot)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get(self, path: str) -> Optional[dict]:
        if not is_enabled():
            return None
        with self._lock:
            entry = self._data.get(path)
            return dict(entry) if entry is not None else None

    def update_stat(self, path: str, ctime: float, mtime: float, size: int) -> None:
        if not is_enabled():
            return
        with self._lock:
            entry = self._data.setdefault(path, {})
            entry["ctime"] = ctime
            entry["mtime"] = mtime
            entry["size"] = size
            self._dirty = True

    def update_dimensions(self, path: str, width: int, height: int) -> None:
        if not is_enabled():
            return
        with self._lock:
            entry = self._data.setdefault(path, {})
            entry["image_width"] = width
            entry["image_height"] = height
            self._dirty = True

    def update_related_image_path(self, path: str, related: str) -> None:
        if not is_enabled():
            return
        with self._lock:
            entry = self._data.setdefault(path, {})
            entry["related_image_path"] = related
            self._dirty = True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop every entry. The next load of a directory re-reads from disk."""
        with self._lock:
            self._data = {}
            self._dirty = True
            self._loaded = True

    def entry_count(self) -> int:
        with self._lock:
            return len(self._data)


file_metadata_cache = FileMetadataCache()

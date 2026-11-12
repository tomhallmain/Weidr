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

Stored in its own file, encrypted like app_info_cache (entries are file paths)
and zlib-compressed by the encryptor; the repeated keys and directory prefixes
compress well. Read and written only by the app's startup/store hooks.
"""

import json
import os
import tempfile
import threading
from typing import Dict, Optional

from utils.config import config
from utils.logging_setup import get_logger
from utils.repo_paths import user_root
from utils.translations import _

logger = get_logger("file_metadata_cache")

CACHE_FILENAME = "file_metadata_cache.enc"
FORMAT_VERSION = 1
# app_info_cache meta key an earlier copy of this cache may still sit under.
CACHE_KEY = "file_metadata_cache"


def is_enabled() -> bool:
    return bool(getattr(config, "enable_file_metadata_cache", False))


def _cache_file_path() -> str:
    override = os.environ.get("WEIDR_CACHE_DIR")
    return os.path.join(override or user_root(), CACHE_FILENAME)


def _write_encrypted(data: bytes, destination: str) -> None:
    """Encrypt to a temp file in the destination's directory, then rename, so
    the destination is always either the old or the new complete file."""
    from utils.constants import AppInfo
    from utils.encryptor import encrypt_data_to_file
    destination_dir = os.path.dirname(destination) or "."
    os.makedirs(destination_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".file_metadata_cache_", suffix=".tmp", dir=destination_dir)
    os.close(fd)
    try:
        encrypt_data_to_file(data, AppInfo.SERVICE_NAME, AppInfo.APP_IDENTIFIER, temp_path)
        os.replace(temp_path, destination)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise


def _read_encrypted(path: str) -> bytes:
    from utils.constants import AppInfo
    from utils.encryptor import decrypt_data_from_file
    return decrypt_data_from_file(path, AppInfo.SERVICE_NAME, AppInfo.APP_IDENTIFIER)


def _drop_copy_in_app_info_cache() -> None:
    """Remove the copy app_info_cache may hold under CACHE_KEY. It is never
    read, and left there it would be rewritten with every app_info_cache store."""
    from utils.app_info_cache import app_info_cache
    app_info_cache.remove_meta(CACHE_KEY)


class FileMetadataCache:
    # Entries are ~180 bytes serialized before compression, and the whole file
    # is rewritten on every store, so this ceiling is a size budget rather than
    # a correctness limit.
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
        """Read the cache file. Idempotent: the first call wins.

        Every AppWindow calls this, so without the guard opening a second
        window would replace the in-memory dict with the on-disk one and drop
        whatever the first window had accumulated. An unreadable file starts
        the cache empty; the next store replaces it.
        """
        _drop_copy_in_app_info_cache()
        if not is_enabled():
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
        path = _cache_file_path()
        if not os.path.exists(path):
            return
        try:
            payload = json.loads(_read_encrypted(path).decode("utf-8"))
        except Exception as e:
            logger.error(_("Failed to load the file metadata cache {0}: {1}").format(path, e))
            return
        if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
            logger.warning(_("Ignoring a file metadata cache with an unknown format: {0}").format(path))
            return
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return
        loaded = {k: dict(v) for k, v in entries.items() if isinstance(k, str) and isinstance(v, dict)}
        with self._lock:
            # Entries recorded before load() finished are newer than the file's.
            for key, entry in loaded.items():
                self._data.setdefault(key, entry)
            count = len(self._data)
        logger.info(f"Loaded file metadata cache with {count} entries")

    def store(self) -> None:
        """Write the cache file if anything changed since the last store/load."""
        if not is_enabled():
            return
        with self._lock:
            if not self._loaded:
                # Writing now would replace the file with this session's entries only.
                logger.warning(_("Skipping the file metadata cache store before load"))
                return
            if not self._dirty:
                return
            if len(self._data) > self.MAX_ENTRIES:
                keys = list(self._data.keys())
                self._data = {k: self._data[k] for k in keys[-self.MAX_ENTRIES:]}
            snapshot = {k: dict(v) for k, v in self._data.items()}
            self._dirty = False
        path = _cache_file_path()
        payload = {"version": FORMAT_VERSION, "entries": snapshot}
        try:
            _write_encrypted(json.dumps(payload, separators=(",", ":")).encode("utf-8"), path)
        except Exception as e:
            # No plaintext fallback: entries hold file paths.
            logger.error(_("Failed to store the file metadata cache {0}: {1}").format(path, e))
            with self._lock:
                self._dirty = True

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

    def reset(self) -> None:
        """Drop in-memory state without marking it for storing (tests)."""
        with self._lock:
            self._data = {}
            self._dirty = False
            self._loaded = False

    def entry_count(self) -> int:
        with self._lock:
            return len(self._data)


file_metadata_cache = FileMetadataCache()

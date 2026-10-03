"""
Persistent classifier prediction scores under the image/audio wrappers'
``predictions_cache``, shared by prevalidations, classifier actions, pipelines
and compare filters across restarts.

A row is valid while the media file and a local model file/directory are
unchanged (a directory's mtime only moves on added/removed entries; hub repo
ids aren't tracked) and its signature matches the score-shaping model
settings. A FrameCache render (first frame, page, sampled frame, SVG/HTML
render) is keyed on its source file plus frame id and validated against the
source, as the render's temp path dies with the session. The file is encrypted
like app_info_cache and is read/written only by the app's startup/store hooks.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Tuple

from lib.file_invalidation_cache import (
    DEFAULT_STALE_ENTRY_MAX_AGE_SECONDS,
    FileKeyedInvalidationCache,
    FileMtimeInvalidationCache,
)
from utils.logging_setup import get_logger
from utils.repo_paths import user_root

logger = get_logger("classifier_prediction_cache")

CACHE_FILENAME = "classifier_prediction_cache.enc"
FORMAT_VERSION = 1


def _cache_file_path() -> str:
    override = os.environ.get("WEIDR_CACHE_DIR")
    return os.path.join(override or user_root(), CACHE_FILENAME)


def model_signature(settings: Dict[str, Any]) -> str:
    """Stable hash of the model settings that shape its scores."""
    payload = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ClassifierPredictionCache:
    # Rows are a few hundred bytes serialized (path, two mtimes, one score per
    # category), and the whole file is rewritten on store, so this is a size
    # budget. Oldest rows are dropped first.
    MAX_ENTRIES = 200_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # model key ("image:<model_name>" / "audio:<model_name>") -> media path key -> row
        self._rows: Dict[str, Dict[str, FileMtimeInvalidationCache[Dict[str, float]]]] = {}
        self._dirty = False
        self._loaded = False

    @staticmethod
    def _tracked_paths(model_location: str, media_path: str) -> tuple:
        # A hub repo id (e.g. audio models' "org/model") is no local path; its
        # row then tracks the media file only, and the signature still
        # carries the id.
        if model_location and os.path.exists(model_location):
            return (media_path, model_location)
        return (media_path,)

    @staticmethod
    def _row_location(media_path: str) -> Optional[Tuple[str, str]]:
        """``(row key, file to validate against)`` for *media_path*, or None if
        it can't be persisted (a render whose source is unknown or itself a
        temp render, e.g. the PDF an HTML page is rendered through)."""
        from utils.media_utils import is_frame_cache_path
        if not is_frame_cache_path(media_path):
            return FileKeyedInvalidationCache._path_key(media_path), media_path
        from image.frame_cache import FrameCache
        located = FrameCache.source_and_frame_id(media_path)
        if located is None or is_frame_cache_path(located[0]):
            return None
        source, frame_id = located
        return f"{FileKeyedInvalidationCache._path_key(source)}#{frame_id}", source

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def get(self, model_key: str, signature: str, model_location: str,
            media_path: str) -> Optional[Dict[str, float]]:
        """Stored scores for *media_path*, or None if absent or invalidated."""
        location = self._row_location(media_path)
        if location is None:
            return None
        row_key, validated_path = location
        with self._lock:
            row = self._rows.get(model_key, {}).get(row_key)
        if row is None or row.signature != signature:
            return None
        ok, scores = row.try_get(self._tracked_paths(model_location, validated_path))
        return dict(scores) if ok and scores is not None else None

    def put(self, model_key: str, signature: str, model_location: str,
            media_path: str, scores: Dict[str, float]) -> None:
        location = self._row_location(media_path)
        if location is None:
            return
        row_key, validated_path = location
        row: FileMtimeInvalidationCache[Dict[str, float]] = FileMtimeInvalidationCache()
        try:
            row.set(self._tracked_paths(model_location, validated_path), dict(scores), signature)
        except OSError as e:
            logger.debug(f"Not caching prediction for {media_path}: {e}")
            return
        with self._lock:
            self._rows.setdefault(model_key, {})[row_key] = row
            self._dirty = True

    def discard(self, model_key: str, media_path: str) -> None:
        location = self._row_location(media_path)
        if location is None:
            return
        with self._lock:
            if self._rows.get(model_key, {}).pop(location[0], None) is not None:
                self._dirty = True

    def clear(self) -> None:
        with self._lock:
            self._rows = {}
            self._dirty = True

    def reset(self) -> None:
        """Drop in-memory state without marking it for storing (tests)."""
        with self._lock:
            self._rows = {}
            self._dirty = False
            self._loaded = False

    def entry_count(self) -> int:
        with self._lock:
            return sum(len(rows) for rows in self._rows.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Read the cache file. Idempotent: the first call wins, so a second
        window's startup doesn't drop rows the first one added."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
        path = _cache_file_path()
        if not os.path.exists(path):
            return
        try:
            from utils.constants import AppInfo
            from utils.encryptor import decrypt_data_from_file
            raw = decrypt_data_from_file(path, AppInfo.SERVICE_NAME, AppInfo.APP_IDENTIFIER)
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to load classifier prediction cache {path}: {e}")
            return
        if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
            logger.warning(f"Ignoring classifier prediction cache with unknown format: {path}")
            return

        now = time.time()
        loaded: Dict[str, Dict[str, FileMtimeInvalidationCache[Dict[str, float]]]] = {}
        count = 0
        for model_key, entries in (payload.get("models") or {}).items():
            if not isinstance(entries, list):
                continue
            rows = loaded.setdefault(model_key, {})
            for item in entries:
                try:
                    path_mtimes = item["path_mtimes"]
                    scores = item["scores"]
                    signature = item["signature"]
                    cached_at = float(item["cached_at_unix"])
                except (KeyError, TypeError, ValueError):
                    continue
                if now - cached_at > DEFAULT_STALE_ENTRY_MAX_AGE_SECONDS:
                    continue
                if not isinstance(path_mtimes, dict) or not isinstance(scores, dict):
                    continue
                media_key = item.get("media_key")
                if not isinstance(media_key, str):
                    continue
                row: FileMtimeInvalidationCache[Dict[str, float]] = FileMtimeInvalidationCache()
                row.load_from_snapshot(path_mtimes, scores, signature, cached_at_unix=cached_at)
                rows[media_key] = row
                count += 1
        with self._lock:
            # Rows added before load() finished are newer than the file's.
            for model_key, rows in loaded.items():
                current = self._rows.setdefault(model_key, {})
                for media_key, row in rows.items():
                    current.setdefault(media_key, row)
        logger.info(f"Loaded classifier prediction cache with {count} entries")

    def store(self) -> None:
        """Write the cache file if anything changed since the last store/load."""
        with self._lock:
            if not self._loaded:
                # Writing now would replace the file with this session's rows only.
                logger.warning("Skipping classifier prediction cache store before load")
                return
            if not self._dirty:
                return
            snapshot = self._snapshot_locked()
            self._dirty = False
        path = _cache_file_path()
        try:
            self._encrypt_atomically(json.dumps(snapshot).encode("utf-8"), path)
            logger.info(f"Stored classifier prediction cache: {path}")
        except Exception as e:
            # No plaintext fallback: rows hold file paths and classification scores.
            logger.error(f"Failed to store classifier prediction cache {path}: {e}")
            with self._lock:
                self._dirty = True

    def _snapshot_locked(self) -> dict:
        now = time.time()
        flat = []
        for model_key, rows in self._rows.items():
            for media_key, row in list(rows.items()):
                snap = row.snapshot_for_persistence()
                cached_at = snap.get("cached_at_unix", 0.0) if snap else 0.0
                if snap is None or now - cached_at > DEFAULT_STALE_ENTRY_MAX_AGE_SECONDS:
                    del rows[media_key]
                    continue
                flat.append((cached_at, model_key, media_key, snap, row.peek_value()))
        if len(flat) > self.MAX_ENTRIES:
            flat.sort(key=lambda t: t[0])
            for _cached_at, model_key, media_key, _snap, _scores in flat[:-self.MAX_ENTRIES]:
                self._rows[model_key].pop(media_key, None)
            flat = flat[-self.MAX_ENTRIES:]
        models: Dict[str, list] = {}
        for cached_at, model_key, media_key, snap, scores in flat:
            models.setdefault(model_key, []).append({
                "media_key": media_key,
                "path_mtimes": snap["path_mtimes"],
                "signature": snap["signature"],
                "cached_at_unix": cached_at,
                "scores": scores,
            })
        return {"version": FORMAT_VERSION, "models": models}

    @staticmethod
    def _encrypt_atomically(data: bytes, destination: str) -> None:
        """Encrypt to a temp file in the destination's directory, then rename,
        so the destination is always either the old or the new complete file."""
        from utils.constants import AppInfo
        from utils.encryptor import encrypt_data_to_file
        destination_dir = os.path.dirname(destination) or "."
        os.makedirs(destination_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".classifier_prediction_cache_", suffix=".tmp", dir=destination_dir)
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


classifier_prediction_cache = ClassifierPredictionCache()

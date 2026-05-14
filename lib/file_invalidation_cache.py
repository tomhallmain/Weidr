"""
Single-slot cache keyed by a set of file paths, a policy signature, and stored mtimes.

Module state:
  - *signature memo*: the current policy fingerprint string (set by the application when
    it computes it). Not consulted on every :meth:`FileKeyedInvalidationCache.try_get`;
    buckets record which signature / policy epoch they were written under.
  - *policy epoch*: incremented when policy changes; :meth:`try_get` compares one integer
    to reject stale buckets without hashing or string work on the hot path.
  - *file buckets*: runtime map ``path_key -> FileKeyedInvalidationCache`` for O(1) lookup.

Serialization policy (JSON-safe, for embedding in ``app_info_cache`` meta)
---------------------------------------------------------------------------
Each :class:`FileKeyedInvalidationCache` exposes :meth:`snapshot_for_persistence` for
one stored row. The application wraps many rows plus a global policy hash in its own
payload (e.g. ``prevalidation_file_invalidation_cache_v2``).

Per-bucket snapshot keys (all JSON-serializable):

``path_mtimes``
    ``dict[str, float]`` — normalized path key (see :meth:`FileKeyedInvalidationCache._path_key`)
    to last-seen mtime/ctime snapshot at :meth:`set` time. Floats are Unix timestamps in
    seconds (same as :func:`os.stat` ``st_mtime`` / ``st_ctime``).

``signature``
    ``str | null`` — policy fingerprint string copied at :meth:`set` time; used for
    debugging and to align with the parent payload’s policy version.

``epoch_at_set``
    ``int`` — :data:`_policy_epoch` value when the bucket was last written; must match
    current epoch for :meth:`try_get` to succeed.

``cached_at_unix``
    ``float`` — wall-clock :func:`time.time` when this cache row was last written
    (:meth:`set` or :meth:`load_from_snapshot`). Used only for stale-row evacuation
    (not for invalidation vs. media files).

Round-trip: :meth:`load_from_snapshot` accepts the same fields; missing ``cached_at_unix``
on legacy data defaults to "now" so old installs do not all expire at once.

Evacuation
----------
:func:`evacuate_stale_file_buckets` removes in-memory buckets whose ``cached_at_unix``
age exceeds *max_age_seconds* (default ~60 days). Empty placeholder buckets are removed.
Call before persisting meta (e.g. from the app’s store hook) to keep on-disk entries
bounded over time.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, Optional, Tuple, TypeVar

T = TypeVar("T")

# ~2 months; evacuation uses wall-clock age of the cache row, not media mtimes.
DEFAULT_STALE_ENTRY_MAX_AGE_SECONDS: float = 60.0 * 24.0 * 3600.0

# Policy fingerprint memo (expensive JSON+hash); owned here, not in the app manager.
_signature_memo: Optional[str] = None
# Bumped when policy invalidates; compared in try_get (single int, no string compare).
_policy_epoch: int = 0
# Normalized path key -> bucket
_file_buckets: Dict[str, "FileKeyedInvalidationCache[Any]"] = {}


def get_signature_memo() -> Optional[str]:
    return _signature_memo


def set_signature_memo(value: str) -> None:
    global _signature_memo
    _signature_memo = value


def invalidate_policy_caches() -> None:
    """Clear signature memo, bump epoch, and drop all file buckets."""
    global _signature_memo, _policy_epoch
    _signature_memo = None
    _policy_epoch += 1
    _file_buckets.clear()


def get_policy_epoch() -> int:
    return _policy_epoch


def get_file_bucket_for_media(media_path: str) -> "FileKeyedInvalidationCache[Any]":
    k = FileKeyedInvalidationCache._path_key(media_path)
    if k not in _file_buckets:
        _file_buckets[k] = FileKeyedInvalidationCache()
    return _file_buckets[k]


def iter_file_buckets() -> Iterator[Tuple[str, "FileKeyedInvalidationCache[Any]"]]:
    return iter(_file_buckets.items())


def estimate_file_buckets_map_overhead_bytes() -> int:
    """Rough ``sys.getsizeof`` for the module-level bucket map (keys are references only)."""
    return sys.getsizeof(_file_buckets)


def install_file_bucket(media_key: str, bucket: "FileKeyedInvalidationCache[Any]") -> None:
    _file_buckets[media_key] = bucket


def evacuate_stale_file_buckets(
    max_age_seconds: float = DEFAULT_STALE_ENTRY_MAX_AGE_SECONDS,
) -> int:
    """
    Drop buckets that are older than *max_age_seconds* (by :attr:`_cached_at_unix`),
    and remove empty placeholder buckets. Returns the number of keys removed.
    """
    now = time.time()
    removed = 0
    for k, b in list(_file_buckets.items()):
        if not b._has_entry:
            del _file_buckets[k]
            removed += 1
            continue
        ts = b._cached_at_unix
        if ts <= 0.0 or (now - ts) > max_age_seconds:
            del _file_buckets[k]
            removed += 1
    return removed


def evacuate_buckets_for_directories(directories: Iterable[str]) -> int:
    """
    Drop buckets whose file path falls under any of the given directories.

    Intended for selective policy invalidation: when only profile-scoped
    prevalidations change, callers can evict just the affected directories
    instead of wiping the entire bucket map.

    Returns the number of keys removed.
    """
    norm_dirs = {
        os.path.normcase(os.path.normpath(os.path.abspath(d)))
        for d in directories
    }
    removed = 0
    for k in list(_file_buckets.keys()):
        parent = os.path.dirname(k)
        if parent in norm_dirs:
            del _file_buckets[k]
            removed += 1
    return removed


class FileKeyedInvalidationCache(Generic[T]):
    """Holds one value for a set of paths; invalidated by epoch, mtime, age, or clear."""

    __slots__ = (
        "_has_entry",
        "_signature",
        "_epoch_at_set",
        "_cached_at_unix",
        "_path_mtimes",
        "_value",
    )

    def __init__(self) -> None:
        self._has_entry: bool = False
        self._signature: Optional[str] = None
        self._epoch_at_set: int = -1
        self._cached_at_unix: float = 0.0
        self._value: Optional[T] = None
        self._path_mtimes: dict[str, float] = {}

    @property
    def signature(self) -> Optional[str]:
        """Policy fingerprint stored when :meth:`set` was last called (for persistence)."""
        return self._signature

    def clear(self) -> None:
        self._has_entry = False
        self._signature = None
        self._epoch_at_set = -1
        self._cached_at_unix = 0.0
        self._value = None
        self._path_mtimes.clear()

    @staticmethod
    def _path_key(p: str) -> str:
        return os.path.normcase(os.path.normpath(os.path.abspath(p)))

    @staticmethod
    def _file_time(p: str) -> float:
        st = os.stat(p)
        return max(st.st_mtime, getattr(st, "st_ctime", st.st_mtime))

    def try_get(self, file_paths: Iterable[str]) -> Tuple[bool, Optional[T]]:
        """
        Return (True, value) if still valid. Uses policy epoch (int) only — no signature
        string comparison on each call.
        """
        if not self._has_entry:
            return False, None
        if self._epoch_at_set != _policy_epoch:
            return False, None
        want = {self._path_key(p) for p in file_paths}
        if want != set(self._path_mtimes.keys()):
            return False, None
        for p in want:
            try:
                cur = self._file_time(p)
            except OSError:
                return False, None
            if cur > self._path_mtimes[p] + 1e-6:
                return False, None
        return True, self._value

    def set(self, file_paths: Iterable[str], value: T, signature: str) -> None:
        """Store *value*, *signature* snapshot, and mtime snapshot for *file_paths*."""
        want = sorted({self._path_key(p) for p in file_paths})
        mtimes: dict[str, float] = {}
        for p in want:
            mtimes[p] = self._file_time(p)
        self._has_entry = True
        self._signature = signature
        self._epoch_at_set = _policy_epoch
        self._cached_at_unix = time.time()
        self._value = value
        self._path_mtimes = mtimes

    def load_from_snapshot(
        self,
        path_mtimes: Dict[str, float],
        value: Optional[T],
        signature: str,
        epoch_at_set: Optional[int] = None,
        cached_at_unix: Optional[float] = None,
    ) -> None:
        """Restore from persisted data (keys may be normalized path strings)."""
        self._has_entry = True
        self._path_mtimes = {self._path_key(p): t for p, t in path_mtimes.items()}
        self._value = value
        self._signature = signature
        self._epoch_at_set = epoch_at_set if epoch_at_set is not None else _policy_epoch
        self._cached_at_unix = (
            float(cached_at_unix) if cached_at_unix is not None else time.time()
        )

    def peek_value(self) -> Optional[T]:
        return self._value if self._has_entry else None

    def has_cached_entry(self) -> bool:
        """True if this bucket currently holds a prevalidation row (value may be ``None``)."""
        return self._has_entry

    def estimated_footprint_bytes(self) -> int:
        """Shallow ``sys.getsizeof`` sum for introspection / UI stats (approximate)."""
        n = sys.getsizeof(self)
        n += sys.getsizeof(self._path_mtimes)
        for pk, pt in self._path_mtimes.items():
            n += sys.getsizeof(pk) + sys.getsizeof(pt)
        sig = self._signature
        if sig is not None:
            n += sys.getsizeof(sig)
        val = self._value
        if val is not None:
            n += sys.getsizeof(val)
        n += sys.getsizeof(self._cached_at_unix)
        n += sys.getsizeof(self._epoch_at_set)
        n += sys.getsizeof(self._has_entry)
        return n

    def snapshot_for_persistence(self) -> Optional[Dict[str, Any]]:
        if not self._has_entry:
            return None
        return {
            "path_mtimes": dict(self._path_mtimes),
            "signature": self._signature,
            "epoch_at_set": self._epoch_at_set,
            "cached_at_unix": self._cached_at_unix,
        }

    def get_or_compute(
        self,
        file_paths: Iterable[str],
        signature: str,
        compute: Callable[[], T],
    ) -> T:
        ok, cached = self.try_get(file_paths)
        if ok:
            return cached  # type: ignore[return-value]
        v = compute()
        self.set(file_paths, v, signature)
        return v

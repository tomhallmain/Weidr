"""Persistent record of batch pipeline runs: the run history, the registry of
runs in progress, and each unfinished run's resumable checkpoint.

Every record holds the directory list the run used, as it was when the run
started, so later warnings and resumes don't depend on the directory profile
still existing or being unchanged.

Files, under WEIDR_CACHE_DIR or user_root(), both encrypted like
app_info_cache and written atomically:
  pipeline_run_history.enc          -- index of all records (small)
  pipeline_run_progress/<id>.enc    -- checkpoint of one run, rewritten as it
                                       advances and deleted once not needed
Nothing is written before load(), so a session that never loaded the file
can't replace it with its own records only.

Several processes (two app instances, a headless MCP server) can share the
files. Each process owns the records of the runs it started; an index write
merges the file's current contents first, taking other processes' records
from disk. A RUNNING record whose process is gone is reconciled to
INTERRUPTED.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List, Optional

from compare.pipeline_run_control import RunCheckpoint, RunProgress
from utils.logging_setup import get_logger
from utils.path_utils import canonical_path, groupings_overlap, same_directory_set
from utils.repo_paths import user_root
from utils.translations import _

logger = get_logger("pipeline_run_history")

HISTORY_FILENAME = "pipeline_run_history.enc"
PROGRESS_DIRNAME = "pipeline_run_progress"
FORMAT_VERSION = 1
MAX_FINISHED_RECORDS = 200
INDEX_HEARTBEAT_SECONDS = 30.0
# Used only when the owning process can't be inspected.
STALE_HEARTBEAT_SECONDS = 120.0
MAX_SUMMARY_CHARS = 4000


def _root() -> str:
    return os.environ.get("WEIDR_CACHE_DIR") or user_root()


def _index_path() -> str:
    return os.path.join(_root(), HISTORY_FILENAME)


def _progress_path(run_id: str) -> str:
    return os.path.join(_root(), PROGRESS_DIRNAME, f"{run_id}.enc")


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    def label(self) -> str:
        return {
            RunStatus.RUNNING: _("Running"),
            RunStatus.COMPLETED: _("Completed"),
            RunStatus.COMPLETED_WITH_ERRORS: _("Completed with errors"),
            RunStatus.CANCELLED: _("Cancelled"),
            RunStatus.FAILED: _("Failed"),
            RunStatus.INTERRUPTED: _("Interrupted"),
        }[self]

    @property
    def is_finished(self) -> bool:
        return self is not RunStatus.RUNNING


RESUMABLE_STATUSES = frozenset({RunStatus.CANCELLED, RunStatus.FAILED, RunStatus.INTERRUPTED})


@dataclass
class RunOwner:
    token: str
    pid: int
    create_time: Optional[float] = None


def _current_process_owner() -> RunOwner:
    create_time = None
    try:
        import psutil
        create_time = psutil.Process(os.getpid()).create_time()
    except Exception:
        pass
    return RunOwner(token=uuid.uuid4().hex, pid=os.getpid(), create_time=create_time)


def _owner_alive(owner: RunOwner, updated_at: float) -> bool:
    try:
        import psutil
        if not psutil.pid_exists(owner.pid):
            return False
        if owner.create_time is None:
            return True
        # A reused pid belongs to a process started at a different time.
        return abs(psutil.Process(owner.pid).create_time() - owner.create_time) <= 1.0
    except Exception:
        return time.time() - updated_at < STALE_HEARTBEAT_SECONDS


@dataclass
class PipelineRunRecord:
    run_id: str
    pipeline_name: str
    pipeline_fingerprint: Optional[str]
    profile_name: Optional[str]
    directories: List[str]
    canonical_directories: List[str]
    source: str  # "gui" | "mcp"
    owner: RunOwner
    started_at: float
    updated_at: float
    finished_at: Optional[float] = None
    status: RunStatus = RunStatus.RUNNING
    stop_requested: bool = False
    directories_done: int = 0
    current_directory: Optional[str] = None
    current_dir_total: int = 0
    current_dir_done: int = 0
    files_evaluated: int = 0
    files_skipped: int = 0
    errors: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    summary: Optional[str] = None
    error: Optional[str] = None
    dump_path: Optional[str] = None
    resumed_from: Optional[str] = None
    resumed_by: Optional[str] = None
    has_checkpoint: bool = False

    @classmethod
    def new(
        cls,
        pipeline_name: str,
        directories: Iterable[str],
        *,
        profile_name: Optional[str],
        source: str,
        owner: RunOwner,
        pipeline_fingerprint: Optional[str] = None,
        resumed_from: Optional[str] = None,
    ) -> "PipelineRunRecord":
        directories = [str(d) for d in directories]
        now = time.time()
        return cls(
            run_id=uuid.uuid4().hex,
            pipeline_name=pipeline_name,
            pipeline_fingerprint=pipeline_fingerprint,
            profile_name=profile_name,
            directories=directories,
            canonical_directories=[canonical_path(d) for d in directories],
            source=source,
            owner=owner,
            started_at=now,
            updated_at=now,
            resumed_from=resumed_from,
        )

    @property
    def is_resumable(self) -> bool:
        return self.status in RESUMABLE_STATUSES and self.has_checkpoint and self.resumed_by is None

    def to_json(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_json(cls, data: dict) -> Optional["PipelineRunRecord"]:
        try:
            owner = data["owner"]
            directories = [str(d) for d in data["directories"]]
            return cls(
                run_id=str(data["run_id"]),
                pipeline_name=str(data["pipeline_name"]),
                pipeline_fingerprint=data.get("pipeline_fingerprint"),
                profile_name=data.get("profile_name"),
                directories=directories,
                canonical_directories=[str(d) for d in data.get("canonical_directories")
                                       or [canonical_path(d) for d in directories]],
                source=str(data.get("source") or "gui"),
                owner=RunOwner(token=str(owner["token"]), pid=int(owner["pid"]),
                               create_time=owner.get("create_time")),
                started_at=float(data["started_at"]),
                updated_at=float(data.get("updated_at") or data["started_at"]),
                finished_at=data.get("finished_at"),
                status=RunStatus(data.get("status", RunStatus.RUNNING.value)),
                stop_requested=bool(data.get("stop_requested", False)),
                directories_done=int(data.get("directories_done") or 0),
                current_directory=data.get("current_directory"),
                current_dir_total=int(data.get("current_dir_total") or 0),
                current_dir_done=int(data.get("current_dir_done") or 0),
                files_evaluated=int(data.get("files_evaluated") or 0),
                files_skipped=int(data.get("files_skipped") or 0),
                errors=int(data.get("errors") or 0),
                action_counts={str(k): int(v) for k, v in (data.get("action_counts") or {}).items()},
                summary=data.get("summary"),
                error=data.get("error"),
                dump_path=data.get("dump_path"),
                resumed_from=data.get("resumed_from"),
                resumed_by=data.get("resumed_by"),
                has_checkpoint=bool(data.get("has_checkpoint", False)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def to_public_dict(self) -> dict:
        """The record as reported over MCP."""
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline_name,
            "profile": self.profile_name,
            "directories": list(self.directories),
            "source": self.source,
            "status": self.status.value,
            "stop_requested": self.stop_requested,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "directories_done": self.directories_done,
            "directories_total": len(self.directories),
            "current_directory": self.current_directory,
            "current_dir_done": self.current_dir_done,
            "current_dir_total": self.current_dir_total,
            "files_evaluated": self.files_evaluated,
            "files_skipped": self.files_skipped,
            "errors": self.errors,
            "action_counts": dict(self.action_counts),
            "summary": self.summary,
            "error": self.error,
            "dump_path": self.dump_path,
            "resumed_from": self.resumed_from,
            "resumed_by": self.resumed_by,
            "resumable": self.is_resumable,
        }


def pipeline_fingerprint(pipeline) -> Optional[str]:
    """Hash of the parts of *pipeline* that decide what a run does, or None
    if it can't be serialized. Name, description and the active flag are left
    out: none of them changes how a file is evaluated."""
    try:
        d = dict(pipeline.to_dict())
        for key in ("name", "description", "is_active"):
            d.pop(key, None)
        payload = json.dumps(d, sort_keys=True, default=str)
    except Exception:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_run_time(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "—"


def _encrypt_atomically(data: bytes, destination: str) -> None:
    from utils.constants import AppInfo
    from utils.encryptor import encrypt_data_to_file
    destination_dir = os.path.dirname(destination) or "."
    os.makedirs(destination_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".pipeline_run_", suffix=".tmp", dir=destination_dir)
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


def _decrypt_json(path: str):
    from utils.constants import AppInfo
    from utils.encryptor import decrypt_data_from_file
    raw = decrypt_data_from_file(path, AppInfo.SERVICE_NAME, AppInfo.APP_IDENTIFIER)
    return json.loads(raw.decode("utf-8"))


def _remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"Could not remove {path}: {e}")


class PipelineRunHistory:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._records: Dict[str, PipelineRunRecord] = {}
        self._checkpoints: Dict[str, RunCheckpoint] = {}
        # Other processes' finished records this process changed (resume link).
        self._local_edits: set = set()
        self._removed: set = set()
        self._loaded = False
        self._dirty = False
        self._owner = _current_process_owner()
        self._last_index_write = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Read the index. Idempotent: the first call wins."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
        disk = self._read_index()
        with self._lock:
            for record in disk.values():
                # Records added before load() are this session's own.
                self._records.setdefault(record.run_id, record)
            # Both mark the history dirty only if they change something.
            self._reconcile_locked()
            self._prune_locked()
        self._remove_orphan_progress_files()
        self.store()

    def store(self) -> None:
        with self._lock:
            if not self._loaded or not self._dirty:
                return
        self._write_index()

    def sync_from_disk(self) -> None:
        """Pick up records other processes have written since the last read."""
        with self._lock:
            if not self._loaded:
                return
        disk = self._read_index()
        with self._lock:
            self._merge_locked(disk)
            self._reconcile_locked()

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._checkpoints.clear()
            self._local_edits.clear()
            self._removed.clear()
            self._loaded = False
            self._dirty = False
            self._last_index_write = 0.0
            self._owner = _current_process_owner()

    @property
    def owner(self) -> RunOwner:
        return self._owner

    def is_own(self, record: PipelineRunRecord) -> bool:
        return record.owner.token == self._owner.token

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_run(self, record: PipelineRunRecord) -> None:
        with self._lock:
            self._records[record.run_id] = record
            self._dirty = True
        self._write_index()

    def update_progress(self, run_id: str, progress: RunProgress) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            record.directories_done = progress.directories_done
            record.current_directory = progress.current_directory
            record.current_dir_total = progress.current_dir_total
            record.current_dir_done = progress.current_dir_done
            record.files_evaluated = progress.files_evaluated
            record.files_skipped = progress.files_skipped
            record.errors = progress.errors
            record.action_counts = dict(progress.action_counts)
            record.updated_at = time.time()
            self._dirty = True
            heartbeat_due = time.monotonic() - self._last_index_write >= INDEX_HEARTBEAT_SECONDS
        if heartbeat_due:
            self._write_index()

    def write_checkpoint(self, run_id: str, checkpoint: RunCheckpoint) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            self._checkpoints[run_id] = checkpoint
            first = not record.has_checkpoint
            record.has_checkpoint = True
            record.updated_at = time.time()
            self._dirty = True
            loaded = self._loaded
        if loaded:
            try:
                _encrypt_atomically(json.dumps(checkpoint.to_json()).encode("utf-8"),
                                    _progress_path(run_id))
            except Exception as e:
                logger.error(f"Failed to write pipeline run checkpoint for {run_id}: {e}")
        if first:
            self._write_index()

    def read_checkpoint(self, run_id: str) -> Optional[RunCheckpoint]:
        with self._lock:
            checkpoint = self._checkpoints.get(run_id)
            loaded = self._loaded
        if checkpoint is not None:
            return checkpoint
        path = _progress_path(run_id)
        if not loaded or not os.path.exists(path):
            return None
        try:
            return RunCheckpoint.from_json(_decrypt_json(path))
        except Exception as e:
            logger.error(f"Failed to read pipeline run checkpoint {path}: {e}")
            return None

    def mark_stop_requested(self, run_id: str) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None or record.status is not RunStatus.RUNNING:
                return
            record.stop_requested = True
            self._dirty = True
        self._write_index()

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        dump_path: Optional[str] = None,
        counters: Optional[RunCheckpoint] = None,
    ) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            now = time.time()
            record.status = status
            record.finished_at = now
            record.updated_at = now
            record.summary = summary[:MAX_SUMMARY_CHARS] if summary else summary
            record.error = error
            record.dump_path = dump_path
            if counters is not None:
                record.files_evaluated = counters.files_evaluated
                record.files_skipped = counters.files_skipped
                record.errors = counters.errors
                record.action_counts = dict(counters.action_counts)
                record.directories_done = len(counters.completed_directories)
            if status in RESUMABLE_STATUSES:
                if counters is not None:
                    self._checkpoints[run_id] = counters
                    record.has_checkpoint = True
            else:
                self._drop_checkpoint_locked(run_id)
                record.has_checkpoint = False
            self._dirty = True
            persist_checkpoint = self._loaded and status in RESUMABLE_STATUSES and counters is not None
        if persist_checkpoint:
            try:
                _encrypt_atomically(json.dumps(counters.to_json()).encode("utf-8"),
                                    _progress_path(run_id))
            except Exception as e:
                logger.error(f"Failed to write final checkpoint for {run_id}: {e}")
        self._write_index()

    def link_resume(self, original_id: str, new_id: str) -> None:
        """Mark *original_id* as continued by *new_id*; its checkpoint now
        lives on in the new run's."""
        with self._lock:
            original = self._records.get(original_id)
            if original is None:
                return
            original.resumed_by = new_id
            original.has_checkpoint = False
            if not self.is_own(original):
                self._local_edits.add(original_id)
            self._drop_checkpoint_locked(original_id)
            self._dirty = True
        self._write_index()

    def remove(self, run_id: str) -> bool:
        with self._lock:
            record = self._records.get(run_id)
            if record is None or record.status is RunStatus.RUNNING:
                return False
            del self._records[run_id]
            self._removed.add(run_id)
            self._drop_checkpoint_locked(run_id)
            self._dirty = True
        self._write_index()
        return True

    # ------------------------------------------------------------------
    # Queries (copies, newest first)
    # ------------------------------------------------------------------
    def records(self) -> List[PipelineRunRecord]:
        with self._lock:
            self._reconcile_locked()
            return [self._copy(r) for r in self._sorted_locked()]

    def get(self, run_id: str) -> Optional[PipelineRunRecord]:
        with self._lock:
            record = self._records.get(run_id)
            return self._copy(record) if record is not None else None

    def running_overlapping(self, directories: Iterable[str]) -> List[PipelineRunRecord]:
        directories = list(directories)
        return [r for r in self.records()
                if r.status is RunStatus.RUNNING and groupings_overlap(r.directories, directories)]

    def last_overlapping(self, pipeline_name: str, directories: Iterable[str]) -> Optional[PipelineRunRecord]:
        """Newest finished run of *pipeline_name* on directories overlapping
        *directories*."""
        directories = list(directories)
        for r in self.records():
            if (r.pipeline_name == pipeline_name and r.status.is_finished
                    and groupings_overlap(r.directories, directories)):
                return r
        return None

    def latest_resumable(
        self, pipeline_name: Optional[str] = None, directories: Optional[Iterable[str]] = None,
    ) -> Optional[PipelineRunRecord]:
        directories = list(directories) if directories is not None else None
        for r in self.records():
            if not r.is_resumable:
                continue
            if pipeline_name is not None and r.pipeline_name != pipeline_name:
                continue
            if directories is not None and not same_directory_set(r.directories, directories):
                continue
            return r
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _copy(record: PipelineRunRecord) -> PipelineRunRecord:
        return PipelineRunRecord.from_json(record.to_json())

    def _sorted_locked(self) -> List[PipelineRunRecord]:
        return sorted(self._records.values(), key=lambda r: r.started_at, reverse=True)

    def _drop_checkpoint_locked(self, run_id: str) -> None:
        self._checkpoints.pop(run_id, None)
        if self._loaded:
            _remove_file(_progress_path(run_id))

    def _reconcile_locked(self) -> None:
        for record in self._records.values():
            if record.status is not RunStatus.RUNNING or self.is_own(record):
                continue
            if _owner_alive(record.owner, record.updated_at):
                continue
            record.status = RunStatus.INTERRUPTED
            record.finished_at = record.updated_at
            record.has_checkpoint = record.has_checkpoint or (
                self._loaded and os.path.exists(_progress_path(record.run_id)))
            self._local_edits.add(record.run_id)
            self._dirty = True

    def _prune_locked(self) -> None:
        finished = [r for r in self._sorted_locked() if r.status.is_finished]
        for record in finished[MAX_FINISHED_RECORDS:]:
            del self._records[record.run_id]
            self._drop_checkpoint_locked(record.run_id)
            self._dirty = True

    def _merge_locked(self, disk: Dict[str, PipelineRunRecord]) -> None:
        # Other processes' records follow the file; one missing there was
        # removed (or pruned) by another process.
        for run_id in [rid for rid, r in self._records.items()
                       if not self.is_own(r) and rid not in disk]:
            del self._records[run_id]
            self._local_edits.discard(run_id)
        for run_id, record in disk.items():
            if run_id in self._removed:
                continue
            if record.owner.token == self._owner.token:
                continue
            current = self._records.get(run_id)
            if current is not None and run_id in self._local_edits:
                # The owner can't still be writing a finished record; keep
                # what this process changed, unless the owner finished it.
                if record.status is not RunStatus.RUNNING:
                    current.status = record.status
                continue
            self._records[run_id] = record

    def _read_index(self) -> Dict[str, PipelineRunRecord]:
        path = _index_path()
        if not os.path.exists(path):
            return {}
        try:
            payload = _decrypt_json(path)
        except Exception as e:
            logger.error(f"Failed to load pipeline run history {path}: {e}")
            self._set_aside_unreadable(path)
            return {}
        if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
            logger.warning(f"Ignoring pipeline run history with unknown format: {path}")
            return {}
        records = {}
        for item in payload.get("runs") or []:
            record = PipelineRunRecord.from_json(item) if isinstance(item, dict) else None
            if record is not None:
                records[record.run_id] = record
        return records

    @staticmethod
    def _set_aside_unreadable(path: str) -> None:
        # Keep the unreadable file once, so the next write doesn't destroy
        # what a key problem (not corruption) may still allow recovering.
        aside = path + ".unreadable"
        if os.path.exists(aside):
            return
        try:
            os.replace(path, aside)
            logger.error(f"Moved unreadable pipeline run history to {aside}")
        except OSError as e:
            logger.error(f"Could not move unreadable pipeline run history {path}: {e}")

    def _write_index(self) -> None:
        with self._lock:
            if not self._loaded:
                return
        with self._io_lock:
            disk = self._read_index()
            with self._lock:
                self._merge_locked(disk)
                self._reconcile_locked()
                self._prune_locked()
                payload = {
                    "version": FORMAT_VERSION,
                    "runs": [r.to_json() for r in self._sorted_locked()],
                }
                self._dirty = False
                self._last_index_write = time.monotonic()
            path = _index_path()
            try:
                _encrypt_atomically(json.dumps(payload).encode("utf-8"), path)
            except Exception as e:
                # No plaintext fallback: records hold directory paths.
                logger.error(f"Failed to store pipeline run history {path}: {e}")
                with self._lock:
                    self._dirty = True

    def _remove_orphan_progress_files(self) -> None:
        progress_dir = os.path.join(_root(), PROGRESS_DIRNAME)
        if not os.path.isdir(progress_dir):
            return
        with self._lock:
            known = set(self._records)
        try:
            names = os.listdir(progress_dir)
        except OSError:
            return
        for name in names:
            run_id, ext = os.path.splitext(name)
            if ext == ".enc" and run_id not in known:
                _remove_file(os.path.join(progress_dir, name))


pipeline_run_history = PipelineRunHistory()

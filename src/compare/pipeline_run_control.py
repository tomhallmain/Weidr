"""Per-run control for a batch pipeline run: cancellation, progress and
resumable checkpoints.

run_pipeline_over_directories() calls the hooks here from its worker thread;
persisting a checkpoint or showing progress is left to the sinks the owner
passes in. A checkpoint records everything a later run needs to continue
where this one stopped: finished directories, the files already processed in
the current directory, the stem-group gate, the cumulative counters, and the
generate/scramble work decided but not yet dispatched.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from utils.path_utils import canonical_path

CHECKPOINT_MIN_FILES = 100
CHECKPOINT_MIN_SECONDS = 5.0
CHECKPOINT_MAX_SECONDS = 15.0

CANCEL_USER = "user"
CANCEL_SHUTDOWN = "shutdown"


@dataclass
class RunCheckpoint:
    completed_directories: List[str] = field(default_factory=list)  # canonical
    current_directory: Optional[str] = None  # canonical
    processed_files: List[str] = field(default_factory=list)  # canonical, current directory only
    processed_stems: Optional[List[str]] = None  # None when the stem-group gate is off
    files_evaluated: int = 0
    errors: int = 0
    files_skipped: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    files_by_directory: Dict[str, int] = field(default_factory=dict)
    pending_generates: List[list] = field(default_factory=list)  # [path, edit_suffix, target_dir]
    pending_scrambles: List[list] = field(default_factory=list)  # [path, modifier]
    updated_at: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "RunCheckpoint":
        stems = data.get("processed_stems")
        return cls(
            completed_directories=[str(d) for d in data.get("completed_directories") or []],
            current_directory=data.get("current_directory"),
            processed_files=[str(p) for p in data.get("processed_files") or []],
            processed_stems=None if stems is None else [str(s) for s in stems],
            files_evaluated=int(data.get("files_evaluated") or 0),
            errors=int(data.get("errors") or 0),
            files_skipped=int(data.get("files_skipped") or 0),
            action_counts={str(k): int(v) for k, v in (data.get("action_counts") or {}).items()},
            files_by_directory={str(k): int(v) for k, v in (data.get("files_by_directory") or {}).items()},
            pending_generates=[list(g) for g in data.get("pending_generates") or []],
            pending_scrambles=[list(s) for s in data.get("pending_scrambles") or []],
            updated_at=float(data.get("updated_at") or 0.0),
        )


@dataclass
class RunProgress:
    directories_total: int
    directories_done: int
    current_directory: Optional[str]
    current_dir_total: int
    current_dir_done: int
    files_evaluated: int
    files_skipped: int
    errors: int
    action_counts: Dict[str, int] = field(default_factory=dict)


class PipelineRunControl:
    """Hooks for one batch run. Cancellation may be requested from any
    thread; every other method is called from the run's worker thread."""

    def __init__(
        self,
        run_id: str,
        directories: Sequence[str],
        *,
        seed: Optional[RunCheckpoint] = None,
        checkpoint_sink: Optional[Callable[[RunCheckpoint], None]] = None,
        progress_sink: Optional[Callable[[RunProgress], None]] = None,
        started_at: Optional[float] = None,
        resumed_from: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self.started_at = started_at
        self.resumed_from = resumed_from
        self._directories_total = len(directories)
        self._seed = seed
        self._checkpoint_sink = checkpoint_sink
        self._progress_sink = progress_sink
        self._cancel = threading.Event()
        self._cancel_reason: Optional[str] = None
        self._lock = threading.Lock()

        self._completed: List[str] = list(seed.completed_directories) if seed else []
        self._completed_set = set(self._completed)
        # Files processed in the seed's current directory; consulted only
        # while that same directory is being walked again.
        self._seed_current = seed.current_directory if seed else None
        self._seed_processed = set(seed.processed_files) if seed else set()
        self._current: Optional[str] = None
        self._processed: List[str] = []
        self._current_dir_total = 0
        self._files_skipped = seed.files_skipped if seed else 0
        self._pending_generates: List[list] = []
        self._pending_scrambles: List[list] = []

        # References to the batch's live counters, set by file_done().
        self._stems = None
        self._evaluated = seed.files_evaluated if seed else 0
        self._errors = seed.errors if seed else 0
        self._actions: Dict[str, int] = dict(seed.action_counts) if seed else {}
        self._files_by_directory: Dict[str, int] = dict(seed.files_by_directory) if seed else {}

        self._files_since_checkpoint = 0
        self._last_checkpoint = time.monotonic()
        self._checkpoint_due = False

    # -- cancellation ----------------------------------------------------
    def request_cancel(self, reason: str = CANCEL_USER) -> None:
        with self._lock:
            if self._cancel_reason is None:
                self._cancel_reason = reason
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def cancel_reason(self) -> Optional[str]:
        with self._lock:
            return self._cancel_reason

    # -- seeds (read once at batch start) ----------------------------------
    def seed_processed_stems(self) -> Optional[set]:
        if self._seed is None or self._seed.processed_stems is None:
            return None
        return set(self._seed.processed_stems)

    def seed_counters(self) -> tuple:
        """``(files_evaluated, errors, action_counts, files_by_directory)``."""
        return self._evaluated, self._errors, dict(self._actions), dict(self._files_by_directory)

    def carried_generates(self) -> List[tuple]:
        return [tuple(g) for g in self._seed.pending_generates] if self._seed else []

    def carried_scrambles(self) -> List[tuple]:
        return [tuple(s) for s in self._seed.pending_scrambles] if self._seed else []

    # -- batch hooks -------------------------------------------------------
    def is_directory_completed(self, directory: str) -> bool:
        return canonical_path(directory) in self._completed_set

    def directory_started(self, directory: str, file_count: int) -> None:
        self._current = canonical_path(directory)
        self._processed = []
        self._current_dir_total = file_count
        self._emit_progress()
        self._write_checkpoint()

    def is_file_processed(self, path: str) -> bool:
        if self._current is None or self._current != self._seed_current:
            return False
        key = canonical_path(path)
        if key in self._seed_processed:
            self._files_skipped += 1
            self._processed.append(key)
            return True
        return False

    def file_done(
        self,
        path: str,
        *,
        processed_stems,
        files_evaluated: int,
        errors: int,
        action_counts: Dict[str, int],
        files_by_directory: Dict[str, int],
    ) -> None:
        self._processed.append(canonical_path(path))
        self._stems = processed_stems
        self._evaluated = files_evaluated
        self._errors = errors
        self._actions = action_counts
        self._files_by_directory = files_by_directory
        self._files_since_checkpoint += 1
        self._emit_progress()
        elapsed = time.monotonic() - self._last_checkpoint
        if (
            self._checkpoint_due
            or (self._files_since_checkpoint >= CHECKPOINT_MIN_FILES and elapsed >= CHECKPOINT_MIN_SECONDS)
            or elapsed >= CHECKPOINT_MAX_SECONDS
        ):
            self._write_checkpoint()

    def directory_finished(self, directory: str) -> None:
        key = canonical_path(directory)
        if key not in self._completed_set:
            self._completed.append(key)
            self._completed_set.add(key)
        self._current = None
        self._processed = []
        self._current_dir_total = 0
        self._emit_progress()
        self._write_checkpoint()

    # Generate/scramble work stays pending until its batch is dispatched. A
    # flush makes the next file_done() checkpoint, so a crash right after a
    # dispatch doesn't replay what was already sent.
    def note_generate(self, path: str, edit_suffix=None, target_dir=None) -> None:
        with self._lock:
            self._pending_generates.append([path, edit_suffix, target_dir])

    def generates_flushed(self) -> None:
        with self._lock:
            self._pending_generates.clear()
        self._checkpoint_due = True

    def note_scramble(self, path: str, modifier=None) -> None:
        with self._lock:
            self._pending_scrambles.append([path, modifier])

    def scrambles_flushed(self) -> None:
        with self._lock:
            self._pending_scrambles.clear()
        self._checkpoint_due = True

    def final_checkpoint(self) -> RunCheckpoint:
        return self._write_checkpoint()

    @property
    def files_skipped(self) -> int:
        return self._files_skipped

    # -- internals ---------------------------------------------------------
    def snapshot(self) -> RunCheckpoint:
        with self._lock:
            pending_generates = [list(g) for g in self._pending_generates]
            pending_scrambles = [list(s) for s in self._pending_scrambles]
        stems = self._stems if self._stems is not None else self.seed_processed_stems()
        return RunCheckpoint(
            completed_directories=list(self._completed),
            current_directory=self._current,
            processed_files=list(self._processed),
            processed_stems=None if stems is None else sorted(stems),
            files_evaluated=self._evaluated,
            errors=self._errors,
            files_skipped=self._files_skipped,
            action_counts=dict(self._actions),
            files_by_directory=dict(self._files_by_directory),
            pending_generates=pending_generates,
            pending_scrambles=pending_scrambles,
            updated_at=time.time(),
        )

    def _write_checkpoint(self) -> RunCheckpoint:
        checkpoint = self.snapshot()
        self._files_since_checkpoint = 0
        self._last_checkpoint = time.monotonic()
        self._checkpoint_due = False
        if self._checkpoint_sink is not None:
            self._checkpoint_sink(checkpoint)
        return checkpoint

    def _emit_progress(self) -> None:
        if self._progress_sink is None:
            return
        self._progress_sink(RunProgress(
            directories_total=self._directories_total,
            directories_done=len(self._completed),
            current_directory=self._current,
            current_dir_total=self._current_dir_total,
            current_dir_done=len(self._processed),
            files_evaluated=self._evaluated,
            files_skipped=self._files_skipped,
            errors=self._errors,
            action_counts=dict(self._actions),
        ))

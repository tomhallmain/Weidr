from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional


@dataclass
class PipelineMessage:
    severity: str   # "INFO", "NOTABLE", "WARNING"
    node: str
    image_path: str
    detail: str
    data: Optional[Any] = None


class PipelineRunReport:
    """Accumulates notable-but-non-fatal events during a single pipeline run.

    Thread-safe: safe to emit from concurrent evaluations if the runner ever
    parallelises condition evaluation in future.
    """

    SEVERITIES = ("INFO", "NOTABLE", "WARNING")

    def __init__(self) -> None:
        self._messages: list[PipelineMessage] = []
        self._lock = Lock()

    def add(
        self,
        severity: str,
        node: str,
        image_path: str,
        detail: str,
        data: Optional[Any] = None,
    ) -> None:
        with self._lock:
            self._messages.append(PipelineMessage(severity, node, image_path, detail, data))

    def messages(self) -> list[PipelineMessage]:
        with self._lock:
            return list(self._messages)

    def has_messages(self) -> bool:
        with self._lock:
            return bool(self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

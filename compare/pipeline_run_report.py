from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

from utils.constants import ClassifierActionType
from utils.translations import _

logger = logging.getLogger(__name__)


@dataclass
class PipelineMessage:
    severity: str   # "INFO", "NOTABLE", "WARNING"
    node: str
    image_path: str
    detail: str
    data: Optional[Any] = None


@dataclass
class PipelineRunStats:
    """Aggregate counters for a single batch pipeline run."""

    pipeline_name: str
    profile_name: Optional[str] = None
    directories: list[str] = field(default_factory=list)
    files_by_directory: dict[str, int] = field(default_factory=dict)
    files_evaluated: int = 0
    errors: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    generates_queued: int = 0
    generation_type_label: Optional[str] = None
    category_map: dict[str, str] = field(default_factory=dict)


class PipelineRunReport:
    """Accumulates notable-but-non-fatal events during a single pipeline run.

    Thread-safe: safe to emit from concurrent evaluations if the runner ever
    parallelises condition evaluation in future.
    """

    SEVERITIES = ("WARNING", "NOTABLE", "INFO")

    def __init__(self) -> None:
        self._messages: list[PipelineMessage] = []
        self._lock = Lock()

    @staticmethod
    def _severity_heading(severity: str) -> str:
        return {
            "WARNING": _("Warnings"),
            "NOTABLE": _("Notable events"),
            "INFO": _("Information"),
        }.get(severity, severity)

    def add(
        self,
        severity: str,
        node: str,
        image_path: str,
        detail: str,
        data: Optional[Any] = None,
    ) -> None:
        if severity not in self.SEVERITIES:
            logger.warning("PipelineRunReport.add: unknown severity %r", severity)
        with self._lock:
            self._messages.append(PipelineMessage(severity, node, image_path, detail, data))

    def messages(self) -> list[PipelineMessage]:
        with self._lock:
            return list(self._messages)

    def messages_by_severity(self, severity: str) -> list[PipelineMessage]:
        with self._lock:
            return [m for m in self._messages if m.severity == severity]

    def has_messages(self) -> bool:
        with self._lock:
            return bool(self._messages)

    def message_count(self) -> int:
        with self._lock:
            return len(self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()

    def format_completion_report(self, stats: PipelineRunStats) -> str:
        """Return a multi-line human-readable end-of-run summary."""
        lines: list[str] = []

        lines.append(
            _("Pipeline {0} — run complete").format(stats.pipeline_name)
        )
        lines.append("─" * 48)

        if stats.profile_name:
            lines.append(_("Profile: {0}").format(stats.profile_name))

        if stats.category_map:
            cats = ", ".join(
                _("{0} → {1}").format(name, suffix)
                for name, suffix in sorted(stats.category_map.items())
            )
            lines.append(_("Categories: {0}").format(cats))

        if stats.directories:
            lines.append(_("Directories scanned:"))
            for directory in stats.directories:
                count = stats.files_by_directory.get(directory, 0)
                lines.append(
                    _("  {0}  ({1} {2})").format(
                        directory,
                        count,
                        _("file(s)"),
                    )
                )

        lines.append("")
        lines.append(
            _("Files evaluated: {0}").format(stats.files_evaluated)
        )
        if stats.errors:
            lines.append(
                _("Errors during evaluation: {0}").format(stats.errors)
            )

        lines.append("")
        lines.append(_("Actions taken:"))
        if stats.action_counts:
            for key, count in sorted(stats.action_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                try:
                    label = ClassifierActionType(key).get_translation()
                except ValueError:
                    label = _(key)
                lines.append(
                    _("  {0}× {1}").format(count, label)
                )
        else:
            lines.append(f"  {_('(none)')}")

        if stats.generates_queued:
            if stats.generation_type_label:
                gen_line = _(
                    "Generations queued: {0} ({1})"
                ).format(stats.generates_queued, stats.generation_type_label)
            else:
                gen_line = _("Generations queued: {0}").format(
                    stats.generates_queued
                )
            lines.append("")
            lines.append(gen_line)

        for severity in self.SEVERITIES:
            section = self._format_message_section(severity, self._severity_heading(severity))
            if section:
                lines.append("")
                lines.extend(section)

        return "\n".join(lines)

    def _format_message_section(
        self, severity: str, heading: str
    ) -> list[str]:
        msgs = self.messages_by_severity(severity)
        if not msgs:
            return []
        lines = [f"── {heading} ({len(msgs)}) ──"]
        for msg in msgs:
            lines.extend(self._format_message_lines(msg))
        return lines

    @staticmethod
    def _format_message_lines(msg: PipelineMessage) -> list[str]:
        image_name = os.path.basename(msg.image_path) if msg.image_path else _("(unknown)")
        lines = [
            f"  [{msg.node}] {image_name}",
            f"    {msg.detail}",
        ]
        data = msg.data
        if not isinstance(data, dict):
            return lines

        matches = data.get("matches")
        if matches:
            shown = matches[:8]
            for path in shown:
                lines.append(f"      · {os.path.basename(path)}")
            remaining = len(matches) - len(shown)
            if remaining > 0:
                lines.append(
                    _("      · … and {0} more").format(remaining)
                )

        unknown_file = data.get("unknown_file")
        if unknown_file and not matches:
            lines.append(f"      · {os.path.basename(unknown_file)}")

        return lines

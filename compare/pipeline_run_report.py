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
    generation_type_value: Optional[str] = None
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
    def _is_seed_category_skip(msg: PipelineMessage) -> bool:
        return isinstance(msg.data, dict) and bool(msg.data.get("seed_category_skip"))

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

    def format_seed_summary(
        self,
        image_path: str,
        action: Optional[Any],
        messages_since: int = 0,
    ) -> str:
        """Return a compact one-or-few-line summary for a single seed image.

        *messages_since* should be the ``message_count()`` captured immediately
        before ``run_pipeline`` was called for this image, so that only messages
        produced during that call are included.
        """
        try:
            action_label = action.get_translation()
        except AttributeError:
            action_label = _("(no action)")
        image_name = os.path.basename(image_path)
        with self._lock:
            recent = self._messages[messages_since:]
        # Seed-category-skip notes fire on every seed image, so surfacing them
        # here would just repeat the same line once per seed file — the
        # completion report's total count already covers it.
        recent = [m for m in recent if not self._is_seed_category_skip(m)]
        if not recent:
            return f"{image_name} → {action_label}"
        lines = [f"{image_name} → {action_label}"]
        for msg in recent:
            lines.append(f"  [{msg.node}] {msg.detail}")
        return "\n".join(lines)

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
        # Group messages with identical (node, detail, data) — e.g. the same
        # stem-uniqueness finding repeated for every file in a stem group.
        groups: dict[tuple, list[PipelineMessage]] = {}
        for msg in msgs:
            key = (msg.node, msg.detail, repr(msg.data))
            groups.setdefault(key, []).append(msg)
        for group in groups.values():
            if self._is_seed_category_skip(group[0]):
                lines.extend(self._format_seed_skip_group_lines(group))
            elif len(group) == 1:
                lines.extend(self._format_message_lines(group[0]))
            else:
                lines.extend(self._format_grouped_message_lines(group))
        return lines

    @staticmethod
    def _format_seed_skip_group_lines(msgs: list[PipelineMessage]) -> list[str]:
        """One clear line with a total count — the skipped files are exactly
        the seed images already implied by pipeline.seed_category, so listing
        each filename would just repeat the seed file list."""
        return [_("  {0} ({1} file(s))").format(msgs[0].detail, len(msgs))]

    @staticmethod
    def _format_grouped_message_lines(msgs: list[PipelineMessage]) -> list[str]:
        node = msgs[0].node
        lines = [_("  [{0}] — {1} files").format(node, len(msgs))]
        for msg in msgs:
            image_name = os.path.basename(msg.image_path) if msg.image_path else _("(unknown)")
            lines.append(f"    · {image_name}")
        lines.append(f"    {msgs[0].detail}")
        lines.extend(PipelineRunReport._format_data_lines(msgs[0].data))
        return lines

    @staticmethod
    def _format_message_lines(msg: PipelineMessage) -> list[str]:
        image_name = os.path.basename(msg.image_path) if msg.image_path else _("(unknown)")
        lines = [
            f"  [{msg.node}] {image_name}",
            f"    {msg.detail}",
        ]
        lines.extend(PipelineRunReport._format_data_lines(msg.data))
        return lines

    @staticmethod
    def _format_data_lines(data: Optional[Any]) -> list[str]:
        if not isinstance(data, dict):
            return []
        lines = []
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

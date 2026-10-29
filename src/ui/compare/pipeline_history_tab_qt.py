"""
PipelineHistoryTab — the batch pipeline run history: every recorded run with
its status, timing, progress and the directories it ran on, plus stopping a
running run and resuming an unfinished one.

The table reads pipeline_run_history in memory once a second while the tab is
visible; worker threads update that state, so no cross-thread signal is
needed. Records written by other processes appear after Refresh or when the
tab is shown again.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from compare import pipeline_run_service
from compare.pipeline_run_history import (
    PipelineRunRecord,
    RunStatus,
    format_run_time,
    pipeline_run_history,
)
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("pipeline_history_tab_qt")

_REFRESH_MS = 1000

(
    _COL_STATUS, _COL_PIPELINE, _COL_PROFILE, _COL_STARTED, _COL_ENDED,
    _COL_PROGRESS, _COL_ERRORS, _COL_SOURCE, _COL_DIRS,
) = range(9)


class PipelineHistoryTab(QWidget):
    """Tab content widget listing pipeline runs, newest first."""

    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(parent)
        self._app_actions = app_actions
        self._records: List[PipelineRunRecord] = []
        self._signature = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        title = QLabel(_("Pipeline Run History"))
        title.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 13pt;"
        )
        root.addWidget(title)

        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels([
            _("Status"), _("Pipeline"), _("Profile"), _("Started"),
            _("Finished / Updated"), _("Progress"), _("Errors"), _("Source"),
            _("Directories"),
        ])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(_COL_DIRS, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._table, 1)

        self._details = QLabel("")
        self._details.setWordWrap(True)
        self._details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._details.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(self._details)

        buttons = QHBoxLayout()
        self._stop_btn = QPushButton(_("Stop"))
        self._stop_btn.clicked.connect(self._stop_selected)
        buttons.addWidget(self._stop_btn)
        self._resume_btn = QPushButton(_("Resume"))
        self._resume_btn.clicked.connect(self._resume_selected)
        buttons.addWidget(self._resume_btn)
        self._resume_last_btn = QPushButton(_("Resume Last Incomplete"))
        self._resume_last_btn.clicked.connect(self._resume_last_incomplete)
        buttons.addWidget(self._resume_last_btn)
        self._remove_btn = QPushButton(_("Remove"))
        self._remove_btn.clicked.connect(self._remove_selected)
        buttons.addWidget(self._remove_btn)
        buttons.addStretch()
        refresh_btn = QPushButton(_("Refresh"))
        refresh_btn.clicked.connect(lambda: self.refresh(from_disk=True))
        buttons.addWidget(refresh_btn)
        root.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()

        self.refresh(from_disk=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh(from_disk=True)

    def _on_timer(self) -> None:
        if self.isVisible():
            self.refresh()

    def refresh(self, from_disk: bool = False) -> None:
        if from_disk:
            pipeline_run_history.sync_from_disk()
        records = pipeline_run_history.records()
        signature = tuple(
            (r.run_id, r.status, r.stop_requested, r.updated_at, r.resumed_by,
             pipeline_run_service.is_active(r.run_id))
            for r in records
        )
        if signature == self._signature:
            return
        selected = self.selected_run_id()
        self._signature = signature
        self._records = records
        self._table.setRowCount(len(records))
        for row, record in enumerate(records):
            self._set_row(row, record)
        self._select_run(selected)
        self._on_selection_changed()

    def _set_row(self, row: int, r: PipelineRunRecord) -> None:
        status = _("Stopping…") if r.status is RunStatus.RUNNING and r.stop_requested else r.status.label()
        if r.status is RunStatus.RUNNING:
            progress = _("Dir {0}/{1} · {2}/{3} files").format(
                min(r.directories_done + 1, len(r.directories)), len(r.directories),
                r.current_dir_done, r.current_dir_total)
            ended = format_run_time(r.updated_at)
        else:
            progress = _("{0} files").format(r.files_evaluated + r.files_skipped)
            ended = format_run_time(r.finished_at)
        values = {
            _COL_STATUS: status,
            _COL_PIPELINE: r.pipeline_name,
            _COL_PROFILE: r.profile_name or "—",
            _COL_STARTED: format_run_time(r.started_at),
            _COL_ENDED: ended,
            _COL_PROGRESS: progress,
            _COL_ERRORS: str(r.errors),
            _COL_SOURCE: self._source_label(r),
            _COL_DIRS: "; ".join(r.directories),
        }
        for col, text in values.items():
            item = QTableWidgetItem(text)
            if col == _COL_STATUS:
                item.setData(Qt.ItemDataRole.UserRole, r.run_id)
            if col == _COL_DIRS:
                item.setToolTip("\n".join(r.directories))
            self._table.setItem(row, col, item)

    @staticmethod
    def _source_label(r: PipelineRunRecord) -> str:
        if not pipeline_run_history.is_own(r):
            return _("Other process")
        return _("MCP") if r.source == "mcp" else _("This app")

    def _details_text(self, r: PipelineRunRecord) -> str:
        lines = [_("Directories at run time:")]
        lines.extend(f"  {d}" for d in r.directories)
        if r.current_directory and r.status is RunStatus.RUNNING:
            lines.append(_("Current directory: {0}").format(r.current_directory))
        if r.files_skipped:
            lines.append(_("Skipped (processed by the earlier run): {0}").format(r.files_skipped))
        if r.action_counts:
            lines.append(_("Actions: {0}").format(
                ", ".join(f"{k}: {v}" for k, v in sorted(r.action_counts.items()))))
        if r.error:
            lines.append(_("Error: {0}").format(r.error))
        if r.resumed_from:
            lines.append(_("Resumes run {0}").format(r.resumed_from))
        if r.resumed_by:
            lines.append(_("Resumed by run {0}").format(r.resumed_by))
        if r.dump_path:
            lines.append(_("Run data: {0}").format(r.dump_path))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def selected_run_id(self) -> Optional[str]:
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows:
            return None
        item = self._table.item(rows[0].row(), _COL_STATUS)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _selected_record(self) -> Optional[PipelineRunRecord]:
        run_id = self.selected_run_id()
        return next((r for r in self._records if r.run_id == run_id), None)

    def _select_run(self, run_id: Optional[str]) -> None:
        if run_id is None:
            return
        for row, r in enumerate(self._records):
            if r.run_id == run_id:
                self._table.selectRow(row)
                return

    def _on_selection_changed(self) -> None:
        record = self._selected_record()
        self._details.setText(self._details_text(record) if record is not None else "")
        self._update_buttons()

    def _update_buttons(self) -> None:
        r = self._selected_record()
        self._stop_btn.setEnabled(
            r is not None and r.status is RunStatus.RUNNING and not r.stop_requested
            and pipeline_run_service.is_active(r.run_id))
        self._resume_btn.setEnabled(r is not None and r.is_resumable)
        self._remove_btn.setEnabled(r is not None and r.status is not RunStatus.RUNNING)
        self._resume_last_btn.setEnabled(any(rec.is_resumable for rec in self._records))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _stop_selected(self) -> None:
        r = self._selected_record()
        if r is None:
            return
        if not qt_alert(
            self, _("Stop Pipeline Run"),
            _("Stop pipeline run '{0}'? Files already processed keep their actions; "
              "the run can be resumed later.").format(r.pipeline_name),
            kind="askyesno",
        ):
            return
        if not pipeline_run_service.request_stop(r.run_id):
            qt_alert(self, _("Stop Pipeline Run"),
                     _("This run is not running in this app."), kind="warning")
        self.refresh()

    def _resume_selected(self) -> None:
        r = self._selected_record()
        if r is not None:
            self._resume(r.run_id)

    def _resume_last_incomplete(self) -> None:
        pipeline_run_history.sync_from_disk()
        r = pipeline_run_history.latest_resumable()
        if r is None:
            qt_alert(self, _("Resume Pipeline Run"), _("There is no unfinished run to resume."))
            return
        self._resume(r.run_id)

    def _resume(self, run_id: str) -> None:
        from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
        ClassifierPipelinesTab.resume_run(self, self._app_actions, run_id)
        self.refresh()

    def _remove_selected(self) -> None:
        r = self._selected_record()
        if r is None:
            return
        if not qt_alert(
            self, _("Remove Run"),
            _("Remove this run of '{0}' from the history?").format(r.pipeline_name),
            kind="askyesno", default_no=True,
        ):
            return
        pipeline_run_history.remove(r.run_id)
        self.refresh()

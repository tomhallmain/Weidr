"""
PySide6 tab widget for managing ClassifierPipelines.

Embedded as the third tab in ClassifierManagementWindow.
Phase 4a: list view with active toggle, delete, move-down, duplicate, and
"Run on Current" buttons.  New/Edit open ClassifierPipelineEditorDialog
(Phase 4b, lazy import).
"""

from __future__ import annotations

import copy
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    PrevalidationPipeline,
)
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("classifier_pipelines_tab_qt")


class ClassifierPipelinesTab(QWidget):
    """
    Tab content widget for managing ClassifierPipelines.

    Can be embedded inside a QTabWidget (ClassifierManagementWindow)
    or used standalone.
    """

    _editor_window = None  # ClassifierPipelineEditorDialog or None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(parent)
        self._app_actions = app_actions

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # -- Title + toolbar --------------------------------------------------
        title_row = QHBoxLayout()

        title_lbl = QLabel(_("Classifier Pipelines"))
        title_lbl.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 13pt;"
        )
        title_row.addWidget(title_lbl)

        new_btn = QPushButton(_("New Pipeline"))
        new_btn.clicked.connect(lambda: self._open_editor())
        title_row.addWidget(new_btn)

        title_row.addStretch()
        root.addLayout(title_row)

        # -- Scrollable list --------------------------------------------------
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {AppStyle.BG_COLOR}; }}"
        )
        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(2)
        self._scroll.setWidget(self._scroll_content)
        root.addWidget(self._scroll, 1)

        ClassifierPipelines.load()
        self._rebuild_rows()

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    # Column indices for the pipeline grid
    _COL_ACTIVE  = 0
    _COL_NAME    = 1
    _COL_TYPE    = 2
    _COL_NODES   = 3
    _COL_PROFILE = 4
    _COL_FLOW    = 5
    _COL_RUN     = 6
    _COL_EDIT    = 7
    _COL_DUP     = 8
    _COL_DEL     = 9
    _COL_DOWN    = 10

    def _rebuild_rows(self) -> None:
        _clear_layout(self._scroll_layout)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(self._COL_NAME, 2)
        grid.setColumnStretch(self._COL_FLOW, 3)

        _FG = AppStyle.FG_COLOR

        # Header row
        for col, text in [
            (self._COL_ACTIVE,  _("Active")),
            (self._COL_NAME,    _("Name")),
            (self._COL_TYPE,    _("Type")),
            (self._COL_NODES,   _("Nodes")),
            (self._COL_PROFILE, _("Profile")),
            (self._COL_FLOW,    _("Flow")),
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {_FG}; font-weight: bold;")
            grid.addWidget(lbl, 0, col)

        for idx, pipeline in enumerate(ClassifierPipelines.get_all_pipelines()):
            r = idx + 1

            active_cb = QCheckBox()
            active_cb.setChecked(pipeline.is_active)
            active_cb.stateChanged.connect(
                lambda state, p=pipeline: self._toggle_active(p, bool(state))
            )
            grid.addWidget(active_cb, r, self._COL_ACTIVE)

            name_lbl = QLabel(pipeline.name)
            name_lbl.setStyleSheet(f"color: {_FG};")
            grid.addWidget(name_lbl, r, self._COL_NAME)

            type_text = (
                _("Prevalidation") if isinstance(pipeline, PrevalidationPipeline)
                else _("General")
            )
            type_lbl = QLabel(type_text)
            type_lbl.setStyleSheet(f"color: {_FG};")
            grid.addWidget(type_lbl, r, self._COL_TYPE)

            nodes_lbl = QLabel(str(len(pipeline.nodes)))
            nodes_lbl.setStyleSheet(f"color: {_FG};")
            nodes_lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(nodes_lbl, r, self._COL_NODES)

            profile_text = getattr(pipeline, "profile_name", None) or "—"
            profile_lbl = QLabel(profile_text)
            profile_lbl.setStyleSheet(f"color: {_FG};")
            grid.addWidget(profile_lbl, r, self._COL_PROFILE)

            flow_lbl = QLabel(pipeline.flow_summary())
            flow_lbl.setStyleSheet(f"color: {_FG};")
            flow_lbl.setWordWrap(True)
            flow_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            grid.addWidget(flow_lbl, r, self._COL_FLOW)

            run_btn = QPushButton(_("Run"))
            run_btn.setToolTip(_("Run on current image"))
            run_btn.clicked.connect(lambda _=False, p=pipeline: self._run_on_current(p))
            grid.addWidget(run_btn, r, self._COL_RUN)

            edit_btn = QPushButton(_("Edit"))
            edit_btn.clicked.connect(lambda _=False, p=pipeline: self._open_editor(p))
            grid.addWidget(edit_btn, r, self._COL_EDIT)

            dup_btn = QPushButton(_("Duplicate"))
            dup_btn.clicked.connect(lambda _=False, p=pipeline: self._duplicate(p))
            grid.addWidget(dup_btn, r, self._COL_DUP)

            del_btn = QPushButton(_("Delete"))
            del_btn.clicked.connect(lambda _=False, p=pipeline: self._delete(p))
            grid.addWidget(del_btn, r, self._COL_DEL)

            down_btn = QPushButton(_("↓"))
            down_btn.setFixedWidth(28)
            down_btn.setToolTip(_("Move down"))
            down_btn.clicked.connect(
                lambda _=False, i=idx, p=pipeline: self._move_down(i, p)
            )
            grid.addWidget(down_btn, r, self._COL_DOWN)

        self._scroll_layout.addLayout(grid)
        self._scroll_layout.addStretch()

    # ------------------------------------------------------------------
    # Public refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        ClassifierPipelines.load()
        self._rebuild_rows()

    # ------------------------------------------------------------------
    # Toolbar / row actions
    # ------------------------------------------------------------------

    def _toggle_active(self, pipeline: ClassifierPipeline, value: bool) -> None:
        pipeline.is_active = value
        ClassifierPipelines.store()

    def _open_editor(self, pipeline: Optional[ClassifierPipeline] = None) -> None:
        try:
            from ui.compare.classifier_pipeline_editor_qt import (
                ClassifierPipelineEditorDialog,
            )
        except ImportError:
            qt_alert(self, _("Not Available"), _("Pipeline editor not yet available."))
            return

        if self._is_editor_valid():
            try:
                ClassifierPipelinesTab._editor_window.close()
            except Exception:
                ClassifierPipelinesTab._editor_window = None

        ClassifierPipelinesTab._editor_window = ClassifierPipelineEditorDialog(
            self.window(),
            self._app_actions,
            self.refresh,
            pipeline,
        )
        ClassifierPipelinesTab._editor_window.show()

    def _duplicate(self, pipeline: ClassifierPipeline) -> None:
        new_pipeline = copy.deepcopy(pipeline)
        base = pipeline.name
        existing_names = {p.name for p in ClassifierPipelines.get_all_pipelines()}
        candidate = base + _(" (copy)")
        counter = 2
        while candidate in existing_names:
            candidate = f"{base} ({_('copy')} {counter})"
            counter += 1
        new_pipeline.name = candidate
        ClassifierPipelines.add_pipeline(new_pipeline)
        ClassifierPipelines.store()
        self._rebuild_rows()

    def _delete(self, pipeline: ClassifierPipeline) -> None:
        ClassifierPipelines.remove_pipeline(pipeline.name)
        ClassifierPipelines.store()
        self._rebuild_rows()

    def _move_down(self, idx: int, pipeline: ClassifierPipeline) -> None:
        pipelines = ClassifierPipelines.get_all_pipelines()
        if idx >= len(pipelines) - 1:
            return
        pipelines[idx], pipelines[idx + 1] = pipelines[idx + 1], pipelines[idx]
        ClassifierPipelines.store()
        self._rebuild_rows()

    def _run_on_current(self, pipeline: ClassifierPipeline) -> None:
        image_path = getattr(self._app_actions, "current_media_path", None)
        if callable(image_path):
            image_path = image_path()
        if not image_path:
            qt_alert(self, _("Run Pipeline"), _("No image is currently open."))
            return

        try:
            from compare.classifier_pipeline_runner import run_pipeline

            def _notify(msg, **_kw):
                try:
                    self._app_actions.title_notify(msg)
                except Exception:
                    pass

            result = run_pipeline(
                pipeline,
                image_path,
                notify_callback=_notify,
            )
            msg = (
                _("Pipeline '{}' result: {}").format(pipeline.name, result)
                if result is not None
                else _("Pipeline '{}': no action taken.").format(pipeline.name)
            )
            qt_alert(self, _("Pipeline Result"), msg)
        except Exception as exc:
            logger.exception("Error running pipeline %r on %s", pipeline.name, image_path)
            qt_alert(self, _("Run Pipeline"), _("Pipeline error: {}").format(exc))

    # ------------------------------------------------------------------
    # Editor window helpers
    # ------------------------------------------------------------------

    @classmethod
    def _is_editor_valid(cls) -> bool:
        win = cls._editor_window
        if win is None:
            return False
        try:
            return win.isVisible()
        except Exception:
            cls._editor_window = None
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())

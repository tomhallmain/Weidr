"""
PySide6 tab widget for managing ClassifierPipelines.

Embedded as the third tab in ClassifierManagementWindow.
Phase 4a: list view with active toggle, delete, move-down, duplicate, and
"Run on Current" buttons.  New/Edit open ClassifierPipelineEditorDialog
(Phase 4b, lazy import).
"""

from __future__ import annotations

import copy
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from compare.classifier_pipeline import (
    ClassifierPipeline,
    ClassifierPipelines,
    PrevalidationPipeline,
)
from files.directory_profile import DirectoryProfile
from lib.qt_alert import qt_alert
from ui.app_style import AppStyle
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("classifier_pipelines_tab_qt")

_PROFILE_CACHE_KEY = "classifier_pipelines_profile"


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

        demo_btn = QPushButton(_("Load Demo"))
        demo_btn.setToolTip(_("Insert the built-in demo pipeline"))
        demo_btn.clicked.connect(self._load_demo)
        title_row.addWidget(demo_btn)

        cat_fill_btn = QPushButton(_("Load Rep. Set Gen."))
        cat_fill_btn.setToolTip(_("Insert the representation set generator demo pipeline"))
        cat_fill_btn.clicked.connect(self._load_category_fill_demo)
        title_row.addWidget(cat_fill_btn)

        title_row.addStretch()
        root.addLayout(title_row)

        # -- Profile selector (for batch run) ---------------------------------
        profile_row = QHBoxLayout()
        profile_lbl = QLabel(_("Run on profile:"))
        profile_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        profile_row.addWidget(profile_lbl)
        self._profile_combo = QComboBox()
        self._refresh_profile_combo()
        saved_profile = app_info_cache.get_meta(_PROFILE_CACHE_KEY, "")
        if saved_profile and self._profile_combo.findText(saved_profile) >= 0:
            self._profile_combo.setCurrentText(saved_profile)
        self._profile_combo.currentTextChanged.connect(
            lambda text: app_info_cache.set_meta(_PROFILE_CACHE_KEY, text)
        )
        profile_row.addWidget(self._profile_combo, 1)
        profile_row.addStretch()
        root.addLayout(profile_row)

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
    _COL_RERUN   = 7
    _COL_EDIT    = 8
    _COL_DUP     = 9
    _COL_DEL     = 10
    _COL_DOWN    = 11

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
            run_btn.setToolTip(_("Run on all files in the selected profile's directories"))
            run_btn.clicked.connect(lambda _=False, p=pipeline: self._run_on_profile(p))
            grid.addWidget(run_btn, r, self._COL_RUN)

            has_dump = ClassifierPipelinesTab._find_latest_dump(pipeline) is not None
            rerun_btn = QPushButton(_("Rerun Last"))
            rerun_btn.setToolTip(
                _("Re-dispatch generates and re-execute scrambles from the most recent run")
            )
            rerun_btn.setEnabled(has_dump)
            rerun_btn.clicked.connect(lambda _=False, p=pipeline: self._rerun_last(p))
            grid.addWidget(rerun_btn, r, self._COL_RERUN)

            edit_btn = QPushButton(_("Edit"))
            edit_btn.clicked.connect(lambda _=False, p=pipeline: self._open_editor(p))
            grid.addWidget(edit_btn, r, self._COL_EDIT)

            dup_btn = QPushButton(_("Copy"))
            dup_btn.clicked.connect(lambda _=False, p=pipeline: self._copy(p))
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
        self._refresh_profile_combo()
        self._rebuild_rows()

    def _refresh_profile_combo(self) -> None:
        current = self._profile_combo.currentText() if hasattr(self, "_profile_combo") else ""
        self._profile_combo.clear()
        names = [p.name for p in DirectoryProfile.directory_profiles]
        self._profile_combo.addItems(names or [_("(no profiles)")])
        if current in names:
            self._profile_combo.setCurrentText(current)
        self._profile_combo.setEnabled(bool(names))

    # ------------------------------------------------------------------
    # Toolbar / row actions
    # ------------------------------------------------------------------

    def _load_demo(self) -> None:
        demo = ClassifierPipelines.build_demo_pipeline()
        self._insert_demo(demo)

    def _load_category_fill_demo(self) -> None:
        demo = ClassifierPipelines.build_category_fill_pipeline()
        self._insert_demo(demo)

    def _insert_demo(self, demo: ClassifierPipeline) -> None:
        existing = {p.name for p in ClassifierPipelines.get_all_pipelines()}
        if demo.name in existing:
            base = demo.name
            counter = 2
            while demo.name in existing:
                demo.name = f"{base} ({counter})"
                counter += 1
        ClassifierPipelines.add_pipeline(demo)
        ClassifierPipelines.store()
        self._rebuild_rows()

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

    def _copy(self, pipeline: ClassifierPipeline) -> None:
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

    def _run_on_profile(self, pipeline: ClassifierPipeline) -> None:
        profile_name = self._profile_combo.currentText().strip()
        profile = DirectoryProfile.get_profile_by_name(profile_name)
        if profile is None:
            qt_alert(self, _("Run Pipeline"), _("No profile selected or profile not found."))
            return

        directories = list(profile.directories)

        details: list[str] = []
        if pipeline.seed_category:
            seed_suffix = pipeline.category_map.get(pipeline.seed_category, "")
            suffix_note = f" ({seed_suffix})" if seed_suffix else ""
            details.append(_("Seed category: {cat}{suffix}").format(
                cat=pipeline.seed_category, suffix=suffix_note))

        _last_profile_key = f"pipeline_last_profile:{pipeline.name}"
        last_profile = app_info_cache.get_meta(_last_profile_key, "")
        if last_profile and last_profile != profile_name:
            details.append(_("Last run on: {profile} (now switching to: {current})").format(
                profile=last_profile, current=profile_name))
        elif last_profile:
            details.append(_("Last run on: {profile}").format(profile=last_profile))

        details_block = ("\n" + "\n".join(details) + "\n") if details else ""
        msg = _("Run pipeline '{name}' on profile '{profile}'?{details}\n\nDirectories:\n{dirs}").format(
            name=pipeline.name,
            profile=profile_name,
            details=details_block,
            dirs="\n".join(f"  {d}" for d in directories),
        )
        if not qt_alert(self, _("Run Pipeline on Profile"), msg, kind="askokcancel"):
            return

        app_info_cache.set_meta(_last_profile_key, profile_name)

        if pipeline.has_generate_action():
            from extensions.sd_runner_client import SDRunnerClient
            if not SDRunnerClient.is_reachable():
                if not qt_alert(
                    self,
                    _("SD Runner Not Available"),
                    _(
                        "This pipeline has GENERATE actions but the SD Runner is not "
                        "reachable.\n\nGenerated images will not be produced unless SD "
                        "Runner is started before the run completes.\n\nContinue anyway?"
                    ),
                    kind="askokcancel",
                ):
                    return

        # Capture the generation type on the main thread before the worker starts.
        # Pipeline-level setting takes priority; fall back to the application's global mode.
        if pipeline.generation_type is not None:
            generation_type = pipeline.generation_type
        else:
            from ui.image.media_details import MediaDetails
            generation_type = MediaDetails.get_image_specific_generation_mode()

        # 0 means "no intermediate flush" (old single-batch-at-end behaviour).
        _gen_batch_cfg = config.pipeline_generate_batch_size
        generate_batch_size: int | None = _gen_batch_cfg if _gen_batch_cfg > 0 else None

        # all_generates: full audit record written to the dump and used by rerun-last.
        # Intermediate-flush state is encapsulated in _make_generate_batch_state().
        all_generates, _on_generate, _dispatch_generate_batch = (
            ClassifierPipelinesTab._make_generate_batch_state(generation_type, generate_batch_size)
        )
        _scr_batch_cfg = config.pipeline_scramble_batch_size
        scramble_batch_size: int | None = _scr_batch_cfg if _scr_batch_cfg > 0 else None
        all_scrambles, _on_scramble, _execute_scramble_batch = (
            ClassifierPipelinesTab._make_scramble_batch_state(scramble_batch_size)
        )

        from compare.action_callbacks import ActionCallbacks
        from files.marked_files import MarkedFiles

        callbacks = ActionCallbacks(
            hide_callback=self._app_actions.hide_current_media,
            notify_callback=self._app_actions.title_notify,
            add_mark_callback=MarkedFiles.add_mark_if_not_present,
            blur_callback=self._app_actions.request_media_blur,
            generate_callback=_on_generate,
            scramble_callback=_on_scramble,
        )

        def _worker():
            from compare.base_compare import gather_files
            from compare.pipeline_run_report import PipelineRunReport, PipelineRunStats
            from compare.classifier_pipeline_runner import run_pipeline
            from files.related_image import clear_base_stem_dir_cache, clear_generate_gate_cache, extract_filename_base_stem
            from utils.constants import ClassifierActionType

            clear_base_stem_dir_cache()
            clear_generate_gate_cache()
            report = PipelineRunReport()
            total = 0
            errors = 0
            actions: dict[str, int] = {}
            files_by_directory: dict[str, int] = {}

            for directory in directories:
                files = list(gather_files(directory))
                files_by_directory[directory] = len(files)
                logger.info(
                    "Pipeline %r: scanning %s — %d file(s)", pipeline.name, directory, len(files)
                )
                for image_path in files:
                    try:
                        msg_snapshot = report.message_count()
                        result = run_pipeline(
                            pipeline, image_path, callbacks,
                            base_directory=directory, report=report,
                        )
                        total += 1
                        key = result.value if isinstance(result, ClassifierActionType) else "(no action)"
                        actions[key] = actions.get(key, 0) + 1
                        if not config.debug:
                            base_stem = extract_filename_base_stem(image_path)
                            file_stem = os.path.splitext(os.path.basename(image_path))[0]
                            if base_stem and file_stem.lower() == base_stem.lower():
                                logger.info(
                                    "Pipeline %r: %s",
                                    pipeline.name,
                                    report.format_seed_summary(image_path, result, msg_snapshot),
                                )
                    except Exception:
                        errors += 1
                        logger.exception("Pipeline run error on %s", image_path)

            gen_label = (
                generation_type.get_text()
                if generation_type is not None
                else None
            )
            stats = PipelineRunStats(
                pipeline_name=pipeline.name,
                profile_name=profile_name,
                directories=directories,
                files_by_directory=files_by_directory,
                files_evaluated=total,
                errors=errors,
                action_counts=actions,
                generates_queued=len(all_generates),
                generation_type_label=gen_label,
                generation_type_value=generation_type.value if generation_type is not None else None,
                category_map=dict(pipeline.category_map or {}),
            )
            summary = report.format_completion_report(stats)
            logger.info("\n%s", summary)
            _dispatch_generate_batch()   # flush generate remainder
            _execute_scramble_batch()    # flush scramble remainder
            self._write_pipeline_run_dump(pipeline, stats, report, all_generates, all_scrambles)
            try:
                self._app_actions.title_notify(summary)
            except Exception:
                pass

        from utils.running_tasks_registry import start_thread
        start_thread(_worker, use_asyncio=False)

    # ------------------------------------------------------------------
    # Scramble helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_one_scramble(
        path: str, modifier: str | None, skip_existing: bool = False
    ) -> None:
        from image.image_ops import ImageOps
        out = ImageOps.new_filepath(path, append_part=modifier) if modifier else None
        if skip_existing and out and os.path.exists(out):
            return
        if modifier and "semi" in modifier:
            ImageOps.semi_scramble_image(path, output_path=out)
        else:
            ImageOps.scramble_image(path, output_path=out)

    @staticmethod
    def _make_scramble_batch_state(
        scramble_batch_size: int | None,
    ) -> tuple[list, object, object]:
        """Build the scramble-execution state for a pipeline run.

        Returns (all_scrambles, on_scramble, execute_batch) where:
          all_scrambles   – grows with every call to on_scramble; written to the dump.
          on_scramble     – use as ActionCallbacks.scramble_callback.
          execute_batch   – call at end-of-run to flush any remainder; also called
                            automatically at each intermediate BATCH_SIZE threshold.

        scramble_batch_size=None means no intermediate flushes; one flush at end-of-run.
        Setting pipeline_scramble_batch_size=0 in config produces None (inline-like behaviour).
        """
        all_scrambles: list[tuple[str, str | None]] = []
        flush_scrambles: list[tuple[str, str | None]] = []

        def execute_batch() -> None:
            if not flush_scrambles:
                return
            batch = list(flush_scrambles)
            flush_scrambles.clear()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(ClassifierPipelinesTab._run_one_scramble, path, modifier)
                    for path, modifier in batch
                ]
                for f in futures:
                    try:
                        f.result()
                    except Exception:
                        logger.exception("Scramble batch item failed")

        def on_scramble(path: str, modifier: str | None = None) -> None:
            all_scrambles.append((path, modifier))
            flush_scrambles.append((path, modifier))
            if scramble_batch_size is not None and len(flush_scrambles) >= scramble_batch_size:
                execute_batch()

        return all_scrambles, on_scramble, execute_batch

    # ------------------------------------------------------------------
    # Generate batch dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _make_generate_batch_state(
        generation_type,
        generate_batch_size: int | None,
    ) -> tuple[list, object, object]:
        """Build the generate-dispatch state for a pipeline run.

        Returns (all_generates, on_generate, dispatch_batch) where:
          all_generates   – grows with every call to on_generate; written to the dump.
          on_generate     – use as ActionCallbacks.generate_callback.
          dispatch_batch  – call at the end-of-run to flush any remainder; also
                            called automatically at each intermediate BATCH_SIZE threshold.

        generate_batch_size=None disables intermediate flushes; one dispatch happens
        at end-of-run.  Setting pipeline_generate_batch_size=0 in config produces None.
        """
        all_generates: list[tuple[str, str | None]] = []
        flush_generates: list[tuple[str, str | None]] = []

        def dispatch_batch() -> None:
            if not flush_generates:
                return
            batch = list(flush_generates)
            flush_generates.clear()
            batch_args = [
                {
                    'image': path,
                    'append': False,
                    **({'edit_suffix': suffix} if suffix else {}),
                }
                for path, suffix in batch
            ]
            try:
                from extensions.sd_runner_client import SDRunnerClient
                SDRunnerClient().run_batch(generation_type, batch_args)
            except Exception:
                logger.exception(
                    "Intermediate generate batch failed; items in all_generates for rerun"
                )

        def on_generate(path: str, edit_suffix: str | None = None) -> None:
            all_generates.append((path, edit_suffix))
            flush_generates.append((path, edit_suffix))
            if generate_batch_size is not None and len(flush_generates) >= generate_batch_size:
                dispatch_batch()

        return all_generates, on_generate, dispatch_batch

    # ------------------------------------------------------------------
    # Rerun last
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_dump(pipeline: ClassifierPipeline):
        """Return the most recent dump Path for *pipeline*, or None if absent."""
        try:
            from utils.logging_setup import get_log_dir
            safe_name = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in pipeline.name
            )
            return max(get_log_dir().glob(f"pipeline_run_*_{safe_name}.json"), default=None)
        except Exception:
            return None

    def _rerun_last(self, pipeline: ClassifierPipeline) -> None:
        dump_path = ClassifierPipelinesTab._find_latest_dump(pipeline)
        if dump_path is None:
            qt_alert(
                self, _("Rerun Last"),
                _("No previous run dump found for '{name}'.").format(name=pipeline.name),
            )
            return

        try:
            import json
            dump = json.loads(dump_path.read_text(encoding="utf-8"))
        except Exception as e:
            qt_alert(self, _("Rerun Last"),
                     _("Could not load dump file:\n{err}").format(err=e))
            return

        generates = dump.get("generates", [])
        scrambles = dump.get("scrambles", [])
        ts_raw = dump.get("timestamp", "")
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = ts_raw

        msg = _(
            "Rerun last run of '{name}'?\n\n"
            "Run date: {ts}\n"
            "Generates to re-dispatch: {n_gen}\n"
            "Scrambles to re-execute:  {n_scr}"
        ).format(name=pipeline.name, ts=ts, n_gen=len(generates), n_scr=len(scrambles))
        if not qt_alert(self, _("Rerun Last"), msg, kind="askokcancel"):
            return

        generation_type_value = dump.get("stats", {}).get("generation_type_value")

        def _worker():
            gen_count = 0
            scr_count = 0

            if generates:
                try:
                    from extensions.sd_runner_client import SDRunnerClient
                    from utils.constants import ImageGenerationType
                    try:
                        gen_type = ImageGenerationType(generation_type_value)
                    except (ValueError, TypeError):
                        from ui.image.media_details import MediaDetails
                        gen_type = MediaDetails.get_image_specific_generation_mode()
                    batch_args = [
                        {
                            "image": g["path"],
                            "append": False,
                            **({'edit_suffix': g["modifier"]} if g.get("modifier") else {}),
                        }
                        for g in generates
                    ]
                    SDRunnerClient().run_batch(gen_type, batch_args)
                    gen_count = len(generates)
                except Exception:
                    logger.exception("Rerun: generate batch dispatch failed")

            if scrambles:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [
                        pool.submit(
                            ClassifierPipelinesTab._run_one_scramble,
                            s["path"], s.get("modifier"), True,
                        )
                        for s in scrambles
                    ]
                    for f in futures:
                        try:
                            f.result()
                            scr_count += 1
                        except Exception:
                            logger.exception("Rerun: scramble item failed")

            summary = _(
                "Rerun complete for '{name}'.\n"
                "Generates dispatched: {n_gen}  Scrambles executed: {n_scr}"
            ).format(name=pipeline.name, n_gen=gen_count, n_scr=scr_count)
            logger.info(summary)
            try:
                self._app_actions.title_notify(summary)
            except Exception:
                pass

        from utils.running_tasks_registry import start_thread
        start_thread(_worker, use_asyncio=False)

    # ------------------------------------------------------------------
    # Pipeline run dump
    # ------------------------------------------------------------------

    @staticmethod
    def _write_pipeline_run_dump(
        pipeline, stats, report,
        all_generates=(), all_scrambles=(),
    ) -> None:
        try:
            import json
            from datetime import datetime
            from utils.logging_setup import get_log_dir
            dump = {
                "timestamp": datetime.now().isoformat(),
                "pipeline": pipeline.to_dict(),
                "stats": {
                    "pipeline_name": stats.pipeline_name,
                    "profile_name": stats.profile_name,
                    "directories": stats.directories,
                    "files_by_directory": stats.files_by_directory,
                    "files_evaluated": stats.files_evaluated,
                    "errors": stats.errors,
                    "action_counts": stats.action_counts,
                    "generates_queued": stats.generates_queued,
                    "generation_type_label": stats.generation_type_label,
                    "generation_type_value": stats.generation_type_value,
                    "category_map": stats.category_map,
                },
                "generates": [
                    {"path": path, "modifier": modifier}
                    for path, modifier in all_generates
                ],
                "scrambles": [
                    {"path": path, "modifier": modifier}
                    for path, modifier in all_scrambles
                ],
                "messages": [
                    {
                        "severity": m.severity,
                        "node": m.node,
                        "image_path": m.image_path,
                        "detail": m.detail,
                        "data": m.data,
                    }
                    for m in report.messages()
                ],
            }
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in pipeline.name)
            dump_path = get_log_dir() / f"pipeline_run_{ts}_{safe_name}.json"
            dump_path.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
            logger.info("Pipeline run data written to %s", dump_path)
        except Exception:
            logger.exception("Failed to write pipeline run dump")

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

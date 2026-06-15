"""
Seek-to-Trigger tab for ClassifierManagementWindow.

Given the currently displayed dynamic media file (video, GIF, or PDF) and a
ClassifierAction (or Prevalidation), scans sampled frames using the action's own
settings, finds the first frame that satisfies the action's full threshold
condition, and seeks the media player to that position.

The scan runs in a background QThread so the UI stays responsive.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from compare.classifier_action import ClassifierAction, Prevalidation, TriggerDetail, TriggerFrameResult
from compare.classifier_actions_manager import ClassifierActionsManager
from ui.app_style import AppStyle
from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger
from utils.media_utils import is_classifier_dynamic_media_path
from utils.translations import _

logger = get_logger("seek_to_trigger_tab_qt")

_NBSP = chr(0x00A0)  # non-breaking space used as a height-preserving placeholder
_LAST_ACTION_CACHE_KEY = "seek_to_trigger_last_action"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class SeekToTriggerWorker(QThread):
    """Background thread that runs ClassifierAction.find_first_trigger_slot()."""

    found = Signal(object)   # TriggerFrameResult
    not_found = Signal(int)  # planned_slots (so the UI can report how many were scanned)
    error = Signal(str)

    def __init__(
        self,
        classifier_action: ClassifierAction,
        media_path: str,
        start_slot: int = 0,
        sample_ratio: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._action = classifier_action
        self._media_path = media_path
        self._start_slot = start_slot
        self._sample_ratio = sample_ratio

    def run(self) -> None:
        try:
            result = self._action.find_first_trigger_slot(
                self._media_path, start_slot=self._start_slot,
                sample_ratio=self._sample_ratio,
            )
            if result is None and self._start_slot > 0:
                # Wrap around: retry from the beginning so a single-trigger
                # video loops back to the same frame on repeated clicks.
                result = self._action.find_first_trigger_slot(
                    self._media_path, start_slot=0, sample_ratio=self._sample_ratio,
                )
            if result is not None:
                self.found.emit(result)
            else:
                from image.frame_cache import FrameCache
                stats = FrameCache.get_dynamic_media_stats(self._media_path)
                planned = getattr(stats, "total_items", 0) or 0
                self.not_found.emit(planned)
        except Exception as exc:
            logger.exception("SeekToTriggerWorker failed for %s", self._media_path)
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class SeekToTriggerTab(QWidget):
    """
    Tab content for "Seek to Trigger".

    Two side-by-side list panels — Classifier Actions (left) and Prevalidations
    (right).  Select a row and click "Seek to trigger" to scan the current
    dynamic media for that action's first trigger point.
    """

    _DEFAULT_SAMPLE_PCT = 30
    _WARN_SAMPLE_PCT    = 75

    # Class-level state shared with the headless keybind path in WindowLauncher.
    _last_action: Optional["ClassifierAction"] = None
    _last_trigger_slot: dict = {}  # (action_id, media_path) → last slot_index

    def __init__(self, parent: QWidget, app_actions) -> None:
        super().__init__(parent)
        self._app_actions = app_actions
        self._worker: Optional[SeekToTriggerWorker] = None
        self._active_media_path: Optional[str] = None
        self._ca_items: list = []   # ClassifierAction objects matching _actions_list rows
        self._pv_items: list = []   # ClassifierAction objects matching _prevals_list rows

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # -- Top button row ----------------------------------------------
        btn_row = QHBoxLayout()

        copy_path_btn = QPushButton(_("Copy path"))
        copy_path_btn.setFixedWidth(100)
        copy_path_btn.clicked.connect(self._copy_path)
        btn_row.addWidget(copy_path_btn)

        refresh_btn = QPushButton(_("Refresh"))
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(refresh_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # -- Sample density row ------------------------------------------
        density_row = QHBoxLayout()

        self._density_check = QCheckBox(_("Custom sample density:"))
        self._density_check.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._density_check.setChecked(False)
        self._density_check.setToolTip(
            _("When unchecked, the action's own sample ratio is used (default).\n"
              "Enable to override with a custom density for this seek scan.")
        )
        self._density_check.toggled.connect(self._on_density_toggled)
        density_row.addWidget(self._density_check)

        self._density_slider = QSlider(Qt.Orientation.Horizontal)
        self._density_slider.setMinimum(5)
        self._density_slider.setMaximum(100)
        self._density_slider.setSingleStep(5)
        self._density_slider.setPageStep(10)
        self._density_slider.setValue(self._DEFAULT_SAMPLE_PCT)
        self._density_slider.setEnabled(False)
        self._density_slider.setToolTip(
            _("Controls what fraction of frames are scanned per seek.\n"
              "Higher values find trigger points more precisely but take longer.\n"
              "Above ~75 % gives diminishing returns and may be very slow for long videos.\n"
              "100 % tests every single frame.")
        )
        self._density_slider.valueChanged.connect(self._on_density_changed)
        density_row.addWidget(self._density_slider, 1)

        self._density_lbl = QLabel(f"{self._DEFAULT_SAMPLE_PCT} %")
        self._density_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._density_lbl.setEnabled(False)
        self._density_lbl.setFixedWidth(40)
        density_row.addWidget(self._density_lbl)

        root.addLayout(density_row)

        self._density_warn_lbl = QLabel(
            _("Tip: values above 75 % give diminishing returns and may be very slow for long videos.")
        )
        self._density_warn_lbl.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-style: italic; font-size: 8pt;"
        )
        self._density_warn_lbl.setWordWrap(True)
        self._density_warn_lbl.setVisible(False)
        root.addWidget(self._density_warn_lbl)

        # -- Last action label -------------------------------------------
        # Persistently shows which action was most recently used for a seek
        # (updated both here and by the headless Ctrl+L keybind path).
        self._last_action_lbl = QLabel(_NBSP)
        self._last_action_lbl.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-size: 9pt; font-style: italic;"
        )
        self._last_action_lbl.setWordWrap(True)
        root.addWidget(self._last_action_lbl)

        # -- Status labels -----------------------------------------------
        # Both labels are always populated (with _NBSP when empty) so the
        # layout height stays constant and elements below never jump.
        self._status_lbl = QLabel(_NBSP)
        self._status_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        self._detail_lbl = QLabel(_NBSP)
        self._detail_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-size: 9pt;")
        self._detail_lbl.setWordWrap(True)
        root.addWidget(self._detail_lbl)

        # -- Two-column action lists -------------------------------------
        list_style = (
            f"QListWidget {{ background: {AppStyle.BG_COLOR}; color: {AppStyle.FG_COLOR};"
            f" border: 1px solid gray; }}"
            f"QListWidget::item:selected {{ background: {AppStyle.FG_COLOR};"
            f" color: {AppStyle.BG_COLOR}; }}"
        )

        columns = QHBoxLayout()
        columns.setSpacing(10)

        # Left: Classifier Actions
        left = QVBoxLayout()
        left_hdr = QLabel(_("Classifier Actions"))
        left_hdr.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-weight: bold;")
        left.addWidget(left_hdr)
        self._actions_list = QListWidget()
        self._actions_list.setStyleSheet(list_style)
        left.addWidget(self._actions_list, 1)
        self._actions_seek_btn = QPushButton(_("Seek to trigger"))
        self._actions_seek_btn.setEnabled(False)
        self._actions_seek_btn.clicked.connect(self._seek_selected_action)
        left.addWidget(self._actions_seek_btn)
        columns.addLayout(left)

        # Right: Prevalidations
        right = QVBoxLayout()
        right_hdr = QLabel(_("Prevalidations"))
        right_hdr.setStyleSheet(f"color: {AppStyle.FG_COLOR}; font-weight: bold;")
        right.addWidget(right_hdr)
        self._prevals_list = QListWidget()
        self._prevals_list.setStyleSheet(list_style)
        right.addWidget(self._prevals_list, 1)
        self._prevals_seek_btn = QPushButton(_("Seek to trigger"))
        self._prevals_seek_btn.setEnabled(False)
        self._prevals_seek_btn.clicked.connect(self._seek_selected_preval)
        right.addWidget(self._prevals_seek_btn)
        columns.addLayout(right)

        root.addLayout(columns, 1)

        self.refresh()
        self._update_last_action_lbl()

    # ------------------------------------------------------------------
    # Refresh / rebuild
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Update which media is active and rebuild the action lists."""
        path: Optional[str] = self._app_actions.get_active_media_filepath()
        self._active_media_path = path
        self._rebuild_rows(bool(path and is_classifier_dynamic_media_path(path)))

    def _rebuild_rows(self, is_dynamic: bool) -> None:
        worker_running = self._worker is not None and self._worker.isRunning()
        btn_enabled = is_dynamic and not worker_running

        # Preserve (or default to 0) selection across rebuilds.
        ca_row = max(0, self._actions_list.currentRow())
        pv_row = max(0, self._prevals_list.currentRow())

        self._ca_items = list(ClassifierActionsManager.classifier_actions)
        self._actions_list.clear()
        for ca in self._ca_items:
            self._actions_list.addItem(ca.name)
        if self._ca_items:
            self._actions_list.setCurrentRow(min(ca_row, len(self._ca_items) - 1))
        self._actions_seek_btn.setEnabled(btn_enabled and bool(self._ca_items))

        self._pv_items = list(ClassifierActionsManager.prevalidations)
        self._prevals_list.clear()
        for pv in self._pv_items:
            self._prevals_list.addItem(pv.name)
        if self._pv_items:
            self._prevals_list.setCurrentRow(min(pv_row, len(self._pv_items) - 1))
        self._prevals_seek_btn.setEnabled(btn_enabled and bool(self._pv_items))

    # ------------------------------------------------------------------
    # Seek entry points (one per panel)
    # ------------------------------------------------------------------

    def _seek_selected_action(self) -> None:
        idx = self._actions_list.currentRow()
        if idx < 0:
            idx = 0
        if idx < len(self._ca_items):
            self._seek_to_trigger(self._ca_items[idx])

    def _seek_selected_preval(self) -> None:
        idx = self._prevals_list.currentRow()
        if idx < 0:
            idx = 0
        if idx < len(self._pv_items):
            self._seek_to_trigger(self._pv_items[idx])

    # ------------------------------------------------------------------
    # Seek logic
    # ------------------------------------------------------------------

    def _seek_to_trigger(self, classifier_action: ClassifierAction) -> None:
        media_path = self._app_actions.get_active_media_filepath()
        self._active_media_path = media_path
        if not media_path or not is_classifier_dynamic_media_path(media_path):
            self._set_status(_("No dynamic media is currently loaded."))
            return

        if self._worker is not None and self._worker.isRunning():
            self._set_status(_("A scan is already in progress."))
            return

        SeekToTriggerTab._last_action = classifier_action
        app_info_cache.set_meta(_LAST_ACTION_CACHE_KEY, classifier_action.name)
        self._update_last_action_lbl()

        nav_key = (id(classifier_action), media_path)
        last_slot = SeekToTriggerTab._last_trigger_slot.get(nav_key)
        start_slot = (last_slot + 1) if last_slot is not None else 0

        self._set_status(
            _("Scanning '{name}' on {file}…").format(
                name=classifier_action.name,
                file=os.path.basename(media_path),
            )
        )
        self._set_detail(_NBSP)
        self._rebuild_rows(is_dynamic=True)

        sample_ratio = (
            self._density_slider.value() / 100.0
            if self._density_check.isChecked()
            else None
        )
        self._worker = SeekToTriggerWorker(
            classifier_action, media_path,
            start_slot=start_slot, sample_ratio=sample_ratio, parent=self,
        )
        self._worker.found.connect(
            lambda r: self._on_found(r, media_path, classifier_action.name, nav_key)
        )
        self._worker.not_found.connect(
            lambda n: self._on_not_found(n, classifier_action.name)
        )
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        path = self._app_actions.get_active_media_filepath()
        self._rebuild_rows(is_dynamic=bool(path and is_classifier_dynamic_media_path(path)))

    def _on_found(
        self, result: TriggerFrameResult, media_path: str, action_name: str, nav_key: tuple
    ) -> None:
        SeekToTriggerTab._last_trigger_slot[nav_key] = result.slot_index

        from image.frame_cache import FrameCache
        pos = FrameCache.slot_index_to_seek_position(
            media_path, result.slot_index, result.total_planned_slots
        )
        if pos is None:
            msg = _(
                "Trigger found (sample {slot}/{total}) but seek position could not be computed."
            ).format(slot=result.slot_index + 1, total=result.total_planned_slots)
            self._set_status(msg)
            self._set_detail(_NBSP)
            self._app_actions.toast(msg)
            return

        if pos.kind == "ms":
            self._app_actions.seek_media(pos.value, pause_after=True)
            minutes, seconds = divmod(pos.value // 1000, 60)
            label = f"{minutes:02d}:{seconds:02d}"
            msg = _(
                "'{name}' triggers at {position} (sample {slot}/{total})."
            ).format(
                name=action_name,
                position=label,
                slot=result.slot_index + 1,
                total=result.total_planned_slots,
            )
        else:
            page_display = pos.value + 1
            msg = _(
                "'{name}' triggers on page {page} (sample {slot}/{total})."
            ).format(
                name=action_name,
                page=page_display,
                slot=result.slot_index + 1,
                total=result.total_planned_slots,
            )

        self._set_status(msg)
        detail_line = _format_trigger_detail(result.detail)
        self._set_detail(detail_line if detail_line else _NBSP)
        self._app_actions.toast(msg)

    def _on_not_found(self, planned_slots: int, action_name: str) -> None:
        msg = _(
            "'{name}' did not trigger on the current media ({n} samples scanned)."
        ).format(name=action_name, n=planned_slots)
        self._set_status(msg)
        self._set_detail(_NBSP)
        self._app_actions.toast(msg)

    def _on_error(self, message: str) -> None:
        logger.error("SeekToTriggerWorker error: %s", message)
        msg = _("Error during trigger scan: {0}").format(message)
        self._set_status(msg)
        self._set_detail(_NBSP)
        self._app_actions.warn(msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_last_action_lbl(self) -> None:
        action = SeekToTriggerTab._last_action
        if action is not None:
            self._last_action_lbl.setText(_("Last action: {name}").format(name=action.name))
        else:
            self._last_action_lbl.setText(_NBSP)

    def _on_density_toggled(self, checked: bool) -> None:
        self._density_slider.setEnabled(checked)
        self._density_lbl.setEnabled(checked)
        self._density_warn_lbl.setVisible(
            checked and self._density_slider.value() >= self._WARN_SAMPLE_PCT
        )

    def _on_density_changed(self, value: int) -> None:
        self._density_lbl.setText(f"{value} %")
        self._density_warn_lbl.setVisible(
            self._density_check.isChecked() and value >= self._WARN_SAMPLE_PCT
        )

    def _copy_path(self) -> None:
        path = self._app_actions.get_active_media_filepath()
        if path:
            QApplication.clipboard().setText(path)

    def _set_status(self, text: str) -> None:
        self._status_lbl.setText(text or _NBSP)

    def _set_detail(self, text: str) -> None:
        self._detail_lbl.setText(text or _NBSP)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()
        self._update_last_action_lbl()

    # ------------------------------------------------------------------
    # Headless / keybind entry point
    # ------------------------------------------------------------------

    @classmethod
    def _restore_last_action_from_cache(cls) -> Optional[ClassifierAction]:
        """Return the ClassifierAction whose name was last saved, searching both
        classifier_actions and prevalidations. Returns None if not found."""
        name = app_info_cache.get_meta(_LAST_ACTION_CACHE_KEY)
        if not name:
            return None
        all_actions = (
            list(ClassifierActionsManager.classifier_actions)
            + list(ClassifierActionsManager.prevalidations)
        )
        for action in all_actions:
            if action.name == name:
                return action
        return None

    @classmethod
    def run_last_seek_to_trigger(cls, app_actions) -> None:
        """Run a seek-to-trigger for the last-used action without opening the window.

        If the ClassifierManagementWindow is already visible, delegates to its
        tab so the tab's labels and cycling state stay in sync.  Otherwise runs
        a standalone background scan and shows results as a toast.
        """
        import os
        from ui.compare.classifier_management_window_qt import ClassifierManagementWindow

        action = cls._last_action
        if action is None:
            action = cls._restore_last_action_from_cache()
            if action is not None:
                cls._last_action = action  # warm the in-memory cache for this session
        if action is None:
            app_actions.toast(
                _("No previous seek-to-trigger action — use the Seek to Trigger tab first.")
            )
            return

        # Delegate to the visible tab so its labels and cycling state stay in sync.
        mgmt = ClassifierManagementWindow._instance
        if mgmt is not None:
            try:
                if mgmt.isVisible():
                    mgmt._seek_to_trigger_tab._seek_to_trigger(action)
                    return
            except (RuntimeError, AttributeError):
                pass

        # Window not up — run headless and emit results as a toast.
        media_path = app_actions.get_active_media_filepath()
        if not media_path:
            app_actions.toast(_("No media loaded."))
            return
        if not is_classifier_dynamic_media_path(media_path):
            app_actions.toast(_("Current media is not video, GIF, or PDF."))
            return

        if getattr(cls, '_headless_worker', None) is not None:
            if cls._headless_worker.isRunning():
                app_actions.toast(_("A seek scan is already in progress."))
                return

        nav_key = (id(action), media_path)
        last_slot = cls._last_trigger_slot.get(nav_key)
        start_slot = (last_slot + 1) if last_slot is not None else 0

        app_actions.toast(
            _("Scanning '{name}' on {file}…").format(
                name=action.name, file=os.path.basename(media_path)
            )
        )

        worker = SeekToTriggerWorker(action, media_path, start_slot=start_slot)
        cls._headless_worker = worker  # prevent GC while running

        def _on_found(result):
            cls._last_trigger_slot[nav_key] = result.slot_index
            from image.frame_cache import FrameCache
            pos = FrameCache.slot_index_to_seek_position(
                media_path, result.slot_index, result.total_planned_slots
            )
            if pos is None:
                msg = _(
                    "Trigger found (sample {slot}/{total}) but seek position could not be computed."
                ).format(slot=result.slot_index + 1, total=result.total_planned_slots)
            elif pos.kind == "ms":
                app_actions.seek_media(pos.value, pause_after=True)
                minutes, seconds = divmod(pos.value // 1000, 60)
                detail_line = _format_trigger_detail(result.detail)
                msg = _(
                    "'{name}' triggers at {position} (sample {slot}/{total})."
                ).format(
                    name=action.name,
                    position=f"{minutes:02d}:{seconds:02d}",
                    slot=result.slot_index + 1,
                    total=result.total_planned_slots,
                )
                if detail_line:
                    msg = f"{msg}  {detail_line}"
            else:
                msg = _(
                    "'{name}' triggers on page {page} (sample {slot}/{total})."
                ).format(
                    name=action.name,
                    page=pos.value + 1,
                    slot=result.slot_index + 1,
                    total=result.total_planned_slots,
                )
            app_actions.toast(msg)

        def _on_not_found(n):
            app_actions.toast(
                _("'{name}' did not trigger on the current media ({n} samples scanned).").format(
                    name=action.name, n=n
                )
            )

        def _on_error(msg):
            logger.error("Headless seek-to-trigger error: %s", msg)
            app_actions.toast(_("Error during trigger scan: {0}").format(msg))

        worker.found.connect(_on_found)
        worker.not_found.connect(_on_not_found)
        worker.error.connect(_on_error)
        worker.finished.connect(lambda: setattr(cls, '_headless_worker', None))
        worker.start()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_trigger_detail(detail: Optional[TriggerDetail]) -> str:
    """Format a TriggerDetail into a human-readable line for the detail label."""
    if detail is None:
        return ""
    if detail.trigger_type == "image_classifier":
        parts = []
        if detail.category:
            parts.append(_("matched category: {cat}").format(cat=detail.category))
        if detail.top_predictions:
            top = detail.top_predictions[:5]
            scores = ", ".join(
                f"{cat} {score:.0%}" for cat, score in top if score > 0.01
            )
            if scores:
                parts.append(f"[{scores}]")
        inner = "  ".join(parts) if parts else ""
        return _("Trigger: image classifier") + (f" — {inner}" if inner else "")
    labels = {
        "embedding": _("Trigger: text embedding"),
        "prompt": _("Trigger: prompt match"),
        "prototype": _("Trigger: prototype"),
        "filename": _("Trigger: filename match"),
    }
    return labels.get(detail.trigger_type, _("Trigger: {t}").format(t=detail.trigger_type))

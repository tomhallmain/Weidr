"""
EmbeddingSeedEditWindow -- create / edit dialog for a single EmbeddingSeed (PySide6).

Three mutually exclusive modes, mirroring the add/edit duality of
LookaheadWindow but with a third capture path:

- ``seed``: rename/retag/describe an existing library entry.
- ``pending_seed``: a new EmbeddingSeed whose vector is already computed
  (e.g. a supergroup centroid) -- just needs a name/tags/description.
- ``pending_media_path``: a raw file path with no vector yet. The user picks
  which embedding architecture to use (independent of whatever compare mode
  is currently active -- see compare.embedding_capture.embedding_capture_modes()),
  and the actual embedding computation happens on Done, not before the
  dialog opens, so nothing is computed for an architecture the user doesn't pick.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QLineEdit, QPushButton, QWidget

from compare.embedding_capture import compute_media_embedding, embedding_capture_modes
from compare.embedding_seed import EmbeddingSeed
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.constants import CompareMode
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("embedding_seed_edit_window_qt")


class EmbeddingSeedEditWindow(SmartDialog):
    """Create / edit dialog for a single EmbeddingSeed's name/tags/description."""

    _instance: Optional['EmbeddingSeedEditWindow'] = None

    def __init__(
        self,
        parent: QWidget,
        app_actions,
        refresh_callback: Callable,
        seed: Optional[EmbeddingSeed] = None,
        pending_seed: Optional[EmbeddingSeed] = None,
        pending_media_path: Optional[str] = None,
        default_name: str = "",
        default_compare_mode: Optional[CompareMode] = None,
        dimensions: str = "500x300",
    ) -> None:
        if seed is not None:
            self._mode = "edit"
            self._seed = seed
        elif pending_seed is not None:
            self._mode = "create_vector_ready"
            self._seed = pending_seed
        elif pending_media_path is not None:
            self._mode = "create_from_media"
            self._seed = None
        else:
            raise ValueError(
                "EmbeddingSeedEditWindow requires one of seed, pending_seed, or pending_media_path"
            )
        self._pending_media_path = pending_media_path
        self._default_name = default_name
        self._default_compare_mode = default_compare_mode

        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Edit Embedding Seed") if self._mode == "edit" else _("Save Embedding Seed"),
            geometry=dimensions,
        )
        EmbeddingSeedEditWindow._instance = self
        self._app_actions = app_actions
        self._refresh_callback = refresh_callback

        self._build_ui()

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_Return), self).activated.connect(self._finalize)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setSpacing(8)
        row = 0

        name_lbl = QLabel(_("Name"))
        name_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(name_lbl, row, 0, Qt.AlignLeft)
        initial_name = self._default_name if self._seed is None else self._seed.name
        self._name_edit = QLineEdit(initial_name)
        grid.addWidget(self._name_edit, row, 1)
        row += 1

        tags_lbl = QLabel(_("Tags (comma-separated)"))
        tags_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(tags_lbl, row, 0, Qt.AlignLeft)
        initial_tags = "" if self._seed is None else ", ".join(self._seed.tags)
        self._tags_edit = QLineEdit(initial_tags)
        grid.addWidget(self._tags_edit, row, 1)
        row += 1

        desc_lbl = QLabel(_("Description"))
        desc_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        grid.addWidget(desc_lbl, row, 0, Qt.AlignLeft)
        initial_description = "" if self._seed is None else self._seed.description
        self._description_edit = QLineEdit(initial_description)
        grid.addWidget(self._description_edit, row, 1)
        row += 1

        self._arch_combo: Optional[QComboBox] = None
        if self._mode == "create_from_media":
            arch_lbl = QLabel(_("Embedding Architecture"))
            arch_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            grid.addWidget(arch_lbl, row, 0, Qt.AlignLeft)

            self._arch_combo = QComboBox()
            modes = embedding_capture_modes()
            for mode in modes:
                self._arch_combo.addItem(mode.get_text(), mode)
            if self._default_compare_mode in modes:
                self._arch_combo.setCurrentIndex(modes.index(self._default_compare_mode))
            grid.addWidget(self._arch_combo, row, 1)
            row += 1

        done_btn = QPushButton(_("Done"))
        done_btn.clicked.connect(self._finalize)
        grid.addWidget(done_btn, row, 0)
        row += 1

        grid.setRowStretch(row, 1)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def _finalize(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            logger.error("Embedding seed name is required")
            return
        tags = [t.strip() for t in self._tags_edit.text().split(",") if t.strip()]
        description = self._description_edit.text().strip()

        if self._mode == "edit":
            ok = EmbeddingSeed.update_seed(
                self._seed.id, name=name, description=description, tags=tags
            )
            if not ok:
                self._app_actions.warn(
                    _("An embedding seed with that name already exists.")
                )
                return
        elif self._mode == "create_vector_ready":
            self._seed.name = name
            self._seed.description = description
            self._seed.tags = tags
            ok = EmbeddingSeed.create_seed(self._seed)
            if not ok:
                self._app_actions.warn(
                    _("An embedding seed with that name already exists.")
                )
                return
            self._app_actions.toast(_("Saved embedding seed: ") + name)
        else:  # create_from_media
            if EmbeddingSeed.get_seed_by_name(name) is not None:
                self._app_actions.warn(
                    _("An embedding seed with that name already exists.")
                )
                return
            compare_mode = self._arch_combo.currentData()
            vector = self._compute_embedding_with_spinner(
                self._pending_media_path, compare_mode
            )
            if vector is None:
                self._app_actions.warn(
                    _("Could not compute an embedding for this file.")
                )
                return
            new_seed = EmbeddingSeed(
                name=name,
                description=description,
                tags=tags,
                positive=vector,
                embedding_model=compare_mode.name,
                embedding_dim=len(vector),
                source={
                    "kind": "single_media",
                    "compare_mode": compare_mode.name,
                    "source_paths_sample": [self._pending_media_path],
                },
            )
            ok = EmbeddingSeed.create_seed(new_seed)
            if not ok:
                self._app_actions.warn(
                    _("An embedding seed with that name already exists.")
                )
                return
            self._app_actions.toast(_("Saved embedding seed: ") + name)

        self.close()
        self._refresh_callback()

    def _compute_embedding_with_spinner(self, media_path: str, compare_mode: CompareMode):
        """
        Run compare.embedding_capture.compute_media_embedding on a worker
        QThread with the spinner badge animating (mirroring
        CompareWrapper._run_dynamic_prevalidation_with_spinner), so picking
        a slow-to-embed architecture or a large video doesn't freeze the
        dialog. Returns None on failure.
        """
        from PySide6.QtCore import QEventLoop, QThread

        result = [None]

        class _Worker(QThread):
            def run(self_inner):
                result[0] = compute_media_embedding(media_path, compare_mode)

        loop = QEventLoop()
        worker = _Worker()
        worker.finished.connect(loop.quit)
        self._app_actions.start_loading_spinner(force=True)
        worker.start()
        loop.exec()
        self._app_actions.stop_loading_spinner()
        return result[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        EmbeddingSeedEditWindow._instance = None
        super().closeEvent(event)

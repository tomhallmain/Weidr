"""
EmbeddingSeedLibraryWindow -- browse/manage the embedding seed library (PySide6).

Global, singleton management window for EmbeddingSeed records: list, filter by
tag, rename/retag, deprecate, delete. See docs/embedding-seed-library.md.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from compare.compare_args import CompareArgs
from compare.embedding_seed import EmbeddingSeed
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.constants import CompareMode, Mode
from utils.translations import _


def _architecture_label(seed: EmbeddingSeed) -> str:
    """Friendly display text for seed.embedding_model (e.g. "CLIP Embedding"
    instead of the raw stored "CLIP_EMBEDDING"), falling back to the raw
    string for anything CompareMode doesn't recognize (hand-edited cache,
    or an architecture removed in a later version)."""
    try:
        return CompareMode.get(seed.embedding_model).get_text()
    except Exception:
        return seed.embedding_model or _("Unknown")


class EmbeddingSeedLibraryWindow(SmartDialog):
    """Singleton window listing/managing the embedding seed library."""

    _instance: Optional['EmbeddingSeedLibraryWindow'] = None

    def __init__(self, parent: QWidget, app_actions, dimensions: str = "700x600") -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=_("Embedding Seed Library"),
            geometry=dimensions,
        )
        EmbeddingSeedLibraryWindow._instance = self
        self._app_actions = app_actions
        self._filtered_seeds: list[EmbeddingSeed] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        title = QLabel(_("Embedding Seed Library"))
        title.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 13pt;"
        )
        root.addWidget(title)

        filter_row = QHBoxLayout()
        filter_lbl = QLabel(_("Filter by tag:"))
        filter_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        filter_row.addWidget(filter_lbl)
        self._filter_edit = QLineEdit()
        self._filter_edit.textChanged.connect(self._refresh_listbox)
        filter_row.addWidget(self._filter_edit, 1)
        root.addLayout(filter_row)

        list_area = QHBoxLayout()
        self._listbox = QListWidget()
        self._listbox.setStyleSheet(
            f"QListWidget {{ background: {AppStyle.BG_COLOR};"
            f" color: {AppStyle.FG_COLOR}; }}"
        )
        self._listbox.doubleClicked.connect(self._edit_seed)
        self._listbox.currentRowChanged.connect(self._update_deprecate_button_label)
        list_area.addWidget(self._listbox, 1)

        btns = QVBoxLayout()
        btns.setSpacing(2)
        search_btn = QPushButton(_("Search"))
        search_btn.clicked.connect(self._search_with_seed)
        btns.addWidget(search_btn)
        negative_search_btn = QPushButton(_("Add to Negative Search"))
        negative_search_btn.clicked.connect(lambda: self._search_with_seed(negative=True))
        btns.addWidget(negative_search_btn)
        edit_btn = QPushButton(_("Rename / Edit Tags"))
        edit_btn.clicked.connect(self._edit_seed)
        btns.addWidget(edit_btn)
        self._deprecate_btn = QPushButton(_("Deprecate"))
        self._deprecate_btn.clicked.connect(self._toggle_deprecate)
        btns.addWidget(self._deprecate_btn)
        delete_btn = QPushButton(_("Delete"))
        delete_btn.clicked.connect(self._delete_seed)
        btns.addWidget(delete_btn)
        btns.addStretch()
        list_area.addLayout(btns)

        root.addLayout(list_area)

        self._empty_lbl = QLabel(
            _("No embedding seeds yet.\n\nRight-click any media and choose "
              "\"Add Current Media to Embedding Seed Library…\", or right-click "
              "while in Group mode and choose \"Save Supergroup as Embedding "
              "Seed…\" to capture the current supergroup's centroid.")
        )
        self._empty_lbl.setWordWrap(True)
        self._empty_lbl.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        root.addWidget(self._empty_lbl)

        self._refresh_listbox()

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # Factory (singleton)
    # ------------------------------------------------------------------
    @classmethod
    def show_window(cls, parent: QWidget, app_actions) -> None:
        if cls._instance is not None:
            try:
                if cls._instance.isVisible():
                    if cls._instance.isMinimized():
                        cls._instance.showNormal()
                    # Rebind to the calling window's app_actions -- this is a
                    # global singleton, so if it was left open from a
                    # different AppWindow, "Search"/"Add to Negative Search"
                    # must act on whichever window just reopened it, not a
                    # stale one.
                    cls._instance._app_actions = app_actions
                    cls._instance.raise_()
                    cls._instance.activateWindow()
                    return
                else:
                    # Exists but hidden (e.g. via reject()) -- discard stale ref.
                    cls._instance = None
            except Exception:
                cls._instance = None
        win = cls(parent, app_actions)
        win.show()

    # ------------------------------------------------------------------
    # List population
    # ------------------------------------------------------------------
    def _refresh_listbox(self) -> None:
        self._listbox.clear()
        tag_filter = self._filter_edit.text().strip().lower()
        self._filtered_seeds = [
            s for s in EmbeddingSeed.list_seeds()
            if not tag_filter or any(tag_filter in t.lower() for t in s.tags)
        ]
        for seed in self._filtered_seeds:
            arch_str = f" ({_architecture_label(seed)})"
            tags_str = f" [{', '.join(seed.tags)}]" if seed.tags else ""
            status = _(" (deprecated)") if seed.deprecated else ""
            self._listbox.addItem(f"{seed.name}{arch_str}{tags_str}{status}")

        has_seeds = len(EmbeddingSeed.seeds) > 0
        self._listbox.setVisible(has_seeds)
        self._empty_lbl.setVisible(not has_seeds)
        self._update_deprecate_button_label()

    def _selected_seed(self) -> Optional[EmbeddingSeed]:
        idx = self._listbox.currentRow()
        if idx < 0 or idx >= len(self._filtered_seeds):
            return None
        return self._filtered_seeds[idx]

    def _update_deprecate_button_label(self, _row=None) -> None:
        seed = self._selected_seed()
        self._deprecate_btn.setText(
            _("Reactivate") if (seed is not None and seed.deprecated) else _("Deprecate")
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _search_with_seed(self, negative: bool = False) -> None:
        """
        Run a search using the selected seed's vector as a positive (or, if
        *negative*, negative) search input, via the generic run_compare
        app_actions entry -- see docs/embedding-seed-library.md, section 6.1.

        Fails fast, before touching any compare machinery, if the seed's
        embedding_model doesn't match the active compare mode of whichever
        AppWindow this library window was (re)opened from.
        """
        seed = self._selected_seed()
        if seed is None:
            return

        active_mode = self._app_actions.get_compare_mode()
        if active_mode is None or not seed.is_compatible_with(active_mode.name):
            active_mode_text = active_mode.get_text() if active_mode else _("no compare mode")
            self._app_actions.warn(
                _("This embedding seed was captured with {0}, which doesn't match "
                  "the active compare mode ({1}). Switch compare mode first.").format(
                    _architecture_label(seed), active_mode_text
                )
            )
            return

        args = CompareArgs()
        if negative:
            args.negative_seed_vectors = [seed.positive]
        else:
            args.positive_seed_vectors = [seed.positive]
            if seed.negative is not None:
                args.negative_seed_vectors = [seed.negative]

        self._app_actions.set_mode(Mode.SEARCH)
        self._app_actions.run_compare(args)
        seed.increment_use()
        self.close()

    def _edit_seed(self) -> None:
        seed = self._selected_seed()
        if seed is None:
            return
        from ui.compare.embedding_seed_edit_window_qt import EmbeddingSeedEditWindow
        EmbeddingSeedEditWindow(
            self, self._app_actions, self._refresh_listbox, seed=seed
        ).show()

    def _toggle_deprecate(self) -> None:
        seed = self._selected_seed()
        if seed is None:
            return
        EmbeddingSeed.update_seed(seed.id, deprecated=not seed.deprecated)
        self._refresh_listbox()

    def _delete_seed(self) -> None:
        seed = self._selected_seed()
        if seed is None:
            return
        EmbeddingSeed.delete_seed(seed.id)
        self._app_actions.toast(_("Deleted embedding seed: ") + seed.name)
        self._refresh_listbox()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        EmbeddingSeedLibraryWindow._instance = None

    def reject(self) -> None:  # noqa: N802  (Escape key -- does NOT call closeEvent)
        self._on_close()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802  (X button -> default QDialog.closeEvent calls reject())
        self._on_close()
        super().closeEvent(event)

"""
FavoritesWindow -- favourited images grouped by directory (PySide6).

Shows favourited images with Open / Remove buttons inside a scrollable area.
Class-level persistence helpers (store/load_favorites, add_favorite,
get_favorites) read and write per-directory lists via app_info_cache.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGridLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.app_actions import AppActions
from utils.app_info_cache import app_info_cache
from utils.translations import _


class FavoritesWindow(SmartDialog):
    """
    Dialog showing favourited images grouped by directory.

    Each favourite row has Open and Remove buttons.
    """

    MAX_ROWS = 30
    has_any_favorites: bool = False

    # ------------------------------------------------------------------
    # Persistence (class-level)
    # ------------------------------------------------------------------
    @staticmethod
    def store_favorites() -> None:
        for d in app_info_cache._get_directory_info().keys():
            favs = app_info_cache.get(d, "favorites", default_val=None)
            if favs is not None:
                app_info_cache.set(d, "favorites", favs)

    @staticmethod
    def load_favorites() -> None:
        any_favs = False
        for d in app_info_cache._get_directory_info().keys():
            favs = app_info_cache.get(d, "favorites", default_val=None)
            if favs is not None and len(favs) > 0:
                any_favs = True
            if favs is not None:
                app_info_cache.set(d, "favorites", favs)
        FavoritesWindow.has_any_favorites = any_favs

    @staticmethod
    def get_favorites(base_dir: str) -> list:
        return list(app_info_cache.get(base_dir, "favorites", default_val=[]))

    @staticmethod
    def add_favorite(base_dir: str, image_path: str, toast_callback=None) -> None:
        favs = app_info_cache.get(base_dir, "favorites", default_val=[])
        if image_path not in favs:
            favs.append(image_path)
            app_info_cache.set(base_dir, "favorites", favs)
            app_info_cache.store()
            if toast_callback:
                toast_callback(_("Added favorite: ") + image_path)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        app_master: QWidget,
        app_actions: AppActions,
        geometry: str = "700x800",
    ) -> None:
        # Position at top of screen, offset from parent
        super().__init__(
            parent=app_master,
            position_parent=app_master,
            title=_("Favorites"),
            geometry=geometry,
            offset_y=0,
            respect_title_bar=True,
        )
        self._app_master = app_master
        self._app_actions = app_actions

        # Scroll area
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: {AppStyle.BG_COLOR}; border: none; }}"
        )

        self._viewport = QWidget()
        self._viewport.setStyleSheet(f"background: {AppStyle.BG_COLOR};")
        self._grid = QGridLayout(self._viewport)
        self._grid.setAlignment(Qt.AlignTop)
        self._grid.setColumnStretch(0, 6)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(2, 1)
        self._row = 0

        self._scroll.setWidget(self._viewport)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self._favorites_by_dir = self._gather_favorites_by_dir()
        self._build_widgets()

        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------
    def _gather_favorites_by_dir(self) -> dict[str, list[str]]:
        open_dirs: list[str] = []
        if hasattr(self._app_actions, "get_open_directories"):
            open_dirs = self._app_actions.get_open_directories()

        recent_dirs: list[str] = []
        try:
            from files.recent_directories import RecentDirectories
            recent_dirs = RecentDirectories.directories[:]
        except Exception:
            pass

        all_dirs: set[str] = set()
        for d in app_info_cache._get_directory_info().keys():
            if app_info_cache.get(d, "favorites", default_val=[]):
                all_dirs.add(d)

        ordered: list[str] = []
        seen: set[str] = set()
        for d in open_dirs + recent_dirs:
            nd = os.path.normpath(os.path.abspath(d))
            if nd in all_dirs and nd not in seen:
                ordered.append(nd)
                seen.add(nd)
        for d in all_dirs:
            if d not in seen:
                ordered.append(d)
                seen.add(d)

        result: dict[str, list[str]] = {}
        total = 0
        for d in ordered:
            favs = app_info_cache.get(d, "favorites", default_val=[])
            if favs:
                result[d] = list(favs)
                total += len(favs)

        FavoritesWindow.has_any_favorites = total > 0
        return result

    # ------------------------------------------------------------------
    # Widget builders
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        self._row = 0

        if not FavoritesWindow.has_any_favorites:
            lbl = QLabel(
                _("No favorites set.\n\nTo add a favorite, right-click an "
                  "image and select 'Add to Favorites'.")
            )
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};")
            self._grid.addWidget(lbl, 0, 0, 1, 3)
            return

        for base_dir, favorites in self._favorites_by_dir.items():
            # Directory header
            header = QLabel(f"{_('Directory')}: {os.path.normpath(base_dir)}")
            header.setStyleSheet(
                f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR}; "
                f"font-weight: bold; padding-top: 8px;"
            )
            self._grid.addWidget(header, self._row, 0, 1, 3, Qt.AlignLeft)
            self._row += 1

            for fav in favorites:
                fav_label = QLabel(os.path.basename(fav))
                fav_label.setStyleSheet(
                    f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
                )
                self._grid.addWidget(fav_label, self._row, 0, Qt.AlignLeft)

                open_btn = QPushButton(_("Open"))
                open_btn.clicked.connect(
                    lambda _c=False, f=fav, bd=base_dir: self.open_favorite(f, bd)
                )
                self._grid.addWidget(open_btn, self._row, 1)

                rm_btn = QPushButton(_("Remove Favorite"))
                rm_btn.clicked.connect(
                    lambda _c=False, f=fav, bd=base_dir: self.remove_favorite(f, bd)
                )
                self._grid.addWidget(rm_btn, self._row, 2)

                self._row += 1

    def _clear_widgets(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _refresh(self) -> None:
        self._clear_widgets()
        self._favorites_by_dir = self._gather_favorites_by_dir()
        self._build_widgets()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def open_favorite(self, fav: str, base_dir: str) -> None:
        if not os.path.isfile(fav):
            self._app_actions.warn(
                _("File not found: ") + os.path.basename(fav)
            )
            return

        from ui.image.media_details import MediaDetails

        try:
            MediaDetails.open_temp_media_canvas(
                master=self._app_master,
                media_path=fav,
                app_actions=self._app_actions,
            )
        except Exception as e:
            self._app_actions.warn(
                _("Error opening image: ") + str(e)
            )

    def remove_favorite(self, fav: str, base_dir: str) -> None:
        favs = app_info_cache.get(base_dir, "favorites", default_val=[])
        if fav in favs:
            favs.remove(fav)
            app_info_cache.set(base_dir, "favorites", favs)
            app_info_cache.store()
            self._app_actions.toast(_("Removed favorite: ") + fav)
            self._refresh()

    def close_window(self, event=None) -> None:
        self.close()

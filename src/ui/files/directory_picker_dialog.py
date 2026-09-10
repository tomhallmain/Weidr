"""
Shared base class for directory-picker windows (RecentDirectoryWindow,
TargetDirectoryWindow, and any future variants).

Provides:
  - Scrollable grid of directory labels + "Select" buttons
  - Filter-by-typing with 4-pass ranked matching
  - Up / Down arrow rolling of the list
  - Browse-for-directory button
  - Clear-all button
  - Escape / Return keybindings
  - Automatic geometry sizing

Subclasses override a small set of hooks to specialise behaviour.
"""

from __future__ import annotations

import os
import string
from abc import abstractmethod
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from lib.fast_directory_picker_qt import get_existing_directory
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.translations import _


class AlphabetAvailabilityDialog(SmartDialog):
    """Simple modal for showing used/unused initial directory letters."""

    def __init__(self, parent: QWidget, *, title: str, message: str) -> None:
        super().__init__(
            parent=parent,
            position_parent=parent,
            title=title,
            geometry="520x220",
            center=True,
        )
        self.setModal(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        label = QLabel(message)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        outer.addWidget(label, 1)

        close_bar = QHBoxLayout()
        close_bar.addStretch()
        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(self.accept)
        close_bar.addWidget(close_btn)
        outer.addLayout(close_bar)


class DirectoryPickerDialog(SmartDialog):
    """
    Abstract base for directory-picker dialogs.

    Subclasses must implement:
        ``_get_all_directories``  -- return the full (unfiltered) directory list
        ``_on_directory_selected`` -- handle the user's selection
        ``_add_directory``        -- persist a newly browsed directory
        ``_remove_directory``     -- remove one directory from backing store
        ``_clear_directories``    -- clear the backing store

    Optionally override:
        ``_title``       -- window title (default: "Select Directory")
        ``_browse_title`` -- file-dialog title
        ``_extra_action_buttons`` -- add more buttons to the action bar
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        parent: QWidget,
        *,
        title: str = "",
        geometry: str | None = None,
        position_parent: QWidget | None = None,
        offset_x: int = 30,
        offset_y: int = 30,
    ) -> None:
        title = title or _("Select Directory")
        geometry = geometry or "420x500"

        super().__init__(
            parent=parent,
            position_parent=position_parent or parent,
            title=title,
            geometry=geometry,
            offset_x=offset_x,
            offset_y=offset_y,
        )

        self._filter_text = ""
        self._filtered_dirs: list[str] = list(self._get_all_directories())

        # --- layout ---
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        # Action bar – all buttons use NoFocus so Up/Down arrow keys
        # always reach keyPressEvent for list rolling / filter-by-typing.
        action_bar = QHBoxLayout()
        browse_btn = QPushButton(_("Add directory"))
        browse_btn.setFocusPolicy(Qt.NoFocus)
        browse_btn.clicked.connect(self._browse_new_directory)
        action_bar.addWidget(browse_btn)

        add_parent_btn = QPushButton(_("Add directories from parent"))
        add_parent_btn.setFocusPolicy(Qt.NoFocus)
        add_parent_btn.clicked.connect(self._add_dirs_from_parent)
        action_bar.addWidget(add_parent_btn)

        clear_btn = QPushButton(_("Clear targets"))
        clear_btn.setFocusPolicy(Qt.NoFocus)
        clear_btn.clicked.connect(self._on_clear_clicked)
        action_bar.addWidget(clear_btn)

        self._add_extra_action_buttons(action_bar)
        action_bar.addStretch()
        outer.addLayout(action_bar)

        # Filter indicator
        self._filter_label = QLabel("")
        self._filter_label.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-style: italic; padding: 2px 0;"
        )
        self._filter_label.setVisible(False)
        outer.addWidget(self._filter_label)

        # Scroll area for directory rows – NoFocus so arrow keys
        # are not consumed for scrolling (the dialog handles them).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.NoFocus)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {AppStyle.BG_COLOR}; border: none; }}"
        )

        self._viewport = QWidget()
        self._viewport.setFocusPolicy(Qt.NoFocus)
        self._viewport.setStyleSheet(f"background: {AppStyle.BG_COLOR};")
        self._grid = QGridLayout(self._viewport)
        self._grid.setAlignment(Qt.AlignTop)
        self._grid.setColumnStretch(0, 8)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(2, 1)

        scroll.setWidget(self._viewport)
        outer.addWidget(scroll, 1)

        self._build_rows()

        # Keybindings
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence(Qt.Key_Return), self).activated.connect(self._on_return)
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(self._show_unused_alphabet_popup)

        # Ensure the dialog itself has focus so keyPressEvent fires
        self.setFocus()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _get_all_directories(self) -> list[str]:
        """Return the full (unfiltered) list of directories."""
        ...

    @abstractmethod
    def _on_directory_selected(self, directory: str) -> None:
        """Called when the user selects a directory (button or Return)."""
        ...

    @abstractmethod
    def _add_directory(self, directory: str) -> None:
        """Persist a newly browsed directory into the backing store."""
        ...

    @abstractmethod
    def _clear_directories(self) -> None:
        """Clear the backing directory store."""
        ...

    @abstractmethod
    def _remove_directory(self, directory: str) -> None:
        """Remove one directory from the backing directory store."""
        ...

    def _browse_dialog_title(self) -> str:
        return _("Select directory")

    def _get_initial_browse_dir(self) -> str:
        dirs = self._get_all_directories()
        if dirs and os.path.isdir(dirs[0]):
            return dirs[0]
        return os.path.expanduser("~")

    def _add_extra_action_buttons(self, layout: QHBoxLayout) -> None:
        """Override to add more buttons to the action bar."""

    def _get_all_directories_copy(self) -> list[str]:
        """Override when the popup should use a different directory source."""
        return list(self._get_all_directories())

    # ------------------------------------------------------------------
    # Row building
    # ------------------------------------------------------------------
    def _clear_rows(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _build_rows(self) -> None:
        for i, _dir in enumerate(self._filtered_dirs):
            lbl = QLabel(_dir)
            lbl.setStyleSheet(
                f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
            )
            self._grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignVCenter)

            btn = QPushButton(_("Set"))
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(
                lambda _c=False, d=_dir: self._select_and_close(d)
            )
            self._grid.addWidget(btn, i, 1)

            remove_btn = QPushButton(_("Clear"))
            remove_btn.setFocusPolicy(Qt.NoFocus)
            remove_btn.clicked.connect(
                lambda _c=False, d=_dir: self._remove_and_refresh(d)
            )
            self._grid.addWidget(remove_btn, i, 2)

    def _rebuild(self) -> None:
        self._clear_rows()
        self._build_rows()

    # ------------------------------------------------------------------
    # Filter-by-typing (4-pass ranked matching, shared by both windows)
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        modifiers = event.modifiers()

        # Ignore if Ctrl/Shift modifiers are held (except for backspace)
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier) and key != Qt.Key_Backspace:
            super().keyPressEvent(event)
            return

        # Up / Down arrow: roll the list
        if key == Qt.Key_Down:
            if self._filtered_dirs:
                self._filtered_dirs = self._filtered_dirs[1:] + [self._filtered_dirs[0]]
                self._rebuild()
            return
        if key == Qt.Key_Up:
            if self._filtered_dirs:
                self._filtered_dirs = [self._filtered_dirs[-1]] + self._filtered_dirs[:-1]
                self._rebuild()
            return

        # Backspace: trim filter
        if key == Qt.Key_Backspace:
            if self._filter_text:
                self._filter_text = self._filter_text[:-1]
            else:
                return
        else:
            text = event.text()
            if not text or not text.isprintable():
                super().keyPressEvent(event)
                return
            self._filter_text += text

        self._apply_filter()

    def _apply_filter(self) -> None:
        all_dirs = self._get_all_directories()

        if not self._filter_text.strip():
            self._filtered_dirs = list(all_dirs)
            self._filter_label.setVisible(False)
        else:
            ft = self._filter_text.lower()
            result: list[str] = []

            # Pass 1: exact basename match
            for d in all_dirs:
                if os.path.basename(os.path.normpath(d)).lower() == ft:
                    result.append(d)
            # Pass 2: basename starts with filter
            for d in all_dirs:
                if d not in result and os.path.basename(os.path.normpath(d)).lower().startswith(ft):
                    result.append(d)
            # Pass 3: parent directory starts with filter
            for d in all_dirs:
                if d not in result:
                    parent = os.path.basename(os.path.dirname(os.path.normpath(d)))
                    if parent and parent.lower().startswith(ft):
                        result.append(d)
            # Pass 4: partial match in basename
            for d in all_dirs:
                if d not in result:
                    bn = os.path.basename(os.path.normpath(d)).lower()
                    if bn and (f" {ft}" in bn or f"_{ft}" in bn):
                        result.append(d)

            self._filtered_dirs = result
            self._filter_label.setText(_("Filter: ") + self._filter_text)
            self._filter_label.setVisible(True)

        self._rebuild()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _select_and_close(self, directory: str) -> None:
        self._on_directory_selected(directory)
        self.close()

    def _on_return(self) -> None:
        if self._filtered_dirs:
            self._select_and_close(self._filtered_dirs[0])

    def _remove_and_refresh(self, directory: str) -> None:
        self._remove_directory(directory)
        # Keep current filter text, but recompute from the updated source list.
        self._apply_filter()

    def _browse_new_directory(self) -> None:
        _dir = get_existing_directory(
            self,
            self._browse_dialog_title(),
            self._get_initial_browse_dir(),
        )
        if _dir and os.path.isdir(_dir):
            _dir = os.path.normpath(_dir)
            self._add_directory(_dir)
            self._filtered_dirs = list(self._get_all_directories())
            self._filter_text = ""
            self._filter_label.setVisible(False)
            self._rebuild()

    def _add_dirs_from_parent(self) -> None:
        parent_dir = get_existing_directory(
            self,
            _("Select parent directory"),
            self._get_initial_browse_dir(),
        )
        if not parent_dir or not os.path.isdir(parent_dir):
            return

        children = sorted(
            [
                os.path.normpath(os.path.join(parent_dir, n))
                for n in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, n))
            ],
            reverse=True,
        )
        for child in children:
            self._add_directory(child)

        self._filtered_dirs = list(self._get_all_directories())
        self._filter_text = ""
        self._filter_label.setVisible(False)
        self._rebuild()

    def _on_clear_clicked(self) -> None:
        self._clear_directories()
        self._filtered_dirs = []
        self._filter_text = ""
        self._filter_label.setVisible(False)
        self._rebuild()

    @staticmethod
    def _used_initial_letters(directories: list[str]) -> set[str]:
        letters: set[str] = set()
        for directory in directories:
            base = os.path.basename(os.path.normpath(directory))
            if not base:
                continue
            first = base[0].upper()
            if first in string.ascii_uppercase:
                letters.add(first)
        return letters

    def _show_unused_alphabet_popup(self) -> None:
        directories = self._get_all_directories_copy()
        used = self._used_initial_letters(directories)
        all_letters = set(string.ascii_uppercase)

        if not used:
            message = _(
                "No alphabet letters are currently used as the first character "
                "of any directory name in this list."
            )
        elif used == all_letters:
            message = _(
                "All letters A-Z are already used as the first character of at "
                "least one directory in this list."
            )
        else:
            unused = sorted(all_letters - used)
            message = _("Unused starting letters:") + "\n\n" + ", ".join(unused)

        dialog = AlphabetAvailabilityDialog(
            self,
            title=_("Alphabet Availability"),
            message=message,
        )
        dialog.exec()

    def close_windows(self, event=None) -> None:
        self.close()

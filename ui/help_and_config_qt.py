"""
PySide6 port of utils/help_and_config.py -- HelpAndConfig window.

Displays keyboard shortcut reference tables (Main Window, Move Marks window,
Image Details, Go To File) and editable config settings inside a scrollable dialog.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QGridLayout, QLabel, QLineEdit,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.config import config
from utils.translations import I18N

_ = I18N._


class HelpAndConfig(SmartDialog):
    """Help & Config dialog with keyboard shortcut tables and config settings."""

    has_run_import = False

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        position_parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            parent=parent,
            position_parent=position_parent or parent,
            title=_("Help and Config"),
            geometry="900x600",
        )
        self._help_labels: list[QLabel] = []

        # -- scrollable container ----------------------------------------
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            f"QScrollArea {{ background: {AppStyle.BG_COLOR}; border: none; }}"
        )

        viewport = QWidget()
        viewport.setStyleSheet(f"background: {AppStyle.BG_COLOR};")
        self._grid = QGridLayout(viewport)
        self._grid.setAlignment(Qt.AlignTop)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 9)
        self._row = 0

        scroll_area.setWidget(viewport)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)

        col_0_width = 250

        # ==============================================================
        # Main Window Shortcuts
        # ==============================================================
        self._add_section_title(_("Main Window Shortcuts"))

        main_help: dict[str, str] = {
            _("Command"): _("Description"),
            "Ctrl+A": _("Search current image in new window"),
            "Ctrl+B": _("Return to Browsing mode"),
            "Ctrl+C": _("Copy marks list"),
            "Ctrl+D": _("Set current marks from previous marks list"),
            "Ctrl+F": _("Open Favorites window"),
            "Ctrl+G": _("Open Go to file window"),
            "Ctrl+H": _("Hide/show sidebar"),
            "Ctrl+J": _("Open content filters window"),
            "Ctrl+K": _("Open marks window (no GUI)"),
            "Ctrl+M": _("Open marks window"),
            "Ctrl+N": _("Open marks action history window"),
            "Ctrl+Q": _("Quit"),
            "Ctrl+P": _("Open security configuration window"),
            "Ctrl+R": _("Run previous marks action"),
            "Ctrl+E": _("Run penultimate marks action"),
            "Ctrl+Shift+R": _("Run third-most-recent marks action"),
            "Ctrl+Return": _("Continue image generation"),
            "Ctrl+Shift+Return": _("Cancel image generation"),
            "Ctrl+S": _("Toggle slideshow"),
            "Ctrl+Shift+S": _("Run next text embedding search preset"),
            "Ctrl+T": _("Run permanent marks action"),
            "Ctrl+V": _("Open type configuration window"),
            "Ctrl+W": _("Open new compare window"),
            "Ctrl+X": _("Move previous marks to a different directory"),
            "Ctrl+Z": _("Undo previous marks changes"),
            "Shift-F / F11": _("Toggle fullscreen"),
            "Home": _("Go to first sorted image"),
            "End": _("Go to last sorted image"),
            "Left/Right Arrow\nMouse Wheel Up/Down": _("Show previous/next image"),
            "Page Up/Down": _("Page through images"),
            "Shift-A": _("Search current image in current window"),
            "Shift+B": _("Clear all hidden images"),
            "Shift+C": _("Clear marks list"),
            "Shift+D": _("Show image details"),
            "Shift+Delete\nMouse Wheel Click": _("Delete image (or marked file group if marks window selected)"),
            "Ctrl+Shift+Delete": _("Delete current base directory and all contents"),
            "Shift+G": _("Go to next mark"),
            "Shift+H": _("Show help window"),
            "Shift+I": _("Run image generation"),
            "Right Click": _("Open context menu"),
            "Shift+J": _("Run content filters for all files in the current directory"),
            "Shift+K": _("View last moved image mark"),
            "Shift+L": _("Toggle content filters"),
            "Shift+M": _("Add or remove a mark for current image"),
            "Shift+N": _("Add all marks between most recently set and current selected inclusive, or all marks in current group"),
            "Shift+O": _("Open media location"),
            "Shift+P": _("Open image in GIMP"),
            "Shift+Q": _("Randomly modify image"),
            "Shift+R": _("View related image (controlnet, etc.)"),
            "Shift+S": _("Capture screenshot"),
            "Shift+T": _("Find related images in open window"),
            "Shift+U": _("Run refacdir"),
            "Shift+V": _("Hide current image"),
            "Shift+Y": _("Set marks from downstream related images"),
            "Shift+Z": _("Undo previous marks changes"),
            "Shift+Left/Right Arrow": _("Show previous/next group"),
        }

        self._add_help_table(main_help, col_0_width)

        # ==============================================================
        # Move Marks window (separate context — chords override main window)
        # ==============================================================
        self._add_divider()
        self._add_section_title(_("Move Marks Window Shortcuts"))

        marks_scope = QLabel(
            _("These shortcuts apply when the Move Marks dialog has keyboard focus "
              "(Ctrl+M full window, or Ctrl+K minimal). They take priority over the "
              "same keys in the main window while this dialog is active."),
        )
        marks_scope.setWordWrap(True)
        marks_scope.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        marks_scope.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        marks_scope.setContentsMargins(0, 0, 0, 8)
        marks_scope.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        self._grid.addWidget(marks_scope, self._row, 0, 1, 2)
        self._row += 1
        self._help_labels.append(marks_scope)

        marks_help: dict[str, str] = {
            _("Command"): _("Description"),
            "Escape": _("Close Move Marks window"),
            "Enter": _("Move marked files to inferred target (filtered row, filter text, or last used)"),
            "Shift+Enter": _("Copy marked files instead of move"),
            "Ctrl+Enter": _("Choose destination folder, then move or copy"),
            "Alt+Enter": _("Move or copy using the penultimate marks action target"),
            "Shift+Delete": _("Delete all marked files"),
            "Ctrl+T": _("Arm permanent mark target (next Move/Copy sets the permanent action)"),
            "Ctrl+Shift+O": _("Sort filtered target directories by CLIP embedding (GUI mode only)"),
            "Ctrl+H": _("Open hotkey actions configuration (password)"),
            "Page Up / Page Down": _("Rotate the filtered target list in chunks"),
            "Up / Down": _("Rotate the filtered target list one row (GUI updates the list)"),
            "Backspace": _("Remove last character from target name filter"),
            _("Letter keys"): _("Filter target directories by name (both window modes)"),
            "Right Click": _("Test marked files in directory (use Ctrl/Alt/Shift for variants)"),
            "Mouse Wheel Click": _("Delete all marked files"),
        }

        self._add_help_table(marks_help, col_0_width)

        # ==============================================================
        # Image Details Shortcuts
        # ==============================================================
        self._add_divider()
        self._add_section_title(_("Image Details Shortcuts"))

        details_help: dict[str, str] = {
            _("Command"): _("Description"),
            "Shift+C": _("Crop Image (Smart Detect)"),
            "Shift+L": _("Rotate Image Left"),
            "Shift+R": _("Rotate Image Right"),
            "Shift+E": _("Enhance Image"),
            "Shift+A": _("Random Crop"),
            "Shift+Q": _("Randomly Modify Image"),
            "Shift+H": _("Flip Image Horizontally"),
            "Shift+V": _("Flip Image Vertically"),
            "Shift+X": _("Copy Without EXIF"),
            "Shift+J": _("Convert to JPG"),
            "Shift+D": _("Show Metadata"),
            "Shift+R": _("Open Related Image"),
            "Shift+I": _("Run Image Generation"),
            "Shift+Y": _("Redo Prompt"),
        }

        self._add_help_table(details_help, col_0_width)

        # Ctrl-behaviour note
        note = QLabel(
            _("Note: Using Ctrl instead of Shift marks the created file "
              "and opens the marks window without GUI."),
        )
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        note.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        note.setContentsMargins(10, 10, 10, 10)
        note.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR}; "
            "padding: 0;"
        )
        self._grid.addWidget(note, self._row, 0, 1, 2)
        note.setMinimumHeight(note.sizeHint().height())
        self._row += 1
        self._help_labels.append(note)

        # ==============================================================
        # Go To File Shortcuts
        # ==============================================================
        self._add_divider()
        self._add_section_title(_("Go To File Shortcuts"))

        gotofile_help: dict[str, str] = {
            _("Command"): _("Description"),
            "Ctrl+G": _("Go To Last Moved"),
            "Ctrl+B": _("Browse File"),
            "Ctrl+R": _("Current Media"),
            "Ctrl+F": _("Find Related Files"),
            "Ctrl+E": _("Extract Base ID"),
            "Ctrl+D": _("Browse Directory"),
        }

        self._add_help_table(gotofile_help, col_0_width)

        # ==============================================================
        # Config Settings
        # ==============================================================
        self._add_divider()
        self._add_section_title(_("Config Settings"))

        # -- boolean settings (checkboxes) --
        self._cb_show_toasts = self._add_checkbox_row(
            _("Show Toasts"), config.show_toasts,
        )
        self._le_slideshow_interval = self._add_entry_row(
            _("Slideshow Interval (sec)"), str(config.slideshow_interval_seconds),
        )
        self._le_file_check_interval = self._add_entry_row(
            _("File Check Interval (sec)"), str(config.file_check_interval_seconds),
        )
        self._le_max_search_results = self._add_entry_row(
            _("Max Search Results"), str(config.max_search_results),
        )

        # Sort By (read-only display)
        sort_label = self._make_label(_("Sort By"), col_0_width)
        sort_label.setFixedWidth(col_0_width)
        sort_value = self._make_label(str(config.sort_by))
        self._grid.addWidget(sort_label, self._row, 0, Qt.AlignLeft | Qt.AlignTop)
        self._grid.addWidget(sort_value, self._row, 1, Qt.AlignLeft | Qt.AlignTop)
        self._row += 1

        self._cb_enable_prevalidations = self._add_checkbox_row(
            _("Enable Prevalidations"), config.enable_prevalidations,
        )
        self._le_toasts_persist = self._add_entry_row(
            _("Toasts Persist (sec)"), str(config.toasts_persist_seconds),
        )
        self._le_title_notify_persist = self._add_entry_row(
            _("Title Notify Persist (sec)"), str(config.title_notify_persist_seconds),
        )
        self._cb_delete_instantly = self._add_checkbox_row(
            _("Delete Instantly"), config.delete_instantly,
        )
        self._le_trash_folder = self._add_entry_row(
            _("Trash Folder"), str(config.trash_folder),
        )
        self._cb_image_tagging = self._add_checkbox_row(
            _("Image Tagging Enabled"), config.image_tagging_enabled,
        )
        self._cb_escape_backslash = self._add_checkbox_row(
            _("Escape Backslash Filepaths"), config.escape_backslash_filepaths,
        )

        # Large-image behavior (static image loading path)
        self._le_large_dim_threshold = self._add_entry_row(
            _("Large Image Threshold (px)"), str(config.large_image_dim_threshold_px),
        )
        self._le_large_preview_overscan = self._add_entry_row(
            _("Large Preview Overscan"), str(config.large_image_preview_overscan),
        )
        self._le_large_preview_max_dim = self._add_entry_row(
            _("Large Preview Max Dimension"), str(config.large_image_preview_max_dim),
        )
        self._cb_large_hq_downscale = self._add_checkbox_row(
            _("Large Image HQ Idle Downscale"), config.large_image_enable_hq_idle_downscale,
        )
        self._cb_large_full_promotion = self._add_checkbox_row(
            _("Large Image Full-Res Promotion"), config.large_image_enable_full_res_promotion,
        )
        self._le_large_hq_ratio_threshold = self._add_entry_row(
            _("Large HQ Downscale Ratio Threshold"), str(config.large_image_hq_downscale_ratio_threshold),
        )
        self._le_large_promotion_min_ram = self._add_entry_row(
            _("Large Promotion Min Free RAM (GB)"), str(config.large_image_promotion_min_free_ram_gb),
        )
        self._le_large_promotion_max_mb = self._add_entry_row(
            _("Large Promotion Max Estimated (MB)"), str(config.large_image_promotion_max_estimated_mb),
        )
        self._le_large_promotion_ram_fraction = self._add_entry_row(
            _("Large Promotion RAM Fraction"), str(config.large_image_promotion_available_ram_fraction),
        )

        # File-scan and scale guardrails
        self._le_skip_file_check_over = self._add_entry_row(
            _("Skip File Check If Files Over"), str(config.file_check_skip_if_n_files_over),
        )
        self._le_screenshot_directory = self._add_entry_row(
            _("Screenshot Directory"), str(config.screenshot_directory),
        )
        self._cb_save_screenshot_same_dir = self._add_checkbox_row(
            _("Save Screenshot To Same Dir"), config.save_screenshot_to_same_dir,
        )

        # TODO: Config editing/persistence for this dialog is intentionally
        # not enabled yet. Fields are display-only for now.

        # -- Escape to close ---------------------------------------------
        shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut.activated.connect(self.close)

    # ==================================================================
    # Helpers
    # ==================================================================
    def close_windows(self, event=None) -> None:  # noqa: D401
        self.close()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _add_section_title(self, text: str) -> None:
        title = QLabel(text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR}; "
            f"font-weight: bold; padding-bottom: 6px;"
        )
        self._grid.addWidget(title, self._row, 0, 1, 2)
        self._row += 1

    def _add_divider(self) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; margin-top: 10px; margin-bottom: 6px;"
        )
        self._grid.addWidget(line, self._row, 0, 1, 2)
        self._row += 1

    def _add_help_table(self, items: dict[str, str], col_0_width: int) -> None:
        for key, value in items.items():
            key_label = self._make_label(key, col_0_width)
            # Keep command labels visually stable: no auto-wrap in column 0.
            # Explicit newlines in the shortcut text still render as intended.
            key_label.setFixedWidth(col_0_width)
            key_label.setWordWrap(False)
            val_label = self._make_label(value)
            self._grid.addWidget(key_label, self._row, 0, Qt.AlignLeft | Qt.AlignTop)
            self._grid.addWidget(val_label, self._row, 1, Qt.AlignTop)
            self._row += 1
            self._help_labels.extend([key_label, val_label])

    def _make_label(self, text: str, max_width: int | None = None) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        if max_width is not None:
            lbl.setMaximumWidth(max_width)
        lbl.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR};"
        )
        return lbl

    def _add_checkbox_row(self, label_text: str, initial: bool) -> QCheckBox:
        lbl = self._make_label(label_text, 250)
        lbl.setFixedWidth(250)
        cb = QCheckBox()
        cb.setChecked(initial)
        cb.setStyleSheet(
            f"QCheckBox {{ color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR}; }}"
        )
        self._grid.addWidget(lbl, self._row, 0, Qt.AlignLeft | Qt.AlignTop)
        self._grid.addWidget(cb, self._row, 1, Qt.AlignLeft | Qt.AlignTop)
        self._row += 1
        return cb

    def _add_entry_row(self, label_text: str, initial: str) -> QLineEdit:
        lbl = self._make_label(label_text, 250)
        lbl.setFixedWidth(250)
        entry = QLineEdit(initial)
        entry.setFixedWidth(300)
        entry.setStyleSheet(
            f"QLineEdit {{ color: {AppStyle.FG_COLOR}; "
            f"background: {AppStyle.BG_INPUT}; "
            f"border: 1px solid {AppStyle.BORDER_COLOR}; "
            f"padding: 2px 4px; }}"
        )
        self._grid.addWidget(lbl, self._row, 0, Qt.AlignLeft | Qt.AlignTop)
        self._grid.addWidget(entry, self._row, 1, Qt.AlignLeft | Qt.AlignTop)
        self._row += 1
        return entry

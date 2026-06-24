"""
PySide6 port of utils/help_and_config.py -- HelpAndConfig window.

Displays keyboard shortcut reference tables (Main Window, Move Marks window,
Image Details, Go To File) on a Help tab and editable config settings on a
Config tab inside a tabbed dialog.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QTabWidget,
    QVBoxLayout, QWidget,
)

from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.config import config
from utils.translations import _


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

        tab_widget = QTabWidget(self)
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {AppStyle.BORDER_COLOR};
                background: {AppStyle.BG_COLOR};
            }}
            QTabBar::tab {{
                color: {AppStyle.FG_COLOR};
                background: {AppStyle.BG_COLOR};
                padding: 6px 18px;
            }}
            QTabBar::tab:selected {{ background: {AppStyle.BG_INPUT}; }}
            QTabBar::tab:hover    {{ background: {AppStyle.BG_INPUT}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tab_widget)

        col_0_width = 250

        # ==============================================================
        # Helper: create a scrollable grid tab and point self._grid at it
        # ==============================================================
        def _start_tab(label: str) -> QVBoxLayout:
            outer = QWidget()
            outer.setStyleSheet(f"background: {AppStyle.BG_COLOR};")
            outer_vbox = QVBoxLayout(outer)
            outer_vbox.setContentsMargins(0, 0, 0, 0)
            outer_vbox.setSpacing(0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet(
                f"QScrollArea {{ background: {AppStyle.BG_COLOR}; border: none; }}"
            )
            viewport = QWidget()
            viewport.setStyleSheet(f"background: {AppStyle.BG_COLOR};")
            grid = QGridLayout(viewport)
            grid.setAlignment(Qt.AlignTop)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 9)
            scroll.setWidget(viewport)
            outer_vbox.addWidget(scroll)
            tab_widget.addTab(outer, label)
            self._grid = grid
            self._row = 0
            return outer_vbox

        # ==============================================================
        # Help tab
        # ==============================================================
        _start_tab(_("Help"))

        # --- Main Window Shortcuts -------------------------------------
        self._add_section_title(_("Main Window Shortcuts"))

        self._add_sub_section_title(_("Navigation"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Left/Right Arrow\nMouse Wheel Up/Down": _("Show previous/next media"),
            "Home": _("Go to first sorted media"),
            "End": _("Go to last sorted media"),
            "Page Up/Down": _("Page through media"),
            "Shift+Backspace": _("Go back to previously viewed media"),
            "Shift+Left/Right Arrow": _("Show previous/next group"),
            "Ctrl+Shift+Left/Right Arrow": _("Show previous/next supergroup"),
        }, col_0_width)

        self._add_sub_section_title(_("View"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Shift+F / F11": _("Toggle fullscreen"),
            "Escape": _("Exit fullscreen"),
            "Ctrl+H": _("Hide/show sidebar"),
            "Ctrl+Shift+G": _("Toggle masonry/grid view"),
            "Ctrl+B": _("Return to browsing mode"),
            "Ctrl+S": _("Toggle slideshow"),
        }, col_0_width)

        self._add_sub_section_title(_("Info / Details"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Shift+D": _("Show media details"),
            "Shift+R": _("View related image (controlnet, etc.)"),
            "Shift+T": _("Find related images in open window"),
            "Shift+E": _("Copy image prompt to clipboard"),
            "Shift+K": _("View last moved media mark"),
            "Ctrl+Shift+K": _("View last system-moved media"),
            "Shift+H / F1": _("Show help window"),
            "Right Click": _("Open context menu"),
        }, col_0_width)

        self._add_sub_section_title(_("Search / Compare"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Shift+A": _("Search current media in current window"),
            "Ctrl+A": _("Search current media in new window"),
            "Shift+Z": _("Add current media to negative search"),
            "Ctrl+Shift+S": _("Run next text embedding search preset"),
            "Ctrl+L": _("Run last seek-to-trigger action"),
        }, col_0_width)

        self._add_sub_section_title(_("Image Generation"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Shift+I": _("Run image generation"),
            "Shift+Q": _("Randomly modify image"),
            "Shift+W": _("Source a random prompt"),
            "Ctrl+Return": _("Continue image generation"),
            "Ctrl+Shift+Return": _("Cancel image generation"),
            "Ctrl+Alt+Return": _("Revert image generation to simple settings"),
        }, col_0_width)

        self._add_sub_section_title(_("Marks"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Shift+M": _("Add or remove a mark for current media"),
            "Shift+N": _("Add all marks between most recently set and current selected inclusive, or all marks in current group"),
            "Shift+G": _("Go to next mark"),
            "Shift+C": _("Clear marks list"),
            "Ctrl+C": _("Copy marks list"),
            "Ctrl+D": _("Set current marks from previous marks list"),
            "Shift+Y": _("Set marks from downstream related images"),
            "Ctrl+Y": _("Mark sources with downstream files in current directory"),
            "Ctrl+Shift+Y": _("Mark all downstream files in current directory"),
            "0-9": _("Run hotkey marks action"),
            "Shift+0-9": _("Run hotkey marks action (shift variant)"),
        }, col_0_width)

        self._add_sub_section_title(_("Marks Actions"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Ctrl+R": _("Run previous marks action"),
            "Ctrl+E": _("Run penultimate marks action"),
            "Ctrl+Shift+E": _("Run third-most-recent marks action"),
            "Ctrl+T": _("Run permanent marks action"),
            "Ctrl+Shift+T": _("Run file action set"),
            "Ctrl+Z": _("Undo previous marks changes"),
            "Ctrl+X": _("Undo last marked files move"),
            "Ctrl+Shift+I": _("Create interpolation GIF from marked files"),
        }, col_0_width)

        self._add_sub_section_title(_("File Operations"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Shift+O": _("Open media location"),
            "Shift+P": _("Open image in GIMP"),
            "Shift+V": _("Hide current media"),
            "Shift+B": _("Clear all hidden media"),
            "Shift+U": _("Run refacdir"),
            "Shift+S": _("Capture screenshot or copy PDF/SVG/HTML image"),
            "Shift+Delete\nMouse Wheel Click": _("Delete media (or marked file group if marks window selected)"),
            "Ctrl+Shift+Delete": _("Delete current base directory and all contents"),
        }, col_0_width)

        self._add_sub_section_title(_("Prevalidation"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Shift+J": _("Run content filters for all files in the current directory"),
            "Shift+L": _("Toggle content filters"),
            "Ctrl+Shift+D": _("Toggle extra debug logging"),
        }, col_0_width)

        self._add_sub_section_title(_("Window Management"))
        self._add_help_table({
            _("Command"): _("Description"),
            "Ctrl+Q": _("Quit"),
            "Ctrl+Tab": _("Cycle to next window"),
            "Ctrl+Shift+Tab": _("Cycle to previous window"),
            "Shift+Escape": _("Close secondary window"),
            "Ctrl+W": _("Open new compare window"),
            "Ctrl+G": _("Open Go to file window"),
            "Ctrl+I": _("Open Go to file window with current media"),
            "Ctrl+F": _("Open Favorites window"),
            "Ctrl+N": _("Open marks action history window"),
            "Ctrl+M": _("Open marks window"),
            "Ctrl+K": _("Open marks window (no GUI)"),
            "Ctrl+J": _("Open content filters window"),
            "Ctrl+P": _("Open security configuration window"),
            "Ctrl+V": _("Open type configuration window"),
            "Ctrl+Shift+C": _("Open compare settings window"),
            "Ctrl+Shift+M": _("Open model manager window"),
            "Ctrl+Shift+N": _("Open directory notes window"),
        }, col_0_width)

        # --- Move Marks Window Shortcuts ------------------------------
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

        self._add_help_table({
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
        }, col_0_width)

        # --- Image Details Shortcuts ----------------------------------
        self._add_divider()
        self._add_section_title(_("Image Details Shortcuts"))

        self._add_help_table({
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
        }, col_0_width)

        note = QLabel(
            _("Note: Using Ctrl instead of Shift marks the created file "
              "and opens the marks window without GUI."),
        )
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        note.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        note.setContentsMargins(10, 10, 10, 10)
        note.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR}; padding: 0;"
        )
        self._grid.addWidget(note, self._row, 0, 1, 2)
        note.setMinimumHeight(note.sizeHint().height())
        self._row += 1
        self._help_labels.append(note)

        # --- Go To File Shortcuts -------------------------------------
        self._add_divider()
        self._add_section_title(_("Go To File Shortcuts"))

        self._add_help_table({
            _("Command"): _("Description"),
            "Ctrl+G": _("Go To Last Moved"),
            "Ctrl+B": _("Browse File"),
            "Ctrl+R": _("Current Media"),
            "Ctrl+F": _("Find Related Files"),
            "Ctrl+E": _("Extract Base ID"),
            "Ctrl+D": _("Browse Directory"),
        }, col_0_width)

        # ==============================================================
        # Config tab
        # ==============================================================
        _config_tab_vbox = _start_tab(_("Config"))

        # --- General --------------------------------------------------
        self._add_sub_section_title(_("General"))
        self._cb_always_open_new_windows = self._add_checkbox_row(
            _("Always Open New Windows"), config.always_open_new_windows,
        )
        self._le_font_size = self._add_entry_row(
            _("Font Size"), str(config.font_size),
        )
        self._le_default_main_window_size = self._add_entry_row(
            _("Default Main Window Size"), str(config.default_main_window_size),
        )
        self._le_default_secondary_window_size = self._add_entry_row(
            _("Default Secondary Window Size"), str(config.default_secondary_window_size),
        )
        self._cb_show_negative_prompt = self._add_checkbox_row(
            _("Show Negative Prompt in Image Details"), config.show_negative_prompt,
        )
        self._cb_media_volume_use_eq = self._add_checkbox_row(
            _("Use Per-Instance EQ Volume Control"), config.media_volume_use_eq,
        )
        self._cb_show_toasts = self._add_checkbox_row(
            _("Show Toasts"), config.show_toasts,
        )
        self._le_toasts_persist = self._add_entry_row(
            _("Toasts Persist (sec)"), str(config.toasts_persist_seconds),
        )
        self._le_title_notify_persist = self._add_entry_row(
            _("Title Notify Persist (sec)"), str(config.title_notify_persist_seconds),
        )

        # --- Slideshow ------------------------------------------------
        self._add_sub_section_title(_("Slideshow"))
        self._le_slideshow_interval = self._add_entry_row(
            _("Interval (sec)"), str(config.slideshow_interval_seconds),
        )
        self._le_slideshow_video_cap = self._add_entry_row(
            _("Video (sec): 0=interval, >0=max, <0=full clip"),
            str(config.slideshow_dynamic_video_max_seconds),
        )
        self._le_slideshow_gif_cap = self._add_entry_row(
            _("GIF/WebP (sec): 0=interval, >0=max, <0=one full loop"),
            str(config.slideshow_dynamic_gif_max_seconds),
        )
        self._le_slideshow_pdf_pages = self._add_entry_row(
            _("PDF pages: 0=interval, N>0=N×interval, N<0=page count×interval"),
            str(config.slideshow_dynamic_pdf_max_pages),
        )

        # --- Comparison / Search -------------------------------------
        self._add_sub_section_title(_("Comparison / Search"))

        sort_label = self._make_label(_("Sort By"), col_0_width)
        sort_label.setFixedWidth(col_0_width)
        sort_value = self._make_label(str(config.sort_by))
        self._grid.addWidget(sort_label, self._row, 0, Qt.AlignLeft | Qt.AlignTop)
        self._grid.addWidget(sort_value, self._row, 1, Qt.AlignLeft | Qt.AlignTop)
        self._row += 1

        self._le_max_search_results = self._add_entry_row(
            _("Max Search Results"), str(config.max_search_results),
        )
        self._cb_presets_exclusive = self._add_checkbox_row(
            _("Text Embedding Presets Exclusive Mode"), config.text_embedding_search_presets_exclusive,
        )
        self._le_dup_threshold_embedding = self._add_entry_row(
            _("Potential Duplicate Threshold (embedding)"), str(config.threshold_potential_duplicate_embedding),
        )
        self._le_dup_threshold_color = self._add_entry_row(
            _("Potential Duplicate Threshold (color)"), str(config.threshold_potential_duplicate_color),
        )
        self._le_embed_sample_ratio = self._add_entry_row(
            _("Embedding Compare Dynamic Media Sample Ratio"), str(config.compare_embedding_dynamic_media_sample_ratio),
        )
        self._le_embed_max_samples = self._add_entry_row(
            _("Embedding Compare Dynamic Media Max Samples"), str(config.compare_embedding_dynamic_media_max_samples),
        )

        # --- File Operations / Marks ---------------------------------
        self._add_sub_section_title(_("File Operations / Marks"))
        self._cb_image_tagging = self._add_checkbox_row(
            _("Image Tagging Enabled"), config.image_tagging_enabled,
        )
        self._cb_escape_backslash = self._add_checkbox_row(
            _("Escape Backslash Filepaths"), config.escape_backslash_filepaths,
        )
        self._cb_delete_instantly = self._add_checkbox_row(
            _("Delete Instantly"), config.delete_instantly,
        )
        self._le_trash_folder = self._add_entry_row(
            _("Trash Folder"), str(config.trash_folder or ""),
        )
        self._cb_clear_marks_errors = self._add_checkbox_row(
            _("Clear Marks with Errors After Move"), config.clear_marks_with_errors_after_move,
        )
        self._cb_overwrite_on_move = self._add_checkbox_row(
            _("Overwrite Existing Files When Moving Marks"), config.move_marks_overwrite_existing_file,
        )
        self._le_file_actions_history_max = self._add_entry_row(
            _("Marks Action History Max"), str(config.file_actions_history_max),
        )

        # --- File Check ----------------------------------------------
        self._add_sub_section_title(_("File Check"))
        self._le_file_check_interval = self._add_entry_row(
            _("File Check Interval (sec)"), str(config.file_check_interval_seconds),
        )
        self._le_skip_file_check_over = self._add_entry_row(
            _("Skip File Check If Files Over"), str(config.file_check_skip_if_n_files_over),
        )

        # --- Screenshots ---------------------------------------------
        self._add_sub_section_title(_("Screenshots"))
        self._le_screenshot_directory = self._add_entry_row(
            _("Screenshot Directory"), str(config.screenshot_directory or ""),
        )
        self._cb_save_screenshot_same_dir = self._add_checkbox_row(
            _("Save Screenshot to Same Directory"), config.save_screenshot_to_same_dir,
        )

        # --- Prevalidation (Dynamic Media) ---------------------------
        self._add_sub_section_title(_("Prevalidation (Dynamic Media)"))
        self._cb_enable_prevalidations = self._add_checkbox_row(
            _("Enable Prevalidations"), config.enable_prevalidations,
        )
        self._le_dyn_min_samples = self._add_entry_row(
            _("Min Sample Count"), str(config.dynamic_media_min_sample_count),
        )
        self._le_dyn_max_frames = self._add_entry_row(
            _("Max Sample Frames"), str(config.dynamic_media_max_sample_frames),
        )
        self._le_dyn_max_pages = self._add_entry_row(
            _("Max Sample Pages"), str(config.dynamic_media_max_sample_pages),
        )
        self._le_dyn_max_duration = self._add_entry_row(
            _("Max Sample Duration (sec, 0=no cap)"), str(config.dynamic_media_max_sample_duration_seconds),
        )
        self._le_dyn_max_size_mb = self._add_entry_row(
            _("Max Sample File Size (MB, 0=no cap)"), str(config.dynamic_media_max_sample_size_mb),
        )

        # --- Large Images --------------------------------------------
        self._add_sub_section_title(_("Large Images"))
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

        # --- External Tools ------------------------------------------
        self._add_sub_section_title(_("External Tools"))
        self._le_gimp_exe = self._add_entry_row(
            _("GIMP Executable Location"), str(config.gimp_exe_loc or ""),
        )
        self._le_sd_prompt_reader = self._add_entry_row(
            _("Stable Diffusion Prompt Reader Location"), str(config.sd_prompt_reader_loc or ""),
        )

        # -- Save button (outside scroll area, always visible) -----------
        save_bar = QWidget()
        save_bar.setStyleSheet(
            f"background: {AppStyle.BG_COLOR}; "
            f"border-top: 1px solid {AppStyle.BORDER_COLOR};"
        )
        save_bar_layout = QHBoxLayout(save_bar)
        save_bar_layout.setContentsMargins(8, 6, 8, 6)
        self._save_btn = QPushButton(_("Save Config"))
        self._save_btn.setStyleSheet(
            f"QPushButton {{ color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_INPUT}; "
            f"border: 1px solid {AppStyle.BORDER_COLOR}; padding: 4px 16px; }}"
            f"QPushButton:hover {{ background: {AppStyle.BG_COLOR}; }}"
        )
        self._save_btn.clicked.connect(self._save_config)
        save_bar_layout.addWidget(self._save_btn)
        save_bar_layout.addStretch()
        _config_tab_vbox.addWidget(save_bar)

        # -- Escape to close ---------------------------------------------
        shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut.activated.connect(self.close)

    # ==================================================================
    # Helpers
    # ==================================================================
    def close_windows(self, event=None) -> None:  # noqa: D401
        self.close()

    def _save_config(self) -> None:
        """Collect Config-tab widget values and delegate validation + persistence."""
        # Checkboxes supply bool directly; line edits supply str for config to coerce.
        checkbox_fields: list[tuple[str, str]] = [
            ("_cb_always_open_new_windows",    "always_open_new_windows"),
            ("_cb_show_negative_prompt",       "show_negative_prompt"),
            ("_cb_media_volume_use_eq",        "media_volume_use_eq"),
            ("_cb_show_toasts",                "show_toasts"),
            ("_cb_presets_exclusive",          "text_embedding_search_presets_exclusive"),
            ("_cb_image_tagging",              "image_tagging_enabled"),
            ("_cb_escape_backslash",           "escape_backslash_filepaths"),
            ("_cb_delete_instantly",           "delete_instantly"),
            ("_cb_clear_marks_errors",         "clear_marks_with_errors_after_move"),
            ("_cb_overwrite_on_move",          "move_marks_overwrite_existing_file"),
            ("_cb_save_screenshot_same_dir",   "save_screenshot_to_same_dir"),
            ("_cb_enable_prevalidations",      "enable_prevalidations"),
            ("_cb_large_hq_downscale",         "large_image_enable_hq_idle_downscale"),
            ("_cb_large_full_promotion",       "large_image_enable_full_res_promotion"),
        ]
        entry_fields: list[tuple[str, str]] = [
            ("_le_font_size",                    "font_size"),
            ("_le_toasts_persist",               "toasts_persist_seconds"),
            ("_le_title_notify_persist",         "title_notify_persist_seconds"),
            ("_le_default_main_window_size",     "default_main_window_size"),
            ("_le_default_secondary_window_size","default_secondary_window_size"),
            ("_le_slideshow_interval",           "slideshow_interval_seconds"),
            ("_le_slideshow_video_cap",          "slideshow_dynamic_video_max_seconds"),
            ("_le_slideshow_gif_cap",            "slideshow_dynamic_gif_max_seconds"),
            ("_le_slideshow_pdf_pages",          "slideshow_dynamic_pdf_max_pages"),
            ("_le_max_search_results",           "max_search_results"),
            ("_le_dup_threshold_embedding",      "threshold_potential_duplicate_embedding"),
            ("_le_dup_threshold_color",          "threshold_potential_duplicate_color"),
            ("_le_embed_sample_ratio",           "compare_embedding_dynamic_media_sample_ratio"),
            ("_le_embed_max_samples",            "compare_embedding_dynamic_media_max_samples"),
            ("_le_trash_folder",                 "trash_folder"),
            ("_le_file_actions_history_max",     "file_actions_history_max"),
            ("_le_file_check_interval",          "file_check_interval_seconds"),
            ("_le_skip_file_check_over",         "file_check_skip_if_n_files_over"),
            ("_le_screenshot_directory",         "screenshot_directory"),
            ("_le_dyn_min_samples",              "dynamic_media_min_sample_count"),
            ("_le_dyn_max_frames",               "dynamic_media_max_sample_frames"),
            ("_le_dyn_max_pages",                "dynamic_media_max_sample_pages"),
            ("_le_dyn_max_duration",             "dynamic_media_max_sample_duration_seconds"),
            ("_le_dyn_max_size_mb",              "dynamic_media_max_sample_size_mb"),
            ("_le_large_dim_threshold",          "large_image_dim_threshold_px"),
            ("_le_large_preview_overscan",       "large_image_preview_overscan"),
            ("_le_large_preview_max_dim",        "large_image_preview_max_dim"),
            ("_le_large_hq_ratio_threshold",     "large_image_hq_downscale_ratio_threshold"),
            ("_le_large_promotion_min_ram",      "large_image_promotion_min_free_ram_gb"),
            ("_le_large_promotion_max_mb",       "large_image_promotion_max_estimated_mb"),
            ("_le_large_promotion_ram_fraction", "large_image_promotion_available_ram_fraction"),
            ("_le_gimp_exe",                     "gimp_exe_loc"),
            ("_le_sd_prompt_reader",             "sd_prompt_reader_loc"),
        ]

        raw: dict[str, object] = {}
        for widget_attr, config_key in checkbox_fields:
            raw[config_key] = getattr(self, widget_attr).isChecked()
        for widget_attr, config_key in entry_fields:
            raw[config_key] = getattr(self, widget_attr).text()

        try:
            errors = config.apply_and_persist(raw)
        except Exception as exc:
            QMessageBox.critical(
                self,
                _("Save Failed"),
                _("Could not write config file:\n\n") + str(exc),
            )
            return

        if errors:
            QMessageBox.warning(
                self,
                _("Config Validation Error"),
                _("Fix these fields before saving:\n\n") + "\n".join(errors),
            )
            return

        QMessageBox.information(
            self,
            _("Config Saved"),
            _("Settings saved successfully.\n\n"
              "Some changes (font size, window sizes, toast timing, etc.) "
              "require restarting the application to take full effect."),
        )

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _add_sub_section_title(self, text: str) -> None:
        title = QLabel(text)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_COLOR}; "
            f"font-weight: bold; padding-top: 8px; padding-bottom: 2px;"
        )
        self._grid.addWidget(title, self._row, 0, 1, 2)
        self._row += 1

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

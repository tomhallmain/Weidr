"""
SidebarPanel -- owns the sidebar QWidget and its layout.

Extracted from the sidebar-building portion of App.__init__ (~200 lines)
and the helper methods: add_button, add_label, apply_to_grid, new_entry,
destroy_grid_element.

Contains the QScrollArea wrapping a QVBoxLayout with all sidebar widgets
(labels, entries, buttons, checkboxes, dropdowns). Exposes references to
key widgets so that controllers can read/write them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from lib.aware_entry_qt import AwareEntry
from lib.loading_spinner_qt import LoadingSpinnerBadge
from lib.scroll_frame_qt import ScrollFrame
from lib.tooltip_qt import create_tooltip
from utils.config import config
from utils.constants import Mode, Sort, SortBy
from utils.logging_setup import get_logger
from utils.translations import I18N

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow

_ = I18N._
logger = get_logger("sidebar_panel")


class SidebarPanel(QWidget):
    """
    Sidebar widget containing all navigation controls, search inputs,
    labels, and dynamically-added mode buttons.

    Mirrors the sidebar built inside the original ``App.__init__``.
    """

    def __init__(self, parent: QWidget, app_window: AppWindow):
        super().__init__(parent)
        self._app = app_window
        self._dynamic_buttons: dict[str, QWidget] = {}

        self._init_ui()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _init_ui(self) -> None:
        """Build the sidebar layout with all persistent widgets."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(2)

        # Scrollable area for sidebar content
        self._scroll = ScrollFrame(self)
        outer.addWidget(self._scroll)
        self._add_spacer()

        # -- Mode label + loading spinner (inline row) ---------------
        mode_row = QWidget(self)
        mode_row_layout = QHBoxLayout(mode_row)
        mode_row_layout.setContentsMargins(0, 0, 0, 0)
        mode_row_layout.setSpacing(4)
        self.label_mode = QLabel(Mode.BROWSE.get_text(), mode_row)
        mode_row_layout.addWidget(self.label_mode, 1)
        self.loading_spinner = LoadingSpinnerBadge(mode_row)
        mode_row_layout.addWidget(self.loading_spinner)
        self._scroll.add_widget(mode_row)

        self.label_state = QLabel(_("Set a directory to run comparison."), self)
        self.label_state.setWordWrap(True)
        self._scroll.add_widget(self.label_state)
        self._add_spacer()

        # ========== Settings UI ===========================================

        # Toggle theme button
        self.toggle_theme_btn = self._make_button(
            _("Toggle theme"), lambda: self._app.toggle_theme()
        )
        create_tooltip(self.toggle_theme_btn, _("Switch between light and dark theme."))
        self._add_spacer()

        # Set directory
        self.set_base_dir_btn = self._make_button(
            _("Set directory"), lambda: self._app.set_base_dir()
        )
        create_tooltip(
            self.set_base_dir_btn,
            _("Set the base directory to browse or run comparisons on."),
        )

        self.set_base_dir_box = AwareEntry(self)
        self.set_base_dir_box.setPlaceholderText(_("Enter base directory..."))
        self.set_base_dir_box.returnPressed.connect(lambda: self._app.set_base_dir())
        self._scroll.add_widget(self.set_base_dir_box)

        self.open_directory_notes_btn = self._make_button(
            _("Directory notes"),
            lambda: self._app.window_launcher.open_directory_notes_window(),
        )
        create_tooltip(
            self.open_directory_notes_btn,
            _("Open and edit notes associated with the current directory."),
        )
        self._add_spacer()

        # Browsing section
        self._add_label(_("Browsing options"))

        # Inclusion pattern (file glob filter)
        self._add_label(_("Filter files by glob pattern"))
        self.inclusion_pattern = AwareEntry(self)
        self.inclusion_pattern.returnPressed.connect(self._on_set_file_filter)
        self._scroll.add_widget(self.inclusion_pattern)

        # Sort by + sort direction (same row)
        self._add_label(_("Browsing mode - Sort by"))
        sort_row = QWidget(self)
        sort_row_layout = QHBoxLayout(sort_row)
        sort_row_layout.setContentsMargins(0, 0, 0, 0)
        sort_row_layout.setSpacing(4)

        self.sort_by_choice = QComboBox(self)
        for text in SortBy.members():
            self.sort_by_choice.addItem(text)
        self.sort_by_choice.setCurrentText(config.sort_by.get_text())
        self.sort_by_choice.currentTextChanged.connect(self._on_sort_by_changed)
        sort_row_layout.addWidget(self.sort_by_choice, 3)

        self.sort_direction_choice = QComboBox(self)
        self.sort_direction_choice.addItem(Sort.ASC.get_text())
        self.sort_direction_choice.addItem(Sort.DESC.get_text())
        self.sort_direction_choice.setEnabled(config.sort_by != SortBy.RANDOMIZE)
        self.sort_direction_choice.currentTextChanged.connect(self._on_sort_changed)
        sort_row_layout.addWidget(self.sort_direction_choice, 1)

        self._scroll.add_widget(sort_row)

        # Checkboxes
        self.recursive_check = QCheckBox(_("Recurse subdirectories"), self)
        self.recursive_check.setChecked(config.image_browse_recursive)
        self.recursive_check.stateChanged.connect(self._on_toggle_recursive)
        self._scroll.add_widget(self.recursive_check)

        self.fill_canvas_check = QCheckBox(_("Media resize to full window"), self)
        self.fill_canvas_check.setChecked(config.fill_canvas)
        self.fill_canvas_check.stateChanged.connect(self._on_toggle_fill_canvas)
        self._scroll.add_widget(self.fill_canvas_check)

        self.search_return_closest_check = QCheckBox(_("Search only return closest"), self)
        self.search_return_closest_check.setChecked(config.search_only_return_closest)
        self.search_return_closest_check.stateChanged.connect(self._on_toggle_search_return_closest)
        self._scroll.add_widget(self.search_return_closest_check)
        self._add_spacer()

        # ========== Search UI =============================================
        self._add_label(_("Search"))

        # Search image
        self.set_search_btn = self._make_button(
            _("Set search file"), lambda: self._app.search_ctrl.set_search_for_image()
        )
        create_tooltip(
            self.set_search_btn,
            _("Set a media file to search for similar media.\n"
              "Uses embedding similarity to find visually similar media."),
        )

        self.search_img_path_box = AwareEntry(self)
        self.search_img_path_box.setPlaceholderText(_("Search media path..."))
        self.search_img_path_box.returnPressed.connect(
            lambda: self._app.search_ctrl.set_search_for_image()
        )
        self._scroll.add_widget(self.search_img_path_box)

        # Negative search image
        self.set_negative_search_btn = self._make_button(
            _("Set negative search file"),
            lambda: self._app.search_ctrl.set_negative_search_for_image(),
        )
        create_tooltip(
            self.set_negative_search_btn,
            _("Set a media file to search away from — results will exclude media similar to this."),
        )
        self.search_img_negative_path_box = AwareEntry(self)
        self.search_img_negative_path_box.setPlaceholderText(_("Negative search media path..."))
        self.search_img_negative_path_box.returnPressed.connect(
            lambda: self._app.search_ctrl.set_negative_search_for_image()
        )
        self._scroll.add_widget(self.search_img_negative_path_box)

        # Search text (embedding)
        self.search_text_btn = self._make_button(
            _("Search text (embedding mode)"),
            lambda: self._app.search_ctrl.set_search_for_text(),
        )
        create_tooltip(
            self.search_text_btn,
            _("Positive text: Find media similar to this text.\n"
              "Negative text: Exclude media similar to this text.\n"
              "Both use embedding similarity matching."),
        )

        # Positive text
        self._add_label(_("Positive text:"))
        self.search_text_box = AwareEntry(self)
        self.search_text_box.returnPressed.connect(
            lambda: self._app.search_ctrl.set_search_for_text()
        )
        self._scroll.add_widget(self.search_text_box)

        # Negative text
        self._add_label(_("Negative text:"))
        self.search_text_negative_box = AwareEntry(self)
        self.search_text_negative_box.returnPressed.connect(
            lambda: self._app.search_ctrl.set_search_for_text()
        )
        self._scroll.add_widget(self.search_text_negative_box)
        self._add_spacer()

        # ========== Compare tools =========================================
        self._add_label(_("Compare tools"))

        self.classifier_actions_btn = self._make_button(
            _("Classifier Actions"),
            lambda: self._app.window_launcher.open_classifier_actions_window(),
        )
        create_tooltip(
            self.classifier_actions_btn,
            _("Configure rules for automatically copying or moving files\n"
              "based on classifier results."),
        )
        self.hf_model_manager_btn = self._make_button(
            _("Model Manager"),
            lambda: self._app.window_launcher.open_hf_model_manager_window(),
        )
        create_tooltip(
            self.hf_model_manager_btn,
            _("Manage HuggingFace models used for media embeddings and comparison."),
        )
        self._add_spacer()

        # ========== Run context-aware UI ==================================
        self._add_label(_("Compare actions"))

        self.compare_settings_btn = self._make_button(
            _("Compare Settings"),
            lambda: self._app.window_launcher.open_compare_settings_window(),
        )
        create_tooltip(
            self.compare_settings_btn,
            _("Configure settings for media comparison and duplicate detection."),
        )

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self._scroll.add_widget(self.progress_bar)

        # Compare buttons
        self.run_compare_btn = self._make_button(
            _("Run media compare"),
            lambda: self._app.search_ctrl.run_compare(),
        )
        create_tooltip(
            self.run_compare_btn,
            _("Run embedding-based similarity comparison on all media in the directory."),
        )
        self.find_duplicates_btn = self._make_button(
            _("Find duplicates"),
            lambda: self._app.search_ctrl.run_compare(find_duplicates=True),
        )
        create_tooltip(
            self.find_duplicates_btn,
            _("Find duplicate or near-duplicate media in the directory."),
        )
        self.image_details_btn = self._make_button(
            _("Media details"),
            lambda: self._app.window_launcher.open_media_details(),
        )
        create_tooltip(
            self.image_details_btn,
            _("Show detailed metadata and information about the current media file."),
        )

        # Search current media
        self.search_current_image_btn = self._make_button(
            _("Search current media"),
            lambda: self._app.search_ctrl.set_current_image_run_search(),
        )
        create_tooltip(
            self.search_current_image_btn,
            _("Search for media similar to the currently displayed media.\n"
              "Uses embedding similarity matching."),
        )
        self._add_spacer()

        self._add_label(_("File actions"))
        # File action buttons
        self.open_media_location_btn = self._make_button(
            _("Open media location"),
            lambda: self._app.file_ops_ctrl.open_media_location(),
        )
        create_tooltip(
            self.open_media_location_btn,
            _("Open the folder containing the current file in the system file manager."),
        )
        self.copy_image_path_btn = self._make_button(
            _("Copy media path"),
            lambda: self._app.file_ops_ctrl.copy_media_path(),
        )
        create_tooltip(
            self.copy_image_path_btn,
            _("Copy the full file path of the current media file to the clipboard."),
        )
        self.copy_image_basename_btn = self._make_button(
            _("Copy media basename"),
            lambda: self._app.file_ops_ctrl.copy_media_basename(),
        )
        create_tooltip(
            self.copy_image_basename_btn,
            _("Copy the filename (without directory path) of the current file to the clipboard."),
        )
        self.delete_image_btn = self._make_button(
            _("---- DELETE ----"),
            lambda: self._app.file_ops_ctrl.delete_image(),
        )
        create_tooltip(
            self.delete_image_btn,
            _("Permanently delete the current media file from disk."),
        )
        self._add_spacer()

        # Current image name label (at the bottom)
        self.label_current_media_name = QLabel("", self)
        self.label_current_media_name.setWordWrap(True)
        self.label_current_media_name.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.label_current_media_name.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        self.label_current_media_name.setMinimumWidth(0)
        self._scroll.add_widget(self.label_current_media_name)
        self._scroll.viewport().installEventFilter(self)

        # -- Mode-specific button container --------------------------------
        self._mode_button_container = QVBoxLayout()
        self._mode_button_container.setContentsMargins(0, 0, 0, 0)
        mode_widget = QWidget(self)
        mode_widget.setLayout(self._mode_button_container)
        self._scroll.add_widget(mode_widget)

    def _add_spacer(self, height: int = 15) -> None:
        """Insert a simple vertical spacer in the sidebar content."""
        spacer = QWidget(self)
        spacer.setFixedHeight(max(0, int(height)))
        self._scroll.add_widget(spacer)

    # ==================================================================
    # Widget factory helpers
    # ==================================================================
    def _add_label(self, text: str) -> QLabel:
        """Add a simple text label to the sidebar."""
        lbl = QLabel(text, self)
        self._scroll.add_widget(lbl)
        return lbl

    def _make_button(self, text: str, command: Callable) -> QPushButton:
        """Create a button, connect its signal, and add it to the scroll area."""
        btn = QPushButton(text, self)
        btn.clicked.connect(command)
        self._scroll.add_widget(btn)
        return btn

    # ==================================================================
    # Dynamic (mode-specific) button management
    # ==================================================================
    def add_button(self, name: str, text: str, command: Callable) -> QPushButton:
        """Add a dynamically-named button to the mode button container."""
        if name in self._dynamic_buttons:
            return self._dynamic_buttons[name]
        btn = QPushButton(_(text), self)
        btn.clicked.connect(command)
        self._mode_button_container.addWidget(btn)
        self._dynamic_buttons[name] = btn
        return btn

    def destroy_button(self, name: str) -> None:
        """Remove a dynamically-added button by name."""
        btn = self._dynamic_buttons.pop(name, None)
        if btn is not None:
            self._mode_button_container.removeWidget(btn)
            btn.deleteLater()

    def add_buttons_for_mode(self) -> None:
        """
        Add buttons appropriate for the current application mode.

        Ported from App._add_buttons_for_mode.
        """
        mode = self._app.mode
        if self._app.has_added_buttons_for_mode.get(mode, False):
            return

        if mode == Mode.SEARCH:
            cm = self._app.compare_manager
            if (cm.search_image_full_path
                    and cm.search_image_full_path.strip() != ""
                    and "toggle_image_view_btn" not in self._dynamic_buttons):
                self.add_button(
                    "toggle_image_view_btn",
                    "Toggle media view",
                    self._app.media_navigator.toggle_image_view,
                )
                self.add_button(
                    "replace_current_image_btn",
                    "Replace with search media",
                    self._app.file_ops_ctrl.replace_current_image_with_search_image,
                )

        elif mode == Mode.GROUP:
            self.add_button(
                "prev_group_btn",
                "Previous group",
                self._app.compare_manager.show_prev_group,
            )
            self.add_button(
                "next_group_btn",
                "Next group",
                self._app.compare_manager.show_next_group,
            )

        elif mode == Mode.DUPLICATES:
            pass  # no extra buttons currently

        self._app.has_added_buttons_for_mode[mode] = True

    def remove_all_mode_buttons(self) -> None:
        """Remove all dynamically-added mode buttons and reset flags."""
        for name in list(self._dynamic_buttons.keys()):
            self.destroy_button(name)
        for mode in self._app.has_added_buttons_for_mode:
            self._app.has_added_buttons_for_mode[mode] = False

    def remove_search_mode_buttons(self) -> None:
        """Remove buttons specific to search mode."""
        for name in ("toggle_image_view_btn", "replace_current_image_btn"):
            self.destroy_button(name)

    def remove_group_mode_buttons(self) -> None:
        """Remove buttons specific to group/duplicates mode."""
        for name in ("prev_group_btn", "next_group_btn"):
            self.destroy_button(name)

    # ==================================================================
    # Progress bar
    # ==================================================================
    def start_progress_bar(self) -> None:
        """Show an indeterminate (bouncing) progress bar."""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)

    def stop_progress_bar(self) -> None:
        """Hide the progress bar."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

    # ==================================================================
    # Sidebar-triggered actions (signal handlers)
    # ==================================================================
    def _on_sort_by_changed(self, text: str) -> None:
        """Handle sort-by dropdown change."""
        try:
            new_sort_by = SortBy.get(text)
            self._app.file_browser.set_sort_by(new_sort_by)
            self._app.file_browser.refresh()
            self.sort_direction_choice.setEnabled(new_sort_by != SortBy.RANDOMIZE)
            if self._app.mode == Mode.BROWSE:
                self._app.media_navigator.show_next_media()
        except Exception as e:
            logger.error(f"Error changing sort: {e}")

    def _on_sort_changed(self, text: str) -> None:
        """Handle sort-direction dropdown change."""
        try:
            sort = Sort.DESC if text == Sort.DESC.get_text() else Sort.ASC
            self._app.file_browser.set_sort(sort)
            self._app.file_browser.refresh()
            if self._app.mode == Mode.BROWSE:
                self._app.media_navigator.show_next_media()
        except Exception as e:
            logger.error(f"Error changing sort direction: {e}")

    def _on_set_file_filter(self) -> None:
        """Handle inclusion pattern entry Return key."""
        if self._app.slideshow_config.end_slideshows():
            self._app.media_navigator.stop_slideshow_timers()
            self._app.notification_ctrl.toast(_("Ended slideshows"))
        pattern = self.inclusion_pattern.text().strip()
        self._app.file_browser.set_filter(pattern if pattern else None)
        self._app.refresh(file_check=False)

    def _on_toggle_recursive(self, state: int) -> None:
        """Handle recursive checkbox toggle."""
        is_recursive = state == Qt.CheckState.Checked.value
        self._app.file_browser.set_recursive(is_recursive)
        if self._app.mode == Mode.BROWSE and self._app.img_path:
            self._app.media_navigator.show_next_media()
        if self._app.mode == Mode.BROWSE:
            self._app.notification_ctrl.set_label_state()

    def _on_toggle_fill_canvas(self, state: int) -> None:
        """Handle fill-canvas checkbox toggle."""
        self._app.media_frame.set_fill_canvas(state == Qt.CheckState.Checked.value)

    def _on_toggle_search_return_closest(self, state: int) -> None:
        """Handle search-return-closest checkbox toggle."""
        self._app.compare_manager.toggle_search_only_return_closest()

    # ==================================================================
    # External update hooks
    # ==================================================================
    def set_mode_label(self, text: str) -> None:
        """Update the mode indicator label."""
        self.label_mode.setText(text)

    def update_base_dir_display(self, base_dir: str) -> None:
        """Update the base directory entry widget."""
        self.set_base_dir_box.setText(base_dir)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._sync_current_image_label_max_width()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_current_image_label_max_width()

    def _sync_current_image_label_max_width(self) -> None:
        """
        Clamp the filename label to the scroll viewport so unbroken long names
        (hashes, URLs) cannot widen the sidebar column and clip centered buttons.
        """
        vp = self._scroll.viewport()
        if vp is None:
            return
        # Layout margins (4×2) + room for vertical scrollbar when shown
        margin = 24
        w = max(48, vp.width() - margin)
        self.label_current_media_name.setMaximumWidth(w)

    def update_current_image_label(self, text: str) -> None:
        """Update the current image name label."""
        self.label_current_media_name.setText(text)
        self.label_current_media_name.setToolTip(text if text else "")
        self._sync_current_image_label_max_width()

    def update_state_label(self, text: str) -> None:
        """Update the file state label (e.g. '5 / 120')."""
        self.label_state.setText(text)

    def set_sort_by_value(self, text: str) -> None:
        """Programmatically set the sort-by combo without triggering the signal."""
        self.sort_by_choice.blockSignals(True)
        self.sort_by_choice.setCurrentText(text)
        self.sort_by_choice.blockSignals(False)

    def set_sort_value(self, text: str) -> None:
        """Programmatically set the sort-direction combo without triggering the signal."""
        self.sort_direction_choice.blockSignals(True)
        self.sort_direction_choice.setCurrentText(text)
        self.sort_direction_choice.blockSignals(False)

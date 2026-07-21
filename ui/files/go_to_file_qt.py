"""
GoToFile window.

All functionality:
  - Go-to-file by name, closest match, or index
  - File picker via QFileDialog
  - Go-to-last-moved shortcut
  - Related-files search by base ID in a target directory
  - Base-ID extraction from filenames
  - Large-directory confirmation with MRU confirmed list
  - Indeterminate progress bar during loading
  - Keyboard shortcuts (Ctrl+B/G/R/F/E/D)
  - Persistence via app_info_cache
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from files.related_image import extract_filename_base_stem, find_files_by_base_stem
from ui.files.marked_file_mover_qt import MarkedFiles
from files.file_action import FileAction
from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.app_actions import AppActions
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.constants import SortBy
from utils.translations import _
from utils.utils import Utils


class GoToFile(SmartDialog):
    """PySide6 GoToFile dialog -- fully self-contained port."""

    _instance: Optional[GoToFile] = None

    # ------------------------------------------------------------------
    # Class-level persisted state
    # ------------------------------------------------------------------
    last_search_text: str = ""
    last_use_closest: bool = False
    last_use_index: bool = False
    last_closest_sort_by: SortBy = SortBy.NAME
    last_target_directory: str = Utils.get_pictures_dir()
    last_base_id: str = ""
    confirmed_directories: list[str] = []

    # Cache keys
    TARGET_DIRECTORY_KEY = "go_to_file.target_directory"
    BASE_ID_KEY = "go_to_file.base_id"
    CONFIRMED_DIRECTORIES_KEY = "go_to_file.confirmed_directories"
    MAX_CONFIRMED_DIRECTORIES = 20

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def load_persisted_data() -> None:
        persisted_target_dir = app_info_cache.get_meta(GoToFile.TARGET_DIRECTORY_KEY)
        if persisted_target_dir and os.path.isdir(persisted_target_dir):
            GoToFile.last_target_directory = persisted_target_dir

        persisted_base_id = app_info_cache.get_meta(GoToFile.BASE_ID_KEY)
        if persisted_base_id:
            GoToFile.last_base_id = persisted_base_id

        # Load, filter invalid, dedupe while preserving order, and cap size
        persisted_confirmed = app_info_cache.get_meta(
            GoToFile.CONFIRMED_DIRECTORIES_KEY, default_val=[]
        )
        cleaned: list[str] = []
        seen: set[str] = set()
        for d in persisted_confirmed:
            try:
                norm = os.path.normpath(os.path.abspath(d))
                if norm in seen:
                    continue
                if os.path.isdir(norm):
                    cleaned.append(norm)
                    seen.add(norm)
            except Exception:
                continue
        if len(cleaned) > GoToFile.MAX_CONFIRMED_DIRECTORIES:
            cleaned = cleaned[: GoToFile.MAX_CONFIRMED_DIRECTORIES]
        GoToFile.confirmed_directories = cleaned

    @staticmethod
    def save_persisted_data() -> None:
        app_info_cache.set_meta(GoToFile.TARGET_DIRECTORY_KEY, GoToFile.last_target_directory)
        app_info_cache.set_meta(GoToFile.BASE_ID_KEY, GoToFile.last_base_id)
        app_info_cache.set_meta(
            GoToFile.CONFIRMED_DIRECTORIES_KEY,
            GoToFile.confirmed_directories[: GoToFile.MAX_CONFIRMED_DIRECTORIES],
        )

    @staticmethod
    def _add_confirmed_directory(directory: str) -> None:
        """Add *directory* to confirmed list as most recent, dedupe, and cap."""
        try:
            norm = os.path.normpath(os.path.abspath(directory))
        except Exception:
            return
        GoToFile.confirmed_directories = [
            d for d in GoToFile.confirmed_directories if d != norm
        ]
        GoToFile.confirmed_directories.insert(0, norm)
        if len(GoToFile.confirmed_directories) > GoToFile.MAX_CONFIRMED_DIRECTORIES:
            GoToFile.confirmed_directories = GoToFile.confirmed_directories[
                : GoToFile.MAX_CONFIRMED_DIRECTORIES
            ]

    @staticmethod
    def get_geometry() -> str:
        return "700x500"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, master: QWidget, app_actions: AppActions) -> None:
        GoToFile.load_persisted_data()

        super().__init__(
            parent=master,
            position_parent=master,
            title=_("Go To File"),
            geometry=self.get_geometry(),
        )
        GoToFile._instance = self
        self._fallback_app_actions = app_actions
        self._loading_bar: Optional[QProgressBar] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ==============================================================
        # Go To File section
        # ==============================================================
        title_lbl = QLabel(_("Go To File"))
        title_lbl.setStyleSheet(
            f"font-size: 12pt; font-weight: bold; color: {AppStyle.FG_COLOR};"
        )
        root.addWidget(title_lbl)

        # Search text entry
        self._search_entry = QLineEdit(GoToFile.last_search_text)
        self._search_entry.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_INPUT}; "
            f"border: 1px solid {AppStyle.BORDER_COLOR}; padding: 4px;"
        )
        self._search_entry.returnPressed.connect(self.go_to_file)
        self._search_entry.textChanged.connect(self._on_filename_changed)
        root.addWidget(self._search_entry)

        # Action buttons
        btn_row = QHBoxLayout()

        go_btn = QPushButton(_("Go To"))
        go_btn.clicked.connect(self.go_to_file)
        btn_row.addWidget(go_btn)

        browse_btn = QPushButton(_("Browse..."))
        browse_btn.clicked.connect(self.pick_file)
        btn_row.addWidget(browse_btn)

        last_moved_btn = QPushButton(_("Go To Last Moved"))
        last_moved_btn.clicked.connect(self.go_to_last_moved)
        btn_row.addWidget(last_moved_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # Closest-file checkbox
        self._use_closest = QCheckBox(_("Go to closest file if exact match not found"))
        self._use_closest.setChecked(GoToFile.last_use_closest)
        self._use_closest.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._use_closest.stateChanged.connect(self._toggle_closest_options)
        root.addWidget(self._use_closest)

        # Index checkbox
        self._use_index = QCheckBox(_("Go to file by index (1-based)"))
        self._use_index.setChecked(GoToFile.last_use_index)
        self._use_index.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(self._use_index)

        # Sort-by row (hidden when closest unchecked)
        self._sort_by_row = QWidget()
        sb_layout = QHBoxLayout(self._sort_by_row)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_label = QLabel(_("Sort by for closest search:"))
        sb_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        sb_layout.addWidget(sb_label)

        self._sort_by_combo = QComboBox()
        for member in SortBy.members():
            self._sort_by_combo.addItem(member)
        self._sort_by_combo.setCurrentText(GoToFile.last_closest_sort_by.get_text())
        sb_layout.addWidget(self._sort_by_combo)
        sb_layout.addStretch()
        root.addWidget(self._sort_by_row)
        self._toggle_closest_options()

        # ==============================================================
        # Separator
        # ==============================================================
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(line)

        # ==============================================================
        # Find Related Filenames section
        # ==============================================================
        related_title = QLabel(_("Find Related Filenames"))
        related_title.setStyleSheet(
            f"font-size: 12pt; font-weight: bold; color: {AppStyle.FG_COLOR};"
        )
        root.addWidget(related_title)

        # Status label
        self._status_label = QLabel(
            _(
                "Enter a base ID manually or click 'Extract from filename' "
                "to auto-extract from the filename above."
            )
        )
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        root.addWidget(self._status_label)

        # Base ID row
        base_id_row = QHBoxLayout()
        bid_label = QLabel(_("Base ID:"))
        bid_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        base_id_row.addWidget(bid_label)

        initial_base_id = GoToFile.last_base_id
        if not initial_base_id:
            text = self._search_entry.text().strip()
            if text:
                initial_base_id = extract_filename_base_stem(text) or ""

        self._base_id_entry = QLineEdit(initial_base_id)
        self._base_id_entry.setFixedWidth(200)
        self._base_id_entry.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_INPUT}; "
            f"border: 1px solid {AppStyle.BORDER_COLOR}; padding: 2px 4px;"
        )
        base_id_row.addWidget(self._base_id_entry)

        extract_btn = QPushButton(_("Extract from filename"))
        extract_btn.clicked.connect(self.extract_and_set_base_id)
        base_id_row.addWidget(extract_btn)
        base_id_row.addStretch()
        root.addLayout(base_id_row)

        # Target directory row
        dir_row = QHBoxLayout()
        dir_label = QLabel(_("Target Directory:"))
        dir_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        dir_row.addWidget(dir_label)

        self._target_dir_entry = QLineEdit(GoToFile.last_target_directory)
        self._target_dir_entry.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; background: {AppStyle.BG_INPUT}; "
            f"border: 1px solid {AppStyle.BORDER_COLOR}; padding: 2px 4px;"
        )
        dir_row.addWidget(self._target_dir_entry, 1)

        browse_dir_btn = QPushButton(_("Browse..."))
        browse_dir_btn.clicked.connect(self.browse_target_directory)
        dir_row.addWidget(browse_dir_btn)
        root.addLayout(dir_row)

        # Related action buttons
        related_btn_row = QHBoxLayout()
        current_media_btn = QPushButton(_("Current Media"))
        current_media_btn.clicked.connect(self.get_current_media_filename)
        related_btn_row.addWidget(current_media_btn)

        find_btn = QPushButton(_("Find Related"))
        find_btn.clicked.connect(self.find_related_files)
        related_btn_row.addWidget(find_btn)
        related_btn_row.addStretch()
        root.addLayout(related_btn_row)

        # Cache scan checkbox (session-only; cache is in-memory so always empty at startup)
        self._cache_scan_checkbox = QCheckBox(_("Cache directory scan"))
        self._cache_scan_checkbox.setChecked(True)
        self._cache_scan_checkbox.setToolTip(
            _("Keep the directory listing in memory so repeated searches "
              "with different base IDs are faster. Uncheck to force a fresh "
              "scan. Cache expires automatically after 5 minutes.")
        )
        self._cache_scan_checkbox.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._cache_scan_checkbox.stateChanged.connect(self._on_cache_scan_toggled)
        root.addWidget(self._cache_scan_checkbox)

        # Loading container
        self._loading_container = QWidget()
        self._loading_container.setFixedHeight(20)
        lc_layout = QHBoxLayout(self._loading_container)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._loading_container)

        # Results list
        self._results_list = QListWidget()
        self._results_list.setStyleSheet(
            f"QListWidget {{ color: {AppStyle.FG_COLOR}; "
            f"background: {AppStyle.BG_COLOR}; "
            f"border: 1px solid {AppStyle.BORDER_COLOR}; }}"
        )
        self._results_list.itemDoubleClicked.connect(self._on_result_double_click)
        root.addWidget(self._results_list, 1)

        # ==============================================================
        # Keyboard shortcuts
        # ==============================================================
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)
        QShortcut(QKeySequence("Ctrl+B"), self).activated.connect(self.pick_file)
        QShortcut(QKeySequence("Ctrl+G"), self).activated.connect(self.go_to_last_moved)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(
            self.get_current_media_filename
        )
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(self.find_related_files)
        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(
            self.extract_and_set_base_id
        )
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(
            self.browse_target_directory
        )

        # Focus the search entry
        QTimer.singleShot(1, self._search_entry.setFocus)

    @property
    def _app_actions(self) -> AppActions:
        """Context of the AppWindow the user last worked in.

        This dialog can stay open while the user switches app windows, so the
        construction-time app_actions can go stale; resolve at request time
        and keep the stored value only as a fallback.
        """
        from ui.app_window.window_manager import WindowManager
        win = WindowManager.get_active_window()
        if win is not None:
            return win.app_actions
        return self._fallback_app_actions

    # ==================================================================
    # Go To File actions
    # ==================================================================
    def go_to_file(self) -> None:
        search_text = self._search_entry.text().strip()
        if not search_text:
            self._app_actions.warn(
                _("Invalid search string, please enter some text.")
            )
            return

        GoToFile.last_search_text = search_text
        GoToFile.last_use_closest = self._use_closest.isChecked()
        GoToFile.last_use_index = self._use_index.isChecked()
        GoToFile.last_closest_sort_by = SortBy.get(self._sort_by_combo.currentText())

        # Index navigation
        if self._use_index.isChecked():
            try:
                index = int(search_text)
                if self._app_actions.go_to_file_by_index(index):
                    self.close()
                return
            except ValueError:
                self._app_actions.warn(
                    _("Index navigation enabled but input is not a valid number.")
                )
                return

        # Normal file search
        closest_sort_by = (
            GoToFile.last_closest_sort_by if GoToFile.last_use_closest else None
        )
        self._app_actions.go_to_file(
            search_text=search_text,
            exact_match=False,
            closest_sort_by=closest_sort_by,
        )
        self.close()

    def pick_file(self) -> None:
        """Open file picker dialog and go to selected file."""
        filters: list[str] = []
        if config.file_types:
            extensions = " ".join(f"*{ext}" for ext in config.file_types)
            filters.append(f"{_('Supported files')} ({extensions})")
            if config.image_types:
                img_ext = " ".join(f"*{ext}" for ext in config.image_types)
                filters.append(f"{_('Image files')} ({img_ext})")
            if config.enable_videos and config.video_types:
                vid_ext = " ".join(f"*{ext}" for ext in config.video_types)
                filters.append(f"{_('Video files')} ({vid_ext})")
            if config.enable_gifs:
                filters.append(f"{_('GIF files')} (*.gif)")
            if config.enable_pdfs:
                filters.append(f"{_('PDF files')} (*.pdf)")
            if config.enable_svgs:
                filters.append(f"{_('SVG files')} (*.svg)")
            if config.enable_html:
                filters.append(f"{_('HTML files')} (*.html *.htm)")
        filters.append(f"{_('All files')} (*)")

        initial_dir = "."
        if hasattr(self._app_actions, "get_base_dir"):
            initial_dir = self._app_actions.get_base_dir()

        selected, _filter = QFileDialog.getOpenFileName(
            self,
            _("Select file to go to"),
            initial_dir,
            ";;".join(filters),
        )
        if selected:
            self._search_entry.setText(selected)
            self.go_to_file()

    def go_to_last_moved(self) -> None:
        """Set closest search, populate with last moved image, and go."""
        last_moved = MarkedFiles.last_moved_image
        if not last_moved:
            action = FileAction.get_history_action(
                start_index=0, auto=False
            )
            if (
                action
                and getattr(action, "new_files", None)
                and len(action.new_files) > 0
            ):
                last_moved = action.new_files[0]
            else:
                self._app_actions.toast(_("No last moved image found."))
                return

        GoToFile.last_use_closest = True
        self._use_closest.setChecked(True)
        self._toggle_closest_options()
        self._search_entry.setText(last_moved)
        self.go_to_file()

    # ==================================================================
    # Closest / Sort-by toggle
    # ==================================================================
    def _toggle_closest_options(self) -> None:
        self._sort_by_row.setVisible(self._use_closest.isChecked())

    # ==================================================================
    # Related files actions
    # ==================================================================
    def _on_cache_scan_toggled(self, state: int) -> None:
        if not state:
            from files.related_image import clear_base_stem_dir_cache
            target_dir = self._target_dir_entry.text().strip()
            if target_dir:
                clear_base_stem_dir_cache(target_dir)
            for d in (config.directories_to_search_for_related_images or []):
                if d:
                    clear_base_stem_dir_cache(d)

    def _on_filename_changed(self, text: str) -> None:
        """Auto-extract base ID when the search text changes."""
        text = text.strip()
        if text:
            base_stem = extract_filename_base_stem(text)
            if base_stem:
                self._base_id_entry.setText(base_stem)

    def extract_and_set_base_id(self) -> None:
        search_text = self._search_entry.text().strip()
        if not search_text:
            self._app_actions.toast(_("Please enter a filename first."))
            return

        base_stem = extract_filename_base_stem(search_text)
        if base_stem:
            self._base_id_entry.setText(base_stem)
            self._update_status(_("Base ID extracted: {}").format(base_stem))
        else:
            self._app_actions.warn(
                _(
                    "Could not extract base ID from filename. "
                    "Please enter it manually."
                )
            )
            self._update_status(
                _("Could not extract base ID. Please enter it manually."),
                warn=True,
            )

    def browse_target_directory(self) -> None:
        """Open the TargetDirectoryWindow to pick a target directory."""
        from ui.files.target_directory_window_qt import TargetDirectoryWindow

        def on_directory_selected(directory: str) -> None:
            self._target_dir_entry.setText(directory)
            GoToFile.last_target_directory = directory
            GoToFile.save_persisted_data()
            TargetDirectoryWindow.add_recent_directory(directory)

            base_id = self._base_id_entry.text().strip()
            if base_id and os.path.isdir(directory):
                self.find_related_files()
            else:
                self.raise_()
                self.activateWindow()

        TargetDirectoryWindow(
            master=self,
            callback=on_directory_selected,
            initial_dir=(
                self._app_actions.get_base_dir()
                if hasattr(self._app_actions, "get_base_dir")
                else "."
            ),
        ).show()

    def get_current_media_filename(self) -> None:
        """Get the current media filename and populate the fields."""
        try:
            self.start_loading(_("Loading current media and searching..."))
            current_filepath = self._app_actions.get_active_media_filepath()
            if current_filepath:
                self._search_entry.setText(current_filepath)
                # Base ID auto-extracted via textChanged signal

                target_dir = self._target_dir_entry.text().strip()
                if target_dir and os.path.isdir(target_dir):
                    self.find_related_files()
                else:
                    self._app_actions.toast(
                        _(
                            "Current media filename loaded. Select a target "
                            "directory to find related filenames."
                        )
                    )
            else:
                self._app_actions.warn(
                    _("No active media file found in the current window.")
                )
        except Exception as e:
            self._app_actions.warn(
                _("Error getting current media filename: {}").format(str(e))
            )
        finally:
            self.stop_loading()

    def find_related_files(self) -> None:
        base_id = self._base_id_entry.text().strip()
        target_dir = self._target_dir_entry.text().strip()

        if not base_id:
            self._app_actions.toast(
                _("Please enter a base ID or extract it from a filename.")
            )
            return
        if not target_dir:
            self._app_actions.toast(
                _("Please select a target directory to search in.")
            )
            return
        if not os.path.isdir(target_dir):
            self._app_actions.warn(_("Target directory does not exist."))
            return

        # Persist
        GoToFile.last_target_directory = target_dir
        GoToFile.last_base_id = base_id
        GoToFile.save_persisted_data()

        from ui.files.target_directory_window_qt import TargetDirectoryWindow

        TargetDirectoryWindow.add_recent_directory(target_dir)

        matching_files = self._find_matching_files(target_dir, base_id)

        # Update results
        self._results_list.clear()
        if matching_files:
            for fp in matching_files:
                self._results_list.addItem(fp)
            self._update_status(
                _("Found {} related filenames").format(len(matching_files)),
            )
        else:
            self._update_status(
                _('No related filenames found with base ID "{0}" in {1}').format(
                    base_id, target_dir
                ),
                warn=True,
            )

    def _on_result_double_click(self, item) -> None:
        file_path = item.text()
        self._search_entry.setText(file_path)
        self._update_status(
            _("Opening file: {}").format(os.path.basename(file_path)),
        )
        self.go_to_file()

    # ==================================================================
    # File matching
    # ==================================================================
    def _find_matching_files(
        self, target_dir: str, base_id: str, threshold: int = 400_000
    ) -> list[str]:
        """Find files whose name starts with *base_id* in target_dir and
        config.directories_to_search_for_related_images."""
        # Build a deduplicated, validated list of directories to search.
        search_dirs: list[str] = []
        seen_norms: set[str] = set()
        for d in ([target_dir] if target_dir else []) + list(
            config.directories_to_search_for_related_images or []
        ):
            if not d:
                continue
            norm = os.path.normpath(os.path.abspath(d))
            if os.path.isdir(d) and norm not in seen_norms:
                search_dirs.append(d)
                seen_norms.add(norm)

        if not search_dirs:
            return []

        # Per-directory confirmation status for this call.
        needs_conf: dict[str, bool] = {
            os.path.normpath(os.path.abspath(d)): (
                os.path.normpath(os.path.abspath(d)) not in GoToFile.confirmed_directories
            )
            for d in search_dirs
        }
        aborted = [False]

        def _on_threshold(directory: str, file_count: int) -> bool:
            norm = os.path.normpath(os.path.abspath(directory))
            if not needs_conf.get(norm, False):
                # Already confirmed — continue without prompting.
                return True
            from lib.qt_alert import qt_alert
            msg = _(
                "The directory '{0}' contains many files "
                "({1} files found so far). Searching may take "
                "a while. Do you want to proceed?"
            ).format(directory, file_count)
            proceed = qt_alert(self, _("Large Directory"), msg, kind="askyesno")
            if proceed:
                GoToFile._add_confirmed_directory(norm)
                GoToFile.save_persisted_data()
                needs_conf[norm] = False
                return True
            self._app_actions.toast(_("Search cancelled by user."))
            aborted[0] = True
            return False

        try:
            matching = find_files_by_base_stem(
                search_dirs,
                base_id,
                threshold=threshold,
                on_threshold_exceeded=_on_threshold,
                use_cache=self._cache_scan_checkbox.isChecked(),
            )
        except Exception as e:
            self._app_actions.warn(
                _("Error searching directory: {}").format(str(e))
            )
            return []

        if not aborted[0]:
            # Auto-confirm any directory that completed without hitting the threshold.
            changed = False
            for d in search_dirs:
                norm = os.path.normpath(os.path.abspath(d))
                if needs_conf.get(norm, False):
                    GoToFile._add_confirmed_directory(norm)
                    changed = True
            if changed:
                GoToFile.save_persisted_data()

        return matching

    # ==================================================================
    # Loading indicator
    # ==================================================================
    def start_loading(self, message: Optional[str] = None) -> None:
        try:
            if message:
                self._update_status(message)
            if self._loading_bar is None:
                self._loading_bar = QProgressBar()
                self._loading_bar.setMinimum(0)
                self._loading_bar.setMaximum(0)  # indeterminate
                self._loading_bar.setFixedWidth(160)
                self._loading_bar.setFixedHeight(16)
                self._loading_container.layout().addWidget(self._loading_bar)
        except Exception:
            pass

    def stop_loading(
        self, final_message: Optional[str] = None, warn: bool = False
    ) -> None:
        try:
            if self._loading_bar is not None:
                self._loading_bar.deleteLater()
                self._loading_bar = None
            if final_message:
                self._update_status(final_message, warn=warn)
        except Exception:
            pass

    # ==================================================================
    # Update with current media (for reusing existing window)
    # ==================================================================
    def update_with_current_media(self, focus: bool = False) -> None:
        self.raise_()
        if focus:
            self.activateWindow()
        QTimer.singleShot(50, self.get_current_media_filename)

    # ==================================================================
    # Helpers
    # ==================================================================
    def _update_status(self, text: str, *, warn: bool = False) -> None:
        color = "orange" if warn else AppStyle.FG_COLOR
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")

    def close_windows(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        GoToFile._instance = None
        super().closeEvent(event)

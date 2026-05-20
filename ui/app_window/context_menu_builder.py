"""
ContextMenuBuilder -- builds the right-click QMenu.

Constructs a context menu from the current application state and controller methods.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMenu

from files.directory_notes import DirectoryNotes
from ui.files.marked_file_mover_qt import MarkedFiles
from utils.config import config
from utils.media_utils import is_video_file
from utils.logging_setup import get_logger
from utils.translations import I18N

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow

_ = I18N._
logger = get_logger("context_menu_builder")


class ContextMenuBuilder:
    """
    Builds and shows the right-click context menu for the media frame.

    Ported from App.show_context_menu.
    """

    def __init__(self, app_window: AppWindow):
        self._app = app_window

    def show(self, global_pos: QPoint) -> None:
        """Build the context menu and display it at the given position."""
        app = self._app
        image_path = app.media_navigator.get_active_media_filepath()
        if not image_path:
            return

        menu = QMenu(app)
        base_dir = app.get_base_dir()

        # ------------------------------------------------------------------
        # Header: filename (italic, disabled)
        # ------------------------------------------------------------------
        header = menu.addAction(os.path.basename(image_path))
        header.setEnabled(False)
        italic_font = QFont()
        italic_font.setItalic(True)
        header.setFont(italic_font)
        menu.addSeparator()

        # ------------------------------------------------------------------
        # Inspection
        # ------------------------------------------------------------------
        menu.addAction(
            _("View Media Details"),
            lambda: app.window_launcher.open_media_details(),
        )

        menu.addAction(
            _("Hide Media"),
            lambda: app.file_ops_ctrl.hide_current_media(),
        )

        # ------------------------------------------------------------------
        # Marks
        # ------------------------------------------------------------------
        in_marks = image_path in MarkedFiles.file_marks
        menu.addAction(
            _("Remove from Marks") if in_marks else _("Add to Marks"),
            lambda: app.file_marks_ctrl.add_or_remove_mark(),
        )

        # ------------------------------------------------------------------
        # Favorites
        # ------------------------------------------------------------------
        try:
            from ui.files.favorites_window_qt import FavoritesWindow
            in_favorites = image_path in FavoritesWindow.get_favorites(base_dir)
            fav_command = (
                FavoritesWindow.remove_favorite if in_favorites else FavoritesWindow.add_favorite
            )
            menu.addAction(
                _("Remove from Favorites") if in_favorites else _("Add to Favorites"),
                lambda: fav_command(base_dir, image_path, app.notification_ctrl.toast),
            )
        except Exception:
            pass  # Favorites module may not be available

        menu.addSeparator()

        # ------------------------------------------------------------------
        # Directory notes
        # ------------------------------------------------------------------
        in_dir_notes = DirectoryNotes.is_marked_file(base_dir, image_path)
        menu.addAction(
            _("Remove from Directory Notes") if in_dir_notes else _("Add to Directory Notes"),
            lambda: app.window_launcher.toggle_directory_note_mark(),
        )
        menu.addAction(
            _("Edit File Note"),
            lambda: app.window_launcher.edit_file_note(),
        )

        menu.addSeparator()

        # ------------------------------------------------------------------
        # External tools
        # ------------------------------------------------------------------
        menu.addAction(
            _("Open in GIMP"),
            lambda: app.file_ops_ctrl.open_image_in_gimp(),
        )
        menu.addAction(
            _("Run Image Generation"),
            lambda: app.search_ctrl.trigger_image_generation(),
        )
        menu.addAction(
            _("Run Image Generation on Directory"),
            lambda: app.search_ctrl.run_image_generation_on_directory(),
        )

        menu.addSeparator()

        # ------------------------------------------------------------------
        # Prevalidations
        # ------------------------------------------------------------------
        prevalidations_toggle_text = (
            _("Disable Prevalidations")
            if config.enable_prevalidations
            else _("Enable Prevalidations")
        )
        menu.addAction(
            _("Run Prevalidations on Directory"),
            lambda: app.window_launcher.run_prevalidations_for_base_dir(),
        )
        menu.addAction(
            prevalidations_toggle_text,
            lambda: app.window_launcher.toggle_prevalidations(),
        )

        menu.addSeparator()

        # ------------------------------------------------------------------
        # Related images
        # ------------------------------------------------------------------
        menu.addAction(
            _("Show Source Media"),
            lambda: app.window_launcher.show_related_media(),
        )
        menu.addAction(
            _("Find Related Media"),
            lambda: app.search_ctrl.find_related_media_in_open_window(),
        )
        menu.addAction(
            _("Set Marks from Downstream Related Images"),
            lambda: app.file_marks_ctrl.set_marks_from_downstream_related_images(),
        )

        menu.addSeparator()

        # ------------------------------------------------------------------
        # Search
        # ------------------------------------------------------------------
        menu.addAction(
            _("Set Current Media as Search Target"),
            lambda: app.search_ctrl.set_current_media_run_search(),
        )
        menu.addAction(
            _("Add Current Media to Negative Search"),
            lambda: app.search_ctrl.add_current_media_to_negative_search(),
        )

        menu.addSeparator()

        # ------------------------------------------------------------------
        # File operations
        # ------------------------------------------------------------------
        menu.addAction(
            _("Copy file path"),
            lambda: app.file_ops_ctrl.copy_media_path(),
        )
        menu.addAction(
            _("Copy file name"),
            lambda: app.file_ops_ctrl.copy_media_basename(),
        )
        menu.addAction(
            _("Open file location"),
            lambda: app.file_ops_ctrl.open_media_location(),
        )

        if is_video_file(image_path):
            menu.addAction(
                _("Save copy without audio"),
                lambda: app.file_ops_ctrl.strip_audio_from_current_video(),
            )
            menu.addAction(
                _("Save copy without metadata"),
                lambda: app.file_ops_ctrl.copy_current_video_without_metadata(),
            )

        menu.addAction(
            _("Convert directory images to JPG"),
            lambda: app.file_ops_ctrl.convert_directory_images_to_jpg(),
        )
        if config.enable_svgs:
            menu.addAction(
                _("Convert directory SVGs to PNG"),
                lambda: app.file_ops_ctrl.convert_directory_svg_to_png(),
            )
        if config.enable_videos: 
            menu.addAction(
                _("Save copies of all videos in directory without metadata"),
                lambda: app.file_ops_ctrl.copy_directory_videos_without_metadata(),
            )

        menu.addSeparator()

        menu.addAction(
            _("Randomize filenames"),
            lambda: app.file_ops_ctrl.run_randomize_filenames(),
        )
        menu.addAction(
            _("Run Refacdir"),
            lambda: app.file_ops_ctrl.run_refacdir(),
        )

        menu.addSeparator()

        # Delete (last, with visual separation)
        menu.addAction(
            _("Delete"),
            lambda: app.file_ops_ctrl.delete_media(),
        )

        menu.exec(global_pos)

"""
WindowLauncher -- opens secondary windows and dialogs.

Each method constructs and shows the appropriate dialog or secondary window,
wiring AppActions and password gates as needed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from ui.auth.password_utils import require_password, check_session_expired
from utils.constants import ClassifierActionType, Mode, ProtectedActions
from utils.logging_setup import get_logger, set_logger_level
from utils.translations import _, compare_running_warn

if TYPE_CHECKING:
    from ui.app_window.app_window import AppWindow
logger = get_logger("window_launcher")


class WindowLauncher:
    """
    Opens every secondary window / dialog. Keeps the "which windows exist"
    knowledge in one place and makes window implementations easy to swap.
    """

    def __init__(self, app_window: AppWindow):
        self._app = app_window
        self._go_to_file_window = None
        self._directory_notes_window = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _handle_error(self, error: Exception, title: str = "Window Error") -> None:
        self._app.notification_ctrl.handle_error(str(error), title=title)

    # ------------------------------------------------------------------
    # Navigation windows
    # ------------------------------------------------------------------
    def open_go_to_file_window(self, event=None) -> None:
        """Open the go-to-file search window."""
        try:
            if self._go_to_file_window is not None:
                try:
                    if self._go_to_file_window.isVisible():
                        self._go_to_file_window.raise_()
                        self._go_to_file_window.activateWindow()
                        return
                except (RuntimeError, AttributeError):
                    self._go_to_file_window = None

            from ui.files.go_to_file_qt import GoToFile
            self._go_to_file_window = GoToFile(self._app, self._app.app_actions)
            self._go_to_file_window.show()
        except Exception as e:
            self._handle_error(e, "Go To File Window Error")

    def open_go_to_file_with_current_media(self, event=None) -> None:
        """Open go-to-file pre-populated with the current media name."""
        try:
            if self._go_to_file_window is not None:
                try:
                    if self._go_to_file_window.isVisible():
                        self._go_to_file_window.update_with_current_media()
                        return
                except (RuntimeError, AttributeError):
                    self._go_to_file_window = None

            from ui.files.go_to_file_qt import GoToFile
            self._go_to_file_window = GoToFile(self._app, self._app.app_actions)
            self._go_to_file_window.show()
            self._go_to_file_window.update_with_current_media(focus=True)
        except Exception as e:
            self._handle_error(e, "Go To File Window Error")

    # ------------------------------------------------------------------
    # Directory windows
    # ------------------------------------------------------------------
    def open_recent_directory_window(
        self,
        event=None,
        open_gui: bool = True,
        run_compare_media: Optional[str] = None,
        extra_callback_args: Optional[list[Any]] = None,
    ) -> None:
        """Open the recent directories window."""
        try:
            from ui.files.recent_directory_window_qt import RecentDirectoryWindow

            window = RecentDirectoryWindow(
                self._app,
                open_gui,
                self._app.app_actions,
                base_dir=self._app.get_base_dir(),
                run_compare_media=run_compare_media,
                extra_callback_args=extra_callback_args,
            )
            window.show()
        except Exception as e:
            self._handle_error(e, "Recent Directory Window Error")

    def open_favorites_window(self, event=None) -> None:
        """Open the favorites directory window."""
        try:
            from ui.files.favorites_window_qt import FavoritesWindow
            FavoritesWindow.show_window(self._app, self._app.app_actions)
        except Exception as e:
            self._handle_error(e, "Favorites Window Error")

    def open_embedding_seed_library_window(self, event=None) -> None:
        """Open the embedding seed library window."""
        try:
            from ui.compare.embedding_seed_library_window_qt import EmbeddingSeedLibraryWindow
            EmbeddingSeedLibraryWindow.show_window(self._app, self._app.app_actions)
        except Exception as e:
            self._handle_error(e, "Embedding Seed Library Window Error")

    def save_supergroup_as_embedding_seed(self, seed_data: dict) -> None:
        """
        Open the create dialog to save a supergroup centroid (resolved by
        CompareManager.get_current_supergroup_seed_data) as a new named
        embedding seed. See docs/embedding-seed-library.md, section 5.1.
        """
        try:
            from datetime import datetime
            from compare.embedding_seed import EmbeddingSeed
            from ui.compare.embedding_seed_edit_window_qt import EmbeddingSeedEditWindow

            base_dir = seed_data.get("base_dir") or self._app.get_base_dir() or ""
            default_name = _("Supergroup from {0} ({1})").format(
                os.path.basename(os.path.normpath(base_dir)) if base_dir else "",
                datetime.now().strftime("%Y-%m-%d"),
            )
            pending = EmbeddingSeed(
                name=default_name,
                positive=seed_data["vector"],
                embedding_model=seed_data["compare_mode"],
                embedding_dim=len(seed_data["vector"]),
                source={
                    "kind": "supergroup_centroid",
                    "compare_mode": seed_data["compare_mode"],
                    "member_count": seed_data["member_count"],
                    "group_indexes": seed_data["group_indexes"],
                    "directory": base_dir or None,
                },
            )
            EmbeddingSeedEditWindow(
                self._app, self._app.app_actions, lambda: None, pending_seed=pending
            ).show()
        except Exception as e:
            self._handle_error(e, "Save Embedding Seed Error")

    def save_current_media_as_embedding_seed(self, media_path: str) -> None:
        """
        Open the create dialog for a new embedding seed captured from a
        single file. The user picks which embedding architecture to use
        (independent of whatever compare mode is currently active, via
        compare.embedding_capture.embedding_capture_modes()) and the actual
        embedding computation is deferred until the dialog is confirmed.
        See docs/embedding-seed-library.md, section 5.4.
        """
        try:
            from datetime import datetime
            from ui.compare.embedding_seed_edit_window_qt import EmbeddingSeedEditWindow

            default_name = _("{0} ({1})").format(
                os.path.basename(media_path), datetime.now().strftime("%Y-%m-%d")
            )
            EmbeddingSeedEditWindow(
                self._app,
                self._app.app_actions,
                lambda: None,
                pending_media_path=media_path,
                default_name=default_name,
                default_compare_mode=self._app.compare_manager.compare_mode,
            ).show()
        except Exception as e:
            self._handle_error(e, "Save Embedding Seed Error")

    def open_related_images_window(self, event=None) -> None:
        """Open the related images actions window."""
        try:
            from ui.files.related_images_window_qt import RelatedImagesWindow
            window = RelatedImagesWindow(self._app)
            window.show()
        except Exception as e:
            self._handle_error(e, "Related Images Window Error")

    def open_directory_notes_window(self, event=None) -> None:
        """Open the directory notes window for the current base directory."""
        try:
            base_dir = self._app.get_base_dir()
            if not base_dir:
                self._app.notification_ctrl.toast(_("Please set a base directory first"))
                return

            # Re-use existing window if still open for the same directory.
            # If the directory changed, close the stale window and open a fresh one.
            if self._directory_notes_window is not None:
                try:
                    if self._directory_notes_window.isVisible():
                        if self._directory_notes_window._base_dir == base_dir:
                            self._directory_notes_window.raise_()
                            self._directory_notes_window.activateWindow()
                            return
                        else:
                            self._directory_notes_window.close()
                except (RuntimeError, AttributeError):
                    pass
                self._directory_notes_window = None

            from ui.files.directory_notes_window_qt import DirectoryNotesWindow
            self._directory_notes_window = DirectoryNotesWindow(
                self._app, self._app.app_actions, base_dir
            )
            self._directory_notes_window.show()
        except Exception as e:
            self._handle_error(e, "Directory Notes Window Error")

    # ------------------------------------------------------------------
    # Settings / configuration windows
    # ------------------------------------------------------------------
    def open_compare_settings_window(self, event=None) -> None:
        """Open the compare settings window."""
        try:
            from ui.compare.compare_settings_window_qt import CompareSettingsWindow
            CompareSettingsWindow.open(
                parent=self._app,
                compare_manager=self._app.compare_manager,
                set_file_filter=self._app.sidebar_panel.file_filter_entry.setText,
            )
        except Exception as e:
            self._handle_error(e, "Compare Settings Window Error")

    @require_password(ProtectedActions.EDIT_PREVALIDATIONS)
    def open_hf_model_manager_window(self, event=None) -> None:
        """Open the HF Hub model manager window."""
        try:
            from ui.compare.hf_model_manager_window_qt import HfModelManagerWindow
            HfModelManagerWindow.show_window(self._app, self._app.app_actions)
        except Exception as e:
            self._handle_error(e, "HF Hub Model Manager Window Error")

    @require_password(ProtectedActions.CONFIGURE_MEDIA_TYPES)
    def open_type_configuration_window(self, event=None) -> None:
        """Open the file type configuration window."""
        from ui.files.type_configuration_window_qt import TypeConfigurationWindow
        TypeConfigurationWindow.show(master=self._app, app_actions=self._app.app_actions)

    @require_password(ProtectedActions.EDIT_PREVALIDATIONS)
    def open_prevalidations_window(self, event=None) -> None:
        """Open the classifier management window, restoring the last-used tab.

        The first-time default is Prevalidations (tab 2), set via the cache
        default in ClassifierManagementWindow.__init__.  Subsequent opens
        restore whatever tab the user was on when they last closed the window.
        """
        try:
            from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
            ClassifierManagementWindow.show_window(self._app, self._app.app_actions)
        except Exception as e:
            self._handle_error(e, "Prevalidations Window Error")

    @require_password(ProtectedActions.EDIT_PREVALIDATIONS, ProtectedActions.RUN_PREVALIDATIONS)
    def open_classifier_actions_window(self, event=None) -> None:
        """Open the classifier management window (classifier actions tab)."""
        try:
            from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
            ClassifierManagementWindow.show_window(self._app, self._app.app_actions)
            mgmt = ClassifierManagementWindow._instance
            if mgmt and hasattr(mgmt, '_tabs'):
                mgmt._tabs.setCurrentIndex(0)
        except Exception as e:
            self._handle_error(e, "Classifier Actions Window Error")

    # ------------------------------------------------------------------
    # File operations windows
    # ------------------------------------------------------------------
    def open_file_actions_window(self, event=None) -> None:
        """Open the file actions window."""
        try:
            from files.marked_files import MarkedFiles
            from ui.files.file_actions_window_qt import FileActionsWindow
            from ui.image.media_details import MediaDetails
            window = FileActionsWindow(
                self._app,
                self._app.app_actions,
                MediaDetails.open_temp_media_canvas,
                MarkedFiles.move_marks_to_dir_static,
            )
            window.show()
        except Exception as e:
            self._handle_error(e, "File Actions Window Error")

    @require_password(ProtectedActions.VIEW_FILE_ACTIONS)
    def open_file_action_sets_window(self, event=None) -> None:
        """Open the file action sets window."""
        try:
            from files.marked_files import MarkedFiles
            from ui.files.file_action_sets_window_qt import FileActionSetsWindow
            if FileActionSetsWindow._instance is not None:
                try:
                    if FileActionSetsWindow._instance.isVisible():
                        if FileActionSetsWindow._instance.isMinimized():
                            FileActionSetsWindow._instance.showNormal()
                        FileActionSetsWindow._instance.raise_()
                        FileActionSetsWindow._instance.activateWindow()
                        return
                    else:
                        FileActionSetsWindow._instance = None
                except (RuntimeError, AttributeError):
                    FileActionSetsWindow._instance = None
            window = FileActionSetsWindow(
                self._app,
                self._app.app_actions,
                MarkedFiles.move_marks_to_dir_static,
            )
            window.show()
        except Exception as e:
            self._handle_error(e, "File Action Sets Window Error")

    # ------------------------------------------------------------------
    # Auth / admin windows
    # ------------------------------------------------------------------
    @require_password(ProtectedActions.ACCESS_ADMIN)
    def open_password_admin_window(self, event=None) -> None:
        """Open the password administration window."""
        try:
            from ui.auth.password_admin_window import PasswordAdminWindow
            PasswordAdminWindow(self._app, self._app.app_actions)
        except Exception as e:
            self._handle_error(e, "Password Admin Window Error")

    # ------------------------------------------------------------------
    # Info windows
    # ------------------------------------------------------------------
    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def open_media_details(
        self,
        event=None,
        media_path: Optional[str] = None,
        manually_keyed: bool = True,
    ) -> None:
        """
        Open the media details / metadata inspector window.

        Manages the singleton MediaDetails window reference stored on AppActions.
        """
        from ui.image.media_details import MediaDetails

        app_actions = self._app.app_actions

        # Close existing window if the session expired
        if app_actions.media_details_window() is not None:
            if check_session_expired(ProtectedActions.VIEW_MEDIA_DETAILS):
                app_actions.media_details_window().close_windows()
                app_actions.set_media_details_window(None)

        preset_media_path = True
        if media_path is None:
            media_path = self._app.media_path
            preset_media_path = False

        if not media_path:
            return

        # Build index text
        if preset_media_path:
            index_text = _("(Open this media as part of a directory to see index details.)")
        elif self._app.mode == Mode.BROWSE or self._app.is_compare_running():
            index_text = self._app.file_browser.get_index_details()
        else:
            cm = self._app.compare_manager
            _index = cm.match_index + 1
            len_matched = len(cm.files_matched)
            if self._app.mode == Mode.GROUP:
                len_groups = len(cm.file_groups)
                group_idx = cm.current_group_index + 1
                index_text = _("{0} of {1} (Group {2} of {3})").format(
                    _index, len_matched, group_idx, len_groups)
            elif self._app.mode == Mode.SEARCH and self._app.is_toggled_view_matches:
                index_text = _("{0} of {1} ({2})").format(
                    _index, len_matched, self._app.file_browser.get_index_details())
            elif self._app.mode == Mode.GROUP_COMPLEMENT:
                index_text = _("{0} of {1} (Ungrouped)").format(_index, len_matched)
            else:
                index_text = ""

        existing = app_actions.media_details_window()
        if existing is not None and not existing.has_closed:
            if existing.do_refresh:
                existing.update_media_details(media_path, index_text)
            if manually_keyed:
                existing.focus()
            else:
                # Keep browsing focus in the main app while details auto-refreshes.
                self._app.refocus()
        else:
            try:
                details_win = MediaDetails(
                    self._app, media_path, index_text,
                    app_actions, do_refresh=not preset_media_path,
                    take_focus=manually_keyed,
                )
                details_win.show()
                app_actions.set_media_details_window(details_win)
            except Exception as e:
                self._handle_error(e, "Image Details Error")

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def copy_prompt(self, event=None) -> None:
        """Copy the AI prompt from the currently viewed image."""
        from ui.image.media_details import MediaDetails
        MediaDetails.copy_prompt_no_break_static(
            self._app.media_navigator.get_active_media_filepath(),
            self._app,
            self._app.app_actions,
        )

    @require_password(ProtectedActions.VIEW_MEDIA_DETAILS)
    def show_related_media(self, event=None) -> None:
        """Show a related media file to the current one."""
        from ui.image.media_details import MediaDetails
        MediaDetails.show_related_image(
            master=self._app,
            image_path=self._app.media_path,
            app_actions=self._app.app_actions,
        )

    def _setup_static_selection(self, media_path: str, video_launch_fn) -> Optional[tuple]:
        """Shared preamble for interactive selection actions (crop / box / freeform).

        Validates media-type support, tears down any previous QGraphicsView
        crop/polygon-selection session, and either dispatches to *video_launch_fn*
        for video (returning None) or resolves the pixel-space source path for
        static media (image, animated GIF, SVG, PDF).

        Returns ``(gv, media_frame, source_path, media_type, restore_original)``
        for static media, or ``None`` if handled elsewhere / unsupported / on error.
        """
        import os
        import tempfile
        from PIL import Image
        from image.frame_cache import FrameCache
        from utils.media_utils import get_media_type_for_path
        from utils.constants import MediaType

        if not media_path:
            return None
        media_type = get_media_type_for_path(media_path)
        if not media_type.is_interactive_crop_supported():
            self._app.notification_ctrl.toast(_("Interactive crop not supported for this media type"))
            return None

        media_frame = self._app.media_frame
        gv = media_frame._graphics_view

        # Tear down any previous QGraphicsView crop/polygon session cleanly.
        # PySide6 emits RuntimeWarning via warnings.warn() (not raise) when
        # disconnecting a signal with no connections, so suppress at that level.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for sig in (gv.crop_confirmed, gv.crop_cancelled, gv.polygon_confirmed):
                try:
                    sig.disconnect()
                except RuntimeError:
                    pass

        is_animated_gif = (media_type == MediaType.GIF and media_frame._gif_is_animated)

        # Video: handled by a separate overlay module — see video_crop_overlay_qt.py.
        if media_type.is_video():
            video_launch_fn(media_frame, media_path, self._app)
            return None

        # -------------------------------------------------------------------
        # Static media (image, animated GIF, SVG, PDF): QGraphicsView selection modes
        # -------------------------------------------------------------------
        if is_animated_gif:
            media_frame.pause_video_if_playing()
            tmp_frame = os.path.join(tempfile.gettempdir(), "weidr_gif_crop_frame.png")
            try:
                with Image.open(media_path) as _img:
                    _img.seek(0)
                    _img.copy().save(tmp_frame, format="PNG")
            except Exception as e:
                self._app.notification_ctrl.toast(_("Could not read GIF frame: ") + str(e))
                return None
            media_frame._show_image_in_view(tmp_frame)
            source_path = media_path
        else:
            # For PDFs, use the currently displayed page rather than always page 0.
            if media_type == MediaType.PDF:
                source_path = (self._app.media_frame.pdf_current_page_path()
                               or FrameCache.get_image_path(media_path))
            else:
                source_path = FrameCache.get_image_path(media_path)

        def _restore_original():
            if is_animated_gif:
                media_frame.show_media(media_path)

        return gv, media_frame, source_path, media_type, _restore_original

    def _finish_static_edit(
        self,
        new_path: str,
        media_path: str,
        media_type,
        output_suffix: str,
        success_msg: str,
        failed_msg: str,
    ) -> None:
        """Shared tail for crop/box/background-box/freeform actions on static media:
        fix up the SVG/PDF sibling extension if needed, then open the result or
        report failure. *new_path* is whatever apply_fn returned.
        """
        import os
        from utils.constants import MediaType
        from utils.utils import Utils

        if media_type in (MediaType.SVG, MediaType.PDF) and new_path and os.path.exists(new_path):
            # SVG/PDF sources are rendered to a raster image for editing (PDF's
            # rendered page is JPEG, SVG's is PNG -- see pdf_current_page_path /
            # FrameCache.get_image_path), and apply_fn writes a real image file
            # in that same format. os.replace() below is a plain rename, not a
            # re-encode, so the sibling must keep apply_fn's actual output
            # extension -- not adopt the original .pdf/.svg extension, which
            # would just mislabel that image as a format it isn't (PDFium then
            # rejects it with a "Data format error" when trying to open it).
            image_ext = os.path.splitext(new_path)[1] or ".png"
            image_stem = os.path.splitext(media_path)[0] + image_ext
            sibling = Utils.unique_sibling_path(image_stem, output_suffix)
            try:
                os.replace(new_path, sibling)
                new_path = sibling
            except OSError:
                pass

        if new_path and os.path.exists(new_path):
            self._app.app_actions.refresh()
            self._app.app_actions.success(success_msg)
            from ui.image.media_details import MediaDetails
            MediaDetails.open_temp_media_canvas(
                master=self._app, media_path=new_path,
                app_actions=self._app.app_actions,
            )
        else:
            self._app.notification_ctrl.toast(failed_msg)

    def _preview_and_confirm_fill(
        self,
        source_path: str,
        fill_size: tuple,
        render_fn,
    ) -> Optional[Any]:
        """Generate a candidate random fill, render a preview of the composited
        result to a temp file and show it, and let the user reroll (regenerate
        the fill and refresh the preview), switch to a plain black/white fill,
        or accept/cancel before the caller commits a real save.

        *fill_size* is the (width, height) to generate the candidate fill at.
        *render_fn* receives (fill_image, output_path) and should write the
        composited preview to *output_path* (typically a closure over apply_fn
        and the already-computed selection geometry).

        Returns the accepted fill image, or None if the user cancelled.
        """
        import tempfile
        from PIL import Image
        from image.image_ops import ImageOps
        from lib.fill_preview_dialog_qt import show_fill_preview_dialog

        ext = os.path.splitext(source_path)[1] or ".png"
        preview_path = os.path.join(tempfile.gettempdir(), "weidr_fill_preview" + ext)

        fill_holder = [ImageOps.generate_box_fill_image(*fill_size)]

        def _render():
            render_fn(fill_holder[0], preview_path)

        def _reroll():
            fill_holder[0] = ImageOps.generate_box_fill_image(*fill_size)
            _render()

        def _set_solid(color: tuple):
            fill_holder[0] = Image.new("RGB", fill_size, color)
            _render()

        _render()
        try:
            accepted = show_fill_preview_dialog(self._app, preview_path, _reroll, _set_solid)
        finally:
            try:
                os.remove(preview_path)
            except OSError:
                pass
        return fill_holder[0] if accepted else None

    def _run_static_rect_action(
        self,
        gv,
        media_frame,
        media_path: str,
        source_path: str,
        media_type,
        restore_original,
        *,
        apply_fn,
        output_suffix: str,
        too_small_msg: str,
        success_msg: str,
        failed_msg: str,
        preview_fill: bool = False,
        fill_covers_full_image: bool = False,
    ) -> None:
        """Wire crop_confirmed/crop_cancelled for one rectangle-selection action
        (crop / box / background box) on static media. *apply_fn* receives
        (source_path, left, upper, right, lower) and returns the new file path
        (or "" on failure).

        When *preview_fill* is True (box / background box only -- crop has no
        fill), *apply_fn* is also called with ``fill_image=``/``output_path=``
        kwargs so a candidate fill can be previewed and rerolled before the
        real save; *fill_covers_full_image* selects whether the previewed fill
        is sized to the selection rect (box) or the whole image (background
        box, which paints everywhere outside the rect).
        """
        def _disconnect_signals():
            for sig, slot in ((gv.crop_confirmed, _on_confirmed), (gv.crop_cancelled, _on_cancelled)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, RuntimeWarning):
                    pass

        def _on_cancelled():
            _disconnect_signals()
            restore_original()

        def _on_confirmed(rect):
            _disconnect_signals()
            gv.end_crop_mode()

            pixmap = media_frame._current_pixmap
            if pixmap is None or pixmap.isNull():
                restore_original()
                return
            pix_w, pix_h = pixmap.width(), pixmap.height()
            img_w = media_frame.imwidth if media_frame.imwidth > 0 else pix_w
            img_h = media_frame.imheight if media_frame.imheight > 0 else pix_h
            scale_x = img_w / pix_w if pix_w > 0 else 1.0
            scale_y = img_h / pix_h if pix_h > 0 else 1.0
            left  = max(0,     int(rect.x() * scale_x))
            upper = max(0,     int(rect.y() * scale_y))
            right = min(img_w, int((rect.x() + rect.width())  * scale_x))
            lower = min(img_h, int((rect.y() + rect.height()) * scale_y))
            if right <= left or lower <= upper:
                self._app.notification_ctrl.toast(too_small_msg)
                restore_original()
                return

            if preview_fill:
                fill_size = (img_w, img_h) if fill_covers_full_image else (right - left, lower - upper)
                fill_image = self._preview_and_confirm_fill(
                    source_path,
                    fill_size,
                    lambda fill, out_path: apply_fn(
                        source_path, left, upper, right, lower, fill_image=fill, output_path=out_path
                    ),
                )
                if fill_image is None:
                    restore_original()
                    return
                new_path = apply_fn(source_path, left, upper, right, lower, fill_image=fill_image)
            else:
                new_path = apply_fn(source_path, left, upper, right, lower)
            self._finish_static_edit(new_path, media_path, media_type, output_suffix, success_msg, failed_msg)

        gv.crop_confirmed.connect(_on_confirmed)
        gv.crop_cancelled.connect(_on_cancelled)
        gv.start_crop_mode()

    def _run_static_polygon_action(
        self,
        gv,
        media_frame,
        media_path: str,
        source_path: str,
        media_type,
        restore_original,
        *,
        apply_fn,
        output_suffix: str,
        too_small_msg: str,
        success_msg: str,
        failed_msg: str,
        preview_fill: bool = False,
    ) -> None:
        """Wire polygon_confirmed/crop_cancelled for one freeform-selection action
        (box / background box) on static media. *apply_fn* receives
        (source_path, points) where points is a list of (x, y) pixel-space
        coordinates, and returns the new file path (or "" on failure).

        When *preview_fill* is True (box / background box only -- crop has no
        fill), *apply_fn* is also called with ``fill_image=``/``output_path=``
        kwargs so a candidate fill can be previewed and rerolled before the
        real save. Unlike the rect case, the fill always covers the whole
        image here -- both freeform box and background-box composite via a
        full-image mask (see draw_box_at_polygon / draw_background_box_at_polygon).
        """
        def _disconnect_signals():
            for sig, slot in ((gv.polygon_confirmed, _on_confirmed), (gv.crop_cancelled, _on_cancelled)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, RuntimeWarning):
                    pass

        def _on_cancelled():
            _disconnect_signals()
            restore_original()

        def _on_confirmed(scene_points):
            _disconnect_signals()
            gv.end_polygon_mode()

            pixmap = media_frame._current_pixmap
            if pixmap is None or pixmap.isNull():
                restore_original()
                return
            pix_w, pix_h = pixmap.width(), pixmap.height()
            img_w = media_frame.imwidth if media_frame.imwidth > 0 else pix_w
            img_h = media_frame.imheight if media_frame.imheight > 0 else pix_h
            scale_x = img_w / pix_w if pix_w > 0 else 1.0
            scale_y = img_h / pix_h if pix_h > 0 else 1.0
            points = [
                (
                    max(0, min(img_w, int(pt.x() * scale_x))),
                    max(0, min(img_h, int(pt.y() * scale_y))),
                )
                for pt in scene_points
            ]
            if len(points) < 3:
                self._app.notification_ctrl.toast(too_small_msg)
                restore_original()
                return

            if preview_fill:
                fill_image = self._preview_and_confirm_fill(
                    source_path,
                    (img_w, img_h),
                    lambda fill, out_path: apply_fn(
                        source_path, points, fill_image=fill, output_path=out_path
                    ),
                )
                if fill_image is None:
                    restore_original()
                    return
                new_path = apply_fn(source_path, points, fill_image=fill_image)
            else:
                new_path = apply_fn(source_path, points)
            self._finish_static_edit(new_path, media_path, media_type, output_suffix, success_msg, failed_msg)

        gv.polygon_confirmed.connect(_on_confirmed)
        gv.crop_cancelled.connect(_on_cancelled)
        gv.start_polygon_mode()

    def interactive_crop(self, event=None) -> None:
        """Enter interactive crop mode for the current media (image, GIF, SVG, PDF, or video)."""
        from image.image_ops import ImageOps
        from ui.app_window.video_crop_overlay_qt import launch_video_crop

        ctx = self._setup_static_selection(self._app.media_path, launch_video_crop)
        if ctx is None:
            return
        gv, media_frame, source_path, media_type, restore_original = ctx
        self._run_static_rect_action(
            gv, media_frame, self._app.media_path, source_path, media_type, restore_original,
            apply_fn=ImageOps.crop_image_to_rect,
            output_suffix="_crop",
            too_small_msg=_("Crop selection too small"),
            success_msg=_("Cropped"),
            failed_msg=_("Crop failed"),
        )
        self._app.notification_ctrl.toast(_("Drag to select crop, Enter to confirm, Escape to cancel"))

    def interactive_box(self, event=None) -> None:
        """Enter interactive box mode for the current media (image, GIF, SVG, PDF, or video):
        paint a random color/pattern box over a selected rectangle, reusing the same
        rect-selection handlers as :meth:`interactive_crop`."""
        from image.image_ops import ImageOps
        from ui.app_window.video_crop_overlay_qt import launch_video_box

        ctx = self._setup_static_selection(self._app.media_path, launch_video_box)
        if ctx is None:
            return
        gv, media_frame, source_path, media_type, restore_original = ctx
        self._run_static_rect_action(
            gv, media_frame, self._app.media_path, source_path, media_type, restore_original,
            apply_fn=ImageOps.draw_box_at_rect,
            output_suffix="_box",
            too_small_msg=_("Box selection too small"),
            success_msg=_("Box added"),
            failed_msg=_("Box draw failed"),
            preview_fill=True,
        )
        self._app.notification_ctrl.toast(_("Drag to select box location, Enter to confirm, Escape to cancel"))

    def interactive_background_box(self, event=None) -> None:
        """Enter interactive background-box mode for the current media (image, GIF,
        SVG, PDF, or video): paint a random color/pattern fill over everything
        *outside* a selected rectangle, leaving the selection itself unchanged.
        Reuses the same rect-selection handlers as :meth:`interactive_crop`."""
        from image.image_ops import ImageOps
        from ui.app_window.video_crop_overlay_qt import launch_video_background_box

        ctx = self._setup_static_selection(self._app.media_path, launch_video_background_box)
        if ctx is None:
            return
        gv, media_frame, source_path, media_type, restore_original = ctx
        self._run_static_rect_action(
            gv, media_frame, self._app.media_path, source_path, media_type, restore_original,
            apply_fn=ImageOps.draw_background_box_at_rect,
            output_suffix="_bgbox",
            too_small_msg=_("Selection too small"),
            success_msg=_("Background box added"),
            failed_msg=_("Background box draw failed"),
            preview_fill=True,
            fill_covers_full_image=True,
        )
        self._app.notification_ctrl.toast(_("Drag to select area to keep, Enter to confirm, Escape to cancel"))

    def _freeform_video_unsupported(self, media_frame, media_path, app) -> None:
        app.notification_ctrl.toast(_("Freeform selection is not supported for video"))

    def interactive_crop_freeform(self, event=None) -> None:
        """Enter freeform polygon crop mode for the current media (image, GIF,
        SVG, or PDF -- video is not supported). Click to add points tracing an
        outline (or click-drag to sweep); close the same way as the other
        freeform actions to crop to the polygon's bounding box, with everything
        outside the polygon itself made transparent. Requires an alpha channel,
        so the output is saved as PNG (static) or animated WebP (GIF) rather
        than matching the source format -- see
        :meth:`image.image_ops.ImageOps.crop_image_to_polygon`. A separate
        option from the rectangle-based :meth:`interactive_crop`, which is left
        unchanged."""
        from image.image_ops import ImageOps

        ctx = self._setup_static_selection(self._app.media_path, self._freeform_video_unsupported)
        if ctx is None:
            return
        gv, media_frame, source_path, media_type, restore_original = ctx
        self._run_static_polygon_action(
            gv, media_frame, self._app.media_path, source_path, media_type, restore_original,
            apply_fn=ImageOps.crop_image_to_polygon,
            output_suffix="_crop",
            too_small_msg=_("Need at least 3 points"),
            success_msg=_("Cropped"),
            failed_msg=_("Crop failed"),
        )
        self._app.notification_ctrl.toast(
            _("Click or click-drag to add points, click near the start to close, Enter to confirm, Escape to cancel")
        )

    def interactive_box_freeform(self, event=None) -> None:
        """Enter freeform polygon box mode for the current media (image, GIF, SVG,
        or PDF -- video is not supported). Click to add points tracing an outline;
        click back near the first point (or press Enter with 3+ points placed,
        Backspace to undo the last point) to close and paint a random
        color/pattern fill inside it. A separate option from the rectangle-based
        :meth:`interactive_box`, which is left unchanged."""
        from image.image_ops import ImageOps

        ctx = self._setup_static_selection(self._app.media_path, self._freeform_video_unsupported)
        if ctx is None:
            return
        gv, media_frame, source_path, media_type, restore_original = ctx
        self._run_static_polygon_action(
            gv, media_frame, self._app.media_path, source_path, media_type, restore_original,
            apply_fn=ImageOps.draw_box_at_polygon,
            output_suffix="_box",
            too_small_msg=_("Need at least 3 points"),
            success_msg=_("Box added"),
            failed_msg=_("Box draw failed"),
            preview_fill=True,
        )
        self._app.notification_ctrl.toast(
            _("Click or click-drag to add points, click near the start to close, Enter to confirm, Escape to cancel")
        )

    def interactive_background_box_freeform(self, event=None) -> None:
        """Enter freeform polygon background-box mode for the current media (image,
        GIF, SVG, or PDF -- video is not supported). Click to add points tracing an
        outline; click back near the first point (or press Enter with 3+ points
        placed, Backspace to undo the last point) to close and paint a random
        color/pattern fill everywhere *outside* it. A separate option from the
        rectangle-based :meth:`interactive_background_box`, which is left unchanged."""
        from image.image_ops import ImageOps

        ctx = self._setup_static_selection(self._app.media_path, self._freeform_video_unsupported)
        if ctx is None:
            return
        gv, media_frame, source_path, media_type, restore_original = ctx
        self._run_static_polygon_action(
            gv, media_frame, self._app.media_path, source_path, media_type, restore_original,
            apply_fn=ImageOps.draw_background_box_at_polygon,
            output_suffix="_bgbox",
            too_small_msg=_("Need at least 3 points"),
            success_msg=_("Background box added"),
            failed_msg=_("Background box draw failed"),
            preview_fill=True,
        )
        self._app.notification_ctrl.toast(
            _("Click or click-drag to add points, click near the start to close, Enter to confirm, Escape to cancel")
        )

    def get_help_and_config(self, event=None) -> None:
        """Open the help and configuration window."""
        try:
            from ui.help_and_config_qt import HelpAndConfig
            dialog = HelpAndConfig(parent=self._app, position_parent=self._app)
            dialog.show()
        except Exception as e:
            self._app.notification_ctrl.alert(
                "Help & Config Error", str(e), kind="error"
            )

    # ------------------------------------------------------------------
    # Secondary compare window
    # ------------------------------------------------------------------
    def open_secondary_compare_window(
        self, event=None, run_compare_media: Optional[str] = None
    ) -> None:
        """Open a new secondary window and optionally start a compare."""
        if run_compare_media is None:
            self.open_recent_directory_window(run_compare_media="")
        elif not os.path.isfile(run_compare_media):
            self._app.notification_ctrl.alert(
                _("No media selected"),
                _("No media file was selected for comparison"),
            )
        else:
            self.open_recent_directory_window(run_compare_media=self._app.media_path)

    # ------------------------------------------------------------------
    # Prevalidations (action, not window)
    # ------------------------------------------------------------------
    @require_password(ProtectedActions.RUN_PREVALIDATIONS)
    def run_prevalidations_for_base_dir(self, event=None) -> None:
        """Run all prevalidations on every file in the current directory."""
        from ui.compare.prevalidations_tab_qt import PrevalidationsTab

        fb = self._app.file_browser
        if fb.is_slow_total_files(threshold=100, use_sortable_files=True):
            ok = self._app.notification_ctrl.alert(
                _("Many Files"),
                _("Are you sure you want to run all prevalidations on directory {0} ? "
                  "This may take a while.").format(self._app.get_base_dir()),
                kind="askokcancel",
            )
            if not ok:
                logger.info("User canceled prevalidations task")
                return

        if self._app.is_compare_running():
            self._app.app_actions.warn(compare_running_warn(_("run prevalidations")))
            return

        logger.warning("Running prevalidations for " + self._app.get_base_dir())
        from PySide6.QtWidgets import QApplication
        from files.marked_files import MarkedFiles

        # Same sidebar badge as dynamic (video/GIF/PDF) prevalidation on navigate
        # (compare_wrapper._run_dynamic_prevalidation_with_spinner); here the work
        # runs on the main thread with processEvents, so the spinner still animates.
        total_files = fb.count()
        self._app.app_actions.start_loading_spinner(force=True)
        self._app.app_actions.start_progress_bar()
        try:
            directory_was_excluded = PrevalidationsTab.remove_directory_from_exclusion_list(self._app.get_base_dir())
            files_checked = 0
            outcomes = 0
            moves = 0
            copies = 0
            deletes = 0
            errors = 0
            for media_path in fb.get_files():
                files_checked += 1
                try:
                    # No blur_callback: this pass does not navigate to each file, so a
                    # display-only BLUR action would have nothing to apply to.
                    result = PrevalidationsTab.prevalidate(
                        media_path,
                        self._app.get_base_dir,
                        self._app.app_actions.prevalidation_callbacks_with_mark,
                        force=False,  # TODO optionally allow force=True via different keybind
                    )
                    if result is not None:
                        outcomes += 1
                        if result == ClassifierActionType.MOVE:
                            moves += 1
                        elif result == ClassifierActionType.COPY:
                            copies += 1
                        elif result == ClassifierActionType.DELETE:
                            deletes += 1
                except Exception as e:
                    errors += 1
                    logger.error(e)
                self._app.app_actions._set_label_state(
                    _("Prevalidations: {0} / {1}").format(files_checked, total_files)
                )
                # Keep the UI responsive during long-running prevalidation
                QApplication.processEvents()
                # closeEvent may have fired inside processEvents — bail out cleanly
                if self._app._closing:
                    logger.info("Prevalidations for base dir: application closing, stopping early")
                    break
            if directory_was_excluded:
                PrevalidationsTab.add_directory_to_exclusion_list(self._app.get_base_dir())

            if not self._app._closing:
                # Rescan from disk so counts and paths match any moves/deletes (fast once
                # the directory is loaded; *force* bypasses the incremental-load refresh guard).
                self._app.refresh(force=True)
                self._app.notification_ctrl.toast(
                    _(
                        "All prevalidations in this directory have finished.\n"
                        "Files checked: {0} — outcomes: {1} — moves: {2} — copies: {3} — deletes: {4} — errors: {5}"
                    ).format(files_checked, outcomes, moves, copies, deletes, errors),
                    time_in_seconds=10,
                )
        finally:
            if not self._app._closing:
                self._app.app_actions.stop_loading_spinner()
                self._app.app_actions.stop_progress_bar()

    @require_password(ProtectedActions.RUN_PREVALIDATIONS)
    def toggle_prevalidations(self, event=None) -> None:
        """Toggle prevalidations on or off for the current base directory."""
        from utils.app_info_cache import app_info_cache
        base_dir = self._app.get_base_dir()
        new_val = not self._app.compare_manager.prevalidations_running
        if base_dir:
            app_info_cache.set(base_dir, "prevalidations_running", new_val)
        self._app.compare_manager.set_prevalidations_running(new_val)
        self._app.setWindowTitle(self._app.get_title_from_base_dir())
        self._app.notification_ctrl.toast(
            _("Prevalidations now running") if new_val
            else _("Prevalidations turned off")
        )

    def toggle_extra_debug_logging(self, event=None) -> None:
        """Toggle the extra-verbose debug logging flag."""
        from utils.config import config as _config
        _config.debug = not _config.debug
        _config.debug2 = not _config.debug2
        set_logger_level(_config.debug)
        self._app.notification_ctrl.toast(
            _("Extra debug logging enabled") if _config.debug2
            else _("Extra debug logging disabled")
        )

    # ------------------------------------------------------------------
    # Directory note operations
    # ------------------------------------------------------------------
    def toggle_directory_note_mark(self, event=None) -> None:
        """Toggle a file's marked status in directory notes."""
        from files.directory_notes import DirectoryNotes

        media_path = self._app.media_navigator.get_active_media_filepath()
        if not media_path:
            return

        base_dir = self._app.get_base_dir()
        if DirectoryNotes.is_marked_file(base_dir, media_path):
            DirectoryNotes.remove_marked_file(base_dir, media_path)
            self._app.notification_ctrl.toast(
                _("Removed from directory notes: {0}").format(os.path.basename(media_path))
            )
        else:
            DirectoryNotes.add_marked_file(base_dir, media_path)
            self._app.notification_ctrl.toast(
                _("Added to directory notes: {0}").format(os.path.basename(media_path))
            )

        # Refresh the notes window if it is open
        if self._directory_notes_window is not None:
            try:
                if self._directory_notes_window.isVisible():
                    self._directory_notes_window._refresh()
            except (RuntimeError, AttributeError):
                pass

    def edit_file_note(self, event=None) -> None:
        """Open a dialog to edit the note for the current file."""
        from files.directory_notes import DirectoryNotes
        from PySide6.QtWidgets import (
            QDialog, QLabel, QPlainTextEdit, QPushButton,
            QHBoxLayout, QVBoxLayout,
        )
        from ui.app_style import AppStyle

        media_path = self._app.media_navigator.get_active_media_filepath()
        if not media_path:
            return

        base_dir = self._app.get_base_dir()
        current_note = DirectoryNotes.get_file_note(base_dir, media_path) or ""

        dialog = QDialog(self._app)
        dialog.setWindowTitle(_("Edit Note - {0}").format(os.path.basename(media_path)))
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)
        path_label = QLabel(media_path, dialog)
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        note_edit = QPlainTextEdit(dialog)
        note_edit.setPlainText(current_note)
        layout.addWidget(note_edit)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton(_("Save"), dialog)
        cancel_btn = QPushButton(_("Cancel"), dialog)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def save_note():
            new_note = note_edit.toPlainText().strip()
            DirectoryNotes.set_file_note(base_dir, media_path, new_note)
            self._app.notification_ctrl.toast(
                _("Note saved for: {0}").format(os.path.basename(media_path))
            )
            dialog.accept()
            if self._directory_notes_window is not None:
                try:
                    if self._directory_notes_window.isVisible():
                        self._directory_notes_window._refresh()
                except (RuntimeError, AttributeError):
                    pass

        save_btn.clicked.connect(save_note)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()


"""
PySide6 port of image/image_details.py -- MediaDetails.

Displays media metadata, file info, prompt extraction, and provides
actions for rotate/crop/flip/enhance/convert/generation/related media.

Non-UI imports (reuse policy):
  - FileBrowser        from files.file_browser
  - FrameCache         from image.frame_cache
  - image_data_extractor from image.image_data_extractor
  - ImageOps           from image.image_ops
  - VideoOps           from image.video_ops (ffprobe for video metadata)
  - Cropper            from image.smart_crop
  - app_info_cache     from utils.app_info_cache
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from datetime import datetime
from typing import Optional

from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget, QDialog,
    QDialogButtonBox,
)

from files.related_image import (
    _VARIANT_SUFFIX_RE_STRICT as _VARIANT_SUFFIX_RE,
    DEFAULT_NODE_ID as _DEFAULT_NODE_ID,
    get_related_image_path as _get_related_image_path,
    get_related_image_text as _get_related_image_text,
)
from image.frame_cache import FrameCache
from image.image_data_extractor import image_data_extractor
from image.image_ops import ImageOps
from image.video_ops import VideoOps
from image.smart_crop import Cropper
from lib.multi_display_qt import SmartWindow
from ui.app_style import AppStyle
from ui.image.metadata_viewer_window_qt import MetadataViewerWindow
from ui.image.ocr_text_window_qt import OCRTextWindow
from ui.image.temp_media_window import TempMediaWindow
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.constants import ImageGenerationType
from utils.media_utils import get_media_type_for_path
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils, ModifierKey
logger = get_logger("media_details")


# ── Utility ───────────────────────────────────────────────────────────


def get_readable_file_size(path: str) -> str:
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024 * 1024:
        return f"{round(size / 1024, 1)} KB"
    else:
        return f"{round(size / (1024 * 1024), 1)} MB"


# ~1000 words at ~6 chars/word; line cap keeps the dialog usable.
_DETAILS_PROMPT_MAX_CHARS = 6000
_DETAILS_PROMPT_MAX_LINES = 42
_DETAILS_ELLIPSIS = "\u2026"


def truncate_details_label_text(
    text: str,
    max_chars: int = _DETAILS_PROMPT_MAX_CHARS,
    max_lines: int = _DETAILS_PROMPT_MAX_LINES,
) -> str:
    """
    Shorten long prompts for on-screen labels: cap line count, then character length.
    Copy actions use the untruncated strings stored on :class:`MediaDetails`.
    """
    if not text:
        return text
    lines = text.splitlines()
    truncated_lines = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated_lines = True
    out = "\n".join(lines)
    if truncated_lines:
        out = out + "\n" + _DETAILS_ELLIPSIS
    if len(out) > max_chars:
        out = out[: max_chars - len(_DETAILS_ELLIPSIS)].rstrip() + _DETAILS_ELLIPSIS
    return out


# ── MediaDetails ──────────────────────────────────────────────────────


class MediaDetails(SmartWindow):
    """Image details / actions dialog."""

    # -- Class-level state -----------------------------------------
    temp_media_canvas: Optional[TempMediaWindow] = None
    image_generation_mode = ImageGenerationType.CONTROL_NET
    # Stores the last generation adapter path (file or directory).
    previous_image_generation_adapter_path: Optional[str] = None
    metadata_viewer_window: Optional[MetadataViewerWindow] = None
    ocr_text_window: Optional[OCRTextWindow] = None

    COL_0_WIDTH = 100

    # Muted colors for special prompt label states
    _PROMPT_NOT_FOUND_COLOR = "#c07830"    # dark orange, differentiable at a glance
    _NEGATIVE_HIDDEN_COLOR = "#55526a"     # dark gray near background, hidden in plain sight

    # -- Static persistence ----------------------------------------
    ASPECT_RATIO_SETTINGS_KEY = "change_aspect_ratio_settings"

    @staticmethod
    def load_image_generation_mode() -> None:
        try:
            MediaDetails.image_generation_mode = ImageGenerationType.get(
                app_info_cache.get_meta(
                    "image_generation_mode",
                    default_val=ImageGenerationType.CONTROL_NET.value,
                )
            )
        except Exception as e:
            logger.error(f"Error loading image generation mode: {e}")

    @staticmethod
    def store_image_generation_mode() -> None:
        app_info_cache.set_meta(
            "image_generation_mode",
            MediaDetails.image_generation_mode.value,
        )

    # -- Construction ----------------------------------------------

    def __init__(
        self,
        parent: QWidget,
        media_path: str,
        index_text: str,
        app_actions,
        do_refresh: bool = True,
        take_focus: bool = True,
    ) -> None:
        super().__init__(
            persistent_parent=None,
            position_parent=parent,
            title=_("Image details"),
            geometry="700x900",
            respect_title_bar=True,
        )
        self._parent_ref = parent
        self._media_path = media_path
        self._image_path = FrameCache.get_image_path(media_path)
        self._temp_path = (
            self._image_path
            if self._image_path != self._media_path
            else None
        )
        media_ext = os.path.splitext(self._media_path)[1].lower()
        self._show_temp_path = media_ext in {".svg", ".pdf"} and self._temp_path is not None
        self._app_actions = app_actions
        self._do_refresh = do_refresh
        self._take_focus = take_focus
        self._has_closed = False
        self.media_type = get_media_type_for_path(self._media_path)
        self._full_positive_for_copy = None
        self._full_negative_for_copy = None

        # -- Determine content type --------------------------------
        # Video: ffprobe on the actual file. Unconfigured: disabled type or bad path.
        if self.media_type.is_video():
            (
                image_mode,
                image_dims,
                positive,
                negative,
                models,
                loras,
                related_image_text,
                self._prompt_extraction_failed,
            ) = self._gather_video_details()
        elif self.media_type.is_audio():
            (
                image_mode,
                image_dims,
                positive,
                negative,
                models,
                loras,
                related_image_text,
                self._prompt_extraction_failed,
            ) = self._gather_audio_details()
        elif self.media_type.is_unconfigured():
            (
                image_mode,
                image_dims,
                positive,
                negative,
                models,
                loras,
                related_image_text,
                self._prompt_extraction_failed,
            ) = self._gather_unconfigured_details()
        else:
            image_mode, image_dims = self._get_image_info()
            (positive, negative, models, loras, prompt_extraction_failed,
            ) = image_data_extractor.get_image_prompts_and_models(self._image_path)
            self._prompt_extraction_failed = prompt_extraction_failed
            related_image_text = self.get_related_image_text()

        mod_time, file_size = self._get_file_info()

        pos_display, neg_display, neg_is_placeholder = (
            self._store_prompt_strings_for_copy_and_get_display(positive, negative)
        )

        # -- Build UI ----------------------------------------------
        self._build_ui(
            image_mode,
            image_dims,
            pos_display,
            neg_display,
            neg_is_placeholder,
            models,
            loras,
            mod_time,
            file_size,
            index_text,
            related_image_text,
        )
        self._bind_shortcuts()
        if self._take_focus:
            self.focus()

    # -- UI construction -------------------------------------------

    def _store_prompt_strings_for_copy_and_get_display(
        self, positive: str, negative: str
    ) -> tuple[str, str, bool]:
        """Keep full prompts for copy; return truncated label text and negative placeholder flag."""
        self._full_positive_for_copy = positive
        self._full_negative_for_copy = negative
        pos_display = truncate_details_label_text(positive)
        neg_is_placeholder = not config.show_negative_prompt or negative == ""
        if neg_is_placeholder:
            neg_display = _("(negative prompt not shown by config setting)")
        else:
            neg_display = truncate_details_label_text(negative)
        return pos_display, neg_display, neg_is_placeholder

    def _build_ui(
        self,
        image_mode: str,
        image_dims: str,
        positive_display: str,
        negative_display: str,
        neg_is_placeholder: bool,
        models,
        loras,
        mod_time: str,
        file_size: str,
        index_text: str,
        related_image_text: str,
    ) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background: {AppStyle.BG_COLOR};")
        self._scroll_area = scroll

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        self._populate_scroll_content(
            image_mode, image_dims, positive_display, negative_display,
            neg_is_placeholder, models, loras, mod_time, file_size,
            index_text, related_image_text,
        )

    def _populate_scroll_content(
        self,
        image_mode: str,
        image_dims: str,
        positive_display: str,
        negative_display: str,
        neg_is_placeholder: bool,
        models,
        loras,
        mod_time: str,
        file_size: str,
        index_text: str,
        related_image_text: str,
    ) -> None:
        """Build (or rebuild) the scrollable content area.

        Safe to call more than once — QScrollArea takes ownership of the new
        content widget and deletes the previous one automatically.
        """
        content = QWidget()
        grid = QGridLayout(content)
        grid.setSpacing(6)

        row = 0

        # -- helpers -----------------------------------------------
        def _header(text: str, r: int, c: int = 0) -> QLabel:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setMaximumWidth(MediaDetails.COL_0_WIDTH)
            lbl.setStyleSheet(
                f"color: {AppStyle.FG_COLOR};"
                f"background: {AppStyle.BG_COLOR};"
            )
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            grid.addWidget(lbl, r, c)
            return lbl

        def _value(text: str, r: int, c: int = 1) -> QLabel:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                f"color: {AppStyle.FG_COLOR};"
                f"background: {AppStyle.BG_COLOR};"
            )
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            grid.addWidget(lbl, r, c)
            return lbl

        def _btn(text: str, callback, r: int, c: int = 0) -> QPushButton:
            b = QPushButton(text)
            b.clicked.connect(callback)
            grid.addWidget(b, r, c)
            return b

        # -- Info labels -------------------------------------------
        _header(_("Image Path"), row)
        self._lbl_path = _value(self._media_path, row)
        row += 1

        self._lbl_temp_path_header = _header(_("Temp Path"), row)
        self._lbl_temp_path = _value(self._temp_path or "", row)
        self._lbl_temp_path_header.setVisible(self._show_temp_path)
        self._lbl_temp_path.setVisible(self._show_temp_path)
        row += 1

        _header(_("File Index"), row)
        self._lbl_index = _value(index_text, row)
        row += 1

        _header(_("Color Mode"), row)
        self._lbl_mode = _value(image_mode, row)
        row += 1

        _header(_("Dimensions"), row)
        self._lbl_dims = _value(image_dims, row)
        row += 1

        _header(_("Size"), row)
        self._lbl_size = _value(file_size, row)
        row += 1

        _header(_("Modification Time"), row)
        self._lbl_mtime = _value(mod_time, row)
        row += 1

        _header(_("Positive"), row)
        self._lbl_positive = _value(positive_display, row)
        if self._prompt_extraction_failed:
            self._lbl_positive.setStyleSheet(
                f"color: {MediaDetails._PROMPT_NOT_FOUND_COLOR};"
                f"background: {AppStyle.BG_COLOR};"
            )
        row += 1

        _header(_("Negative"), row)
        self._lbl_negative = _value(negative_display, row)
        if neg_is_placeholder:
            self._lbl_negative.setStyleSheet(
                f"color: {MediaDetails._NEGATIVE_HIDDEN_COLOR};"
                f"background: {AppStyle.BG_COLOR};"
            )
        row += 1

        _header(_("Models"), row)
        self._lbl_models = _value(", ".join(models), row)
        row += 1

        _header(_("LoRAs"), row)
        self._lbl_loras = _value(", ".join(loras), row)
        row += 1

        # -- Action buttons (two columns) --------------------------
        _btn(_("Copy Prompt"), self.copy_prompt, row, 0)
        _btn(_("Copy Prompt No BREAK"), self.copy_prompt_no_break, row, 1)
        row += 1

        if self.media_type.supports_raster_image_details():
            _btn(_("Rotate Image Left"), lambda: self.rotate_image(right=False), row, 0)
            _btn(_("Rotate Image Right"), lambda: self.rotate_image(right=True), row, 1)
            row += 1

            _btn(_("Crop Image (Smart Detect)"), lambda: self.crop_image(), row, 0)
            _btn(_("Enhance Image"), lambda: self.enhance_image(), row, 1)
            row += 1

            _btn(_("Random Crop"), lambda: self.random_crop(), row, 0)
            _btn(_("Randomly Modify"), lambda: self.random_modification(), row, 1)
            row += 1

            _btn(_("Flip Image Horizontally"), lambda: self.flip_image(), row, 0)
            _btn(_("Flip Image Vertically"), lambda: self.flip_image(top_bottom=True), row, 1)
            row += 1

            _btn(_("Change Aspect Ratio"), self.open_change_aspect_ratio_dialog, row, 0)
            _btn(_("Flip Aspect Ratio"), self.flip_aspect_ratio, row, 1)
            row += 1

            _btn(_("Copy Without EXIF"), lambda: self.copy_without_exif(), row, 0)
            _btn(_("Convert to JPG"), lambda: self.convert_to_jpg(), row, 1)
            row += 1

        _btn(_("Show Metadata"), lambda: self.show_metadata(), row, 0)
        if self.media_type.supports_raster_image_details():
            _btn(_("Run OCR"), lambda: self.run_ocr(), row, 1)
        row += 1

        if self.media_type.supports_raster_image_details():
            _btn(_("Color Diversity Score"), lambda: self.show_color_diversity_score(), row, 0)
            _btn(_("Hue Breadth Score"), lambda: self.show_hue_breadth_score(), row, 1)
            row += 1

        _btn(_("Open Related Image"), self.open_related_image, row, 0)
        self._lbl_related_image = _value(related_image_text, row)
        row += 1

        # -- Image generation section (raster only) ----------------
        if self.media_type.supports_raster_image_details():
            _header(_("Image Generation"), row)
            self._gen_mode_combo = QComboBox()
            for member in ImageGenerationType:
                self._gen_mode_combo.addItem(member.get_text(), member.value)
            selected_index = self._gen_mode_combo.findData(
                MediaDetails.image_generation_mode.value
            )
            if selected_index >= 0:
                self._gen_mode_combo.setCurrentIndex(selected_index)
            self._gen_mode_combo.currentIndexChanged.connect(self._on_gen_mode_changed)
            grid.addWidget(self._gen_mode_combo, row, 1)
            row += 1

            _btn(_("Run Image Generation"), self.run_image_generation, row, 0)
            _value(_("Press Shift+I on a main app window to run this"), row)
            row += 1

            _btn(_("Redo Prompt"), self.run_redo_prompt, row, 0)
            row += 1

        # -- Tags section (conditional) ----------------------------
        if config.image_tagging_enabled and self.media_type.supports_raster_image_details():
            _header(_("Tags"), row)
            tags = image_data_extractor.extract_tags(self._image_path)
            self._tags = tags if tags else []
            tags_str = ", ".join(self._tags) if self._tags else ""
            self._tags_entry = QLineEdit(tags_str)
            grid.addWidget(self._tags_entry, row, 1)
            row += 1

            _btn(_("Update Tags"), self.update_tags, row, 0)
            row += 1

        # Stretch at bottom
        grid.setRowStretch(row, 1)

        self._scroll_area.setWidget(content)

    # -- Shortcuts -------------------------------------------------

    def _bind_shortcuts(self) -> None:
        def sc(key: str, fn) -> None:
            s = QShortcut(QKeySequence(key), self)
            s.activated.connect(fn)

        sc("Escape", self.close_windows)
        sc("Shift+Escape", self.close_windows)

        # Shift+key -- action only
        sc("Shift+C", lambda: self.crop_image())
        sc("Shift+L", lambda: self.rotate_image(right=False))
        sc("Shift+P", lambda: self.rotate_image(right=True))
        sc("Shift+R", self.open_related_image)
        sc("Shift+E", lambda: self.copy_prompt_no_break())
        sc("Shift+B", lambda: self.enhance_image())
        sc("Shift+A", lambda: self.random_crop())
        sc("Shift+Q", lambda: self.random_modification())
        sc("Ctrl+Shift+Q", lambda: self.scramble_image())
        sc("Shift+H", lambda: self.flip_image())
        sc("Shift+V", lambda: self.flip_image(top_bottom=True))
        sc("Shift+X", lambda: self.copy_without_exif())
        sc("Shift+J", lambda: self.convert_to_jpg())
        sc("Shift+K", lambda: self.convert_to_jpg())
        sc("Shift+D", lambda: self.show_metadata())
        sc("Shift+O", lambda: self.run_ocr())
        sc("Shift+I", self.run_image_generation)
        sc("Shift+Y", self.run_redo_prompt)

        # Ctrl+key -- action + mark (opens marks window without GUI)
        sc("Ctrl+C", lambda: self._crop_image_and_mark())
        sc("Ctrl+L", lambda: self._rotate_image_and_mark(right=False))
        sc("Ctrl+R", lambda: self._rotate_image_and_mark(right=True))
        sc("Ctrl+E", lambda: self._enhance_image_and_mark())
        sc("Ctrl+A", lambda: self._random_crop_and_mark())
        sc("Ctrl+Q", lambda: self._random_modification_and_mark())
        sc("Ctrl+H", lambda: self._flip_image_and_mark())
        sc("Ctrl+V", lambda: self._flip_image_and_mark(top_bottom=True))
        sc("Ctrl+X", lambda: self._copy_without_exif_and_mark())
        sc("Ctrl+J", lambda: self._convert_to_jpg_and_mark())
        sc("Ctrl+K", lambda: self._convert_to_jpg_and_mark())

    # -- Focus -----------------------------------------------------

    def focus(self) -> None:
        QTimer.singleShot(
            1, lambda: (self.raise_(), self.activateWindow())
        )

    def _gather_unconfigured_details(self) -> tuple[str, str, str, str, list[str], list[str], str, bool]:
        """Placeholder fields when the path is invalid or the file type is disabled in config."""
        related_image_text = _("(Not available)")
        return (
            _("Unavailable"),
            "",
            _(
                "This media type is not enabled in configuration, or the path is invalid."
            ),
            "",
            [],
            [],
            related_image_text,
            True,
        )

    def _gather_video_details(self) -> tuple[str, str, str, str, list[str], list[str], str, bool]:
        """Load mode, dimensions, and tag-derived prompts from the video file via ffprobe.

        If the probe reveals no video stream (e.g. an M4A/M4P file misclassified
        as video), falls back to ``_gather_audio_details`` using the same probe
        so ffprobe is only run once.
        """
        probe: dict | None = None
        try:
            if VideoOps.find_ffprobe_executable():
                probe = VideoOps.ffprobe_json(self._media_path)
        except RuntimeError as e:
            logger.debug("ffprobe failed: %s", e)

        if probe is not None:
            has_video_stream = any(
                s.get("codec_type") == "video"
                and not s.get("disposition", {}).get("attached_pic", 0)
                for s in (probe.get("streams") or [])
            )
            if not has_video_stream:
                logger.debug(
                    "%s has no video stream — treating as audio", self._media_path
                )
                return self._gather_audio_details(probe)

        try:
            if probe:
                image_mode, image_dims = VideoOps.ffprobe_video_mode_and_dims(probe)
                positive, negative, models, loras, prompt_failed = (
                    VideoOps.ffprobe_prompt_fields_from_tags(probe)
                )
            else:
                image_mode = _("Video (ffprobe unavailable)")
                image_dims = ""
                positive = ""
                negative = ""
                models = []
                loras = []
                prompt_failed = True
            related_image_text = self.get_related_image_text()
        except Exception as e:
            logger.warning("Video details gather failed: %s", e)
            return (
                _("Video"),
                "",
                "",
                "",
                [],
                [],
                _("(Could not load video details)"),
                True,
            )
        return (
            image_mode,
            image_dims,
            positive,
            negative,
            models,
            loras,
            related_image_text,
            prompt_failed,
        )

    def _gather_audio_details(
        self, probe: dict | None = None
    ) -> tuple[str, str, str, str, list[str], list[str], str, bool]:
        """Load metadata from an audio file via ffprobe.

        *probe* may be passed in when the caller already ran ffprobe (e.g.
        ``_gather_video_details`` discovered there is no video stream).
        """
        if probe is None:
            try:
                if VideoOps.find_ffprobe_executable():
                    probe = VideoOps.ffprobe_json(self._media_path)
            except RuntimeError as e:
                logger.debug("ffprobe failed for audio: %s", e)

        if probe is None:
            return (
                _("Audio (ffprobe unavailable)"),
                "",
                "",
                "",
                [],
                [],
                _("(Not available)"),
                True,
            )

        try:
            codec, duration_str, bitrate_str, sample_rate_str, title, artist, album = (
                VideoOps.ffprobe_audio_info(probe)
            )

            image_mode = _("Audio ({0})").format(codec) if codec else _("Audio")

            dims_parts = [p for p in [duration_str, bitrate_str, sample_rate_str] if p]
            image_dims = " · ".join(dims_parts)

            if title or artist or album:
                tag_lines = []
                if title:
                    tag_lines.append(f"{_('Title')}: {title}")
                if artist:
                    tag_lines.append(f"{_('Artist')}: {artist}")
                if album:
                    tag_lines.append(f"{_('Album')}: {album}")
                positive = "\n".join(tag_lines)
                prompt_failed = False
            else:
                tags = VideoOps.merge_ffprobe_tag_dicts(probe)
                if tags:
                    positive = "\n".join(f"{k}: {v}" for k, v in sorted(tags.items()))
                    prompt_failed = False
                else:
                    positive = ""
                    prompt_failed = True

        except Exception as e:
            logger.warning("Audio details gather failed: %s", e)
            return _("Audio"), "", "", "", [], [], _("(Could not load audio details)"), True

        return image_mode, image_dims, positive, "", [], [], _("(Not available)"), prompt_failed

    def _path_for_metadata_window(self) -> str:
        """Path shown in the metadata viewer title (media file, not a cached frame)."""
        if self.media_type.is_video() or self.media_type.is_audio() or self.media_type.is_unconfigured():
            return self._media_path
        return self._image_path

    # -- Info helpers ----------------------------------------------

    def _get_image_info(self) -> tuple[str, str]:
        try:
            with Image.open(self._image_path) as image:
                image_mode = str(image.mode)
                image_dims = f"{image.size[0]}x{image.size[1]}"
            return image_mode, image_dims
        except Exception as e:
            logger.warning("Could not read image info for %s: %s", self._image_path, e)
            return _("Unknown"), ""

    def _get_file_info(self) -> tuple[str, str]:
        if self.media_type.is_video() or self.media_type.is_audio() or self.media_type.is_unconfigured():
            stat_path = self._media_path
        else:
            stat_path = self._image_path
        mod_time = datetime.fromtimestamp(
            os.path.getmtime(stat_path)
        ).strftime("%Y-%m-%d %H:%M")
        file_size = get_readable_file_size(stat_path)
        return mod_time, file_size

    def update_media_details(self, media_path: str, index_text: str) -> None:
        """Refresh all displayed fields for new media."""
        self._media_path = media_path
        self._image_path = FrameCache.get_image_path(media_path)
        self._temp_path = (
            self._image_path
            if self._image_path != self._media_path
            else None
        )
        media_ext = os.path.splitext(self._media_path)[1].lower()
        self._show_temp_path = media_ext in {".svg", ".pdf"} and self._temp_path is not None
        old_supports_raster = self.media_type.supports_raster_image_details()
        self.media_type = get_media_type_for_path(self._media_path)
        if self.media_type.is_video():
            (
                image_mode,
                image_dims,
                positive,
                negative,
                models,
                loras,
                related_image_text,
                self._prompt_extraction_failed,
            ) = self._gather_video_details()
        elif self.media_type.is_audio():
            (
                image_mode,
                image_dims,
                positive,
                negative,
                models,
                loras,
                related_image_text,
                self._prompt_extraction_failed,
            ) = self._gather_audio_details()
        elif self.media_type.is_unconfigured():
            (
                image_mode,
                image_dims,
                positive,
                negative,
                models,
                loras,
                related_image_text,
                self._prompt_extraction_failed,
            ) = self._gather_unconfigured_details()
        else:
            image_mode, image_dims = self._get_image_info()
            (
                positive,
                negative,
                models,
                loras,
                prompt_extraction_failed,
            ) = image_data_extractor.get_image_prompts_and_models(
                self._image_path
            )
            self._prompt_extraction_failed = prompt_extraction_failed
            related_image_text = self.get_related_image_text()

        mod_time, file_size = self._get_file_info()
        pos_display, neg_display, neg_is_placeholder = (
            self._store_prompt_strings_for_copy_and_get_display(positive, negative)
        )
        if self.media_type.supports_raster_image_details() != old_supports_raster:
            self._populate_scroll_content(
                image_mode, image_dims, pos_display, neg_display, neg_is_placeholder,
                models, loras, mod_time, file_size, index_text, related_image_text,
            )
        else:
            self._lbl_path.setText(self._media_path)
            if self._lbl_temp_path is not None and self._lbl_temp_path_header is not None:
                self._lbl_temp_path_header.setVisible(self._show_temp_path)
                self._lbl_temp_path.setVisible(self._show_temp_path)
                self._lbl_temp_path.setText(self._temp_path if self._show_temp_path else "")
            self._lbl_index.setText(index_text)
            self._lbl_mode.setText(image_mode)
            self._lbl_dims.setText(image_dims)
            self._lbl_mtime.setText(mod_time)
            self._lbl_size.setText(file_size)
            self._lbl_positive.setText(pos_display)
            if self._prompt_extraction_failed:
                self._lbl_positive.setStyleSheet(
                    f"color: {MediaDetails._PROMPT_NOT_FOUND_COLOR};"
                    f"background: {AppStyle.BG_COLOR};"
                )
            else:
                self._lbl_positive.setStyleSheet(
                    f"color: {AppStyle.FG_COLOR};"
                    f"background: {AppStyle.BG_COLOR};"
                )
            self._lbl_negative.setText(neg_display)
            if neg_is_placeholder:
                self._lbl_negative.setStyleSheet(
                    f"color: {MediaDetails._NEGATIVE_HIDDEN_COLOR};"
                    f"background: {AppStyle.BG_COLOR};"
                )
            else:
                self._lbl_negative.setStyleSheet(
                    f"color: {AppStyle.FG_COLOR};"
                    f"background: {AppStyle.BG_COLOR};"
                )
            self._lbl_models.setText(", ".join(models))
            self._lbl_loras.setText(", ".join(loras))
            self._lbl_related_image.setText(related_image_text)

        # Refresh open metadata viewer
        if MediaDetails.metadata_viewer_window is not None:
            if MediaDetails.metadata_viewer_window.has_closed:
                MediaDetails.metadata_viewer_window = None
            else:
                self.show_metadata()

    # ── Clipboard operations ──────────────────────────────────────

    def copy_prompt(self) -> None:
        positive = self._full_positive_for_copy
        if positive is None:
            positive = self._lbl_positive.text()
        MediaDetails._copy_prompt_static(
            positive, self._app_actions, self._prompt_extraction_failed
        )

    def copy_prompt_no_break(self) -> None:
        positive = self._full_positive_for_copy
        if positive is None:
            positive = self._lbl_positive.text()
        MediaDetails._copy_prompt_static(
            positive,
            self._app_actions,
            self._prompt_extraction_failed,
            remove_emphases=True,
        )

    @staticmethod
    def copy_prompt_no_break_static(
        image_path: str, master, app_actions
    ) -> None:
        positive, _neg, _mod, _lor, prompt_extraction_failed = (
            image_data_extractor.get_image_prompts_and_models(image_path)
        )
        MediaDetails._copy_prompt_static(
            positive,
            app_actions,
            prompt_extraction_failed,
            remove_emphases=True,
        )

    @staticmethod
    def _copy_prompt_static(
        positive,
        app_actions,
        prompt_extraction_failed,
        remove_emphases=False,
    ) -> None:
        if (
            prompt_extraction_failed
            or positive is None
            or positive.strip() == ""
        ):
            app_actions.warn(_("No prompt found"))
        else:
            if remove_emphases:
                if "BREAK" in positive:
                    positive = positive[positive.index("BREAK") + 6 :]
                positive = MediaDetails.remove_emphases(positive)
            QGuiApplication.clipboard().setText(positive)
            app_actions.toast(_("Copied prompt without BREAK"))

    @staticmethod
    def remove_emphases(prompt: str) -> str:
        prompt = prompt.replace("(", "").replace(")", "")
        prompt = prompt.replace("[", "").replace("]", "")
        if ":" in prompt:
            prompt = re.sub(r":[0-9]*\.[0-9]+", "", prompt)
        if "<" in prompt:
            prompt = re.sub(r"<[^>]*>", "", prompt)
        return prompt

    @staticmethod
    def source_random_prompt(file_browser, master, app_actions) -> None:
        """
        Find a random file from the file browser that contains a prompt,
        copy the prompt to clipboard, and notify the user.
        """
        if not file_browser.has_files():
            app_actions.warn(_("No files found in current directory"))
            return

        files = list(file_browser.get_files())
        if not files:
            app_actions.warn(_("No files available"))
            return

        random.shuffle(files)
        max_attempts = min(500, len(files))
        for i in range(max_attempts):
            file_path = files[i]
            try:
                (
                    positive, _neg, _mod, _lor, prompt_extraction_failed,
                ) = image_data_extractor.get_image_prompts_and_models(
                    file_path
                )
                if (
                    not prompt_extraction_failed
                    and positive is not None
                    and positive.strip() != ""
                ):
                    prompt_text = positive
                    if "BREAK" in prompt_text:
                        prompt_text = prompt_text[prompt_text.index("BREAK") + 6:]
                    prompt_text = MediaDetails.remove_emphases(prompt_text)
                    QGuiApplication.clipboard().setText(prompt_text)
                    filename = os.path.basename(file_path)
                    app_actions.success(
                        _("Copied prompt from {0}").format(filename)
                    )
                    return
            except Exception as e:
                logger.debug(f"Error extracting prompt from {file_path}: {e}")
                continue

        app_actions.warn(_("No files with prompts found in current directory"))

    # ── Unified image-action handler ──────────────────────────────

    def _handle_action_result(
        self,
        new_filepath: str,
        success_msg: str,
        *,
        mark: bool = False,
        close: bool = True,
    ) -> None:
        """Common post-processing for image manipulation actions.

        Parameters
        ----------
        new_filepath : str
            Path returned by the image operation.
        success_msg : str
            Toast text shown to the user.
        mark : bool
            If *True*, open the marks window (no GUI) instead of the
            temp image canvas.
        close : bool
            If *True*, close this MediaDetails window before refreshing.
        """
        if close:
            self.close_windows()
        self._app_actions.refresh()
        self._app_actions.success(success_msg)
        if new_filepath and os.path.exists(new_filepath):
            if mark:
                self._app_actions.open_move_marks_window(
                    filepath=new_filepath, open_gui=False
                )
            else:
                MediaDetails.open_temp_media_canvas(
                    master=self._parent_ref,
                    media_path=new_filepath,
                    app_actions=self._app_actions,
                )

    # ── Image manipulation actions ────────────────────────────────

    def rotate_image(self, right: bool = False) -> None:
        new_filepath = ImageOps.rotate_image(self._image_path, right)
        msg = (
            _("Rotated image right") if right else _("Rotated image left")
        )
        self._handle_action_result(new_filepath, msg)

    def crop_image(self, event=None) -> None:
        saved_files = Cropper.smart_crop_multi_detect(self._image_path, "")
        if len(saved_files) > 0:
            self.close_windows()
            self._app_actions.refresh()
            self._app_actions.success(_("Cropped image"))
            MediaDetails.open_temp_media_canvas(
                master=self._parent_ref,
                media_path=saved_files[0],
                app_actions=self._app_actions,
            )
        else:
            self._app_actions.toast(_("No crops found"))

    def enhance_image(self) -> None:
        new_filepath = ImageOps.enhance_image(self._image_path)
        self._handle_action_result(new_filepath, _("Enhanced image"))

    def random_crop(self) -> None:
        new_filepath = ImageOps.random_crop_and_upscale(self._image_path)
        self._handle_action_result(new_filepath, _("Randomly cropped image"))

    def random_modification(self) -> None:
        MediaDetails.randomly_modify_image(
            self._image_path, self._app_actions, self._parent_ref
        )

    @staticmethod
    def randomly_modify_image(
        image_path: str, app_actions, master=None
    ) -> None:
        new_filepath = ImageOps.randomly_modify_image(image_path)
        app_actions.refresh()
        if os.path.exists(new_filepath):
            app_actions.success(_("Randomly modified image"))
            if master is not None:
                MediaDetails.open_temp_media_canvas(
                    master=master,
                    media_path=new_filepath,
                    app_actions=app_actions,
                )
        else:
            app_actions.toast(_("No new image created"))

    def scramble_image(self) -> None:
        MediaDetails.scramble_image_static(
            self._image_path, self._app_actions, self._parent_ref
        )

    @staticmethod
    def scramble_image_static(
        image_path: str, app_actions, master=None
    ) -> None:
        new_filepath = ImageOps.scramble_image(image_path)
        app_actions.refresh()
        if os.path.exists(new_filepath):
            app_actions.success(_("Scrambled image"))
            if master is not None:
                MediaDetails.open_temp_media_canvas(
                    master=master,
                    media_path=new_filepath,
                    app_actions=app_actions,
                )
        else:
            app_actions.toast(_("No new image created"))

    def _scramble_image_and_mark(self) -> None:
        new_filepath = ImageOps.scramble_image(self._image_path)
        self.close_windows()
        self._app_actions.refresh()
        if new_filepath and os.path.exists(new_filepath):
            self._app_actions.toast(_("Scrambled image"))
            self._app_actions.open_move_marks_window(
                filepath=new_filepath, open_gui=False
            )
        else:
            self._app_actions.toast(_("No new image created"))

    def flip_image(self, top_bottom: bool = False) -> None:
        if top_bottom:
            from lib.qt_alert import qt_alert
            if not qt_alert(
                self,
                _("Confirm Vertical Flip"),
                _(
                    "Are you sure you want to flip this image vertically? "
                    "This is an uncommon operation and may have been "
                    "clicked by accident."
                ),
                kind="askokcancel",
            ):
                return
        new_filepath = ImageOps.flip_image(
            self._image_path, top_bottom=top_bottom
        )
        self._handle_action_result(new_filepath, _("Flipped image"))

    def _get_current_dimensions(self) -> tuple[int, int]:
        with Image.open(self._image_path) as image:
            return image.size

    @staticmethod
    def _ratio_text(width: int, height: int) -> str:
        divisor = math.gcd(width, height)
        if divisor <= 0:
            return f"{width}:{height}"
        return f"{width // divisor}:{height // divisor}"

    def _store_aspect_ratio_settings(self, target_ratio: str) -> None:
        app_info_cache.set_meta(
            MediaDetails.ASPECT_RATIO_SETTINGS_KEY,
            {"target_ratio": target_ratio},
        )

    def _get_saved_aspect_ratio(self) -> str | None:
        settings = app_info_cache.get_meta(
            MediaDetails.ASPECT_RATIO_SETTINGS_KEY,
            default_val={},
        )
        if isinstance(settings, dict):
            value = settings.get("target_ratio")
            if isinstance(value, str) and value.strip() != "":
                return value
        return None

    def _apply_aspect_ratio_change(self, target_ratio: str) -> bool:
        ratio_text = target_ratio.strip()
        if ratio_text == "":
            self._app_actions.warn(_("Please enter a target ratio"))
            return False
        try:
            new_filepath = ImageOps.change_aspect_ratio(
                self._image_path,
                ratio_text,
            )
            self._store_aspect_ratio_settings(ratio_text)
            self._handle_action_result(
                new_filepath,
                _("Changed image aspect ratio"),
            )
            return True
        except Exception as e:
            logger.error(f"Error changing image aspect ratio: {e}")
            self._app_actions.warn(_("Error changing image aspect ratio"))
            return False

    def flip_aspect_ratio(self) -> None:
        if not self.media_type.supports_raster_image_details():
            self._app_actions.toast(_("Aspect ratio changes are only available for images"))
            return
        width, height = self._get_current_dimensions()
        self._apply_aspect_ratio_change(f"{height}:{width}")

    def open_change_aspect_ratio_dialog(self) -> None:
        if not self.media_type.supports_raster_image_details():
            self._app_actions.toast(_("Aspect ratio changes are only available for images"))
            return

        width, height = self._get_current_dimensions()
        current_ratio = MediaDetails._ratio_text(width, height)
        saved_ratio = self._get_saved_aspect_ratio()
        default_ratio = saved_ratio if saved_ratio is not None else current_ratio

        dialog = QDialog(self)
        dialog.setWindowTitle(_("Change Aspect Ratio"))
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        current_label = QLabel(
            _("Current ratio: {0}").format(current_ratio)
        )
        target_label = QLabel(_("Target ratio (e.g. 16:9 or 1.777):"))
        ratio_input = QLineEdit(default_ratio)
        ratio_input.selectAll()

        layout.addWidget(current_label)
        layout.addWidget(target_label)
        layout.addWidget(ratio_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        def _apply_from_input() -> None:
            if self._apply_aspect_ratio_change(ratio_input.text()):
                dialog.accept()

        buttons.accepted.connect(_apply_from_input)
        buttons.rejected.connect(dialog.reject)
        dialog.exec()

    def copy_without_exif(self) -> None:
        try:
            new_filepath = image_data_extractor.copy_without_exif(self._image_path)
            self._handle_action_result(
                new_filepath,
                _("Copied image without EXIF data"),
                close=False,
            )
        except Exception as e:
            logger.error(f"Error copying image without EXIF: {e}")
            self._app_actions.warn(_("Error copying image without EXIF"))

    def convert_to_jpg(self) -> None:
        try:
            new_filepath = ImageOps.convert_to_jpg(self._image_path)
            self._handle_action_result(new_filepath, _("Converted image to JPG"))
        except Exception as e:
            logger.error(f"Error converting image to JPG: {e}")
            self._app_actions.warn(_("Error converting image to JPG"))

    # ── Mark-and-action variants ──────────────────────────────────

    def _rotate_image_and_mark(self, right: bool = False) -> None:
        new_filepath = ImageOps.rotate_image(self._image_path, right)
        msg = _("Rotated image right") if right else _("Rotated image left")
        self._handle_action_result(new_filepath, msg, mark=True)

    def _crop_image_and_mark(self, event=None) -> None:
        saved_files = Cropper.smart_crop_multi_detect(self._image_path, "")
        if len(saved_files) > 0:
            self.close_windows()
            self._app_actions.refresh()
            self._app_actions.toast(_("Cropped image"))
            self._app_actions.open_move_marks_window(
                filepath=saved_files[0], open_gui=False
            )
        else:
            self._app_actions.toast(_("No crops found"))

    def _enhance_image_and_mark(self) -> None:
        """Enhance and mark.  Uses toast (not success) per original."""
        new_filepath = ImageOps.enhance_image(self._image_path)
        self.close_windows()
        self._app_actions.refresh()
        self._app_actions.toast(_("Enhanced image"))
        if new_filepath and os.path.exists(new_filepath):
            self._app_actions.open_move_marks_window(
                filepath=new_filepath, open_gui=False
            )

    def _random_crop_and_mark(self) -> None:
        new_filepath = ImageOps.random_crop_and_upscale(self._image_path)
        self._handle_action_result(
            new_filepath, _("Randomly cropped image"), mark=True
        )

    def _random_modification_and_mark(self) -> None:
        new_filepath = ImageOps.randomly_modify_image(self._image_path)
        self.close_windows()
        self._app_actions.refresh()
        if new_filepath and os.path.exists(new_filepath):
            self._app_actions.toast(_("Randomly modified image"))
            self._app_actions.open_move_marks_window(
                filepath=new_filepath, open_gui=False
            )
        else:
            self._app_actions.toast(_("No new image created"))

    def _flip_image_and_mark(self, top_bottom: bool = False) -> None:
        new_filepath = ImageOps.flip_image(self._image_path, top_bottom=top_bottom)
        self._handle_action_result(new_filepath, _("Flipped image"), mark=True)

    def _copy_without_exif_and_mark(self) -> None:
        try:
            new_filepath = image_data_extractor.copy_without_exif(self._image_path)
            self._handle_action_result(
                new_filepath,
                _("Copied image without EXIF data"),
                mark=True,
                close=False,
            )
        except Exception as e:
            logger.error(f"Error copying image without EXIF: {e}")
            self._app_actions.warn(_("Error copying image without EXIF"))

    def _convert_to_jpg_and_mark(self) -> None:
        try:
            new_filepath = ImageOps.convert_to_jpg(self._image_path)
            self._handle_action_result(new_filepath, _("Converted image to JPG"), mark=True)
        except Exception as e:
            logger.error(f"Error converting image to JPG: {e}")
            self._app_actions.warn(_("Error converting image to JPG"))

    # ── Metadata viewer ──────────────────────────────────────────

    def show_metadata(self, event=None) -> None:
        if self.media_type.is_unconfigured():
            self._app_actions.toast(_("Metadata is not available for this path"))
            return
        if self.media_type.is_video() or self.media_type.is_audio():
            try:
                if not VideoOps.find_ffprobe_executable():
                    self._app_actions.warn(_("ffprobe not found on PATH"))
                    return
                data = VideoOps.ffprobe_json(self._media_path)
                metadata_text = json.dumps(data, indent=2, ensure_ascii=False)
            except RuntimeError as e:
                self._app_actions.warn(str(e))
                return
            if not metadata_text or metadata_text.strip() in ("", "{}"):
                self._app_actions.toast(_("No metadata found"))
                return
            self._show_metadata_window(metadata_text)
            return

        metadata_text = image_data_extractor.get_raw_metadata_text(self._image_path)
        if metadata_text is None:
            self._app_actions.toast(_("No metadata found"))
        else:
            self._show_metadata_window(metadata_text)

    def _show_metadata_window(self, metadata_text: str) -> None:
        path_for_title = self._path_for_metadata_window()
        mvw = MediaDetails.metadata_viewer_window
        if mvw is None or mvw.has_closed:
            MediaDetails.metadata_viewer_window = MetadataViewerWindow(
                self, self._app_actions, metadata_text, path_for_title
            )
            MediaDetails.metadata_viewer_window.show()
        else:
            mvw.update_metadata(metadata_text, path_for_title)

    # ── OCR ──────────────────────────────────────────────────────

    def run_ocr(self) -> None:
        """Run Surya OCR on the current image and show the result."""
        if not self.media_type.supports_raster_image_details():
            self._app_actions.toast(_("OCR is only available for images"))
            return
        if not ImageOps.is_surya_ocr_available():
            self._app_actions.warn(_("Surya OCR is not installed"))
            return
        try:
            result = ImageOps.run_ocr(self._image_path)
            if not result.has_text:
                self._app_actions.toast(_("No text found in image"))
                return
            self._show_ocr_window(result.text, result.avg_confidence)
        except RuntimeError as e:
            self._app_actions.warn(str(e))
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            self._app_actions.warn(_("OCR failed: ") + str(e))

    def _show_ocr_window(self, ocr_text: str, confidence: float | None) -> None:
        w = MediaDetails.ocr_text_window
        if w is None or w.has_closed:
            MediaDetails.ocr_text_window = OCRTextWindow(
                self, self._app_actions, ocr_text, self._image_path,
                confidence=confidence,
            )
            MediaDetails.ocr_text_window.show()
        else:
            w.update_text(ocr_text, self._image_path, confidence)

    # ── Color analysis ───────────────────────────────────────────

    def show_color_diversity_score(self) -> None:
        try:
            score = ImageOps.color_diversity_score(self._image_path)
        except Exception as e:
            logger.error("Color diversity score failed: %s", e)
            self._app_actions.toast(_("Could not compute color diversity score"))
            return
        if score >= 0.75:
            bracket = _("high")
        elif score >= 0.4:
            bracket = _("medium")
        else:
            bracket = _("low")
        self._app_actions.toast(
            _("Color diversity score: {0:.2f} ({1})").format(score, bracket)
        )

    def show_hue_breadth_score(self) -> None:
        try:
            score = ImageOps.hue_breadth_score(self._image_path)
        except Exception as e:
            logger.error("Hue breadth score failed: %s", e)
            self._app_actions.toast(_("Could not compute hue breadth score"))
            return
        if score >= 0.75:
            bracket = _("high")
        elif score >= 0.4:
            bracket = _("medium")
        else:
            bracket = _("low")
        self._app_actions.toast(
            _("Hue breadth score: {0:.2f} ({1})").format(score, bracket)
        )

    # ── Related images ───────────────────────────────────────────

    def get_related_image_text(self) -> str:
        # Related-image metadata is read via PIL / workflow JSON on image files only.
        if self.media_type.is_unconfigured():
            return _("(Not available)")
        if self.media_type.is_video():
            return _("(Related image lookup is not available for video)")
        if self.media_type.is_audio():
            return _("(Related image lookup is not available for audio)")
        return _get_related_image_text(self._image_path, _DEFAULT_NODE_ID)

    def open_related_image(self, event=None) -> None:
        if self.media_type.is_unconfigured():
            self._app_actions.toast(_("Related image is not available for this path"))
            return
        if self.media_type.is_video():
            self._app_actions.toast(_("Related image is not available for video files"))
            return
        MediaDetails.show_related_image(
            self._parent_ref, None, self._image_path, self._app_actions
        )

    @staticmethod
    def get_related_image_path(
        image_path: str,
        node_id: str | None = None,
        check_extra_directories: bool | None = True,
    ) -> tuple[str | None, bool]:
        if node_id is None or node_id == "":
            node_id = _DEFAULT_NODE_ID
        return _get_related_image_path(image_path, node_id, check_extra_directories)

    @staticmethod
    def show_related_image(
        master=None, node_id=None, image_path="", app_actions=None
    ) -> None:
        if master is None or image_path == "":
            raise Exception("No master or image path given")
        related_image_path, exact_match = (
            MediaDetails.get_related_image_path(image_path, node_id)
        )
        if related_image_path is None or related_image_path == "":
            app_actions.toast(_("(No related image found)"))
            return
        elif not exact_match:
            app_actions.toast(_(" (Exact Match Not Found)"))
            return
        MediaDetails.open_temp_media_canvas(
            master=master,
            media_path=related_image_path,
            app_actions=app_actions,
        )

    @staticmethod
    def open_temp_media_canvas(
        master=None,
        media_path=None,
        app_actions=None,
        skip_get_window_check=False,
    ) -> None:
        if media_path is None:
            return
        base_dir = os.path.dirname(media_path)
        if not skip_get_window_check:
            if (
                app_actions.get_window(
                    base_dir=base_dir,
                    media_path=media_path,
                    refocus=True,
                    disallow_if_compare_state=True,
                    new_media=True,
                )
                is not None
            ):
                return
        was_open = MediaDetails.temp_media_canvas is not None
        if MediaDetails.temp_media_canvas is None:
            MediaDetails.set_temp_media_canvas(
                master, media_path, app_actions
            )
        try:
            MediaDetails.temp_media_canvas.create_media(media_path)
        except Exception:
            # Re-create the canvas window if the old one was destroyed
            was_open = False
            MediaDetails.set_temp_media_canvas(
                master, media_path, app_actions
            )
            MediaDetails.temp_media_canvas.create_media(media_path)
        MediaDetails._check_temp_canvas_load(media_path, was_open, app_actions)

    @staticmethod
    def _check_temp_canvas_load(
        media_path: str, was_open: bool, app_actions
    ) -> None:
        """Close a freshly-opened canvas and alert the user if media failed to load."""
        from ui.app_window.media_frame import VideoUI
        canvas = MediaDetails.temp_media_canvas
        if canvas is None:
            return
        frame = canvas._media_frame
        # VLC video loading is async — media_displayed stays False until the
        # first frame renders, so skip the check for active VLC sessions.
        if frame.media_displayed or isinstance(frame._video_ui, VideoUI):
            return
        if not was_open:
            canvas.close()
            MediaDetails.temp_media_canvas = None
        app_actions._alert(
            _("Unable to open media"),
            _('Could not load: "{0}"').format(os.path.basename(media_path)),
        )

    @staticmethod
    def set_temp_media_canvas(
        master, media_path: str, app_actions
    ) -> None:
        width, height = MediaDetails._get_temp_canvas_dimensions(media_path)
        canvas = TempMediaWindow(
            parent=master,
            title=media_path,
            dimensions=f"{width}x{height}",
            app_actions=app_actions,
        )
        canvas.show()
        MediaDetails.temp_media_canvas = canvas

    @staticmethod
    def _get_temp_canvas_dimensions(
        media_path: str, max_width: int = 700
    ) -> tuple[int, int]:
        """Return (width, height) for TempMediaWindow canvas sizing.

        Tries QImageReader (fast, no external library) then PIL for static
        images.  Falls back to a sensible default for video, PDF, and other
        media that PIL cannot identify.
        """
        from PySide6.QtGui import QImageReader
        reader = QImageReader(media_path)
        size = reader.size()
        if size.isValid() and size.width() > 0 and size.height() > 0:
            w = min(max_width, size.width())
            return w, max(1, int(size.height() * w / size.width()))
        try:
            with Image.open(media_path) as image:
                if image.size[0] > 0 and image.size[1] > 0:
                    w = min(max_width, image.size[0])
                    return w, max(1, int(image.size[1] * w / image.size[0]))
        except Exception:
            pass
        return 800, 600

    # ── Image generation ─────────────────────────────────────────

    def _on_gen_mode_changed(self, _index: int) -> None:
        mode_value = self._gen_mode_combo.currentData()
        if mode_value is not None:
            MediaDetails.image_generation_mode = ImageGenerationType.get(mode_value)

    def set_image_generation_mode(self, event=None) -> None:
        mode_value = self._gen_mode_combo.currentData()
        if mode_value is not None:
            MediaDetails.image_generation_mode = ImageGenerationType.get(mode_value)

    def run_image_generation(self, event=None) -> None:
        MediaDetails.run_image_generation_static(self._app_actions)

    def run_redo_prompt(self, event=None) -> None:
        MediaDetails.run_image_generation_static(
            self._app_actions, _type=ImageGenerationType.REDO_PROMPT
        )

    @staticmethod
    def run_image_generation_static(
        app_actions, _type=None, modify_call=False, event=None
    ) -> None:
        if event is not None:
            if Utils.modifier_key_pressed(event, [ModifierKey.SHIFT]):
                _type = ImageGenerationType.CANCEL
            elif Utils.modifier_key_pressed(event, [ModifierKey.ALT]):
                _type = ImageGenerationType.REVERT_TO_SIMPLE_GEN
            else:
                _type = ImageGenerationType.LAST_SETTINGS
            app_actions.run_image_generation(
                _type=_type,
                media_path=MediaDetails.previous_image_generation_adapter_path,
                modify_call=modify_call,
            )
        else:
            if _type is None:
                _type = MediaDetails.image_generation_mode
            app_actions.run_image_generation(
                _type=_type, modify_call=modify_call
            )

    @staticmethod
    def get_image_specific_generation_mode():
        if MediaDetails.image_generation_mode in [
            ImageGenerationType.REDO_PROMPT,
            ImageGenerationType.TAKE_PROMPT,
            ImageGenerationType.CONTROL_NET,
            ImageGenerationType.IP_ADAPTER,
        ]:
            return MediaDetails.image_generation_mode
        return ImageGenerationType.CONTROL_NET

    # ── Tags ─────────────────────────────────────────────────────

    def update_tags(self) -> None:
        logger.info(f"Updating tags for {self._image_path}")
        tags_str = self._tags_entry.text()
        if tags_str == "":
            self._tags = []
        else:
            self._tags = [t.strip() for t in tags_str.split(",")]
        image_data_extractor.set_tags(self._image_path, self._tags)
        logger.info("Updated tags for " + self._image_path)
        self._app_actions.success(_("Updated tags for {0}").format(self._image_path))

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def has_closed(self) -> bool:
        return self._has_closed

    @property
    def do_refresh(self) -> bool:
        return self._do_refresh

    def close_windows(self, event=None) -> None:
        self._app_actions.set_media_details_window(None)
        self._has_closed = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._app_actions.set_media_details_window(None)
        self._has_closed = True
        super().closeEvent(event)

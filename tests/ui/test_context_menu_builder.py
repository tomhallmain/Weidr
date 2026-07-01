"""
Tests for ContextMenuBuilder media-type-dependent action visibility.

Each test patches get_media_type_for_path to a fixed MediaType value and
inspects which action labels are added to the menu.  Directory-scoped actions
(Run Image Generation on Directory, prevalidation controls, etc.) are expected
to appear unconditionally regardless of the active media type.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu

from files.marked_files import MarkedFiles
from ui.app_window.context_menu_builder import ContextMenuBuilder
from utils.constants import MediaType
from utils.translations import _


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NoExecMenu(QMenu):
    """QMenu subclass that suppresses exec() so tests don't block."""
    def exec(self, pos=None):
        pass


def _make_app_mock(media_path: str, base_dir: str = "/some/dir") -> MagicMock:
    app = MagicMock()
    app.media_navigator.get_active_media_filepath.return_value = media_path
    app.get_base_dir.return_value = base_dir
    return app


def _build_menu_actions(monkeypatch, media_path: str, media_type: MediaType) -> list[str]:
    """Return the text of every non-separator action built for *media_type*."""
    captured: list[_NoExecMenu] = []

    def _menu_factory(parent):
        m = _NoExecMenu(None)
        captured.append(m)
        return m

    monkeypatch.setattr("ui.app_window.context_menu_builder.QMenu", _menu_factory)
    monkeypatch.setattr(
        "ui.app_window.context_menu_builder.get_media_type_for_path",
        lambda path: media_type,
    )

    ContextMenuBuilder(_make_app_mock(media_path)).show(QPoint(0, 0))

    assert captured, "QMenu was never constructed"
    return [a.text() for a in captured[0].actions() if not a.isSeparator()]


# ---------------------------------------------------------------------------
# Action label constants (must match strings in context_menu_builder.py)
# ---------------------------------------------------------------------------

_OPEN_IN_GIMP           = _("Open in GIMP")
_RUN_IMAGE_GEN          = _("Run Image Generation")
_RUN_IMAGE_GEN_DIR      = _("Run Image Generation on Directory")
_REDO_EDIT_SUFFIX       = _("Redo image edit from suffix")
_SAVE_WITHOUT_AUDIO     = _("Save copy without audio")
_SAVE_WITHOUT_METADATA  = _("Save copy without metadata")
_CUT_VIDEO              = _("Cut video at current position…")
_RUN_PREVALIDATIONS_DIR = _("Run Prevalidations on Directory")
_INTERACTIVE_CROP       = _("Interactive Crop…")
_INTERACTIVE_BOX        = _("Interactive Box…")

_SET_RELATED  = _("Set Marked File as Related Image of Current")

_VIDEO_ONLY = {_SAVE_WITHOUT_AUDIO, _SAVE_WITHOUT_METADATA, _CUT_VIDEO}
_IMAGE_ONLY = {_RUN_IMAGE_GEN, _REDO_EDIT_SUFFIX}
_DIR_SCOPED = {_RUN_IMAGE_GEN_DIR, _RUN_PREVALIDATIONS_DIR}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImageMediaType:
    def test_image_actions(self, qapp, monkeypatch):
        actions = _build_menu_actions(monkeypatch, "/dir/file.png", MediaType.IMAGE)
        assert _OPEN_IN_GIMP in actions
        assert _RUN_IMAGE_GEN in actions
        assert _REDO_EDIT_SUFFIX in actions
        assert not _VIDEO_ONLY & set(actions)
        assert _DIR_SCOPED <= set(actions)
        assert _INTERACTIVE_CROP in actions
        assert _INTERACTIVE_BOX in actions


class TestVideoMediaType:
    def test_video_actions(self, qapp, monkeypatch):
        actions = _build_menu_actions(monkeypatch, "/dir/clip.mp4", MediaType.VIDEO)
        assert _SAVE_WITHOUT_AUDIO in actions
        assert _SAVE_WITHOUT_METADATA in actions
        assert _CUT_VIDEO in actions
        assert _OPEN_IN_GIMP not in actions
        assert not _IMAGE_ONLY & set(actions)
        assert _DIR_SCOPED <= set(actions)
        assert _INTERACTIVE_CROP in actions
        assert _INTERACTIVE_BOX in actions


class TestGifMediaType:
    def test_gif_actions(self, qapp, monkeypatch):
        actions = _build_menu_actions(monkeypatch, "/dir/anim.gif", MediaType.GIF)
        assert _OPEN_IN_GIMP in actions
        assert not _IMAGE_ONLY & set(actions)
        assert not _VIDEO_ONLY & set(actions)
        assert _INTERACTIVE_CROP in actions
        assert _INTERACTIVE_BOX in actions


class TestSvgMediaType:
    def test_svg_actions(self, qapp, monkeypatch):
        actions = _build_menu_actions(monkeypatch, "/dir/icon.svg", MediaType.SVG)
        assert _OPEN_IN_GIMP in actions
        assert not _IMAGE_ONLY & set(actions)
        assert not _VIDEO_ONLY & set(actions)
        assert _INTERACTIVE_CROP in actions
        assert _INTERACTIVE_BOX in actions


class TestPdfMediaType:
    def test_pdf_actions(self, qapp, monkeypatch):
        actions = _build_menu_actions(monkeypatch, "/dir/doc.pdf", MediaType.PDF)
        assert _OPEN_IN_GIMP not in actions
        assert not _IMAGE_ONLY & set(actions)
        assert not _VIDEO_ONLY & set(actions)
        assert _DIR_SCOPED <= set(actions)
        assert _INTERACTIVE_CROP in actions
        assert _INTERACTIVE_BOX in actions


class TestSetAsRelatedImageMenuItem:
    """Menu item visibility: IMAGE + exactly 1 mark required."""

    @pytest.fixture(autouse=True)
    def _restore_marks(self):
        saved = MarkedFiles.file_marks[:]
        yield
        MarkedFiles.file_marks = saved[:]

    def test_appears_for_image_with_one_mark(self, qapp, monkeypatch):
        MarkedFiles.file_marks = ["/dir/marked.png"]
        actions = _build_menu_actions(monkeypatch, "/dir/source.png", MediaType.IMAGE)
        assert _SET_RELATED in actions

    def test_hidden_when_no_marks(self, qapp, monkeypatch):
        MarkedFiles.file_marks = []
        actions = _build_menu_actions(monkeypatch, "/dir/source.png", MediaType.IMAGE)
        assert _SET_RELATED not in actions

    def test_hidden_for_non_image_media_type(self, qapp, monkeypatch):
        MarkedFiles.file_marks = ["/dir/marked.png"]
        actions = _build_menu_actions(monkeypatch, "/dir/clip.mp4", MediaType.VIDEO)
        assert _SET_RELATED not in actions

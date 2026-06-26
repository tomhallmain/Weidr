"""Tests for FileMarksController.set_marked_file_as_related_to_current."""

from __future__ import annotations

from unittest.mock import MagicMock

import PySide6.QtWidgets
import pytest

from files.marked_files import MarkedFiles


def _fake_dialog(suffix: str, ok: bool = True):
    fake = MagicMock()
    fake.getText = staticmethod(lambda *a, **kw: (suffix, ok))
    return fake


class TestSetMarkedFileAsRelatedToCurrent:

    @pytest.fixture(autouse=True)
    def _restore_marks(self):
        saved = MarkedFiles.file_marks[:]
        was_performing = MarkedFiles.is_performing_action
        yield
        MarkedFiles.file_marks = saved[:]
        MarkedFiles.is_performing_action = was_performing

    def test_renames_file_and_updates_mark(
        self, tmp_path, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        source = tmp_path / "source.png"
        marked = tmp_path / "marked.png"
        source.write_bytes(b"")
        marked.write_bytes(b"pixel data")

        win.media_path = str(source)
        MarkedFiles.file_marks = [str(marked)]

        monkeypatch.setattr(PySide6.QtWidgets, "QInputDialog", _fake_dialog("edit"))
        win.file_marks_ctrl.set_marked_file_as_related_to_current()

        expected = tmp_path / "source_edit.png"
        assert expected.exists() and expected.read_bytes() == b"pixel data"
        assert not marked.exists()
        assert MarkedFiles.file_marks == [str(expected)]

    def test_invalid_suffix_leaves_file_unchanged(
        self, tmp_path, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        source = tmp_path / "source.png"
        marked = tmp_path / "marked.png"
        source.write_bytes(b"")
        marked.write_bytes(b"")

        win.media_path = str(source)
        MarkedFiles.file_marks = [str(marked)]

        monkeypatch.setattr(PySide6.QtWidgets, "QInputDialog", _fake_dialog("123"))
        win.file_marks_ctrl.set_marked_file_as_related_to_current()

        assert marked.exists()
        assert MarkedFiles.file_marks == [str(marked)]

    def test_no_rename_when_target_already_exists(
        self, tmp_path, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        source = tmp_path / "source.png"
        marked = tmp_path / "marked.png"
        collision = tmp_path / "source_edit.png"
        source.write_bytes(b"")
        marked.write_bytes(b"")
        collision.write_bytes(b"original")

        win.media_path = str(source)
        MarkedFiles.file_marks = [str(marked)]

        monkeypatch.setattr(PySide6.QtWidgets, "QInputDialog", _fake_dialog("edit"))
        win.file_marks_ctrl.set_marked_file_as_related_to_current()

        assert marked.exists()
        assert collision.read_bytes() == b"original"

    def test_no_dialog_when_no_marks(
        self, tmp_path, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        source = tmp_path / "source.png"
        source.write_bytes(b"")
        win.media_path = str(source)
        MarkedFiles.file_marks = []

        opened = []
        fake = MagicMock()
        fake.getText = staticmethod(lambda *a, **kw: opened.append(True) or ("edit", True))
        monkeypatch.setattr(PySide6.QtWidgets, "QInputDialog", fake)

        win.file_marks_ctrl.set_marked_file_as_related_to_current()

        assert not opened

    def test_no_dialog_while_transfer_running(
        self, tmp_path, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        source = tmp_path / "source.png"
        marked = tmp_path / "marked.png"
        source.write_bytes(b"")
        marked.write_bytes(b"")

        win.media_path = str(source)
        MarkedFiles.file_marks = [str(marked)]
        MarkedFiles.is_performing_action = True

        opened = []
        fake = MagicMock()
        fake.getText = staticmethod(lambda *a, **kw: opened.append(True) or ("edit", True))
        monkeypatch.setattr(PySide6.QtWidgets, "QInputDialog", fake)

        win.file_marks_ctrl.set_marked_file_as_related_to_current()

        assert not opened
        assert marked.exists()

"""Diff PDFs: an ePub input is passed to diff-pdf as its derived PDF."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pypdfium2")

import files.pdf_creator as pdf_creator
from files.pdf_creator import PDFCreator
from utils.config import config
from utils.translations import _


def _run_diff(monkeypatch, tmp_path, derived):
    book = tmp_path / "book.epub"
    book.write_bytes(b"x")
    other = tmp_path / "other.pdf"
    other.write_bytes(b"x")
    monkeypatch.setattr(config, "enable_epubs", True)
    monkeypatch.setattr(pdf_creator.shutil, "which", lambda name: "/usr/bin/diff-pdf")
    monkeypatch.setattr(pdf_creator, "get_paged_document_pdf", derived)
    commands = []

    def _fake_run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(pdf_creator.subprocess, "run", _fake_run)
    actions = MagicMock()
    PDFCreator.create_diff_pdf_from_files(
        [str(book), str(other)], actions, output_path=str(tmp_path / "out.pdf")
    )
    return commands, actions, str(other)


def test_epub_passes_derived_pdf(monkeypatch, tmp_path):
    derived = str(tmp_path / "derived_from_epub.pdf")
    commands, _actions, other = _run_diff(monkeypatch, tmp_path, lambda path: derived)
    assert len(commands) == 1
    assert commands[0][-2:] == [derived, other]


def test_epub_render_failure_alerts(monkeypatch, tmp_path):
    def _fail(path):
        raise RuntimeError("render failed")

    commands, actions, _other = _run_diff(monkeypatch, tmp_path, _fail)
    assert commands == []
    message = actions.alert.call_args[0][1]
    assert message == _("Failed to render ePub for diff: {0}").format(str(tmp_path / "book.epub"))

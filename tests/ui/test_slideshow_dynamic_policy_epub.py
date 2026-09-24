"""Slideshow page dwell for ePubs: never builds the derived PDF from a tick."""

import pytest

from image.frame_cache import FrameCache
from ui.app_window import slideshow_dynamic_policy as policy
from utils.config import config


@pytest.fixture
def book(tmp_path, monkeypatch):
    path = tmp_path / "book.epub"
    path.write_bytes(b"x")
    monkeypatch.setattr(config, "enable_epubs", True)
    monkeypatch.setattr(FrameCache, "epub_pdf_cache", {})

    def _no_build(cls, epub_path):
        raise AssertionError("slideshow policy started an ePub build")

    monkeypatch.setattr(FrameCache, "get_epub_pdf", classmethod(_no_build))
    return str(path)


def test_intrinsic_budget_is_one_before_build(book):
    assert policy._pdf_effective_page_budget(book, -1) == 1


def test_fixed_budget_unchanged(book):
    assert policy._pdf_effective_page_budget(book, 3) == 3


def test_intrinsic_budget_uses_cached_build(book, tmp_path, monkeypatch):
    pdfium = pytest.importorskip("pypdfium2")
    derived = str(tmp_path / "derived.pdf")
    pdf = pdfium.PdfDocument.new()
    for _ in range(5):
        pdf.new_page(100, 100)
    pdf.save(derived)
    pdf.close()
    monkeypatch.setattr(FrameCache, "epub_pdf_cache", {book: derived})
    assert policy._pdf_effective_page_budget(book, -1) == 5


def test_epub_uses_page_dwell_rule(book, monkeypatch):
    monkeypatch.setattr(config, "slideshow_dynamic_pdf_max_pages", -1)
    assert policy.skip_classic_slideshow_primary_tick(None, book)
    assert policy.slideshow_poll_should_run(None, book)


def test_disabled_epub_has_no_dwell_rule(book, monkeypatch):
    monkeypatch.setattr(config, "enable_epubs", False)
    monkeypatch.setattr(config, "slideshow_dynamic_pdf_max_pages", -1)
    assert not policy.skip_classic_slideshow_primary_tick(None, book)

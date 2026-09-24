"""FrameCache ePub support: cover fast path, derived-PDF paging and sampling,
fallbacks. The Chromium build is replaced by a stub that writes an N-page PDF."""

import os
from types import SimpleNamespace

import pytest

pdfium = pytest.importorskip("pypdfium2")

import image.frame_cache as frame_cache_module
from image.epub_document import EpubError
from image.frame_cache import FrameCache
from utils.config import config
from utils.media_utils import get_paged_document_pdf, get_pdf_page_count
from tests.fixtures.epub_fixtures import build_epub

AES = "http://www.w3.org/2001/04/xmlenc#aes128-cbc"


@pytest.fixture
def fc(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    monkeypatch.setattr(FrameCache, "temporary_directory", SimpleNamespace(name=str(frames)))
    for attr in ("cache", "sampled_cache", "media_stats_cache", "epub_pdf_cache", "epub_failures", "_epub_build_locks"):
        monkeypatch.setattr(FrameCache, attr, {})
    monkeypatch.setattr(config, "enable_epubs", True)
    monkeypatch.setattr(frame_cache_module, "has_imported_pyppeteer", True)
    return FrameCache


def _write_pdf(path, n_pages):
    pdf = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        pdf.new_page(200, 300)
    pdf.save(path)
    pdf.close()


@pytest.fixture
def stub_build(fc, monkeypatch):
    """Replace the Chromium build; records each call."""
    calls = []

    def _fake(cls, epub_path, n_pages=4):
        calls.append(epub_path)
        out = os.path.join(cls.temporary_directory.name, "derived.pdf")
        _write_pdf(out, n_pages)
        return out, n_pages

    monkeypatch.setattr(FrameCache, "_build_epub_pdf", classmethod(_fake))
    return calls


class TestCoverFastPath:
    def test_get_image_path_returns_cover_without_building(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub", cover="epub3")
        frame = fc.get_image_path(epub)
        assert frame != epub
        assert frame.endswith(".jpg") and os.path.isfile(frame)
        assert stub_build == []

    def test_no_cover_builds_and_uses_page_zero(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub", cover=None)
        frame = fc.get_image_path(epub)
        assert stub_build == [epub]
        assert os.path.isfile(frame)
        assert fc.media_stats_cache[epub].total_items == 4


class TestPagingThroughDerivedPdf:
    def test_pages_and_stats_keyed_by_epub_path(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub")
        page = fc.get_pdf_page(epub, 2)
        assert os.path.isfile(page)
        assert fc.media_stats_cache[epub].media_type == "pdf"
        assert fc.media_stats_cache[epub].total_items == 4
        assert epub in fc.epub_pdf_cache
        derived = fc.epub_pdf_cache[epub]
        assert derived not in fc.media_stats_cache
        assert derived not in fc.cache

    def test_build_happens_once(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub")
        fc.get_pdf_page(epub, 0)
        fc.get_pdf_page(epub, 1)
        get_paged_document_pdf(epub)
        assert stub_build == [epub]

    def test_page_zero_replaces_cover_in_cache(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub")
        cover = fc.get_image_path(epub)
        page0 = fc.get_pdf_page(epub, 0)
        assert fc.cache[epub] == page0 != cover

    def test_page_count(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub")
        assert get_pdf_page_count(epub, build=False) == 0
        assert stub_build == []
        assert get_pdf_page_count(epub) == 4

    def test_sampling_keyed_by_epub_path(self, fc, stub_build, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "dynamic_media_min_sample_count", 2)
        monkeypatch.setattr(config, "dynamic_media_max_sample_pages", 10)
        epub = build_epub(tmp_path / "book.epub")
        samples = fc.get_frame_samples(epub, sample_ratio=1.0)
        assert len(samples) == 4
        assert all(p.endswith(".jpg") and os.path.isfile(p) for p in samples)
        assert any(key.startswith(epub + "|") for key in fc.sampled_cache)
        assert fc.media_stats_cache[epub].total_items == 4


class TestUnpageable:
    def test_build_failure_caches_placeholder_not_source(self, fc, monkeypatch, tmp_path):
        def _boom(cls, epub_path):
            raise RuntimeError("chromium failed")

        monkeypatch.setattr(FrameCache, "_build_epub_pdf", classmethod(_boom))
        epub = build_epub(tmp_path / "book.epub", cover=None)
        frame = fc.get_image_path(epub)
        assert frame != epub
        assert frame.endswith(".jpg") and os.path.isfile(frame)
        assert fc.is_epub_unpageable(epub)
        assert fc.media_stats_cache[epub].total_items == 1

    def test_failure_is_not_retried(self, fc, monkeypatch, tmp_path):
        calls = []

        def _boom(cls, epub_path):
            calls.append(epub_path)
            raise RuntimeError("chromium failed")

        monkeypatch.setattr(FrameCache, "_build_epub_pdf", classmethod(_boom))
        epub = build_epub(tmp_path / "book.epub")
        for _ in range(2):
            with pytest.raises(EpubError):
                fc.get_epub_pdf(epub)
        assert len(calls) == 1

    def test_broken_zip_gets_placeholder(self, fc, stub_build, tmp_path):
        epub = tmp_path / "broken.epub"
        epub.write_bytes(b"not a zip")
        frame = fc.get_image_path(str(epub))
        assert frame != str(epub) and os.path.isfile(frame)
        assert stub_build == []

    def test_drm_uses_cover_and_never_builds(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub", encryption_entries=[(AES, "OEBPS/chapter1.xhtml")])
        frame = fc.get_image_path(epub)
        assert os.path.isfile(frame) and frame != epub
        assert fc.get_pdf_page(epub, 0) == frame
        with pytest.raises(IndexError):
            fc.get_pdf_page(epub, 1)
        assert fc.media_stats_cache[epub].total_items == 1
        assert stub_build == []

    def test_sampling_unpageable_yields_frame_not_epub(self, fc, monkeypatch, tmp_path):
        def _boom(cls, epub_path):
            raise RuntimeError("chromium failed")

        monkeypatch.setattr(FrameCache, "_build_epub_pdf", classmethod(_boom))
        epub = build_epub(tmp_path / "book.epub")
        samples = fc.get_frame_samples(epub)
        assert samples and epub not in samples
        assert all(os.path.isfile(p) for p in samples)


class TestRemoveFromCache:
    def test_drops_derived_pdf(self, fc, stub_build, tmp_path):
        epub = build_epub(tmp_path / "book.epub")
        derived = fc.get_epub_pdf(epub)
        fc.remove_from_cache(epub, delete_temp_file=True)
        assert epub not in fc.epub_pdf_cache
        assert not os.path.exists(derived)
        assert os.path.isfile(epub)

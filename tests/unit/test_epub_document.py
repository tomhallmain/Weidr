"""Unit tests for image/epub_document.py: OPF parsing, DRM detection, safe extraction."""

import zipfile

import pytest

from image import epub_document
from image.epub_document import (
    EpubError,
    document_only_wraps_image,
    is_url_inside,
    read_cover_bytes,
    read_epub_info,
    safe_extract,
    viewport_size,
)
from tests.fixtures.epub_fixtures import COVER_WRAPPER_XHTML, build_epub, chapter_xhtml

IDPF_OBFUSCATION = "http://www.idpf.org/2008/embedding"
ADOBE_OBFUSCATION = "http://ns.adobe.com/pdf/enc#RC"
AES = "http://www.w3.org/2001/04/xmlenc#aes128-cbc"


def _info(path):
    with zipfile.ZipFile(path) as zf:
        return read_epub_info(zf)


class TestReadEpubInfo:
    def test_epub3_cover_image_property(self, tmp_path):
        info = _info(build_epub(tmp_path / "b.epub", cover="epub3"))
        assert info.cover_image_name == "OEBPS/images/cover.png"

    def test_epub2_cover_meta(self, tmp_path):
        info = _info(build_epub(tmp_path / "b.epub", cover="epub2"))
        assert info.cover_image_name == "OEBPS/images/cover.png"

    def test_no_cover(self, tmp_path):
        info = _info(build_epub(tmp_path / "b.epub", cover=None))
        assert info.cover_image_name is None

    def test_spine_resolved_against_opf_dir_and_non_linear_skipped(self, tmp_path):
        path = build_epub(
            tmp_path / "b.epub",
            chapters=[("c1", "text/one.xhtml", "yes"), ("notes", "notes.xhtml", "no"), ("c2", "two.xhtml", "yes")],
        )
        info = _info(path)
        assert [s.name for s in info.spine] == ["OEBPS/text/one.xhtml", "OEBPS/two.xhtml"]

    def test_fixed_layout_detected(self, tmp_path):
        info = _info(build_epub(tmp_path / "b.epub", fixed_layout=True))
        assert all(s.fixed_layout for s in info.spine)

    def test_reflowable_by_default(self, tmp_path):
        info = _info(build_epub(tmp_path / "b.epub"))
        assert not any(s.fixed_layout for s in info.spine)

    def test_empty_spine_raises(self, tmp_path):
        path = build_epub(tmp_path / "b.epub", chapters=[("n", "n.xhtml", "no")])
        with pytest.raises(EpubError):
            _info(path)

    def test_missing_container_raises(self, tmp_path):
        path = tmp_path / "b.epub"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
        with pytest.raises(EpubError):
            _info(path)


class TestDrmDetection:
    def test_font_obfuscation_only_is_not_drm(self, tmp_path):
        path = build_epub(
            tmp_path / "b.epub",
            encryption_entries=[
                (IDPF_OBFUSCATION, "OEBPS/fonts/a.otf"),
                (ADOBE_OBFUSCATION, "OEBPS/fonts/b.otf"),
            ],
        )
        assert not _info(path).drm

    def test_other_algorithm_on_content_document_is_drm(self, tmp_path):
        path = build_epub(tmp_path / "b.epub", encryption_entries=[(AES, "OEBPS/chapter1.xhtml")])
        assert _info(path).drm

    def test_encrypted_cover_is_not_read(self, tmp_path):
        path = build_epub(tmp_path / "b.epub", encryption_entries=[(AES, "OEBPS/images/cover.png")])
        with zipfile.ZipFile(path) as zf:
            info = read_epub_info(zf)
            assert read_cover_bytes(zf, info) is None

    def test_unencrypted_cover_is_read(self, tmp_path):
        path = build_epub(tmp_path / "b.epub", encryption_entries=[(AES, "OEBPS/chapter1.xhtml")])
        with zipfile.ZipFile(path) as zf:
            info = read_epub_info(zf)
            assert read_cover_bytes(zf, info)


class TestSafeExtract:
    def _zip_with(self, path, name, data=b"x"):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(name, data)
        return path

    @pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/abs/evil.txt", "C:/evil.txt"])
    def test_unsafe_names_rejected(self, tmp_path, name):
        path = self._zip_with(tmp_path / "b.epub", name)
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(path) as zf, pytest.raises(EpubError):
            safe_extract(zf, str(dest))
        assert not (tmp_path / "evil.txt").exists()

    def test_oversized_total_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(epub_document, "MAX_TOTAL_UNCOMPRESSED_BYTES", 100)
        path = self._zip_with(tmp_path / "b.epub", "big.bin", b"0" * 1000)
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(path) as zf, pytest.raises(EpubError):
            safe_extract(zf, str(dest))

    def test_too_many_entries_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(epub_document, "MAX_ENTRY_COUNT", 2)
        path = tmp_path / "b.epub"
        with zipfile.ZipFile(path, "w") as zf:
            for i in range(3):
                zf.writestr(f"f{i}.txt", "x")
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(path) as zf, pytest.raises(EpubError):
            safe_extract(zf, str(dest))

    def test_normal_book_extracts(self, tmp_path):
        path = build_epub(tmp_path / "b.epub")
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(path) as zf:
            safe_extract(zf, str(dest))
        assert (dest / "OEBPS" / "chapter1.xhtml").is_file()


class TestDocumentHelpers:
    def test_cover_wrapper_detected(self):
        assert document_only_wraps_image(COVER_WRAPPER_XHTML, "OEBPS/cover.xhtml", "OEBPS/images/cover.png")

    def test_wrapper_of_other_image_not_detected(self):
        assert not document_only_wraps_image(COVER_WRAPPER_XHTML, "OEBPS/cover.xhtml", "OEBPS/images/other.png")

    def test_chapter_with_text_not_detected(self):
        text = chapter_xhtml('Once upon a time <img src="images/cover.png"/>')
        assert not document_only_wraps_image(text, "OEBPS/c.xhtml", "OEBPS/images/cover.png")

    def test_viewport_size(self):
        text = chapter_xhtml("x", '<meta name="viewport" content="width=600, height=800"/>')
        assert viewport_size(text) == (600, 800)

    def test_viewport_missing(self):
        assert viewport_size(chapter_xhtml("x")) is None

    def test_is_url_inside(self, tmp_path):
        inside = tmp_path / "book" / "a.css"
        inside.parent.mkdir()
        inside.write_text("")
        assert is_url_inside(inside.as_uri(), str(tmp_path))
        assert not is_url_inside((tmp_path.parent / "x.css").as_uri(), str(tmp_path))
        assert not is_url_inside("https://example.com/tracker.gif", str(tmp_path))
        assert is_url_inside("data:image/png;base64,AAAA", str(tmp_path))

"""Tests for compare/example_pipelines.py: packing the example pipeline JSON
files into the encrypted archive and extracting them again.

Every test works on its own temp directory and archive; the repository's
assets are not read or written.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from compare import example_pipelines


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.json").write_text('{"name": "one", "nodes": []}', encoding="utf-8")
    (source / "two.json").write_text('{"name": "two", "nodes": []}', encoding="utf-8")
    (source / "notes.txt").write_text("not packed", encoding="utf-8")
    return source, tmp_path / "extracted", str(tmp_path / "examples.enc")


def _files(directory) -> dict:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.suffix == ".json"}


class TestPackAndExtract:
    def test_pack_returns_the_json_file_names(self, paths):
        source, _dest, archive = paths
        assert example_pipelines.pack(str(source), archive) == ["one.json", "two.json"]

    def test_archive_is_not_plain_json_or_zip(self, paths):
        source, _dest, archive = paths
        example_pipelines.pack(str(source), archive)
        with open(archive, "rb") as f:
            data = f.read()
        assert b'"name"' not in data
        assert not zipfile.is_zipfile(io.BytesIO(data))

    def test_read_archive_returns_the_packed_files(self, paths):
        source, _dest, archive = paths
        example_pipelines.pack(str(source), archive)
        assert example_pipelines.read_archive(archive) == _files(source)

    def test_extracts_into_an_empty_directory(self, paths):
        source, dest, archive = paths
        example_pipelines.pack(str(source), archive)
        assert example_pipelines.ensure_extracted(str(dest), archive) is True
        assert _files(dest) == _files(source)

    def test_unchanged_archive_keeps_local_edits(self, paths):
        source, dest, archive = paths
        example_pipelines.pack(str(source), archive)
        example_pipelines.ensure_extracted(str(dest), archive)
        (dest / "one.json").write_text("edited", encoding="utf-8")
        assert example_pipelines.ensure_extracted(str(dest), archive) is False
        assert (dest / "one.json").read_text(encoding="utf-8") == "edited"

    def test_packing_a_directory_marks_it_as_extracted(self, paths):
        source, _dest, archive = paths
        example_pipelines.pack(str(source), archive)
        assert example_pipelines.ensure_extracted(str(source), archive) is False

    def test_changed_archive_overwrites_its_files_and_keeps_others(self, paths, tmp_path):
        source, dest, archive = paths
        example_pipelines.pack(str(source), archive)
        example_pipelines.ensure_extracted(str(dest), archive)
        (dest / "local_only.json").write_text("{}", encoding="utf-8")
        (source / "one.json").write_text('{"name": "one v2", "nodes": []}', encoding="utf-8")
        example_pipelines.pack(str(source), archive)
        assert example_pipelines.ensure_extracted(str(dest), archive) is True
        assert (dest / "one.json").read_bytes() == (source / "one.json").read_bytes()
        assert (dest / "local_only.json").exists()

    def test_missing_archive_extracts_nothing(self, tmp_path):
        dest = tmp_path / "dest"
        assert example_pipelines.ensure_extracted(str(dest), str(tmp_path / "missing.enc")) is False
        assert not dest.exists()

    def test_unreadable_archive_extracts_nothing(self, tmp_path):
        archive = tmp_path / "bad.enc"
        archive.write_bytes(b"\x00" * 64)
        dest = tmp_path / "dest"
        assert example_pipelines.ensure_extracted(str(dest), str(archive)) is False
        assert not dest.exists()

    def test_archive_from_another_app_identifier_is_unreadable(self, paths, monkeypatch):
        source, dest, archive = paths
        example_pipelines.pack(str(source), archive)
        monkeypatch.setattr(example_pipelines.AppInfo, "APP_IDENTIFIER", "another_app")
        assert example_pipelines.ensure_extracted(str(dest), archive) is False

    def test_entries_that_are_not_plain_json_names_are_ignored(self, tmp_path):
        from utils.encryptor import symmetric_encrypt_data_to_file
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("ok.json", "{}")
            zf.writestr("../escape.json", "{}")
            zf.writestr("sub/nested.json", "{}")
            zf.writestr("readme.txt", "text")
        archive = str(tmp_path / "crafted.enc")
        symmetric_encrypt_data_to_file(buffer.getvalue(), archive, example_pipelines._passphrase(),
                                       compress=False)
        assert set(example_pipelines.read_archive(archive)) == {"ok.json"}
        dest = tmp_path / "dest"
        example_pipelines.ensure_extracted(str(dest), archive)
        assert not (tmp_path / "escape.json").exists()
        assert sorted(p.name for p in dest.iterdir() if p.suffix == ".json") == ["ok.json"]

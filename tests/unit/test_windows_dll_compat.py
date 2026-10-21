"""Conda Cairo DLLs must not be the copies Qt and fontconfig bind to."""

import json

import pytest

from utils import windows_dll_compat as compat


def test_system_icu_is_loaded_before_qt(tmp_path, monkeypatch):
    system32 = tmp_path / "System32"
    system32.mkdir()
    icu = system32 / "icuuc.dll"
    icu.write_bytes(b"icu")
    loaded = []
    monkeypatch.setattr(compat.sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.setattr(compat.ctypes, "WinDLL", lambda path: loaded.append(path))

    compat.prefer_system_icu()

    assert loaded == [str(icu)]


def test_system_icu_preload_is_windows_only(monkeypatch):
    monkeypatch.setattr(compat.sys, "platform", "linux")
    monkeypatch.setattr(compat.ctypes, "WinDLL", lambda path: pytest.fail(path))

    compat.prefer_system_icu()


def test_packaged_libintl_preloads_when_installed_copy_is_smaller(tmp_path, monkeypatch):
    bin_dir = tmp_path / "Library" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "intl-8.dll").write_bytes(b"x" * 100)

    extracted = tmp_path / "pkgs" / "libintl"
    extracted_bin = extracted / "Library" / "bin"
    extracted_bin.mkdir(parents=True)
    packaged = extracted_bin / "intl-8.dll"
    packaged.write_bytes(b"y" * 500)
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    (meta / "libintl-0.22.5-h1.json").write_text(
        json.dumps({"extracted_package_dir": str(extracted)}),
        encoding="utf-8",
    )

    loaded = []
    monkeypatch.setattr(compat.sys, "platform", "win32")
    monkeypatch.setattr(compat.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(compat.ctypes, "WinDLL", lambda path: loaded.append(path))

    compat.prefer_packaged_libintl()

    assert loaded == [str(packaged)]


def test_packaged_libintl_is_left_alone_when_installed_copy_is_larger(tmp_path, monkeypatch):
    bin_dir = tmp_path / "Library" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "intl-8.dll").write_bytes(b"x" * 500)

    extracted = tmp_path / "pkgs" / "libintl"
    extracted_bin = extracted / "Library" / "bin"
    extracted_bin.mkdir(parents=True)
    (extracted_bin / "intl-8.dll").write_bytes(b"y" * 100)
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    (meta / "libintl-0.22.5-h1.json").write_text(
        json.dumps({"extracted_package_dir": str(extracted)}),
        encoding="utf-8",
    )

    monkeypatch.setattr(compat.sys, "platform", "win32")
    monkeypatch.setattr(compat.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(compat.ctypes, "WinDLL", lambda path: pytest.fail(path))

    compat.prefer_packaged_libintl()


def test_frozen_executable_preloads_libintl_from_conda_prefix(tmp_path, monkeypatch):
    unpack_dir = tmp_path / "unpacked"
    unpack_dir.mkdir()
    conda_prefix = tmp_path / "conda"
    bin_dir = conda_prefix / "Library" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "intl-8.dll").write_bytes(b"x" * 100)
    extracted = tmp_path / "pkgs" / "libintl"
    extracted_bin = extracted / "Library" / "bin"
    extracted_bin.mkdir(parents=True)
    packaged = extracted_bin / "intl-8.dll"
    packaged.write_bytes(b"y" * 500)
    meta = conda_prefix / "conda-meta"
    meta.mkdir()
    (meta / "libintl-0.22.5-h1.json").write_text(
        json.dumps({"extracted_package_dir": str(extracted)}),
        encoding="utf-8",
    )

    loaded = []
    monkeypatch.setattr(compat.sys, "platform", "win32")
    monkeypatch.setattr(compat.sys, "prefix", str(unpack_dir))
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setattr(compat.ctypes, "WinDLL", lambda path: loaded.append(path))

    compat.prefer_packaged_libintl()

    assert loaded == [str(packaged)]


def test_missing_package_cache_skips_libintl_preload(tmp_path, monkeypatch):
    bin_dir = tmp_path / "Library" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "intl-8.dll").write_bytes(b"x" * 100)
    meta = tmp_path / "conda-meta"
    meta.mkdir()
    (meta / "libintl-0.22.5-h1.json").write_text(
        json.dumps({"extracted_package_dir": str(tmp_path / "missing")}),
        encoding="utf-8",
    )

    monkeypatch.setattr(compat.sys, "platform", "win32")
    monkeypatch.setattr(compat.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(compat.ctypes, "WinDLL", lambda path: pytest.fail(path))

    compat.prefer_packaged_libintl()

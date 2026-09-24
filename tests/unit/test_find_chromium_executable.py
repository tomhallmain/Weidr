"""Browser resolution for pyppeteer (HTML and ePub rendering)."""

import sys
import types

import pytest

import image.frame_cache as frame_cache_module
from image.frame_cache import find_chromium_executable
from utils.config import config


@pytest.fixture
def downloaded(monkeypatch):
    """Control whether pyppeteer reports its own Chromium as downloaded."""
    state = {"present": False}
    fake = types.ModuleType("pyppeteer.chromium_downloader")
    fake.check_chromium = lambda: state["present"]
    monkeypatch.setitem(sys.modules, "pyppeteer.chromium_downloader", fake)
    return state


def _exe(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"")
    return str(path)


def test_configured_path_wins(tmp_path, monkeypatch, downloaded):
    downloaded["present"] = True
    exe = _exe(tmp_path, "chrome.exe")
    monkeypatch.setattr(config, "chromium_exe_loc", exe)
    assert find_chromium_executable() == exe


def test_downloaded_chromium_preferred_over_installed(tmp_path, monkeypatch, downloaded):
    downloaded["present"] = True
    monkeypatch.setattr(config, "chromium_exe_loc", None)
    monkeypatch.setattr(frame_cache_module, "_installed_browser_candidates", lambda: [_exe(tmp_path, "msedge.exe")])
    assert find_chromium_executable() is None


def test_installed_browser_used_when_nothing_downloaded(tmp_path, monkeypatch, downloaded):
    monkeypatch.setattr(config, "chromium_exe_loc", None)
    edge = _exe(tmp_path, "msedge.exe")
    monkeypatch.setattr(
        frame_cache_module, "_installed_browser_candidates", lambda: [str(tmp_path / "missing.exe"), edge]
    )
    assert find_chromium_executable() == edge


def test_bad_configured_path_falls_back(tmp_path, monkeypatch, downloaded):
    monkeypatch.setattr(config, "chromium_exe_loc", str(tmp_path / "nope.exe"))
    edge = _exe(tmp_path, "msedge.exe")
    monkeypatch.setattr(frame_cache_module, "_installed_browser_candidates", lambda: [edge])
    assert find_chromium_executable() == edge


def test_nothing_found_lets_pyppeteer_download(monkeypatch, downloaded):
    monkeypatch.setattr(config, "chromium_exe_loc", None)
    monkeypatch.setattr(frame_cache_module, "_installed_browser_candidates", lambda: [])
    assert find_chromium_executable() is None

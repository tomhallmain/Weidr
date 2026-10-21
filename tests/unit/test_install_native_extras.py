"""Cairo and VLC are installed by scripts/install_native_extras.py."""

import importlib.util
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "install_native_extras.py")
_spec = importlib.util.spec_from_file_location("install_native_extras", _SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def test_check_reports_both_without_installing(monkeypatch, capsys):
    monkeypatch.setattr(script, "cairo_status", lambda: "ok")
    monkeypatch.setattr(script, "find_vlc_library", lambda: r"C:\VLC\libvlc.dll")
    monkeypatch.setattr(script, "run_command", lambda command: pytest.fail(str(command)))

    assert script.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "Cairo: available" in out
    assert "VLC: available" in out


def test_check_fails_when_vlc_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(script, "cairo_status", lambda: "ok")
    monkeypatch.setattr(script, "find_vlc_library", lambda: None)

    assert script.main(["--check"]) == 1
    assert "VLC: not found" in capsys.readouterr().out


def test_windows_cairo_install_uses_the_active_conda_prefix(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    prefix.mkdir()
    conda = tmp_path / "conda.exe"
    conda.write_bytes(b"")
    calls = []

    monkeypatch.setattr(script.sys, "platform", "win32")
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setattr(script, "find_conda", lambda: str(conda))
    monkeypatch.setattr(script, "cairo_status", lambda: "library")

    def run(command):
        calls.append(command)
        monkeypatch.setattr(script, "cairo_status", lambda: "ok")
        return 0

    assert script.ensure_cairo(run) == 0
    assert calls == [[str(conda), "install", "-y", "-p", str(prefix), "-c", "conda-forge", "cairo"]]


def test_cairo_install_needs_an_activated_conda_env(monkeypatch, capsys):
    monkeypatch.setattr(script.sys, "platform", "win32")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(script, "cairo_status", lambda: "library")
    monkeypatch.setattr(script, "find_conda", lambda: None)

    assert script.ensure_cairo(lambda command: pytest.fail(str(command))) == 1
    assert "Activate the conda environment" in capsys.readouterr().out


def test_windows_vlc_install_uses_winget(monkeypatch):
    calls = []
    monkeypatch.setattr(script.sys, "platform", "win32")
    monkeypatch.setattr(script.shutil, "which", lambda name: r"C:\winget.exe" if name == "winget" else None)
    found = {"path": None}
    monkeypatch.setattr(script, "find_vlc_library", lambda: found["path"])

    def run(command):
        calls.append(command)
        found["path"] = r"C:\Program Files\VideoLAN\VLC\libvlc.dll"
        return 0

    assert script.ensure_vlc(run) == 0
    assert calls[0][1:4] == ["install", "--id", "VideoLAN.VLC"]


def test_find_vlc_library_uses_an_install_directory(tmp_path, monkeypatch):
    folder = tmp_path / "VLC"
    folder.mkdir()
    library = folder / "libvlc.dll"
    library.write_bytes(b"vlc")
    monkeypatch.setattr(script.sys, "platform", "win32")
    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    monkeypatch.setattr(script, "_windows_vlc_install_dirs", lambda: [str(folder)])

    assert script.find_vlc_library() == str(library)


def test_linux_vlc_install_uses_apt(monkeypatch):
    calls = []
    monkeypatch.setattr(script.sys, "platform", "linux")
    monkeypatch.setattr(script, "find_vlc_library", lambda: None)
    monkeypatch.setattr(script.shutil, "which", lambda name: "/usr/bin/apt-get" if name == "apt-get" else "/usr/bin/sudo")
    monkeypatch.setattr(script.os, "geteuid", lambda: 1000, raising=False)

    def run(command):
        calls.append(command)
        monkeypatch.setattr(script, "find_vlc_library", lambda: "libvlc.so.5")
        return 0

    assert script.ensure_vlc(run) == 0
    assert calls == [["/usr/bin/sudo", "/usr/bin/apt-get", "install", "-y", "vlc"]]

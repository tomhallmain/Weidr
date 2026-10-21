"""Install the native libraries pip does not ship: Cairo (SVG) and VLC (playback).

Run this from the environment you use to start Weidr:

    python scripts/install_native_extras.py
    python scripts/install_native_extras.py --check

Windows installs Cairo with conda into the active environment, and VLC with
winget. The application loads the Windows ICU and the packaged gettext
library on startup, so the Cairo install does not need a manual DLL fix.
macOS uses Homebrew. Linux uses apt, dnf, or pacman.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Callable, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

Runner = Callable[[list[str]], int]

_CAIRO_MODULES = ("cairosvg", "cairocffi")


def find_conda() -> Optional[str]:
    """conda executable for the active environment, if this is a conda env."""
    configured = os.environ.get("CONDA_EXE")
    if configured and os.path.isfile(configured):
        return configured
    found = shutil.which("conda")
    if found:
        return found
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    parent = os.path.dirname(prefix)
    base = os.path.dirname(parent) if os.path.basename(parent).lower() == "envs" else prefix
    candidate = os.path.join(base, "Scripts", "conda.exe")
    if os.path.isfile(candidate):
        return candidate
    return None


def cairo_status() -> str:
    """``ok``, ``package`` (cairosvg missing), or ``library`` (Cairo DLL missing)."""
    from utils.windows_dll_compat import prefer_packaged_libintl

    prefer_packaged_libintl()
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        return "package"
    except OSError:
        return "library"
    return "ok"


def _drop_cairo_modules() -> None:
    for name in list(sys.modules):
        if name in _CAIRO_MODULES or name.startswith(_CAIRO_MODULES[0] + ".") or name.startswith(_CAIRO_MODULES[1] + "."):
            del sys.modules[name]


def _windows_vlc_install_dirs() -> list[str]:
    folders: list[str] = []
    try:
        import winreg
    except ImportError:
        winreg = None  # type: ignore[assignment]
    if winreg is not None:
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, r"Software\VideoLAN\VLC") as key:
                    install_dir, _value_type = winreg.QueryValueEx(key, "InstallDir")
            except OSError:
                continue
            if isinstance(install_dir, str) and install_dir:
                folders.append(install_dir)
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if root:
            folders.append(os.path.join(root, "VideoLAN", "VLC"))
    return folders


def find_vlc_library() -> Optional[str]:
    """Path or soname of libvlc, or None when VLC is not installed."""
    if sys.platform == "win32":
        on_path = shutil.which("libvlc.dll")
        if on_path:
            return on_path
        for folder in _windows_vlc_install_dirs():
            candidate = os.path.join(folder, "libvlc.dll")
            if os.path.isfile(candidate):
                return candidate
        return None
    if sys.platform == "darwin":
        candidate = "/Applications/VLC.app/Contents/MacOS/lib/libvlc.dylib"
        return candidate if os.path.isfile(candidate) else None
    import ctypes.util

    found = ctypes.util.find_library("vlc")
    if found:
        return found
    for candidate in (
        "/usr/lib/libvlc.so.5",
        "/usr/lib/x86_64-linux-gnu/libvlc.so.5",
        "/usr/lib64/libvlc.so.5",
        "/usr/local/lib/libvlc.so.5",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _privileged(command: list[str]) -> list[str]:
    if sys.platform == "win32":
        return command
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None and geteuid() == 0:
        return command
    sudo = shutil.which("sudo")
    if sudo:
        return [sudo, *command]
    return command


def _linux_package_install(run: Runner, packages: dict[str, list[str]]) -> Optional[int]:
    for manager, names in packages.items():
        executable = shutil.which(manager)
        if not executable:
            continue
        if manager == "pacman":
            command = [executable, "-S", "--noconfirm", *names]
        else:
            command = [executable, "install", "-y", *names]
        return run(_privileged(command))
    return None


def _install_cairo_library(run: Runner) -> int:
    if sys.platform == "win32":
        prefix = os.environ.get("CONDA_PREFIX")
        conda = find_conda()
        if not prefix or not conda:
            print(
                "Cairo was not found. Activate the conda environment you use for Weidr "
                "and run this script again."
            )
            return 1
        print(f"Installing Cairo into {prefix}")
        code = run([conda, "install", "-y", "-p", prefix, "-c", "conda-forge", "cairo"])
    elif sys.platform == "darwin":
        brew = shutil.which("brew")
        if not brew:
            print("Cairo was not found. Install Homebrew, then run: brew install cairo")
            return 1
        code = run([brew, "install", "cairo"])
    else:
        code = _linux_package_install(
            run,
            {"apt-get": ["libcairo2"], "dnf": ["cairo"], "pacman": ["cairo"]},
        )
        if code is None:
            print("Cairo was not found. Install your distribution's cairo package and run this script again.")
            return 1
    if code != 0:
        return code
    _drop_cairo_modules()
    if cairo_status() != "ok":
        print("Cairo was installed, but it still did not load. Restart the terminal and run this script again.")
        return 1
    print("Cairo is available.")
    return 0


def ensure_cairo(run: Runner) -> int:
    status = cairo_status()
    if status == "ok":
        print("Cairo is already available.")
        return 0
    if status == "package":
        print("Installing cairosvg.")
        code = run([sys.executable, "-m", "pip", "install", "cairosvg"])
        if code != 0:
            return code
        _drop_cairo_modules()
        if cairo_status() == "ok":
            print("Cairo is available.")
            return 0
    return _install_cairo_library(run)


def ensure_vlc(run: Runner) -> int:
    found = find_vlc_library()
    if found:
        print(f"VLC is already available ({found}).")
        return 0
    code: Optional[int]
    if sys.platform == "win32":
        winget = shutil.which("winget")
        if not winget:
            print(
                "VLC was not found, and winget is not available. "
                "Install it from https://www.videolan.org and restart."
            )
            return 1
        print("Installing VLC. Windows may ask for administrator approval.")
        code = run([
            winget, "install", "--id", "VideoLAN.VLC", "-e",
            "--accept-package-agreements", "--accept-source-agreements",
            "--disable-interactivity",
        ])
    elif sys.platform == "darwin":
        brew = shutil.which("brew")
        if not brew:
            print("VLC was not found. Install Homebrew, then run: brew install --cask vlc")
            return 1
        code = run([brew, "install", "--cask", "vlc"])
    else:
        code = _linux_package_install(run, {"apt-get": ["vlc"], "dnf": ["vlc"], "pacman": ["vlc"]})
        if code is None:
            print("VLC was not found. Install your distribution's vlc package and restart.")
            return 1
    found = find_vlc_library()
    if found:
        print(f"VLC is available ({found}).")
        return 0
    if code != 0:
        print("VLC install did not succeed. Install it from https://www.videolan.org and restart.")
        return code if code is not None else 1
    print("The installer finished, but libvlc was not found. Restart the terminal and run this script again.")
    return 1


def run_command(command: list[str]) -> int:
    print("+", " ".join(command))
    completed = subprocess.run(command)
    return completed.returncode


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Install Cairo and VLC for Weidr.")
    parser.add_argument("--check", action="store_true", help="Report what is missing and install nothing.")
    args = parser.parse_args(argv)
    if args.check:
        cairo_ok = cairo_status() == "ok"
        vlc = find_vlc_library()
        print("Cairo: available" if cairo_ok else "Cairo: not found")
        print(f"VLC: available ({vlc})" if vlc else "VLC: not found")
        return 0 if cairo_ok and vlc else 1
    cairo_code = ensure_cairo(run_command)
    vlc_code = ensure_vlc(run_command)
    if cairo_code == 0 and vlc_code == 0:
        print("Restart Weidr.")
        return 0
    return cairo_code or vlc_code or 1


if __name__ == "__main__":
    sys.exit(main())

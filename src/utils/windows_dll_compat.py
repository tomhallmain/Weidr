"""Keep conda-forge Cairo from shadowing the DLLs pip PySide6 and fontconfig need.

``scripts/install_native_extras.py`` installs Cairo. These loads still run at
startup: a one-time install cannot stop conda's ``Library\\bin`` from being
searched ahead of the Windows ICU.

conda install -c conda-forge cairo puts two conflicting copies on the DLL
search path that conda Python always includes (``Library\\bin``):

* ICU 78's ``icuuc.dll``. Qt 6 imports ``icuuc.dll`` and expects the Windows
  ICU. The conda copy is found first and is missing a procedure Qt needs.
* defaults ``libglib``'s ``intl-8.dll``, which can replace conda-forge
  libintl's copy of the same filename. fontconfig imports ``libintl_*`` from
  that name; the glib copy only exports ``g_libintl_*``.
"""

import ctypes
import json
import os
import sys
from typing import Optional


def prefer_system_icu() -> None:
    """Load the Windows ICU before the first PySide6 import.

    A full path load registers ``icuuc.dll`` in this process, so Qt binds to
    it instead of a newer conda ICU later on the search path. Safe when the
    conda copy is absent: Qt would have loaded this file anyway.
    """
    if sys.platform != "win32":
        return
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    icu_path = os.path.join(system_root, "System32", "icuuc.dll")
    if not os.path.isfile(icu_path):
        return
    try:
        ctypes.WinDLL(icu_path)
    except OSError:
        return


def prefer_packaged_libintl() -> None:
    """Load conda-forge libintl before Cairo when ``Library\\bin`` has the glib shim.

    Preloading the packaged DLL registers the ``intl-8.dll`` filename, so
    fontconfig binds to the copy that exports ``libintl_*``. The glib shim
    then cannot load in this process. Weidr does not use GLib. Skipped when
    the installed file is already at least as large as the packaged one.

    A frozen executable's ``sys.prefix`` is its unpack directory. When it is
    started from an activated conda environment, Cairo is still found on
    ``PATH`` in that environment, so the gettext DLL is taken from
    ``CONDA_PREFIX`` instead.
    """
    if sys.platform != "win32":
        return
    if _preload_libintl_from(sys.prefix) or _prefix_has_intl(sys.prefix):
        return
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    if os.path.normcase(os.path.abspath(conda_prefix)) == os.path.normcase(os.path.abspath(sys.prefix)):
        return
    _preload_libintl_from(conda_prefix)


def _prefix_has_intl(prefix: str) -> bool:
    return os.path.isfile(os.path.join(prefix, "Library", "bin", "intl-8.dll"))


def _preload_libintl_from(prefix: str) -> bool:
    installed = os.path.join(prefix, "Library", "bin", "intl-8.dll")
    packaged = _packaged_libintl_dll(prefix)
    if not packaged or not os.path.isfile(installed):
        return False
    try:
        if os.path.getsize(packaged) <= os.path.getsize(installed):
            return False
        ctypes.WinDLL(packaged)
    except OSError:
        return False
    return True


def _packaged_libintl_dll(prefix: str) -> Optional[str]:
    """``intl-8.dll`` from the extracted conda libintl package, if the cache remains."""
    meta_dir = os.path.join(prefix, "conda-meta")
    if not os.path.isdir(meta_dir):
        return None
    best: Optional[str] = None
    best_size = -1
    try:
        names = os.listdir(meta_dir)
    except OSError:
        return None
    for name in names:
        if not (name.startswith("libintl-") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(meta_dir, name), encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        extracted = data.get("extracted_package_dir")
        if not isinstance(extracted, str):
            continue
        candidate = os.path.join(extracted, "Library", "bin", "intl-8.dll")
        try:
            size = os.path.getsize(candidate)
        except OSError:
            continue
        if size > best_size:
            best = candidate
            best_size = size
    return best

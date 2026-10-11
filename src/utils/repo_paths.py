"""Where Weidr's top-level files live, resolved once here instead of
independently at each call site that needs them.

Several modules need top-level siblings that are never imported as packages
-- configs/, locale/, assets/ -- and a hop count from a file's own __file__ is
only correct for the depth that file happens to live at. Routing every lookup
through this module means a move only needs this file's hop count corrected.

A Nuitka build splits those files in two:

- resource_root(): files shipped with the app (locale/, assets/,
  configs/config_example.json, configs/suggested_classifier_models.json).
- user_root(): files the app writes (configs/config.json, app_info_cache.*,
  the classifier prediction cache, the extracted example pipelines). In a
  build these must not live beside the shipped files, which a new build
  replaces and which may not be writable, so they go to app_data_dir().

From a source checkout both are the repo root. Logs go to app_data_dir() in
both.
"""

import os
import sys

# Nuitka defines __compiled__ in the globals of every module it compiles.
_COMPILED = "__compiled__" in globals()

APP_DIR_NAME = "Weidr"
APP_DATA_DIR_ENV = "WEIDR_APP_DATA_DIR"


def is_compiled() -> bool:
    """True when running as a Nuitka build, not from source."""
    return _COMPILED


def repo_root() -> str:
    """The source checkout's root: this file is src/utils/repo_paths.py."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resource_root() -> str:
    """Directory holding the shipped data files.

    A build has no src/ level: utils/ sits directly in the build folder, and
    build_exe.py places the data files there too.
    """
    if _COMPILED:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return repo_root()


def app_data_dir() -> str:
    """Weidr's per-user data directory; not created here.

    %APPDATA%\\Weidr (the roaming profile) on Windows, ~/.local/share/Weidr
    elsewhere, macOS included. WEIDR_APP_DATA_DIR overrides it, for tests and
    the build's smoke test. Distinct from Utils.user_data_dir(), the base of
    the encryptor's key store, which other apps share.
    """
    override = os.environ.get(APP_DATA_DIR_ENV)
    if override:
        return override
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, APP_DIR_NAME)


def logs_dir() -> str:
    """Directory for log files, created if missing."""
    path = os.path.join(app_data_dir(), "logs")
    os.makedirs(path, exist_ok=True)
    return path


def user_root() -> str:
    """Directory for files the app writes; not created here.

    WEIDR_CONFIGS_DIR and WEIDR_CACHE_DIR still take precedence at their own
    call sites.
    """
    if _COMPILED:
        return app_data_dir()
    return repo_root()

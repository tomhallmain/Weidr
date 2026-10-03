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
  replaces and which may not be writable.

From a source checkout both are the repo root.
"""

import os

# Nuitka defines __compiled__ in the globals of every module it compiles.
_COMPILED = "__compiled__" in globals()

APP_DIR_NAME = "Weidr"


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


def user_root() -> str:
    """Directory for files the app writes; not created here.

    WEIDR_CONFIGS_DIR and WEIDR_CACHE_DIR still take precedence at their own
    call sites.
    """
    if _COMPILED:
        from utils.utils import Utils
        return os.path.join(Utils.user_data_dir(), APP_DIR_NAME)
    return repo_root()

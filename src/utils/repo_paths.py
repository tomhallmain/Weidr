"""Where Weidr's top-level files live, resolved once here instead of
independently at each call site that needs them.

Several modules need top-level siblings that are never imported as packages
-- configs/, locale/, assets/ -- and a hop count from a file's own __file__ is
only correct for the depth that file happens to live at. Routing every lookup
through this module means a move only needs this file's hop count corrected.

Those files come in two kinds:

- resource_root(): files shipped with the app (locale/, assets/,
  configs/config_example.json, configs/suggested_classifier_models.json):
  the repo root in a checkout, the build folder in a Nuitka build.
- user_root(): files the app writes (configs/config.json, app_info_cache.*,
  the classifier prediction cache, the extracted example pipelines), and
  logs: app_data_dir() in both, so a checkout and a build share them. A
  build's folder is replaced by each new build and may not be writable.

The first user_root() call in a process moves user files left at their old
locations into app_data_dir(), replacing older files there (_move_path()),
and not while an override variable is set: from the repo (a checkout only; a
build cannot see it, so running from the checkout once is what migrates
them) and from ~/.weidr (checkout and build).
"""

import glob
import os
import shutil
import sys
import threading

# Nuitka defines __compiled__ in the globals of every module it compiles.
_COMPILED = "__compiled__" in globals()

APP_DIR_NAME = "Weidr"
APP_DATA_DIR_ENV = "WEIDR_APP_DATA_DIR"
# Any of these points the app at a scratch or test directory.
_OVERRIDE_ENVS = (APP_DATA_DIR_ENV, "WEIDR_CONFIGS_DIR", "WEIDR_CACHE_DIR")

# configs/ entries that are shipped, not written; every other entry in the
# repo's configs/ is user data.
_SHIPPED_CONFIGS = ("config_example.json", "suggested_classifier_models.json")
# User files a checkout wrote at the repo root, kept at the same relative
# path under app_data_dir(). Temporary files of an interrupted save are left.
_LEGACY_USER_FILE_PATTERNS = (
    "app_info_cache.enc",
    "app_info_cache.enc.bak*",
    "app_info_cache.json",
    "app_info_cache.no_oqs.json",
    "classifier_prediction_cache.enc",
    # config.file_paths_json_path's default, which resolved against the
    # working directory: the repo when started from there.
    "file_paths.json",
    os.path.join("assets", "pipelines"),
)
# Utils.get_no_directory_compare_cache_dir()'s folder under the cache root.
NO_DIRECTORY_COMPARE_CACHE_NAME = "compare_cache_no_dir"
# Where it was before, under the home directory; the folder is removed once empty.
_LEGACY_HOME_DIR = ".weidr"
_LEGACY_HOME_ENTRIES = (NO_DIRECTORY_COMPARE_CACHE_NAME,)

_migration_done = False
_migration_lock = threading.Lock()


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

    WEIDR_APP_DATA_DIR overrides default_app_data_dir(), for tests and the
    build's smoke test. Distinct from Utils.user_data_dir(), the base of the
    encryptor's key store, which other apps share.
    """
    return os.environ.get(APP_DATA_DIR_ENV) or default_app_data_dir()


def default_app_data_dir() -> str:
    """%APPDATA%\\Weidr (the roaming profile) on Windows, ~/.local/share/Weidr
    elsewhere, macOS included. build_exe.py unpacks builds under it."""
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
    root = app_data_dir()
    _migrate_legacy_user_files(root)
    return root


def _logger():
    # Imported here: logging_setup imports this module.
    from utils.logging_setup import get_logger
    return get_logger("repo_paths")


def _migrate_legacy_user_files(target: str) -> None:
    """Once per process: move user files from their old locations into *target*."""
    global _migration_done
    with _migration_lock:
        if _migration_done:
            return
        _migration_done = True
        if any(os.environ.get(name) for name in _OVERRIDE_ENVS):
            return
        if not _COMPILED:
            try:
                _move_legacy_user_files(repo_root(), target)
            except Exception as e:
                _logger().error(f"Moving user files from {repo_root()} to {target} failed: {e}")
        home = os.path.expanduser("~")
        try:
            _move_legacy_home_files(home, target)
        except Exception as e:
            _logger().error(f"Moving user files from {os.path.join(home, _LEGACY_HOME_DIR)} "
                            f"to {target} failed: {e}")


def _move_legacy_user_files(source_root: str, target_root: str) -> None:
    configs = os.path.join(source_root, "configs")
    relative_paths = []
    if os.path.isdir(configs):
        relative_paths += [os.path.join("configs", name) for name in sorted(os.listdir(configs))
                           if name not in _SHIPPED_CONFIGS]
    for pattern in _LEGACY_USER_FILE_PATTERNS:
        relative_paths += sorted(os.path.relpath(path, source_root)
                                 for path in glob.glob(os.path.join(glob.escape(source_root), pattern)))
    _move_entries(source_root, target_root, relative_paths)


def _move_legacy_home_files(home: str, target_root: str) -> None:
    legacy = os.path.join(home, _LEGACY_HOME_DIR)
    if not os.path.isdir(legacy):
        return
    _move_entries(legacy, target_root,
                  [name for name in _LEGACY_HOME_ENTRIES if os.path.lexists(os.path.join(legacy, name))])
    try:
        os.rmdir(legacy)
        _logger().info(f"Removed the empty folder {legacy}")
    except OSError:
        pass  # Not empty: something else is still there.


def _move_entries(source_root: str, target_root: str, relative_paths: list) -> None:
    """Move each of *relative_paths* from *source_root* to the same path under *target_root*."""
    for relative_path in relative_paths:
        source = os.path.join(source_root, relative_path)
        destination = os.path.join(target_root, relative_path)
        try:
            _move_path(source, destination)
        except OSError as e:
            _logger().error(f"Could not move {source} to {destination}: {e}")


def _move_path(source: str, destination: str) -> None:
    """Move *source* to *destination*. An existing destination file is
    replaced unless *source* was modified more recently; then both stay and
    the log says so. Folders are merged file by file under the same rule,
    and the source folder is removed once empty."""
    if not os.path.lexists(destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
        _logger().info(f"Moved {source} to {destination}")
        return
    if os.path.isdir(source) and os.path.isdir(destination):
        for name in sorted(os.listdir(source)):
            try:
                _move_path(os.path.join(source, name), os.path.join(destination, name))
            except OSError as e:
                _logger().error(f"Could not move {os.path.join(source, name)}: {e}")
        try:
            os.rmdir(source)
        except OSError:
            pass  # Not empty: an entry was kept.
        return
    if os.path.isdir(source) or os.path.isdir(destination):
        _logger().warning(f"Not moving {source}: {destination} exists and only one of them is a folder")
        return
    if os.path.getmtime(source) > os.path.getmtime(destination):
        _logger().warning(f"Not moving {source}: it is newer than {destination}. "
                          "Remove one of them to choose which one Weidr uses.")
        return
    try:
        os.replace(source, destination)
    except OSError:
        # os.replace cannot cross drives or filesystems.
        shutil.copy2(source, destination)
        os.remove(source)
    _logger().info(f"Moved {source} to {destination}, replacing the file there")

"""
Root conftest for the Weidr test suite.

IMPORTANT: The env vars below must be set at module load time — before any app
module is imported — because both `app_info_cache` and `config` are module-level
singletons instantiated on first import. Any nested conftest.py files must
mirror this same module-level assignment for the same reason.
"""

import atexit
import os
import shutil
import sys
import tempfile

# ---------------------------------------------------------------------------
# Manual / prospective scripts that live in tests/ but are NOT pytest suites.
# Exclude them from collection so `pytest tests/` never accidentally runs them.
#
# test_gegl_operations.py    — requires GIMP 3 + a CLI image path argument
# test_gimp_gegl_direct.py   — requires GIMP 3; invokes it via raw subprocess
# test_compare_embedding_matrix.py — calls input() (blocks on stdin); reads from
#                              a hardcoded user path; imports tests.analysis
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
collect_ignore = [
    os.path.join(_here, "test_gegl_operations.py"),
    os.path.join(_here, "test_gimp_gegl_direct.py"),
    os.path.join(_here, "test_compare_embedding_matrix.py"),
]

# Ensure the project root is on sys.path so that app packages (ui/, utils/,
# etc.) are importable regardless of which directory pytest is invoked from.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Bootstrap a safe temporary location so that the singletons created during
# initial import never touch the real cache or config files.
_bootstrap_tmp = tempfile.mkdtemp(prefix="weidr_tests_")
os.environ.setdefault("WEIDR_CACHE_DIR", os.path.join(_bootstrap_tmp, "cache"))
os.environ.setdefault("WEIDR_CONFIGS_DIR", os.path.join(_bootstrap_tmp, "configs"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.makedirs(os.environ["WEIDR_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["WEIDR_CONFIGS_DIR"], exist_ok=True)
_src_example = os.path.join(os.path.dirname(__file__), "..", "configs", "config_example.json")
shutil.copy(_src_example, os.path.join(os.environ["WEIDR_CONFIGS_DIR"], "config.json"))
atexit.register(shutil.rmtree, _bootstrap_tmp, True)

import pytest


@pytest.fixture(autouse=True)
def isolated_singletons(tmp_path, monkeypatch):
    """Re-initialise the app_info_cache and config singletons for each test,
    pointing at a fresh per-test temp directory. No production files are touched."""
    cache_dir = tmp_path / "cache"
    configs_dir = tmp_path / "configs"
    cache_dir.mkdir()
    configs_dir.mkdir()
    shutil.copy(_src_example, configs_dir / "config.json")

    monkeypatch.setenv("WEIDR_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("WEIDR_CONFIGS_DIR", str(configs_dir))

    import utils.app_info_cache as aic
    import utils.config as cfg

    new_cache = aic.AppInfoCache()
    monkeypatch.setattr(aic, "app_info_cache", new_cache)

    # cache_controller.py holds a module-level `from utils.app_info_cache import
    # app_info_cache` that bypasses the monkeypatch above.  Patch it directly so
    # AppWindow.__init__ reads from the fresh per-test cache, not a stale one.
    try:
        import ui.app_window.cache_controller as _cc
        monkeypatch.setattr(_cc, "app_info_cache", new_cache)
    except Exception:
        pass

    try:
        import files.file_action as _fa
        monkeypatch.setattr(_fa, "app_info_cache", new_cache)
    except Exception:
        pass

    try:
        import compare.classifier_pipeline as _cp
        monkeypatch.setattr(_cp, "app_info_cache", new_cache)
    except Exception:
        pass

    try:
        import ui.compare.seek_to_trigger_tab_qt as _stt
        monkeypatch.setattr(_stt, "app_info_cache", new_cache)
    except Exception:
        pass

    try:
        import ui.compare.classifier_management_window_qt as _cmw
        monkeypatch.setattr(_cmw, "app_info_cache", new_cache)
    except Exception:
        pass

    # Silence startup log spam; patch before instantiation so __init__ skips the print.
    monkeypatch.setattr(cfg.Config, "print_config_settings", lambda self: None)
    monkeypatch.setattr(cfg, "config", cfg.Config())


@pytest.fixture(autouse=True)
def reset_app_globals():
    """Reset class-level mutable state on the shared singletons that are not
    covered by isolated_singletons (which only handles config + app_info_cache).

    Runs before each test so any state leaked by a previous test does not
    pollute the next one.  Teardown after yield is a courtesy reset so that a
    failing test leaves the process in a clean state for potential post-run
    inspection fixtures.
    """
    def _reset():
        # MarkedFiles
        try:
            from files.marked_files import MarkedFiles
            MarkedFiles.file_marks = []
            MarkedFiles.is_performing_action = False
            MarkedFiles.delete_lock = False
        except Exception:
            pass

        # FileAction
        try:
            from files.file_action import FileAction
            FileAction.action_history = []
            FileAction.permanent_action = None
            FileAction.hotkey_actions = {}
        except Exception:
            pass

        # ClassifierActionsManager
        try:
            from compare.classifier_actions_manager import ClassifierActionsManager
            ClassifierActionsManager.prevalidated_cache.clear()
            ClassifierActionsManager.prevalidations = []
            ClassifierActionsManager._prevalidations_initialized = False
        except Exception:
            pass

        # ClassifierPipelines
        try:
            from compare.classifier_pipeline import ClassifierPipelines
            ClassifierPipelines.pipelines = []
            ClassifierPipelines._prevalidation_pipelines = []
            ClassifierPipelines._action_pipelines = []
        except Exception:
            pass

        # ClassifierPipelinesTab — class-level editor window reference
        try:
            from ui.compare.classifier_pipelines_tab_qt import ClassifierPipelinesTab
            ClassifierPipelinesTab._editor_window = None
        except Exception:
            pass

        # ClassifierManagementWindow — singleton dialog reference
        try:
            from ui.compare.classifier_management_window_qt import ClassifierManagementWindow
            ClassifierManagementWindow._instance = None
        except Exception:
            pass

        # Lookahead — shared list of lookahead definitions
        try:
            from compare.lookahead import Lookahead
            Lookahead.lookaheads = []
        except Exception:
            pass

        # DirectoryProfile — shared list of directory profiles
        try:
            from files.directory_profile import DirectoryProfile
            DirectoryProfile.directory_profiles = []
        except Exception:
            pass

        # SeekToTriggerTab — class-level action cache and cycling state
        try:
            from ui.compare.seek_to_trigger_tab_qt import SeekToTriggerTab
            SeekToTriggerTab._last_action = None
            SeekToTriggerTab._last_trigger_slot.clear()
            SeekToTriggerTab._headless_worker = None
        except Exception:
            pass

        # FrameCache — clear in-memory dicts only; leave temp dir intact
        try:
            from image.frame_cache import FrameCache
            FrameCache.clear()
        except Exception:
            pass

        # WindowManager — UI tests should unregister on close; reset if any leaked
        try:
            from ui.app_window.window_manager import WindowManager
            WindowManager._windows.clear()
            WindowManager._primary = None
            WindowManager._secondary_toplevels.clear()
            WindowManager._cycle_index = 0
        except Exception:
            pass

        # FileBrowser — confirmed-directories list persists across tests otherwise
        try:
            from files.file_browser import FileBrowser
            FileBrowser.have_confirmed_directories.clear()
        except Exception:
            pass

        # Downstream related-image cache keyed by path; stale entries from one
        # test would silently skip the refresh in the next
        try:
            import files.related_image as _ri
            _ri._downstream_cache.clear()
            _ri._downstream_index = 0
            _ri._downstream_browser = None
        except Exception:
            pass

    _reset()
    yield
    _reset()

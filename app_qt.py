"""
Weidr - Media Handler -- PySide6 entry point.

Creates the QApplication, handles startup authentication, signal handlers,
single-instance locking, and launches the main AppWindow.
"""

import os
import signal
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from ui.app_style import AppStyle
from utils.config import config
from utils.logging_setup import get_logger
from utils.repo_paths import resource_root
from utils.translations import I18N, _
from utils.utils import Utils
logger = get_logger("app_qt")


def _start_mcp_server():
    """Start the MCP server on a daemon thread, if configured.

    Resolves the active window fresh on every tool/resource call rather than
    capturing one here -- see extensions/mcp_server.py's module docstring on
    why a captured reference would go stale as windows open, switch focus,
    and close.
    """
    import threading

    from extensions.mcp_server import MCPServerExtension
    from ui.app_window.mcp_session_qt import QtWindowMCPSession
    from ui.app_window.window_manager import WindowManager

    def _resolve_session():
        window = WindowManager.get_active_window()
        return QtWindowMCPSession(window) if window is not None else None

    threading.Thread(
        target=MCPServerExtension(session_resolver=_resolve_session).start,
        daemon=True, name="mcp-server",
    ).start()


# Third-party modules imported only when a feature is first used; a build
# that left one out would fail then rather than at startup.
_SMOKE_TEST_LAZY_MODULES = (
    "av", "cairosvg", "insightface", "onnxruntime", "open_clip", "pyppeteer",
    "pypdfium2", "send2trash", "tensorflow", "tensorflow_hub", "tf_keras", "vlc",
)

_SMOKE_TEST_DATA_FILES = (
    ("assets", "icon.png"),
    ("assets", "example_pipelines.enc"),
    ("assets", "sounds"),
    ("configs", "config_example.json"),
    ("configs", "suggested_classifier_models.json"),
    ("locale", "en", "LC_MESSAGES", "base.mo"),
)


def _smoke_test_checks() -> list[str]:
    """Problems found by run_smoke_test(); empty when there are none."""
    import importlib.util

    import keyring
    from keyring.backends import fail as keyring_fail

    from utils import pillow_plugins

    problems = []
    I18N.install_locale(config.locale, verbose=False)

    pillow_plugins.ensure_pillow_plugins_registered()
    problems += [f"Pillow plugin not importable: {name}"
                 for name in ("pillow_avif", "pillow_heif", "pillow_jxl")
                 if name not in sys.modules]

    problems += [f"Data file missing: {os.path.join(*parts)}"
                 for parts in _SMOKE_TEST_DATA_FILES
                 if not os.path.exists(os.path.join(resource_root(), *parts))]

    problems += [f"Module missing: {name}" for name in _SMOKE_TEST_LAZY_MODULES
                 if importlib.util.find_spec(name) is None]

    if isinstance(keyring.get_keyring(), keyring_fail.Keyring):
        problems.append("No keyring backend available")

    qt_app = QApplication(sys.argv[:1])
    if QIcon(os.path.join(resource_root(), "assets", "icon.png")).isNull():
        problems.append("Qt cannot load assets/icon.png (image format plugins missing?)")

    # Imports every compare engine and the main window's import graph; no
    # model is loaded and no window is created.
    from compare import model
    import compare.compare_manager  # noqa: F401
    import ui.app_window.app_window  # noqa: F401
    if not model.eva_clip_loaded:
        problems.append("open_clip failed to import")
    if not model.insightface_loaded:
        problems.append("insightface failed to import")

    qt_app.quit()
    return problems


def run_smoke_test() -> int:
    """Check that a build starts: config, translations, Pillow plugins, data
    files, compare modules and a QApplication on the offscreen platform.
    Takes no single-instance lock, asks for no password, opens no window and
    loads no model. Returns the process exit code.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        problems = _smoke_test_checks()
    except BaseException:
        logger.exception("Smoke test failed")
        traceback.print_exc()
        return 1
    for problem in problems:
        logger.error("Smoke test: %s", problem)
        print(f"Smoke test: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("Smoke test passed")
    return 0


def main():
    # Single instance check -- prevent multiple instances from running
    lock_file, cleanup_lock = Utils.check_single_instance("Weidr")

    I18N.install_locale(config.locale, verbose=config.print_settings)

    # Create QApplication (must exist before any widgets)
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Weidr")
    qt_app.setStyleSheet(AppStyle.get_stylesheet())

    # Application icon
    icon_path = os.path.join(resource_root(), "assets", "icon.png")
    if os.path.isfile(icon_path):
        qt_app.setWindowIcon(QIcon(icon_path))

    # ------------------------------------------------------------------
    # Graceful shutdown handler
    # ------------------------------------------------------------------
    app_window = None  # will be set after startup auth succeeds

    def graceful_shutdown(signum, frame):
        logger.info("Caught signal, shutting down gracefully...")
        if app_window is not None:
            app_window.on_closing()
        cleanup_lock()
        os._exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # ------------------------------------------------------------------
    # Startup authentication callback
    # ------------------------------------------------------------------
    def startup_callback(result: bool) -> None:
        nonlocal app_window

        if not result:
            logger.info("User cancelled password dialog, exiting application")
            cleanup_lock()
            sys.exit(0)

        # Password verified or not required -- create the main window
        from ui.app_window.app_window import AppWindow
        from ui.files.type_configuration_window_qt import TypeConfigurationWindow

        try:
            # Keep behavior in sync with Tk startup: apply persisted type toggles
            # before FileBrowser and windows are initialized.
            TypeConfigurationWindow.load_pending_changes()
            TypeConfigurationWindow.apply_changes()
            app_window = AppWindow()
            app_window.show()

            # Bring window to front and give it focus
            app_window.raise_()
            app_window.activateWindow()

            _start_mcp_server()
        except Exception as e:
            logger.critical(f"Failed to create main window: {e}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                None, "Startup Error",
                f"Failed to create main window:\n\n{e}"
            )
            cleanup_lock()
            os._exit(1)

    # ------------------------------------------------------------------
    # Check if startup password is required
    # ------------------------------------------------------------------
    from ui.auth.app_startup_auth_qt import check_startup_password_required
    check_startup_password_required(callback=startup_callback)

    # ------------------------------------------------------------------
    # Run the event loop
    # ------------------------------------------------------------------
    try:
        exit_code = qt_app.exec()
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        cleanup_lock()

    sys.exit(exit_code)


if __name__ == "__main__":
    if "--smoke-test" in sys.argv[1:]:
        sys.exit(run_smoke_test())
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()

"""
Weidr - Media Handler -- PySide6 entry point.

Creates the QApplication, handles startup authentication, signal handlers,
single-instance locking, and launches the main AppWindow.
"""

import logging
import os
import signal
import sys
import threading
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from ui.app_style import AppStyle
from utils.config import config
from utils.logging_setup import get_logger
from utils.repo_paths import is_compiled, resource_root
from utils.translations import I18N, _
from utils.unpack_cleanup import remove_stale_unpack_dirs
from utils.utils import Utils
from utils.version import APP_VERSION, build_info, describe as describe_version, source_drift
logger = get_logger("app_qt")


_QT_MSG_LOG_LEVELS = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def _log_qt_message(msg_type, context, message):
    logger.log(_QT_MSG_LOG_LEVELS.get(msg_type, logging.WARNING), "Qt: %s", message)


def _route_qt_messages_to_logger():
    """Send Qt's own warnings and errors through the app logger.

    Call before creating the QApplication. Qt's default handler writes byte
    strings to the C-level stderr. A Nuitka build with
    --windows-console-mode=attach puts that stream in wide-character
    (_O_U8TEXT) mode, and the C runtime aborts the process (0xc0000409 in
    ucrtbase.dll) on a byte write to it, so any Qt warning would end the app.
    """
    qInstallMessageHandler(_log_qt_message)


def _report_source_drift(app_window):
    """In a build, log whether the source checkout it was built from has
    changed since, and tell the user with a toast when it has."""
    try:
        result = source_drift()
    except Exception as e:
        logger.error(f"Could not compare this build with its source checkout: {e}")
        return
    if result:
        drifted, message = result
        if drifted:
            logger.warning(message)
            app_window.notification_ctrl.toast(
                _("This build no longer matches its source code. See the log for details."))
        else:
            logger.info(message)


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


def _smoke_test_checks(expect_oqs: bool = False) -> list[str]:
    """Problems found by run_smoke_test(); empty when there are none."""
    import importlib.util

    import keyring
    from keyring.backends import fail as keyring_fail

    from utils import encryptor, pillow_plugins
    from utils.crash_log import enable_fault_log

    problems = []
    logger.info("Smoke test: %s", describe_version())
    if is_compiled():
        info = build_info()
        if not info:
            problems.append("Build info missing")
        elif info.get("version") != APP_VERSION:
            problems.append(f"Build info names version {info.get('version')}, expected {APP_VERSION}")
    try:
        logger.info("Smoke test: fault log at %s", enable_fault_log())
    except Exception as e:
        problems.append(f"Fault log cannot be opened: {e}")

    if encryptor.KeyEncapsulation is None:
        if expect_oqs:
            problems.append("OQS not available, but the build was made with --with-oqs")
    else:
        logger.info("Smoke test: OQS available")

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

    backend = keyring.get_keyring()
    logger.info("Smoke test: keyring backend %s", type(backend).__name__)
    if isinstance(backend, keyring_fail.Keyring):
        problems.append("No keyring backend available")

    _route_qt_messages_to_logger()
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


def run_smoke_test(expect_oqs: bool = False) -> int:
    """Check that a build starts: build info, fault log, OQS (with
    *expect_oqs*), config, translations, Pillow plugins, data files, keyring,
    compare modules and a QApplication on the offscreen platform. Takes no
    single-instance lock, asks for no password, opens no window and loads no
    model. Returns the process exit code.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        problems = _smoke_test_checks(expect_oqs)
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

    logger.info(describe_version())
    try:
        from utils.crash_log import enable_fault_log
        enable_fault_log()
    except Exception as e:
        logger.error(f"Could not open the fault log: {e}")
    # Deleting an older build's unpack folder can take a while (several GB).
    threading.Thread(target=remove_stale_unpack_dirs, name="unpack-cleanup", daemon=True).start()

    I18N.install_locale(config.locale, verbose=config.print_settings)

    # Create QApplication (must exist before any widgets)
    _route_qt_messages_to_logger()
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
            # Runs git, which can take a moment.
            threading.Thread(target=_report_source_drift, args=(app_window,),
                             name="source-drift", daemon=True).start()
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
        sys.exit(run_smoke_test(expect_oqs="--expect-oqs" in sys.argv[1:]))
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()

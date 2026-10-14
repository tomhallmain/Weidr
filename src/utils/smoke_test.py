"""The checks behind ``--smoke-test``, which build_exe.py runs on each
executable it builds (app_qt.py's and app_headless.py's): build info, fault
log, OQS, translations, Pillow plugins, data files, lazily imported modules,
keyring and the compare engines. Each entry point adds its own checks.
"""

import importlib.util
import os
import sys
import traceback
from typing import Callable

from utils.logging_setup import get_logger

logger = get_logger("smoke_test")

# Third-party modules imported only when a feature is first used; a build
# that left one out would fail then rather than at startup. Both entry points
# reach these.
LAZY_MODULES = (
    "av", "cairosvg", "insightface", "mcp", "onnxruntime", "open_clip", "pyppeteer",
    "pypdfium2", "send2trash", "tensorflow", "tensorflow_hub", "tf_keras", "uvicorn",
)

DATA_FILES = (
    ("VERSION",),
    ("assets", "icon.png"),
    ("assets", "example_pipelines.enc"),
    ("assets", "sounds"),
    ("configs", "config_example.json"),
    ("configs", "suggested_classifier_models.json"),
    ("locale", "en", "LC_MESSAGES", "base.mo"),
)


def startup_checks(expect_oqs: bool, lazy_modules=LAZY_MODULES) -> list[str]:
    """Problems with what the app needs before it loads anything heavy;
    empty when there are none."""
    import keyring
    from keyring.backends import fail as keyring_fail

    from utils import encryptor, pillow_plugins
    from utils.config import config
    from utils.crash_log import enable_fault_log
    from utils.repo_paths import is_compiled, resource_root
    from utils.translations import I18N
    from utils.version import app_version, build_info, describe

    problems = []
    logger.info("Smoke test: %s", describe())
    if is_compiled():
        info = build_info()
        if not info:
            problems.append("Build info missing")
        elif info.get("version") != app_version():
            problems.append(f"Build info names version {info.get('version')}, expected {app_version()}")
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
                 for parts in DATA_FILES
                 if not os.path.exists(os.path.join(resource_root(), *parts))]

    problems += [f"Module missing: {name}" for name in lazy_modules
                 if importlib.util.find_spec(name) is None]

    backend = keyring.get_keyring()
    logger.info("Smoke test: keyring backend %s", type(backend).__name__)
    if isinstance(backend, keyring_fail.Keyring):
        problems.append("No keyring backend available")
    return problems


def compare_engine_checks() -> list[str]:
    """Import every compare engine (no model is loaded); problems found."""
    from compare import model
    import compare.compare_manager  # noqa: F401

    problems = []
    if not model.eva_clip_loaded:
        problems.append("open_clip failed to import")
    if not model.insightface_loaded:
        problems.append("insightface failed to import")
    return problems


def run(checks: Callable[[], list]) -> int:
    """Run *checks*, report what they found, and return the process exit code."""
    try:
        problems = checks()
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

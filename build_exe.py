"""
Build Weidr with Nuitka: ``python build_exe.py``.

Nuitka compiles the code to C, then to a native executable that embeds the
Python runtime. Needs a C compiler: Xcode Command Line Tools on macOS; on
Windows, Visual Studio Build Tools, or Nuitka downloads MinGW itself. Nuitka
does not cross-compile, so run this on each target OS.

Builds inside a dedicated ``.venv-build`` at the repo root, populated from
requirements-build.txt, so packages from the active conda env or venv aren't
bundled. The venv is based on the interpreter running this script, and the
build embeds that interpreter's runtime. pip installs torch from PyPI there,
which on Windows is the CPU build; for CUDA torch, install it into
.venv-build from the PyTorch index first, or use ``--current-env``, which
builds with the running interpreter instead (it must already have
requirements-build.txt installed).

The output is a folder, dist/Weidr/, holding the Weidr (Weidr.exe) executable
beside its libraries and data files; distribute the whole folder. It is
smoke-tested with ``Weidr --smoke-test``. Other args are forwarded to Nuitka,
e.g. ``--show-scons`` or ``--jobs=4``.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import venv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(REPO_ROOT, ".venv-build")
ENTRY = os.path.join(REPO_ROOT, "app_qt.py")
BUILD_DIR = os.path.join(REPO_ROOT, "build", "nuitka")
# Nuitka names the standalone folder after the entry script.
BUILD_DIST_DIR = os.path.join(BUILD_DIR, "app_qt.dist")
DIST_DIR = os.path.join(REPO_ROOT, "dist", "Weidr")
PACKAGE_CONFIG = os.path.join(REPO_ROOT, "nuitka-package.config.yml")
EXE_NAME = "Weidr.exe" if sys.platform == "win32" else "Weidr"
EXE_PATH = os.path.join(DIST_DIR, EXE_NAME)

# Shipped files, at the same relative paths utils.repo_paths.resource_root()
# resolves them under. assets/pipelines/ is left out: it is extracted from
# example_pipelines.enc into the user data directory.
DATA_DIRS = ("locale", os.path.join("assets", "sounds"))
DATA_FILES = (
    os.path.join("assets", "icon.png"),
    os.path.join("assets", "example_pipelines.enc"),
    os.path.join("configs", "config_example.json"),
    os.path.join("configs", "suggested_classifier_models.json"),
)


def _venv_python(venv_dir: str) -> str:
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _ensure_build_venv() -> str:
    python = _venv_python(VENV_DIR)
    if not os.path.exists(python):
        print(f"Creating build venv at {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    subprocess.run(
        [python, "-m", "pip", "install", "-q", "-r", os.path.join(REPO_ROOT, "requirements-build.txt")],
        check=True,
    )
    return python


def _nuitka_args(extra_args) -> list:
    args = [
        "--standalone",
        f"--output-dir={BUILD_DIR}",
        f"--output-filename={EXE_NAME}",
        # Lets Nuitka fetch build helpers (e.g. MinGW on Windows) without prompting.
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        # sound_player uses QSoundEffect.
        "--include-qt-plugins=multimedia",
        # utils/pillow_plugins.py imports these by name, which Nuitka can't follow.
        "--include-package=pillow_avif",
        "--include-package=pillow_heif",
        "--include-package=pillow_jxl",
        # clip reads its BPE vocab from beside its modules. (Nuitka ships
        # open_clip's model configs by itself.)
        "--include-package-data=clip",
        f"--user-package-configuration-file={PACKAGE_CONFIG}",
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=_pytest",
        "--nofollow-import-to=pytestqt",
    ]
    args += [f"--include-data-dir={d}={d}" for d in DATA_DIRS]
    args += [f"--include-data-files={f}={f}" for f in DATA_FILES]
    if sys.platform == "win32":
        args += [
            # No console window when started from Explorer; when started from
            # a terminal, output (e.g. the smoke test's) goes to that terminal.
            "--windows-console-mode=attach",
            f"--windows-icon-from-ico={os.path.join('assets', 'icon.ico')}",
        ]
    return [*args, *extra_args, ENTRY]


def _nuitka_env() -> dict:
    # app_qt.py puts src/ on sys.path at runtime; Nuitka resolves imports at
    # build time and needs it there already.
    env = dict(os.environ)
    paths = [os.path.join(REPO_ROOT, "src"), REPO_ROOT]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _copy_to_dist() -> None:
    if os.path.exists(DIST_DIR):
        shutil.rmtree(DIST_DIR)
    shutil.copytree(BUILD_DIST_DIR, DIST_DIR)


def _smoke_test() -> None:
    # Run from outside the repo so a passing test can't depend on the checkout,
    # with config and caches in a scratch directory so the user's are untouched.
    with tempfile.TemporaryDirectory(prefix="weidr-smoke-") as scratch:
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["WEIDR_CACHE_DIR"] = scratch
        env["WEIDR_CONFIGS_DIR"] = os.path.join(scratch, "configs")
        result = subprocess.run([EXE_PATH, "--smoke-test"], cwd=os.path.expanduser("~"), env=env)
    if result.returncode != 0:
        sys.exit(
            f"Smoke test failed: `{EXE_PATH} --smoke-test` exited {result.returncode}. "
            "Details are also in Weidr's log directory."
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--current-env" in args:
        args.remove("--current-env")
        python = sys.executable
    else:
        python = _ensure_build_venv()

    subprocess.run([python, "-m", "nuitka"] + _nuitka_args(args), cwd=REPO_ROOT, env=_nuitka_env(), check=True)
    _copy_to_dist()
    _smoke_test()
    print(f"Built {DIST_DIR}")

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

Builds a single-file executable in dist/ (``.exe`` on Windows), Weidr from
app_qt.py. ``--with-headless`` also builds WeidrHeadless from app_headless.py,
which serves a browsing/comparison session over MCP with no window and
contains no Qt; it is a second full Nuitka compile. At its first start each unpacks
its runtime to ``unpacked/<prefix>-<build id>`` in the Weidr data directory;
later starts of the same build reuse that folder, and each removes its own
older builds' folders (utils/unpack_cleanup.py). Each is smoke-tested with
``--smoke-test``. Other args are forwarded to Nuitka, e.g. ``--show-scons`` or
``--jobs=4``.

``--with-oqs`` adds quantum-safe (OQS) key encapsulation: it installs
liboqs-python, finds the liboqs shared library the build interpreter loads, and
bundles it. liboqs-python builds liboqs on first import if none is installed,
which needs git, CMake and a C compiler; or point OQS_INSTALL_PATH at an
existing install. Without the option the build has no OQS: it cannot read
data encrypted with OQS keys, and refuses password-protected actions when the
stored keys are OQS keys.

The executable's version information comes from the VERSION file at the repo
root, which is also bundled, with the build id (build time) in its description. Each
build writes ``build/nuitka/build_info.json`` (version, build id, whether it
has OQS, Python and key package versions, and this checkout's path, commit,
the commit's tree and a fingerprint of its uncommitted changes), bundled so the app logs it at startup and compares
itself with this checkout, and ``build/nuitka/build-requirements.lock.txt``,
the build venv's ``pip freeze``. ``--lock=<file>`` installs the build venv
from such a file instead of requirements-build.txt, to repeat a known good
build; packages already in the venv but absent from the file stay installed,
so delete .venv-build first for an exact copy.
"""
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import venv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(REPO_ROOT, ".venv-build")
BUILD_DIR = os.path.join(REPO_ROOT, "build", "nuitka")
DIST_DIR = os.path.join(REPO_ROOT, "dist")
PACKAGE_CONFIG = os.path.join(REPO_ROOT, "nuitka-package.config.yml")
# utils.repo_paths.default_app_data_dir() spelled for Nuitka, which has no variable
# for %APPDATA%. utils/unpack_cleanup.py expects the folder names TARGETS gives.
APP_DATA_SPEC = "{HOME}/AppData/Roaming/Weidr" if sys.platform == "win32" else "{HOME}/.local/share/Weidr"
VERSION_FILE = "VERSION"
VERSION_MODULE = os.path.join("src", "utils", "version.py")
BUILD_INFO_PATH = os.path.join(BUILD_DIR, "build_info.json")
LOCK_PATH = os.path.join(BUILD_DIR, "build-requirements.lock.txt")
# Versions the build output and build_info.json record.
RECORDED_DISTRIBUTIONS = (
    "Nuitka", "PySide6", "torch", "torchvision", "transformers", "tensorflow", "onnxruntime-gpu", "mcp",
)

# Shipped files, at the same relative paths utils.repo_paths.resource_root()
# resolves them under. assets/pipelines/ is left out: it is extracted from
# example_pipelines.enc into the user data directory.
DATA_DIRS = ("locale", os.path.join("assets", "sounds"))
DATA_FILES = (
    os.path.join("assets", "icon.png"),
    os.path.join("assets", "example_pipelines.enc"),
    os.path.join("configs", "config_example.json"),
    os.path.join("configs", "suggested_classifier_models.json"),
    VERSION_FILE,
)

# transformers reads the installed version of these packages when it finds
# them importable (_is_package_available(..., return_version=True)); without
# their metadata it imports the whole package to look for __version__, and
# Nuitka's own transformers configuration ends the process when that fails.
# Those of its probed packages that are in requirements-build.txt.
TRANSFORMERS_PROBED_DISTRIBUTIONS = ("huggingface_hub", "numpy", "torch", "torchvision")
# The transformers release the list above was taken from.
TRANSFORMERS_PROBED_VERSION = "5.18"

# Prints {"python": ..., "packages": {name: version}} for the names in argv.
_VERSIONS_PROBE = (
    "import json, sys\n"
    "from importlib.metadata import PackageNotFoundError, version\n"
    "def get(name):\n"
    "    try:\n"
    "        return version(name)\n"
    "    except PackageNotFoundError:\n"
    "        return None\n"
    "print(json.dumps({'python': sys.version.split()[0],\n"
    "                  'packages': {n: get(n) for n in sys.argv[1:]}}))\n"
)
# Same source as requirements-optional.txt; not on PyPI.
LIBOQS_PYTHON = "git+https://github.com/open-quantum-safe/liboqs-python.git"
# Prints the path of the liboqs shared library liboqs-python loaded.
_LIBOQS_PROBE = (
    "import oqs.oqs as m\n"
    "lib = m.native() if hasattr(m, 'native') else m._liboqs\n"
    "print(lib._name)\n"
)


def _exe_name(base_name: str) -> str:
    return f"{base_name}.exe" if sys.platform == "win32" else base_name


def _venv_python(venv_dir: str) -> str:
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _ensure_build_venv(extra_packages=(), requirements=None) -> str:
    """The build venv's interpreter, after installing *requirements*
    (requirements-build.txt by default, or a lock file) and *extra_packages*."""
    python = _venv_python(VENV_DIR)
    if not os.path.exists(python):
        print(f"Creating build venv at {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    requirements = requirements or os.path.join(REPO_ROOT, "requirements-build.txt")
    subprocess.run(
        [python, "-m", "pip", "install", "-q", "-r", requirements, *extra_packages],
        check=True,
    )
    return python


def _version_module() -> dict:
    """src/utils/version.py's globals, loaded without importing the app."""
    return runpy.run_path(os.path.join(REPO_ROOT, VERSION_MODULE))


def _app_version() -> str:
    with open(os.path.join(REPO_ROOT, VERSION_FILE), "r", encoding="utf-8") as f:
        version = f.read().strip()
    if not re.fullmatch(r"\d+(\.\d+){0,3}", version):
        sys.exit(f"{VERSION_FILE} holds {version!r}; it must be up to four dot-separated numbers.")
    return version


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_has(lock_path: str, distribution: str) -> bool:
    """Whether the lock file pins *distribution* (``name==...`` or ``name @ url``)."""
    with open(lock_path, "r", encoding="utf-8") as f:
        for line in f:
            name = re.split(r"[=@<>~!\s;\[]", line.strip(), maxsplit=1)[0]
            if name and _canonical_name(name) == _canonical_name(distribution):
                return True
    return False


def _write_lock_file(python: str) -> None:
    freeze = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(freeze.stdout)
    print(f"Build environment recorded in {LOCK_PATH}")


def _environment_versions(python: str) -> dict:
    probe = subprocess.run(
        [python, "-c", _VERSIONS_PROBE, *RECORDED_DISTRIBUTIONS], capture_output=True, text=True, check=True,
    )
    return json.loads(probe.stdout.strip().splitlines()[-1])


def _check_transformers_version(packages: dict) -> None:
    installed = packages.get("transformers")
    if installed and installed.split(".")[:2] != TRANSFORMERS_PROBED_VERSION.split("."):
        print(
            f"WARNING: transformers {installed} is installed, but TRANSFORMERS_PROBED_DISTRIBUTIONS "
            f"was taken from transformers {TRANSFORMERS_PROBED_VERSION}. If the smoke test exits at a "
            "package metadata lookup, update the list from transformers' "
            "_is_package_available(..., return_version=True) calls."
        )


def _write_build_info(version: str, build_id: str, environment: dict, with_oqs: bool) -> None:
    """build_info.json, bundled at the resource root (utils/version.py reads it)."""
    info = {
        "version": version,
        "build_id": build_id,
        "with_oqs": with_oqs,
        "python": environment["python"],
        "packages": {name: v for name, v in environment["packages"].items() if v},
        # Lets the build compare itself with this checkout at startup.
        "source": {"repo": REPO_ROOT, **_version_module()["source_state"](REPO_ROOT)},
    }
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(BUILD_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("Building " + ", ".join(
        [f"Weidr {version}", f"build {build_id}",
         f"from {info['source']['commit'][:10] or 'unknown commit'}"
         + (" with uncommitted changes" if info["source"]["changes"] else ""),
         f"Python {info['python']}"]
        + [f"{name} {v}" for name, v in sorted(info["packages"].items())]
    ))


def _common_args(exe_name: str, unpack_prefix: str, version: str, build_id: str) -> list:
    args = [
        "--onefile",
        # The exe unpacks its runtime to this folder. A fixed path (no
        # {PID}/{TIME}) makes Nuitka keep it between runs, so only the first
        # run pays the unpacking cost; the build id gives each build its own
        # folder, so a rebuilt exe never reuses a previous build's files.
        # Nothing the app writes goes under unpacked/.
        f"--onefile-tempdir-spec={APP_DATA_SPEC}/unpacked/{unpack_prefix}-{build_id}",
        f"--output-dir={BUILD_DIR}",
        f"--output-filename={exe_name}",
        # Lets Nuitka fetch build helpers (e.g. MinGW on Windows) without prompting.
        "--assume-yes-for-downloads",
        # Version information; Windows shows it in the executable's properties
        # and crash reports. The file version takes numbers only, so the build
        # id goes in the description.
        "--product-name=Weidr",
        f"--product-version={version}",
        f"--file-version={version}",
        f"--file-description=Weidr media browser, build {build_id}",
        f"--include-data-files={BUILD_INFO_PATH}=build_info.json",
        # keyring finds its backends, including the OS one, through package
        # metadata entry points; without it encrypted saves cannot work.
        "--include-distribution-metadata=keyring",
        *[f"--include-distribution-metadata={p}" for p in TRANSFORMERS_PROBED_DISTRIBUTIONS],
        # TorchScript compiles Python functions from their source, which a
        # build does not ship; nothing in Weidr uses it.
        "--module-parameter=torch-disable-jit=yes",
        # The MCP SDK reads its own version from package metadata, and uvicorn
        # loads its protocol and event-loop modules by name.
        "--include-distribution-metadata=mcp",
        "--include-package=uvicorn",
        # utils/pillow_plugins.py imports these by name, which Nuitka can't follow.
        "--include-package=pillow_avif",
        "--include-package=pillow_heif",
        "--include-package=pillow_jxl",
        # clip and open_clip read their BPE vocab (bpe_simple_vocab_16e6.txt.gz)
        # from beside their modules. Nuitka's own open_clip entry ships its
        # model configs and *.tar.gz files only.
        "--include-package-data=clip",
        "--include-package-data=open_clip",
        f"--user-package-configuration-file={PACKAGE_CONFIG}",
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=_pytest",
        "--nofollow-import-to=pytestqt",
    ]
    args += [f"--include-data-dir={d}={d}" for d in DATA_DIRS]
    args += [f"--include-data-files={f}={f}" for f in DATA_FILES]
    if sys.platform == "win32":
        args.append(f"--windows-icon-from-ico={os.path.join('assets', 'icon.ico')}")
    return args


def _gui_args() -> list:
    args = [
        "--enable-plugin=pyside6",
        # sound_player uses QSoundEffect.
        "--include-qt-plugins=multimedia",
    ]
    if sys.platform == "win32":
        # No console window when started from Explorer; when started from a
        # terminal, output (e.g. the smoke test's) goes to that terminal.
        args.append("--windows-console-mode=attach")
    return args


def _headless_args() -> list:
    # app_headless.py's import chain imports no Qt at load time
    # (tests/unit/test_app_headless_qt_freedom.py); it keeps its console window.
    return ["--nofollow-import-to=PySide6"]


def _oqs_args(python: str, with_oqs: bool) -> list:
    if not with_oqs:
        # Keep liboqs-python out even if the build environment has it: without
        # its shared library it would try to build liboqs at the app's startup.
        return ["--nofollow-import-to=oqs"]
    probe = subprocess.run([python, "-c", _LIBOQS_PROBE], capture_output=True, text=True)
    library = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
    if probe.returncode != 0 or not os.path.isabs(library) or not os.path.isfile(library):
        sys.exit(
            "--with-oqs: could not locate the liboqs shared library through liboqs-python.\n"
            f"{probe.stdout}{probe.stderr}\n"
            "Install liboqs (or let liboqs-python build it: needs git, CMake and a C "
            "compiler) or set OQS_INSTALL_PATH to an existing install, then rebuild."
        )
    # The layout liboqs-python searches under OQS_INSTALL_PATH, which
    # utils/encryptor.py points at this folder in the build.
    lib_dir = "bin" if sys.platform == "win32" else "lib"
    print(f"Bundling liboqs: {library}")
    return [
        "--include-package=oqs",
        "--include-distribution-metadata=liboqs-python",
        f"--include-data-files={library}=liboqs/{lib_dir}/{os.path.basename(library)}",
    ]


def _nuitka_env() -> dict:
    # The entry scripts put src/ on sys.path at runtime; Nuitka resolves imports at
    # build time and needs it there already.
    env = dict(os.environ)
    paths = [os.path.join(REPO_ROOT, "src"), REPO_ROOT]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _copy_to_dist(exe_name: str) -> str:
    exe_path = os.path.join(DIST_DIR, exe_name)
    if os.path.isdir(exe_path):
        # Left by a folder build; on macOS/Linux it has the executable's name.
        print(f"Removing the folder build at {exe_path}")
        shutil.rmtree(exe_path)
    os.makedirs(DIST_DIR, exist_ok=True)
    shutil.copy2(os.path.join(BUILD_DIR, exe_name), exe_path)
    return exe_path


def _smoke_test(exe_path: str, with_oqs: bool) -> None:
    # Run from outside the repo so a passing test can't depend on the checkout,
    # with all app data (logs included), config, caches and the key store in a
    # scratch directory so the user's are untouched. The exe still unpacks to
    # the real unpacked/ folder, which a later start of this build reuses.
    command = [exe_path, "--smoke-test", *(["--expect-oqs"] if with_oqs else [])]
    with tempfile.TemporaryDirectory(prefix="weidr-smoke-") as scratch:
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["WEIDR_APP_DATA_DIR"] = scratch
        env["WEIDR_CACHE_DIR"] = os.path.join(scratch, "cache")
        env["WEIDR_CONFIGS_DIR"] = os.path.join(scratch, "configs")
        result = subprocess.run(command, cwd=os.path.expanduser("~"), env=env)
    if result.returncode != 0:
        sys.exit(f"Smoke test failed: `{' '.join(command)}` exited {result.returncode}.")


def _pop_option(args: list, name: str):
    """Remove ``name=value`` from *args* and return the value, or None."""
    for arg in list(args):
        if arg.startswith(name + "="):
            args.remove(arg)
            return arg.split("=", 1)[1]
    return None


# exe base name, entry script, unpack folder prefix, target-specific args
TARGETS = (
    ("Weidr", "app_qt.py", "weidr", _gui_args),
    ("WeidrHeadless", "app_headless.py", "weidr-headless", _headless_args),
)
HEADLESS_TARGET = "WeidrHeadless"


def _targets(with_headless: bool) -> tuple:
    return tuple(t for t in TARGETS if with_headless or t[0] != HEADLESS_TARGET)


if __name__ == "__main__":
    args = sys.argv[1:]
    with_oqs = "--with-oqs" in args
    if with_oqs:
        args.remove("--with-oqs")
    with_headless = "--with-headless" in args
    if with_headless:
        args.remove("--with-headless")
    lock = _pop_option(args, "--lock")
    version = _app_version()
    build_id = time.strftime("%Y%m%d-%H%M%S")

    if "--current-env" in args:
        args.remove("--current-env")
        if lock:
            sys.exit("--lock installs into the build venv; it cannot be combined with --current-env.")
        python = sys.executable
    else:
        if lock:
            lock = os.path.abspath(lock)
            if not os.path.isfile(lock):
                sys.exit(f"--lock: {lock} not found.")
        # A lock file from an OQS build already pins liboqs-python.
        needs_liboqs = with_oqs and not (lock and _lock_has(lock, "liboqs-python"))
        python = _ensure_build_venv((LIBOQS_PYTHON,) if needs_liboqs else (), requirements=lock)

    _write_lock_file(python)
    environment = _environment_versions(python)
    _check_transformers_version(environment["packages"])
    _write_build_info(version, build_id, environment, with_oqs)
    oqs_args = _oqs_args(python, with_oqs)

    for base_name, entry, unpack_prefix, target_args in _targets(with_headless):
        exe_name = _exe_name(base_name)
        nuitka_args = [
            *_common_args(exe_name, unpack_prefix, version, build_id), *target_args(), *oqs_args,
            *args, os.path.join(REPO_ROOT, entry),
        ]
        subprocess.run([python, "-m", "nuitka", *nuitka_args], cwd=REPO_ROOT, env=_nuitka_env(), check=True)
        exe_path = _copy_to_dist(exe_name)
        _smoke_test(exe_path, with_oqs)
        print(f"Built {exe_path}")

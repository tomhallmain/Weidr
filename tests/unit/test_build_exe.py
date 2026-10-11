"""Helpers of build_exe.py, loaded by file path without running a build."""

import importlib.util
import os

import pytest

_BUILD_EXE = os.path.join(os.path.dirname(__file__), "..", "..", "build_exe.py")


@pytest.fixture(scope="module")
def build_exe():
    spec = importlib.util.spec_from_file_location("build_exe_under_test", _BUILD_EXE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pop_option(build_exe):
    args = ["--jobs=4", "--lock=a.txt", "--show-scons"]
    assert build_exe._pop_option(args, "--lock") == "a.txt"
    assert args == ["--jobs=4", "--show-scons"]
    assert build_exe._pop_option(args, "--lock") is None


def test_lock_has(build_exe, tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("torch==2.14.1\nliboqs-python @ git+https://example.invalid/liboqs-python\n",
                    encoding="utf-8")
    assert build_exe._lock_has(str(lock), "liboqs_python")
    assert build_exe._lock_has(str(lock), "Torch")
    assert not build_exe._lock_has(str(lock), "torchvision")


def test_nuitka_args(build_exe):
    args = build_exe._nuitka_args("1.2.3", "20261010-120000")
    assert "--include-distribution-metadata=keyring" in args
    assert "--module-parameter=torch-disable-jit=yes" in args
    assert "--file-version=1.2.3" in args
    assert any("build 20261010-120000" in a for a in args if a.startswith("--file-description="))
    for name in build_exe.TRANSFORMERS_PROBED_DISTRIBUTIONS:
        assert f"--include-distribution-metadata={name}" in args


def test_onefile_unpacks_to_a_folder_unpack_cleanup_recognizes(build_exe):
    from utils import unpack_cleanup

    args = build_exe._nuitka_args("1.2.3", "20261010-120000")
    assert "--onefile" in args and "--standalone" not in args
    spec = build_exe._pop_option(args, "--onefile-tempdir-spec")
    parent, folder = spec.split("/")[-2:]
    assert parent == unpack_cleanup.UNPACK_PARENT_NAME
    assert folder == unpack_cleanup.UNPACK_DIR_PREFIX + "20261010-120000"


def test_build_without_oqs_leaves_it_out(build_exe):
    assert build_exe._oqs_args(python=None, with_oqs=False) == ["--nofollow-import-to=oqs"]


def test_app_version_is_numeric(build_exe):
    assert build_exe._app_version().replace(".", "").isdigit()

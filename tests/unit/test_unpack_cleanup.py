"""Tests for utils/unpack_cleanup.py, run as if this were a build unpacked
under tmp_path. The process list is stubbed; every file operation stays in
tmp_path."""

import os

import pytest

from utils import unpack_cleanup

CURRENT = "weidr-20261009-120000"


@pytest.fixture
def unpacked(monkeypatch, tmp_path):
    """An unpacked/ dir holding the running build's folder, with no other
    processes running from it; returns the dir."""
    root = tmp_path / unpack_cleanup.UNPACK_PARENT_NAME
    (root / CURRENT / "utils").mkdir(parents=True)
    monkeypatch.setattr(unpack_cleanup, "is_compiled", lambda: True)
    in_build(monkeypatch, root / CURRENT)
    running(monkeypatch, set())
    return root


def in_build(monkeypatch, folder):
    monkeypatch.setattr(unpack_cleanup, "resource_root", lambda: str(folder))


def running(monkeypatch, executables):
    """Stub the running processes' executables (None: cannot be listed)."""
    value = None if executables is None else {unpack_cleanup._normalize(str(e)) for e in executables}
    monkeypatch.setattr(unpack_cleanup, "_running_executables", lambda: value)


def _build_folder(root, name):
    folder = root / name
    (folder / "locale").mkdir(parents=True)
    (folder / "locale" / "base.mo").write_bytes(b"x")
    (folder / "Weidr").write_bytes(b"exe")
    return folder


@pytest.fixture(params=[True, False], ids=["windows", "macos-linux"])
def platform(request, monkeypatch):
    monkeypatch.setattr(unpack_cleanup, "_IS_WINDOWS", request.param)
    return request.param


class TestRemoveStaleUnpackDirs:
    def test_removes_older_builds_and_keeps_the_running_one(self, unpacked, platform):
        old = _build_folder(unpacked, "weidr-20261001-090000")
        removed = unpack_cleanup.remove_stale_unpack_dirs()
        assert not old.exists()
        assert not os.path.exists(str(old) + ".removing")
        assert (unpacked / CURRENT).is_dir()
        assert len(removed) == 1

    def test_skips_a_folder_a_running_process_executes_from(self, unpacked, platform, monkeypatch):
        in_use = _build_folder(unpacked, "weidr-20261005-100000")
        idle = _build_folder(unpacked, "weidr-20261001-090000")
        running(monkeypatch, {in_use / "Weidr"})
        unpack_cleanup.remove_stale_unpack_dirs()
        assert in_use.is_dir()
        assert not idle.exists()

    def test_a_similar_folder_name_is_not_mistaken_for_in_use(self, unpacked, platform, monkeypatch):
        old = _build_folder(unpacked, "weidr-20261001-09")
        _build_folder(unpacked, "weidr-20261001-0900")
        running(monkeypatch, {unpacked / "weidr-20261001-0900" / "Weidr"})
        unpack_cleanup.remove_stale_unpack_dirs()
        assert not old.exists()

    def test_keeps_the_running_folder_seen_under_another_name(self, unpacked, platform, monkeypatch):
        # A symlink stands in for the Windows 8.3 short name of the running folder.
        alias = unpacked / "weidr-~1"
        try:
            alias.symlink_to(unpacked / CURRENT, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks unavailable")
        in_build(monkeypatch, alias)
        old = _build_folder(unpacked, "weidr-20261001-090000")
        unpack_cleanup.remove_stale_unpack_dirs()
        assert (unpacked / CURRENT).is_dir()
        assert not old.exists()

    def test_leaves_other_entries_alone(self, unpacked, platform):
        other = _build_folder(unpacked, "something-else")
        stray_file = unpacked / "weidr-notes.txt"
        stray_file.write_text("keep", encoding="utf-8")
        unpack_cleanup.remove_stale_unpack_dirs()
        assert other.is_dir()
        assert stray_file.is_file()

    def test_skips_a_folder_that_cannot_be_renamed(self, unpacked, platform, monkeypatch):
        locked = _build_folder(unpacked, "weidr-20261005-100000")
        real_rename = os.rename

        def rename(src, dst):
            if os.path.basename(src) == locked.name:
                raise PermissionError("in use")
            real_rename(src, dst)

        monkeypatch.setattr(unpack_cleanup.os, "rename", rename)
        assert unpack_cleanup.remove_stale_unpack_dirs() == []
        assert locked.is_dir()

    def test_finishes_a_folder_renamed_at_an_earlier_start(self, unpacked, platform):
        leftover = _build_folder(unpacked, "weidr-20261001-090000.removing")
        unpack_cleanup.remove_stale_unpack_dirs()
        assert not leftover.exists()

    def test_without_a_process_list_windows_relies_on_the_rename(self, unpacked, monkeypatch):
        monkeypatch.setattr(unpack_cleanup, "_IS_WINDOWS", True)
        running(monkeypatch, None)
        old = _build_folder(unpacked, "weidr-20261001-090000")
        unpack_cleanup.remove_stale_unpack_dirs()
        assert not old.exists()

    def test_without_a_process_list_other_platforms_remove_nothing(self, unpacked, monkeypatch):
        monkeypatch.setattr(unpack_cleanup, "_IS_WINDOWS", False)
        running(monkeypatch, None)
        old = _build_folder(unpacked, "weidr-20261001-090000")
        assert unpack_cleanup.remove_stale_unpack_dirs() == []
        assert old.is_dir()

    def test_does_nothing_in_a_checkout(self, unpacked, platform, monkeypatch):
        old = _build_folder(unpacked, "weidr-20261001-090000")
        monkeypatch.setattr(unpack_cleanup, "is_compiled", lambda: False)
        assert unpack_cleanup.remove_stale_unpack_dirs() == []
        assert old.is_dir()

    def test_does_nothing_outside_an_unpack_folder(self, unpacked, platform, monkeypatch, tmp_path):
        old = _build_folder(unpacked, "weidr-20261001-090000")
        in_build(monkeypatch, tmp_path / "repo")
        assert unpack_cleanup.remove_stale_unpack_dirs() == []
        assert old.is_dir()


class TestRunningExecutables:
    def test_lists_this_process(self):
        psutil = pytest.importorskip("psutil")
        executables = unpack_cleanup._running_executables()
        assert unpack_cleanup._normalize(psutil.Process().exe()) in executables

    def test_none_without_psutil(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_psutil)
        assert unpack_cleanup._running_executables() is None

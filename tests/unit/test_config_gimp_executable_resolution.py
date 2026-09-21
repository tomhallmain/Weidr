"""Tests for Config's GIMP executable resolution.

Regression coverage for a real failure: with ``gimp_exe_loc`` set to a bare
command name (the shipped default "gimp-2.10", or the "gimp-3.0" in
config_example.json), every Open-in-GIMP request died with
``FileNotFoundError: [WinError 2]``.  GIMP used to be launched through
``shell=True``, so cmd.exe resolved the name against PATH/PATHEXT; once the
launch became a plain argv list nothing resolved it any more, and Windows
CreateProcess only appends ".exe" when the name has no extension -- the ".0" in
"gimp-3.0" reads as one, so the bare name was looked up verbatim and missed.

Validation already resolved the name via shutil.which() to run its --version
check, but discarded the result and left ``gimp_exe_loc`` as the bare name.
These tests pin that the resolved path is what gets kept.

No real GIMP required -- resolution and the version subprocess are both mocked.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import utils.config as _cfg_module


RESOLVED = os.path.join("C:", os.sep, "Program Files", "GIMP 3", "bin", "gimp-3.0.exe")


def _config():
    """The per-test config instance (patched by isolated_singletons)."""
    return _cfg_module.config


@pytest.fixture()
def cfg():
    c = _config()
    c._gimp_validated = False
    return c


class TestResolveGimpExecutable:
    def test_bare_name_resolves_through_path_lookup(self, cfg):
        with patch("utils.config.shutil.which", return_value=RESOLVED) as which:
            assert cfg._resolve_gimp_executable("gimp-3.0") == RESOLVED
        which.assert_called_once_with("gimp-3.0")

    def test_bare_name_not_on_path_returns_none(self, cfg):
        with patch("utils.config.shutil.which", return_value=None):
            assert cfg._resolve_gimp_executable("gimp-3.0") is None

    def test_existing_file_is_used_without_path_lookup(self, cfg, tmp_path):
        exe = tmp_path / "gimp-3.0.exe"
        exe.write_text("")
        with patch("utils.config.shutil.which") as which:
            assert cfg._resolve_gimp_executable(str(exe)) == str(exe)
        which.assert_not_called()

    def test_relative_existing_path_is_made_absolute(self, cfg, tmp_path, monkeypatch):
        """A relative configured path must survive a later cwd change."""
        exe = tmp_path / "gimp-3.0.exe"
        exe.write_text("")
        monkeypatch.chdir(tmp_path)
        resolved = cfg._resolve_gimp_executable("gimp-3.0.exe")
        assert os.path.isabs(resolved)
        assert os.path.basename(resolved) == "gimp-3.0.exe"

    @pytest.mark.parametrize("empty", [None, ""])
    def test_empty_input_returns_none(self, cfg, empty):
        assert cfg._resolve_gimp_executable(empty) is None


class TestValidateAndFindGimpStoresResolvedPath:
    def test_configured_bare_name_is_replaced_by_resolved_path(self, cfg):
        """The bug: a bare name survived validation and reached subprocess.Popen."""
        cfg.gimp_exe_loc = "gimp-3.0"
        with patch.object(_cfg_module.Config, "_resolve_gimp_executable", return_value=RESOLVED), \
             patch.object(_cfg_module.Config, "_is_valid_gimp_installation", return_value=True), \
             patch.object(_cfg_module.Config, "_check_gimp_version_for_gegl"):
            cfg.validate_and_find_gimp()
        assert cfg.gimp_exe_loc == RESOLVED

    def test_validation_runs_against_the_resolved_path(self, cfg):
        cfg.gimp_exe_loc = "gimp-3.0"
        with patch.object(_cfg_module.Config, "_resolve_gimp_executable", return_value=RESOLVED), \
             patch.object(_cfg_module.Config, "_is_valid_gimp_installation", return_value=True) as valid, \
             patch.object(_cfg_module.Config, "_check_gimp_version_for_gegl"):
            cfg.validate_and_find_gimp()
        valid.assert_called_once_with(RESOLVED)

    def test_unresolvable_name_falls_through_to_autodetect(self, cfg):
        cfg.gimp_exe_loc = "gimp-3.0"
        with patch.object(_cfg_module.Config, "_resolve_gimp_executable", side_effect=[None, RESOLVED]), \
             patch.object(_cfg_module.Config, "_find_gimp_installation", return_value="gimp-2.10"), \
             patch.object(_cfg_module.Config, "_check_gimp_version_for_gegl"):
            cfg.validate_and_find_gimp()
        assert cfg.gimp_exe_loc == RESOLVED

    def test_autodetected_bare_name_is_also_resolved(self, cfg):
        """_find_gimp_unix returns bare names like "gimp-2.10", not paths."""
        cfg.gimp_exe_loc = None
        with patch.object(_cfg_module.Config, "_resolve_gimp_executable", return_value=RESOLVED), \
             patch.object(_cfg_module.Config, "_find_gimp_installation", return_value="gimp-2.10"), \
             patch.object(_cfg_module.Config, "_check_gimp_version_for_gegl"):
            cfg.validate_and_find_gimp()
        assert cfg.gimp_exe_loc == RESOLVED

    def test_no_gimp_found_leaves_none_and_disables_gegl(self, cfg):
        cfg.gimp_exe_loc = None
        cfg.gimp_gegl_enabled = True
        with patch.object(_cfg_module.Config, "_find_gimp_installation", return_value=None):
            cfg.validate_and_find_gimp()
        assert cfg.gimp_exe_loc is None
        assert cfg.gimp_gegl_enabled is False

    def test_resolution_does_not_rewrite_the_users_config_file(self, cfg):
        """persist() rebuilds from self.dict, so the portable configured name stays.

        Resolution is a runtime concern -- baking a machine-specific absolute path
        into the user's config.json would break it on any other machine.
        """
        cfg.gimp_exe_loc = "gimp-3.0"
        cfg.dict["gimp_exe_loc"] = "gimp-3.0"
        with patch.object(_cfg_module.Config, "_resolve_gimp_executable", return_value=RESOLVED), \
             patch.object(_cfg_module.Config, "_is_valid_gimp_installation", return_value=True), \
             patch.object(_cfg_module.Config, "_check_gimp_version_for_gegl"):
            cfg.validate_and_find_gimp()
        assert cfg.gimp_exe_loc == RESOLVED
        assert cfg._build_persisted_config_dict()["gimp_exe_loc"] == "gimp-3.0"


class TestGimpVersionProbeExecutable:
    """On Windows, GIMP 3.2's GUI exe prints --version to its own console window,
    leaving captured stdout empty; the gimp-console sibling is probed instead."""

    def test_windows_prefers_console_sibling(self, cfg, tmp_path, monkeypatch):
        monkeypatch.setattr(_cfg_module.sys, "platform", "win32")
        gui = tmp_path / "gimp-3.2.exe"
        console = tmp_path / "gimp-console-3.2.exe"
        gui.write_text("")
        console.write_text("")
        assert cfg.gimp_version_probe_executable(str(gui)) == str(console)

    def test_windows_without_console_sibling_keeps_gui_exe(self, cfg, tmp_path, monkeypatch):
        monkeypatch.setattr(_cfg_module.sys, "platform", "win32")
        gui = tmp_path / "gimp-3.2.exe"
        gui.write_text("")
        assert cfg.gimp_version_probe_executable(str(gui)) == str(gui)

    def test_non_windows_is_unchanged(self, cfg, tmp_path, monkeypatch):
        monkeypatch.setattr(_cfg_module.sys, "platform", "linux")
        gui = tmp_path / "gimp-3.2"
        (tmp_path / "gimp-console-3.2").write_text("")
        assert cfg.gimp_version_probe_executable(str(gui)) == str(gui)

    def test_validation_probes_console_but_keeps_gui_path(self, cfg, tmp_path, monkeypatch):
        monkeypatch.setattr(_cfg_module.sys, "platform", "win32")
        gui = tmp_path / "gimp-3.2.exe"
        console = tmp_path / "gimp-console-3.2.exe"
        gui.write_text("")
        console.write_text("")
        cfg.gimp_exe_loc = str(gui)
        version = type("R", (), {"returncode": 0,
                                 "stdout": "GNU Image Manipulation Program version 3.2.6\n"})()
        with patch("utils.config.subprocess.run", return_value=version) as run:
            cfg.validate_and_find_gimp()
        assert cfg.gimp_exe_loc == str(gui)
        assert all(call.args[0][0] == str(console) for call in run.call_args_list)
        assert cfg.gimp_gegl_enabled is True

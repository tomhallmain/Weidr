"""Optional Pillow plugin registration (HEIF, AVIF, JXL)."""

import importlib.util

import pytest
from PIL import Image

from utils import pillow_plugins as pp
from utils.pillow_plugins import ensure_pillow_plugins_registered


@pytest.fixture(autouse=True)
def _reset_pillow_plugin_registration():
    pp._registered = False
    pp.has_imported_pillow_jxl = False
    yield
    pp._registered = False
    pp.has_imported_pillow_jxl = False


def test_ensure_pillow_plugins_registered_is_idempotent():
    ensure_pillow_plugins_registered()
    ensure_pillow_plugins_registered()
    assert pp._registered is True


def test_jxl_flag_matches_installed_plugin():
    ensure_pillow_plugins_registered()
    installed = importlib.util.find_spec("pillow_jxl") is not None
    assert pp.has_imported_pillow_jxl is installed


@pytest.mark.skipif(
    importlib.util.find_spec("pillow_jxl") is None,
    reason="pillow-jxl-plugin not installed",
)
def test_jxl_roundtrip_when_plugin_installed(tmp_path):
    ensure_pillow_plugins_registered()
    path = tmp_path / "sample.jxl"
    Image.new("RGB", (6, 6), (10, 20, 30)).save(path)
    with Image.open(path) as img:
        assert img.size == (6, 6)

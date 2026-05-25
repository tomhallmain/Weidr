"""Config defaults and classification for expanded image/video extensions (phase 1, no audio)."""

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from files.file_browser import FileBrowser
from utils.config import Config, config
from utils.constants import MediaType
from utils.media_utils import (
    DEFAULT_VIDEO_EXTENSIONS,
    get_media_type_for_path,
    is_video_path_by_extension,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_EXAMPLE = _REPO_ROOT / "configs" / "config_example.json"

TIER_A_IMAGE_EXTENSIONS = (".jfif", ".jpe", ".pjpeg", ".pjp", ".ico", ".heif", ".apng")


def _load_config_example() -> dict:
    return json.loads(_CONFIG_EXAMPLE.read_text(encoding="utf-8"))


def test_config_example_lists_tier_a_image_extensions():
    data = _load_config_example()
    for ext in TIER_A_IMAGE_EXTENSIONS:
        assert ext in data["image_types"]


def test_config_example_video_types_exclude_m4a_and_match_fallback():
    data = _load_config_example()
    vt = data["video_types"]
    assert ".m4a" not in vt
    assert set(vt) == set(DEFAULT_VIDEO_EXTENSIONS)


def test_migrate_removes_m4a_from_video_types():
    cfg = Config.__new__(Config)
    cfg.dict = {"video_types": [".mp4", ".m4a", ".webm"]}
    Config._migrate_legacy_keys(cfg)
    assert ".m4a" not in cfg.dict["video_types"]
    assert ".webm" in cfg.dict["video_types"]


@pytest.mark.parametrize("ext", TIER_A_IMAGE_EXTENSIONS)
def test_get_media_type_tier_a_extensions_are_image(tmp_path, ext):
    path = str(tmp_path / f"sample{ext}")
    (tmp_path / f"sample{ext}").write_bytes(b"\xff\xd8\xff" if ext != ".ico" else b"\x00\x00\x01\x00")
    assert get_media_type_for_path(path) == MediaType.IMAGE


@pytest.mark.parametrize("ext", (".webm", ".ogv", ".mpeg", ".mpg"))
def test_get_media_type_video_extensions(tmp_path, monkeypatch, ext):
    path = str(tmp_path / f"clip{ext}")
    (tmp_path / f"clip{ext}").write_bytes(b"\x00")
    monkeypatch.setattr(config, "enable_videos", True)
    monkeypatch.setattr(config, "video_types", list(DEFAULT_VIDEO_EXTENSIONS))
    assert get_media_type_for_path(path) == MediaType.VIDEO
    assert is_video_path_by_extension(path)


def test_file_browser_gathers_jfif_when_in_file_types(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "file_types", [".jfif", ".png"])
    root = tmp_path
    Image.new("RGB", (4, 4), (100, 100, 100)).save(root / "a.jfif", format="JPEG")
    Image.new("RGB", (4, 4), (50, 50, 50)).save(root / "b.png", format="PNG")
    (root / "skip.jpg").write_bytes(b"\xff\xd8\xff")

    fb = FileBrowser(str(root), recursive=False)
    fb.set_directory(str(root))
    names = sorted(os.path.basename(p) for p in fb.get_files())
    assert names == ["a.jfif", "b.png"]


def test_compare_args_include_gifs_follows_enable_gifs(monkeypatch):
    from compare.compare_args import CompareArgs

    monkeypatch.setattr(config, "enable_gifs", True)
    monkeypatch.setattr(config, "video_types", [".mp4"])
    assert CompareArgs().include_gifs is True

    monkeypatch.setattr(config, "enable_gifs", False)
    assert CompareArgs().include_gifs is False

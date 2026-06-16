"""Phase 2 audio: classification, file_types, and prevalidation gating."""

import json
from pathlib import Path

import pytest

from files.file_browser import FileBrowser
import os

from utils.audio_media import (
    DEFAULT_AUDIO_EXTENSIONS,
    is_audio_path_by_extension,
)
import utils.audio_media as audio_media
import utils.media_utils as media_utils
from utils.config import config
from utils.constants import MediaType
from utils.media_utils import (
    get_media_type_for_path,
    is_video_for_display,
    is_video_file,
    is_video_path_by_extension,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_EXAMPLE = _REPO_ROOT / "configs" / "config_example.json"


def test_config_example_lists_audio_types():
    data = json.loads(_CONFIG_EXAMPLE.read_text(encoding="utf-8"))
    assert ".mp3" in data["audio_types"]
    assert data.get("enable_audio") is True
    assert "enable_audio_prevalidation" not in data


def test_m4a_is_audio_not_video(tmp_path, monkeypatch):
    path = str(tmp_path / "track.m4a")
    (tmp_path / "track.m4a").write_bytes(b"\x00")
    monkeypatch.setattr(config, "enable_audio", True)
    monkeypatch.setattr(config, "enable_videos", True)
    monkeypatch.setattr(config, "audio_types", list(DEFAULT_AUDIO_EXTENSIONS))
    monkeypatch.setattr(config, "video_types", [".mp4", ".mkv"])
    assert get_media_type_for_path(path) == MediaType.AUDIO
    assert is_audio_path_by_extension(path)
    assert not is_video_path_by_extension(path)


def test_m4a_with_real_video_stream_classifies_as_video(tmp_path, monkeypatch):
    """A .m4a whose container actually carries a real (non-cover-art) video
    stream -- a mislabeled/dual-purpose file -- should classify as VIDEO, not
    AUDIO, once probed via has_real_video_stream()."""
    path = str(tmp_path / "movie.m4a")
    (tmp_path / "movie.m4a").write_bytes(b"\x00")
    monkeypatch.setattr(config, "enable_audio", True)
    monkeypatch.setattr(config, "enable_videos", True)
    monkeypatch.setattr(config, "audio_types", list(DEFAULT_AUDIO_EXTENSIONS))
    monkeypatch.setattr(config, "video_types", [".mp4", ".mkv"])

    # is_audio_path_by_extension() calls has_real_video_stream() directly (same module).
    monkeypatch.setattr(audio_media, "has_real_video_stream", lambda p: True)
    # media_utils imported has_real_video_stream by name, so it needs its own patch target.
    monkeypatch.setattr(media_utils, "has_real_video_stream", lambda p: True)

    assert not is_audio_path_by_extension(path)
    assert is_video_path_by_extension(path)
    assert is_video_for_display(path)
    assert is_video_file(path)
    assert get_media_type_for_path(path) == MediaType.VIDEO


def test_m4a_with_unprobeable_content_stays_audio(tmp_path, monkeypatch):
    """ffprobe failure (garbage bytes) must not flip classification -- the
    override only fires when a real video stream is positively confirmed."""
    path = str(tmp_path / "track.m4a")
    (tmp_path / "track.m4a").write_bytes(b"\x00")
    monkeypatch.setattr(config, "enable_audio", True)
    monkeypatch.setattr(config, "enable_videos", True)
    monkeypatch.setattr(config, "audio_types", list(DEFAULT_AUDIO_EXTENSIONS))
    monkeypatch.setattr(config, "video_types", [".mp4", ".mkv"])

    assert is_audio_path_by_extension(path)
    assert not is_video_path_by_extension(path)
    assert get_media_type_for_path(path) == MediaType.AUDIO


def test_get_media_type_audio_respects_enable_flag(tmp_path, monkeypatch):
    path = str(tmp_path / "a.mp3")
    (tmp_path / "a.mp3").write_bytes(b"\x00")
    monkeypatch.setattr(config, "enable_audio", True)
    assert get_media_type_for_path(path) == MediaType.AUDIO
    monkeypatch.setattr(config, "enable_audio", False)
    assert get_media_type_for_path(path) == MediaType.UNCONFIGURED


def test_file_browser_gathers_mp3_when_in_file_types(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "file_types", [".mp3", ".png"])
    root = tmp_path
    (root / "a.mp3").write_bytes(b"ID3")
    (root / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (root / "skip.jpg").write_bytes(b"\xff\xd8\xff")

    fb = FileBrowser(str(root), recursive=False)
    fb.set_directory(str(root))
    names = sorted(os.path.basename(p) for p in fb.get_files())
    assert names == ["a.mp3", "b.png"]

"""
Audio path classification and browse/slideshow policy (phase 2).

Playback UI lives in ``ui.app_window.audio_playback``; this module stays
free of Qt/VLC so tests and file ops can import it cheaply.
"""

from __future__ import annotations

import os

# Default browse extensions when ``audio_types`` is missing from config.
DEFAULT_AUDIO_EXTENSIONS = (
    ".mp3", ".flac", ".ogg", ".opus", ".wav", ".m4a", ".m4b",
    ".aac", ".wma", ".weba", ".aiff", ".aif", ".au", ".ape", ".wv",
)

# Masonry grid placeholder when no thumbnail is decoded.
MASONRY_AUDIO_TILE_LABEL = "\u266a"  # ♪


def get_audio_extensions() -> tuple[str, ...]:
    """Configured audio extensions, lowercased."""
    from utils.config import config
    at = getattr(config, "audio_types", None)
    if at is None:
        return DEFAULT_AUDIO_EXTENSIONS
    return tuple(str(e).lower() for e in at)


def is_audio_path_by_extension(path: str) -> bool:
    """True when *path*'s suffix is a configured audio extension (no existence check)."""
    if not path:
        return False
    path_lower = path.lower()
    return any(path_lower.endswith(ext) for ext in get_audio_extensions())


def is_audio_for_display(path: str) -> bool:
    """True when audio is enabled in config and *path* looks like an audio file."""
    from utils.config import config
    return bool(getattr(config, "enable_audio", False)) and is_audio_path_by_extension(path)


def is_audio_file(path: str) -> bool:
    """Existing file that is enabled audio (not for ffmpeg video ops)."""
    from utils.config import config
    if not path or not os.path.isfile(path):
        return False
    if not getattr(config, "enable_audio", False):
        return False
    return is_audio_path_by_extension(path)


def audio_placeholder_title(path: str) -> str:
    """Short label for media frame when VLC is unavailable."""
    from utils.translations import I18N
    _ = I18N._
    return _("Audio: ") + os.path.basename(path)

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

# MP4-family extension that may legitimately carry a real video stream (the
# container is identical to .mp4/.m4v) -- unlike .mp3/.flac/etc, which never do.
# Probed lazily via ffprobe before trusting the extension; see has_real_video_stream.
AMBIGUOUS_AUDIO_VIDEO_EXTENSIONS = (".m4a",)


def is_ambiguous_audio_video_extension(path: str) -> bool:
    """True for audio extensions whose container can also hold a real video stream."""
    return bool(path) and path.lower().endswith(AMBIGUOUS_AUDIO_VIDEO_EXTENSIONS)


def has_real_video_stream(path: str) -> bool:
    """
    True when *path* (an :data:`AMBIGUOUS_AUDIO_VIDEO_EXTENSIONS` file) actually
    contains a real, non-cover-art video stream rather than being audio-only.

    Lazily imports the ffprobe-backed :func:`image.video_ops.probe_has_video_stream`
    to avoid a module import cycle (video_ops depends on media_utils, which depends
    on this module). Only meaningful for the handful of ambiguous extensions --
    never consulted for ordinary audio types, so bulk directory scans (which
    classify by extension set only) are unaffected.
    """
    if not is_ambiguous_audio_video_extension(path) or not os.path.isfile(path):
        return False
    try:
        from image.video_ops import probe_has_video_stream
        return probe_has_video_stream(path)
    except Exception:
        return False


def get_audio_extensions() -> tuple[str, ...]:
    """Configured audio extensions, lowercased."""
    from utils.config import config
    at = getattr(config, "audio_types", None)
    if at is None:
        return DEFAULT_AUDIO_EXTENSIONS
    return tuple(str(e).lower() for e in at)


def is_audio_path_by_extension(path: str) -> bool:
    """True when *path*'s suffix is a configured audio extension.

    No existence check, except for :data:`AMBIGUOUS_AUDIO_VIDEO_EXTENSIONS`
    (currently just ``.m4a``) -- those are probed via :func:`has_real_video_stream`
    and excluded when the file actually contains a real video stream.
    """
    if not path:
        return False
    path_lower = path.lower()
    if not any(path_lower.endswith(ext) for ext in get_audio_extensions()):
        return False
    if is_ambiguous_audio_video_extension(path) and has_real_video_stream(path):
        return False
    return True


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
    from utils.translations import _
    return _("Audio: ") + os.path.basename(path)

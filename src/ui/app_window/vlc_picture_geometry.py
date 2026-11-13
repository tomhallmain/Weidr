"""
Shape of the picture libvlc shows for the current video.

video_get_size() reports decoded pixels. libvlc then applies the sample
aspect ratio and the orientation tag: a phone clip stored as 1920x1080 with a
90-degree tag plays as portrait, and a 720x576 stream with SAR 16:15 plays
768 wide. Sizing to video_get_size() alone gives the wrong shape for those
files, and libvlc fills the difference with black bars.
"""

from __future__ import annotations

import ctypes
import functools
from dataclasses import dataclass

from utils.logging_setup import get_logger

logger = get_logger("vlc_picture_geometry")

# libvlc_track_type_t: libvlc_track_video
_VIDEO_TRACK_TYPE = 1
# libvlc_video_orient_t left_top, left_bottom, right_top, right_bottom: the
# picture is turned by a quarter, so width and height trade places.
_QUARTER_TURN_ORIENTATIONS = frozenset({4, 5, 6, 7})


@dataclass(frozen=True)
class VideoTrackShape:
    """Fields of a libvlc video track that change the displayed shape."""

    width: int
    height: int
    sar_num: int
    sar_den: int
    orientation: int


@dataclass(frozen=True)
class VlcPicture:
    """The picture as displayed, and the pixel grid edits are made in.

    ``display_w``/``display_h`` give the on-screen aspect (SAR and orientation
    applied). ``frame_w``/``frame_h`` are decoded pixels after orientation,
    without SAR: the grid ffmpeg's filters see, since ffmpeg auto-rotates by
    default and crops in stored pixels.
    """

    display_w: float
    display_h: float
    frame_w: int
    frame_h: int


class _LibvlcVideoTrack(ctypes.Structure):
    """Leading fields of libvlc_video_track_t (projection and pose follow, unread)."""

    _fields_ = (
        ("height", ctypes.c_uint),
        ("width", ctypes.c_uint),
        ("sar_num", ctypes.c_uint),
        ("sar_den", ctypes.c_uint),
        ("frame_rate_num", ctypes.c_uint),
        ("frame_rate_den", ctypes.c_uint),
        ("orientation", ctypes.c_int),
    )


class _LibvlcMediaTrack(ctypes.Structure):
    """libvlc_media_track_t as libvlc 3 lays it out.

    python-vlc 3.0.21203's MediaTrack declares the audio/video/subtitle union
    as three separate pointers, so its ``video`` field reads i_bitrate and the
    padding after it. Dereferencing that is an access violation for any track
    with a nonzero bitrate, which kills the process. ``details`` is the union:
    one pointer, cast by ``type``.
    """

    _fields_ = (
        ("codec", ctypes.c_uint32),
        ("original_fourcc", ctypes.c_uint32),
        ("id", ctypes.c_int),
        ("type", ctypes.c_int),
        ("profile", ctypes.c_int),
        ("level", ctypes.c_int),
        ("details", ctypes.c_void_p),
        ("bitrate", ctypes.c_uint),
        ("language", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
    )


_TrackArray = ctypes.POINTER(ctypes.POINTER(_LibvlcMediaTrack))


@functools.lru_cache(maxsize=1)
def _libvlc_track_functions():
    """(tracks_get, tracks_release) bound with the layout above."""
    import vlc

    tracks_get = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(_TrackArray)
    )(("libvlc_media_tracks_get", vlc.dll))
    tracks_release = ctypes.CFUNCTYPE(None, _TrackArray, ctypes.c_uint)(
        ("libvlc_media_tracks_release", vlc.dll)
    )
    return tracks_get, tracks_release


def shapes_from_track_array(array, count: int) -> list[tuple[int, VideoTrackShape]]:
    """(track id, shape) of each video track in a libvlc track array."""
    shapes = []
    for index in range(count):
        entry = array[index]
        if not entry:
            continue
        track = entry.contents
        if track.type != _VIDEO_TRACK_TYPE or not track.details:
            continue
        video = _LibvlcVideoTrack.from_address(track.details)
        shapes.append((
            int(track.id),
            VideoTrackShape(
                width=int(video.width),
                height=int(video.height),
                sar_num=int(video.sar_num),
                sar_den=int(video.sar_den),
                orientation=int(video.orientation),
            ),
        ))
    return shapes


def list_video_tracks(media) -> list[tuple[int, VideoTrackShape]]:
    """(track id, shape) of each video track libvlc lists for a python-vlc Media.

    Calls libvlc directly: python-vlc's Media.tracks_get() reads the track
    struct with the wrong layout (see _LibvlcMediaTrack) and never releases
    the array.
    """
    handle = getattr(media, "_as_parameter_", None)
    if not isinstance(handle, ctypes.c_void_p) or not handle:
        raise TypeError("not a libvlc media handle: {0!r}".format(media))
    tracks_get, tracks_release = _libvlc_track_functions()
    array = _TrackArray()
    count = int(tracks_get(handle, ctypes.byref(array)))
    if count <= 0 or not array:
        return []
    try:
        return shapes_from_track_array(array, count)
    finally:
        tracks_release(array, count)


def read_video_track_shape(player) -> VideoTrackShape | None:
    """Shape fields of the playing video track, or None before libvlc lists it."""
    try:
        media = player.get_media()
    except Exception:
        return None
    if media is None:
        return None
    try:
        selected = int(player.video_get_track())
    except Exception:
        selected = None
    try:
        tracks = list_video_tracks(media)
    except Exception as exc:
        logger.debug("read_video_track_shape: could not list tracks: %s", exc)
        return None
    for track_id, shape in tracks:
        if selected is not None and track_id == selected:
            return shape
    return tracks[0][1] if tracks else None


def _reported_already_turned(width: int, height: int, track: VideoTrackShape) -> bool:
    """True when video_get_size() already reports the picture turned.

    libvlc 3 reports the decoder's unturned size. This compares against the
    track's own orientation of width and height so a build that reports the
    turned size is not turned a second time.
    """
    if track.width <= 0 or track.height <= 0 or track.width == track.height:
        return False
    if width == height:
        return False
    return (width > height) != (track.width > track.height)


def picture_from(width: int, height: int, track: VideoTrackShape | None) -> VlcPicture:
    """Displayed shape of a ``width`` x ``height`` decoded picture."""
    stored_w, stored_h = width, height
    sar = 1.0
    quarter_turn = False
    if track is not None:
        if track.sar_num > 0 and track.sar_den > 0:
            sar = track.sar_num / track.sar_den
        quarter_turn = track.orientation in _QUARTER_TURN_ORIENTATIONS
        if quarter_turn and _reported_already_turned(width, height, track):
            stored_w, stored_h = height, width
    # SAR stretches the stored x axis, so it is applied before turning.
    display_w, display_h = stored_w * sar, float(stored_h)
    frame_w, frame_h = stored_w, stored_h
    if quarter_turn:
        display_w, display_h = display_h, display_w
        frame_w, frame_h = frame_h, frame_w
    return VlcPicture(display_w, display_h, frame_w, frame_h)


def read_picture(player, track: VideoTrackShape | None) -> VlcPicture | None:
    """Displayed picture of ``player``, or None until libvlc reports a size."""
    try:
        width, height = player.video_get_size(0)
    except Exception:
        return None
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return picture_from(width, height, track)


def fit_rect(
    container_w: int, container_h: int, aspect_w: float, aspect_h: float
) -> tuple[int, int, int, int]:
    """(x, y, w, h) of an ``aspect_w:aspect_h`` box fitted and centred in the container."""
    container_w = max(int(container_w), 1)
    container_h = max(int(container_h), 1)
    scale = min(container_w / aspect_w, container_h / aspect_h)
    width = max(1, int(aspect_w * scale))
    height = max(1, int(aspect_h * scale))
    return (container_w - width) // 2, (container_h - height) // 2, width, height

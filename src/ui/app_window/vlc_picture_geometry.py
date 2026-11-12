"""
Shape of the picture libvlc shows for the current video.

video_get_size() reports decoded pixels. libvlc then applies the sample
aspect ratio and the orientation tag: a phone clip stored as 1920x1080 with a
90-degree tag plays as portrait, and a 720x576 stream with SAR 16:15 plays
768 wide. Sizing to video_get_size() alone gives the wrong shape for those
files, and libvlc fills the difference with black bars.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _track_shape(track) -> VideoTrackShape | None:
    try:
        kind = getattr(track, "type", None)
        kind = getattr(kind, "value", kind)
        if kind is None or int(kind) != _VIDEO_TRACK_TYPE:
            return None
        # python-vlc builds differ: the union member is either on the track
        # or under ``u``, as a struct or a pointer to one.
        video = getattr(track, "video", None)
        if video is None:
            video = getattr(getattr(track, "u", None), "video", None)
        video = getattr(video, "contents", video)
        if video is None:
            return None
        return VideoTrackShape(
            width=int(getattr(video, "width", 0) or 0),
            height=int(getattr(video, "height", 0) or 0),
            sar_num=int(getattr(video, "sar_num", 0) or 0),
            sar_den=int(getattr(video, "sar_den", 0) or 0),
            orientation=int(getattr(video, "orientation", 0) or 0),
        )
    except Exception:
        return None


def read_video_track_shape(player) -> VideoTrackShape | None:
    """Shape fields of the playing video track, or None before libvlc lists it.

    python-vlc's Media.tracks_get() does not release the array libvlc returns,
    so callers should cache the result per media rather than poll it.
    """
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
        tracks = list(media.tracks_get() or ())
    except Exception:
        return None
    first = None
    for track in tracks:
        shape = _track_shape(track)
        if shape is None:
            continue
        if selected is not None and getattr(track, "id", None) == selected:
            return shape
        if first is None:
            first = shape
    return first


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

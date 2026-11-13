"""Displayed shape of a libvlc picture: SAR, orientation tag, fitting."""

import ctypes
from types import SimpleNamespace

import ui.app_window.vlc_picture_geometry as geometry
from ui.app_window.vlc_picture_geometry import (
    VideoTrackShape,
    fit_rect,
    picture_from,
    read_picture,
    read_video_track_shape,
    shapes_from_track_array,
)

_RIGHT_TOP = 6  # libvlc_video_orient_right_top: a 90-degree phone clip


def _shape(width=1920, height=1080, *, sar=(1, 1), orientation=0):
    return VideoTrackShape(width, height, sar[0], sar[1], orientation)


def _player(monkeypatch, tracks, *, size=(1920, 1080), selected=0):
    """Player whose media lists ``tracks``: (track id, VideoTrackShape) pairs."""
    media = object()
    monkeypatch.setattr(
        geometry, "list_video_tracks", lambda m: list(tracks) if m is media else []
    )
    return SimpleNamespace(
        get_media=lambda: media,
        video_get_track=lambda: selected,
        video_get_size=lambda _num: size,
    )


def _track_array(*tracks):
    """A libvlc track array: (type, id, bitrate, video struct or None) per entry."""
    keep = []
    entries = (ctypes.POINTER(geometry._LibvlcMediaTrack) * len(tracks))()
    for index, (kind, track_id, bitrate, video) in enumerate(tracks):
        details = ctypes.addressof(video) if video is not None else None
        track = geometry._LibvlcMediaTrack(
            type=kind, id=track_id, bitrate=bitrate, details=details
        )
        keep.extend([track, video])
        entries[index] = ctypes.pointer(track)
    array = ctypes.cast(entries, geometry._TrackArray)
    return array, (entries, keep)


def _video_struct(width, height, *, sar=(1, 1), orientation=0):
    return geometry._LibvlcVideoTrack(
        height=height, width=width, sar_num=sar[0], sar_den=sar[1], orientation=orientation
    )


class TestPictureFrom:
    def test_square_pixels_keep_the_decoded_shape(self):
        picture = picture_from(1920, 1080, VideoTrackShape(1920, 1080, 1, 1, 0))
        assert (picture.display_w, picture.display_h) == (1920, 1080)
        assert (picture.frame_w, picture.frame_h) == (1920, 1080)

    def test_no_track_uses_the_decoded_shape(self):
        picture = picture_from(640, 480, None)
        assert (picture.display_w, picture.display_h) == (640, 480)

    def test_sample_aspect_ratio_widens_the_display_only(self):
        picture = picture_from(720, 576, VideoTrackShape(720, 576, 16, 15, 0))
        assert (picture.display_w, picture.display_h) == (768, 576)
        assert (picture.frame_w, picture.frame_h) == (720, 576)

    def test_a_quarter_turn_tag_makes_a_landscape_frame_portrait(self):
        picture = picture_from(1920, 1080, VideoTrackShape(1920, 1080, 1, 1, _RIGHT_TOP))
        assert (picture.display_w, picture.display_h) == (1080, 1920)
        assert (picture.frame_w, picture.frame_h) == (1080, 1920)

    def test_a_size_already_reported_turned_is_not_turned_again(self):
        picture = picture_from(1080, 1920, VideoTrackShape(1920, 1080, 1, 1, _RIGHT_TOP))
        assert (picture.display_w, picture.display_h) == (1080, 1920)
        assert (picture.frame_w, picture.frame_h) == (1080, 1920)

    def test_sar_applies_to_the_stored_axis_before_turning(self):
        picture = picture_from(720, 576, VideoTrackShape(720, 576, 16, 15, _RIGHT_TOP))
        assert (picture.display_w, picture.display_h) == (576, 768)
        assert (picture.frame_w, picture.frame_h) == (576, 720)

    def test_a_half_turn_keeps_the_shape(self):
        bottom_right = 3
        picture = picture_from(1920, 1080, VideoTrackShape(1920, 1080, 1, 1, bottom_right))
        assert (picture.display_w, picture.display_h) == (1920, 1080)


class TestReadTrack:
    def test_reads_the_selected_video_track(self, monkeypatch):
        tracks = [(0, _shape(640, 480)), (5, _shape(orientation=_RIGHT_TOP))]
        shape = read_video_track_shape(_player(monkeypatch, tracks, selected=5))
        assert shape == VideoTrackShape(1920, 1080, 1, 1, _RIGHT_TOP)

    def test_falls_back_to_the_first_video_track(self, monkeypatch):
        tracks = [(2, _shape(320, 240)), (3, _shape(640, 480))]
        shape = read_video_track_shape(_player(monkeypatch, tracks, selected=-1))
        assert shape == VideoTrackShape(320, 240, 1, 1, 0)

    def test_no_media_gives_none(self):
        player = SimpleNamespace(get_media=lambda: None)
        assert read_video_track_shape(player) is None

    def test_a_media_without_a_libvlc_handle_is_not_passed_to_libvlc(self):
        player = SimpleNamespace(get_media=lambda: object(), video_get_track=lambda: 0)
        assert read_video_track_shape(player) is None

    def test_no_size_gives_no_picture(self, monkeypatch):
        assert read_picture(_player(monkeypatch, [], size=(0, 0)), None) is None


class TestFitRect:
    def test_portrait_picture_is_pillarboxed(self):
        assert fit_rect(1000, 500, 1080, 1920) == (359, 0, 281, 500)

    def test_landscape_picture_is_letterboxed(self):
        assert fit_rect(500, 500, 1920, 1080) == (0, 109, 500, 281)


class TestTrackArray:
    """libvlc_media_track_t keeps audio/video/subtitle in one union pointer."""

    def test_the_union_is_one_pointer_before_the_bitrate(self):
        track = geometry._LibvlcMediaTrack
        assert track.bitrate.offset == track.details.offset + ctypes.sizeof(ctypes.c_void_p)

    def test_reads_video_fields_through_the_union(self):
        video = _video_struct(720, 576, sar=(16, 15), orientation=_RIGHT_TOP)
        array, _keep = _track_array((1, 4, 5_000_000, video))
        assert shapes_from_track_array(array, 1) == [
            (4, VideoTrackShape(720, 576, 16, 15, _RIGHT_TOP))
        ]

    def test_skips_other_track_types_and_missing_details(self):
        audio_kind = 0
        video = _video_struct(640, 480)
        array, _keep = _track_array(
            (audio_kind, 0, 128_000, _video_struct(1, 1)),
            (1, 1, 0, None),
            (1, 2, 0, video),
        )
        assert shapes_from_track_array(array, 3) == [(2, VideoTrackShape(640, 480, 1, 1, 0))]

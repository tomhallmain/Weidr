"""Displayed shape of a libvlc picture: SAR, orientation tag, fitting."""

from types import SimpleNamespace

from ui.app_window.vlc_picture_geometry import (
    VideoTrackShape,
    fit_rect,
    picture_from,
    read_picture,
    read_video_track_shape,
)

_RIGHT_TOP = 6  # libvlc_video_orient_right_top: a 90-degree phone clip


def _track(track_id=0, *, width=1920, height=1080, sar=(1, 1), orientation=0, kind=1, nested=True):
    video = SimpleNamespace(
        width=width, height=height, sar_num=sar[0], sar_den=sar[1], orientation=orientation
    )
    if nested:
        return SimpleNamespace(type=kind, id=track_id, u=SimpleNamespace(video=SimpleNamespace(contents=video)))
    return SimpleNamespace(type=kind, id=track_id, video=video)


def _player(tracks, *, size=(1920, 1080), selected=0):
    media = SimpleNamespace(tracks_get=lambda: iter(tracks))
    return SimpleNamespace(
        get_media=lambda: media,
        video_get_track=lambda: selected,
        video_get_size=lambda _num: size,
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
    def test_reads_the_selected_video_track(self):
        tracks = [
            _track(0, width=640, height=480),
            _track(5, width=1920, height=1080, orientation=_RIGHT_TOP),
        ]
        shape = read_video_track_shape(_player(tracks, selected=5))
        assert shape == VideoTrackShape(1920, 1080, 1, 1, _RIGHT_TOP)

    def test_falls_back_to_the_first_video_track(self):
        tracks = [_track(0, kind=0), _track(2, width=320, height=240)]
        shape = read_video_track_shape(_player(tracks, selected=-1))
        assert shape == VideoTrackShape(320, 240, 1, 1, 0)

    def test_accepts_the_union_member_on_the_track_itself(self):
        shape = read_video_track_shape(_player([_track(0, sar=(4, 3), nested=False)]))
        assert shape == VideoTrackShape(1920, 1080, 4, 3, 0)

    def test_no_media_gives_none(self):
        player = SimpleNamespace(get_media=lambda: None)
        assert read_video_track_shape(player) is None

    def test_no_size_gives_no_picture(self):
        assert read_picture(_player([], size=(0, 0)), None) is None


class TestFitRect:
    def test_portrait_picture_is_pillarboxed(self):
        assert fit_rect(1000, 500, 1080, 1920) == (359, 0, 281, 500)

    def test_landscape_picture_is_letterboxed(self):
        assert fit_rect(500, 500, 1920, 1080) == (0, 109, 500, 281)

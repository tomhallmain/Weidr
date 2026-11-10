"""Video follows the media frame onto another display.

Images are painted by Qt and move with the window. libvlc's Direct3D swap
chain does not, so a window dragged from a 1080p monitor onto a 4K HDMI TV
keeps presenting on the display where playback started. VlcVideoEmbed owns
the surface that fixes that, including the rebuild when the display changes.
"""

import platform

import pytest

from ui.app_window.media_frame import VideoUI, _VLC_AVAILABLE

pytestmark = pytest.mark.skipif(not _VLC_AVAILABLE, reason="python-vlc not installed")


class _Media:
    def release(self):
        return None


def _stub_player(player, calls):
    player.stop = lambda: calls.append("stop")
    player.get_time = lambda: 1500
    player.get_rate = lambda: 1.0
    player.is_playing = lambda: True
    player.set_media = lambda _media: calls.append("media")
    player.play = lambda: calls.append("play") or 0
    player.set_time = lambda ms: calls.append(("time", ms))
    player.set_rate = lambda _rate: None
    player.set_pause = lambda _paused: None
    player.set_hwnd = lambda hwnd: calls.append(("hwnd", int(hwnd)))


class TestDisplayChange:
    def test_no_video_does_not_schedule_a_rebuild(self, media_frame, monkeypatch):
        embed = media_frame._video_embed
        embed._screen = object()
        monkeypatch.setattr(embed, "_screen_of_frame", lambda: object())

        embed.note_screen()

        assert embed._timer.isActive() is False

    def test_the_same_display_does_not_schedule_a_rebuild(self, media_frame, monkeypatch):
        embed = media_frame._video_embed
        screen = object()
        embed._screen = screen
        embed.bound_hwnd = 1
        media_frame._video_ui = VideoUI("clip.mp4", has_video=True)
        monkeypatch.setattr(embed, "_screen_of_frame", lambda: screen)

        embed.note_screen()

        assert embed._timer.isActive() is False

    def test_the_first_display_is_remembered_without_a_rebuild(self, media_frame, monkeypatch):
        embed = media_frame._video_embed
        screen = object()
        monkeypatch.setattr(embed, "_screen_of_frame", lambda: screen)

        embed.note_screen()

        assert embed._screen is screen
        assert embed._timer.isActive() is False

    @pytest.mark.skipif(platform.system() != "Windows", reason="HWND rebuild is Windows-only")
    def test_a_display_change_schedules_one_rebuild(self, media_frame, monkeypatch):
        embed = media_frame._video_embed
        embed.bound_hwnd = 1
        media_frame._video_ui = VideoUI("clip.mp4", has_video=True)
        second = object()
        embed._screen = object()
        monkeypatch.setattr(embed, "_screen_of_frame", lambda: second)

        embed.note_screen()

        assert embed._timer.isActive() is True
        embed._timer.stop()
        embed.note_screen()
        assert embed._timer.isActive() is False

    @pytest.mark.skipif(platform.system() != "Windows", reason="HWND rebuild is Windows-only")
    def test_crossing_displays_reopens_playback_on_a_new_window(self, media_frame, monkeypatch):
        from PySide6.QtCore import QPoint, Qt

        calls = []
        _stub_player(media_frame.vlc_media_player, calls)
        media_frame.vlc_instance.media_new = lambda _path: _Media()
        media_frame._video_ui = VideoUI("clip.mp4", has_video=True)
        media_frame.path = "clip.mp4"
        media_frame._video_embed.bound_hwnd = 1

        media_frame._video_embed._rebind()

        assert calls[0] == "stop"
        hwnds = [item[1] for item in calls if isinstance(item, tuple) and item[0] == "hwnd"]
        assert hwnds and hwnds[-1] != 0
        host = media_frame._video_embed.host
        assert host is not None
        assert host.isWindow() is True
        assert host.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) is False
        assert host.winId() == hwnds[-1]
        top_left = media_frame.mapToGlobal(QPoint(0, 0))
        assert host.geometry().topLeft() == top_left
        assert host.width() == media_frame.width()
        assert ("time", 1500) in calls

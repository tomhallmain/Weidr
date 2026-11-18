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

    @pytest.mark.skipif(platform.system() != "Windows", reason="HWND rebuild is Windows-only")
    def test_a_portrait_video_covers_only_the_picture(self, media_frame):
        from PySide6.QtCore import QPoint

        media_frame.vlc_media_player.video_get_size = lambda _track: (100, 200)
        media_frame._video_embed.prepare_hwnd()

        host = media_frame._video_embed.host
        frame_w = media_frame.width()
        frame_h = media_frame.height()
        scale = min(frame_w / 100, frame_h / 200)
        picture_w = max(1, int(100 * scale))
        picture_h = max(1, int(200 * scale))
        origin = media_frame.mapToGlobal(
            QPoint((frame_w - picture_w) // 2, (frame_h - picture_h) // 2)
        )
        assert host.geometry().topLeft() == origin
        assert host.width() == picture_w
        assert host.height() == picture_h
        assert picture_w < frame_w or picture_h < frame_h


class TestRevealAfterFirstPicture:
    def test_the_host_stays_hidden_until_vlc_has_a_video_output(self, media_frame):
        player = media_frame.vlc_media_player
        player.has_vout = lambda: 0
        player.video_get_size = lambda _track: (0, 0)
        embed = media_frame._video_embed

        embed.bound_hwnd = embed.prepare_hwnd()
        embed._reveal_when_ready()

        assert embed.host.isHidden() is True
        assert embed._reveal_timer.isActive() is True

        player.has_vout = lambda: 1
        player.video_get_size = lambda _track: (100, 200)
        embed._reveal_when_ready()

        assert embed.host.isHidden() is False
        assert embed._reveal_timer.isActive() is False
        assert embed.host.height() <= media_frame.height()

    def test_stopping_playback_cancels_a_pending_reveal(self, media_frame):
        media_frame.vlc_media_player.has_vout = lambda: 0
        embed = media_frame._video_embed
        embed.bound_hwnd = embed.prepare_hwnd()

        media_frame.detach_vlc_output()
        embed._reveal_when_ready()

        assert embed.host.isHidden() is True
        assert embed._reveal_timer.isActive() is False
        assert embed._awaiting_picture is False


def _list_one_video_track(monkeypatch, width, height, *, sar=(1, 1), orientation=0):
    """Make every media list one video track of this shape."""
    import ui.app_window.vlc_picture_geometry as geometry
    from ui.app_window.vlc_picture_geometry import VideoTrackShape

    shape = VideoTrackShape(width, height, sar[0], sar[1], orientation)
    monkeypatch.setattr(geometry, "list_video_tracks", lambda _media: [(0, shape)])


class TestPictureShape:
    def test_a_rotated_phone_clip_gets_a_portrait_host(self, media_frame, monkeypatch):
        player = media_frame.vlc_media_player
        player.video_get_size = lambda _track: (1920, 1080)
        player.get_media = lambda: object()
        _list_one_video_track(monkeypatch, 1920, 1080, orientation=6)
        player.video_get_track = lambda: 0
        media_frame.resize(640, 480)

        embed = media_frame._video_embed
        embed.prepare_hwnd()

        host = embed.host
        assert host.height() > host.width()
        assert host.height() == media_frame.height()

    def test_a_new_file_reads_its_own_track(self, media_frame, monkeypatch):
        player = media_frame.vlc_media_player
        player.video_get_size = lambda _track: (1920, 1080)
        player.get_media = lambda: object()
        _list_one_video_track(monkeypatch, 1920, 1080, orientation=6)
        player.video_get_track = lambda: 0
        embed = media_frame._video_embed
        embed.prepare_hwnd()

        embed.note_new_media()
        _list_one_video_track(monkeypatch, 1920, 1080)
        embed.prepare_hwnd()

        assert embed.host.width() > embed.host.height()


class TestHostKind:
    @pytest.mark.skipif(platform.system() == "Windows", reason="Windows uses a tool window")
    def test_other_platforms_draw_into_a_child_of_the_frame(self, media_frame):
        from PySide6.QtCore import Qt

        embed = media_frame._video_embed
        wid = embed.prepare_hwnd()

        host = embed.host
        assert host.isWindow() is False
        assert host.parentWidget() is media_frame
        assert host.testAttribute(Qt.WidgetAttribute.WA_NativeWindow) is True
        assert wid == int(host.winId())
        assert media_frame.get_media_frame_handle() == wid


def _revealed_host(media_frame):
    player = media_frame.vlc_media_player
    player.has_vout = lambda: 1
    player.video_get_size = lambda _track: (1920, 1080)
    embed = media_frame._video_embed
    embed.bound_hwnd = embed.prepare_hwnd()
    embed._reveal_when_ready()
    return embed


class TestPlacement:
    def test_the_host_is_positioned_before_it_is_shown(self, media_frame, monkeypatch):
        player = media_frame.vlc_media_player
        player.has_vout = lambda: 1
        player.video_get_size = lambda _track: (1920, 1080)
        embed = media_frame._video_embed
        embed.bound_hwnd = embed.prepare_hwnd()
        hidden_at_placement = []
        original = embed._apply_geometry

        def _record(host, rect):
            hidden_at_placement.append(host.isHidden())
            original(host, rect)

        monkeypatch.setattr(embed, "_apply_geometry", _record)
        embed._reveal_when_ready()

        assert hidden_at_placement and hidden_at_placement[0] is True
        assert embed.host.isHidden() is False

    def test_a_scale_change_alone_places_the_host_again(self, media_frame, monkeypatch):
        embed = _revealed_host(media_frame)
        calls = []
        monkeypatch.setattr(embed, "_apply_geometry", lambda host, rect: calls.append(rect))

        embed.sync_geometry()
        assert calls == []

        monkeypatch.setattr(media_frame, "devicePixelRatio", lambda: 2.5)
        embed.sync_geometry()
        assert len(calls) == 1


class _EventRecorder:
    def __init__(self, widget):
        from PySide6.QtCore import QObject

        recorder = self

        class _Filter(QObject):
            def eventFilter(self, _watched, event):
                recorder.types.append(event.type())
                return False

        self.types = []
        self._filter = _Filter()
        widget.installEventFilter(self._filter)


class TestInputOverTheVideo:
    @pytest.mark.skipif(platform.system() != "Windows", reason="Only the tool-window host forwards")
    def test_wheel_context_menu_and_middle_click_reach_the_frame(self, media_frame):
        from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
        from PySide6.QtGui import QContextMenuEvent, QMouseEvent, QWheelEvent
        from PySide6.QtWidgets import QApplication

        host = _revealed_host(media_frame).host
        recorder = _EventRecorder(media_frame)
        centre = QPointF(host.width() / 2, host.height() / 2)
        global_centre = QPointF(host.mapToGlobal(centre.toPoint()))

        QApplication.sendEvent(host, QWheelEvent(
            centre, global_centre, QPoint(), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        ))
        QApplication.sendEvent(host, QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, centre.toPoint(), global_centre.toPoint(),
            Qt.KeyboardModifier.NoModifier,
        ))
        QApplication.sendEvent(host, QMouseEvent(
            QEvent.Type.MouseButtonPress, centre, global_centre,
            Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton,
            Qt.KeyboardModifier.NoModifier,
        ))

        assert QEvent.Type.Wheel in recorder.types
        assert QEvent.Type.ContextMenu in recorder.types
        assert QEvent.Type.MouseButtonPress in recorder.types


class TestWaylandWarning:
    def _patch(self, monkeypatch, system, platform_name):
        from types import SimpleNamespace

        import ui.app_window.vlc_video_embed as embed_mod

        monkeypatch.setattr(embed_mod, "_wayland_warning_logged", False)
        monkeypatch.setattr(embed_mod.platform, "system", lambda: system)
        app = SimpleNamespace(platformName=lambda: platform_name)
        monkeypatch.setattr(embed_mod, "QApplication", SimpleNamespace(instance=lambda: app))
        return embed_mod

    def test_wayland_is_warned_about_once(self, monkeypatch):
        embed_mod = self._patch(monkeypatch, "Linux", "wayland")
        assert embed_mod.warn_if_wayland() is True
        assert embed_mod.warn_if_wayland() is False

    def test_xwayland_and_other_systems_are_not_warned_about(self, monkeypatch):
        embed_mod = self._patch(monkeypatch, "Linux", "xcb")
        assert embed_mod.warn_if_wayland() is False
        embed_mod = self._patch(monkeypatch, "Darwin", "wayland")
        assert embed_mod.warn_if_wayland() is False


class TestTeardown:
    def test_the_drawable_is_reset_before_release_on_every_platform(self, media_frame):
        calls = []
        player = media_frame.vlc_media_player
        for name in ("set_hwnd", "set_nsobject", "set_xwindow"):
            setattr(player, name, lambda wid, name=name: calls.append((name, int(wid or 0))))

        media_frame.dispose_vlc_for_test_teardown()

        expected = {"Windows": "set_hwnd", "Darwin": "set_nsobject"}.get(
            platform.system(), "set_xwindow"
        )
        assert calls == [(expected, 0)]


class TestMacosReveal:
    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS GL view refresh")
    def test_showing_the_host_resizes_it_once_and_back(self, media_frame, monkeypatch):
        embed = media_frame._video_embed
        player = media_frame.vlc_media_player
        player.has_vout = lambda: 1
        player.video_get_size = lambda _track: (1920, 1080)
        embed.bound_hwnd = embed.prepare_hwnd()
        host = embed.host
        sizes = []
        original = host.setGeometry
        monkeypatch.setattr(host, "setGeometry", lambda rect: (sizes.append(rect.width()), original(rect)))

        embed._reveal_when_ready()

        assert len(sizes) >= 3
        assert sizes[-2] == sizes[-1] + 1
        assert host.isHidden() is False

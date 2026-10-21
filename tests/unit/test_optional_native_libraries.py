"""Missing Cairo and VLC libraries limit features without stopping the app."""

import image.frame_cache as frame_cache
from ui.app_window.media_frame import video_playback_unavailable_message
from ui.files.type_configuration_window_qt import svg_dependency_info
from utils.translations import _


def test_cairo_library_notice_names_the_install(monkeypatch):
    monkeypatch.setattr(frame_cache, "has_imported_cairosvg", False)
    monkeypatch.setattr(frame_cache, "cairosvg_unavailable_reason", "library")

    message = frame_cache.svg_conversion_unavailable_message()
    hint = frame_cache.svg_conversion_install_hint()
    notice = frame_cache.svg_conversion_unavailable_notice()

    assert message == _("SVG conversion is unavailable because the Cairo library was not found.")
    assert hint == _("Run python scripts/install_native_extras.py from the repository, then restart.")
    assert notice == f"{message} {hint}"

    info = svg_dependency_info()
    assert info["available"] is False
    assert info["description"] == message
    assert info["detail"] == hint


def test_missing_cairosvg_package_points_at_pip(monkeypatch):
    monkeypatch.setattr(frame_cache, "has_imported_cairosvg", False)
    monkeypatch.setattr(frame_cache, "cairosvg_unavailable_reason", "package")

    message = frame_cache.svg_conversion_unavailable_message()
    hint = frame_cache.svg_conversion_install_hint()

    assert message == _("SVG conversion is unavailable because cairosvg is not installed.")
    assert hint == _("pip install cairosvg")
    assert frame_cache.svg_conversion_unavailable_notice() == f"{message} ({hint})"

    info = svg_dependency_info()
    assert info["available"] is False
    assert "detail" not in info


def test_svg_notice_is_absent_when_cairosvg_loaded(monkeypatch):
    monkeypatch.setattr(frame_cache, "has_imported_cairosvg", True)
    monkeypatch.setattr(frame_cache, "cairosvg_unavailable_reason", None)

    assert frame_cache.svg_conversion_unavailable_message() is None
    assert frame_cache.svg_conversion_install_hint() is None
    assert frame_cache.svg_conversion_unavailable_notice() is None
    assert svg_dependency_info()["available"] is True


def test_video_notice_follows_vlc_availability(monkeypatch):
    import ui.app_window.media_frame as media_frame

    monkeypatch.setattr(media_frame, "_VLC_AVAILABLE", False)
    assert video_playback_unavailable_message() == _(
        "Video playback is unavailable because VLC was not found."
    )

    monkeypatch.setattr(media_frame, "_VLC_AVAILABLE", True)
    assert video_playback_unavailable_message() is None

"""Animated raster extension policy (GIF/WebP/APNG) — no Qt/VLC required."""

from ui.app_window.media_frame import ANIMATED_IMAGE_SUFFIXES


def test_animated_image_suffixes_include_webp_and_apng():
    assert ".webp" in ANIMATED_IMAGE_SUFFIXES
    assert ".apng" in ANIMATED_IMAGE_SUFFIXES
    assert ".gif" in ANIMATED_IMAGE_SUFFIXES
    assert ".png" not in ANIMATED_IMAGE_SUFFIXES


def test_animated_candidate_path_matching():
    def is_candidate(path: str) -> bool:
        if not path:
            return False
        return path.lower().endswith(ANIMATED_IMAGE_SUFFIXES)

    assert is_candidate("/dir/frame.webp")
    assert is_candidate("/dir/anim.apng")
    assert is_candidate("/dir/x.gif")
    assert not is_candidate("/dir/static.png")
    assert not is_candidate("")

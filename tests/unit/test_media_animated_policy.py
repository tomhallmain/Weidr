"""Animated raster extension policy (GIF/WebP/APNG) — no Qt/VLC required."""

from utils.media_utils import ANIMATED_IMAGE_SUFFIXES, is_animated_image_candidate


def test_animated_image_suffixes_include_webp_and_apng():
    assert ".webp" in ANIMATED_IMAGE_SUFFIXES
    assert ".apng" in ANIMATED_IMAGE_SUFFIXES
    assert ".gif" in ANIMATED_IMAGE_SUFFIXES
    assert ".png" not in ANIMATED_IMAGE_SUFFIXES


def test_animated_candidate_path_matching():
    assert is_animated_image_candidate("/dir/frame.webp")
    assert is_animated_image_candidate("/dir/anim.apng")
    assert is_animated_image_candidate("/dir/x.gif")
    assert not is_animated_image_candidate("/dir/static.png")
    assert not is_animated_image_candidate("")

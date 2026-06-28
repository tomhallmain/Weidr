"""
Unit tests for ImageOps texture backgrounds and partial rotation.

Covers logic exercised by temp_test_texture_rotation.py (manual script): noise
textures, tiled cache, probability-based texture vs solid fill, and file output.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

pytest.importorskip("cv2")

from image.image_ops import ImageOps


def _sample_bgr_image(height: int = 80, width: int = 120) -> np.ndarray:
    """Small BGR test image with colored regions (similar to the manual script)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 40), (255, 0, 0), -1)
    cv2.rectangle(img, (70, 10), (110, 40), (0, 255, 0), -1)
    cv2.rectangle(img, (10, 50), (60, 70), (0, 0, 255), -1)
    return img


@pytest.fixture(autouse=True)
def _reset_texture_caches():
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()
    yield
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()


@pytest.mark.parametrize("texture_type", ImageOps.TEXTURE_DRAW_TYPES)
def test_generate_noise_texture_returns_bgr_uint8(texture_type):
    texture = ImageOps.generate_noise_texture(64, 48, texture_type)
    assert texture.shape == (48, 64, 3)
    assert texture.dtype == np.uint8


def test_generate_noise_texture_tiles_cached_smaller_than_tile_size():
    w, h = 100, 80
    texture = ImageOps.generate_noise_texture(w, h, "perlin", background_color=(10, 20, 30))
    assert texture.shape == (h, w, 3)
    assert len(ImageOps._texture_tile_cache) == 1


def test_get_random_texture_type_is_known():
    for _ in range(20):
        assert ImageOps.get_random_texture_type() in ImageOps.TEXTURE_DRAW_TYPES


def test_rotate_image_partial_preserves_shape_texture_and_solid(tmp_path, monkeypatch):
    src = tmp_path / "sample.png"
    original = _sample_bgr_image()
    cv2.imwrite(str(src), original)

    # Without pinning, get_random_color() has a 25% chance of returning black,
    # which can make both paths produce near-identical all-dark arrays.
    monkeypatch.setattr(
        ImageOps, "get_random_color",
        staticmethod(lambda true_random_chance=0.75: (200, 100, 50)),
    )

    textured = ImageOps._rotate_image_partial(original.copy(), angle=45, use_texture=True)
    solid = ImageOps._rotate_image_partial(original.copy(), angle=45, use_texture=False)

    assert textured.shape == original.shape
    assert solid.shape == original.shape
    assert not np.array_equal(textured, solid)


def test_rotate_image_partial_probability_zero_uses_solid_path(tmp_path, monkeypatch):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), _sample_bgr_image())
    calls: list[bool] = []

    def _capture_rotate(image, angle=90, center=None, scale=1.0, use_texture=True):
        calls.append(use_texture)
        return image

    monkeypatch.setattr(ImageOps, "_rotate_image_partial", staticmethod(_capture_rotate))
    monkeypatch.setattr("image.image_ops.random.random", lambda: 0.99)

    ImageOps.rotate_image_partial(str(src), angle=30, texture_probability=0.0)

    assert calls == [False]


def test_rotate_image_partial_probability_one_uses_texture_path(tmp_path, monkeypatch):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), _sample_bgr_image())
    calls: list[bool] = []

    def _capture_rotate(image, angle=90, center=None, scale=1.0, use_texture=True):
        calls.append(use_texture)
        return image

    monkeypatch.setattr(ImageOps, "_rotate_image_partial", staticmethod(_capture_rotate))
    monkeypatch.setattr("image.image_ops.random.random", lambda: 0.01)

    ImageOps.rotate_image_partial(str(src), angle=30, texture_probability=1.0)

    assert calls == [True]


def test_rotate_image_partial_writes_rot_file(tmp_path, monkeypatch):
    src = tmp_path / "photo.png"
    cv2.imwrite(str(src), _sample_bgr_image())
    monkeypatch.setattr("image.image_ops.random.random", lambda: 1.0)

    out = ImageOps.rotate_image_partial(str(src), angle=30, texture_probability=0.0)

    assert out == ImageOps.new_filepath(str(src), append_part="_rot")
    assert os.path.isfile(out)
    loaded = cv2.imread(out)
    assert loaded is not None
    assert loaded.shape[:2] == (80, 120)


def test_texture_cache_reuses_tile_for_same_type_and_color():
    bg = (40, 50, 60)
    ImageOps.generate_noise_texture(200, 200, "gaussian", background_color=bg)
    key = ("gaussian", bg)
    assert key in ImageOps._texture_tile_cache
    _, _, use_count_after_first = ImageOps._texture_tile_cache[key]

    ImageOps.generate_noise_texture(300, 250, "gaussian", background_color=bg)
    _, _, use_count_after_second = ImageOps._texture_tile_cache[key]
    assert use_count_after_second == use_count_after_first + 1

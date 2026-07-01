"""Unit tests for ImageOps.draw_box_at_rect and ImageOps.generate_box_fill_image."""

import os
import random
import pytest
from PIL import Image

from image.image_ops import ImageOps


@pytest.fixture(autouse=True)
def _reset_texture_caches():
    """generate_box_fill_image reuses the texture-tile cache and used-color list
    that ImageOps keeps at class level; isolate them per test like
    test_image_ops_texture_rotation.py does, so runs don't leak state."""
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()
    yield
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()


class TestGenerateBoxFillImage:
    def test_solid_fill_has_requested_size(self):
        img = ImageOps.generate_box_fill_image(30, 20, use_texture=False)
        assert img.size == (30, 20)

    def test_solid_fill_is_single_color(self):
        img = ImageOps.generate_box_fill_image(10, 10, use_texture=False)
        colors = img.convert("RGB").getcolors()
        assert len(colors) == 1

    def test_texture_fill_has_requested_size(self):
        img = ImageOps.generate_box_fill_image(40, 25, use_texture=True)
        assert img.size == (40, 25)

    def test_default_gate_forces_texture_branch(self, monkeypatch):
        # use_texture=None defers to config.image_edit_configuration.texture_draw_probability
        # via random.random(); pin it below the probability to force the texture branch.
        monkeypatch.setattr(random, "random", lambda: 0.0)
        img = ImageOps.generate_box_fill_image(12, 8)
        assert img.size == (12, 8)

    def test_default_gate_forces_solid_branch(self, monkeypatch):
        # Pin random.random() above any configured probability to force solid fill.
        monkeypatch.setattr(random, "random", lambda: 0.999999)
        img = ImageOps.generate_box_fill_image(12, 8)
        assert img.size == (12, 8)
        colors = img.convert("RGB").getcolors()
        assert len(colors) == 1


class TestDrawBoxOutputPath:
    def test_output_is_sibling_with_box_suffix(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        result = ImageOps.draw_box_at_rect(src, 0, 0, 50, 50)
        assert result == str(tmp_path / "photo_box.png")
        assert os.path.exists(result)

    def test_output_preserves_extension(self, tmp_path):
        src = str(tmp_path / "image.jpg")
        Image.new("RGB", (100, 100)).save(src, format="JPEG")
        result = ImageOps.draw_box_at_rect(src, 0, 0, 50, 50)
        assert result.endswith("_box.jpg")
        assert os.path.exists(result)

    def test_source_file_untouched(self, tmp_path):
        src = tmp_path / "photo.png"
        Image.new("RGB", (100, 100), color=(1, 2, 3)).save(src)
        original_bytes = src.read_bytes()
        ImageOps.draw_box_at_rect(str(src), 0, 0, 50, 50)
        assert src.read_bytes() == original_bytes

    def test_second_box_of_same_source_does_not_overwrite_first(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        first = ImageOps.draw_box_at_rect(src, 0, 0, 50, 50)
        second = ImageOps.draw_box_at_rect(src, 0, 0, 40, 40)
        assert first != second
        assert os.path.exists(first)
        assert os.path.exists(second)


class TestDrawBoxStaticImage:
    def test_output_dimensions_match_original(self, tmp_path):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (100, 80)).save(src)
        result = ImageOps.draw_box_at_rect(src, 10, 20, 60, 70)
        with Image.open(result) as out:
            assert out.size == (100, 80)

    def test_solid_box_fills_exact_region_with_one_color(self, tmp_path, monkeypatch):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (50, 50), color=(0, 0, 0)).save(src)
        monkeypatch.setattr(
            ImageOps, "generate_box_fill_image",
            staticmethod(lambda w, h, use_texture=None: Image.new("RGB", (w, h), (200, 100, 50))),
        )
        result = ImageOps.draw_box_at_rect(src, 10, 10, 30, 30)
        with Image.open(result) as out:
            assert out.getpixel((15, 15)) == (200, 100, 50)
            # Outside the box, original background is preserved.
            assert out.getpixel((1, 1)) == (0, 0, 0)

    def test_bad_path_returns_empty_string(self):
        result = ImageOps.draw_box_at_rect("/nonexistent/no.png", 0, 0, 10, 10)
        assert result == ""


class TestDrawBoxAnimatedGif:
    def _make_gif(self, path: str, n_frames: int, size=(50, 50)) -> None:
        frames = [Image.new("RGB", size, color=(i * 30, i * 30, i * 30)) for i in range(n_frames)]
        frames[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

    def test_animated_gif_frame_count_preserved(self, tmp_path):
        src = str(tmp_path / "anim.gif")
        self._make_gif(src, n_frames=5)
        result = ImageOps.draw_box_at_rect(src, 5, 5, 30, 30)
        assert result.endswith("_box.gif")
        with Image.open(result) as out:
            assert getattr(out, "n_frames", 1) == 5

    def test_animated_gif_size_unchanged(self, tmp_path):
        src = str(tmp_path / "anim.gif")
        self._make_gif(src, n_frames=3, size=(60, 60))
        result = ImageOps.draw_box_at_rect(src, 0, 0, 40, 40)
        with Image.open(result) as out:
            assert out.size == (60, 60)

    def test_single_frame_gif_handled_as_static(self, tmp_path):
        src = str(tmp_path / "single.gif")
        Image.new("P", (50, 50), color=0).save(src, format="GIF")
        result = ImageOps.draw_box_at_rect(src, 5, 5, 25, 25)
        with Image.open(result) as out:
            assert out.size == (50, 50)

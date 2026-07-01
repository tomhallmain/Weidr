"""Unit tests for ImageOps.draw_box_at_polygon and ImageOps.draw_background_box_at_polygon
(the freeform counterparts to draw_box_at_rect / draw_background_box_at_rect)."""

import os
import pytest
from PIL import Image

from image.image_ops import ImageOps


@pytest.fixture(autouse=True)
def _reset_texture_caches():
    """These reuse generate_box_fill_image, which touches the texture-tile
    cache and used-color list ImageOps keeps at class level; isolate them per
    test like test_image_ops_texture_rotation.py does."""
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()
    yield
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()


_TRIANGLE = [(10, 10), (40, 10), (25, 40)]


class TestPolygonMask:
    def test_mask_is_l_mode_matching_size(self):
        mask = ImageOps._polygon_mask((50, 50), _TRIANGLE)
        assert mask.mode == "L"
        assert mask.size == (50, 50)

    def test_inside_polygon_is_filled(self):
        mask = ImageOps._polygon_mask((50, 50), _TRIANGLE)
        assert mask.getpixel((25, 20)) == 255

    def test_outside_polygon_is_empty(self):
        mask = ImageOps._polygon_mask((50, 50), _TRIANGLE)
        assert mask.getpixel((1, 1)) == 0


class TestDrawBoxAtPolygonOutputPath:
    def test_output_is_sibling_with_box_suffix(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        result = ImageOps.draw_box_at_polygon(src, _TRIANGLE)
        assert result == str(tmp_path / "photo_box.png")
        assert os.path.exists(result)

    def test_source_file_untouched(self, tmp_path):
        src = tmp_path / "photo.png"
        Image.new("RGB", (100, 100), color=(1, 2, 3)).save(src)
        original_bytes = src.read_bytes()
        ImageOps.draw_box_at_polygon(str(src), _TRIANGLE)
        assert src.read_bytes() == original_bytes

    def test_does_not_collide_with_rect_box_output(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        rect_out = ImageOps.draw_box_at_rect(src, 0, 0, 50, 50)
        poly_out = ImageOps.draw_box_at_polygon(src, _TRIANGLE)
        assert rect_out != poly_out

    def test_bad_path_returns_empty_string(self):
        result = ImageOps.draw_box_at_polygon("/nonexistent/no.png", _TRIANGLE)
        assert result == ""


class TestDrawBoxAtPolygonStaticImage:
    def test_output_dimensions_match_original(self, tmp_path):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (100, 80)).save(src)
        result = ImageOps.draw_box_at_polygon(src, _TRIANGLE)
        with Image.open(result) as out:
            assert out.size == (100, 80)

    def test_fills_inside_and_preserves_outside_polygon(self, tmp_path, monkeypatch):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (50, 50), color=(0, 0, 0)).save(src)
        monkeypatch.setattr(
            ImageOps, "generate_box_fill_image",
            staticmethod(lambda w, h, use_texture=None: Image.new("RGB", (w, h), (200, 100, 50))),
        )
        result = ImageOps.draw_box_at_polygon(src, _TRIANGLE)
        with Image.open(result) as out:
            # A point well inside the triangle gets the fill color.
            assert out.getpixel((25, 20)) == (200, 100, 50)
            # A point outside the triangle keeps the original background.
            assert out.getpixel((1, 1)) == (0, 0, 0)


class TestDrawBoxAtPolygonAnimatedGif:
    def _make_gif(self, path: str, n_frames: int, size=(50, 50)) -> None:
        frames = [Image.new("RGB", size, color=(i * 30, i * 30, i * 30)) for i in range(n_frames)]
        frames[0].save(
            path, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0,
        )

    def test_animated_gif_frame_count_preserved(self, tmp_path):
        src = str(tmp_path / "anim.gif")
        self._make_gif(src, n_frames=5)
        result = ImageOps.draw_box_at_polygon(src, _TRIANGLE)
        assert result.endswith("_box.gif")
        with Image.open(result) as out:
            assert getattr(out, "n_frames", 1) == 5


class TestDrawBackgroundBoxAtPolygonOutputPath:
    def test_output_is_sibling_with_bgbox_suffix(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        result = ImageOps.draw_background_box_at_polygon(src, _TRIANGLE)
        assert result == str(tmp_path / "photo_bgbox.png")
        assert os.path.exists(result)

    def test_source_file_untouched(self, tmp_path):
        src = tmp_path / "photo.png"
        Image.new("RGB", (100, 100), color=(1, 2, 3)).save(src)
        original_bytes = src.read_bytes()
        ImageOps.draw_background_box_at_polygon(str(src), _TRIANGLE)
        assert src.read_bytes() == original_bytes

    def test_bad_path_returns_empty_string(self):
        result = ImageOps.draw_background_box_at_polygon("/nonexistent/no.png", _TRIANGLE)
        assert result == ""


class TestDrawBackgroundBoxAtPolygonStaticImage:
    def test_output_dimensions_match_original(self, tmp_path):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (100, 80)).save(src)
        result = ImageOps.draw_background_box_at_polygon(src, _TRIANGLE)
        with Image.open(result) as out:
            assert out.size == (100, 80)

    def test_fills_outside_and_preserves_inside_polygon(self, tmp_path, monkeypatch):
        """Inverse of draw_box_at_polygon: the polygon interior keeps the
        original content, everything outside it is replaced by the fill."""
        src = str(tmp_path / "img.png")
        Image.new("RGB", (50, 50), color=(0, 0, 0)).save(src)
        monkeypatch.setattr(
            ImageOps, "generate_box_fill_image",
            staticmethod(lambda w, h, use_texture=None: Image.new("RGB", (w, h), (200, 100, 50))),
        )
        result = ImageOps.draw_background_box_at_polygon(src, _TRIANGLE)
        with Image.open(result) as out:
            # Inside the triangle, original content (black) is preserved.
            assert out.getpixel((25, 20)) == (0, 0, 0)
            # Outside the triangle, the fill color is applied.
            assert out.getpixel((1, 1)) == (200, 100, 50)
            assert out.getpixel((45, 45)) == (200, 100, 50)


class TestDrawBackgroundBoxAtPolygonAnimatedGif:
    def _make_gif(self, path: str, n_frames: int, size=(50, 50)) -> None:
        frames = [Image.new("RGB", size, color=(i * 30, i * 30, i * 30)) for i in range(n_frames)]
        frames[0].save(
            path, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0,
        )

    def test_animated_gif_frame_count_preserved(self, tmp_path):
        src = str(tmp_path / "anim.gif")
        self._make_gif(src, n_frames=5)
        result = ImageOps.draw_background_box_at_polygon(src, _TRIANGLE)
        assert result.endswith("_bgbox.gif")
        with Image.open(result) as out:
            assert getattr(out, "n_frames", 1) == 5

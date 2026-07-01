"""Unit tests for ImageOps.draw_background_box_at_rect."""

import os
import pytest
from PIL import Image

from image.image_ops import ImageOps


@pytest.fixture(autouse=True)
def _reset_texture_caches():
    """draw_background_box_at_rect reuses generate_box_fill_image, which touches
    the texture-tile cache and used-color list ImageOps keeps at class level;
    isolate them per test like test_image_ops_texture_rotation.py does."""
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()
    yield
    ImageOps._texture_tile_cache.clear()
    ImageOps._used_random_colors.clear()


class TestDrawBackgroundBoxOutputPath:
    def test_output_is_sibling_with_bgbox_suffix(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        result = ImageOps.draw_background_box_at_rect(src, 0, 0, 50, 50)
        assert result == str(tmp_path / "photo_bgbox.png")
        assert os.path.exists(result)

    def test_output_preserves_extension(self, tmp_path):
        src = str(tmp_path / "image.jpg")
        Image.new("RGB", (100, 100)).save(src, format="JPEG")
        result = ImageOps.draw_background_box_at_rect(src, 0, 0, 50, 50)
        assert result.endswith("_bgbox.jpg")
        assert os.path.exists(result)

    def test_source_file_untouched(self, tmp_path):
        src = tmp_path / "photo.png"
        Image.new("RGB", (100, 100), color=(1, 2, 3)).save(src)
        original_bytes = src.read_bytes()
        ImageOps.draw_background_box_at_rect(str(src), 0, 0, 50, 50)
        assert src.read_bytes() == original_bytes

    def test_second_call_on_same_source_does_not_overwrite_first(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        first = ImageOps.draw_background_box_at_rect(src, 0, 0, 50, 50)
        second = ImageOps.draw_background_box_at_rect(src, 0, 0, 40, 40)
        assert first != second
        assert os.path.exists(first)
        assert os.path.exists(second)

    def test_does_not_collide_with_plain_box_output(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        box_out = ImageOps.draw_box_at_rect(src, 0, 0, 50, 50)
        bgbox_out = ImageOps.draw_background_box_at_rect(src, 0, 0, 50, 50)
        assert box_out != bgbox_out


class TestDrawBackgroundBoxStaticImage:
    def test_output_dimensions_match_original(self, tmp_path):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (100, 80)).save(src)
        result = ImageOps.draw_background_box_at_rect(src, 10, 20, 60, 70)
        with Image.open(result) as out:
            assert out.size == (100, 80)

    def test_fills_outside_and_preserves_inside_selection(self, tmp_path, monkeypatch):
        """Inverse of draw_box_at_rect: the selected rectangle keeps the
        original content, everything outside it is replaced by the fill."""
        src = str(tmp_path / "img.png")
        Image.new("RGB", (50, 50), color=(0, 0, 0)).save(src)
        monkeypatch.setattr(
            ImageOps, "generate_box_fill_image",
            staticmethod(lambda w, h, use_texture=None: Image.new("RGB", (w, h), (200, 100, 50))),
        )
        result = ImageOps.draw_background_box_at_rect(src, 10, 10, 30, 30)
        with Image.open(result) as out:
            # Inside the selection, original content (black) is preserved.
            assert out.getpixel((15, 15)) == (0, 0, 0)
            # Outside the selection, the fill color is applied.
            assert out.getpixel((1, 1)) == (200, 100, 50)
            assert out.getpixel((45, 45)) == (200, 100, 50)

    def test_bad_path_returns_empty_string(self):
        result = ImageOps.draw_background_box_at_rect("/nonexistent/no.png", 0, 0, 10, 10)
        assert result == ""


class TestDrawBackgroundBoxAnimatedGif:
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
        result = ImageOps.draw_background_box_at_rect(src, 5, 5, 30, 30)
        assert result.endswith("_bgbox.gif")
        with Image.open(result) as out:
            assert getattr(out, "n_frames", 1) == 5

    def test_animated_gif_size_unchanged(self, tmp_path):
        src = str(tmp_path / "anim.gif")
        self._make_gif(src, n_frames=3, size=(60, 60))
        result = ImageOps.draw_background_box_at_rect(src, 0, 0, 40, 40)
        with Image.open(result) as out:
            assert out.size == (60, 60)

    def test_single_frame_gif_handled_as_static(self, tmp_path):
        src = str(tmp_path / "single.gif")
        Image.new("P", (50, 50), color=0).save(src, format="GIF")
        result = ImageOps.draw_background_box_at_rect(src, 5, 5, 25, 25)
        with Image.open(result) as out:
            assert out.size == (50, 50)

"""Unit tests for ImageOps.crop_image_to_rect."""

import os
import pytest
from PIL import Image

from image.image_ops import ImageOps


class TestCropImageOutputPath:
    def test_output_is_sibling_with_crop_suffix(self, tmp_path):
        src = str(tmp_path / "photo.png")
        Image.new("RGB", (100, 100)).save(src)
        result = ImageOps.crop_image_to_rect(src, 0, 0, 50, 50)
        assert result == str(tmp_path / "photo_crop.png")
        assert os.path.exists(result)

    def test_output_preserves_extension(self, tmp_path):
        src = str(tmp_path / "image.jpg")
        Image.new("RGB", (100, 100)).save(src, format="JPEG")
        result = ImageOps.crop_image_to_rect(src, 0, 0, 50, 50)
        assert result.endswith("_crop.jpg")
        assert os.path.exists(result)


class TestCropStaticImage:
    def test_cropped_dimensions_match(self, tmp_path):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (100, 80)).save(src)
        result = ImageOps.crop_image_to_rect(src, 10, 20, 60, 70)
        with Image.open(result) as out:
            assert out.size == (50, 50)

    def test_full_crop_matches_original_size(self, tmp_path):
        src = str(tmp_path / "img.png")
        Image.new("RGB", (30, 40), color=(255, 0, 0)).save(src)
        result = ImageOps.crop_image_to_rect(src, 0, 0, 30, 40)
        with Image.open(result) as out:
            assert out.size == (30, 40)

    def test_pixel_color_preserved(self, tmp_path):
        src = str(tmp_path / "colored.png")
        img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        img.putpixel((5, 5), (123, 45, 67))
        img.save(src)
        result = ImageOps.crop_image_to_rect(src, 3, 3, 8, 8)
        with Image.open(result) as out:
            # pixel (5,5) in original → (2,2) in crop (offset by 3,3)
            assert out.getpixel((2, 2)) == (123, 45, 67)

    def test_bad_path_returns_empty_string(self):
        result = ImageOps.crop_image_to_rect("/nonexistent/no.png", 0, 0, 10, 10)
        assert result == ""


class TestCropAnimatedGif:
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
        result = ImageOps.crop_image_to_rect(src, 5, 5, 30, 30)
        assert result.endswith("_crop.gif")
        with Image.open(result) as out:
            assert getattr(out, "n_frames", 1) == 5

    def test_animated_gif_frame_size_correct(self, tmp_path):
        src = str(tmp_path / "anim.gif")
        self._make_gif(src, n_frames=3, size=(60, 60))
        result = ImageOps.crop_image_to_rect(src, 0, 0, 40, 40)
        with Image.open(result) as out:
            assert out.size == (40, 40)

    def test_single_frame_gif_handled_as_static(self, tmp_path):
        src = str(tmp_path / "single.gif")
        Image.new("P", (50, 50), color=0).save(src, format="GIF")
        result = ImageOps.crop_image_to_rect(src, 5, 5, 25, 25)
        with Image.open(result) as out:
            assert out.size == (20, 20)

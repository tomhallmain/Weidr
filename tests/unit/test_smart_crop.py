"""Unit tests for image/smart_crop.py.

All fixtures are generated programmatically — no large image files are
committed to the repository.  Images are kept small so the Python-level
pixel-diff loops in detect_perfectly_vertical/horizontal_divisions finish
in a reasonable time during CI.
"""

import numpy as np
import pytest
from PIL import Image

from image.smart_crop import Cropper, smart_consolidate_diffs, validate_division


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _solid(width: int, height: int, color, mode: str = "RGB") -> Image.Image:
    """Solid-colour image in the requested mode."""
    return Image.new(mode, (width, height), color)


def _bordered(interior_color, border_color=(255, 255, 255),
              border=10, size=80, mode="RGB") -> Image.Image:
    """Solid interior with a uniform border of *border* pixels on every side."""
    im = Image.new(mode, (size, size), border_color)
    inner = Image.new(mode, (size - 2 * border, size - 2 * border), interior_color)
    im.paste(inner, (border, border))
    return im


def _no_border_image(width=60, height=60) -> Image.Image:
    """Red image with one blue pixel on each of the four edges (centre of each side).

    This prevents remove_borders from treating any edge row/column as uniformly
    coloured, so the function finds no border to strip.
    """
    im = Image.new("RGB", (width, height), (200, 0, 0))
    cx, cy = width // 2, height // 2
    for x, y in [(cx, 0), (cx, height - 1), (0, cy), (width - 1, cy)]:
        im.putpixel((x, y), (0, 0, 200))
    return im


def _noise_panel(width: int, height: int, seed: int = 42) -> Image.Image:
    """Fully random RGB panel — entropy >> 5, guaranteed to survive is_low_entropy."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _vertical_division_solid(panel_w=40, height=60,
                              left_color=(200, 0, 0),
                              divider_color=(0, 0, 0),
                              right_color=(0, 0, 200),
                              divider_px=2) -> Image.Image:
    """Two solid-colour panels separated by a solid divider column.

    Used by validate_division tests (which only need a sharp contrast at the
    division position, not high entropy).
    """
    width = panel_w * 2 + divider_px
    im = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            if x < panel_w:
                im.putpixel((x, y), left_color)
            elif x < panel_w + divider_px:
                im.putpixel((x, y), divider_color)
            else:
                im.putpixel((x, y), right_color)
    return im


def _horizontal_division_solid(width=60, panel_h=40,
                                top_color=(200, 0, 0),
                                divider_color=(0, 0, 0),
                                bottom_color=(0, 0, 200),
                                divider_px=2) -> Image.Image:
    """Two solid-colour panels separated by a solid divider row."""
    height = panel_h * 2 + divider_px
    im = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            if y < panel_h:
                im.putpixel((x, y), top_color)
            elif y < panel_h + divider_px:
                im.putpixel((x, y), divider_color)
            else:
                im.putpixel((x, y), bottom_color)
    return im


def _vertical_division_noisy(panel_w=50, height=50, divider_px=2) -> Image.Image:
    """Two random-noise panels separated by a black divider column.

    Entropy is well above 5 so panels survive is_low_entropy; the black divider
    contrasts enough that validate_division accepts the split position.
    """
    width = panel_w * 2 + divider_px
    rng = np.random.RandomState(7)
    left = Image.fromarray(rng.randint(0, 256, (height, panel_w, 3), dtype=np.uint8), "RGB")
    right = Image.fromarray(rng.randint(0, 256, (height, panel_w, 3), dtype=np.uint8), "RGB")
    divider = Image.new("RGB", (divider_px, height), (0, 0, 0))
    im = Image.new("RGB", (width, height))
    im.paste(left, (0, 0))
    im.paste(divider, (panel_w, 0))
    im.paste(right, (panel_w + divider_px, 0))
    return im


def _horizontal_division_noisy(width=50, panel_h=50, divider_px=2) -> Image.Image:
    """Two random-noise panels separated by a black divider row."""
    height = panel_h * 2 + divider_px
    rng = np.random.RandomState(13)
    top = Image.fromarray(rng.randint(0, 256, (panel_h, width, 3), dtype=np.uint8), "RGB")
    bottom = Image.fromarray(rng.randint(0, 256, (panel_h, width, 3), dtype=np.uint8), "RGB")
    divider = Image.new("RGB", (width, divider_px), (0, 0, 0))
    im = Image.new("RGB", (width, height))
    im.paste(top, (0, 0))
    im.paste(divider, (0, panel_h))
    im.paste(bottom, (0, panel_h + divider_px))
    return im


def _grid_noisy(panel=50, divider_px=2) -> Image.Image:
    """2×2 grid of random-noise panels separated by black dividers on both axes."""
    side = panel * 2 + divider_px
    rng = np.random.RandomState(99)
    im = Image.new("RGB", (side, side), (0, 0, 0))
    offsets = [(0, 0), (panel + divider_px, 0), (0, panel + divider_px),
               (panel + divider_px, panel + divider_px)]
    for i, (ox, oy) in enumerate(offsets):
        patch = Image.fromarray(rng.randint(0, 256, (panel, panel, 3), dtype=np.uint8), "RGB")
        im.paste(patch, (ox, oy))
    return im


# ---------------------------------------------------------------------------
# is_close_color — non-RGB mode safety (#13)
# ---------------------------------------------------------------------------

class TestIsCloseColor:
    def test_rgb_tuple_matches(self):
        assert Cropper.is_close_color((100, 100, 100), (100, 100, 100))

    def test_rgb_tuple_outside_tolerance(self):
        assert not Cropper.is_close_color((100, 100, 100), (200, 100, 100))

    def test_grayscale_int_treated_as_grey(self):
        # getpixel on an 'L' image returns a plain int
        assert Cropper.is_close_color(128, (128, 128, 128))

    def test_grayscale_int_outside_tolerance(self):
        assert not Cropper.is_close_color(0, (128, 128, 128))

    def test_rgba_tuple_alpha_ignored(self):
        # Only the first three channels should be compared
        assert Cropper.is_close_color((100, 100, 100, 0), (100, 100, 100, 255))

    def test_rgba_tuple_rgb_mismatch(self):
        assert not Cropper.is_close_color((100, 100, 100, 255), (200, 100, 100, 255))


# ---------------------------------------------------------------------------
# remove_borders
# ---------------------------------------------------------------------------

class TestRemoveBorders:
    def test_white_border_stripped(self):
        im = _bordered(interior_color=(200, 0, 0), border_color=(255, 255, 255),
                       border=10, size=80)
        result, cropped = Cropper.remove_borders(im)
        assert cropped is True
        # The interior was 60×60
        assert result.size == (60, 60)
        # Corner pixel should be the interior colour, not the border colour
        assert result.getpixel((0, 0)) == (200, 0, 0)

    def test_no_border_unchanged(self):
        # Image whose four edge-rows/columns each have at least one pixel
        # differing from (0,0) — remove_borders must find nothing to strip.
        im = _no_border_image(60, 60)
        result, cropped = Cropper.remove_borders(im)
        assert cropped is False
        assert result.size == (60, 60)

    def test_grayscale_border_no_crash(self):
        # 'L' mode returns int from getpixel — is_close_color must handle it
        im = _bordered(interior_color=100, border_color=255,
                       border=8, size=60, mode="L")
        result, cropped = Cropper.remove_borders(im)
        assert cropped is True
        assert result.size == (44, 44)

    def test_rgba_border_no_crash(self):
        im = _bordered(interior_color=(200, 0, 0, 255),
                       border_color=(255, 255, 255, 255),
                       border=8, size=60, mode="RGBA")
        result, cropped = Cropper.remove_borders(im)
        assert cropped is True


# ---------------------------------------------------------------------------
# smart_consolidate_diffs — transitive grouping fix (#9)
# ---------------------------------------------------------------------------

class TestSmartConsolidateDiffs:
    def test_three_within_gap_collapse_to_one(self):
        # Positions 10, 15, 20 are all within min_gap=20 of each other.
        # Old code let all three survive; fixed code keeps only the strongest.
        diffs = {10: 0.5, 15: 0.9, 20: 0.4}
        result = smart_consolidate_diffs(diffs, image_size=200, min_gap=20)
        assert len(result) == 1
        assert 15 in result

    def test_well_separated_positions_both_survive(self):
        diffs = {10: 0.5, 100: 0.8}
        result = smart_consolidate_diffs(diffs, image_size=200, min_gap=20)
        assert len(result) == 2

    def test_empty_input(self):
        assert smart_consolidate_diffs({}, image_size=100) == {}


# ---------------------------------------------------------------------------
# validate_division (#10)
# ---------------------------------------------------------------------------

class TestValidateDivision:
    def test_too_close_to_top_edge_rejected(self):
        im = _vertical_division_solid()
        assert not validate_division(im, 5, is_horizontal=True)

    def test_too_close_to_left_edge_rejected(self):
        im = _horizontal_division_solid()
        assert not validate_division(im, 5, is_horizontal=False)

    def test_clear_vertical_division_accepted(self):
        # Division at the black column between two strongly contrasting panels
        im = _vertical_division_solid(panel_w=40, height=60)
        division_col = 40  # first column of the divider
        assert validate_division(im, division_col, is_horizontal=False)

    def test_clear_horizontal_division_accepted(self):
        im = _horizontal_division_solid(width=60, panel_h=40)
        division_row = 40
        assert validate_division(im, division_row, is_horizontal=True)

    def test_uniform_column_rejected(self):
        # Solid image — adjacent columns are identical, coverage will be 0
        im = _solid(100, 80, (128, 128, 128))
        assert not validate_division(im, 50, is_horizontal=False)

    def test_uniform_row_rejected(self):
        im = _solid(100, 80, (128, 128, 128))
        assert not validate_division(im, 40, is_horizontal=True)


# ---------------------------------------------------------------------------
# split_image
# ---------------------------------------------------------------------------

class TestSplitImage:
    def test_no_divisions_returns_original(self):
        im = _solid(80, 60, (200, 0, 0))
        result = Cropper.split_image(im, {}, {})
        assert len(result) == 1
        assert result[0] is im

    def test_vertical_split_produces_two_panels(self):
        # Noisy panels so entropy check passes; divider at column 50 and 52
        im = _vertical_division_noisy(panel_w=50, height=50)
        result = Cropper.split_image(im, {}, {50: 1.0, 52: 1.0})
        # Divider strip (2px) removed as too small; two main panels survive
        assert len(result) >= 2
        for piece in result:
            assert piece.size[0] >= 30

    def test_horizontal_split_produces_two_panels(self):
        im = _horizontal_division_noisy(width=50, panel_h=50)
        result = Cropper.split_image(im, {50: 1.0, 52: 1.0}, {})
        assert len(result) >= 2

    def test_grid_split(self):
        im = _grid_noisy(panel=50, divider_px=2)
        # Dividers at x=50,52 and y=50,52
        result = Cropper.split_image(im, {50: 1.0, 52: 1.0}, {50: 1.0, 52: 1.0})
        # All four noisy panels should survive entropy and size checks
        assert len(result) >= 4

    def test_excessive_grid_returns_original(self):
        # 16 positions on each axis → (16-1)*(16-1)=225 cells > 200 limit
        # range(0, 160, 10) = [0,10,...,150] and image width/height=100 is
        # already in that list, so no extra boundary is appended.
        im = _solid(100, 100, (200, 0, 0))
        v = {x: 1.0 for x in range(0, 160, 10)}
        h = {y: 1.0 for y in range(0, 160, 10)}
        result = Cropper.split_image(im, h, v)
        assert len(result) == 1
        assert result[0] is im

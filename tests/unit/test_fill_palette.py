"""FillPalette: fill colours drawn from a source image's chromatic profile.

Sampling is random by design, so the assertions are about the distribution the
draws fall in (hue near the source's, saturation absent for a black-and-white
source) rather than about any single colour.
"""

import colorsys
import random

import pytest
from PIL import Image

from image.fill_palette import FillPalette
from image.image_ops import ImageOps
from utils.config import config


def _solid(path, color, size=(64, 64)) -> str:
    Image.new("RGB", size, color).save(str(path), format="PNG")
    return str(path)


def _greyscale(path, size=(64, 64)) -> str:
    image = Image.new("L", size)
    image.putdata([(x * 4) % 256 for x in range(size[0] * size[1])])
    image.convert("RGB").save(str(path), format="PNG")
    return str(path)


def _hue_of(color) -> float:
    r, g, b = (c / 255.0 for c in color)
    return colorsys.rgb_to_hsv(r, g, b)[0]


def _saturation_of(color) -> float:
    r, g, b = (c / 255.0 for c in color)
    return colorsys.rgb_to_hsv(r, g, b)[1]


def _hue_distance(a: float, b: float) -> float:
    """Shortest distance around the colour wheel, in turns."""
    raw = abs(a - b) % 1.0
    return min(raw, 1.0 - raw)


@pytest.fixture(autouse=True)
def _match_on(monkeypatch):
    """Matching is opt-in; these tests are about what it does when on."""
    monkeypatch.setattr(
        config.image_edit_configuration, "fill_palette_match_enabled", True
    )
    monkeypatch.setattr(config.image_edit_configuration, "fill_palette_mode", "sampled")
    monkeypatch.setattr(config.image_edit_configuration, "fill_palette_strength", 1.0)
    random.seed(20260912)


class TestAchromaticSources:
    def test_greyscale_source_is_detected(self, tmp_path):
        palette = FillPalette.for_image(_greyscale(tmp_path / "grey.png"))
        assert palette.is_achromatic() is True

    def test_greyscale_source_yields_only_greys(self, tmp_path):
        palette = FillPalette.for_image(_greyscale(tmp_path / "grey.png"))

        for _ in range(40):
            r, g, b = palette.color()
            assert r == g == b

    def test_saturated_source_is_not_achromatic(self, tmp_path):
        palette = FillPalette.for_image(_solid(tmp_path / "red.png", (220, 20, 20)))
        assert palette.is_achromatic() is False


class TestSampling:
    def test_sampled_hues_stay_near_the_source_hue(self, tmp_path):
        source_color = (30, 160, 220)  # a blue
        palette = FillPalette.for_image(_solid(tmp_path / "blue.png", source_color))
        source_hue = _hue_of(source_color)

        drawn = [_hue_of(palette.color()) for _ in range(40)]

        # One 10-degree histogram bin either side covers the sampling spread.
        assert all(_hue_distance(hue, source_hue) <= 2 / 36 for hue in drawn)

    def test_complementary_mode_lands_opposite_the_source_hue(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            config.image_edit_configuration, "fill_palette_mode", "complementary"
        )
        source_color = (30, 160, 220)
        palette = FillPalette.for_image(_solid(tmp_path / "blue.png", source_color))
        opposite = (_hue_of(source_color) + 0.5) % 1.0

        drawn = [_hue_of(palette.color()) for _ in range(40)]

        # The bin spread plus the mode's own 15-degree jitter.
        assert all(_hue_distance(hue, opposite) <= 2 / 36 + 15 / 360 for hue in drawn)

    def test_colours_are_quantized_so_the_texture_cache_key_space_stays_bounded(self, tmp_path):
        palette = FillPalette.for_image(_solid(tmp_path / "blue.png", (30, 160, 220)))

        for _ in range(40):
            for channel in palette.color():
                # 5 bits: bucket centres sit on a 8-step grid.
                assert channel % 8 == 4


class TestSessionOverride:
    def test_config_is_followed_when_nothing_overrides_it(self, monkeypatch):
        monkeypatch.setattr(
            config.image_edit_configuration, "fill_palette_match_enabled", False
        )
        assert FillPalette.match_enabled() is False

    def test_session_value_wins_over_config(self, monkeypatch):
        monkeypatch.setattr(
            config.image_edit_configuration, "fill_palette_match_enabled", False
        )
        FillPalette.set_session_match_enabled(True)
        assert FillPalette.match_enabled() is True

        FillPalette.reset_session_match_enabled()
        assert FillPalette.match_enabled() is False

    def test_no_palette_is_built_while_matching_is_off(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            config.image_edit_configuration, "fill_palette_match_enabled", False
        )
        source = _solid(tmp_path / "blue.png", (30, 160, 220))

        assert FillPalette.for_image_if_enabled(source) is None

    def test_an_unreadable_source_falls_back_to_random(self, tmp_path):
        assert FillPalette.for_image_if_enabled(str(tmp_path / "missing.png")) is None
        assert FillPalette.for_image_if_enabled(None) is None


class TestStrength:
    def test_zero_strength_reproduces_the_unmatched_distribution(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config.image_edit_configuration, "fill_palette_strength", 0.0)
        palette = FillPalette.for_image(_solid(tmp_path / "blue.png", (30, 160, 220)))
        monkeypatch.setattr(ImageOps, "get_random_color", lambda *a, **kw: (255, 0, 0))

        # Fully diluted: every draw is the random colour, quantized.
        assert {palette.color() for _ in range(20)} == {(252, 4, 4)}


class TestUnmatchedCallSites:
    def test_box_fill_without_a_palette_is_unchanged(self, tmp_path):
        fill = ImageOps.generate_box_fill_image(8, 8, use_texture=False)
        assert fill.size == (8, 8)
        assert fill.mode == "RGB"

    def test_box_fill_with_a_palette_uses_palette_colours(self, tmp_path):
        palette = FillPalette.for_image(_greyscale(tmp_path / "grey.png"))

        fill = ImageOps.generate_box_fill_image(8, 8, use_texture=False, palette=palette)

        r, g, b = fill.getpixel((0, 0))
        assert r == g == b  # the achromatic source's greys, not a random hue

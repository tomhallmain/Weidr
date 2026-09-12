"""Sampling fill colours from an image's own chromatic profile.

The random fills -- a box or background-box fill, and the shapes and textures
the random image modification draws -- pick their colours with no reference to
the image they land in, so a muted photograph can be given a saturated magenta
box, and a black-and-white image a coloured one.

A FillPalette samples from the source image's own HSV histogram instead. It is
opt-in (``image_edit_configuration.fill_palette_match_enabled``, overridable
for the session from the preview dialog); with it off every caller passes None
and keeps the original random behaviour.
"""

from __future__ import annotations

import colorsys
import random
from typing import Optional, Sequence, Tuple

from utils.config import config
from utils.logging_setup import get_logger

logger = get_logger("fill_palette")

# ImageOps.color_histogram's layout: hue bins, then saturation, then value.
_H_BINS, _S_BINS, _V_BINS = 36, 16, 16

# A pixel below this HSV saturation reads as achromatic -- the threshold
# ImageOps.hue_breadth_score applies when deciding which pixels carry a hue.
_MIN_CHROMATIC_SAT = 0.08
# Below this share of chromatic pixels the source counts as black and white,
# and the palette returns greys only.
_ACHROMATIC_PIXEL_SHARE = 0.05

_COMPLEMENTARY_JITTER = 15.0 / 360.0
_ANALOGOUS_MIN, _ANALOGOUS_MAX = 15.0 / 360.0, 45.0 / 360.0

MODE_SAMPLED = "sampled"
MODE_COMPLEMENTARY = "complementary"
MODE_ANALOGOUS = "analogous"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quantize(color: Tuple[int, int, int], bits: int) -> Tuple[int, int, int]:
    """Round each channel to *bits* of precision.

    ImageOps' texture tile cache is keyed by (texture_type, background_color)
    and only drops a key when that key is looked up again, so colours sampled
    per image would grow it without bound. A tile's colour is blended in at
    30%, so the rounding is not visible in the result.
    """
    if bits <= 0 or bits >= 8:
        return tuple(color)
    step = 1 << (8 - bits)
    return tuple(min(255, (c // step) * step + step // 2) for c in color)


class FillPalette:
    """Colours drawn from one image's chromatic profile."""

    # Session override for fill_palette_match_enabled; None follows config.
    # Process-wide on purpose -- turning matching off in the preview dialog
    # means the session, not one window -- and never persisted, so the next
    # start follows the configured value again.
    _session_match_enabled: Optional[bool] = None

    def __init__(
        self,
        hue_weights: Sequence[float],
        sat_weights: Sequence[float],
        val_weights: Sequence[float],
        achromatic: bool,
    ) -> None:
        self._hue = list(hue_weights)
        self._sat = list(sat_weights)
        self._val = list(val_weights)
        # bool(): the histogram is a numpy array, so a comparison over it
        # yields numpy.bool_, which is truthy but is not True.
        self._achromatic = bool(achromatic)

    # ------------------------------------------------------------------
    # Session switch
    # ------------------------------------------------------------------
    @classmethod
    def match_enabled(cls) -> bool:
        """Whether fills should match the source image right now."""
        if cls._session_match_enabled is not None:
            return cls._session_match_enabled
        return bool(config.image_edit_configuration.fill_palette_match_enabled)

    @classmethod
    def set_session_match_enabled(cls, enabled: bool) -> None:
        cls._session_match_enabled = bool(enabled)

    @classmethod
    def reset_session_match_enabled(cls) -> None:
        """Follow the configured value again."""
        cls._session_match_enabled = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def for_image(cls, image_path: str) -> "FillPalette":
        """Read *image_path*'s chromatic profile. One histogram, one decode."""
        # Lazy: ImageOps imports this module.
        from image.image_ops import ImageOps

        hist = ImageOps.color_histogram(
            image_path, h_bins=_H_BINS, s_bins=_S_BINS, v_bins=_V_BINS
        )
        hue = list(hist[:_H_BINS])
        sat = list(hist[_H_BINS:_H_BINS + _S_BINS])
        val = list(hist[_H_BINS + _S_BINS:])

        # The histogram is the only read of the image, so the achromatic test
        # works on saturation bins whose centre clears the per-pixel threshold
        # hue_breadth_score uses, rather than decoding a second time.
        first_chromatic = next(
            (i for i in range(_S_BINS) if (i + 0.5) / _S_BINS > _MIN_CHROMATIC_SAT),
            _S_BINS,
        )
        chromatic_share = sum(sat[first_chromatic:])
        return cls(hue, sat, val, achromatic=chromatic_share < _ACHROMATIC_PIXEL_SHARE)

    @classmethod
    def for_image_if_enabled(cls, image_path: Optional[str]) -> Optional["FillPalette"]:
        """A palette for *image_path*, or None when matching is off.

        None is also the answer when the image cannot be read: callers treat it
        as "draw at random", which is the behaviour without this feature.
        """
        if not image_path or not cls.match_enabled():
            return None
        try:
            return cls.for_image(image_path)
        except Exception:
            logger.exception("Could not read a fill palette from %s", image_path)
            return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def is_achromatic(self) -> bool:
        """True when the source carries almost no hue (black and white)."""
        return self._achromatic

    def color(self) -> Tuple[int, int, int]:
        """One colour from the palette. Random within the source's profile."""
        if self._achromatic:
            # No hue, whatever the mode: a grey from the source's own tonal
            # range. fill_palette_strength deliberately does not dilute this --
            # blending toward a random colour is what it exists to avoid.
            level = int(round(self._draw(self._val, _V_BINS) * 255))
            picked = (level, level, level)
        else:
            hue = self._apply_mode(self._draw(self._hue, _H_BINS))
            sat = self._draw(self._sat, _S_BINS)
            val = self._draw(self._val, _V_BINS)
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            picked = (
                int(round(r * 255)), int(round(g * 255)), int(round(b * 255)),
            )
            picked = self._blend_with_random(picked)
        return _quantize(
            picked, int(config.image_edit_configuration.fill_palette_color_quantize_bits)
        )

    @staticmethod
    def _draw(weights: Sequence[float], bins: int) -> float:
        """A value in [0, 1), drawn by bin weight then uniformly inside it.

        Weighted rather than "the dominant colour" so repeated rerolls stay
        varied while staying inside the image's palette.
        """
        total = sum(weights)
        if total <= 0:
            return random.random()
        index = random.choices(range(bins), weights=weights, k=1)[0]
        return (index + random.random()) / bins

    @staticmethod
    def _apply_mode(hue: float) -> float:
        mode = str(config.image_edit_configuration.fill_palette_mode or MODE_SAMPLED).lower()
        if mode == MODE_COMPLEMENTARY:
            jitter = random.uniform(-_COMPLEMENTARY_JITTER, _COMPLEMENTARY_JITTER)
            return (hue + 0.5 + jitter) % 1.0
        if mode == MODE_ANALOGOUS:
            offset = random.uniform(_ANALOGOUS_MIN, _ANALOGOUS_MAX)
            return (hue + (offset if random.random() < 0.5 else -offset)) % 1.0
        return hue

    @staticmethod
    def _blend_with_random(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Move *color* toward a fully random one by 1 - fill_palette_strength."""
        from image.image_ops import ImageOps

        strength = _clamp(
            float(config.image_edit_configuration.fill_palette_strength), 0.0, 1.0
        )
        if strength >= 1.0:
            return color
        random_color = ImageOps.get_random_color()
        return tuple(
            int(round(rand * (1.0 - strength) + picked * strength))
            for rand, picked in zip(random_color, color)
        )

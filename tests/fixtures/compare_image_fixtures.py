"""
Synthetic solid-color PNG sets for CompareColors pipeline integration tests.

Each fixture returns a dict with keys:
  "dir"      — absolute path to the temp directory
  "red"      — list of 5 red-family PNG paths
  "blue"     — list of 5 blue-family PNG paths
  "green"    — list of 5 green-family PNG paths
  "outliers" — list of 3 outlier PNG paths (gray/gray/yellow)

All images are 48×48 solid-color PNGs.  Within each color family the hue is
nearly identical (only the primary channel varies by ±10), so the LAB ΔE76
between siblings is << 15.  Across families the ΔE76 is >> 60, making the
COLOR_MATCHING algorithm cleanly separate them at the default threshold of 15.
"""

import os

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Color definitions
# ---------------------------------------------------------------------------

_RED_COLORS = [
    (220, 0, 0),
    (230, 0, 0),
    (215, 0, 0),
    (225, 0, 0),
    (210, 0, 0),
]

_BLUE_COLORS = [
    (0, 0, 220),
    (0, 0, 230),
    (0, 0, 215),
    (0, 0, 225),
    (0, 0, 210),
]

_GREEN_COLORS = [
    (0, 220, 0),
    (0, 230, 0),
    (0, 215, 0),
    (0, 225, 0),
    (0, 210, 0),
]

# Three images unlikely to be similar to any of the above families:
#   dark gray, light gray, saturated yellow
_OUTLIER_COLORS = [
    (40, 40, 40),
    (180, 180, 180),
    (220, 220, 0),
]


def _write_png(directory: str, name: str, color: tuple) -> str:
    path = os.path.join(directory, name)
    Image.new("RGB", (48, 48), color).save(path, format="PNG")
    return path


@pytest.fixture
def compare_colors_dir(tmp_path):
    """
    Return a catalog of 18 solid-color PNGs spread across three families.

    Structure::

        {
          "dir":      "/tmp/.../",
          "red":      ["/tmp/.../red_00.png", ...],   # 5 images
          "blue":     ["/tmp/.../blue_00.png", ...],  # 5 images
          "green":    ["/tmp/.../green_00.png", ...], # 5 images
          "outliers": ["/tmp/.../outlier_00.png", ...], # 3 images
        }
    """
    d = str(tmp_path)
    catalog = {"dir": d, "red": [], "blue": [], "green": [], "outliers": []}

    for i, color in enumerate(_RED_COLORS):
        catalog["red"].append(_write_png(d, f"red_{i:02d}.png", color))

    for i, color in enumerate(_BLUE_COLORS):
        catalog["blue"].append(_write_png(d, f"blue_{i:02d}.png", color))

    for i, color in enumerate(_GREEN_COLORS):
        catalog["green"].append(_write_png(d, f"green_{i:02d}.png", color))

    for i, color in enumerate(_OUTLIER_COLORS):
        catalog["outliers"].append(_write_png(d, f"outlier_{i:02d}.png", color))

    return catalog

"""
The last interactive selection action (crop / box / background box, rect or
freeform) and its geometry, kept so it can be reapplied to another image.

Geometry is stored in pixels of the image it was drawn on, together with that
image's size. Reapplying to an image of a different size scales each
coordinate proportionally per axis; on differently framed images this can put
the selection in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, Tuple, Union

from utils.translations import _

Rect = Tuple[int, int, int, int]            # (left, upper, right, lower)
Points = Tuple[Tuple[int, int], ...]        # polygon vertices (x, y)
Size = Tuple[int, int]                      # (width, height)


class SelectionKind(Enum):
    CROP_RECT = "crop_rect"
    CROP_POLYGON = "crop_polygon"
    BOX_RECT = "box_rect"
    BOX_POLYGON = "box_polygon"
    BACKGROUND_BOX_RECT = "background_box_rect"
    BACKGROUND_BOX_POLYGON = "background_box_polygon"

    @property
    def is_polygon(self) -> bool:
        return self in (
            SelectionKind.CROP_POLYGON,
            SelectionKind.BOX_POLYGON,
            SelectionKind.BACKGROUND_BOX_POLYGON,
        )

    def get_translation(self) -> str:
        return {
            SelectionKind.CROP_RECT: _("Crop"),
            SelectionKind.CROP_POLYGON: _("Crop (Freeform)"),
            SelectionKind.BOX_RECT: _("Box"),
            SelectionKind.BOX_POLYGON: _("Box (Freeform)"),
            SelectionKind.BACKGROUND_BOX_RECT: _("Background Box"),
            SelectionKind.BACKGROUND_BOX_POLYGON: _("Background Box (Freeform)"),
        }[self]


@dataclass(frozen=True)
class StoredSelection:
    kind: SelectionKind
    geometry: Union[Rect, Points]
    source_size: Size
    source_path: str    # the file it was drawn on; shown to the user only


class LastSelection:
    """One process-wide slot, shared by every window, for this session only."""

    _stored: ClassVar[Optional[StoredSelection]] = None

    @classmethod
    def get(cls) -> Optional[StoredSelection]:
        return cls._stored

    @classmethod
    def set(cls, stored: StoredSelection) -> None:
        cls._stored = stored

    @classmethod
    def clear(cls) -> None:
        cls._stored = None


def _scale(value: int, from_extent: int, to_extent: int) -> int:
    if from_extent <= 0 or from_extent == to_extent:
        return int(value)
    return int(round(value * to_extent / from_extent))


def map_rect(rect: Rect, from_size: Size, to_size: Size) -> Optional[Rect]:
    """*rect* moved from an image of *from_size* to one of *to_size*, clamped to
    its bounds. None when nothing of it is left (no area)."""
    (fw, fh), (tw, th) = from_size, to_size
    left, upper, right, lower = rect
    left = min(max(_scale(left, fw, tw), 0), tw)
    right = min(max(_scale(right, fw, tw), 0), tw)
    upper = min(max(_scale(upper, fh, th), 0), th)
    lower = min(max(_scale(lower, fh, th), 0), th)
    if right <= left or lower <= upper:
        return None
    return left, upper, right, lower


def map_points(points: Points, from_size: Size, to_size: Size) -> Optional[List[Tuple[int, int]]]:
    """*points* moved from an image of *from_size* to one of *to_size*, clamped
    to its bounds. None when fewer than 3 distinct points remain."""
    (fw, fh), (tw, th) = from_size, to_size
    mapped = [
        (min(max(_scale(x, fw, tw), 0), tw), min(max(_scale(y, fh, th), 0), th))
        for x, y in points
    ]
    if len(set(mapped)) < 3:
        return None
    return mapped

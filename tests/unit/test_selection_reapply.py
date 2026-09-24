"""Unit tests for image/selection_reapply.py: stored selection and geometry mapping."""

from image.selection_reapply import (
    LastSelection,
    SelectionKind,
    StoredSelection,
    map_points,
    map_rect,
)
from utils.translations import _


class TestMapRect:
    def test_same_size_is_identity(self):
        assert map_rect((10, 20, 30, 40), (100, 100), (100, 100)) == (10, 20, 30, 40)

    def test_scales_each_axis(self):
        assert map_rect((10, 20, 30, 40), (100, 100), (200, 50)) == (20, 10, 60, 20)

    def test_clamped_to_bounds(self):
        assert map_rect((-5, -5, 150, 150), (100, 100), (100, 100)) == (0, 0, 100, 100)

    def test_no_area_left_is_rejected(self):
        assert map_rect((10, 10, 11, 11), (1000, 1000), (10, 10)) is None


class TestMapPoints:
    def test_same_size_is_identity(self):
        pts = ((0, 0), (50, 0), (0, 50))
        assert map_points(pts, (100, 100), (100, 100)) == list(pts)

    def test_scales_each_axis(self):
        pts = ((10, 10), (50, 10), (10, 50))
        assert map_points(pts, (100, 100), (200, 50)) == [(20, 5), (100, 5), (20, 25)]

    def test_clamped_to_bounds(self):
        pts = ((-10, -10), (500, 0), (0, 500))
        assert map_points(pts, (100, 100), (100, 100)) == [(0, 0), (100, 0), (0, 100)]

    def test_collapsed_polygon_is_rejected(self):
        pts = ((10, 10), (11, 10), (10, 11))
        assert map_points(pts, (1000, 1000), (10, 10)) is None


class TestLastSelection:
    def test_empty_by_default(self):
        assert LastSelection.get() is None

    def test_set_and_clear(self):
        stored = StoredSelection(SelectionKind.BOX_RECT, (1, 2, 3, 4), (10, 10), "/a.png")
        LastSelection.set(stored)
        assert LastSelection.get() is stored
        LastSelection.clear()
        assert LastSelection.get() is None


def test_kind_polygon_flag_and_labels():
    assert SelectionKind.BOX_POLYGON.is_polygon
    assert not SelectionKind.BOX_RECT.is_polygon
    assert SelectionKind.BACKGROUND_BOX_POLYGON.get_translation() == _("Background Box (Freeform)")

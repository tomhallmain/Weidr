"""
MediaFrame state across the transitions either side of displaying something:
an unusable path, an undecodable file, clear/release, and the fill-canvas
switch.

tests/ui/test_media_frame.py covers the VLC stop deadlock and is skipped
without VLC; nothing here needs VLC. Rendering itself is not asserted -- that
needs a real display -- only the state each transition leaves behind.
"""

import pytest

from tests.fixtures.show_media_assets import show_media_files  # noqa: F401
from utils.translations import _

pytest.importorskip("PIL")


def _wait_displayed(media_frame, path, qtbot) -> None:
    qtbot.waitUntil(
        lambda: media_frame.media_displayed and media_frame.path == path,
        timeout=8000,
    )


class TestUnusablePaths:
    def test_a_missing_path_clears_rather_than_erroring(self, media_frame, tmp_path):
        missing = str(tmp_path / "does_not_exist.png")

        media_frame.show_media(missing)

        assert media_frame.media_displayed is False
        assert media_frame._current_pixmap is None
        # The path is recorded before the existence check, so a caller asking
        # what was last requested still gets an answer.
        assert media_frame.path == missing

    @pytest.mark.parametrize("path", ["", None, "."])
    def test_an_empty_path_clears(self, media_frame, path):
        media_frame.show_media(path)

        assert media_frame.media_displayed is False
        assert media_frame._current_pixmap is None
        assert media_frame.path == "."

    def test_an_unusable_path_clears_a_displayed_image(
        self, media_frame, show_media_files, qtbot
    ):
        media_frame.show_media(show_media_files["png"])
        _wait_displayed(media_frame, show_media_files["png"], qtbot)

        media_frame.show_media("")

        assert media_frame.media_displayed is False
        assert media_frame._current_pixmap is None

    def test_an_undecodable_file_reports_its_name(self, media_frame, tmp_path):
        """Silence would leave the previous image up, which reads as success."""
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"this is not a png")

        media_frame.show_media(str(broken))

        text = media_frame._placeholder_label.text()
        assert media_frame._placeholder_label.isVisible() is True
        assert "broken.png" in text
        assert _("Unable to display this file: ") in text
        assert media_frame.media_displayed is False


class TestSurfaceSwitching:
    def test_a_still_image_takes_the_graphics_view(
        self, media_frame, show_media_files, qtbot
    ):
        media_frame.show_media(show_media_files["png"])
        _wait_displayed(media_frame, show_media_files["png"], qtbot)

        assert media_frame._graphics_view.isVisible() is True
        assert media_frame._gif_label.isVisible() is False
        assert media_frame._placeholder_label.isVisible() is False

    def test_a_placeholder_replaces_a_displayed_image(
        self, media_frame, show_media_files, qtbot, tmp_path
    ):
        media_frame.show_media(show_media_files["png"])
        _wait_displayed(media_frame, show_media_files["png"], qtbot)

        broken = tmp_path / "broken.png"
        broken.write_bytes(b"this is not a png")
        media_frame.show_media(str(broken))

        assert media_frame._placeholder_label.isVisible() is True
        assert media_frame._current_pixmap is None


class TestClearAndRelease:
    def test_clear_drops_the_image(self, media_frame, show_media_files, qtbot):
        media_frame.show_media(show_media_files["png"])
        _wait_displayed(media_frame, show_media_files["png"], qtbot)

        media_frame.clear()

        assert media_frame._current_pixmap is None
        assert media_frame.media_displayed is False
        assert media_frame._placeholder_label.text() == ""

    def test_clear_is_safe_with_nothing_displayed(self, media_frame):
        media_frame.clear()
        media_frame.clear()

        assert media_frame.media_displayed is False

    def test_release_media_drops_the_pixmap(
        self, media_frame, show_media_files, qtbot
    ):
        media_frame.show_media(show_media_files["png"])
        _wait_displayed(media_frame, show_media_files["png"], qtbot)

        media_frame.release_media()

        assert media_frame._current_pixmap is None
        assert media_frame.media_displayed is False

    def test_release_media_tears_down_an_animation(
        self, media_frame, show_media_files, qtbot
    ):
        """A live QMovie keeps the file handle open, which blocks a move or
        delete of that file on Windows."""
        path = show_media_files["gif"]
        media_frame.show_media(path)
        qtbot.waitUntil(lambda: media_frame._gif_movie is not None, timeout=8000)

        media_frame.release_media()

        assert media_frame._gif_movie is None
        assert media_frame._gif_is_animated is False
        assert media_frame.media_displayed is False

    def test_release_media_is_safe_to_repeat(self, media_frame):
        media_frame.release_media()
        media_frame.release_media()

        assert media_frame.media_displayed is False


class TestFillCanvas:
    def test_it_follows_the_constructor_argument(self, media_frame):
        assert media_frame.fill_canvas is False

    def test_toggling_it_does_not_decode_the_file_again(
        self, media_frame, show_media_files, qtbot
    ):
        """Changing the fit mode is a view transform, so the decoded pixmap
        has to survive it."""
        media_frame.show_media(show_media_files["png"])
        _wait_displayed(media_frame, show_media_files["png"], qtbot)
        pixmap = media_frame._current_pixmap

        media_frame.set_fill_canvas(True)

        assert media_frame.fill_canvas is True
        assert media_frame._current_pixmap is pixmap

    def test_it_is_safe_with_nothing_displayed(self, media_frame):
        media_frame.set_fill_canvas(True)

        assert media_frame.fill_canvas is True
        assert media_frame.media_displayed is False

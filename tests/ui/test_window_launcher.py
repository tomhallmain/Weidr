"""UI tests for WindowLauncher secondary windows."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from ui.app_window.window_launcher import WindowLauncher
from utils.constants import MediaType


class TestWindowLauncher:
    def test_window_launcher_attached(self, window):
        assert window.window_launcher is not None
        assert window.window_launcher._app is window

    def test_open_go_to_file_window(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.window_launcher.open_go_to_file_window()
        qtbot.waitUntil(
            lambda: win.window_launcher._go_to_file_window is not None
            and win.window_launcher._go_to_file_window.isVisible(),
            timeout=3000,
        )
        first = win.window_launcher._go_to_file_window
        win.window_launcher.open_go_to_file_window()
        assert win.window_launcher._go_to_file_window is first

    def test_open_recent_directory_window(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.window_launcher.open_recent_directory_window()
        qtbot.waitExposed(win, timeout=2000)


# ---------------------------------------------------------------------------
# _run_static_rect_action — SVG/PDF sibling naming regression
# ---------------------------------------------------------------------------
#
# Bug: PDF/SVG post-processing renamed apply_fn's real image output (e.g. a
# .jpg from a PDF's rendered page -- see pdf_current_page_path()) to a sibling
# using the *original* media's extension (.pdf/.svg). Since the move is a
# plain os.replace (no re-encode), this mislabeled real image bytes as a
# .pdf/.svg file, which then failed to open ("PDFium: Data format error").
# Fixed by keeping apply_fn's own output extension for the sibling.

class _FakeSignal:
    """Minimal stand-in for a Qt Signal: single-slot connect/disconnect/emit."""
    def __init__(self):
        self._slot = None

    def connect(self, slot):
        self._slot = slot

    def disconnect(self, slot=None):
        self._slot = None

    def emit(self, *args):
        if self._slot:
            self._slot(*args)


def _make_launcher_and_gv():
    launcher = WindowLauncher(MagicMock())
    gv = MagicMock()
    gv.crop_confirmed = _FakeSignal()
    gv.crop_cancelled = _FakeSignal()
    return launcher, gv


def _make_media_frame(size=100):
    media_frame = MagicMock()
    pixmap = MagicMock()
    pixmap.isNull.return_value = False
    pixmap.width.return_value = size
    pixmap.height.return_value = size
    media_frame._current_pixmap = pixmap
    media_frame.imwidth = size
    media_frame.imheight = size
    return media_frame


class TestRunStaticRectActionPdfSvgSiblingNaming:
    def test_pdf_sibling_keeps_jpeg_extension_not_pdf(self, tmp_path, monkeypatch):
        """crop_image_to_rect on a PDF's rendered (JPEG) page must end up as a
        .jpg sibling of the .pdf source, never renamed to a .pdf extension."""
        monkeypatch.setattr(
            "ui.image.media_details.MediaDetails.open_temp_media_canvas",
            lambda **kwargs: None,
        )
        media_path = str(tmp_path / "paper.pdf")
        open(media_path, "w").close()

        produced = tmp_path / "page0_crop.jpg"
        produced.write_bytes(b"fake-jpeg-bytes")

        def fake_apply_fn(source_path, left, upper, right, lower):
            return str(produced)

        launcher, gv = _make_launcher_and_gv()
        media_frame = _make_media_frame()

        launcher._run_static_rect_action(
            gv, media_frame, media_path, "/tmp/source.jpg", MediaType.PDF,
            restore_original=lambda: None,
            apply_fn=fake_apply_fn,
            output_suffix="_crop",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
        )
        rect = SimpleNamespace(x=lambda: 0, y=lambda: 0, width=lambda: 50, height=lambda: 50)
        gv.crop_confirmed.emit(rect)

        expected = tmp_path / "paper_crop.jpg"
        assert expected.exists()
        assert not produced.exists()  # moved, not copied
        assert not (tmp_path / "paper_crop.pdf").exists()

    def test_svg_sibling_keeps_png_extension_not_svg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ui.image.media_details.MediaDetails.open_temp_media_canvas",
            lambda **kwargs: None,
        )
        media_path = str(tmp_path / "icon.svg")
        open(media_path, "w").close()

        produced = tmp_path / "rendered_box.png"
        produced.write_bytes(b"fake-png-bytes")

        def fake_apply_fn(source_path, left, upper, right, lower):
            return str(produced)

        launcher, gv = _make_launcher_and_gv()
        media_frame = _make_media_frame()

        launcher._run_static_rect_action(
            gv, media_frame, media_path, "/tmp/source.png", MediaType.SVG,
            restore_original=lambda: None,
            apply_fn=fake_apply_fn,
            output_suffix="_box",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
        )
        rect = SimpleNamespace(x=lambda: 0, y=lambda: 0, width=lambda: 50, height=lambda: 50)
        gv.crop_confirmed.emit(rect)

        expected = tmp_path / "icon_box.png"
        assert expected.exists()
        assert not produced.exists()
        assert not (tmp_path / "icon_box.svg").exists()

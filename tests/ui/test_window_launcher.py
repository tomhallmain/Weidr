"""UI tests for WindowLauncher secondary windows."""

import os
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


# ---------------------------------------------------------------------------
# _preview_and_confirm_fill — fill preview + reroll/accept/cancel
# ---------------------------------------------------------------------------

class TestPreviewAndConfirmFill:
    """WindowLauncher._preview_and_confirm_fill in isolation."""

    def test_accept_returns_the_generated_fill(self, tmp_path, monkeypatch):
        launcher = WindowLauncher(MagicMock())
        fills = [object(), object()]
        monkeypatch.setattr(
            "image.image_ops.ImageOps.generate_box_fill_image",
            MagicMock(side_effect=fills),
        )
        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog",
            lambda master, preview_path, on_reroll, on_solid, **kwargs: True,
        )
        rendered = []
        result = launcher._preview_and_confirm_fill(
            str(tmp_path / "source.png"),
            (10, 10),
            lambda fill, out_path: rendered.append((fill, out_path)),
        )
        assert result is fills[0]
        assert len(rendered) == 1
        assert rendered[0][0] is fills[0]

    def test_cancel_returns_none(self, tmp_path, monkeypatch):
        launcher = WindowLauncher(MagicMock())
        monkeypatch.setattr(
            "image.image_ops.ImageOps.generate_box_fill_image",
            MagicMock(return_value=object()),
        )
        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog",
            lambda master, preview_path, on_reroll, on_solid, **kwargs: False,
        )
        result = launcher._preview_and_confirm_fill(
            str(tmp_path / "source.png"), (10, 10), lambda fill, out_path: None,
        )
        assert result is None

    def test_reroll_regenerates_and_rerenders(self, tmp_path, monkeypatch):
        launcher = WindowLauncher(MagicMock())
        fills = [object(), object(), object()]
        monkeypatch.setattr(
            "image.image_ops.ImageOps.generate_box_fill_image",
            MagicMock(side_effect=fills),
        )

        def fake_dialog(master, preview_path, on_reroll, on_solid, **kwargs):
            on_reroll()  # simulate one reroll click before accepting
            return True

        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog", fake_dialog,
        )
        rendered = []
        result = launcher._preview_and_confirm_fill(
            str(tmp_path / "source.png"), (10, 10),
            lambda fill, out_path: rendered.append(fill),
        )
        # Initial render + one reroll render = 2 calls; the second (rerolled) fill wins.
        assert rendered == [fills[0], fills[1]]
        assert result is fills[1]

    def test_solid_color_button_switches_to_plain_fill(self, tmp_path, monkeypatch):
        launcher = WindowLauncher(MagicMock())
        monkeypatch.setattr(
            "image.image_ops.ImageOps.generate_box_fill_image",
            MagicMock(return_value=object()),  # the initial random fill; not asserted on here
        )

        def fake_dialog(master, preview_path, on_reroll, on_solid, **kwargs):
            on_solid((0, 0, 0))  # simulate clicking "Black"
            return True

        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog", fake_dialog,
        )
        rendered = []
        result = launcher._preview_and_confirm_fill(
            str(tmp_path / "source.png"), (10, 6),
            lambda fill, out_path: rendered.append(fill),
        )
        assert result.size == (10, 6)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (0, 0, 0)
        # Initial random render, then the solid-black render.
        assert len(rendered) == 2
        assert rendered[-1] is result

    def test_preview_file_is_cleaned_up(self, tmp_path, monkeypatch):
        launcher = WindowLauncher(MagicMock())
        monkeypatch.setattr(
            "image.image_ops.ImageOps.generate_box_fill_image",
            MagicMock(return_value=object()),
        )
        captured_path = []

        def fake_dialog(master, preview_path, on_reroll, on_solid, **kwargs):
            captured_path.append(preview_path)
            with open(preview_path, "wb") as f:
                f.write(b"x")
            return True

        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog", fake_dialog,
        )
        launcher._preview_and_confirm_fill(
            str(tmp_path / "source.png"), (10, 10), lambda fill, out_path: None,
        )
        assert not os.path.exists(captured_path[0])


# ---------------------------------------------------------------------------
# _run_static_rect_action / _run_static_polygon_action — preview_fill wiring
# ---------------------------------------------------------------------------

def _make_launcher_and_gv_polygon():
    launcher = WindowLauncher(MagicMock())
    gv = MagicMock()
    gv.polygon_confirmed = _FakeSignal()
    gv.crop_cancelled = _FakeSignal()
    return launcher, gv


class TestRunStaticRectActionPreviewFill:
    def test_accept_saves_with_the_previewed_fill(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ui.image.media_details.MediaDetails.open_temp_media_canvas",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog",
            lambda master, preview_path, on_reroll, on_solid, **kwargs: True,
        )
        media_path = str(tmp_path / "photo.png")
        open(media_path, "w").close()
        produced = tmp_path / "photo_box.png"
        produced.write_bytes(b"fake-png-bytes")

        calls = []

        def fake_apply_fn(source_path, left, upper, right, lower, fill_image=None, output_path=None):
            calls.append((fill_image, output_path))
            return str(produced) if output_path is None else output_path

        launcher, gv = _make_launcher_and_gv()
        media_frame = _make_media_frame()

        launcher._run_static_rect_action(
            gv, media_frame, media_path, "/tmp/source.png", MediaType.IMAGE,
            restore_original=lambda: None,
            apply_fn=fake_apply_fn,
            output_suffix="_box",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
            preview_fill=True,
        )
        rect = SimpleNamespace(x=lambda: 0, y=lambda: 0, width=lambda: 50, height=lambda: 50)
        gv.crop_confirmed.emit(rect)

        # One call to render the preview (output_path set), one to commit the
        # real save (output_path=None -> default sibling path); same fill both times.
        assert len(calls) == 2
        assert calls[0][1] is not None
        assert calls[1][1] is None
        assert calls[0][0] is calls[1][0]
        assert produced.exists()

    def test_cancel_does_not_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog",
            lambda master, preview_path, on_reroll, on_solid, **kwargs: False,
        )
        media_path = str(tmp_path / "photo.png")
        open(media_path, "w").close()

        calls = []

        def fake_apply_fn(source_path, left, upper, right, lower, fill_image=None, output_path=None):
            calls.append(output_path)
            return ""

        restored = []
        launcher, gv = _make_launcher_and_gv()
        media_frame = _make_media_frame()

        launcher._run_static_rect_action(
            gv, media_frame, media_path, "/tmp/source.png", MediaType.IMAGE,
            restore_original=lambda: restored.append(True),
            apply_fn=fake_apply_fn,
            output_suffix="_box",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
            preview_fill=True,
        )
        rect = SimpleNamespace(x=lambda: 0, y=lambda: 0, width=lambda: 50, height=lambda: 50)
        gv.crop_confirmed.emit(rect)

        # Only the preview render happened; no real (output_path=None) save call.
        assert len(calls) == 1
        assert calls[0] is not None
        assert restored == [True]

    def test_fill_covers_full_image_uses_full_image_size_for_background_box(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog",
            lambda master, preview_path, on_reroll, on_solid, **kwargs: True,
        )
        gen_calls = []
        monkeypatch.setattr(
            "image.image_ops.ImageOps.generate_box_fill_image",
            lambda w, h, use_texture=None, palette=None: gen_calls.append((w, h)) or object(),
        )
        media_path = str(tmp_path / "photo.png")
        open(media_path, "w").close()
        produced = tmp_path / "photo_bgbox.png"
        produced.write_bytes(b"x")

        def fake_apply_fn(source_path, left, upper, right, lower, fill_image=None, output_path=None):
            return str(produced)

        launcher, gv = _make_launcher_and_gv()
        media_frame = _make_media_frame(size=100)  # img_w = img_h = 100

        launcher._run_static_rect_action(
            gv, media_frame, media_path, "/tmp/source.png", MediaType.IMAGE,
            restore_original=lambda: None,
            apply_fn=fake_apply_fn,
            output_suffix="_bgbox",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
            preview_fill=True,
            fill_covers_full_image=True,
        )
        # A 20x20 selection well inside a 100x100 image.
        rect = SimpleNamespace(x=lambda: 10, y=lambda: 10, width=lambda: 20, height=lambda: 20)
        gv.crop_confirmed.emit(rect)

        # Background-box fill covers the whole image, not just the selection.
        assert gen_calls[0] == (100, 100)


class TestRunStaticPolygonActionPreviewFill:
    def test_accept_saves_with_the_previewed_fill(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ui.image.media_details.MediaDetails.open_temp_media_canvas",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "lib.fill_preview_dialog_qt.show_fill_preview_dialog",
            lambda master, preview_path, on_reroll, on_solid, **kwargs: True,
        )
        media_path = str(tmp_path / "photo.png")
        open(media_path, "w").close()
        produced = tmp_path / "photo_box.png"
        produced.write_bytes(b"x")

        calls = []

        def fake_apply_fn(source_path, points, fill_image=None, output_path=None):
            calls.append((fill_image, output_path))
            return str(produced) if output_path is None else output_path

        launcher, gv = _make_launcher_and_gv_polygon()
        media_frame = _make_media_frame()

        launcher._run_static_polygon_action(
            gv, media_frame, media_path, "/tmp/source.png", MediaType.IMAGE,
            restore_original=lambda: None,
            apply_fn=fake_apply_fn,
            output_suffix="_box",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
            preview_fill=True,
        )
        points = [
            SimpleNamespace(x=lambda: 0, y=lambda: 0),
            SimpleNamespace(x=lambda: 50, y=lambda: 0),
            SimpleNamespace(x=lambda: 0, y=lambda: 50),
        ]
        gv.polygon_confirmed.emit(points)

        assert len(calls) == 2
        assert calls[0][0] is calls[1][0]
        assert produced.exists()


# ---------------------------------------------------------------------------
# Reapply the last selection to the current media
# ---------------------------------------------------------------------------

from image.selection_reapply import LastSelection, SelectionKind, StoredSelection  # noqa: E402
from utils.translations import _  # noqa: E402


def _png_of_size(path, size):
    from PIL import Image
    Image.new("RGB", size, (200, 200, 200)).save(path)
    return str(path)


def _no_canvas(monkeypatch):
    monkeypatch.setattr(
        "ui.image.media_details.MediaDetails.open_temp_media_canvas",
        lambda **kwargs: None,
    )


def _accept_fill_preview(monkeypatch, accepted=True):
    calls = []

    def _dialog(master, preview_path, on_reroll, on_solid, **kwargs):
        calls.append(preview_path)
        return accepted

    monkeypatch.setattr("lib.fill_preview_dialog_qt.show_fill_preview_dialog", _dialog)
    return calls


class TestRecordLastSelection:
    def _run_box(self, tmp_path, monkeypatch, accepted, kind=SelectionKind.BOX_RECT):
        _no_canvas(monkeypatch)
        _accept_fill_preview(monkeypatch, accepted)
        media_path = str(tmp_path / "photo.png")
        open(media_path, "w").close()
        produced = tmp_path / "photo_box.png"
        produced.write_bytes(b"x")

        def fake_apply_fn(source_path, left, upper, right, lower, fill_image=None, output_path=None):
            return str(produced) if output_path is None else output_path

        launcher, gv = _make_launcher_and_gv()
        launcher._run_static_rect_action(
            gv, _make_media_frame(), media_path, "/tmp/source.png", MediaType.IMAGE,
            restore_original=lambda: None,
            apply_fn=fake_apply_fn,
            output_suffix="_box",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
            preview_fill=True,
            kind=kind,
        )
        rect = SimpleNamespace(x=lambda: 10, y=lambda: 20, width=lambda: 30, height=lambda: 40)
        gv.crop_confirmed.emit(rect)
        return media_path

    def test_accepted_edit_is_stored(self, tmp_path, monkeypatch):
        media_path = self._run_box(tmp_path, monkeypatch, accepted=True)
        stored = LastSelection.get()
        assert stored == StoredSelection(SelectionKind.BOX_RECT, (10, 20, 40, 60), (100, 100), media_path)

    def test_cancelled_preview_keeps_previous(self, tmp_path, monkeypatch):
        previous = StoredSelection(SelectionKind.CROP_RECT, (0, 0, 5, 5), (10, 10), "/old.png")
        LastSelection.set(previous)
        self._run_box(tmp_path, monkeypatch, accepted=False)
        assert LastSelection.get() is previous

    def test_without_kind_nothing_is_stored(self, tmp_path, monkeypatch):
        self._run_box(tmp_path, monkeypatch, accepted=True, kind=None)
        assert LastSelection.get() is None

    def test_polygon_edit_is_stored(self, tmp_path, monkeypatch):
        _no_canvas(monkeypatch)
        media_path = str(tmp_path / "photo.png")
        open(media_path, "w").close()
        produced = tmp_path / "photo_crop.png"
        produced.write_bytes(b"x")
        launcher, gv = _make_launcher_and_gv_polygon()
        launcher._run_static_polygon_action(
            gv, _make_media_frame(), media_path, "/tmp/source.png", MediaType.IMAGE,
            restore_original=lambda: None,
            apply_fn=lambda source_path, points: str(produced),
            output_suffix="_crop",
            too_small_msg="too small",
            success_msg="ok",
            failed_msg="failed",
            kind=SelectionKind.CROP_POLYGON,
        )
        gv.polygon_confirmed.emit([
            SimpleNamespace(x=lambda: 0, y=lambda: 0),
            SimpleNamespace(x=lambda: 50, y=lambda: 0),
            SimpleNamespace(x=lambda: 0, y=lambda: 50),
        ])
        stored = LastSelection.get()
        assert stored.kind == SelectionKind.CROP_POLYGON
        assert stored.geometry == ((0, 0), (50, 0), (0, 50))


def _reapply_launcher(tmp_path, monkeypatch, media_path, media_type, kind, produced_name):
    """Launcher whose tool table has fake apply_fns recording their calls."""
    _no_canvas(monkeypatch)
    monkeypatch.setattr("utils.media_utils.get_media_type_for_path", lambda p: media_type)
    app = MagicMock()
    app.media_path = media_path
    launcher = WindowLauncher(app)
    produced = tmp_path / produced_name
    calls = []

    def fake_rect(source_path, left, upper, right, lower, fill_image=None, output_path=None):
        calls.append((source_path, (left, upper, right, lower), output_path))
        if output_path is not None:
            return output_path
        produced.write_bytes(b"x")
        return str(produced)

    def fake_polygon(source_path, points, fill_image=None, output_path=None):
        calls.append((source_path, list(points), output_path))
        if output_path is not None:
            return output_path
        produced.write_bytes(b"x")
        return str(produced)

    real_tool = launcher._selection_tool

    def _tool(k):
        tool = dict(real_tool(k))
        tool["apply_fn"] = fake_polygon if k.is_polygon else fake_rect
        return tool

    monkeypatch.setattr(launcher, "_selection_tool", _tool)
    return launcher, app, calls


class TestReapplyLastSelection:
    def test_nothing_stored_toasts(self, tmp_path, monkeypatch):
        media_path = _png_of_size(tmp_path / "a.png", (100, 100))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, media_path, MediaType.IMAGE, None, "a_box.png"
        )
        launcher.reapply_last_selection()
        app.notification_ctrl.toast.assert_called_once_with(_("No selection to reapply yet"))
        assert calls == []

    def test_video_is_not_supported(self, tmp_path, monkeypatch):
        LastSelection.set(StoredSelection(SelectionKind.BOX_RECT, (0, 0, 5, 5), (10, 10), "/a.png"))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, str(tmp_path / "clip.mp4"), MediaType.VIDEO, None, "x.png"
        )
        launcher.reapply_last_selection()
        app.notification_ctrl.toast.assert_called_once_with(
            _("Reapplying a selection is not supported for this media type")
        )
        assert calls == []

    def test_box_opens_preview_with_scaled_geometry(self, tmp_path, monkeypatch):
        stored = StoredSelection(SelectionKind.BOX_RECT, (10, 20, 30, 40), (100, 100), "/first.png")
        LastSelection.set(stored)
        media_path = _png_of_size(tmp_path / "other.png", (200, 50))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, media_path, MediaType.IMAGE, SelectionKind.BOX_RECT, "other_box.png"
        )
        previews = _accept_fill_preview(monkeypatch, accepted=True)

        launcher.reapply_last_selection()

        assert len(previews) == 1
        assert calls[0][1] == (20, 10, 60, 20)       # preview render
        assert calls[-1] == (media_path, (20, 10, 60, 20), None)  # real save
        assert (tmp_path / "other_box.png").exists()
        assert LastSelection.get() is stored        # reapply keeps the original

    def test_box_cancel_writes_nothing(self, tmp_path, monkeypatch):
        LastSelection.set(StoredSelection(SelectionKind.BOX_RECT, (10, 20, 30, 40), (100, 100), "/first.png"))
        media_path = _png_of_size(tmp_path / "other.png", (100, 100))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, media_path, MediaType.IMAGE, SelectionKind.BOX_RECT, "other_box.png"
        )
        _accept_fill_preview(monkeypatch, accepted=False)
        launcher.reapply_last_selection()
        assert all(call[2] is not None for call in calls)
        assert not (tmp_path / "other_box.png").exists()

    def test_crop_applies_directly_without_preview(self, tmp_path, monkeypatch):
        LastSelection.set(StoredSelection(SelectionKind.CROP_RECT, (10, 20, 30, 40), (100, 100), "/first.png"))
        media_path = _png_of_size(tmp_path / "other.png", (100, 100))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, media_path, MediaType.IMAGE, SelectionKind.CROP_RECT, "other_crop.png"
        )

        def _no_dialog(*args, **kwargs):
            raise AssertionError("crop reapply must not open a preview")

        monkeypatch.setattr("lib.fill_preview_dialog_qt.show_fill_preview_dialog", _no_dialog)
        launcher.reapply_last_selection()
        assert calls == [(media_path, (10, 20, 30, 40), None)]
        assert (tmp_path / "other_crop.png").exists()

    def test_freeform_crop_scales_points(self, tmp_path, monkeypatch):
        LastSelection.set(StoredSelection(
            SelectionKind.CROP_POLYGON, ((0, 0), (50, 0), (0, 50)), (100, 100), "/first.png"
        ))
        media_path = _png_of_size(tmp_path / "other.png", (200, 200))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, media_path, MediaType.IMAGE, SelectionKind.CROP_POLYGON, "other_crop.png"
        )
        launcher.reapply_last_selection()
        assert calls == [(media_path, [(0, 0), (100, 0), (0, 100)], None)]

    def test_selection_too_small_after_scaling_toasts(self, tmp_path, monkeypatch):
        LastSelection.set(StoredSelection(SelectionKind.CROP_RECT, (10, 10, 11, 11), (1000, 1000), "/first.png"))
        media_path = _png_of_size(tmp_path / "tiny.png", (10, 10))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, media_path, MediaType.IMAGE, SelectionKind.CROP_RECT, "tiny_crop.png"
        )
        launcher.reapply_last_selection()
        app.notification_ctrl.toast.assert_called_once_with(_("Crop selection too small"))
        assert calls == []

    def test_pdf_uses_current_page_and_keeps_raster_extension(self, tmp_path, monkeypatch):
        LastSelection.set(StoredSelection(SelectionKind.CROP_RECT, (0, 0, 50, 50), (100, 100), "/first.png"))
        media_path = str(tmp_path / "paper.pdf")
        open(media_path, "w").close()
        page = _png_of_size(tmp_path / "page_3.jpg", (100, 100))
        launcher, app, calls = _reapply_launcher(
            tmp_path, monkeypatch, media_path, MediaType.PDF, SelectionKind.CROP_RECT, "page_3_crop.jpg"
        )
        app.media_frame.pdf_current_page_path.return_value = page
        launcher.reapply_last_selection()
        assert calls[0][0] == page
        assert (tmp_path / "paper_crop.jpg").exists()

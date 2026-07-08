"""
Regression tests for large-image preview decoding and full-resolution promotion.

Images with a dimension above large_image_dim_threshold_px are first decoded to
a viewport-sized preview (QImageReader.setScaledSize) and later promoted to full
resolution in a background worker when RAM budgets allow.

setScaledSize() scales to *exactly* the size it is given — it does not preserve
aspect ratio. _preview_target_size() therefore has to hand it a size that already
matches the source aspect ratio. It originally clamped width and height to the
viewport independently, so every sufficiently large image was stretched to the
media frame's aspect ratio (and stayed stretched whenever promotion was gated
off by the RAM/allocation-limit checks).
"""

import pytest

import ui.app_window.media_frame as mf_module
from utils.media_utils import large_image_dim_threshold


def _make_large_jpeg(tmp_path, width, height):
    from PIL import Image

    path = str(tmp_path / f"large_{width}x{height}.jpg")
    Image.new("RGB", (width, height), (40, 120, 200)).save(path, format="JPEG")
    return path


def _aspect(w, h):
    return w / h


@pytest.mark.parametrize("source_dims", [(6000, 2000), (2000, 6000)])
def test_preview_target_size_preserves_source_aspect_ratio(media_frame, source_dims):
    """The preview decode target must keep the source aspect ratio.

    The media frame viewport is roughly square here while the source is 3:1
    (and 1:3), so the pre-fix independent per-axis clamp would return a
    viewport-shaped (square) target for both cases.
    """
    src_w, src_h = source_dims
    assert max(source_dims) > large_image_dim_threshold(), (
        "test dims must qualify for the large-image path"
    )

    target = media_frame._preview_target_size(source_dims)

    assert target.width() > 0 and target.height() > 0
    assert target.width() <= src_w and target.height() <= src_h
    assert _aspect(target.width(), target.height()) == pytest.approx(
        _aspect(src_w, src_h), rel=0.05
    ), "preview target was clamped per-axis instead of fit with the source aspect ratio"


def test_large_image_preview_displays_with_source_aspect_ratio(
    media_frame, qtbot, tmp_path, monkeypatch
):
    """show_media() on a large image must render an aspect-correct preview.

    Promotion is disabled so the test observes the preview pixmap itself — the
    original bug was masked whenever the full-res promotion succeeded and
    silently replaced the stretched preview.
    """
    monkeypatch.setattr(
        mf_module, "large_image_full_res_promotion_enabled", lambda: False
    )
    path = _make_large_jpeg(tmp_path, 6000, 2000)

    media_frame.show_media(path)
    qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=5000)

    pix = media_frame._current_pixmap
    assert pix is not None and not pix.isNull()
    assert pix.width() < 6000, "expected a downscaled preview, not a full-res decode"
    assert _aspect(pix.width(), pix.height()) == pytest.approx(3.0, rel=0.05), (
        "large-image preview was stretched to the media frame's aspect ratio"
    )
    # Source dimensions must still be reported for analysis/zoom logic.
    assert (media_frame.imwidth, media_frame.imheight) == (6000, 2000)


def test_large_image_promotion_restores_full_resolution(
    media_frame, qtbot, tmp_path, monkeypatch
):
    """After background promotion the displayed image is full-res with source aspect ratio.

    RAM gates are pinned open so _can_promote_large_image() passes regardless of
    the machine running the suite.
    """
    monkeypatch.setattr(
        mf_module, "large_image_full_res_promotion_enabled", lambda: True
    )
    monkeypatch.setattr(
        mf_module.Utils, "calculate_available_ram", staticmethod(lambda: 64.0)
    )
    monkeypatch.setattr(
        mf_module, "large_image_promotion_min_free_ram_gb", lambda: 0.0
    )
    path = _make_large_jpeg(tmp_path, 6000, 2000)

    media_frame.show_media(path)
    qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=5000)

    preview = media_frame._current_pixmap
    assert _aspect(preview.width(), preview.height()) == pytest.approx(3.0, rel=0.05)

    qtbot.waitUntil(
        lambda: media_frame._image is not None
        and media_frame._image.width() == 6000,
        timeout=15000,
    )
    promoted = media_frame._current_pixmap
    assert (promoted.width(), promoted.height()) == (6000, 2000)
    assert (media_frame.imwidth, media_frame.imheight) == (6000, 2000)


def test_switching_media_mid_promotion_is_safe(
    media_frame, qtbot, tmp_path, monkeypatch
):
    """Navigating away while a promotion decode is in flight must not crash or
    let the stale decode overwrite the new image.

    Cancelling a running _ImageDecodeWorker must not delete its QThread while
    the thread is still executing (that aborts the process), and a late decoded
    signal from the cancelled worker must be discarded via the request-id guard.
    """
    from PIL import Image

    monkeypatch.setattr(
        mf_module, "large_image_full_res_promotion_enabled", lambda: True
    )
    monkeypatch.setattr(
        mf_module.Utils, "calculate_available_ram", staticmethod(lambda: 64.0)
    )
    monkeypatch.setattr(
        mf_module, "large_image_promotion_min_free_ram_gb", lambda: 0.0
    )
    large_path = _make_large_jpeg(tmp_path, 6000, 2000)
    small_path = str(tmp_path / "small.png")
    Image.new("RGB", (10, 10), (200, 50, 50)).save(small_path, format="PNG")

    media_frame.show_media(large_path)
    qtbot.waitUntil(lambda: media_frame.media_displayed, timeout=5000)
    # Switch immediately — the promotion worker for large_path is likely still decoding.
    media_frame.show_media(small_path)
    qtbot.waitUntil(
        lambda: media_frame._image is not None and media_frame._image.width() == 10,
        timeout=5000,
    )

    # Give any stale decoded signal time to arrive; the request-id guard must drop it.
    qtbot.wait(750)
    assert media_frame._image.width() == 10, (
        "stale promotion decode replaced the newly shown image"
    )

"""Build synthetic raster/video files under tmp_path for show_media tests."""

from __future__ import annotations

import os
import subprocess
from io import BytesIO
from typing import Any, Dict, Optional

import pytest
from PIL import Image

from utils.config import config
from utils.pillow_plugins import ensure_pillow_plugins_registered


def pillow_can_save(format_name: str) -> bool:
    buf = BytesIO()
    try:
        Image.new("RGB", (4, 4), (90, 100, 110)).save(buf, format=format_name)
        return buf.tell() > 0
    except Exception:
        return False


def _write_animated(path: str, fmt: str, base: Image.Image, frame2: Image.Image) -> bool:
    try:
        base.save(
            path,
            format=fmt,
            save_all=True,
            append_images=[frame2],
            duration=100,
            loop=0,
        )
        return os.path.isfile(path)
    except Exception:
        return False


def _try_make_mp4(tmp_path) -> Optional[str]:
    try:
        from image.video_ops import VideoOps

        ffmpeg = VideoOps.find_ffmpeg_executable()
    except Exception:
        return None
    if not ffmpeg:
        return None
    out = os.path.join(tmp_path, "short.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=64x64:d=0.2",
        "-pix_fmt",
        "yuv420p",
        out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except Exception:
        return None
    return out if os.path.isfile(out) and os.path.getsize(out) > 0 else None


@pytest.fixture
def show_media_files(tmp_path) -> Dict[str, Any]:
    """
    Map of logical kind -> absolute path (or None when the format is unavailable).

    All files live under the per-test tmp_path; nothing touches user directories.
    """
    ensure_pillow_plugins_registered()
    catalog: Dict[str, Any] = {}
    base = Image.new("RGB", (32, 24), (40, 80, 120))
    frame2 = Image.new("RGB", (32, 24), (200, 60, 60))

    for key, ext, fmt in (
        ("png", ".png", "PNG"),
        ("jpg", ".jpg", "JPEG"),
        ("jpeg", ".jpeg", "JPEG"),
    ):
        path = os.path.join(tmp_path, f"sample{ext}")
        base.save(path, format=fmt)
        catalog[key] = path

    static_webp = os.path.join(tmp_path, "static.webp")
    base.save(static_webp, format="WEBP")
    catalog["webp_static"] = static_webp

    gif_path = os.path.join(tmp_path, "anim.gif")
    _write_animated(gif_path, "GIF", base, frame2)
    catalog["gif"] = gif_path

    anim_webp = os.path.join(tmp_path, "anim.webp")
    catalog["webp_animated"] = (
        anim_webp if _write_animated(anim_webp, "WEBP", base, frame2) else None
    )

    if pillow_can_save("HEIF"):
        heic_path = os.path.join(tmp_path, "sample.heic")
        try:
            base.save(heic_path, format="HEIF")
            catalog["heic"] = heic_path
        except Exception:
            catalog["heic"] = None
    else:
        catalog["heic"] = None

    if pillow_can_save("AVIF"):
        avif_path = os.path.join(tmp_path, "sample.avif")
        try:
            base.save(avif_path, format="AVIF")
            catalog["avif"] = avif_path
        except Exception:
            catalog["avif"] = None
    else:
        catalog["avif"] = None

    catalog["mp4"] = _try_make_mp4(tmp_path)

    svg_path = os.path.join(tmp_path, "shape.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
            '<rect width="20" height="20" fill="#336699"/></svg>'
        )
    catalog["svg"] = svg_path

    catalog["pdf"] = os.path.join(tmp_path, "doc.pdf")
    with open(catalog["pdf"], "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")

    catalog["html"] = os.path.join(tmp_path, "page.html")
    with open(catalog["html"], "w", encoding="utf-8") as f:
        f.write("<html><body><p>test</p></body></html>")

    return catalog


def require_extension_in_config(ext: str) -> None:
    types = [t.lower() for t in getattr(config, "image_types", [])]
    if ext.lower() not in types:
        pytest.skip(f"{ext} not in config.image_types for this test run")

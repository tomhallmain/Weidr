"""Hand-built ePub archives for tests (no third-party ePub library)."""

from __future__ import annotations

import io
import zipfile
from typing import Optional, Sequence

CONTAINER_XML = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

COVER_WRAPPER_XHTML = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Cover</title></head>
<body><div><img src="images/cover.png" alt=""/></div></body></html>
"""


def chapter_xhtml(text: str, head_extra: str = "") -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title>'
        f"{head_extra}</head><body><p>{text}</p></body></html>\n"
    )


def png_bytes(size=(60, 90), color=(200, 30, 30)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def build_epub(
    path,
    chapters: Sequence[tuple] = (("ch1", "chapter1.xhtml", "yes"),),
    cover: Optional[str] = "epub3",
    cover_wrapper_first: bool = False,
    fixed_layout: bool = False,
    encryption_entries: Sequence[tuple] = (),
    chapter_head_extra: str = "",
) -> str:
    """Write an ePub at *path*.

    *chapters*: ``(id, href, linear)`` spine entries, each written as a one-paragraph chapter.
    *cover*: ``"epub3"`` (manifest ``properties="cover-image"``), ``"epub2"``
    (``<meta name="cover">``) or None.
    *encryption_entries*: ``(algorithm, uri)`` pairs for META-INF/encryption.xml.
    """
    manifest = []
    spine = []
    files = {}
    meta = ""
    if cover is not None:
        props = ' properties="cover-image"' if cover == "epub3" else ""
        manifest.append(f'<item id="cover-img" href="images/cover.png" media-type="image/png"{props}/>')
        files["OEBPS/images/cover.png"] = png_bytes()
        if cover == "epub2":
            meta += '<meta name="cover" content="cover-img"/>'
    if cover_wrapper_first:
        manifest.append('<item id="cover-page" href="cover.xhtml" media-type="application/xhtml+xml"/>')
        spine.append('<itemref idref="cover-page"/>')
        files["OEBPS/cover.xhtml"] = COVER_WRAPPER_XHTML
    for item_id, href, linear in chapters:
        manifest.append(f'<item id="{item_id}" href="{href}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{item_id}" linear="{linear}"/>')
        files[f"OEBPS/{href}"] = chapter_xhtml(f"Text of {item_id}", chapter_head_extra)
    if fixed_layout:
        meta += '<meta property="rendition:layout">pre-paginated</meta>'
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">'
        f'<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Book</dc:title>{meta}</metadata>'
        f'<manifest>{"".join(manifest)}</manifest>'
        f'<spine>{"".join(spine)}</spine></package>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", opf)
        for name, data in files.items():
            zf.writestr(name, data)
        if encryption_entries:
            entries = "".join(
                '<enc:EncryptedData><enc:EncryptionMethod Algorithm="{0}"/>'
                '<enc:CipherData><enc:CipherReference URI="{1}"/></enc:CipherData>'
                "</enc:EncryptedData>".format(alg, uri)
                for alg, uri in encryption_entries
            )
            zf.writestr(
                "META-INF/encryption.xml",
                '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                'xmlns:enc="http://www.w3.org/2001/04/xmlenc#">' + entries + "</encryption>",
            )
    return str(path)

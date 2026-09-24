"""
ePub container reading for FrameCache's derived-PDF build.

Parses ``META-INF/container.xml`` and the OPF package document (spine,
cover, fixed-layout flag), detects DRM from ``META-INF/encryption.xml``,
and extracts the archive with zip-slip and zip-bomb guards. Rendering the
spine into a PDF happens in :mod:`image.frame_cache`.
"""

from __future__ import annotations

import os
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

# Extraction limits: a crafted archive must not fill the disk or the inode table.
MAX_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_ENTRY_COUNT = 20000
MAX_COVER_BYTES = 64 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024

# Font obfuscation (IDPF, Adobe) scrambles embedded fonts only; it is not DRM.
FONT_OBFUSCATION_ALGORITHMS = frozenset({
    "http://www.idpf.org/2008/embedding",
    "http://ns.adobe.com/pdf/enc#RC",
})

_CONTAINER_PATH = "META-INF/container.xml"
_ENCRYPTION_PATH = "META-INF/encryption.xml"


class EpubError(Exception):
    """The ePub can't be paged: unreadable archive, missing OPF, empty spine, DRM, or a failed render."""


@dataclass
class EpubSpineItem:
    name: str               # archive member name, resolved against the OPF directory
    fixed_layout: bool      # pre-paginated: one document is one page


@dataclass
class EpubInfo:
    opf_name: str
    spine: List[EpubSpineItem]
    cover_image_name: Optional[str]
    drm: bool
    encrypted_names: Set[str] = field(default_factory=set)


def _local(tag: str) -> str:
    """Element or attribute name without its ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


def _attr(elem: ET.Element, name: str) -> Optional[str]:
    for key, value in elem.attrib.items():
        if _local(key) == name:
            return value
    return None


def _iter_local(root: ET.Element, name: str):
    for elem in root.iter():
        if _local(elem.tag) == name:
            yield elem


def resolve_href(base_name: str, href: str) -> str:
    """Archive member name for *href* as written in the member *base_name*
    (URL-decoded, fragment dropped, relative to *base_name*'s directory)."""
    href = unquote(href.split("#", 1)[0])
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_name), href))


def _read_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        data = zf.read(name)
    except KeyError:
        raise EpubError(f"Missing {name}")
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise EpubError(f"Malformed XML in {name}: {e}")


def _find_opf_name(zf: zipfile.ZipFile) -> str:
    root = _read_xml(zf, _CONTAINER_PATH)
    for rootfile in _iter_local(root, "rootfile"):
        full_path = _attr(rootfile, "full-path")
        if full_path:
            return unquote(full_path)
    raise EpubError("container.xml names no OPF package document")


def _encrypted_names_by_algorithm(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    """``(member_name, algorithm)`` for every entry in ``META-INF/encryption.xml``."""
    if _ENCRYPTION_PATH not in zf.namelist():
        return []
    root = _read_xml(zf, _ENCRYPTION_PATH)
    entries = []
    for data in _iter_local(root, "EncryptedData"):
        algorithm = ""
        for method in _iter_local(data, "EncryptionMethod"):
            algorithm = _attr(method, "Algorithm") or ""
            break
        for ref in _iter_local(data, "CipherReference"):
            uri = _attr(ref, "URI")
            if uri:
                # URIs here are relative to the container root, not to the OPF.
                entries.append((posixpath.normpath(unquote(uri)), algorithm))
    return entries


def _find_cover_image_name(opf_root: ET.Element, manifest: dict, opf_name: str) -> Optional[str]:
    # ePub 3: <item properties="cover-image">
    for item in _iter_local(opf_root, "item"):
        props = (_attr(item, "properties") or "").split()
        href = _attr(item, "href")
        if "cover-image" in props and href:
            return resolve_href(opf_name, href)
    # ePub 2: <meta name="cover" content="<manifest id>"/>
    for meta in _iter_local(opf_root, "meta"):
        if _attr(meta, "name") == "cover":
            item = manifest.get(_attr(meta, "content") or "")
            if item is not None:
                return item[0]
    return None


def _is_package_fixed_layout(opf_root: ET.Element) -> bool:
    for meta in _iter_local(opf_root, "meta"):
        if _attr(meta, "property") == "rendition:layout":
            return (meta.text or "").strip() == "pre-paginated"
    return False


def read_epub_info(zf: zipfile.ZipFile) -> EpubInfo:
    """Parse the container, OPF and encryption list of an open ePub archive.

    Spine items with ``linear="no"`` (notes and pop-ups reached by links) are
    left out. Raises :class:`EpubError` when there is no OPF or no readable spine.
    """
    opf_name = _find_opf_name(zf)
    opf_root = _read_xml(zf, opf_name)

    manifest: dict = {}  # id -> (member_name, media_type)
    for item in _iter_local(opf_root, "item"):
        item_id = _attr(item, "id")
        href = _attr(item, "href")
        if item_id and href:
            manifest[item_id] = (resolve_href(opf_name, href), _attr(item, "media-type") or "")

    package_fixed = _is_package_fixed_layout(opf_root)
    spine: List[EpubSpineItem] = []
    for itemref in _iter_local(opf_root, "itemref"):
        if (_attr(itemref, "linear") or "yes").strip().lower() == "no":
            continue
        item = manifest.get(_attr(itemref, "idref") or "")
        if item is None:
            continue
        props = (_attr(itemref, "properties") or "").split()
        fixed = package_fixed
        if "rendition:layout-pre-paginated" in props:
            fixed = True
        elif "rendition:layout-reflowable" in props:
            fixed = False
        spine.append(EpubSpineItem(name=item[0], fixed_layout=fixed))
    if not spine:
        raise EpubError("Empty spine")

    encrypted = _encrypted_names_by_algorithm(zf)
    drm_names = {name for name, alg in encrypted if alg not in FONT_OBFUSCATION_ALGORITHMS}
    spine_names = {s.name for s in spine}
    return EpubInfo(
        opf_name=opf_name,
        spine=spine,
        cover_image_name=_find_cover_image_name(opf_root, manifest, opf_name),
        drm=bool(drm_names & spine_names),
        encrypted_names=drm_names,
    )


def _is_unsafe_member_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return True
    return ".." in normalized.split("/")


def safe_extract(zf: zipfile.ZipFile, dest_dir: str) -> None:
    """Extract every member of *zf* under *dest_dir*.

    Rejects absolute names, drive letters and ``..`` components (zip-slip),
    more than :data:`MAX_ENTRY_COUNT` entries, and more than
    :data:`MAX_TOTAL_UNCOMPRESSED_BYTES` in total. The size cap counts bytes
    as they are written, since the sizes in the archive's header can be forged.
    """
    infos = zf.infolist()
    if len(infos) > MAX_ENTRY_COUNT:
        raise EpubError(f"Too many archive entries ({len(infos)})")
    dest_root = os.path.realpath(dest_dir)
    written = 0
    for info in infos:
        if _is_unsafe_member_name(info.filename):
            raise EpubError(f"Unsafe archive entry name: {info.filename!r}")
        target = os.path.realpath(os.path.join(dest_root, *info.filename.replace("\\", "/").split("/")))
        if target != dest_root and not target.startswith(dest_root + os.sep):
            raise EpubError(f"Archive entry escapes extraction folder: {info.filename!r}")
        if info.is_dir():
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise EpubError("Archive exceeds the uncompressed size limit")
                dst.write(chunk)


def read_cover_bytes(zf: zipfile.ZipFile, info: EpubInfo) -> Optional[bytes]:
    """Raw bytes of the cover image, or None when there is none, it is encrypted,
    or it is larger than :data:`MAX_COVER_BYTES`."""
    name = info.cover_image_name
    if not name or name in info.encrypted_names:
        return None
    try:
        with zf.open(name) as f:
            data = f.read(MAX_COVER_BYTES + 1)
    except KeyError:
        return None
    if len(data) > MAX_COVER_BYTES:
        return None
    return data


def is_url_inside(url: str, root_dir: str) -> bool:
    """True for a ``data:`` URL or a ``file://`` URL naming a path under *root_dir*.
    The ePub renderer lets only these requests through."""
    if url.startswith("data:"):
        return True
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return False
    local = os.path.realpath(url2pathname(parsed.path))
    root = os.path.realpath(root_dir)
    return local == root or local.startswith(root + os.sep)


_TAG_RE = re.compile(r"<[^>]*>", re.DOTALL)
_HEAD_RE = re.compile(r"<head\b.*?</head\s*>", re.DOTALL | re.IGNORECASE)
_IMAGE_REF_RE = re.compile(
    r"<(?:img\b[^>]*?\bsrc|image\b[^>]*?\b(?:xlink:)?href)\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


def document_only_wraps_image(document_text: str, document_name: str, image_name: str) -> bool:
    """True when the spine document *document_name* shows *image_name* and has no body text
    (an ePub's cover page wrapper)."""
    refs = [resolve_href(document_name, ref) for ref in _IMAGE_REF_RE.findall(document_text)]
    if not refs or any(ref != image_name for ref in refs):
        return False
    body = _HEAD_RE.sub("", document_text)
    text = _TAG_RE.sub("", body)
    return not re.sub(r"&nbsp;|&#160;|&#xa0;|\s", "", text, flags=re.IGNORECASE)


_VIEWPORT_RE = re.compile(
    r"<meta\b[^>]*\bname\s*=\s*[\"']viewport[\"'][^>]*>", re.IGNORECASE
)
_CONTENT_RE = re.compile(r"\bcontent\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)


def viewport_size(document_text: str) -> Optional[Tuple[int, int]]:
    """``(width, height)`` in CSS pixels from a fixed-layout document's
    ``<meta name="viewport" content="width=…, height=…">``, or None."""
    meta = _VIEWPORT_RE.search(document_text)
    if not meta:
        return None
    content = _CONTENT_RE.search(meta.group(0))
    if not content:
        return None
    values = {}
    for part in re.split(r"[,;]", content.group(1)):
        key, _sep, value = part.partition("=")
        number = re.match(r"\s*(\d+(?:\.\d+)?)", value)
        if number:
            values[key.strip().lower()] = int(float(number.group(1)))
    width, height = values.get("width"), values.get("height")
    if width and height and width > 0 and height > 0:
        return width, height
    return None

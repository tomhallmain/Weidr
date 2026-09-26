"""
Example pipelines shipped with the app, stored encrypted.

The example pipeline JSON files are kept in the repository as one zip archive,
encrypted with a key derived from the app identifier (EXAMPLE_PIPELINES_ARCHIVE).
The first time the examples are needed, the archive is decrypted and its JSON
files are extracted into EXAMPLE_PIPELINES_DIRECTORY, which is not tracked.
The extracted files are the ones to edit; scripts/update_example_pipelines.py
packs them back into the archive.

A marker file in the directory records the SHA-256 of the archive the files
came from. A changed archive (e.g. after pulling) is extracted again,
overwriting the files it contains; an unchanged one leaves local edits alone.
Files not in the archive are never removed.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from typing import Optional

from utils.constants import AppInfo
from utils.logging_setup import get_logger
from utils.repo_paths import repo_root

logger = get_logger("example_pipelines")

EXAMPLE_PIPELINES_DIRECTORY = os.path.join(repo_root(), "assets", "pipelines")
EXAMPLE_PIPELINES_ARCHIVE = os.path.join(repo_root(), "assets", "example_pipelines.enc")
SOURCE_MARKER_FILE = ".archive_sha256"
# Changing the app identifier makes the archive unreadable until it is
# regenerated with scripts/update_example_pipelines.py.
_PASSPHRASE_SUFFIX = "_example_pipelines"


def _passphrase() -> bytes:
    return (AppInfo.APP_IDENTIFIER + _PASSPHRASE_SUFFIX).encode("utf-8")


def _sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def json_files_in(directory: str) -> list[str]:
    """Names of the .json files directly in *directory*, sorted."""
    try:
        return sorted(n for n in os.listdir(directory) if n.lower().endswith(".json"))
    except OSError:
        return []


def pack(directory: Optional[str] = None, archive: Optional[str] = None) -> list[str]:
    """Zip every .json file in *directory*, encrypt the zip to *archive*, and
    record *archive*'s hash in *directory* so the files are not extracted
    over. Returns the packed file names."""
    from utils.encryptor import symmetric_encrypt_data_to_file

    directory = directory or EXAMPLE_PIPELINES_DIRECTORY
    archive = archive or EXAMPLE_PIPELINES_ARCHIVE
    names = json_files_in(directory)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            with open(os.path.join(directory, name), "rb") as f:
                zf.writestr(zipfile.ZipInfo(name), f.read(),
                            compress_type=zipfile.ZIP_DEFLATED)
    # The zip is already deflated, so the encryptor's own compression is skipped.
    symmetric_encrypt_data_to_file(buffer.getvalue(), archive, _passphrase(), compress=False)
    _write_marker(directory, _sha256_of_file(archive))
    return names


def read_archive(archive: Optional[str] = None) -> dict[str, bytes]:
    """File name -> contents of every .json file in *archive*. Entries that
    are not plain .json file names (subdirectories, "..") are ignored."""
    from utils.encryptor import symmetric_decrypt_data_from_file

    archive = archive or EXAMPLE_PIPELINES_ARCHIVE
    data = symmetric_decrypt_data_from_file(archive, _passphrase())
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            name = info.filename
            if (info.is_dir() or os.path.basename(name) != name
                    or name in (".", "..") or not name.lower().endswith(".json")):
                logger.warning("Example pipeline archive: ignoring entry %r", name)
                continue
            files[name] = zf.read(info)
    return files


def _read_marker(directory: str) -> str:
    try:
        with open(os.path.join(directory, SOURCE_MARKER_FILE), "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _write_marker(directory: str, digest: str) -> None:
    with open(os.path.join(directory, SOURCE_MARKER_FILE), "w", encoding="utf-8") as f:
        f.write(digest + "\n")


def ensure_extracted(directory: Optional[str] = None, archive: Optional[str] = None) -> bool:
    """Extract *archive* into *directory* unless the files there came from
    this same archive. Returns True when files were extracted. A missing or
    unreadable archive is logged and leaves *directory* as it is."""
    directory = directory or EXAMPLE_PIPELINES_DIRECTORY
    archive = archive or EXAMPLE_PIPELINES_ARCHIVE
    if not os.path.isfile(archive):
        return False
    digest = _sha256_of_file(archive)
    if _read_marker(directory) == digest:
        return False
    try:
        files = read_archive(archive)
    except Exception:
        logger.exception("Could not read the example pipeline archive %s", archive)
        return False
    os.makedirs(directory, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(directory, name), "wb") as f:
            f.write(content)
    _write_marker(directory, digest)
    logger.info("Extracted %d example pipeline(s) to %s", len(files), directory)
    return True

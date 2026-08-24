"""Qt-free directory-level file management.

The media-processing counterpart lives in image/directory_ops.py; this is for
operations on directories as such, independent of what the files contain.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from utils.config import config
from utils.logging_setup import get_logger
from utils.utils import Utils

logger = get_logger("files_directory_ops")


@dataclass
class DirectoryMergeResult:
    moved: int = 0
    skipped: int = 0


def move_directory_contents_then_delete(
    base_dir: str, target_dir: str
) -> DirectoryMergeResult:
    """Move everything out of *base_dir* into *target_dir*, then remove *base_dir*.

    An entry whose destination name already exists is skipped and left where it
    is -- this deliberately does not use Utils.move_file, which raises on a
    name collision. Because skipped entries stay behind, *base_dir* is not
    necessarily empty at the end; it is removed regardless, honouring the
    configured trash-folder setting like any other delete.
    """
    logger.info(f"Moving contents of {base_dir} to {target_dir}")
    result = DirectoryMergeResult()

    for entry in os.listdir(base_dir):
        src = os.path.join(base_dir, entry)
        dst = os.path.join(target_dir, entry)
        if os.path.exists(dst):
            logger.warning(f"Skipping {src}: destination {dst} already exists")
            result.skipped += 1
            continue
        shutil.move(src, dst)
        result.moved += 1

    Utils.remove_path(
        base_dir,
        delete_instantly=config.delete_instantly,
        trash_folder=config.trash_folder,
        is_directory=True,
    )
    return result

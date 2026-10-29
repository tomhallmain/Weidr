"""Directory comparison helpers for directory groupings (e.g. a directory
profile's directory list).

Overlap means one directory equals or contains the other: batch scans are
recursive, so a run on a parent also covers every subdirectory.
"""

import os
from typing import Iterable, List, Tuple


def canonical_path(path: str) -> str:
    """Absolute, normalized, case-folded (on Windows) form of *path*.

    Symlinks are not resolved: resolving can block on unreachable network
    drives, so two aliases of one directory compare as different.
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(os.path.expanduser(path))))


def paths_overlap(a: str, b: str) -> bool:
    """True if *a* and *b* are the same directory or one contains the other."""
    ca, cb = canonical_path(a), canonical_path(b)
    if ca == cb:
        return True
    try:
        common = os.path.commonpath([ca, cb])
    except ValueError:
        # Different drives, or a mix of absolute and relative paths.
        return False
    return common in (ca, cb)


def overlapping_pairs(dirs_a: Iterable[str], dirs_b: Iterable[str]) -> List[Tuple[str, str]]:
    """Every ``(a, b)`` pair, as given, where the two directories overlap."""
    dirs_b = list(dirs_b)
    return [(a, b) for a in dirs_a for b in dirs_b if paths_overlap(a, b)]


def groupings_overlap(dirs_a: Iterable[str], dirs_b: Iterable[str]) -> bool:
    dirs_b = list(dirs_b)
    return any(paths_overlap(a, b) for a in dirs_a for b in dirs_b)


def same_directory_set(dirs_a: Iterable[str], dirs_b: Iterable[str]) -> bool:
    return {canonical_path(d) for d in dirs_a} == {canonical_path(d) for d in dirs_b}

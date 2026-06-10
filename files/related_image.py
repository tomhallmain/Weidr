import glob
import os
import re

from image.image_data_extractor import image_data_extractor
from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("related_image")

# Default ComfyUI node class used to locate the related image in workflow metadata.
DEFAULT_NODE_ID = "LoadImage"

# Module-level state for downstream related-image navigation.
# These replace the class-level attributes that previously lived on MediaDetails.
_downstream_index: int = 0
_downstream_cache: dict = {}
_downstream_browser = None  # FileBrowser, created lazily to avoid circular import at load time

# Broad pattern: underscore + 1–8 alphanumeric chars at end of stem.
# Covers generator-appended suffixes such as _edit, _v2, _1, _abc1.
# Used by the file browser for sort grouping where a false positive is low-cost.
_VARIANT_SUFFIX_RE = re.compile(r'^(.+)_[A-Za-z0-9]{1,8}$')

# Strict pattern: underscore + 1–8 alpha-only chars at end of stem.
# Used for cross-directory downstream detection where false positives are costly.
_VARIANT_SUFFIX_RE_STRICT = re.compile(r'^(.+)_[A-Za-z]{1,8}$')


def _ensure_related_path(entry) -> str:
    """Return the resolved related_image_path for entry (empty string if none)."""
    if entry.related_image_path is None:
        entry.set_related_image_path()
    return entry.related_image_path or ""


def get_origin_basename(entry, basename_lookup: dict, visited: set = None) -> str:
    """
    Return the basename of the root origin image for entry within basename_lookup.

    Resolution order:
      1. Follow the metadata-stored related_image chain within the lookup.
         If the chain leads outside the lookup, return that basename as-is
         (the chain is broken at the directory boundary).
      2. When no metadata chain exists, attempt filename-pattern inference by
         stripping a trailing variant suffix (_[A-Za-z0-9]{1,8}) to find a
         parent in the lookup, then continue the chain from there.
      3. Return entry.basename if neither source is resolvable.

    entry must expose .basename (str), .related_image_path (str | None), and
    .set_related_image_path() — satisfied by SortableFile.
    basename_lookup maps basename → entry-like object.
    """
    if visited is None:
        visited = set()
    visited.add(entry.basename)

    related_path = _ensure_related_path(entry)

    if related_path:
        related_basename = os.path.basename(related_path)
        if related_basename not in visited and related_basename in basename_lookup:
            return get_origin_basename(basename_lookup[related_basename], basename_lookup, visited)
        return related_basename

    # No metadata chain — try stripping a variant suffix to infer a parent.
    stem, ext = os.path.splitext(entry.basename)
    m = _VARIANT_SUFFIX_RE.match(stem)
    if m:
        candidate = m.group(1) + ext
        if candidate in basename_lookup and candidate not in visited:
            return get_origin_basename(basename_lookup[candidate], basename_lookup, visited)

    return entry.basename


def get_related_image_path(
    image_path: str,
    node_id: str = "LoadImage",
    check_extra_directories: bool | None = True,
) -> tuple[str | None, bool]:
    """
    Resolve the related image path for image_path.

    Returns (path, exact_match). exact_match is True only when the file
    was found on disk. check_extra_directories=None skips the existence
    check entirely and returns the raw metadata value with exact_match=False.
    """
    related_image_path = image_data_extractor.get_related_image_path(image_path, node_id)
    if related_image_path is None or related_image_path == "":
        return None, False
    elif check_extra_directories is None:
        return related_image_path, False
    elif not os.path.isfile(related_image_path):
        if not check_extra_directories:
            return related_image_path, False
        logger.info(f"{image_path} - Related image {related_image_path} not found")
        related_image_path_found = False
        if len(config.directories_to_search_for_related_images) > 0:
            basename = os.path.basename(related_image_path)
            for directory in config.directories_to_search_for_related_images:
                dir_filepaths = glob.glob(os.path.join(directory, "**/*"), recursive=True)
                for file_path in dir_filepaths:
                    if file_path == image_path:
                        continue
                    if file_path.endswith(basename) and os.path.basename(file_path) == basename:
                        related_image_path = file_path
                        related_image_path_found = True
                        break
                if related_image_path_found:
                    break
        if not related_image_path_found or not os.path.isfile(related_image_path):
            return related_image_path, False
        logger.info(f"{image_path} - Possibly related image {related_image_path} found")
    return related_image_path, True


def get_related_image_text(image_path: str, node_id: str = "LoadImage") -> str:
    """Return a display string describing the related image for image_path."""
    related_image_path, exact_match = get_related_image_path(
        image_path, node_id, check_extra_directories=False
    )
    if related_image_path is not None:
        return (
            related_image_path if exact_match
            else related_image_path + _(" (Exact Match Not Found)")
        )
    return _("(No related image found)")


def refresh_downstream_related_image_cache(
    key: str, image_path: str, other_base_dir: str
) -> None:
    global _downstream_browser, _downstream_cache
    from files.file_browser import FileBrowser
    if _downstream_browser is None or _downstream_browser.directory != other_base_dir:
        _downstream_browser = FileBrowser(directory=other_base_dir)
    _downstream_browser._gather_files()
    downstream: list[str] = []
    image_basename = os.path.basename(image_path)
    for path in _downstream_browser.filepaths:
        if path == image_path:
            continue
        related, _exact = get_related_image_path(path, check_extra_directories=None)
        if related is not None:
            if related == image_path:
                downstream.append(path)
            else:
                file_basename = os.path.basename(related)
                if len(file_basename) > 10 and image_basename == file_basename:
                    # NOTE: relation criteria is intentionally loose
                    downstream.append(path)
    _downstream_cache[key] = downstream


def get_downstream_related_images(
    image_path: str,
    other_base_dir: str,
    app_actions,
    force_refresh: bool = False,
):
    global _downstream_index, _downstream_cache
    key = image_path + "/" + other_base_dir
    if force_refresh or key not in _downstream_cache:
        refresh_downstream_related_image_cache(key, image_path, other_base_dir)
        downstream = _downstream_cache[key]
        toast_text = _("{0} downstream image(s) found.").format(len(downstream))
    else:
        downstream = _downstream_cache[key]
        toast_text = _("{0} (cached) downstream image(s) found.").format(len(downstream))
        if _downstream_index >= len(downstream):
            refresh_downstream_related_image_cache(key, image_path, other_base_dir)
            downstream = _downstream_cache[key]
            toast_text = _("{0} downstream image(s) found.").format(len(downstream))
    if len(downstream) == 0:
        app_actions.toast(_("No downstream related images found in") + f"\n{other_base_dir}")
        return None
    app_actions.toast(toast_text)
    return downstream


def next_downstream_related_image(
    image_path: str, other_base_dir: str, app_actions
) -> str | None:
    global _downstream_index
    downstream = get_downstream_related_images(image_path, other_base_dir, app_actions)
    if downstream is None:
        return None
    if _downstream_index >= len(downstream):
        _downstream_index = 0
    path = downstream[_downstream_index]
    _downstream_index += 1
    return path


def get_sources_with_downstream_in_dir(
    source_paths: list[str],
    other_base_dir: str,
) -> list[str]:
    """Return the subset of source_paths that have at least one downstream image in other_base_dir."""
    from files.file_browser import FileBrowser
    browser = FileBrowser(directory=other_base_dir)
    browser._gather_files()

    # Build reverse lookups from dir Y in one pass — O(|Y|) instead of O(|X|*|Y|).
    # exact_sources:   raw related-path string (path-level match)
    # basename_sources: basename string (loose match, only for len > 10)
    # stem_prefixes:   stem prefix after stripping a generator suffix
    #                  (_[A-Za-z]{1,8}) — catches derivatives whose metadata
    #                  was wiped but whose name still encodes the source stem.
    exact_sources: set[str] = set()
    basename_sources: set[str] = set()
    stem_prefixes: set[tuple[str, str]] = set()  # (prefix, ext)
    for candidate in browser.filepaths:
        stem, ext = os.path.splitext(os.path.basename(candidate))
        m = _VARIANT_SUFFIX_RE_STRICT.match(stem)
        if m:
            stem_prefixes.add((m.group(1), ext.lower()))
        related, _exact = get_related_image_path(candidate, check_extra_directories=None)
        if related is None:
            continue
        exact_sources.add(related)
        b = os.path.basename(related)
        if len(b) > 10:
            basename_sources.add(b)

    results = []
    for p in source_paths:
        if p in exact_sources or os.path.basename(p) in basename_sources:
            results.append(p)
            continue
        stem, ext = os.path.splitext(os.path.basename(p))
        if (stem, ext.lower()) in stem_prefixes:
            results.append(p)
    return results


def get_downstream_files_for_sources(
    source_paths: list[str],
    other_base_dir: str,
) -> list[str]:
    """Return files in other_base_dir that are downstream of any path in source_paths."""
    from files.file_browser import FileBrowser
    # Build lookup sets from dir X in one pass — O(|X|).
    source_path_set: set[str] = set(source_paths)
    source_basename_set: set[str] = set()
    source_stems: set[tuple[str, str]] = set()  # (stem, ext) for variant-suffix match
    for p in source_paths:
        b = os.path.basename(p)
        if len(b) > 10:
            source_basename_set.add(b)
        stem, ext = os.path.splitext(b)
        source_stems.add((stem, ext.lower()))

    # Scan dir Y once — O(|Y|).
    browser = FileBrowser(directory=other_base_dir)
    browser._gather_files()

    results = []
    for candidate in browser.filepaths:
        related, _exact = get_related_image_path(candidate, check_extra_directories=None)
        if related is not None:
            if related in source_path_set:
                results.append(candidate)
                continue
            b = os.path.basename(related)
            if len(b) > 10 and b in source_basename_set:
                results.append(candidate)
                continue
        stem, ext = os.path.splitext(os.path.basename(candidate))
        m = _VARIANT_SUFFIX_RE_STRICT.match(stem)
        if m and (m.group(1), ext.lower()) in source_stems:
            results.append(candidate)
    return results


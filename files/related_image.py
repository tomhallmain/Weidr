import glob
import os
import re
import time
from enum import Enum
from typing import Callable

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


def _stem_matches_any_suffix(stem: str, suffixes: list) -> bool:
    """Return True if *stem* ends with (a truncation of) any entry in *suffixes*.

    Matching rules (all case-insensitive):
    - Leading separators (``_``, space) in the configured suffix are stripped
      before comparison, so ``_cherry`` and ``cherry`` are equivalent specs.
    - A non-alphanumeric character must immediately precede the matched portion
      in the stem, but the exact form of the separator is unrestricted: ``__cher``
      matches ``_cherry`` (double-underscore separator, truncated word).
    - Right-side truncation: ``_cher`` and ``_che`` both match ``_cherry``.
    - Trailing variant markers are stripped before matching so that
      ``_cherry_2``, ``_cherry 2``, and ``_cherry2`` all match ``_cherry``.
    """
    s = stem.lower()
    # Build a variant-stripped copy: remove optional (_, space) then digits at end.
    s_base = re.sub(r"[_ ]*\d+$", "", s) if s and s[-1].isdigit() else s
    candidates = (s_base, s) if s_base != s else (s,)

    for sf in suffixes:
        sf_core = sf.lower().lstrip("_ ")
        if not sf_core:
            continue
        for k in range(len(sf_core), 0, -1):
            prefix = sf_core[:k]
            for candidate in candidates:
                if candidate.endswith(prefix):
                    pos = len(candidate) - len(prefix) - 1
                    # Accept if the preceding character is non-alphanumeric,
                    # or there is no preceding character (prefix fills the stem).
                    if pos < 0 or not candidate[pos].isalnum():
                        return True
    return False


def suffix_is_numeric(suffix: str) -> bool:
    """Return True if *suffix* is purely numeric after stripping leading separators.

    Applies to raw suffix strings (e.g. ``"_1280"``, ``"001"``).  A numeric
    suffix is never a meaningful category marker.
    """
    core = suffix.lstrip("_ ")
    return bool(core) and core.isdigit()


def _stem_suffix_is_numeric(stem: str) -> bool:
    """Return True if the entire suffix portion of *stem* is only digits (and separators).

    Uses ``extract_filename_base_stem`` to find where the base ends, then
    delegates to ``suffix_is_numeric``.  ``_apple_1`` returns False (contains
    alpha); ``_1280`` returns True.
    """
    base = extract_filename_base_stem(stem)
    if not base or len(base) >= len(stem):
        return False
    return suffix_is_numeric(stem[len(base):])


def _matches_any_valid_suffix(stem: str, suffixes: list) -> bool:
    """Return True if *stem* ends with a known suffix OR has a numeric-only tail.

    A purely numeric tail (e.g. ``_1280``) is never a meaningful suffix and
    should not be flagged as unrecognised.  Use this instead of
    ``_stem_matches_any_suffix`` wherever returning False would cause the stem
    to be reported as having an unknown suffix.
    """
    return _stem_matches_any_suffix(stem, suffixes) or _stem_suffix_is_numeric(stem)


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


# Cache keyed by search_dir storing (filepath, related_path_or_None) for every
# file in that directory. Amortises the filesystem scan and metadata reads across
# all images prevalidated against the same directory.  Cleared by
# clear_generate_gate_cache() after any file is written into a search directory.
_generate_gate_dir_cache: dict[str, list[tuple[str, str | None]]] = {}


def clear_generate_gate_cache(search_dir: str | None = None) -> None:
    """Invalidate the generate-gate directory cache.

    Call with the directory path after generating a file there so subsequent
    gate checks pick up the new file.  Call with no argument to clear all entries.
    """
    if search_dir is None:
        _generate_gate_dir_cache.clear()
    else:
        _generate_gate_dir_cache.pop(search_dir, None)


def _scan_dir_cached(directory: str) -> list[tuple[str, str | None]]:
    """Return (filepath, related_path_or_None) for every file in directory.

    Results are cached per directory path.  Call clear_generate_gate_cache()
    after any write to ensure subsequent calls reflect the new state.
    """
    if directory not in _generate_gate_dir_cache:
        from files.file_browser import FileBrowser
        browser = FileBrowser(directory=directory)
        browser._gather_files()
        _generate_gate_dir_cache[directory] = [
            (fp, get_related_image_path(fp, check_extra_directories=None)[0])
            for fp in browser.filepaths
        ]
    return _generate_gate_dir_cache[directory]


def should_run_generate_action(
    image_path: str,
    edit_suffix: str,
    search_dir: str,
    count_threshold: int = 1,
) -> bool:
    """
    Return True when a generate action should fire for image_path.

    Returns False immediately if image_path is itself a downstream image of another file
    that is present in search_dir. A related-image pointer to a source outside search_dir
    does not block generation -- only downstream-ness relative to the current directory
    matters. Otherwise counts how many downstream images of image_path in search_dir have
    a stem ending with edit_suffix, edit_suffix followed directly by an integer, or
    edit_suffix followed by an underscore separator and an integer (e.g. "_edit",
    "_edit1", "_edit2", "_edit_2"). Returns True if that count is below count_threshold (including
    zero), False if it meets or exceeds the threshold.

    Directory listings and related-image metadata are cached per search_dir to avoid
    re-scanning the directory for every image in a prevalidation batch.
    """
    dir_entries = _scan_dir_cached(search_dir)

    related_path, _ = get_related_image_path(image_path, check_extra_directories=None)
    if related_path is not None:
        related_basename = os.path.basename(related_path)
        source_in_search_dir = any(
            fp == related_path
            or (len(related_basename) > 10 and os.path.basename(fp) == related_basename)
            for fp, _r in dir_entries
        )
        if source_in_search_dir:
            return False

    source_basename = os.path.basename(image_path)
    source_stem, source_ext = os.path.splitext(source_basename)

    downstream_stems: list[str] = []
    for fp, related in dir_entries:
        if related is not None:
            if related == image_path:
                downstream_stems.append(os.path.splitext(os.path.basename(fp))[0])
                continue
            if len(os.path.basename(related)) > 10 and os.path.basename(related) == source_basename:
                downstream_stems.append(os.path.splitext(os.path.basename(fp))[0])
                continue
        fp_stem, fp_ext = os.path.splitext(os.path.basename(fp))
        # Filename-pattern fallback when metadata is absent.  Use a direct prefix
        # break (not _VARIANT_SUFFIX_RE_STRICT) so double-underscore / truncated
        # generator suffixes (e.g. __appl, __cher_2) are collected; edit_suffix
        # filtering happens below via _matches_any_valid_suffix.
        if (
            fp_ext.lower() == source_ext.lower()
            and len(fp_stem) > len(source_stem)
            and fp_stem[: len(source_stem)].lower() == source_stem.lower()
            and not fp_stem[len(source_stem)].isalnum()
        ):
            downstream_stems.append(fp_stem)

    suffix_count = sum(1 for stem in downstream_stems if _matches_any_valid_suffix(stem, [edit_suffix]))
    return suffix_count < count_threshold


def get_image_edit_redo_params(
    image_path: str,
) -> tuple[str | None, str | None]:
    """Return (related_path, edit_suffix) if image_path can be redone as an image edit.

    Eligible when:
    - The image has a related image reference in its metadata (path need not exist
      locally — a sufficiently unique basename is likely the correct source), OR
      the base stem of the filename matches exactly one file in the same directory
      whose stem is exactly that base stem (no appended suffix).
    - The image's stem ends with a non-numeric variant suffix (_VARIANT_SUFFIX_RE).

    The suffix is the tail of the current stem beyond the related image's stem
    (e.g. source ``img.png``, edit ``img_edit.png`` → suffix ``_edit``).
    Returns (None, None) when either condition is not met.
    """
    related_path, _exact = get_related_image_path(image_path)

    if related_path is None:
        base_stem = extract_filename_base_stem(image_path)
        if base_stem:
            same_dir = os.path.dirname(os.path.abspath(image_path))
            matches = find_files_by_base_stem([same_dir], base_stem)
            source_matches = [
                f for f in matches
                if os.path.splitext(os.path.basename(f))[0] == base_stem
                and f != image_path
            ]
            if len(source_matches) == 1:
                related_path = source_matches[0]

    if related_path is None:
        return None, None

    stem = os.path.splitext(os.path.basename(image_path))[0]
    if not _VARIANT_SUFFIX_RE.match(stem):
        return None, None

    related_stem = os.path.splitext(os.path.basename(related_path))[0]
    if stem.lower().startswith(related_stem.lower()):
        edit_suffix = stem[len(related_stem):]
    else:
        m = _VARIANT_SUFFIX_RE.match(stem)
        edit_suffix = stem[len(m.group(1)):]

    if suffix_is_numeric(edit_suffix):
        return None, None

    return related_path, edit_suffix


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


# ---------------------------------------------------------------------------
# Filename base stem matching
# ---------------------------------------------------------------------------

class _CharCategory(Enum):
    ALPHA = "alpha"
    DIGIT = "digit"
    OTHER = "other"


def _get_char_category(char: str) -> _CharCategory:
    if char.isalpha():
        return _CharCategory.ALPHA
    if char.isdigit():
        return _CharCategory.DIGIT
    return _CharCategory.OTHER


def extract_filename_base_stem(filename: str) -> str | None:
    """
    Extract the base stem from filename using common delimiter heuristics.

    Example: "SDWebUI_17602175357792320_0_s.png" -> "SDWebUI_17602175357792320"

    Works for any file type, not just images.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]

    delimiter_pattern = r"([_\s\-\.]+)"
    parts = re.split(delimiter_pattern, basename)

    if len(parts) == 0:
        return basename

    if len(parts) == 1:
        single_part = parts[0]
        if len(single_part) >= 8 and any(c.isalnum() for c in single_part):
            return single_part
        return basename

    base_stem = parts[0]
    first_part_len = len(parts[0])

    if first_part_len >= 30:
        return base_stem

    if len(parts) < 3:
        if len(base_stem) >= 3 and any(c.isalnum() for c in base_stem):
            return base_stem
        return basename

    delimiter = parts[1]
    second_part = parts[2]

    if not second_part or len(second_part) <= 4:
        if len(base_stem) >= 3 and any(c.isalnum() for c in base_stem):
            return base_stem
        return basename

    base_stem = base_stem + delimiter + second_part

    if len(base_stem) >= 10:
        return base_stem

    for i in range(3, len(parts), 2):
        if i + 1 < len(parts):
            delim = parts[i]
            next_part = parts[i + 1]

            if not next_part or len(next_part) <= 4:
                break

            candidate = base_stem + delim + next_part
            if len(candidate) >= 3 and any(c.isalnum() for c in candidate):
                base_stem = candidate
                if len(base_stem) >= 10:
                    break
            else:
                break
        else:
            break

    if len(base_stem) >= 3 and any(c.isalnum() for c in base_stem):
        return base_stem
    return basename


def _check_base_stem_boundary(
    filename_no_ext: str, base_stem: str, last_category: _CharCategory
) -> bool:
    """Return True if filename_no_ext is a valid base stem match for base_stem."""
    if not filename_no_ext.startswith(base_stem):
        return False
    if len(filename_no_ext) == len(base_stem):
        return True
    next_char = filename_no_ext[len(base_stem)]
    if next_char in ("_", " ", "-", "."):
        return True
    return last_category != _get_char_category(next_char)


# Per-directory scan cache used by find_files_by_base_stem(use_cache=True).
# Each entry is (entries, cached_at) where entries is a list of
# (filepath, stem-without-extension) for every file found via os.walk.
# Entries expire after _BASE_STEM_CACHE_TTL seconds.
_BASE_STEM_CACHE_TTL: float = 300.0  # 5 minutes
_base_stem_dir_cache: dict[str, tuple[list[tuple[str, str]], float]] = {}
# Results cache: (directory, base_stem) → matching file paths. Keyed the same
# way as _base_stem_dir_cache so it can be invalidated in lockstep.
_base_stem_results_cache: dict[tuple[str, str], list[str]] = {}


def clear_base_stem_dir_cache(search_dir: str | None = None) -> None:
    """Invalidate the base stem directory scan cache.

    Pass a directory path to evict just that entry, or no argument to clear
    everything. Call this after writing a file into a previously-scanned
    directory so the next find_files_by_base_stem(use_cache=True) call picks
    up the new entry.
    """
    if search_dir is None:
        _base_stem_dir_cache.clear()
        _base_stem_results_cache.clear()
    else:
        _base_stem_dir_cache.pop(search_dir, None)
        # Evict all results entries for this directory.
        for key in [k for k in _base_stem_results_cache if k[0] == search_dir]:
            del _base_stem_results_cache[key]


def find_files_by_base_stem(
    directories: list[str],
    base_stem: str,
    threshold: int = 400_000,
    on_threshold_exceeded: Callable[[str, int], bool] | None = None,
    use_cache: bool = False,
) -> list[str]:
    """
    Walk each directory in `directories` and return files whose stem starts
    with `base_stem` (with appropriate delimiter / character-category boundary
    check). Results are sorted by basename, lower-cased.

    When use_cache=False (default) each call performs a fresh os.walk().
    If the total file count within a directory exceeds `threshold`,
    `on_threshold_exceeded(directory, file_count)` is called. Returning False
    aborts and returns []. If `on_threshold_exceeded` is None the threshold is
    silently passed and the walk continues.

    When use_cache=True a per-directory listing of (filepath, stem) pairs is
    built on the first cold walk and reused on subsequent calls. The threshold
    is still honoured for user-confirmation callbacks; when no callback is
    provided the walk always completes so the cache is fully populated. Call
    clear_base_stem_dir_cache() after writing files into cached directories.
    Intended for batch callers (pipeline, ClassifierAction) that search the
    same directories many times within a single run.

    Works for any file type, not just images.
    """
    last_category = (
        _get_char_category(base_stem[-1]) if base_stem else _CharCategory.OTHER
    )
    matching: list[str] = []

    for directory in directories:
        if use_cache:
            results_key = (directory, base_stem)
            cached = _base_stem_dir_cache.get(directory)
            dir_expired = cached is None or (time.time() - cached[1]) > _BASE_STEM_CACHE_TTL
            if not dir_expired and results_key in _base_stem_results_cache:
                matching.extend(_base_stem_results_cache[results_key])
                continue
            if dir_expired:
                entries: list[tuple[str, str]] = []
                file_count = 0
                threshold_cleared = False
                try:
                    for root_dir, _dirs, files in os.walk(directory):
                        for fname in files:
                            file_count += 1
                            if not threshold_cleared and file_count > threshold:
                                if on_threshold_exceeded is not None:
                                    if not on_threshold_exceeded(directory, file_count):
                                        return []
                                threshold_cleared = True
                            entries.append((
                                os.path.join(root_dir, fname),
                                os.path.splitext(fname)[0],
                            ))
                except Exception:
                    logger.exception(
                        "Error scanning %r for base stem cache", directory
                    )
                _base_stem_dir_cache[directory] = (entries, time.time())
                scan_entries = entries
            else:
                scan_entries = cached[0]

            dir_matches: list[str] = []
            for filepath, filename_no_ext in scan_entries:
                if _check_base_stem_boundary(filename_no_ext, base_stem, last_category):
                    dir_matches.append(filepath)
            _base_stem_results_cache[results_key] = dir_matches
            matching.extend(dir_matches)
        else:
            file_count = 0
            threshold_cleared = False
            try:
                for root_dir, _dirs, files in os.walk(directory):
                    for fname in files:
                        file_count += 1

                        if not threshold_cleared and file_count > threshold:
                            if on_threshold_exceeded is not None:
                                if not on_threshold_exceeded(directory, file_count):
                                    return []
                                threshold_cleared = True
                            else:
                                return matching

                        filename_no_ext = os.path.splitext(fname)[0]
                        if _check_base_stem_boundary(
                            filename_no_ext, base_stem, last_category
                        ):
                            matching.append(os.path.join(root_dir, fname))
            except Exception:
                logger.exception(
                    "Error searching %r for base stem %r", directory, base_stem
                )

    matching.sort(key=lambda x: os.path.basename(x).lower())
    return matching


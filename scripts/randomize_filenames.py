#!/usr/bin/env python3
"""
Randomize media filenames under a directory tree while preserving extensions and paths.

Behavior overview:
- Only media files are considered (images, GIF, video, audio, PDF, SVG, etc.).
- Renames are globally unique across:
  - files currently on disk,
  - names assigned earlier in the same run,
  - previously assigned names in file_mapping_cache.json.
- Existing cache mappings are preserved and merged with the new run.
- If a media file's path (relative to the file root) matches a cached subdirectory mapping
  entry (that subdirectory key plus original filename), it is moved to <root>/review instead
  of being renamed. Basename-only matching is not used.
- Default mode is dry run; use --execute to apply filesystem changes.

Examples:
1) Dry run using default cache path (<dir>/file_mapping_cache.json):
   python randomize_files.py "C:\\path\\to\\files"

2) Execute changes:
   python randomize_files.py "C:\\path\\to\\files" --execute

3) Dry run with exclusions:
   python randomize_files.py "C:\\path\\to\\files" --exclude "*.tmp" --exclude "README*"

4) Execute with verbose logging:
   python randomize_files.py "C:\\path\\to\\files" --execute --verbose

5) Use a custom cache file path:
   python randomize_filenames.py "C:\\path\\to\\files" --output-json "D:\\maps\\cache.json"

6) Defaults from <dir>/randomize_filenames_config.json (CLI flags override):
   {
     "verbose": true,
     "exclude": ["*.tmp"],
     "log_file": "randomize_filenames.log",
     "log_append": true
   }

Dry runs do not write planned renames to the cache. The only exception is pruning invalid keys
listed in <cache-dir>/files_to_remove_from_mapping.json (a JSON array of basenames and/or full
paths under the file root — absolute Windows paths are accepted).
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import logging
import os
import re
import secrets
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# 32 bytes -> 64 hex chars; vanishing collision risk vs any existing or assigned name.
_RANDOM_ID_BYTES = 32
_CACHE_FILENAME = "file_mapping_cache.json"
_CONFIG_FILENAME = "randomize_filenames_config.json"
_REMOVAL_LIST_FILENAME = "files_to_remove_from_mapping.json"
_KNOWN_CONFIG_KEYS = frozenset(
    {"execute", "verbose", "exclude", "output_json", "log_file", "log_append"}
)
# argparse dest -> CLI flags that override the config file when present on argv
_CONFIG_OVERRIDE_FLAGS: dict[str, tuple[str, ...]] = {
    "execute": ("--execute", "--dry-run"),
    "verbose": ("-v", "--verbose"),
    "exclude": ("--exclude",),
    "output_json": ("-o", "--output-json"),
    "log_file": ("--log-file",),
    "log_append": ("--log-append",),
}
_RANDOM_BASENAME_RE = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_EXTENSIONS = {
    ".3g2",
    ".3gp",
    ".avif",
    ".aac",
    ".aiff",
    ".alac",
    ".amr",
    ".ape",
    ".apng",
    ".arw",
    ".asf",
    ".au",
    ".avi",
    ".bmp",
    ".cr2",
    ".dng",
    ".f4v",
    ".flac",
    ".flv",
    ".gif",
    ".heic",
    ".heif",
    ".ico",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".m4v",
    ".m4a",
    ".m4b",
    ".m2ts",
    ".m2v",
    ".mid",
    ".midi",
    ".mkv",
    ".mov",
    ".mp1",
    ".mp2",
    ".mp3",
    ".mp4",
    ".mts",
    ".nef",
    ".oga",
    ".ogg",
    ".ogv",
    ".opus",
    ".orf",
    ".mpeg",
    ".mpg",
    ".pdf",
    ".pjp",
    ".pjpeg",
    ".png",
    ".psd",
    ".qt",
    ".ra",
    ".raw",
    ".rm",
    ".rmvb",
    ".rw2",
    ".svg",
    ".ts",
    ".tif",
    ".tiff",
    ".vob",
    ".wav",
    ".weba",
    ".webm",
    ".webp",
    ".wma",
    ".wmv",
    ".wv",
    ".y4m",
}


def _argv_mentions_flag(argv: list[str], flags: tuple[str, ...]) -> bool:
    """Return True if any of *flags* appear on *argv* (including --flag=value)."""
    flag_set = set(flags)
    i = 0
    while i < len(argv):
        tok = argv[i]
        base = tok.split("=", 1)[0]
        if base in flag_set:
            return True
        i += 1
    return False


def _resolve_config_path(value: str | Path, root: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p)


def load_config_defaults(config_path: Path, root: Path) -> dict:
    """
    Parse ``randomize_filenames_config.json`` into :func:`argparse` default kwargs.

    Relative ``output_json`` and ``log_file`` paths are resolved under *root*.
    """
    with config_path.open(encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config root must be a JSON object")

    unknown = set(data.keys()) - _KNOWN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}")

    out: dict = {}
    if "execute" in data:
        out["execute"] = bool(data["execute"])
    if "verbose" in data:
        out["verbose"] = bool(data["verbose"])
    if "exclude" in data:
        raw_exclude = data["exclude"]
        if isinstance(raw_exclude, str):
            raw_exclude = [raw_exclude]
        if not isinstance(raw_exclude, list) or not all(isinstance(x, str) for x in raw_exclude):
            raise ValueError('"exclude" must be a string or list of strings')
        out["exclude"] = list(raw_exclude)
    if data.get("output_json"):
        out["output_json"] = _resolve_config_path(data["output_json"], root)
    if data.get("log_file"):
        out["log_file"] = _resolve_config_path(data["log_file"], root)
    if "log_append" in data:
        out["log_append"] = bool(data["log_append"])
    return out


def setup_logging(
    verbose: bool,
    log_file: Path | None = None,
    *,
    log_append: bool = False,
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(levelname)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    if log_file is not None:
        log_file = log_file.resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_file, mode="a" if log_append else "w", encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)


def should_exclude(name: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def should_exclude_dir(path: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        for path_part in re.split(r"[/\\]", path):
            if fnmatch.fnmatch(path_part, pat):
                return True
    return False


def is_media_file(path: Path) -> bool:
    return path.suffix.lower() in _MEDIA_EXTENSIONS


def is_already_randomized_name(path: Path) -> bool:
    return _RANDOM_BASENAME_RE.fullmatch(path.stem) is not None


def random_basename(extension: str) -> str:
    """extension should include leading dot if non-empty, e.g. '.png' or ''."""
    core = secrets.token_hex(_RANDOM_ID_BYTES)
    return f"{core}{extension}"


def collect_files(root: Path, exclude_patterns: list[str]) -> list[Path]:
    """Media files under root to rename, respecting exclusions (basename matched only)."""
    out: list[Path] = []
    review_dir_name = "review"
    root_resolved = root.resolve()
    for dirpath, _dirnames, filenames in os.walk(root):
        dirpath_obj = Path(dirpath)
        if dirpath_obj.resolve() == (root_resolved / review_dir_name):
            logging.debug("Skip directory: %s", dirpath_obj)
            continue
        if should_exclude_dir(os.path.normpath(dirpath_obj.resolve()), exclude_patterns):
            logging.info("Exclude directory: %s", dirpath_obj)
            continue
        for fn in filenames:
            if should_exclude(fn, exclude_patterns):
                logging.debug("Skip (excluded pattern): %s", Path(dirpath) / fn)
                continue
            full_path = Path(dirpath) / fn
            if not is_media_file(full_path):
                logging.debug("Skip (non-media file): %s", full_path)
                continue
            if is_already_randomized_name(full_path):
                logging.debug("Skip (already randomized name): %s", full_path)
                continue
            out.append(full_path)
    return sorted(out)


def collect_all_files(root: Path) -> list[Path]:
    """Every file under root (for global basename collision checks)."""
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            out.append(Path(dirpath) / fn)
    return out


def load_cache_payload(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Could not load existing cache %s: %s", cache_path, e)
        return {}
    if not isinstance(data, dict):
        logging.warning("Ignoring invalid cache format in %s (expected object).", cache_path)
        return {}
    return data


def load_keys_to_remove(removal_path: Path) -> list[str]:
    if not removal_path.exists():
        logging.debug("No removal list at %s (optional).", removal_path)
        return []
    try:
        with removal_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Could not load removal list %s: %s", removal_path, e)
        return []
    if not isinstance(data, list):
        logging.warning("Ignoring invalid removal list in %s (expected JSON array).", removal_path)
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            out.append(item)
    logging.info("Loaded %s removal entr(y/ies) from %s.", len(out), removal_path)
    return out


def sort_cache_payload(payload: dict) -> dict:
    """Return a copy with root_files / subdirectories sorted like merge_cache_mappings."""
    p = copy.deepcopy(payload)
    rf = p.get("root_files")
    if isinstance(rf, dict):
        p["root_files"] = dict(sorted(rf.items(), key=lambda kv: kv[0].lower()))
    sd = p.get("subdirectories")
    if isinstance(sd, dict):
        p["subdirectories"] = dict(
            sorted(
                (
                    subdir,
                    dict(sorted(name_map.items(), key=lambda kv: kv[0].lower())),
                )
                for subdir, name_map in sd.items()
                if isinstance(name_map, dict)
            )
        )
    return p


def _match_subdir_cache_key(subdirs: dict, want: str) -> str | None:
    """Return the actual key in subdirs whose relative path equals want (case-insensitive on Windows)."""
    wn = os.path.normcase(want.replace("\\", "/").strip("/"))
    for k in subdirs:
        if not isinstance(k, str):
            continue
        kn = os.path.normcase(k.replace("\\", "/").strip("/"))
        if kn == wn:
            return k
    return None


def _match_mapping_source_key(name_map: dict, want: str) -> str | None:
    """Return the actual source-name key matching want (case-insensitive on Windows)."""
    wn = os.path.normcase(want)
    for k in name_map:
        if isinstance(k, str) and os.path.normcase(k) == wn:
            return k
    return None


def parse_removal_entry(raw: str, root: Path) -> tuple[str, str, str] | None:
    """
    Map one removal line to (kind, filename, subdir_key).

    kind is:
    - "all": basename-only; remove filename from root_files and every subdirectories map;
    - "root": file lives directly under root (cache root_files only);
    - "subdir": remove subdirectories[subdir_key][filename] (subdir_key uses forward slashes).

    Accepts absolute Windows paths, paths relative to the file root, or a plain basename.
    """
    raw = raw.strip()
    if not raw:
        return None

    looks_like_path = (
        Path(raw).is_absolute()
        or "/" in raw
        or "\\" in raw
        or (len(raw) >= 2 and raw[1] == ":")
    )
    if not looks_like_path:
        return ("all", raw, "")

    p = Path(raw)
    if not p.is_absolute():
        p = root / raw
    try:
        p = p.resolve()
    except OSError:
        logging.warning("Could not resolve removal path: %s", raw)
        return None

    root_r = root.resolve()
    try:
        rel = p.relative_to(root_r)
    except ValueError:
        logging.warning("Removal path not under file root %s: %s", root_r, raw)
        return None

    if not rel.parts:
        return None

    name = rel.parts[-1]
    if len(rel.parts) == 1:
        return ("root", name, "")
    sub = "/".join(rel.parts[:-1])
    return ("subdir", name, sub)


def apply_key_removals(
    cache: dict, keys_to_remove: list[str], root: Path
) -> tuple[dict, bool]:
    """
    Drop mapping entries listed for removal.

    Entries may be absolute paths under the file root, paths relative to that root, or a plain
    basename (removes that basename from root_files and every subdirectories map).
    """
    if not keys_to_remove:
        return cache, False
    payload = copy.deepcopy(cache)
    rf = payload.get("root_files")
    if not isinstance(rf, dict):
        rf = {}
        payload["root_files"] = rf
    sd = payload.get("subdirectories")
    if not isinstance(sd, dict):
        sd = {}
        payload["subdirectories"] = sd

    changed = False
    for raw in keys_to_remove:
        spec = parse_removal_entry(raw, root)
        if spec is None:
            continue
        kind, name, sub_key = spec
        if kind == "all":
            rk = _match_mapping_source_key(rf, name)
            if rk is not None:
                del rf[rk]
                changed = True
            for subdir_key in list(sd.keys()):
                sub_map = sd.get(subdir_key)
                if not isinstance(sub_map, dict):
                    continue
                nk = _match_mapping_source_key(sub_map, name)
                if nk is not None:
                    del sub_map[nk]
                    changed = True
                    if not sub_map:
                        del sd[subdir_key]
        elif kind == "root":
            rk = _match_mapping_source_key(rf, name)
            if rk is not None:
                del rf[rk]
                changed = True
        else:
            actual = _match_subdir_cache_key(sd, sub_key)
            if actual is None:
                logging.debug(
                    "Removal subdir not in cache (nothing to delete): %s (wanted %r)",
                    raw,
                    sub_key,
                )
                continue
            sub_map = sd.get(actual)
            if isinstance(sub_map, dict):
                nk = _match_mapping_source_key(sub_map, name)
                if nk is not None:
                    del sub_map[nk]
                    changed = True
                    if not sub_map:
                        del sd[actual]

    return sort_cache_payload(payload), changed


def extract_cached_target_names(cache_payload: dict) -> set[str]:
    names: set[str] = set()

    root_files = cache_payload.get("root_files")
    if isinstance(root_files, dict):
        for v in root_files.values():
            if isinstance(v, str):
                names.add(v)

    subdirs = cache_payload.get("subdirectories")
    if isinstance(subdirs, dict):
        for mapping in subdirs.values():
            if not isinstance(mapping, dict):
                continue
            for v in mapping.values():
                if isinstance(v, str):
                    names.add(v)

    return names


def _norm_cache_rel_key(fragment: str) -> str:
    """Normalize a subdirectory key from the cache for comparison (case on Windows)."""
    s = fragment.replace("\\", "/").strip().strip("/")
    return os.path.normcase(s)


def _parent_rel_key_under_root(root: Path, parent: Path) -> str:
    """Relative path from root to parent using forward slashes; empty string if parent is root."""
    try:
        rel = parent.resolve().relative_to(root.resolve())
    except ValueError:
        return ""
    if rel == Path(".") or not rel.parts:
        return ""
    return str(rel).replace("\\", "/")


def extract_cached_subdirectory_review_pairs(cache_payload: dict) -> set[tuple[str, str]]:
    """
    Pairs (normalized subdir key, normalized basename) for files that should go to review.
    Matches the cache layout: subdirectories[rel_path][original_basename].
    """
    pairs: set[tuple[str, str]] = set()
    subdirs = cache_payload.get("subdirectories")
    if not isinstance(subdirs, dict):
        return pairs
    for subdir_key, mapping in subdirs.items():
        if not isinstance(subdir_key, str) or not isinstance(mapping, dict):
            continue
        sk = _norm_cache_rel_key(subdir_key)
        for source_name in mapping:
            if isinstance(source_name, str):
                pairs.add((sk, os.path.normcase(source_name)))
    return pairs


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _next_available_destination(
    dst_dir: Path,
    requested_name: str,
    reserved: set[Path],
) -> Path:
    base = Path(requested_name).stem
    ext = Path(requested_name).suffix
    candidate = dst_dir / requested_name
    idx = 1
    while candidate.exists() or candidate in reserved:
        candidate = dst_dir / f"{base}_{idx}{ext}"
        idx += 1
    reserved.add(candidate)
    return candidate


def plan_review_moves(review_files: list[Path], review_dir: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    reserved: set[Path] = set()
    for src in sorted(review_files, key=lambda p: str(p).lower()):
        dst = _next_available_destination(review_dir, src.name, reserved)
        moves.append((src, dst))
    return moves


def merge_cache_mappings(
    root: Path,
    existing_cache_payload: dict,
    new_per_dir_mappings: dict[Path, dict[str, str]],
) -> dict:
    """Preserve old cache entries and overlay new mappings for this run."""
    merged = build_json_structure(root, new_per_dir_mappings)

    existing_root_files = existing_cache_payload.get("root_files")
    if isinstance(existing_root_files, dict):
        for old_name, new_name in existing_root_files.items():
            if isinstance(old_name, str) and isinstance(new_name, str):
                merged["root_files"].setdefault(old_name, new_name)

    existing_subdirs = existing_cache_payload.get("subdirectories")
    if isinstance(existing_subdirs, dict):
        for subdir_key, mapping in existing_subdirs.items():
            if not isinstance(subdir_key, str) or not isinstance(mapping, dict):
                continue
            target_map = merged["subdirectories"].setdefault(subdir_key, {})
            for old_name, new_name in mapping.items():
                if isinstance(old_name, str) and isinstance(new_name, str):
                    target_map.setdefault(old_name, new_name)

    merged["root_files"] = dict(
        sorted(merged["root_files"].items(), key=lambda kv: kv[0].lower())
    )
    merged["subdirectories"] = dict(
        sorted(
            (
                subdir,
                dict(sorted(name_map.items(), key=lambda kv: kv[0].lower())),
            )
            for subdir, name_map in merged["subdirectories"].items()
        )
    )
    return merged


def plan_global_renames(
    batch: list[Path],
    all_files: list[Path],
    occupied_extra: set[str] | None = None,
) -> tuple[list[tuple[Path, Path]], dict[Path, dict[str, str]]] | tuple[None, None]:
    """
    Assign a new basename per file in batch such that:
    - each new basename is unique across the whole tree (among new names and vs existing files);
    - extensions preserved on each path.

    Single-phase rename is then safe: new names never match any current on-disk basename.
    """
    if not batch:
        return [], {}

    occupied = {p.name for p in all_files}
    if occupied_extra:
        occupied.update(occupied_extra)
    assigned: set[str] = set()
    pairs: list[tuple[Path, Path]] = []
    per_dir: dict[Path, dict[str, str]] = defaultdict(dict)

    max_per_file = 50_000
    for p in sorted(batch, key=lambda x: (str(x.parent).lower(), x.name.lower())):
        ext = p.suffix
        for _ in range(max_per_file):
            candidate = random_basename(ext)
            if candidate in occupied or candidate in assigned:
                continue
            assigned.add(candidate)
            new_path = p.parent / candidate
            pairs.append((p, new_path))
            per_dir[p.parent][p.name] = candidate
            break
        else:
            logging.error(
                "Could not assign a globally unique name for %s after %s attempts",
                p,
                max_per_file,
            )
            return None, None

    return pairs, dict(per_dir)


def build_json_structure(
    root: Path,
    per_dir_mappings: dict[Path, dict[str, str]],
) -> dict:
    """Split root-level vs subdirs; sort by original filename within each."""
    root_resolved = root.resolve()
    root_files: dict[str, str] = {}
    subdirs: dict[str, dict[str, str]] = {}

    for parent, name_map in sorted(per_dir_mappings.items(), key=lambda x: str(x[0])):
        sorted_map = dict(sorted(name_map.items(), key=lambda kv: kv[0].lower()))
        if parent.resolve() == root_resolved:
            root_files.update(sorted_map)
        else:
            try:
                rel = parent.relative_to(root_resolved)
            except ValueError:
                rel = parent
            key = str(rel).replace("\\", "/")
            subdirs[key] = sorted_map

    return {
        "root_directory": str(root_resolved),
        "root_files": dict(sorted(root_files.items(), key=lambda kv: kv[0].lower())),
        "subdirectories": dict(sorted(subdirs.items(), key=lambda kv: kv[0].lower())),
    }


def try_rename_file(src: Path, dst: Path, *, verbose: bool = False) -> bool:
    """
    Rename src to dst in one step. Returns True on success.
    Logs and returns False on OSError (permissions, existing target, I/O errors, etc.).
    """
    try:
        src.rename(dst)
    except OSError as e:
        logging.error(
            "Rename failed: %s -> %s (%s: %s)",
            src,
            dst,
            e.__class__.__name__,
            e,
        )
        if verbose:
            logging.debug("Rename traceback", exc_info=True)
        return False
    logging.info("Rename: %s -> %s", src, dst.name)
    return True


def _build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Randomize filenames under DIR (extensions and paths preserved). "
            "New basenames are unique across the entire tree. "
            "Default is dry run; use --execute to apply."
        )
    )
    p.add_argument(
        "directory",
        type=Path,
        help="Root directory to walk",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform renames (default: dry run only)",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan renames only; overrides \"execute\" in the config file",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Glob pattern for filenames to skip (fnmatch). "
            "May be repeated, e.g. --exclude '*.tmp' --exclude 'README*'"
        ),
    )
    p.add_argument(
        "-o",
        "--output-json",
        type=Path,
        default=None,
        help=f"Path for mapping JSON (default: <directory>/{_CACHE_FILENAME})",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="More detailed logging",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write log output to this file (stderr still used; see --log-append)",
    )
    p.add_argument(
        "--log-append",
        action="store_true",
        help="Append to the log file instead of truncating (config: log_append)",
    )
    return p


def _config_file_candidate(directory: Path) -> Path:
    """Path where ``randomize_filenames_config.json`` is expected for *directory*."""
    expanded = directory.expanduser()
    if expanded.is_dir():
        return (expanded.resolve() / _CONFIG_FILENAME)
    return expanded / _CONFIG_FILENAME


def _apply_config_file_values_to_args(
    args: argparse.Namespace,
    loaded_from_file: dict,
    argv_list: list[str],
) -> None:
    """Apply config-file fields argparse may not pick up (e.g. ``exclude`` append)."""
    if "exclude" in loaded_from_file and not _argv_mentions_flag(argv_list, ("--exclude",)):
        args.exclude = list(loaded_from_file["exclude"])
    if "log_append" in loaded_from_file and not _argv_mentions_flag(
        argv_list, ("--log-append",)
    ):
        args.log_append = bool(loaded_from_file["log_append"])


def _log_config_resolution(args: argparse.Namespace) -> None:
    """Log whether the per-directory JSON config was checked, found, and applied."""
    checked: Path | None = getattr(args, "config_checked_path", None)
    if checked is None:
        logging.info(
            "Config: no directory argument; did not look for %s",
            _CONFIG_FILENAME,
        )
        return

    logging.info("Config: checked for %s", checked)
    used: Path | None = getattr(args, "config_file_used", None)
    if used is None:
        logging.info(
            "Config: file not found; using command-line and built-in defaults only"
        )
        return

    logging.info("Config: loaded %s", used)
    from_file: tuple[str, ...] = getattr(args, "config_keys_from_file", ()) or ()
    if from_file:
        logging.info("Config: keys in file: %s", ", ".join(from_file))
    else:
        logging.info("Config: file contained no recognized option keys")

    overridden: tuple[str, ...] = getattr(args, "config_keys_overridden_by_cli", ()) or ()
    if overridden:
        logging.info(
            "Config: overridden by command line: %s",
            ", ".join(overridden),
        )

    if args.exclude:
        exclude_source = (
            "command line"
            if "exclude" in overridden
            else "config file"
            if "exclude" in from_file
            else "unknown"
        )
        logging.info(
            "Config: active exclude patterns (%s, fnmatch on basenames): %s",
            exclude_source,
            args.exclude,
        )
    else:
        logging.info("Config: no exclude patterns active")

    if args.log_file:
        logging.info(
            "Config: log file %s will be %s",
            args.log_file,
            "appended" if getattr(args, "log_append", False) else "truncated",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    If ``<directory>/randomize_filenames_config.json`` exists, its fields become
    argparse defaults. Any option also passed on the command line overrides the
    config file value.
    """
    argv_list = list(argv if argv is not None else sys.argv[1:])

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("directory", type=Path, nargs="?")
    pre_args, _ = pre.parse_known_args(argv_list)

    config_defaults: dict = {}
    config_path: Path | None = None
    config_checked_path: Path | None = None
    if pre_args.directory is not None:
        root_guess = pre_args.directory.expanduser()
        config_checked_path = _config_file_candidate(root_guess)
        if config_checked_path.is_file():
            config_path = config_checked_path.resolve()
            root_for_paths = root_guess.resolve() if root_guess.is_dir() else root_guess
            try:
                config_defaults = load_config_defaults(config_path, root_for_paths)
            except (OSError, json.JSONDecodeError, ValueError) as e:
                raise SystemExit(f"{config_checked_path}: {e}") from e

    loaded_from_file = dict(config_defaults)
    config_keys_overridden: list[str] = []
    for dest, flags in _CONFIG_OVERRIDE_FLAGS.items():
        if dest in config_defaults and _argv_mentions_flag(argv_list, flags):
            del config_defaults[dest]
            config_keys_overridden.append(dest)

    p = _build_argument_parser()
    p.set_defaults(**config_defaults)
    args = p.parse_args(argv_list)
    _apply_config_file_values_to_args(args, loaded_from_file, argv_list)
    if getattr(args, "dry_run", False):
        args.execute = False
    args.config_checked_path = config_checked_path
    args.config_file_used = config_path
    args.config_keys_from_file = tuple(sorted(loaded_from_file.keys()))
    args.config_keys_overridden_by_cli = tuple(sorted(config_keys_overridden))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(
        args.verbose,
        args.log_file,
        log_append=bool(getattr(args, "log_append", False)),
    )
    _log_config_resolution(args)

    root = args.directory
    if not root.is_dir():
        logging.error("Not a directory: %s", root)
        return 1

    root = root.resolve()
    exclude = list(args.exclude)

    out_path = args.output_json or (root / _CACHE_FILENAME)
    removal_path = out_path.parent / _REMOVAL_LIST_FILENAME
    raw_cache = load_cache_payload(out_path)
    keys_to_remove = load_keys_to_remove(removal_path)
    existing_cache_payload, cache_modified_by_removals = apply_key_removals(
        raw_cache, keys_to_remove, root
    )
    if cache_modified_by_removals:
        logging.info(
            "Removed invalid cache key(s) per %s (see %s).",
            removal_path.name,
            removal_path,
        )
    elif keys_to_remove:
        logging.warning(
            "None of the %s removal entr(y/ies) changed %s. "
            "Use a basename to drop that name from all maps, or a full path under %s "
            "(absolute or relative to the file root).",
            len(keys_to_remove),
            out_path,
            root,
        )
    cached_target_names = extract_cached_target_names(existing_cache_payload)
    cached_subdir_review_pairs = extract_cached_subdirectory_review_pairs(
        existing_cache_payload
    )
    if cached_subdir_review_pairs:
        logging.info(
            "Review rule: %s cached subdirectory path + filename pair(s) "
            "(files must match both path under root and name).",
            len(cached_subdir_review_pairs),
        )
    review_dir = root / "review"
    review_dir_norm = _normalized_path(review_dir)

    all_files = collect_all_files(root)
    batch = collect_files(root, exclude)
    rename_batch: list[Path] = []
    review_batch: list[Path] = []
    for p in batch:
        rel_parent_key = _norm_cache_rel_key(_parent_rel_key_under_root(root, p.parent))
        review_key = (rel_parent_key, os.path.normcase(p.name))
        if review_key in cached_subdir_review_pairs:
            if _normalized_path(p.parent) == review_dir_norm:
                logging.debug(
                    "Conflict filename already under review dir, skipping: %s",
                    p,
                )
            else:
                review_batch.append(p)
            continue
        rename_batch.append(p)

    review_moves = plan_review_moves(review_batch, review_dir)
    logging.info(
        (
            "Found %s file(s) on disk; %s media file(s) selected under %s "
            "(%s planned rename(s), %s planned review move(s))"
        ),
        len(all_files),
        len(batch),
        root,
        len(rename_batch),
        len(review_moves),
    )

    if cached_target_names:
        logging.info(
            "Loaded %s existing cached target name(s) for uniqueness checks.",
            len(cached_target_names),
        )

    pairs, per_dir_mappings = plan_global_renames(
        rename_batch,
        all_files,
        occupied_extra=cached_target_names,
    )
    if pairs is None:
        return 1

    if pairs:
        logging.info("Planned %s rename(s) with globally unique target basenames.", len(pairs))

    if args.execute:
        payload = merge_cache_mappings(root, existing_cache_payload, per_dir_mappings)
        payload["dry_run"] = False
        payload["total_renames"] = len(pairs)
        payload["total_review_moves"] = len(review_moves)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logging.info("Wrote mapping: %s", out_path)
    elif cache_modified_by_removals:
        # Dry run: do not persist planned renames; only persist cache key pruning from the removal list.
        payload = copy.deepcopy(existing_cache_payload)
        payload["root_directory"] = str(root.resolve())
        payload["dry_run"] = True
        payload["total_renames"] = 0
        payload["total_review_moves"] = 0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logging.info(
            "Updated %s (removed invalid key(s) only; dry run did not record planned renames).",
            out_path,
        )
    else:
        logging.info(
            "Dry run: not writing %s (planned renames are not cached until --execute).",
            out_path,
        )

    ordered = sorted(pairs, key=lambda on: str(on[0]).lower())
    review_ordered = sorted(review_moves, key=lambda on: str(on[0]).lower())

    if not args.execute:
        logging.info("Dry run - no changes will be made. Pass --execute to apply.")
        for old, new in review_ordered:
            logging.info("[dry-run][review] %s -> %s", old, new)
        for old, new in ordered:
            logging.info("[dry-run] %s -> %s", old, new.name)
    else:
        logging.info(
            "Executing %s review move(s) and %s rename(s)...",
            len(review_ordered),
            len(ordered),
        )
        failed = 0
        if review_ordered:
            review_dir.mkdir(parents=True, exist_ok=True)
        for old, new in review_ordered:
            if not try_rename_file(old, new, verbose=args.verbose):
                failed += 1
        for old, new in ordered:
            if not try_rename_file(old, new, verbose=args.verbose):
                failed += 1
        if failed:
            logging.error(
                "Finished with %s failed operation(s) out of %s.",
                failed,
                len(review_ordered) + len(ordered),
            )
            return 1
        logging.info("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

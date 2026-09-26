#!/usr/bin/env python3
"""
Pack the example pipeline JSON files (assets/pipelines/*.json) into the
encrypted archive the app extracts them from (assets/example_pipelines.enc).

Edit or add the JSON files in assets/pipelines, then run this and commit the
archive; the JSON files themselves are not tracked. Every file must load as a
pipeline, or nothing is written. After writing, the archive is decrypted again
and compared with the files.

Usage:
  python scripts/update_example_pipelines.py --dry-run   # report changes only
  python scripts/update_example_pipelines.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from compare import example_pipelines  # noqa: E402
from compare.classifier_pipeline import ClassifierPipelines  # noqa: E402


def _current_files() -> dict[str, bytes]:
    directory = example_pipelines.EXAMPLE_PIPELINES_DIRECTORY
    files = {}
    for name in example_pipelines.json_files_in(directory):
        with open(os.path.join(directory, name), "rb") as f:
            files[name] = f.read()
    return files


def _archived_files() -> dict[str, bytes]:
    if not os.path.isfile(example_pipelines.EXAMPLE_PIPELINES_ARCHIVE):
        print("No existing archive.")
        return {}
    try:
        return example_pipelines.read_archive()
    except Exception as e:
        print(f"Existing archive could not be read ({e}); treating it as empty.")
        return {}


def _report(current: dict, archived: dict) -> bool:
    """Print the differences; True when there are any."""
    added = sorted(set(current) - set(archived))
    removed = sorted(set(archived) - set(current))
    changed = sorted(n for n in set(current) & set(archived) if current[n] != archived[n])
    unchanged = len(set(current) & set(archived)) - len(changed)
    print(f"Unchanged: {unchanged}  Added: {len(added)}  Changed: {len(changed)}  Removed: {len(removed)}")
    for label, names in (("+", added), ("~", changed), ("-", removed)):
        for name in names:
            print(f"  {label} {name}")
    return bool(added or removed or changed)


def _invalid_files(current: dict) -> list[str]:
    problems = []
    directory = example_pipelines.EXAMPLE_PIPELINES_DIRECTORY
    for name in current:
        try:
            ClassifierPipelines.read_json_file(os.path.join(directory, name))
        except (OSError, ValueError) as e:
            problems.append(f"{name}: {e}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Report what would change without writing the archive.")
    args = parser.parse_args()

    current = _current_files()
    if not current:
        print(f"No JSON files in {example_pipelines.EXAMPLE_PIPELINES_DIRECTORY}; nothing to pack.")
        return 1
    changes = _report(current, _archived_files())
    if args.dry_run:
        if not changes:
            print("No changes: the archive is up to date.")
        return 0

    problems = _invalid_files(current)
    if problems:
        print("\nNot written; these files do not load as pipelines:")
        for p in problems:
            print(f"  {p}")
        return 1

    names = example_pipelines.pack()
    print(f"\nWrote {example_pipelines.EXAMPLE_PIPELINES_ARCHIVE} ({len(names)} file(s)).")
    if example_pipelines.read_archive() != current:
        print("Verification failed: the decrypted archive does not match the files.")
        return 1
    print("Verified: the decrypted archive matches the files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

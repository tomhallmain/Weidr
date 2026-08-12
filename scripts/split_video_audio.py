#!/usr/bin/env python3
"""
Batch-split videos in a directory into a video-only copy and an audio-only
copy, for every video that actually has an audio stream.

For a directory with N video files that contain audio, this produces N
"*_noaudio.<ext>" video files and N "*_audio.mka" audio files. Videos with no
audio stream are skipped and reported as such. Original files are never
modified or deleted.

Reuses the same VideoOps helpers as the "Save copy without audio" context
menu action (ui/app_window/file_ops_controller.py) for the video-only copy,
plus the analogous audio-only extraction added alongside it in
image/video_ops.py.

Usage:
  python scripts/split_video_audio.py <dir>
  python scripts/split_video_audio.py <dir> --recursive
  python scripts/split_video_audio.py <dir> --dry-run
  python scripts/split_video_audio.py <dir> --json-output report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from image.video_ops import VideoOps  # noqa: E402
from utils.config import config  # noqa: E402
from utils.media_utils import get_video_extensions, is_video_file  # noqa: E402

# This script's entire purpose is operating on video files, so the app's
# enable_videos toggle (persisted in config.json, and possibly overridden by
# a cached value read on top of it) must never suppress that -- force it on
# for this process regardless of what was loaded from disk.
config.enable_videos = True


def _debug_candidate(p: Path) -> None:
    ext = p.suffix.lower()
    print(
        f"DEBUG: {p} ext={ext!r} enable_videos={config.enable_videos} "
        f"video_types={get_video_extensions()} is_video_file={is_video_file(str(p))}",
        file=sys.stderr,
    )


def iter_video_files(root: Path, recursive: bool, debug: bool = False) -> Iterable[Path]:
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            for name in filenames:
                p = d / name
                if debug:
                    _debug_candidate(p)
                if is_video_file(str(p)):
                    yield p
    else:
        for p in sorted(root.iterdir()):
            if not p.is_file():
                continue
            if debug:
                _debug_candidate(p)
            if is_video_file(str(p)):
                yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dir", help="Directory of videos to process.")
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be done without writing any files.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print, per file scanned, why is_video_file() accepted or rejected it (to stderr).",
    )
    parser.add_argument(
        "--json-output", metavar="PATH",
        help="Write the summary (splits, skipped, errors, and counts) as JSON to this path.",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    if not VideoOps.find_ffmpeg_executable():
        parser.error("ffmpeg not found on PATH.")
    if not VideoOps.find_ffprobe_executable():
        parser.error("ffprobe not found on PATH.")

    if args.debug:
        print(f"DEBUG: config_path={config.config_path}", file=sys.stderr)

    videos = list(iter_video_files(root, args.recursive, debug=args.debug))
    if not videos:
        print(f"No video files found in {root}.")
        return 0

    splits: list[tuple[Path, str, str]] = []
    skipped: list[Path] = []
    errors: list[tuple[Path, str]] = []

    for video_path in videos:
        rel = video_path.relative_to(root)
        if not VideoOps.probe_has_audio_stream(str(video_path)):
            skipped.append(rel)
            continue

        if args.dry_run:
            noaudio_out = VideoOps.default_output_path_copy_without_audio(str(video_path))
            audio_out = VideoOps.default_output_path_audio_only(str(video_path))
            splits.append((rel, Path(noaudio_out).name, Path(audio_out).name))
            continue

        try:
            noaudio_out = VideoOps.copy_video_without_audio(str(video_path))
            audio_out = VideoOps.extract_audio_only(str(video_path))
        except RuntimeError as e:
            errors.append((rel, str(e)))
            continue

        splits.append((rel, Path(noaudio_out).name, Path(audio_out).name))

    verb = "WOULD SPLIT" if args.dry_run else "SPLIT"
    for rel, noaudio_name, audio_name in splits:
        print(f"{verb}: {rel}")
        print(f"    video -> {noaudio_name}")
        print(f"    audio -> {audio_name}")

    if skipped:
        print("\nSkipped (no audio stream):")
        for rel in skipped:
            print(f"    {rel}")

    if errors:
        print("\nErrors:")
        for rel, msg in errors:
            print(f"    {rel}: {msg}")

    mode = "Dry run" if args.dry_run else "Done"
    print(
        f"\n{mode}. {len(splits)} split, {len(skipped)} skipped (no audio), "
        f"{len(errors)} errors, out of {len(videos)} video file(s)."
    )

    if args.json_output:
        report = {
            "dir": str(root),
            "recursive": args.recursive,
            "dry_run": args.dry_run,
            "counts": {
                "total": len(videos),
                "split": len(splits),
                "skipped": len(skipped),
                "errors": len(errors),
            },
            "splits": [
                {"source": str(rel), "video_output": noaudio_name, "audio_output": audio_name}
                for rel, noaudio_name, audio_name in splits
            ],
            "skipped": [str(rel) for rel in skipped],
            "errors": [{"source": str(rel), "error": msg} for rel, msg in errors],
        }
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote summary JSON to {args.json_output}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

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
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from image.video_ops import VideoOps  # noqa: E402
from utils.media_utils import is_video_file  # noqa: E402


def iter_video_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            for name in filenames:
                p = d / name
                if is_video_file(str(p)):
                    yield p
    else:
        for p in sorted(root.iterdir()):
            if p.is_file() and is_video_file(str(p)):
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
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    if not VideoOps.find_ffmpeg_executable():
        parser.error("ffmpeg not found on PATH.")
    if not VideoOps.find_ffprobe_executable():
        parser.error("ffprobe not found on PATH.")

    videos = list(iter_video_files(root, args.recursive))
    if not videos:
        print(f"No video files found in {root}.")
        return 0

    split_count = 0
    skipped_count = 0
    error_count = 0

    for video_path in videos:
        rel = video_path.relative_to(root)
        if not VideoOps.probe_has_audio_stream(str(video_path)):
            print(f"SKIP (no audio): {rel}")
            skipped_count += 1
            continue

        if args.dry_run:
            noaudio_out = VideoOps.default_output_path_copy_without_audio(str(video_path))
            audio_out = VideoOps.default_output_path_audio_only(str(video_path))
            print(f"WOULD SPLIT: {rel}")
            print(f"    video -> {Path(noaudio_out).name}")
            print(f"    audio -> {Path(audio_out).name}")
            split_count += 1
            continue

        try:
            noaudio_out = VideoOps.copy_video_without_audio(str(video_path))
            audio_out = VideoOps.extract_audio_only(str(video_path))
        except RuntimeError as e:
            print(f"ERROR: {rel}: {e}")
            error_count += 1
            continue

        print(f"SPLIT: {rel}")
        print(f"    video -> {Path(noaudio_out).name}")
        print(f"    audio -> {Path(audio_out).name}")
        split_count += 1

    mode = "Dry run" if args.dry_run else "Done"
    print(
        f"\n{mode}. {split_count} split, {skipped_count} skipped (no audio), "
        f"{error_count} errors, out of {len(videos)} video file(s)."
    )
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())

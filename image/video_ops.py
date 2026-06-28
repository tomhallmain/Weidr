"""
Video file operations (ffmpeg), kept separate from :mod:`image.image_ops`.

Container metadata is format-specific (MP4 ``moov`` atoms, Matroska tags, etc.).
Remuxing with stream copy plus ``-map_metadata -1`` strips global/container tags
without re-encoding; use :func:`ffprobe_json` (or ``ffprobe`` in a shell) later
for a details UI and to verify strips.

Reading rich tags in Python often goes through **ffprobe** (same install as ffmpeg)
or ExifTool; see project notes on format coverage.

Use :meth:`VideoOps.merge_ffprobe_tag_dicts`, :meth:`VideoOps.ffprobe_video_mode_and_dims`,
and :meth:`VideoOps.ffprobe_prompt_fields_from_tags` to interpret :meth:`VideoOps.ffprobe_json` output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from enum import Enum
from typing import Any

from utils.logging_setup import get_logger
from utils.media_utils import is_video_file
from utils.translations import _

logger = get_logger("video_ops")


class VideoCutSide(Enum):
    """Which half of a video to keep when cutting at a single position."""
    KEEP_BEGINNING = "before"
    KEEP_END = "after"


class VideoOps:
    """Static helpers for video processing."""

    @staticmethod
    def find_ffmpeg_executable() -> str | None:
        """Return the ``ffmpeg`` executable path if it is on PATH, else ``None``."""
        return shutil.which("ffmpeg")

    @staticmethod
    def default_output_path_copy_without_metadata(video_path: str) -> str:
        """
        ``dir/foo.ext`` → ``dir/foo_nometa.ext``, or ``foo_nometa_N.ext`` if that exists.
        """
        dirname = os.path.dirname(os.path.abspath(video_path)) or "."
        stem, ext = os.path.splitext(os.path.basename(video_path))
        base = os.path.join(dirname, f"{stem}_nometa")
        candidate = f"{base}{ext}"
        n = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{n}{ext}"
            n += 1
        return candidate

    @staticmethod
    def default_output_path_copy_without_audio(video_path: str) -> str:
        """
        ``dir/foo.ext`` → ``dir/foo_noaudio.ext``, or ``foo_noaudio_N.ext`` if that exists.
        """
        dirname = os.path.dirname(os.path.abspath(video_path)) or "."
        stem, ext = os.path.splitext(os.path.basename(video_path))
        base = os.path.join(dirname, f"{stem}_noaudio")
        candidate = f"{base}{ext}"
        n = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{n}{ext}"
            n += 1
        return candidate

    @staticmethod
    def default_output_path_crop(video_path: str) -> str:
        """``dir/foo.ext`` → ``dir/foo_crop.ext`` (collision-safe)."""
        dirname = os.path.dirname(os.path.abspath(video_path)) or "."
        stem, ext = os.path.splitext(os.path.basename(video_path))
        base = os.path.join(dirname, f"{stem}_crop")
        candidate = f"{base}{ext}"
        n = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{n}{ext}"
            n += 1
        return candidate

    @staticmethod
    def crop_video(
        video_path: str,
        x: int,
        y: int,
        w: int,
        h: int,
        output_path: str | None = None,
    ) -> str:
        """Write a new sibling file spatially cropped to the rectangle (x, y, w, h).

        Uses FFmpeg's ``crop`` filter with libx264 re-encode.
        Returns the output path on success.
        Raises RuntimeError if ffmpeg is missing, inputs are invalid, or ffmpeg fails.
        """
        if not is_video_file(video_path):
            raise RuntimeError("Not a video file")
        if w <= 0 or h <= 0:
            raise RuntimeError(f"Invalid crop dimensions: w={w} h={h}")
        ffmpeg = VideoOps.find_ffmpeg_executable()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH")

        out_path = output_path or VideoOps.default_output_path_crop(video_path)
        if os.path.exists(out_path):
            try:
                os.unlink(out_path)
            except OSError as e:
                raise RuntimeError(f"Could not remove existing output file: {e}") from e

        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", video_path,
            "-vf", f"crop={w}:{h}:{x}:{y}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            out_path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=3600,
            )
        except subprocess.TimeoutExpired as e:
            try:
                if os.path.isfile(out_path):
                    os.unlink(out_path)
            except OSError:
                pass
            raise RuntimeError("ffmpeg timed out while cropping video") from e
        except OSError as e:
            raise RuntimeError(f"Failed to run ffmpeg: {e}") from e

        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"ffmpeg crop failed{detail}")

        logger.info("Wrote cropped video: %s", out_path)
        return out_path

    @staticmethod
    def copy_video_without_audio(
        video_path: str, output_path: str | None = None
    ) -> str:
        """
        Write a new file with audio streams removed (ffmpeg video stream copy, ``-an``).

        Does not modify *video_path*. Default output is a sibling like ``foo_noaudio.ext``
        (see :meth:`default_output_path_copy_without_audio`).

        Returns:
            Path to the written file on success.

        Raises:
            RuntimeError: If the file is not a video, ffmpeg is missing, or processing fails.
        """
        if not is_video_file(video_path):
            raise RuntimeError("Not a video file")
        ffmpeg = VideoOps.find_ffmpeg_executable()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH")

        out_path = output_path or VideoOps.default_output_path_copy_without_audio(
            video_path
        )
        if os.path.exists(out_path):
            try:
                os.unlink(out_path)
            except OSError as e:
                raise RuntimeError(f"Could not remove existing output file: {e}") from e

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-c:v",
            "copy",
            "-an",
            out_path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=3600,
            )
        except subprocess.TimeoutExpired as e:
            try:
                if os.path.isfile(out_path):
                    os.unlink(out_path)
            except OSError:
                pass
            raise RuntimeError("ffmpeg timed out while stripping audio") from e
        except OSError as e:
            raise RuntimeError(f"Failed to run ffmpeg: {e}") from e

        if proc.returncode != 0:
            try:
                if os.path.isfile(out_path):
                    os.unlink(out_path)
            except OSError:
                pass
            err = (proc.stderr or proc.stdout or "").strip()
            detail = f": {err}" if err else ""
            raise RuntimeError(f"ffmpeg failed{detail}")

        logger.info("Wrote video without audio: %s", out_path)
        return out_path

    @staticmethod
    def copy_video_without_metadata(
        video_path: str,
        output_path: str | None = None,
    ) -> str:
        """
        Write a **new file** next to the source with container metadata stripped.

        Uses ffmpeg stream copy (no re-encode): ``-map_metadata -1`` removes global
        tags; ``-map_chapters -1`` drops chapters. Does not modify *video_path*.

        If *output_path* is omitted, uses :meth:`default_output_path_copy_without_metadata`.

        Returns:
            Path to the written file.

        Raises:
            RuntimeError: If validation fails, ffmpeg is missing, or ffmpeg errors.
        """
        if not is_video_file(video_path):
            raise RuntimeError("Not a video file")
        ffmpeg = VideoOps.find_ffmpeg_executable()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found on PATH")

        out = output_path or VideoOps.default_output_path_copy_without_metadata(
            video_path
        )
        out = os.path.abspath(out)
        if os.path.abspath(video_path) == out:
            raise RuntimeError("Output path must differ from the source file")
        if os.path.exists(out):
            raise RuntimeError(f"Output file already exists: {out}")

        out_dir = os.path.dirname(out)
        if out_dir and not os.path.isdir(out_dir):
            raise RuntimeError(f"Output directory does not exist: {out_dir}")

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            video_path,
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-c",
            "copy",
            out,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            try:
                if os.path.isfile(out):
                    os.unlink(out)
            except OSError:
                pass
            raise RuntimeError("ffmpeg timed out while copying video") from None
        except OSError as e:
            raise RuntimeError(f"Failed to run ffmpeg: {e}") from e

        if proc.returncode != 0:
            try:
                if os.path.isfile(out):
                    os.unlink(out)
            except OSError:
                pass
            err = (proc.stderr or proc.stdout or "").strip()
            detail = f": {err}" if err else ""
            raise RuntimeError(f"ffmpeg failed{detail}")

        logger.info("Wrote video without metadata: %s -> %s", video_path, out)
        return out

    @staticmethod
    def default_output_path_cut(video_path: str, side: "VideoCutSide", cut_ms: int) -> str:
        """
        ``dir/foo.ext`` → ``dir/foo_cut_before_01m23s456.ext`` (keep beginning)
        or ``dir/foo_cut_after_01m23s456.ext`` (keep end), collision-safe.
        """
        dirname = os.path.dirname(os.path.abspath(video_path)) or "."
        stem, ext = os.path.splitext(os.path.basename(video_path))
        total_s, ms = divmod(cut_ms, 1000)
        total_m, s = divmod(total_s, 60)
        tag = f"{total_m:02d}m{s:02d}s{ms:03d}"
        label = "before" if side == VideoCutSide.KEEP_BEGINNING else "after"
        base = os.path.join(dirname, f"{stem}_cut_{label}_{tag}")
        candidate = f"{base}{ext}"
        n = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{n}{ext}"
            n += 1
        return candidate

    @staticmethod
    def cut_video_at_ms(
        video_path: str,
        cut_ms: int,
        side: "VideoCutSide",
        duration_ms: int,
        output_path: str | None = None,
    ) -> str:
        """
        Write a new sibling file trimmed at *cut_ms* milliseconds.

        *side* controls which segment is kept:

        - ``KEEP_BEGINNING`` — ``[0, cut_ms]`` via output-side ``-to`` (output from start).
        - ``KEEP_END`` — ``[cut_ms, duration_ms]`` via input-side ``-ss`` + ``-t`` duration.

        Stream copy (no re-encode); keyframe snapping applies.
        The source file is never modified.

        Returns:
            Path to the written output file.

        Raises:
            RuntimeError: Validation failure, missing ffmpeg, or ffmpeg error.
        """
        if not is_video_file(video_path):
            raise RuntimeError(_("Not a video file"))
        ffmpeg = VideoOps.find_ffmpeg_executable()
        if not ffmpeg:
            raise RuntimeError(_("ffmpeg not found on PATH"))

        if cut_ms <= 0:
            raise RuntimeError(_("Cut position must be after the start of the video"))
        if cut_ms >= duration_ms:
            raise RuntimeError(_("Cut position must be before the end of the video"))

        out = output_path or VideoOps.default_output_path_cut(video_path, side, cut_ms)
        out = os.path.abspath(out)
        if os.path.abspath(video_path) == out:
            raise RuntimeError(_("Output path must differ from the source file"))
        if os.path.exists(out):
            raise RuntimeError(_("Output file already exists: {0}").format(out))

        t_sec = cut_ms / 1000.0

        if side == VideoCutSide.KEEP_BEGINNING:
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", video_path,
                "-to", str(t_sec),
                "-map", "0", "-c", "copy",
                out,
            ]
        else:
            remain_sec = (duration_ms - cut_ms) / 1000.0
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-ss", str(t_sec), "-i", video_path,
                "-t", str(remain_sec),
                "-map", "0", "-c", "copy",
                out,
            ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            try:
                if os.path.isfile(out):
                    os.unlink(out)
            except OSError:
                pass
            raise RuntimeError(_("ffmpeg timed out while cutting video")) from None
        except OSError as e:
            raise RuntimeError(_("Failed to run ffmpeg: {0}").format(e)) from e

        if proc.returncode != 0:
            try:
                if os.path.isfile(out):
                    os.unlink(out)
            except OSError:
                pass
            err = (proc.stderr or proc.stdout or "").strip()
            detail = f": {err}" if err else ""
            raise RuntimeError(f"ffmpeg failed{detail}")

        logger.info("Cut video (%s) at %d ms → %s", side.value, cut_ms, out)
        return out

    @staticmethod
    def find_ffprobe_executable() -> str | None:
        """Return ``ffprobe`` on PATH if present (pairs with ffmpeg installs)."""
        return shutil.which("ffprobe")

    @staticmethod
    def ffprobe_json(video_path: str) -> dict[str, Any]:
        """
        Run ffprobe and return parsed JSON (format + streams). For future details UI.

        Raises:
            RuntimeError: If ffprobe is missing, the path is invalid, or JSON parse fails.
        """
        if not video_path or not os.path.isfile(video_path):
            raise RuntimeError("Not a file")
        ffprobe = VideoOps.find_ffprobe_executable()
        if not ffprobe:
            raise RuntimeError("ffprobe not found on PATH")

        cmd = [
            ffprobe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            raise RuntimeError(f"ffprobe failed: {e}") from e

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            detail = f": {err}" if err else ""
            raise RuntimeError(f"ffprobe failed{detail}")

        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid ffprobe JSON: {e}") from e
        return data if isinstance(data, dict) else {}

    @staticmethod
    def merge_ffprobe_tag_dicts(probe: dict[str, Any]) -> dict[str, str]:
        """Lowercased keys from format tags + first video stream tags."""
        merged: dict[str, str] = {}
        fmt_tags = (probe.get("format") or {}).get("tags") or {}
        for k, v in fmt_tags.items():
            merged[str(k).lower()] = str(v)
        for s in probe.get("streams") or []:
            if s.get("codec_type") != "video":
                continue
            for k, v in (s.get("tags") or {}).items():
                lk = str(k).lower()
                if lk not in merged:
                    merged[lk] = str(v)
            break
        return merged

    @staticmethod
    def ffprobe_audio_info(
        probe: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str, str]:
        """Extract display strings from an audio ffprobe result.

        Returns ``(codec, duration_str, bitrate_str, sample_rate_str,
        title, artist, album)``.  Any field may be an empty string when
        the information is absent.
        """
        codec = ""
        sample_rate_str = ""
        for s in probe.get("streams") or []:
            if s.get("codec_type") == "audio":
                codec = str(s.get("codec_name") or "")
                sr = s.get("sample_rate")
                if sr:
                    try:
                        sample_rate_str = f"{int(sr):,} Hz"
                    except (ValueError, TypeError):
                        pass
                break

        fmt = probe.get("format") or {}

        duration_str = ""
        raw_dur = fmt.get("duration")
        if raw_dur:
            try:
                total_secs = float(raw_dur)
                mins = int(total_secs // 60)
                secs = int(total_secs % 60)
                duration_str = f"{mins}:{secs:02d}"
            except (ValueError, TypeError):
                pass

        bitrate_str = ""
        raw_br = fmt.get("bit_rate")
        if raw_br:
            try:
                bitrate_str = f"{int(raw_br) // 1000} kbps"
            except (ValueError, TypeError):
                pass

        tags: dict[str, str] = {}
        for k, v in ((fmt.get("tags") or {}).items()):
            tags[str(k).lower()] = str(v)
        # Also pick up any audio-stream-level tags not present at format level
        for s in probe.get("streams") or []:
            if s.get("codec_type") == "audio":
                for k, v in (s.get("tags") or {}).items():
                    lk = str(k).lower()
                    if lk not in tags:
                        tags[lk] = str(v)
                break

        title = tags.get("title", "")
        artist = tags.get("artist", "") or tags.get("album_artist", "")
        album = tags.get("album", "")

        return codec, duration_str, bitrate_str, sample_rate_str, title, artist, album

    @staticmethod
    def ffprobe_video_mode_and_dims(probe: dict[str, Any]) -> tuple[str, str]:
        """Display strings for first video stream: label (translated) and ``WxH``."""
        vcodec = ""
        width = height = None
        for s in probe.get("streams") or []:
            if s.get("codec_type") == "video":
                vcodec = str(s.get("codec_name") or "")
                width = s.get("width")
                height = s.get("height")
                break
        mode = _("Video ({0})").format(vcodec) if vcodec else _("Video")
        dims = ""
        if width and height:
            dims = f"{int(width)}x{int(height)}"
        return mode, dims

    @staticmethod
    def ffprobe_prompt_fields_from_tags(
        probe: dict[str, Any],
    ) -> tuple[str, str, list[str], list[str], bool]:
        """
        Map container / stream tags to positive & negative prompt fields.

        Returns (positive, negative, models, loras, extraction_failed).
        """
        tags = VideoOps.merge_ffprobe_tag_dicts(probe)
        if not tags:
            return "", "", [], [], True

        positive = (
            tags.get("comment")
            or tags.get("description")
            or tags.get("title")
            or ""
        )
        negative = (
            tags.get("negative")
            or tags.get("negative prompt")
            or tags.get("com.apple.quicktime.description.negative")
            or ""
        )
        if not positive.strip():
            lines = [f"{k}: {v}" for k, v in sorted(tags.items())]
            positive = "\n".join(lines)
        return positive, negative, [], [], False

    @staticmethod
    def extract_attached_pic(media_path: str, cache_dir: str) -> str | None:
        """
        Extract the first ``attached_pic`` stream from *media_path* into *cache_dir*.

        Returns the path to the extracted JPEG, or ``None`` if there is no
        such stream, ffmpeg/ffprobe is unavailable, or extraction fails.
        Cached: a previously-extracted file for the same path is reused.
        """
        stem = os.path.splitext(os.path.basename(media_path))[0]
        out_path = os.path.join(cache_dir, f"{stem}_cover.jpg")
        if os.path.isfile(out_path):
            return out_path

        try:
            probe = VideoOps.ffprobe_json(media_path)
        except Exception:
            return None
        streams = probe.get("streams") or []
        has_attached_pic = any(
            s.get("codec_type") == "video"
            and (s.get("disposition") or {}).get("attached_pic", 0)
            for s in streams
        )
        if not has_attached_pic:
            return None

        ffmpeg = VideoOps.find_ffmpeg_executable()
        if not ffmpeg:
            return None

        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            return None

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            media_path,
            "-map",
            "0:v:0",
            "-vframes",
            "1",
            "-update",
            "1",
            out_path,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("ffmpeg attached_pic extraction failed for %s: %s", media_path, e)
            return None

        if proc.returncode != 0 or not os.path.isfile(out_path):
            return None
        return out_path


# Caches probe_has_video_stream() results keyed by (path, mtime) so repeated
# calls during a single playback session don't re-run ffprobe.
_has_video_stream_cache: dict[tuple[str, float], bool] = {}


def probe_has_video_stream(path: str) -> bool:
    """
    Return True if *path* contains at least one real (non-cover-art) video stream.

    A stream with ``disposition.attached_pic == 1`` (e.g. an embedded MP3/FLAC
    cover) is a still image, not a video, and is excluded. Checks via ffprobe
    first; falls back to PyAV then OpenCV (existing logic) when ffprobe is
    unavailable or fails. Falls back to True when nothing is available so
    callers are never worse off than before this guard existed.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    cache_key = (path, mtime)
    cached = _has_video_stream_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _probe_has_video_stream_uncached(path)
    _has_video_stream_cache[cache_key] = result
    return result


def _probe_has_video_stream_uncached(path: str) -> bool:
    try:
        probe = VideoOps.ffprobe_json(path)
        streams = probe.get("streams") or []
        if streams:  # ffprobe succeeded — trust it
            real_video = [
                s for s in streams
                if s.get("codec_type") == "video"
                and not (s.get("disposition") or {}).get("attached_pic", 0)
            ]
            return bool(real_video)
    except Exception:
        pass

    try:
        import av as _av

        container = _av.open(path, metadata_errors="ignore")
        try:
            return bool(container.streams.video)
        finally:
            container.close()
    except Exception:
        pass
    try:
        import cv2 as _cv2

        cap = _cv2.VideoCapture(path)
        try:
            return cap.isOpened() and cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) > 0
        finally:
            cap.release()
    except Exception:
        pass
    return True

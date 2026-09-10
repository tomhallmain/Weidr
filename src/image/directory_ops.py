"""Qt-free media operations that act on a whole directory's worth of files.

ImageOps and friends work one file at a time; the rules for applying them
across a directory -- which extensions count, which files to skip, what a
target path is called -- lived only inside the Qt file-operations controller,
so nothing but the GUI could run them. They live here instead, and the
controller calls in, so the GUI and a headless caller share one implementation.

Each operation is split in two: a survey that reports what a run would do
without touching anything, and the run itself. The GUI needs the survey to
populate its confirmation dialog; a headless caller can skip straight to the
run. The answer the dialog collects is passed in as a plain argument, so no
part of this reaches back into a user interface.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

from utils.config import config
from utils.logging_setup import get_logger

logger = get_logger("directory_ops")

JPG_EXTENSIONS = frozenset((".jpg", ".jpeg"))


def _configured_image_extensions() -> frozenset:
    return frozenset(
        e.lower() for e in getattr(config, "image_types", []) if isinstance(e, str)
    )


def is_non_animated_gif(filepath: str) -> bool:
    """True for a GIF with at most one frame.

    GIF is dynamic media, so only effectively single-frame ones are worth
    converting. A file that cannot be inspected is treated as animated: on a
    decode or read failure the conservative answer is to leave it alone.
    """
    try:
        from PIL import Image
        with Image.open(filepath) as im:
            return int(getattr(im, "n_frames", 1) or 1) <= 1
    except Exception:
        logger.debug("Skipping GIF conversion (unable to inspect frames): %s", filepath)
        return False


def target_jpg_path(filepath: str) -> str:
    """Where *filepath* would be written as JPG. A .jpeg source stays .jpeg."""
    source_ext = os.path.splitext(filepath)[1].lower()
    target_ext = ".jpeg" if source_ext == ".jpeg" else ".jpg"
    return os.path.splitext(filepath)[0] + target_ext


@dataclass
class JpgConversionSurvey:
    """What a JPG conversion run would act on.

    image_files       – every convertible image, existing JPGs included.
    convert_candidates – the subset that is not already JPG/JPEG.
    existing_jpg_count – how many were already JPG/JPEG.
    existing_target_count – candidates whose .jpg target already exists.
    """

    image_files: list = field(default_factory=list)
    convert_candidates: list = field(default_factory=list)
    existing_jpg_count: int = 0
    existing_target_count: int = 0

    def has_nothing_to_do(self) -> bool:
        return not self.image_files and self.existing_jpg_count == 0


@dataclass
class JpgConversionResult:
    converted: int = 0
    failed: int = 0
    skipped_existing: int = 0


def survey_jpg_conversion(files) -> JpgConversionSurvey:
    """Classify *files* for a JPG conversion run without touching anything.

    SVG is included deliberately even when it is not among the configured
    image types: rasterising an SVG to JPG is the point of this workflow.
    """
    convertible = _configured_image_extensions() | {".svg", ".gif"}
    survey = JpgConversionSurvey()

    for filepath in files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in convertible:
            # Non-image media is not part of this action.
            continue
        if ext == ".gif" and not is_non_animated_gif(filepath):
            continue
        survey.image_files.append(filepath)
        if ext in JPG_EXTENSIONS:
            survey.existing_jpg_count += 1
        else:
            survey.convert_candidates.append(filepath)
            if os.path.exists(target_jpg_path(filepath)):
                survey.existing_target_count += 1

    return survey


def convert_files_to_jpg(
    survey: JpgConversionSurvey, *, overwrite_existing: bool
) -> JpgConversionResult:
    """Run the conversion described by *survey*.

    overwrite_existing=True also re-encodes files that are already JPG, which
    is how this workflow strips EXIF from them, and overwrites targets that
    already exist. False converts only non-JPG sources and leaves any existing
    target in place.
    """
    from image.frame_cache import FrameCache
    from image.image_ops import ImageOps

    result = JpgConversionResult()
    candidates = survey.image_files if overwrite_existing else survey.convert_candidates

    for filepath in candidates:
        try:
            ext = os.path.splitext(filepath)[1].lower()

            if ext in JPG_EXTENSIONS:
                # Re-encode in place to strip EXIF.
                ImageOps.convert_to_jpg(filepath, output_path=filepath)
                result.converted += 1
                continue

            output_path = target_jpg_path(filepath)
            if os.path.exists(output_path) and not overwrite_existing:
                result.skipped_existing += 1
                continue

            # SVG has to be rasterised first; FrameCache does that and may
            # already hand back a JPG, in which case a copy is enough.
            source_path = filepath
            if ext == ".svg":
                source_path = FrameCache.get_image_path(filepath)
                if os.path.splitext(source_path)[1].lower() in JPG_EXTENSIONS:
                    shutil.copy2(source_path, output_path)
                    result.converted += 1
                    continue
            ImageOps.convert_to_jpg(source_path, output_path=output_path)
            result.converted += 1
        except Exception as e:
            result.failed += 1
            logger.warning(f"Failed to convert to JPG: {filepath} - {e}")

    return result


# ---------------------------------------------------------------------------
# SVG -> PNG
# ---------------------------------------------------------------------------

def target_png_path(filepath: str) -> str:
    """Where *filepath* would be written as PNG."""
    return os.path.splitext(filepath)[0] + ".png"


@dataclass
class SvgConversionSurvey:
    """What an SVG-to-PNG run would act on.

    convert_candidates – every .svg among the files given.
    existing_target_count – candidates whose .png target already exists.
    """

    convert_candidates: list = field(default_factory=list)
    existing_target_count: int = 0

    def has_nothing_to_do(self) -> bool:
        return not self.convert_candidates


@dataclass
class SvgConversionResult:
    converted: int = 0
    failed: int = 0
    skipped_existing: int = 0


def survey_svg_conversion(files) -> SvgConversionSurvey:
    """Classify *files* for an SVG-to-PNG run without touching anything."""
    survey = SvgConversionSurvey()
    for filepath in files:
        if os.path.splitext(filepath)[1].lower() != ".svg":
            continue
        survey.convert_candidates.append(filepath)
        if os.path.exists(target_png_path(filepath)):
            survey.existing_target_count += 1
    return survey


def convert_svgs_to_png(
    survey: SvgConversionSurvey, *, overwrite_existing: bool
) -> SvgConversionResult:
    """Rasterise each surveyed SVG to a PNG beside it.

    The raster comes from FrameCache, which already renders SVGs for display,
    so this copies that render out rather than rendering a second time.
    overwrite_existing=False leaves any existing target in place.
    """
    from image.frame_cache import FrameCache

    result = SvgConversionResult()
    for filepath in survey.convert_candidates:
        try:
            target_path = target_png_path(filepath)
            if os.path.exists(target_path) and not overwrite_existing:
                result.skipped_existing += 1
                continue
            shutil.copy2(FrameCache.get_image_path(filepath), target_path)
            result.converted += 1
        except Exception as e:
            result.failed += 1
            logger.warning(f"Failed to convert to PNG: {filepath} - {e}")
    return result


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

@dataclass
class ScaleSurvey:
    """What a scaling run would act on.

    target_side is an *equivalent square side*: the goal is a total pixel
    count of target_side², with the aspect ratio preserved, so a non-square
    image ends up with neither dimension equal to target_side.

    An image whose pixel count is already at or below the target is counted in
    already_within. One that cannot be read is not counted there -- it stays in
    to_scale so the run attempts it and reports the failure, rather than being
    silently written off as fine.
    """

    candidates: list = field(default_factory=list)
    already_within: int = 0
    target_side: int = 0

    @property
    def target_pixels(self) -> int:
        return self.target_side * self.target_side

    @property
    def to_scale(self) -> int:
        return len(self.candidates) - self.already_within

    def has_no_candidates(self) -> bool:
        return not self.candidates

    def nothing_to_scale(self) -> bool:
        return self.to_scale == 0


@dataclass
class ScaleResult:
    scaled: int = 0
    skipped: int = 0
    failed: int = 0


def survey_image_scaling(files, target_side: int) -> ScaleSurvey:
    """Classify *files* for a scaling run without touching anything.

    SVG is not included: unlike JPG conversion there is nothing to rasterise
    into here, so the candidate set is the configured image types plus GIF.
    """
    from PIL import Image

    scaleable = _configured_image_extensions() | {".gif"}
    survey = ScaleSurvey(target_side=target_side)
    survey.candidates = [
        f for f in files if os.path.splitext(f)[1].lower() in scaleable
    ]
    target_pixels = survey.target_pixels

    for filepath in survey.candidates:
        try:
            with Image.open(filepath) as img:
                width, height = img.size
            if width * height <= target_pixels:
                survey.already_within += 1
        except Exception:
            # Unreadable: leave it in to_scale so the run reports the failure.
            logger.debug("Could not measure image for scaling: %s", filepath)

    return survey


def scale_images(survey: ScaleSurvey) -> ScaleResult:
    """Scale every surveyed candidate in place.

    Every candidate is attempted, including those counted as already_within:
    scale_image_to_equivalent_pixels() decides per file and reports back
    whether it actually resized, which is what separates scaled from skipped.
    """
    from image.image_ops import ImageOps

    result = ScaleResult()
    for filepath in survey.candidates:
        try:
            _unused, was_scaled = ImageOps.scale_image_to_equivalent_pixels(
                filepath, survey.target_side
            )
            if was_scaled:
                result.scaled += 1
            else:
                result.skipped += 1
        except Exception as e:
            result.failed += 1
            logger.warning("Failed to scale image: %s — %s", filepath, e)
    return result


# ---------------------------------------------------------------------------
# Video metadata stripping
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadataStripSurvey:
    """Which files a metadata-stripping run would copy.

    Selection goes through utils.media_utils.is_video_file, which already
    requires the file to exist, videos to be enabled in config, and the suffix
    to be a configured video type -- so a caller does not gate on
    config.enable_videos separately to get an empty survey.
    """

    videos: list = field(default_factory=list)

    def has_nothing_to_do(self) -> bool:
        return not self.videos


@dataclass
class VideoMetadataStripResult:
    written: int = 0
    failed: int = 0


def survey_video_metadata_strip(files) -> VideoMetadataStripSurvey:
    """Pick the videos out of *files* without touching anything."""
    from utils.media_utils import is_video_file

    return VideoMetadataStripSurvey(
        videos=[fp for fp in files if is_video_file(fp)]
    )


def strip_video_metadata(
    survey: VideoMetadataStripSurvey,
) -> VideoMetadataStripResult:
    """Write a sibling copy of each surveyed video with container metadata gone.

    Nothing is overwritten: each copy is a new file beside its source, so there
    is no skip-or-replace decision to make. Requires ffmpeg; without it every
    file fails rather than raising, and the caller sees it in the count.
    """
    from image.video_ops import VideoOps

    result = VideoMetadataStripResult()
    for filepath in survey.videos:
        try:
            VideoOps.copy_video_without_metadata(filepath)
            result.written += 1
        except Exception as e:
            result.failed += 1
            logger.warning("Copy without metadata failed for %s: %s", filepath, e)
    return result

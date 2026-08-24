"""Directory-wide media operations, without Qt.

These ran only inside the Qt file-operations controller and had no tests at
all, while rewriting every matching file in a directory. They now live in
image.directory_ops; these tests are the safety net they never had, so they
lean on pinning the rules rather than on breadth.

Rules covered here, none of which are obvious from the call sites:
  - JPG: SVG is converted even though it is not a configured image type;
    only single-frame GIFs are converted and unreadable ones are left alone;
    a .jpeg source keeps .jpeg; overwrite_existing both re-encodes existing
    JPGs (that is how EXIF is stripped) and decides whether an existing
    target is replaced or skipped.
  - SVG->PNG: only .svg is taken, so a configured-image-type change cannot
    widen or narrow it; overwrite_existing decides replace vs skip.
"""

import os

import pytest
from PIL import Image

from image import directory_ops
from utils.config import config


def _png(path):
    Image.new("RGB", (8, 8), (10, 120, 200)).save(str(path), format="PNG")
    return str(path)


def _jpg(path):
    Image.new("RGB", (8, 8), (200, 120, 10)).save(str(path), format="JPEG")
    return str(path)


def _animated_gif(path):
    """Write a genuinely multi-frame GIF.

    The frames must differ visibly. Filling "P" images with different palette
    indices is not enough: the default palette can map them to the same
    colour, and the GIF encoder then collapses the frames into one, producing
    a single-frame file that looks animated but is not. The assertion keeps a
    fixture failure from being mistaken for a survey bug.
    """
    frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 0, 255))]
    frames[0].save(
        str(path), format="GIF", save_all=True,
        append_images=frames[1:], duration=100, loop=0,
    )
    with Image.open(str(path)) as im:
        assert getattr(im, "n_frames", 1) > 1, "fixture did not write a multi-frame GIF"
    return str(path)


def _static_gif(path):
    """Write a single-frame GIF (the counterpart to _animated_gif)."""
    Image.new("RGB", (8, 8), (0, 200, 0)).save(str(path), format="GIF")
    with Image.open(str(path)) as im:
        assert getattr(im, "n_frames", 1) == 1, "fixture wrote more than one frame"
    return str(path)


@pytest.fixture(autouse=True)
def _pinned_image_types(monkeypatch):
    # The survey reads config.image_types; pin it so the test does not depend
    # on the shipped default list.
    monkeypatch.setattr(config, "image_types", [".png", ".jpg", ".jpeg"])


class TestTargetPath:
    def test_png_becomes_jpg(self):
        assert directory_ops.target_jpg_path("/d/a.png") == "/d/a.jpg"

    def test_jpeg_keeps_its_spelling(self):
        # Not normalised to .jpg -- otherwise converting a .jpeg would write a
        # second file beside it instead of replacing it.
        assert directory_ops.target_jpg_path("/d/a.jpeg") == "/d/a.jpeg"

    def test_uppercase_extension_is_recognised(self):
        assert directory_ops.target_jpg_path("/d/a.PNG") == "/d/a.jpg"


class TestSurvey:
    def test_counts_existing_jpgs_separately_from_candidates(self, tmp_path):
        a = _png(tmp_path / "a.png")
        b = _jpg(tmp_path / "b.jpg")
        survey = directory_ops.survey_jpg_conversion([a, b])
        assert survey.convert_candidates == [a]
        assert survey.existing_jpg_count == 1
        assert set(survey.image_files) == {a, b}

    def test_non_image_files_are_ignored(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("x", encoding="utf-8")
        survey = directory_ops.survey_jpg_conversion([str(txt)])
        assert survey.image_files == []
        assert survey.has_nothing_to_do()

    def test_svg_is_included_though_not_a_configured_image_type(self, tmp_path):
        # config.image_types is pinned to png/jpg/jpeg above; SVG must still
        # be picked up, because rasterising it is the point of this workflow.
        svg = tmp_path / "vector.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        survey = directory_ops.survey_jpg_conversion([str(svg)])
        assert survey.convert_candidates == [str(svg)]

    def test_animated_gif_is_skipped(self, tmp_path):
        gif = _animated_gif(tmp_path / "moving.gif")
        survey = directory_ops.survey_jpg_conversion([gif])
        assert survey.image_files == []

    def test_static_gif_is_a_candidate(self, tmp_path):
        gif = _static_gif(tmp_path / "still.gif")
        survey = directory_ops.survey_jpg_conversion([gif])
        assert survey.convert_candidates == [gif]

    def test_unreadable_gif_is_treated_as_animated(self, tmp_path):
        broken = tmp_path / "broken.gif"
        broken.write_bytes(b"not a gif")
        survey = directory_ops.survey_jpg_conversion([str(broken)])
        assert survey.image_files == []

    def test_existing_target_is_counted(self, tmp_path):
        a = _png(tmp_path / "a.png")
        _jpg(tmp_path / "a.jpg")  # the target already exists
        survey = directory_ops.survey_jpg_conversion([a])
        assert survey.existing_target_count == 1


class TestConversion:
    def test_converts_a_png_and_leaves_the_source(self, tmp_path):
        a = _png(tmp_path / "a.png")
        survey = directory_ops.survey_jpg_conversion([a])

        result = directory_ops.convert_files_to_jpg(survey, overwrite_existing=False)

        assert result.converted == 1
        assert result.failed == 0
        assert os.path.isfile(tmp_path / "a.jpg")
        # Conversion writes a sibling; removing the source is not part of it.
        assert os.path.isfile(a)

    def test_existing_target_is_skipped_when_not_overwriting(self, tmp_path):
        a = _png(tmp_path / "a.png")
        target = _jpg(tmp_path / "a.jpg")
        before = os.path.getsize(target)
        survey = directory_ops.survey_jpg_conversion([a])

        result = directory_ops.convert_files_to_jpg(survey, overwrite_existing=False)

        assert result.skipped_existing == 1
        assert result.converted == 0
        assert os.path.getsize(target) == before

    def test_existing_jpgs_are_reencoded_only_when_overwriting(self, tmp_path):
        b = _jpg(tmp_path / "b.jpg")
        survey = directory_ops.survey_jpg_conversion([b])

        # Not overwriting: an existing JPG is not a convert candidate at all.
        assert directory_ops.convert_files_to_jpg(
            survey, overwrite_existing=False
        ).converted == 0

        # Overwriting: it is re-encoded in place, which is the EXIF strip.
        assert directory_ops.convert_files_to_jpg(
            survey, overwrite_existing=True
        ).converted == 1
        assert os.path.isfile(b)

    def test_failures_are_counted_not_raised(self, tmp_path, monkeypatch):
        a = _png(tmp_path / "a.png")
        survey = directory_ops.survey_jpg_conversion([a])

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("image.image_ops.ImageOps.convert_to_jpg", _boom)
        result = directory_ops.convert_files_to_jpg(survey, overwrite_existing=False)

        assert result.failed == 1
        assert result.converted == 0

    def test_empty_survey_does_nothing(self):
        result = directory_ops.convert_files_to_jpg(
            directory_ops.JpgConversionSurvey(), overwrite_existing=True
        )
        assert (result.converted, result.failed, result.skipped_existing) == (0, 0, 0)


# ---------------------------------------------------------------------------
# SVG -> PNG
# ---------------------------------------------------------------------------
#
# Deliberately narrower than the JPG survey: this one takes only .svg, so a
# configured-image-type change cannot widen or narrow it.

class TestSvgTargetPath:
    def test_extension_is_replaced(self):
        assert directory_ops.target_png_path("/d/a.svg") == "/d/a.png"

    def test_uppercase_source_is_handled(self):
        assert directory_ops.target_png_path("/d/a.SVG") == "/d/a.png"


class TestSvgSurvey:
    def test_only_svgs_are_candidates(self, tmp_path):
        svg = tmp_path / "vector.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        png = _png(tmp_path / "raster.png")

        survey = directory_ops.survey_svg_conversion([str(svg), png])

        assert survey.convert_candidates == [str(svg)]
        assert not survey.has_nothing_to_do()

    def test_uppercase_extension_is_matched(self, tmp_path):
        svg = tmp_path / "vector.SVG"
        svg.write_text("<svg/>", encoding="utf-8")
        survey = directory_ops.survey_svg_conversion([str(svg)])
        assert survey.convert_candidates == [str(svg)]

    def test_no_svgs_means_nothing_to_do(self, tmp_path):
        survey = directory_ops.survey_svg_conversion([_png(tmp_path / "a.png")])
        assert survey.has_nothing_to_do()

    def test_existing_png_target_is_counted(self, tmp_path):
        svg = tmp_path / "a.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        _png(tmp_path / "a.png")  # target already present

        survey = directory_ops.survey_svg_conversion([str(svg)])

        assert survey.existing_target_count == 1


class TestSvgConversion:
    """FrameCache does the rasterising, so it is stubbed: what is under test is
    the skip/overwrite decision and the counting, not SVG rendering."""

    def _stub_framecache(self, monkeypatch, tmp_path):
        rendered = _png(tmp_path / "_rendered_source.png")
        monkeypatch.setattr(
            "image.frame_cache.FrameCache.get_image_path",
            staticmethod(lambda media_path: rendered),
        )
        return rendered

    def test_converts_an_svg(self, tmp_path, monkeypatch):
        self._stub_framecache(monkeypatch, tmp_path)
        svg = tmp_path / "a.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        survey = directory_ops.survey_svg_conversion([str(svg)])

        result = directory_ops.convert_svgs_to_png(survey, overwrite_existing=False)

        assert result.converted == 1
        assert result.failed == 0
        assert os.path.isfile(tmp_path / "a.png")

    def test_existing_target_is_skipped_when_not_overwriting(self, tmp_path, monkeypatch):
        self._stub_framecache(monkeypatch, tmp_path)
        svg = tmp_path / "a.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        target = _png(tmp_path / "a.png")
        before = os.path.getsize(target)
        survey = directory_ops.survey_svg_conversion([str(svg)])

        result = directory_ops.convert_svgs_to_png(survey, overwrite_existing=False)

        assert result.skipped_existing == 1
        assert result.converted == 0
        assert os.path.getsize(target) == before

    def test_existing_target_is_replaced_when_overwriting(self, tmp_path, monkeypatch):
        rendered = self._stub_framecache(monkeypatch, tmp_path)
        svg = tmp_path / "a.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        _png(tmp_path / "a.png")
        survey = directory_ops.survey_svg_conversion([str(svg)])

        result = directory_ops.convert_svgs_to_png(survey, overwrite_existing=True)

        assert result.converted == 1
        assert result.skipped_existing == 0
        assert os.path.getsize(tmp_path / "a.png") == os.path.getsize(rendered)

    def test_render_failures_are_counted_not_raised(self, tmp_path, monkeypatch):
        svg = tmp_path / "a.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        survey = directory_ops.survey_svg_conversion([str(svg)])

        def _boom(media_path):
            raise OSError("cannot rasterise")

        monkeypatch.setattr(
            "image.frame_cache.FrameCache.get_image_path", staticmethod(_boom)
        )
        result = directory_ops.convert_svgs_to_png(survey, overwrite_existing=True)

        assert result.failed == 1
        assert result.converted == 0

    def test_empty_survey_does_nothing(self):
        result = directory_ops.convert_svgs_to_png(
            directory_ops.SvgConversionSurvey(), overwrite_existing=True
        )
        assert (result.converted, result.failed, result.skipped_existing) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def _png_sized(path, width, height):
    Image.new("RGB", (width, height), (10, 120, 200)).save(str(path), format="PNG")
    return str(path)


class TestScaleSurvey:
    def test_target_pixels_is_the_side_squared(self):
        survey = directory_ops.ScaleSurvey(target_side=320)
        assert survey.target_pixels == 320 * 320

    def test_only_configured_image_types_and_gif_are_candidates(self, tmp_path):
        png = _png(tmp_path / "a.png")
        gif = _static_gif(tmp_path / "b.gif")
        txt = tmp_path / "c.txt"
        txt.write_text("x", encoding="utf-8")

        survey = directory_ops.survey_image_scaling([png, gif, str(txt)], 100)

        assert set(survey.candidates) == {png, gif}

    def test_svg_is_not_scaleable(self, tmp_path):
        # Contrast with JPG conversion, which does take SVG: there is nothing
        # to rasterise into here, so a vector file is not a candidate.
        svg = tmp_path / "vector.svg"
        svg.write_text("<svg/>", encoding="utf-8")
        survey = directory_ops.survey_image_scaling([str(svg)], 100)
        assert survey.candidates == []
        assert survey.has_no_candidates()

    def test_image_at_or_below_target_counts_as_already_within(self, tmp_path):
        small = _png_sized(tmp_path / "small.png", 2, 2)  # 4 px
        survey = directory_ops.survey_image_scaling([small], 4)  # target 16 px
        assert survey.already_within == 1
        assert survey.to_scale == 0
        assert survey.nothing_to_scale()

    def test_image_exactly_on_the_limit_counts_as_within(self, tmp_path):
        exact = _png_sized(tmp_path / "exact.png", 4, 4)  # 16 px
        survey = directory_ops.survey_image_scaling([exact], 4)  # target 16 px
        assert survey.already_within == 1

    def test_larger_image_is_left_to_scale(self, tmp_path):
        big = _png_sized(tmp_path / "big.png", 8, 8)  # 64 px
        survey = directory_ops.survey_image_scaling([big], 4)  # target 16 px
        assert survey.already_within == 0
        assert survey.to_scale == 1

    def test_unreadable_image_is_not_written_off_as_within(self, tmp_path):
        # It stays in to_scale so the run attempts it and reports the failure,
        # rather than being silently counted as already fine.
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a png")
        survey = directory_ops.survey_image_scaling([str(broken)], 4)
        assert survey.candidates == [str(broken)]
        assert survey.already_within == 0
        assert survey.to_scale == 1


class TestScaleRun:
    def test_every_candidate_is_attempted_not_just_the_oversized_ones(
        self, tmp_path, monkeypatch
    ):
        # scale_image_to_equivalent_pixels decides per file and reports back
        # whether it resized; that is what separates scaled from skipped.
        small = _png_sized(tmp_path / "small.png", 2, 2)
        big = _png_sized(tmp_path / "big.png", 8, 8)
        seen = []

        def _fake(image_path, target_side, output_path=None):
            seen.append(image_path)
            return image_path, image_path == big

        monkeypatch.setattr(
            "image.image_ops.ImageOps.scale_image_to_equivalent_pixels", _fake
        )
        survey = directory_ops.survey_image_scaling([small, big], 4)
        result = directory_ops.scale_images(survey)

        assert set(seen) == {small, big}
        assert result.scaled == 1
        assert result.skipped == 1
        assert result.failed == 0

    def test_failures_are_counted_not_raised(self, tmp_path, monkeypatch):
        a = _png_sized(tmp_path / "a.png", 8, 8)

        def _boom(image_path, target_side, output_path=None):
            raise OSError("cannot write")

        monkeypatch.setattr(
            "image.image_ops.ImageOps.scale_image_to_equivalent_pixels", _boom
        )
        result = directory_ops.scale_images(
            directory_ops.survey_image_scaling([a], 4)
        )

        assert result.failed == 1
        assert result.scaled == 0

    def test_empty_survey_does_nothing(self):
        result = directory_ops.scale_images(directory_ops.ScaleSurvey(target_side=4))
        assert (result.scaled, result.skipped, result.failed) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Video metadata stripping
# ---------------------------------------------------------------------------
#
# ffmpeg is not invoked here: VideoOps.copy_video_without_metadata is stubbed,
# so what is under test is the selection and the counting, not the transcode.

class TestVideoStripSurvey:
    def test_only_videos_are_selected(self, tmp_path, monkeypatch):
        png = _png(tmp_path / "a.png")
        vid = tmp_path / "b.mp4"
        vid.write_bytes(b"fake")

        monkeypatch.setattr(
            "utils.media_utils.is_video_file", lambda p: str(p).endswith(".mp4")
        )
        survey = directory_ops.survey_video_metadata_strip([png, str(vid)])

        assert survey.videos == [str(vid)]
        assert not survey.has_nothing_to_do()

    def test_no_videos_means_nothing_to_do(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.media_utils.is_video_file", lambda p: False)
        survey = directory_ops.survey_video_metadata_strip([_png(tmp_path / "a.png")])
        assert survey.has_nothing_to_do()

    def test_disabled_videos_yield_an_empty_survey(self, tmp_path, monkeypatch):
        # is_video_file consults config.enable_videos itself, so a caller gets
        # an empty survey without gating on the flag separately.
        vid = tmp_path / "b.mp4"
        vid.write_bytes(b"fake")
        monkeypatch.setattr(config, "enable_videos", False)

        survey = directory_ops.survey_video_metadata_strip([str(vid)])

        assert survey.has_nothing_to_do()


class TestVideoStripRun:
    def test_each_video_is_copied(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "image.video_ops.VideoOps.copy_video_without_metadata",
            lambda video_path, output_path=None: seen.append(video_path) or video_path,
        )
        survey = directory_ops.VideoMetadataStripSurvey(videos=["/v/a.mp4", "/v/b.mp4"])

        result = directory_ops.strip_video_metadata(survey)

        assert seen == ["/v/a.mp4", "/v/b.mp4"]
        assert result.written == 2
        assert result.failed == 0

    def test_failures_are_counted_and_do_not_stop_the_run(self, monkeypatch):
        def _fake(video_path, output_path=None):
            if video_path.endswith("bad.mp4"):
                raise OSError("ffmpeg exploded")
            return video_path

        monkeypatch.setattr(
            "image.video_ops.VideoOps.copy_video_without_metadata", _fake
        )
        survey = directory_ops.VideoMetadataStripSurvey(
            videos=["/v/bad.mp4", "/v/good.mp4"]
        )

        result = directory_ops.strip_video_metadata(survey)

        # The bad one comes first: the good one must still be attempted.
        assert result.failed == 1
        assert result.written == 1

    def test_empty_survey_does_nothing(self):
        result = directory_ops.strip_video_metadata(
            directory_ops.VideoMetadataStripSurvey()
        )
        assert (result.written, result.failed) == (0, 0)

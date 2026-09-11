"""image.directory_ops's PEEK survey/execute pair, without the optional peek
dependency installed -- image.peek_frame_selector.extract_peek_frames is
mocked so these test the batch-orchestration/counting logic in directory_ops
itself (survey classification, per-file try/except, count aggregation), not
PEEK's own model behavior.
"""

from unittest.mock import patch

from image import directory_ops
from image.peek_frame_selector import PeekFrameExtraction


def _touch(path):
    with open(path, "wb") as f:
        f.write(b"\x00")
    return str(path)


class TestSurveyPeekExtraction:
    def test_classifies_video_and_gif_as_eligible(self, tmp_path):
        video = _touch(tmp_path / "a.mp4")
        gif = _touch(tmp_path / "b.gif")
        survey = directory_ops.survey_peek_extraction([video, gif])
        assert set(survey.eligible_files) == {video, gif}
        assert survey.ineligible_count == 0

    def test_pdf_counted_as_ineligible_not_dropped_silently(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        with patch("utils.media_utils.is_classifier_dynamic_media_path", return_value=True):
            survey = directory_ops.survey_peek_extraction([str(pdf)])
        assert survey.eligible_files == []
        assert survey.ineligible_count == 1

    def test_plain_image_is_neither(self, tmp_path):
        img = _touch(tmp_path / "a.png")
        survey = directory_ops.survey_peek_extraction([img])
        assert survey.eligible_files == []
        assert survey.ineligible_count == 0

    def test_has_nothing_to_do(self):
        assert directory_ops.PeekExtractionSurvey().has_nothing_to_do()


class TestExtractPeekFramesForDirectory:
    def test_aggregates_across_files(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        b = _touch(tmp_path / "b.mp4")
        survey = directory_ops.PeekExtractionSurvey(eligible_files=[a, b])

        def fake_extract(media_path, k=None, fps=None, target_dir=None):
            return PeekFrameExtraction(media_path=media_path, frames_written=["x.jpg", "y.jpg"])

        with patch("image.peek_frame_selector.extract_peek_frames", side_effect=fake_extract):
            result = directory_ops.extract_peek_frames_for_directory(survey)

        assert result.extracted == 2
        assert result.frames_written == 4
        assert result.failed == 0
        assert result.skipped == 0

    def test_zero_frames_counts_as_skipped_not_failed(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        survey = directory_ops.PeekExtractionSurvey(eligible_files=[a])

        def fake_extract(media_path, k=None, fps=None, target_dir=None):
            return PeekFrameExtraction(media_path=media_path, frames_written=[])

        with patch("image.peek_frame_selector.extract_peek_frames", side_effect=fake_extract):
            result = directory_ops.extract_peek_frames_for_directory(survey)

        assert result.extracted == 0
        assert result.skipped == 1
        assert result.failed == 0

    def test_one_file_failing_does_not_stop_the_rest(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        b = _touch(tmp_path / "b.mp4")
        survey = directory_ops.PeekExtractionSurvey(eligible_files=[a, b])

        def fake_extract(media_path, k=None, fps=None, target_dir=None):
            if media_path == a:
                raise RuntimeError("boom")
            return PeekFrameExtraction(media_path=media_path, frames_written=["x.jpg"])

        with patch("image.peek_frame_selector.extract_peek_frames", side_effect=fake_extract):
            result = directory_ops.extract_peek_frames_for_directory(survey)

        assert result.failed == 1
        assert result.extracted == 1
        assert result.frames_written == 1

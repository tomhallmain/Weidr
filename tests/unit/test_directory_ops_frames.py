"""image.directory_ops's frame-extraction survey/execute pair, with the
per-file call mocked -- these test the batch-orchestration logic in
directory_ops itself (survey classification, per-file try/except, count
aggregation, progress and cancellation), not any strategy's own behaviour.
"""

from unittest.mock import patch

from image import directory_ops
from image.frame_extraction import FrameExtraction


def _touch(path):
    with open(path, "wb") as f:
        f.write(b"\x00")
    return str(path)


def _extraction(media_path, frames=("x.png",), duplicates=0):
    return FrameExtraction(
        media_path=media_path,
        frames_written=list(frames),
        duplicates_skipped=duplicates,
    )


class TestSurveyFrameExtraction:
    def test_classifies_video_and_gif_as_eligible(self, tmp_path):
        video = _touch(tmp_path / "a.mp4")
        gif = _touch(tmp_path / "b.gif")
        survey = directory_ops.survey_frame_extraction([video, gif])
        assert set(survey.eligible_files) == {video, gif}
        assert survey.ineligible_count == 0

    def test_pdf_counted_as_ineligible_not_dropped_silently(self, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        with patch("utils.media_utils.is_classifier_dynamic_media_path", return_value=True):
            survey = directory_ops.survey_frame_extraction([str(pdf)])
        assert survey.eligible_files == []
        assert survey.ineligible_count == 1

    def test_plain_image_is_neither(self, tmp_path):
        img = _touch(tmp_path / "a.png")
        survey = directory_ops.survey_frame_extraction([img])
        assert survey.eligible_files == []
        assert survey.ineligible_count == 0

    def test_has_nothing_to_do(self):
        assert directory_ops.FrameExtractionSurvey().has_nothing_to_do()


class TestExtractFramesForDirectory:
    def test_aggregates_across_files(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        b = _touch(tmp_path / "b.mp4")
        survey = directory_ops.FrameExtractionSurvey(eligible_files=[a, b])

        def fake_extract(media_path, strategy="peek", **kwargs):
            return _extraction(media_path, frames=("x.png", "y.png"))

        with patch("image.frame_extraction.extract_frames", side_effect=fake_extract):
            result = directory_ops.extract_frames_for_directory(survey, "peek")

        assert result.extracted == 2
        assert result.frames_written == 4
        assert result.failed == 0
        assert result.skipped == 0

    def test_zero_frames_counts_as_skipped_not_failed(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        survey = directory_ops.FrameExtractionSurvey(eligible_files=[a])

        def fake_extract(media_path, strategy="peek", **kwargs):
            return _extraction(media_path, frames=())

        with patch("image.frame_extraction.extract_frames", side_effect=fake_extract):
            result = directory_ops.extract_frames_for_directory(survey, "peek")

        assert result.extracted == 0
        assert result.skipped == 1
        assert result.failed == 0

    def test_one_file_failing_does_not_stop_the_rest(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        b = _touch(tmp_path / "b.mp4")
        survey = directory_ops.FrameExtractionSurvey(eligible_files=[a, b])

        def fake_extract(media_path, strategy="peek", **kwargs):
            if media_path == a:
                raise RuntimeError("boom")
            return _extraction(media_path)

        with patch("image.frame_extraction.extract_frames", side_effect=fake_extract):
            result = directory_ops.extract_frames_for_directory(survey, "peek")

        assert result.failed == 1
        assert result.extracted == 1
        assert result.frames_written == 1

    def test_the_strategy_reaches_every_file(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        b = _touch(tmp_path / "b.gif")
        survey = directory_ops.FrameExtractionSurvey(eligible_files=[a, b])
        seen = []

        def fake_extract(media_path, strategy="peek", **kwargs):
            seen.append((media_path, strategy, kwargs.get("action_name")))
            return _extraction(media_path)

        with patch("image.frame_extraction.extract_frames", side_effect=fake_extract):
            directory_ops.extract_frames_for_directory(
                survey, "trigger", action_name="Rotate check",
            )

        assert seen == [(a, "trigger", "Rotate check"), (b, "trigger", "Rotate check")]

    def test_duplicates_are_totalled(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        b = _touch(tmp_path / "b.mp4")
        survey = directory_ops.FrameExtractionSurvey(eligible_files=[a, b])

        def fake_extract(media_path, strategy="peek", **kwargs):
            return _extraction(media_path, duplicates=2)

        with patch("image.frame_extraction.extract_frames", side_effect=fake_extract):
            result = directory_ops.extract_frames_for_directory(survey, "peek")

        assert result.duplicates_skipped == 4


class TestDefaultRunsEveryStrategy:
    def test_no_strategy_goes_through_the_combined_run(self, tmp_path):
        """Omitting a strategy must reach extract_frames_all, not one strategy."""
        a = _touch(tmp_path / "a.mp4")
        survey = directory_ops.FrameExtractionSurvey(eligible_files=[a])
        combined = []

        with patch(
            "image.frame_extraction.extract_frames_all",
            side_effect=lambda media_path, **kw: combined.append(media_path) or _extraction(media_path),
        ):
            result = directory_ops.extract_frames_for_directory(survey)

        assert combined == [a]
        assert result.extracted == 1

    def test_a_named_strategy_runs_only_that_one(self, tmp_path):
        a = _touch(tmp_path / "a.mp4")
        survey = directory_ops.FrameExtractionSurvey(eligible_files=[a])
        single = []

        with patch(
            "image.frame_extraction.extract_frames",
            side_effect=lambda media_path, strategy="peek", **kw:
                single.append(strategy) or _extraction(media_path),
        ):
            directory_ops.extract_frames_for_directory(survey, "first")

        assert single == ["first"]


class TestProgressAndCancellation:
    def _survey(self, tmp_path, count):
        files = [_touch(tmp_path / f"f{i}.mp4") for i in range(count)]
        return directory_ops.FrameExtractionSurvey(eligible_files=files)

    def test_progress_is_reported_after_every_file(self, tmp_path):
        survey = self._survey(tmp_path, 3)
        reported = []

        with patch(
            "image.frame_extraction.extract_frames_all",
            side_effect=lambda media_path, **kw: _extraction(media_path),
        ):
            result = directory_ops.extract_frames_for_directory(
                survey, progress_callback=lambda done, total: reported.append((done, total)),
            )

        assert reported == [(1, 3), (2, 3), (3, 3)]
        assert result.cancelled is False

    def test_returning_false_stops_the_run(self, tmp_path):
        survey = self._survey(tmp_path, 4)
        calls = []

        def _stop_after_two(done, total):
            calls.append(done)
            return False if done >= 2 else None

        with patch(
            "image.frame_extraction.extract_frames_all",
            side_effect=lambda media_path, **kw: _extraction(media_path),
        ):
            result = directory_ops.extract_frames_for_directory(
                survey, progress_callback=_stop_after_two,
            )

        assert calls == [1, 2]
        assert result.extracted == 2
        assert result.cancelled is True

    def test_a_callback_returning_none_does_not_cancel(self, tmp_path):
        """The trap this guards: a callback that only refreshes a widget
        returns None, which must not read as 'stop'."""
        survey = self._survey(tmp_path, 3)

        with patch(
            "image.frame_extraction.extract_frames_all",
            side_effect=lambda media_path, **kw: _extraction(media_path),
        ):
            result = directory_ops.extract_frames_for_directory(
                survey, progress_callback=lambda done, total: None,
            )

        assert result.extracted == 3
        assert result.cancelled is False

"""image.frame_extraction -- the mechanics shared by every selection strategy.

Eligibility (video/GIF only, not PDF), the GIF frame timeline built from
per-frame durations, duration reading, target-dir/output-path resolution, and
the related_image tag written into each frame. None of it needs the optional
peek dependency, which is the point of the module existing separately.
"""

import os

import pytest
from PIL import Image

from image import frame_extraction as fe
from utils.config import config


def _mp4(path):
    """Extension-only stand-in -- eligibility is a suffix check, not a decode."""
    with open(path, "wb") as f:
        f.write(b"\x00")
    return str(path)


def _gif(path, durations_ms):
    frames = [Image.new("RGB", (4, 4), c) for c in [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
    ][: len(durations_ms)]]
    frames[0].save(
        str(path), format="GIF", save_all=True,
        append_images=frames[1:], duration=durations_ms, loop=0,
    )
    return str(path)


def _pdf(path):
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n")
    return str(path)


class TestEligibility:
    def test_video_extension_is_eligible(self, tmp_path):
        assert fe.is_frame_extraction_eligible(_mp4(tmp_path / "a.mp4"))

    def test_gif_is_eligible(self, tmp_path):
        assert fe.is_frame_extraction_eligible(_gif(tmp_path / "a.gif", [100, 100]))

    def test_pdf_is_not_eligible(self, tmp_path):
        assert not fe.is_frame_extraction_eligible(_pdf(tmp_path / "a.pdf"))

    def test_missing_file_is_not_eligible(self, tmp_path):
        assert not fe.is_frame_extraction_eligible(str(tmp_path / "nope.mp4"))


class TestGifFrameTimeline:
    def test_cumulative_starts_match_declared_durations(self, tmp_path):
        path = _gif(tmp_path / "a.gif", [100, 200, 50])
        assert fe.gif_frame_timeline_ms(path) == [0, 100, 300]

    def test_single_frame_gif(self, tmp_path):
        path = _gif(tmp_path / "a.gif", [100])
        assert fe.gif_frame_timeline_ms(path) == [0]

    def test_gif_duration_comes_from_the_timeline(self, tmp_path):
        path = _gif(tmp_path / "a.gif", [100, 200, 50])
        # The last frame's start, in seconds -- the same value the caps use.
        assert fe.media_duration_seconds(path) == 0.3


class TestTargetDirResolution:
    def test_explicit_target_dir_wins(self, tmp_path):
        media = str(tmp_path / "a.mp4")
        assert fe.resolve_target_dir(media, str(tmp_path / "out")) == str(tmp_path / "out")

    def test_defaults_to_source_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "peek_output_directory", None)
        monkeypatch.setattr(config, "peek_save_to_same_dir", True)
        media = str(tmp_path / "sub" / "a.mp4")
        assert fe.resolve_target_dir(media, None) == str(tmp_path / "sub")

    def test_configured_output_directory_used_when_not_same_dir(self, tmp_path, monkeypatch):
        configured = str(tmp_path / "peek_out")
        monkeypatch.setattr(config, "peek_output_directory", configured)
        monkeypatch.setattr(config, "peek_save_to_same_dir", False)
        media = str(tmp_path / "sub" / "a.mp4")
        assert fe.resolve_target_dir(media, None) == configured


class TestFrameOutputPaths:
    def test_naming_and_count(self, tmp_path):
        media = str(tmp_path / "clip.mp4")
        paths = fe.frame_output_paths(media, str(tmp_path), 3, "_peek_")
        assert paths == [
            str(tmp_path / "clip_peek_0.png"),
            str(tmp_path / "clip_peek_1.png"),
            str(tmp_path / "clip_peek_2.png"),
        ]

    def test_suffix_is_the_callers_choice(self, tmp_path):
        media = str(tmp_path / "clip.mp4")
        assert fe.frame_output_paths(media, str(tmp_path), 1, "_trigger_") == [
            str(tmp_path / "clip_trigger_0.png"),
        ]


class TestRelatedImageTag:
    def test_written_png_carries_related_image_pointing_at_source(self, tmp_path):
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        out_path = str(tmp_path / "frame.png")
        source_path = str(tmp_path / "source.mp4")

        assert fe.write_png_with_related_image(img, out_path, source_path)

        with Image.open(out_path) as written:
            assert written.info.get("related_image") == source_path


class TestStrategyDispatch:
    def test_unknown_strategy_is_refused(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100])
        with pytest.raises(RuntimeError, match="strategy"):
            fe.extract_frames(source, "nonsense")

    def test_ineligible_file_is_refused_for_every_strategy(self, tmp_path):
        path = _pdf(tmp_path / "a.pdf")
        for strategy in fe.STRATEGIES:
            with pytest.raises(RuntimeError, match="video or GIF"):
                fe.extract_frames(path, strategy)

    def test_peek_delegates_to_the_peek_selector(self, tmp_path, monkeypatch):
        """The optional dependency is reached only by the peek strategy."""
        source = _gif(tmp_path / "a.gif", [100, 100])
        calls = []

        import image.peek_frame_selector as pfs

        monkeypatch.setattr(
            pfs, "extract_peek_frames",
            lambda path, **kwargs: calls.append((path, kwargs))
            or fe.FrameExtraction(media_path=path, frames_written=[]),
        )

        fe.extract_frames(source, fe.STRATEGY_PEEK, k=3, fps=1.5)

        assert len(calls) == 1
        path, kwargs = calls[0]
        assert path == source
        assert (kwargs["k"], kwargs["fps"], kwargs["target_dir"]) == (3, 1.5, None)
        # Forwarded so a combined run can share one duplicate-skip set.
        assert "seen" in kwargs


class TestFirstAndLast:
    def test_first_writes_the_opening_frame(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])

        outcome = fe.extract_frames(source, fe.STRATEGY_FIRST)

        expected = str(tmp_path / "a_first.png")
        assert outcome.frames_written == [expected]
        with Image.open(expected) as written, Image.open(source) as original:
            original.seek(0)
            assert written.convert("RGB").getpixel((0, 0)) == original.convert("RGB").getpixel((0, 0))

    def test_last_writes_the_closing_frame(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])

        outcome = fe.extract_frames(source, fe.STRATEGY_LAST)

        expected = str(tmp_path / "a_last.png")
        assert outcome.frames_written == [expected]
        with Image.open(expected) as written, Image.open(source) as original:
            original.seek(2)
            assert written.convert("RGB").getpixel((0, 0)) == original.convert("RGB").getpixel((0, 0))

    def test_first_and_last_differ_and_both_carry_the_tag(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])

        first = fe.extract_frames(source, fe.STRATEGY_FIRST).frames_written[0]
        last = fe.extract_frames(source, fe.STRATEGY_LAST).frames_written[0]

        with Image.open(first) as a, Image.open(last) as b:
            assert a.convert("RGB").getpixel((0, 0)) != b.convert("RGB").getpixel((0, 0))
            assert a.info.get("related_image") == source
            assert b.info.get("related_image") == source

    def test_last_refuses_a_video_whose_duration_cannot_be_read(self, tmp_path, monkeypatch):
        source = _mp4(tmp_path / "a.mp4")
        monkeypatch.setattr(fe, "video_duration_and_fps", lambda path: (None, None))

        with pytest.raises(RuntimeError, match="duration"):
            fe.extract_frames(source, fe.STRATEGY_LAST)


class TestTriggerStrategy:
    def _patch_report(self, monkeypatch, report):
        import compare.trigger_scan as ts
        monkeypatch.setattr(ts, "find_trigger", lambda *a, **kw: report)

    def test_writes_the_triggering_frame_named_by_slot(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        self._patch_report(monkeypatch, {
            "media_kind": "dynamic", "matched": True, "slot_index": 2,
            "total_planned_slots": 3, "position": {"kind": "ms", "value": 150},
        })

        outcome = fe.extract_frames(
            source, fe.STRATEGY_TRIGGER, action_name="Rotate check",
        )

        expected = str(tmp_path / "a_trigger_2.png")
        assert outcome.frames_written == [expected]
        with Image.open(expected) as written:
            assert written.info.get("related_image") == source

    def test_nothing_triggering_writes_nothing_and_is_not_an_error(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100])
        self._patch_report(monkeypatch, {
            "media_kind": "dynamic", "matched": False, "samples_scanned": 2,
        })

        outcome = fe.extract_frames(source, fe.STRATEGY_TRIGGER, action_name="Rotate check")

        assert outcome.frames_written == []
        assert not list(tmp_path.glob("*.png"))

    def test_an_unresolvable_position_is_reported(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100])
        self._patch_report(monkeypatch, {
            "media_kind": "dynamic", "matched": True, "slot_index": 1,
            "total_planned_slots": 2, "position": None,
        })

        with pytest.raises(RuntimeError, match="position"):
            fe.extract_frames(source, fe.STRATEGY_TRIGGER, action_name="Rotate check")

    def test_a_caller_error_from_the_scan_surfaces(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100])

        import compare.trigger_scan as ts

        def _raise(*_a, **_kw):
            raise ValueError("no classifier_action named 'Nope'")

        monkeypatch.setattr(ts, "find_trigger", _raise)

        with pytest.raises(RuntimeError, match="Nope"):
            fe.extract_frames(source, fe.STRATEGY_TRIGGER, action_name="Nope")

    def test_without_an_action_name_it_says_so(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100])
        with pytest.raises(RuntimeError, match="classifier action"):
            fe.extract_frames(source, fe.STRATEGY_TRIGGER)


class TestAllStrategiesInOnePass:
    """One action, every enabled strategy, deduped -- the normal run."""

    @pytest.fixture(autouse=True)
    def _only_the_cheap_strategies(self, monkeypatch):
        """First and last on; peek and trigger need a package and an action."""
        monkeypatch.setattr(config, "frame_extraction_use_first_frame", True)
        monkeypatch.setattr(config, "frame_extraction_use_last_frame", True)
        monkeypatch.setattr(config, "frame_extraction_use_trigger", True)
        monkeypatch.setattr(config, "frame_extraction_trigger_action", None)
        monkeypatch.setattr(config, "enable_peek_frame_detection", False)

    def test_enabled_strategies_are_ordered_deterministic_first(self, monkeypatch):
        monkeypatch.setattr(config, "enable_peek_frame_detection", True)
        monkeypatch.setattr(config, "frame_extraction_trigger_action", "Rotate check")

        assert fe.enabled_strategies() == [
            fe.STRATEGY_FIRST, fe.STRATEGY_LAST, fe.STRATEGY_TRIGGER, fe.STRATEGY_PEEK,
        ]

    def test_trigger_is_left_out_without_an_action(self):
        assert fe.STRATEGY_TRIGGER not in fe.enabled_strategies()

    def test_an_action_passed_in_brings_trigger_back(self):
        assert fe.STRATEGY_TRIGGER in fe.enabled_strategies("Rotate check")

    def test_switched_off_strategies_do_not_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "frame_extraction_use_last_frame", False)
        source = _gif(tmp_path / "a.gif", [100, 100, 100])

        outcome = fe.extract_frames_all(source)

        assert outcome.frames_written == [str(tmp_path / "a_first.png")]

    def test_one_pass_writes_each_enabled_strategy(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])

        outcome = fe.extract_frames_all(source)

        assert outcome.frames_written == [
            str(tmp_path / "a_first.png"), str(tmp_path / "a_last.png"),
        ]
        assert outcome.duplicates_skipped == 0
        assert outcome.errors == []

    def test_strategies_landing_on_one_frame_write_it_once(self, tmp_path):
        """A single-frame GIF is its own first and last frame."""
        source = _gif(tmp_path / "a.gif", [100])

        outcome = fe.extract_frames_all(source)

        assert outcome.frames_written == [str(tmp_path / "a_first.png")]
        assert outcome.duplicates_skipped == 1

    def test_rerunning_replaces_its_own_output(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        first = fe.extract_frames_all(source)

        again = fe.extract_frames_all(source)

        assert again.frames_written == first.frames_written
        assert again.duplicates_skipped == 0

    def test_a_failing_strategy_is_reported_and_the_others_still_run(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        monkeypatch.setattr(config, "enable_peek_frame_detection", True)

        import image.peek_frame_selector as pfs

        def _no_peek(*_a, **_kw):
            raise RuntimeError("peek is required for frame detection.")

        monkeypatch.setattr(pfs, "extract_peek_frames", _no_peek)

        outcome = fe.extract_frames_all(source)

        assert outcome.frames_written == [
            str(tmp_path / "a_first.png"), str(tmp_path / "a_last.png"),
        ]
        assert len(outcome.errors) == 1
        assert "peek" in outcome.errors[0]

    def test_the_configured_trigger_action_is_used(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        monkeypatch.setattr(config, "frame_extraction_trigger_action", "Rotate check")
        monkeypatch.setattr(config, "frame_extraction_use_first_frame", False)
        monkeypatch.setattr(config, "frame_extraction_use_last_frame", False)
        asked = []

        import compare.trigger_scan as ts

        def _find(action_name, kind, media_path, **kwargs):
            asked.append((action_name, kind))
            return {
                "media_kind": "dynamic", "matched": True, "slot_index": 1,
                "total_planned_slots": 3, "position": {"kind": "ms", "value": 100},
            }

        monkeypatch.setattr(ts, "find_trigger", _find)

        outcome = fe.extract_frames_all(source)

        assert asked == [("Rotate check", "classifier_action")]
        assert outcome.frames_written == [str(tmp_path / "a_trigger_1.png")]

    def test_nothing_enabled_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "frame_extraction_use_first_frame", False)
        monkeypatch.setattr(config, "frame_extraction_use_last_frame", False)
        monkeypatch.setattr(config, "frame_extraction_use_trigger", False)
        source = _gif(tmp_path / "a.gif", [100, 100])

        outcome = fe.extract_frames_all(source)

        assert outcome.frames_written == []
        assert not list(tmp_path.glob("*.png"))

    def test_an_ineligible_file_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="video or GIF"):
            fe.extract_frames_all(_pdf(tmp_path / "a.pdf"))


class TestDuplicateFrames:
    """A frame already pulled out of a file is not written again."""

    def _trigger_at(self, monkeypatch, slot, ms):
        import compare.trigger_scan as ts
        monkeypatch.setattr(ts, "find_trigger", lambda *a, **kw: {
            "media_kind": "dynamic", "matched": True, "slot_index": slot,
            "total_planned_slots": 3, "position": {"kind": "ms", "value": ms},
        })

    def test_another_strategy_landing_on_the_same_frame_writes_nothing(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        fe.extract_frames(source, fe.STRATEGY_FIRST)
        self._trigger_at(monkeypatch, slot=0, ms=0)  # the opening frame again

        outcome = fe.extract_frames(source, fe.STRATEGY_TRIGGER, action_name="Check")

        assert outcome.frames_written == []
        assert outcome.duplicates_skipped == 1
        assert sorted(p.name for p in tmp_path.glob("*.png")) == ["a_first.png"]

    def test_a_different_frame_is_still_written(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        fe.extract_frames(source, fe.STRATEGY_FIRST)
        self._trigger_at(monkeypatch, slot=2, ms=200)  # the third frame

        outcome = fe.extract_frames(source, fe.STRATEGY_TRIGGER, action_name="Check")

        assert outcome.frames_written == [str(tmp_path / "a_trigger_2.png")]
        assert outcome.duplicates_skipped == 0

    def test_rerunning_a_strategy_replaces_its_own_output(self, tmp_path):
        """Its own names are exempt, or a rerun would skip everything."""
        source = _gif(tmp_path / "a.gif", [100, 100])
        first = fe.extract_frames(source, fe.STRATEGY_FIRST)

        again = fe.extract_frames(source, fe.STRATEGY_FIRST)

        assert again.frames_written == first.frames_written
        assert again.duplicates_skipped == 0

    def test_frames_of_another_source_do_not_suppress_this_one(self, tmp_path):
        other = _gif(tmp_path / "other.gif", [100, 100])
        fe.extract_frames(other, fe.STRATEGY_FIRST)
        # Same pixels, different source: its tag names other.gif.
        source = _gif(tmp_path / "a.gif", [100, 100])

        outcome = fe.extract_frames(source, fe.STRATEGY_FIRST)

        assert outcome.frames_written == [str(tmp_path / "a_first.png")]
        assert outcome.duplicates_skipped == 0

    def test_switching_the_safeguard_off_writes_the_duplicate(self, tmp_path, monkeypatch):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        fe.extract_frames(source, fe.STRATEGY_FIRST)
        monkeypatch.setattr(config, "frame_extraction_skip_duplicate_frames", False)
        self._trigger_at(monkeypatch, slot=0, ms=0)

        outcome = fe.extract_frames(source, fe.STRATEGY_TRIGGER, action_name="Check")

        assert outcome.frames_written == [str(tmp_path / "a_trigger_0.png")]
        assert outcome.duplicates_skipped == 0

    def test_two_timestamps_on_one_frame_collapse_within_a_call(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100])
        out_paths = [str(tmp_path / "out_0.png"), str(tmp_path / "out_1.png")]

        # Both land inside the first frame.
        writes = fe.extract_frames_at_timestamps(source, [0.0, 0.05], out_paths, set())

        assert writes.written == [out_paths[0]]
        assert writes.duplicates == 1

    def test_signature_set_is_none_when_the_safeguard_is_off(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "frame_extraction_skip_duplicate_frames", False)
        assert fe.signatures_to_skip(str(tmp_path / "a.gif"), str(tmp_path)) is None


class TestGifFrameWriting:
    def test_timestamps_map_to_frames_and_carry_the_tag(self, tmp_path):
        source = _gif(tmp_path / "a.gif", [100, 100, 100])
        out_paths = [str(tmp_path / f"out_{i}.png") for i in range(2)]

        # 0ms is the first frame; 150ms falls inside the second.
        writes = fe.extract_frames_at_timestamps(source, [0.0, 0.15], out_paths)

        assert writes.written == out_paths
        assert writes.duplicates == 0
        for path in out_paths:
            assert os.path.isfile(path)
            with Image.open(path) as frame:
                assert frame.info.get("related_image") == source
        with Image.open(out_paths[0]) as first, Image.open(out_paths[1]) as second:
            assert first.convert("RGB").getpixel((0, 0)) != second.convert("RGB").getpixel((0, 0))

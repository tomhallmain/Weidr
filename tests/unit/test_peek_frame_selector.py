"""image.peek_frame_selector -- the Qt-free helpers, without the optional
peek dependency actually installed.

These pin the parts that don't need peek itself: eligibility (video/GIF
only, not PDF), the candidate-density cap math, the GIF frame-timeline
built from per-frame durations, and target-dir/output-path resolution.
extract_peek_frames' top-level guard (raises for an ineligible path before
ever trying to import peek) is covered too, since that path never reaches
the optional import either.
"""

import os

import pytest
from PIL import Image

from image import peek_frame_selector as pfs
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
        assert pfs.is_peek_eligible_media_path(_mp4(tmp_path / "a.mp4"))

    def test_gif_is_eligible(self, tmp_path):
        assert pfs.is_peek_eligible_media_path(_gif(tmp_path / "a.gif", [100, 100]))

    def test_pdf_is_not_eligible(self, tmp_path):
        assert not pfs.is_peek_eligible_media_path(_pdf(tmp_path / "a.pdf"))

    def test_missing_file_is_not_eligible(self, tmp_path):
        assert not pfs.is_peek_eligible_media_path(str(tmp_path / "nope.mp4"))


class TestCandidateDensityCap:
    def test_short_duration_keeps_requested_fps(self, monkeypatch):
        monkeypatch.setattr(config, "peek_max_candidate_frames", 300)
        assert pfs._effective_candidate_fps(10.0, 2.0) == 2.0

    def test_long_duration_is_capped(self, monkeypatch):
        monkeypatch.setattr(config, "peek_max_candidate_frames", 300)
        # 1 hour at 2.0 fps would be 7200 candidates; capped to 300 total.
        effective = pfs._effective_candidate_fps(3600.0, 2.0)
        assert effective < 2.0
        assert 3600.0 * effective == pytest.approx(300.0, rel=1e-6)

    def test_unknown_duration_is_not_capped(self, monkeypatch):
        monkeypatch.setattr(config, "peek_max_candidate_frames", 300)
        assert pfs._effective_candidate_fps(None, 2.0) == 2.0


class TestGifFrameTimeline:
    def test_cumulative_starts_match_declared_durations(self, tmp_path):
        path = _gif(tmp_path / "a.gif", [100, 200, 50])
        timeline = pfs._gif_frame_timeline_ms(path)
        assert timeline == [0, 100, 300]

    def test_single_frame_gif(self, tmp_path):
        path = _gif(tmp_path / "a.gif", [100])
        assert pfs._gif_frame_timeline_ms(path) == [0]


class TestTargetDirResolution:
    def test_explicit_target_dir_wins(self, tmp_path):
        media = str(tmp_path / "a.mp4")
        out = pfs._resolve_target_dir(media, str(tmp_path / "out"))
        assert out == str(tmp_path / "out")

    def test_defaults_to_source_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "peek_output_directory", None)
        monkeypatch.setattr(config, "peek_save_to_same_dir", True)
        media = str(tmp_path / "sub" / "a.mp4")
        assert pfs._resolve_target_dir(media, None) == str(tmp_path / "sub")

    def test_configured_output_directory_used_when_not_same_dir(self, tmp_path, monkeypatch):
        configured = str(tmp_path / "peek_out")
        monkeypatch.setattr(config, "peek_output_directory", configured)
        monkeypatch.setattr(config, "peek_save_to_same_dir", False)
        media = str(tmp_path / "sub" / "a.mp4")
        assert pfs._resolve_target_dir(media, None) == configured


class TestFrameOutputPaths:
    def test_naming_and_count(self, tmp_path):
        media = str(tmp_path / "clip.mp4")
        paths = pfs._frame_output_paths(media, str(tmp_path), 3)
        assert paths == [
            str(tmp_path / "clip_peek_0.png"),
            str(tmp_path / "clip_peek_1.png"),
            str(tmp_path / "clip_peek_2.png"),
        ]


class TestScaledDefaultK:
    def test_short_video_uses_base_k(self, monkeypatch):
        monkeypatch.setattr(config, "peek_default_k", 4)
        monkeypatch.setattr(config, "peek_minutes_per_extra_frame", 2.0)
        assert pfs._scaled_default_k(30.0) == 4

    def test_long_video_scales_up(self, monkeypatch):
        monkeypatch.setattr(config, "peek_default_k", 4)
        monkeypatch.setattr(config, "peek_minutes_per_extra_frame", 2.0)
        # 10 minutes / 2 minutes-per-extra = +5
        assert pfs._scaled_default_k(600.0) == 9

    def test_unknown_duration_uses_base_k(self, monkeypatch):
        monkeypatch.setattr(config, "peek_default_k", 4)
        assert pfs._scaled_default_k(None) == 4

    def test_scaling_disabled_when_per_extra_is_zero(self, monkeypatch):
        monkeypatch.setattr(config, "peek_default_k", 4)
        monkeypatch.setattr(config, "peek_minutes_per_extra_frame", 0)
        assert pfs._scaled_default_k(3600.0) == 4


class TestMaxDurationGuard:
    def test_long_video_is_refused(self, tmp_path, monkeypatch):
        media = _mp4(tmp_path / "long.mp4")
        monkeypatch.setattr(config, "peek_max_video_duration_seconds", 600)
        monkeypatch.setattr(pfs, "_video_duration_and_fps", lambda path: (700.0, 30.0))
        with pytest.raises(RuntimeError, match="long"):
            pfs.extract_peek_frames(media)

    def test_short_video_is_not_refused_by_duration_guard(self, tmp_path, monkeypatch):
        """Passes the duration guard, then legitimately fails at the (mocked
        as missing) peek import -- proves the guard itself did not fire."""
        media = _mp4(tmp_path / "short.mp4")
        monkeypatch.setattr(config, "peek_max_video_duration_seconds", 600)
        monkeypatch.setattr(pfs, "_video_duration_and_fps", lambda path: (30.0, 30.0))
        monkeypatch.setattr(
            pfs, "_load_peek",
            lambda: (_ for _ in ()).throw(RuntimeError("pip install marker")),
        )
        with pytest.raises(RuntimeError, match="pip install marker"):
            pfs.extract_peek_frames(media)

    def test_zero_disables_the_guard(self, tmp_path, monkeypatch):
        media = _mp4(tmp_path / "long.mp4")
        monkeypatch.setattr(config, "peek_max_video_duration_seconds", 0)
        monkeypatch.setattr(pfs, "_video_duration_and_fps", lambda path: (7200.0, 30.0))
        monkeypatch.setattr(
            pfs, "_load_peek",
            lambda: (_ for _ in ()).throw(RuntimeError("pip install marker")),
        )
        with pytest.raises(RuntimeError, match="pip install marker"):
            pfs.extract_peek_frames(media)


class TestRelatedImageTag:
    def test_written_png_carries_related_image_pointing_at_source(self, tmp_path):
        img = Image.new("RGB", (4, 4), (10, 20, 30))
        out_path = str(tmp_path / "frame.png")
        source_path = str(tmp_path / "source.mp4")

        assert pfs._write_png_with_related_image(img, out_path, source_path)

        with Image.open(out_path) as written:
            assert written.info.get("related_image") == source_path


class TestExtractPeekFramesGuard:
    def test_ineligible_path_raises_without_importing_peek(self, tmp_path):
        """Guard runs before the optional peek import, so this must raise
        cleanly even though peek is not installed in this environment."""
        path = _pdf(tmp_path / "a.pdf")
        with pytest.raises(RuntimeError):
            pfs.extract_peek_frames(path)

    def test_missing_dependency_raises_actionable_error(self, tmp_path, monkeypatch):
        path = _mp4(tmp_path / "a.mp4")
        monkeypatch.setattr(
            pfs, "_load_peek",
            lambda: (_ for _ in ()).throw(
                RuntimeError("peek is required for frame detection. Install with: pip install git+https://github.com/momentslab/peek")
            ),
        )
        with pytest.raises(RuntimeError, match="pip install"):
            pfs.extract_peek_frames(path)

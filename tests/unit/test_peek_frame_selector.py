"""image.peek_frame_selector -- the PEEK-specific parts, without the optional
peek dependency actually installed.

These pin what belongs to PEEK itself: the candidate-density cap math, the
scaled default frame count, the duration guard, and extract_peek_frames' own
guards. The decode/write mechanics moved to image/frame_extraction.py and are
covered by tests/unit/test_frame_extraction.py.
"""

import pytest
from PIL import Image

from image import peek_frame_selector as pfs
from utils.config import config


def _mp4(path):
    with open(path, "wb") as f:
        f.write(b"\x00")
    return str(path)


def _pdf(path):
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n")
    return str(path)


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
        monkeypatch.setattr(pfs, "media_duration_seconds", lambda path: 700.0)
        with pytest.raises(RuntimeError, match="long"):
            pfs.extract_peek_frames(media)

    def test_short_video_is_not_refused_by_duration_guard(self, tmp_path, monkeypatch):
        """Passes the duration guard, then legitimately fails at the (mocked
        as missing) peek import -- proves the guard itself did not fire."""
        media = _mp4(tmp_path / "short.mp4")
        monkeypatch.setattr(config, "peek_max_video_duration_seconds", 600)
        monkeypatch.setattr(pfs, "media_duration_seconds", lambda path: 30.0)
        monkeypatch.setattr(
            pfs, "_load_peek",
            lambda: (_ for _ in ()).throw(RuntimeError("pip install marker")),
        )
        with pytest.raises(RuntimeError, match="pip install marker"):
            pfs.extract_peek_frames(media)

    def test_zero_disables_the_guard(self, tmp_path, monkeypatch):
        media = _mp4(tmp_path / "long.mp4")
        monkeypatch.setattr(config, "peek_max_video_duration_seconds", 0)
        monkeypatch.setattr(pfs, "media_duration_seconds", lambda path: 7200.0)
        monkeypatch.setattr(
            pfs, "_load_peek",
            lambda: (_ for _ in ()).throw(RuntimeError("pip install marker")),
        )
        with pytest.raises(RuntimeError, match="pip install marker"):
            pfs.extract_peek_frames(media)


class TestExtractPeekFramesGuard:
    def test_ineligible_path_raises_without_importing_peek(self, tmp_path):
        """Guard runs before the optional peek import, so this must raise
        cleanly even though peek is not installed in this environment."""
        path = _pdf(tmp_path / "a.pdf")
        with pytest.raises(RuntimeError):
            pfs.extract_peek_frames(path)

    def test_missing_dependency_raises_actionable_error(self, tmp_path, monkeypatch):
        path = _mp4(tmp_path / "a.mp4")
        monkeypatch.setattr(pfs, "media_duration_seconds", lambda path: 10.0)
        monkeypatch.setattr(
            pfs, "_load_peek",
            lambda: (_ for _ in ()).throw(
                RuntimeError("peek is required for frame detection. Install with: pip install git+https://github.com/momentslab/peek")
            ),
        )
        with pytest.raises(RuntimeError, match="pip install"):
            pfs.extract_peek_frames(path)

    def test_no_selected_timestamps_writes_nothing(self, tmp_path, monkeypatch):
        path = _mp4(tmp_path / "a.mp4")
        monkeypatch.setattr(pfs, "select_peek_timestamps", lambda *a, **kw: [])

        outcome = pfs.extract_peek_frames(path)

        assert outcome.frames_written == []
        assert not list(tmp_path.glob("*.png"))

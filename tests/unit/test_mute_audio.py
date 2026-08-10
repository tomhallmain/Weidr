"""
Unit tests for VideoOps.mute_audio_range_ms and related helpers.
All subprocess calls are mocked — no real ffmpeg required.
"""

import os
import re
import pytest
from unittest.mock import patch, MagicMock

from image.video_ops import VideoOps
from utils.translations import _


# ---------------------------------------------------------------------------
# default_output_path_mute_audio — path naming
# ---------------------------------------------------------------------------

class TestDefaultOutputPathMuteAudio:
    def test_muted_label(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_mute_audio(src, 20500, 21500)
        assert "_muted_" in out
        assert out.endswith(".mp4")

    def test_time_tag_format(self, tmp_path):
        src = str(tmp_path / "video.mkv")
        open(src, "w").close()
        out = VideoOps.default_output_path_mute_audio(src, 20500, 21500)
        assert "00m20s500-00m21s500" in out

    def test_zero_start_tag(self, tmp_path):
        src = str(tmp_path / "video.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_mute_audio(src, 0, 1000)
        assert "00m00s000-00m01s000" in out

    def test_collision_suffix(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        first = VideoOps.default_output_path_mute_audio(src, 20500, 21500)
        open(first, "w").close()
        second = VideoOps.default_output_path_mute_audio(src, 20500, 21500)
        assert second != first
        assert not os.path.exists(second)


# ---------------------------------------------------------------------------
# mute_audio_range_ms — ffmpeg argv and validation
# ---------------------------------------------------------------------------

def _make_proc(returncode=0, stderr=""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = stderr
    proc.stdout = ""
    return proc


@pytest.fixture()
def fake_video(tmp_path):
    p = tmp_path / "source.mp4"
    p.write_bytes(b"fake")
    return str(p)


class TestMuteAudioRangeMs:
    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_uses_volume_filter_with_between_expression(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.mute_audio_range_ms(fake_video, 5000, 6000, out)
        cmd = mock_run.call_args[0][0]
        assert "-af" in cmd
        af = cmd[cmd.index("-af") + 1]
        assert "between(t,5.0,6.0)" in af
        assert "volume=0" in af

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_copies_video_reencodes_audio(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        """Video must be stream-copied (no quality loss / no re-encode); only the
        audio track goes through the filter graph and so must be re-encoded."""
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.mute_audio_range_ms(fake_video, 5000, 6000, out)
        cmd = mock_run.call_args[0][0]
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-c:a" in cmd
        assert cmd[cmd.index("-c:a") + 1] == "aac"

    @patch("image.video_ops.is_video_file", return_value=False)
    def test_raises_if_not_video(self, _is_video, fake_video):
        with pytest.raises(RuntimeError):
            VideoOps.mute_audio_range_ms(fake_video, 5000, 6000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value=None)
    def test_raises_if_no_ffmpeg(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match=re.escape(_("ffmpeg not found on PATH"))):
            VideoOps.mute_audio_range_ms(fake_video, 5000, 6000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_if_start_negative(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(
            RuntimeError,
            match=re.escape(_("Mute range must start at or after the beginning of the video")),
        ):
            VideoOps.mute_audio_range_ms(fake_video, -1, 1000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_if_end_before_start(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match=re.escape(_("Mute range end must be after its start"))):
            VideoOps.mute_audio_range_ms(fake_video, 6000, 5000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_if_end_equals_start(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match=re.escape(_("Mute range end must be after its start"))):
            VideoOps.mute_audio_range_ms(fake_video, 5000, 5000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_raises_on_ffmpeg_nonzero(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc(returncode=1, stderr="codec error")
        out = str(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            VideoOps.mute_audio_range_ms(fake_video, 5000, 6000, out)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_returns_output_path_on_success(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        result = VideoOps.mute_audio_range_ms(fake_video, 5000, 6000, out)
        assert result == out

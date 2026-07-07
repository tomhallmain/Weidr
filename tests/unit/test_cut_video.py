"""
Unit tests for VideoOps.cut_video_at_ms and related helpers.
All subprocess calls are mocked — no real ffmpeg required.
"""

import os
import re
import pytest
from unittest.mock import patch, MagicMock

from image.video_ops import VideoCutSide, VideoOps
from utils.translations import _


# ---------------------------------------------------------------------------
# default_output_path_cut — path naming
# ---------------------------------------------------------------------------

class TestDefaultOutputPathCut:
    def test_keep_beginning_label(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_cut(src, VideoCutSide.KEEP_BEGINNING, 83456)
        assert "_cut_before_" in out
        assert out.endswith(".mp4")

    def test_keep_end_label(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_cut(src, VideoCutSide.KEEP_END, 83456)
        assert "_cut_after_" in out
        assert out.endswith(".mp4")

    def test_time_tag_format(self, tmp_path):
        src = str(tmp_path / "video.mkv")
        open(src, "w").close()
        # 1 minute 23 seconds 456 ms = 83456 ms
        out = VideoOps.default_output_path_cut(src, VideoCutSide.KEEP_BEGINNING, 83456)
        assert "01m23s456" in out

    def test_zero_ms_tag(self, tmp_path):
        src = str(tmp_path / "video.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_cut(src, VideoCutSide.KEEP_END, 0)
        assert "00m00s000" in out

    def test_collision_suffix(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        first = VideoOps.default_output_path_cut(src, VideoCutSide.KEEP_BEGINNING, 5000)
        open(first, "w").close()
        second = VideoOps.default_output_path_cut(src, VideoCutSide.KEEP_BEGINNING, 5000)
        assert second != first
        assert not os.path.exists(second)


# ---------------------------------------------------------------------------
# cut_video_at_ms — ffmpeg argv and validation
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


class TestCutVideoAtMs:
    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_keep_beginning_uses_to_flag(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.cut_video_at_ms(fake_video, 5000, VideoCutSide.KEEP_BEGINNING, 60000, out)
        cmd = mock_run.call_args[0][0]
        assert "-to" in cmd
        assert str(5.0) in cmd
        assert "-ss" not in cmd

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_keep_end_uses_ss_and_t(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.cut_video_at_ms(fake_video, 10000, VideoCutSide.KEEP_END, 60000, out)
        cmd = mock_run.call_args[0][0]
        assert "-ss" in cmd
        assert str(10.0) in cmd
        assert "-t" in cmd
        # remaining duration = (60000 - 10000) / 1000 = 50.0
        assert str(50.0) in cmd
        assert "-to" not in cmd

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_maps_all_streams(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.cut_video_at_ms(fake_video, 3000, VideoCutSide.KEEP_BEGINNING, 30000, out)
        cmd = mock_run.call_args[0][0]
        assert "-map" in cmd
        assert "0" in cmd
        assert "-c" in cmd
        assert "copy" in cmd

    @patch("image.video_ops.is_video_file", return_value=False)
    def test_raises_if_not_video(self, _is_video, fake_video, tmp_path):
        with pytest.raises(RuntimeError):
            VideoOps.cut_video_at_ms(fake_video, 5000, VideoCutSide.KEEP_BEGINNING, 60000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value=None)
    def test_raises_if_no_ffmpeg(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match=re.escape(_("ffmpeg not found on PATH"))):
            VideoOps.cut_video_at_ms(fake_video, 5000, VideoCutSide.KEEP_BEGINNING, 60000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_if_cut_at_zero(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match=re.escape(_("Cut position must be after the start of the video"))):
            VideoOps.cut_video_at_ms(fake_video, 0, VideoCutSide.KEEP_BEGINNING, 60000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_if_cut_at_or_past_end(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match=re.escape(_("Cut position must be before the end of the video"))):
            VideoOps.cut_video_at_ms(fake_video, 60000, VideoCutSide.KEEP_BEGINNING, 60000)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_raises_on_ffmpeg_nonzero(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc(returncode=1, stderr="codec error")
        out = str(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            VideoOps.cut_video_at_ms(fake_video, 5000, VideoCutSide.KEEP_BEGINNING, 60000, out)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_returns_output_path_on_success(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        result = VideoOps.cut_video_at_ms(fake_video, 5000, VideoCutSide.KEEP_BEGINNING, 60000, out)
        assert result == out

"""
Unit tests for VideoOps.default_output_path_crop and VideoOps.crop_video.
All subprocess calls are mocked — no real ffmpeg required.
"""

import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from image.video_ops import VideoOps


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# default_output_path_crop — naming
# ---------------------------------------------------------------------------

class TestDefaultOutputPathCrop:
    def test_contains_crop_suffix(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_crop(src)
        assert "_crop" in out

    def test_preserves_extension(self, tmp_path):
        src = str(tmp_path / "clip.mkv")
        open(src, "w").close()
        out = VideoOps.default_output_path_crop(src)
        assert out.endswith(".mkv")

    def test_sibling_of_source(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_crop(src)
        assert os.path.dirname(os.path.abspath(out)) == os.path.dirname(os.path.abspath(src))

    def test_collision_avoidance(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        first = VideoOps.default_output_path_crop(src)
        open(first, "w").close()
        second = VideoOps.default_output_path_crop(src)
        assert second != first
        assert not os.path.exists(second)


# ---------------------------------------------------------------------------
# crop_video — validation
# ---------------------------------------------------------------------------

class TestCropVideoValidation:
    @patch("image.video_ops.is_video_file", return_value=False)
    def test_raises_if_not_video(self, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="Not a video file"):
            VideoOps.crop_video(fake_video, 0, 0, 100, 100)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value=None)
    def test_raises_if_no_ffmpeg(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            VideoOps.crop_video(fake_video, 0, 0, 100, 100)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_on_zero_width(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="Invalid crop dimensions"):
            VideoOps.crop_video(fake_video, 0, 0, 0, 100)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_on_zero_height(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="Invalid crop dimensions"):
            VideoOps.crop_video(fake_video, 0, 0, 100, 0)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_on_negative_dimension(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="Invalid crop dimensions"):
            VideoOps.crop_video(fake_video, 0, 0, -10, 100)


# ---------------------------------------------------------------------------
# crop_video — ffmpeg command construction
# ---------------------------------------------------------------------------

class TestCropVideoCommand:
    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_crop_filter_format(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.crop_video(fake_video, x=10, y=20, w=640, h=360, output_path=out)
        cmd = mock_run.call_args[0][0]
        assert "crop=640:360:10:20" in " ".join(cmd)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_uses_libx264_encoder(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.crop_video(fake_video, 0, 0, 100, 100, output_path=out)
        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_copies_audio_stream(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.crop_video(fake_video, 0, 0, 100, 100, output_path=out)
        cmd = mock_run.call_args[0][0]
        # -c:a copy must appear together in the command list
        assert "copy" in cmd
        ca_idx = next(i for i, v in enumerate(cmd) if v == "-c:a")
        assert cmd[ca_idx + 1] == "copy"

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_output_path_is_last_arg(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.crop_video(fake_video, 0, 0, 100, 100, output_path=out)
        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == out

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_uses_explicit_output_path(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "custom_name.mp4")
        result = VideoOps.crop_video(fake_video, 0, 0, 100, 100, output_path=out)
        assert result == out

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_default_output_path_contains_crop(self, mock_run, _ffmpeg, _is_video, fake_video):
        mock_run.return_value = _make_proc()
        result = VideoOps.crop_video(fake_video, 0, 0, 100, 100)
        assert "_crop" in result


# ---------------------------------------------------------------------------
# crop_video — error / edge-case handling
# ---------------------------------------------------------------------------

class TestCropVideoErrors:
    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_raises_on_nonzero_returncode(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc(returncode=1, stderr="some ffmpeg error")
        out = str(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError, match="ffmpeg crop failed"):
            VideoOps.crop_video(fake_video, 0, 0, 100, 100, output_path=out)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=3600))
    def test_raises_on_timeout(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        out = str(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError, match="timed out"):
            VideoOps.crop_video(fake_video, 0, 0, 100, 100, output_path=out)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_removes_existing_output_before_run(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        open(out, "w").close()
        assert os.path.exists(out)
        VideoOps.crop_video(fake_video, 0, 0, 100, 100, output_path=out)
        # file is removed before ffmpeg runs; mock didn't recreate it
        assert not os.path.exists(out)

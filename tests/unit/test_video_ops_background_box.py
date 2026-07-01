"""
Unit tests for VideoOps.default_output_path_background_box and
VideoOps.draw_background_box_on_video.
All subprocess calls are mocked — no real ffmpeg required.
"""

import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from image.video_ops import VideoOps


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
# default_output_path_background_box — naming
# ---------------------------------------------------------------------------

class TestDefaultOutputPathBackgroundBox:
    def test_contains_bgbox_suffix(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        out = VideoOps.default_output_path_background_box(src)
        assert "_bgbox" in out

    def test_always_uses_mp4_regardless_of_source_extension(self, tmp_path):
        # H.264/AAC re-encoded output isn't valid in every source container
        # (e.g. WebM only allows VP8/VP9/AV1), so the output always standardizes
        # on MP4 rather than preserving the source extension.
        src = str(tmp_path / "clip.webm")
        open(src, "w").close()
        out = VideoOps.default_output_path_background_box(src)
        assert out.endswith(".mp4")

    def test_collision_avoidance(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        first = VideoOps.default_output_path_background_box(src)
        open(first, "w").close()
        second = VideoOps.default_output_path_background_box(src)
        assert second != first
        assert not os.path.exists(second)

    def test_does_not_collide_with_box_or_crop_output(self, tmp_path):
        src = str(tmp_path / "clip.mp4")
        open(src, "w").close()
        crop_out = VideoOps.default_output_path_crop(src)
        box_out = VideoOps.default_output_path_box(src)
        bgbox_out = VideoOps.default_output_path_background_box(src)
        assert len({crop_out, box_out, bgbox_out}) == 3


# ---------------------------------------------------------------------------
# draw_background_box_on_video — validation
# ---------------------------------------------------------------------------

class TestDrawBackgroundBoxOnVideoValidation:
    @patch("image.video_ops.is_video_file", return_value=False)
    def test_raises_if_not_video(self, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="Not a video file"):
            VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value=None)
    def test_raises_if_no_ffmpeg(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_on_zero_width(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="Invalid box dimensions"):
            VideoOps.draw_background_box_on_video(fake_video, 0, 0, 0, 100)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    def test_raises_on_negative_dimension(self, _ffmpeg, _is_video, fake_video):
        with pytest.raises(RuntimeError, match="Invalid box dimensions"):
            VideoOps.draw_background_box_on_video(fake_video, 0, 0, -10, 100)


# ---------------------------------------------------------------------------
# draw_background_box_on_video — ffmpeg command construction
# ---------------------------------------------------------------------------

class TestDrawBackgroundBoxOnVideoCommand:
    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_four_drawbox_strips_tile_frame_minus_kept_rect(
        self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path
    ):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.draw_background_box_on_video(fake_video, x=10, y=20, w=100, h=50, output_path=out)
        cmd = mock_run.call_args[0][0]
        vf_idx = next(i for i, v in enumerate(cmd) if v == "-vf")
        strips = cmd[vf_idx + 1]
        assert strips.count("drawbox=") == 4
        # top strip: full width, down to the kept rect's y
        assert "drawbox=x=0:y=0:w=iw:h=20:" in strips
        # bottom strip: full width, from below the kept rect to the frame bottom
        assert "drawbox=x=0:y=70:w=iw:h=ih-70:" in strips
        # left strip: spans just the kept rect's row, up to its x
        assert "drawbox=x=0:y=20:w=10:h=50:" in strips
        # right strip: spans just the kept rect's row, from beyond its x+w
        assert "drawbox=x=110:y=20:w=iw-110:h=50:" in strips

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_all_four_strips_share_the_same_color(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100, output_path=out)
        cmd = mock_run.call_args[0][0]
        vf_idx = next(i for i, v in enumerate(cmd) if v == "-vf")
        strips = cmd[vf_idx + 1]
        colors = [seg.split("color=")[1].split(":")[0] for seg in strips.split(",")]
        assert len(set(colors)) == 1

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_uses_libx264_encoder(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100, output_path=out)
        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_transcodes_audio_to_aac(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        # Audio is re-encoded to AAC (not stream-copied): a copied Vorbis/Opus
        # stream from a source like WebM would be invalid once muxed into MP4.
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100, output_path=out)
        cmd = mock_run.call_args[0][0]
        ca_idx = next(i for i, v in enumerate(cmd) if v == "-c:a")
        assert cmd[ca_idx + 1] == "aac"

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_uses_explicit_output_path(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "custom_name.mp4")
        result = VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100, output_path=out)
        assert result == out

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_default_output_path_contains_bgbox(self, mock_run, _ffmpeg, _is_video, fake_video):
        mock_run.return_value = _make_proc()
        result = VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100)
        assert "_bgbox" in result


# ---------------------------------------------------------------------------
# draw_background_box_on_video — error / edge-case handling
# ---------------------------------------------------------------------------

class TestDrawBackgroundBoxOnVideoErrors:
    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_raises_on_nonzero_returncode(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc(returncode=1, stderr="some ffmpeg error")
        out = str(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError, match="ffmpeg background box draw failed"):
            VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100, output_path=out)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=3600))
    def test_raises_on_timeout(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        out = str(tmp_path / "out.mp4")
        with pytest.raises(RuntimeError, match="timed out"):
            VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100, output_path=out)

    @patch("image.video_ops.is_video_file", return_value=True)
    @patch("image.video_ops.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_removes_existing_output_before_run(self, mock_run, _ffmpeg, _is_video, fake_video, tmp_path):
        mock_run.return_value = _make_proc()
        out = str(tmp_path / "out.mp4")
        open(out, "w").close()
        assert os.path.exists(out)
        VideoOps.draw_background_box_on_video(fake_video, 0, 0, 100, 100, output_path=out)
        assert not os.path.exists(out)

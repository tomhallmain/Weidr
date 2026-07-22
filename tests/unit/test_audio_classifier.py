"""
Unit tests for image.audio_classifier.

decode_audio_waveform: ffmpeg subprocess calls are mocked, no real ffmpeg required
(mirrors tests/unit/test_video_ops_background_box.py's mocking pattern).

AudioClassifierWrapper: config-validation-before-load path needs no torch/transformers
(load_classifier() gracefully sets can_run=False on ImportError). The split-positive
classify_audio() logic is tested against a manually-populated instance with
predict_audio() monkeypatched to canned predictions, so it doesn't need a real model
either -- mirrors how image_classifier.py's equivalent logic has no dedicated test file
of its own to follow, so this establishes the pattern for the audio side directly.
"""

import struct
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from image.audio_classifier import (
    AudioClassifierWrapper,
    decode_audio_waveform,
    derive_neutral_categories_from_positive_groups,
)
from image.audio_classifier_model_config import AudioClassifierModelConfig


def _make_proc(returncode=0, stdout=b"", stderr=b""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _pcm_bytes(*samples: float) -> bytes:
    return struct.pack(f"<{len(samples)}f", *samples)


# ---------------------------------------------------------------------------
# decode_audio_waveform
# ---------------------------------------------------------------------------

class TestDecodeAudioWaveform:
    @patch("image.audio_classifier.VideoOps.find_ffmpeg_executable", return_value=None)
    def test_raises_if_no_ffmpeg(self, _ffmpeg):
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            decode_audio_waveform("/fake/audio.mp3", 16000, 10.0)

    @patch("image.audio_classifier.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_nonzero_returncode_raises(self, mock_run, _ffmpeg):
        mock_run.return_value = _make_proc(returncode=1, stderr=b"Invalid data found")
        with pytest.raises(RuntimeError, match="ffmpeg audio decode failed"):
            decode_audio_waveform("/fake/audio.mp3", 16000, 10.0)

    @patch("image.audio_classifier.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(cmd="ffmpeg", timeout=120))
    def test_timeout_raises(self, _run, _ffmpeg):
        with pytest.raises(RuntimeError, match="timed out"):
            decode_audio_waveform("/fake/audio.mp3", 16000, 10.0)

    @patch("image.audio_classifier.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_empty_output_raises(self, mock_run, _ffmpeg):
        mock_run.return_value = _make_proc(returncode=0, stdout=b"")
        with pytest.raises(RuntimeError, match="no audio samples"):
            decode_audio_waveform("/fake/audio.mp3", 16000, 10.0)

    @patch("image.audio_classifier.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_success_decodes_float32_samples(self, mock_run, _ffmpeg):
        mock_run.return_value = _make_proc(returncode=0, stdout=_pcm_bytes(0.1, -0.2, 0.3))
        waveform = decode_audio_waveform("/fake/audio.mp3", 16000, 10.0)
        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float32
        np.testing.assert_allclose(waveform, [0.1, -0.2, 0.3], atol=1e-6)

    @patch("image.audio_classifier.VideoOps.find_ffmpeg_executable", return_value="/usr/bin/ffmpeg")
    @patch("subprocess.run")
    def test_command_requests_mono_and_target_rate(self, mock_run, _ffmpeg):
        mock_run.return_value = _make_proc(returncode=0, stdout=_pcm_bytes(0.0))
        decode_audio_waveform("/fake/audio.mp3", 22050, 5.0)
        cmd = mock_run.call_args[0][0]
        assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
        assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "22050"
        assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "5.0"
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "f32le"


# ---------------------------------------------------------------------------
# derive_neutral_categories_from_positive_groups
# ---------------------------------------------------------------------------

class TestDeriveNeutralCategories:
    def test_complement_of_positive_groups(self):
        result = derive_neutral_categories_from_positive_groups(
            ["safe", "mild", "explicit"], [["mild", "explicit"]]
        )
        assert result == ["safe"]

    def test_no_positive_groups_returns_all(self):
        result = derive_neutral_categories_from_positive_groups(["a", "b"], [])
        assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# AudioClassifierWrapper -- config validation before load (no torch needed)
# ---------------------------------------------------------------------------

class TestAudioClassifierWrapperValidation:
    def test_empty_model_categories_marks_unrunnable(self):
        cfg = AudioClassifierModelConfig(
            model_name="m", model_location="org/model", model_categories=[]
        )
        w = AudioClassifierWrapper(cfg)
        assert w.can_run is False

    def test_blank_model_name_marks_unrunnable(self):
        cfg = AudioClassifierModelConfig(
            model_name="   ", model_location="org/model", model_categories=["a"]
        )
        w = AudioClassifierWrapper(cfg)
        assert w.can_run is False

    def test_positive_groups_referencing_unknown_category_marks_unrunnable(self):
        cfg = AudioClassifierModelConfig(
            model_name="m", model_location="org/model", model_categories=["a", "b"],
            positive_groups=[["a", "not_a_category"]],
        )
        w = AudioClassifierWrapper(cfg)
        assert w.can_run is False

    def test_missing_transformers_or_torch_marks_unrunnable_not_raises(self):
        # load_classifier() catches ImportError gracefully -- valid config, but
        # if torch/transformers aren't importable in this environment, construction
        # must not raise.
        cfg = AudioClassifierModelConfig(
            model_name="m", model_location="org/model", model_categories=["safe", "explicit"]
        )
        w = AudioClassifierWrapper(cfg)  # should not raise regardless of environment
        assert isinstance(w.can_run, bool)


# ---------------------------------------------------------------------------
# AudioClassifierWrapper.classify_audio -- split-positive-group logic
# ---------------------------------------------------------------------------

def _make_wrapper_for_logic_test(categories, positive_groups=None, neutral_categories=None, severity_order=None):
    """Build a wrapper instance for testing classify_audio's scoring logic in
    isolation, without touching torch/transformers (predict_audio is monkeypatched).

    can_run is forced True regardless of what load_classifier() actually determined --
    these tests exercise the split-positive scoring logic only, and must not depend on
    whether torch/transformers happen to be importable in whatever environment runs them.
    """
    cfg = AudioClassifierModelConfig(
        model_name="m",
        model_location="org/model",
        model_categories=categories,
        positive_groups=positive_groups or [],
        neutral_categories=neutral_categories or [],
        severity_order=severity_order or [],
    )
    w = AudioClassifierWrapper(cfg)
    w.can_run = True
    return w


class TestClassifyAudioSplitPositive:
    def test_no_positive_groups_picks_highest_score(self):
        w = _make_wrapper_for_logic_test(["safe", "explicit"])
        w.predict_audio = lambda path: {"safe": 0.2, "explicit": 0.8}
        assert w.classify_audio("/fake.mp3") == "explicit"

    def test_split_group_wins_over_neutral(self):
        w = _make_wrapper_for_logic_test(
            ["safe", "mild", "explicit"],
            positive_groups=[["mild", "explicit"]],
        )
        # derived neutral_categories = ["safe"]; combined positive mass (0.3+0.3=0.6)
        # clears neutral (0.4) by more than the 0.05 margin.
        w.predict_audio = lambda path: {"safe": 0.4, "mild": 0.3, "explicit": 0.3}
        assert w.classify_audio("/fake.mp3") in ("mild", "explicit")

    def test_severity_order_breaks_ties_within_group(self):
        w = _make_wrapper_for_logic_test(
            ["safe", "mild", "explicit"],
            positive_groups=[["mild", "explicit"]],
            severity_order=["explicit", "mild"],
        )
        w.predict_audio = lambda path: {"safe": 0.1, "mild": 0.45, "explicit": 0.45}
        assert w.classify_audio("/fake.mp3") == "explicit"

    def test_neutral_dominant_falls_through_to_argmax(self):
        w = _make_wrapper_for_logic_test(
            ["safe", "mild", "explicit"],
            positive_groups=[["mild", "explicit"]],
        )
        # neutral mass (0.9) dominates -- no split detected, falls through to argmax.
        w.predict_audio = lambda path: {"safe": 0.9, "mild": 0.05, "explicit": 0.05}
        assert w.classify_audio("/fake.mp3") == "safe"


class TestTestAudioForCategory:
    def test_above_threshold_true(self):
        w = _make_wrapper_for_logic_test(["safe", "explicit"])
        w.predict_audio = lambda path: {"safe": 0.2, "explicit": 0.8}
        assert w.test_audio_for_category("/fake.mp3", "explicit", 0.5) is True

    def test_below_threshold_false(self):
        w = _make_wrapper_for_logic_test(["safe", "explicit"])
        w.predict_audio = lambda path: {"safe": 0.2, "explicit": 0.8}
        assert w.test_audio_for_category("/fake.mp3", "explicit", 0.9) is False


class TestPredictAudioRanked:
    def test_sorted_descending(self):
        w = _make_wrapper_for_logic_test(["a", "b", "c"])
        w.predict_audio = lambda path: {"a": 0.1, "b": 0.7, "c": 0.2}
        ranked = w.predict_audio_ranked("/fake.mp3")
        assert [name for name, _score in ranked] == ["b", "c", "a"]


class TestWrapperDunderMethods:
    def test_equality_by_model_name(self):
        w1 = _make_wrapper_for_logic_test(["a"])
        w2 = _make_wrapper_for_logic_test(["a"])
        assert w1 == w2  # both named "m"

    def test_hash_by_model_name(self):
        w = _make_wrapper_for_logic_test(["a"])
        assert hash(w) == hash("m")

"""
Audio classification via Hugging Face ``transformers``.

Deliberately narrow first pass: only the general
``AutoFeatureExtractor`` / ``AutoModelForAudioClassification`` path is
supported (no custom architectures, no TensorFlow/H5 backend, no
multi-segment sampling for long files -- one fixed-length window per file).
See :mod:`image.image_classifier` for the much more general image
counterpart this deliberately does not try to match in scope.

Audio is decoded to a mono float32 waveform via an ``ffmpeg`` subprocess
(mirroring :class:`image.video_ops.VideoOps`'s pattern) rather than adding a
new audio-decoding dependency -- ``ffmpeg`` is already required for video
handling in this project.
"""

from __future__ import annotations

import subprocess
from typing import Dict, Optional

import numpy as np

from image.audio_classifier_model_config import AudioClassifierModelConfig
from image.classifier_prediction_cache import classifier_prediction_cache, model_signature
from image.classifier_utils import (
    derive_neutral_categories_from_positive_groups,
    format_prediction_line,
    from_pretrained_checked,
    logits_to_probabilities,
    map_scores_to_categories,
    pick_split_positive,
    top_category,
)
from image.video_ops import VideoOps
from utils.config import config
from utils.logging_setup import get_logger
from utils.subprocess_flags import NO_CONSOLE_WINDOW

logger = get_logger("audio_classifier")

# Split-positive assignment only applies when a group's combined mass exceeds
# aggregate neutral mass by more than this margin (probability points on the
# model's output scale). Kept separate from image_classifier.py's identical
# constant so the two classifier domains can be tuned independently.
_SPLIT_GROUP_OVER_NEUTRAL_MARGIN = 0.05


def decode_audio_waveform(audio_path: str, sample_rate: int, max_duration_seconds: float) -> np.ndarray:
    """Decode *audio_path* to a mono float32 PCM waveform at *sample_rate*, capped to
    *max_duration_seconds*.

    Raises RuntimeError if ffmpeg is missing, the file can't be decoded, or
    decoding produces no samples.
    """
    ffmpeg = VideoOps.find_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; required to decode audio for classification")

    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", audio_path,
        "-t", str(max_duration_seconds),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=120, creationflags=NO_CONSOLE_WINDOW,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out decoding audio: {audio_path}") from e
    except OSError as e:
        raise RuntimeError(f"Failed to run ffmpeg: {e}") from e

    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"ffmpeg audio decode failed for {audio_path}{detail}")

    waveform = np.frombuffer(proc.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError(f"ffmpeg produced no audio samples for {audio_path}")
    return waveform


class AudioClassifierWrapper:
    def __init__(self, model_config: AudioClassifierModelConfig):
        """Load and run a Hugging Face audio classifier from a single
        :class:`AudioClassifierModelConfig`.

        Split-positive options (``positive_groups``, ``neutral_categories``, ``severity_order``)
        are read from ``model_config``, with the same semantics as the image classifier: if
        ``positive_groups`` is non-empty and ``neutral_categories`` is omitted or empty,
        neutrals are derived as the complement of the union of all positive groups within
        ``model_categories``.
        """
        self.model_name = model_config.model_name
        self.model_categories = list(model_config.model_categories)
        self.model_location = model_config.model_location
        self.model_kwargs = dict(model_config.model_kwargs)
        self.sample_rate = int(model_config.sample_rate)
        self.max_duration_seconds = float(model_config.max_duration_seconds)
        self.positive_groups = [list(g) for g in model_config.positive_groups]
        self.neutral_categories = list(model_config.neutral_categories)
        self.severity_order = list(model_config.severity_order)

        if self.positive_groups and not self.neutral_categories:
            self.neutral_categories = derive_neutral_categories_from_positive_groups(
                self.model_categories, self.positive_groups
            )
            if config.debug2 and self.neutral_categories:
                logger.debug(
                    f"Derived neutral_categories for {self.model_name}: {self.neutral_categories}"
                )

        self.can_run = True
        self.device = None
        self.feature_extractor = None
        self.model = None
        self.predictions_cache: Dict[str, Dict[str, float]] = {}
        self._prediction_signature: Optional[str] = None

        if self.can_run:
            try:
                self.model_name = str(self.model_name).strip()
                if not self.model_name:
                    raise Exception("Invalid model name: " + self.model_name)
                if not isinstance(self.model_categories, list) or len(self.model_categories) == 0 \
                        or any(type(c) != str for c in self.model_categories):
                    raise Exception(f"Invalid model categories: {self.model_categories}")
                if not isinstance(self.model_location, str) or not self.model_location.strip():
                    raise Exception(f"Invalid model location: {self.model_location}")
                allowed = set(self.model_categories)
                if self.positive_groups:
                    if not isinstance(self.positive_groups, list):
                        raise Exception(f"positive_groups must be a list, got {type(self.positive_groups)}")
                    for grp in self.positive_groups:
                        if not isinstance(grp, list):
                            raise Exception(f"positive_groups entries must be lists, got {type(grp)}")
                        for c in grp:
                            if c not in allowed:
                                raise Exception(
                                    f"positive_groups references unknown category {c!r} "
                                    f"(not in model_categories)"
                                )
                for c in self.neutral_categories:
                    if c not in allowed:
                        raise Exception(
                            f"neutral_categories references unknown category {c!r} "
                            f"(not in model_categories)"
                        )
            except Exception:
                self.can_run = False
                logger.exception(
                    "Audio classifier %r: config validation failed before load (location=%r, "
                    "positive_groups=%s, neutral_categories=%s)",
                    self.model_name,
                    self.model_location,
                    self.positive_groups,
                    self.neutral_categories,
                )
            if self.can_run:
                self.load_classifier()

    def load_classifier(self) -> None:
        assert self.can_run is True
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as e:
            self.can_run = False
            logger.error(f"transformers/torch not installed, cannot load audio classifier {self.model_name}: {e}")
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_location)
            self.model = from_pretrained_checked(
                AutoModelForAudioClassification, self.model_location, self.model_name,
                log=logger, **self.model_kwargs
            ).to(self.device)
            self.model.eval()

            # The feature extractor's own sampling rate always wins over the
            # configured value -- decoding at the wrong rate silently degrades
            # every prediction rather than raising, so this is worth getting
            # right automatically rather than trusting user configuration.
            fe_rate = getattr(self.feature_extractor, "sampling_rate", None)
            if fe_rate:
                if int(fe_rate) != self.sample_rate:
                    logger.info(
                        "Audio classifier %r: overriding configured sample_rate=%s with "
                        "feature extractor's sampling_rate=%s",
                        self.model_name, self.sample_rate, fe_rate,
                    )
                self.sample_rate = int(fe_rate)

            self.can_run = True
            logger.info(f"Loaded audio classifier {self.model_name!r} from {self.model_location!r} on {self.device}")
        except Exception as e:
            self.can_run = False
            self.feature_extractor = None
            self.model = None
            logger.error(f"Failed to load audio classifier {self.model_name!r} from {self.model_location!r}: {e}")

    def _persisted_prediction_key(self) -> tuple:
        """(model key, settings signature) for classifier_prediction_cache."""
        if self._prediction_signature is None:
            self._prediction_signature = model_signature({
                "location": self.model_location,
                "categories": self.model_categories,
                "model_kwargs": self.model_kwargs,
                "sample_rate": self.sample_rate,
                "max_duration_seconds": self.max_duration_seconds,
            })
        return f"audio:{self.model_name}", self._prediction_signature

    def discard_cached_prediction(self, audio_path: str) -> None:
        """Forget the session and persisted scores for *audio_path*."""
        self.predictions_cache.pop(audio_path, None)
        model_key, _sig = self._persisted_prediction_key()
        classifier_prediction_cache.discard(model_key, audio_path)

    def predict_audio(self, audio_path: str) -> Dict[str, float]:
        if audio_path in self.predictions_cache:
            return self.predictions_cache[audio_path]

        model_key, signature = self._persisted_prediction_key()
        persisted = classifier_prediction_cache.get(
            model_key, signature, self.model_location, audio_path)
        if persisted is not None:
            self.predictions_cache[audio_path] = persisted
            return dict(persisted)

        if self.model is None or self.feature_extractor is None:
            raise ValueError("Classifier not initialized")

        import torch

        waveform = decode_audio_waveform(audio_path, self.sample_rate, self.max_duration_seconds)
        max_length = int(self.sample_rate * self.max_duration_seconds)
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits if hasattr(output, "logits") else output[0]
        scores = logits_to_probabilities(logits.float().cpu().numpy())[0]
        classed_predictions = map_scores_to_categories(
            scores, self.model_categories, self.model_name, log=logger)

        self.predictions_cache[audio_path] = dict(classed_predictions)
        classifier_prediction_cache.put(
            model_key, signature, self.model_location, audio_path, classed_predictions)
        return classed_predictions

    def predict_audio_ranked(self, audio_path: str) -> list[tuple[str, float]]:
        """Return (category, score) pairs sorted by score descending (rank 1 = highest)."""
        return sorted(self.predict_audio(audio_path).items(), key=lambda kv: kv[1], reverse=True)

    def classify_audio(self, audio_path: str) -> str:
        if not self.can_run:
            raise Exception("Invalid state: Audio classifier failed to initialize, unable to classify audio")

        classed_predictions = self.predict_audio(audio_path)

        if self.positive_groups:
            split = pick_split_positive(
                classed_predictions, self.positive_groups, self.neutral_categories,
                self.severity_order, _SPLIT_GROUP_OVER_NEUTRAL_MARGIN,
            )
            if split is not None:
                best_category, best_group, best_combined_prob = split
                if config.debug2:
                    logger.debug(
                        f"Audio classifier prediction map ({self.model_name}): "
                        f"{format_prediction_line(classed_predictions)}"
                    )
                    logger.debug(
                        f"Split positive in group '{'+'.join(best_group)}': "
                        f"combined={best_combined_prob:.6f}, assigned={best_category}"
                    )
                return best_category

        classed_category = top_category(classed_predictions, self.model_categories)
        if config.debug2:
            logger.debug(
                f"Audio classifier prediction map ({self.model_name}): "
                f"{format_prediction_line(classed_predictions)}"
            )
        return classed_category

    def test_audio_for_categories(self, audio_path: str, categories) -> bool:
        if not self.can_run:
            raise Exception("Invalid state: Audio classifier failed to initialize, unable to classify audio")
        category = self.classify_audio(audio_path)
        return category in categories

    def test_audio_for_category(self, audio_path: str, category: str, threshold: float) -> bool:
        if self.can_run:
            return self.predict_audio(audio_path).get(category, 0.0) > threshold
        raise Exception("Invalid state: Audio classifier failed to initialize, unable to classify audio")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', categories={self.model_categories})"

    def __hash__(self) -> int:
        return hash(self.model_name)

    def __eq__(self, other):
        if not isinstance(other, AudioClassifierWrapper):
            raise TypeError(f"Invalid type for comparison: {type(other)}")
        return self.model_name == other.model_name

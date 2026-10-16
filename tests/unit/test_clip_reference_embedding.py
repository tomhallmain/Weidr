"""Fresh CLIP embeddings of compare.clip_reference's image against the recorded
reference, through both loaders in compare.model. Skipped where the weights
are not downloaded yet (ViT-B/32 is about 340 MB) or the reference has not
been generated."""

import os

import pytest

from compare import clip_reference


def _weights_path():
    from clip.clip import _MODELS
    url = _MODELS[clip_reference.REFERENCE_MODEL]
    return os.path.join(os.path.expanduser("~/.cache/clip"), os.path.basename(url))


pytestmark = [
    pytest.mark.skipif(not os.path.isfile(clip_reference.reference_path()),
                       reason="run scripts/generate_clip_reference_embedding.py first"),
    pytest.mark.skipif(not os.path.isfile(_weights_path()),
                       reason="CLIP ViT-B/32 weights not downloaded"),
]


def test_reference_names_the_reference_model():
    assert clip_reference.read_reference()["model"] == clip_reference.REFERENCE_MODEL


def test_load_clip_matches_reference():
    assert clip_reference.check() == []


def test_loading_without_torchscript_matches_reference():
    # The packaged build's path (PYTORCH_JIT=0), run here with TorchScript on.
    from compare import model
    assert clip_reference.check(load=model.load_clip_without_torchscript) == []

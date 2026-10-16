"""A recorded CLIP embedding of a generated image, compared against a fresh one
to check that the weights load and give the same result in this environment
(a packaged build, another torch or clip version). The reference lives in
assets/clip_reference_embedding.json and is written from a source checkout by
scripts/generate_clip_reference_embedding.py.
"""

import json
import math
import os

from PIL import Image

from utils.repo_paths import resource_root

REFERENCE_PATH_PARTS = ("assets", "clip_reference_embedding.json")
REFERENCE_MODEL = "ViT-B/32"
# On the CPU in float32, the same weights give the same embedding up to rounding
# that varies with the torch version and CPU; weights loaded wrongly give a
# far lower similarity.
MIN_COSINE_SIMILARITY = 0.999
_IMAGE_SIZE = 224


def reference_path() -> str:
    return os.path.join(resource_root(), *REFERENCE_PATH_PARTS)


def reference_image() -> Image.Image:
    """A 224x224 RGB pattern computed pixel by pixel, so it is the same
    everywhere and needs no resizing by ViT-B/32's preprocess."""
    pixels = bytes(
        channel
        for y in range(_IMAGE_SIZE)
        for x in range(_IMAGE_SIZE)
        for channel in (x, y, x ^ y)
    )
    return Image.frombytes("RGB", (_IMAGE_SIZE, _IMAGE_SIZE), pixels)


def compute_embedding(model_name: str = REFERENCE_MODEL, load=None) -> list[float]:
    """Normalized image embedding of reference_image(), computed on the CPU.
    *load* defaults to compare.model.load_clip."""
    import torch
    from compare import model

    model_, preprocess = (load or model.load_clip)(model_name, device="cpu")
    with torch.no_grad():
        embedding = model_.encode_image(preprocess(reference_image()).unsqueeze(0))
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding[0].tolist()


def read_reference() -> dict:
    with open(reference_path(), encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def check(load=None) -> list[str]:
    """Problems found comparing a fresh embedding with the reference; empty
    when there are none."""
    try:
        reference = read_reference()
    except (OSError, ValueError) as e:
        return [f"CLIP reference embedding unreadable: {e}"]
    expected = reference["embedding"]
    actual = compute_embedding(reference["model"], load)
    if len(actual) != len(expected):
        return [f"CLIP embedding has {len(actual)} values, the reference {len(expected)}"]
    similarity = cosine_similarity(actual, expected)
    if similarity < MIN_COSINE_SIMILARITY:
        return [f"CLIP embedding differs from the reference: cosine similarity "
                f"{similarity:.6f}, expected at least {MIN_COSINE_SIMILARITY}"]
    return []

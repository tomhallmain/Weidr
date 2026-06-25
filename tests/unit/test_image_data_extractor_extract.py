"""Unit tests for ImageDataExtractor.extract() — ComfyUI prompt graph traversal."""

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image.image_data_extractor import ImageDataExtractor, image_data_extractor


def _make_png_with_prompt(tmp_dir: str, prompt: dict) -> str:
    path = Path(tmp_dir) / "test.png"
    pnginfo = PngInfo()
    pnginfo.add_text("prompt", json.dumps(prompt))
    Image.new("RGB", (8, 8)).save(str(path), pnginfo=pnginfo)
    return str(path)


# Minimal KSampler prompt: positive and negative connect directly to CLIPTextEncode nodes.
DIRECT_KSAMPLER_PROMPT = {
    "1": {
        "class_type": "KSampler",
        "inputs": {
            "positive": ["2", 0],
            "negative": ["3", 0],
            "seed": 42,
        },
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a photo of a cat"},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "blurry, low quality"},
    },
}

# CFGGuider prompt where positive/negative route through ReferenceLatent before
# reaching CLIPTextEncode — the case that was previously unhandled.
REFERENCE_LATENT_PROMPT = {
    "1": {
        "class_type": "CFGGuider",
        "inputs": {
            "positive": ["2", 0],
            "negative": ["3", 0],
        },
    },
    "2": {
        "class_type": "ReferenceLatent",
        "inputs": {
            "conditioning": ["4", 0],
            "latent": ["6", 0],
        },
    },
    "3": {
        "class_type": "ReferenceLatent",
        "inputs": {
            "conditioning": ["5", 0],
            "latent": ["6", 0],
        },
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "positive prompt via reference latent"},
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "negative prompt via reference latent"},
    },
    "6": {
        "class_type": "VAEEncode",
        "inputs": {},
    },
}


def test_extract_direct_ksampler():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, DIRECT_KSAMPLER_PROMPT)
        positive, negative = image_data_extractor.extract(path)
    assert positive == "a photo of a cat"
    assert negative == "blurry, low quality"


def test_extract_conditioning_via_reference_latent():
    """CFGGuider → ReferenceLatent → CLIPTextEncode chains are resolved correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, REFERENCE_LATENT_PROMPT)
        positive, negative = image_data_extractor.extract(path)
    assert positive == "positive prompt via reference latent"
    assert negative == "negative prompt via reference latent"


def test_extract_conditioning_via_reference_latent_empty_negative():
    """Empty negative prompt is returned as an empty string, not None."""
    prompt = {
        "1": {
            "class_type": "CFGGuider",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
            },
        },
        "2": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["4", 0]},
        },
        "3": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["5", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "detailed scene description"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ""},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, prompt)
        positive, negative = image_data_extractor.extract(path)
    assert positive == "detailed scene description"
    assert negative == ""

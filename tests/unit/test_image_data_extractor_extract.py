"""Unit tests for ImageDataExtractor.extract() prompt graph traversal."""

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


def _krea2_switch_prompt(lora_enabled, trigger=""):
    """KSampler positive text is CLIPTextEncode → ComfySwitchNode.

    ``on_false`` is the user prompt primitive. ``on_true`` concatenates that
    prompt with a LoRA trigger word. Matches image_krea2_turbo_t2i.json.
    """
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["13", 0],
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["28", 0]},
        },
        "13": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality"},
        },
        "19": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "drawing, people doing bravo !\n\n"},
        },
        "23": {
            "class_type": "PrimitiveBoolean",
            "inputs": {"value": lora_enabled},
        },
        "27": {
            "class_type": "StringConcatenate",
            "inputs": {
                "string_a": ["19", 0],
                "string_b": trigger,
                "delimiter": ", ",
            },
        },
        "28": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["23", 0],
                "on_false": ["19", 0],
                "on_true": ["27", 0],
            },
        },
    }


def test_extract_positive_via_switch_to_primitive_string():
    """LoRA switch off: positive prompt is the PrimitiveStringMultiline behind the switch."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, _krea2_switch_prompt(False))
        positive, negative = image_data_extractor.extract(path)
    assert positive == "drawing, people doing bravo !\n\n"
    assert negative == "blurry, low quality"


def test_extract_positive_via_switch_to_string_concatenate():
    """LoRA switch on: positive prompt is string_a + delimiter + string_b."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, _krea2_switch_prompt(True, trigger="darkbrush"))
        positive, negative = image_data_extractor.extract(path)
    assert positive == "drawing, people doing bravo !\n\n, darkbrush"
    assert negative == "blurry, low quality"


def test_extract_clip_text_linked_directly_to_primitive_string():
    """A one-hop link from CLIPTextEncode to PrimitiveStringMultiline still resolves."""
    prompt = {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["4", 0]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ""},
        },
        "4": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "hello from primitive"},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, prompt)
        positive, negative = image_data_extractor.extract(path)
    assert positive == "hello from primitive"
    assert negative == ""


def test_extract_clip_text_via_preview_any_and_switch():
    """PreviewAny forwards ``source``; the switch then selects the primitive string."""
    prompt = {
        "1": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["4", 0]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "negative stays literal"},
        },
        "4": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["5", 0]},
        },
        "5": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": False,
                "on_false": ["6", 0],
                "on_true": ["7", 0],
            },
        },
        "6": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "prompt before upsampling"},
        },
        "7": {
            "class_type": "TextGenerate",
            "inputs": {"prompt": ["6", 0]},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, prompt)
        positive, negative = image_data_extractor.extract(path)
    assert positive == "prompt before upsampling"
    assert negative == "negative stays literal"


USER_PROMPT = (
    "A high-resolution, surreal digital illustration showing a human hand holding a martini glass."
)
SYSTEM_PROMPT = "You are an expert prompt engineer.\n\nUser's Input:\n"


def _krea2_template_prompt(refine=True, lora_enabled=False, trigger="muted minimalist sketch style", user=USER_PROMPT):
    """Official Krea-2 Turbo template shape.

    Prompt enhancement (``TextGenerate``) is on by default. Its output is not
    stored in the prompt graph. The negative input is ``ConditioningZeroOut``.
    """
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {"positive": ["6", 0], "negative": ["13", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ["28", 0]},
        },
        "13": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "16": {
            "class_type": "TextGenerate",
            "inputs": {"prompt": ["17", 0]},
        },
        "17": {
            "class_type": "StringConcatenate",
            "inputs": {
                "string_a": ["18", 0],
                "string_b": ["19", 0],
                "delimiter": "",
            },
        },
        "18": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": SYSTEM_PROMPT},
        },
        "19": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": user},
        },
        "20": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["21", 0]},
        },
        "21": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["24", 0],
                "on_false": ["19", 0],
                "on_true": ["16", 0],
            },
        },
        "23": {
            "class_type": "PrimitiveBoolean",
            "inputs": {"value": lora_enabled},
        },
        "24": {
            "class_type": "PrimitiveBoolean",
            "inputs": {"value": refine},
        },
        "27": {
            "class_type": "StringConcatenate",
            "inputs": {
                "string_a": ["20", 0],
                "string_b": trigger,
                "delimiter": ", ",
            },
        },
        "28": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "switch": ["23", 0],
                "on_false": ["20", 0],
                "on_true": ["27", 0],
            },
        },
    }


def test_extract_krea_template_refine_on_uses_user_prompt():
    """Prompt enhancement output is not in the graph; the typed prompt is."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, _krea2_template_prompt(refine=True, lora_enabled=False))
        positive, negative = image_data_extractor.extract(path)
    assert positive == USER_PROMPT
    assert SYSTEM_PROMPT not in positive
    assert negative == ""


def test_extract_krea_template_refine_on_with_lora_trigger():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(
            tmp, _krea2_template_prompt(refine=True, lora_enabled=True, trigger="rainy window style")
        )
        positive, negative = image_data_extractor.extract(path)
    assert positive == USER_PROMPT + ", rainy window style"
    assert negative == ""


def test_extract_krea_template_refine_off():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, _krea2_template_prompt(refine=False, lora_enabled=False))
        positive, negative = image_data_extractor.extract(path)
    assert positive == USER_PROMPT
    assert negative == ""


def test_empty_selected_switch_branch_does_not_use_the_other_branch():
    """An empty typed prompt stays empty. Only an unstored generator falls back."""
    prompt = _krea2_switch_prompt(False, trigger="darkbrush")
    prompt["19"]["inputs"]["value"] = ""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, prompt)
        positive, negative = image_data_extractor.extract(path)
    assert positive == ""
    assert negative == "blurry, low quality"


def test_extract_saved_krea_template_metadata():
    meta_path = Path(__file__).resolve().parents[1] / "fixtures" / "prompt_graphs" / "krea2_turbo_template_prompt.json"
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    user_prompt = saved["prompt"]["30:19"]["inputs"]["value"]
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, saved["prompt"])
        positive, negative = image_data_extractor.extract(path)
    assert positive == user_prompt
    assert negative == ""


def test_extract_reference_multi_latent_slots():
    """KSampler positive and negative are two outputs of the same ReferenceMultiLatent node."""
    prompt = {
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["17", 0],
                "negative": ["17", 1],
            },
        },
        "17": {
            "class_type": "ReferenceMultiLatent",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["58", 0],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "Make the cat into a dragon instead."},
        },
        "58": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": ""},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, prompt)
        positive, negative = image_data_extractor.extract(path)
    assert positive == "Make the cat into a dragon instead."
    assert negative == ""


def test_extract_reference_multi_latent_negative_echoes_positive_when_unconnected():
    prompt = {
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "positive": ["17", 0],
                "negative": ["17", 1],
            },
        },
        "17": {
            "class_type": "ReferenceMultiLatent",
            "inputs": {
                "positive": ["4", 0],
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "shared conditioning"},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, prompt)
        positive, negative = image_data_extractor.extract(path)
    assert positive == "shared conditioning"
    assert negative == "shared conditioning"


def test_extract_saved_flux_klein_edit_template_metadata():
    meta_path = Path(__file__).resolve().parents[1] / "fixtures" / "prompt_graphs" / "flux_klein_edit_template_prompt.json"
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_png_with_prompt(tmp, saved["prompt"])
        positive, negative = image_data_extractor.extract(path)
    assert positive == "Make the cat into a dragon instead."
    assert negative == ""

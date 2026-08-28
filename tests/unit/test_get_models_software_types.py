"""Unit tests for ImageDataExtractor.get_models() software-type handling.

SoftwareType.OTHER is a normal outcome, not an error: extract_prompt returns
it when a ComfyUI "prompt" key holds text that isn't valid JSON. get_models
used to fall through to a raise for that case, and the raise itself was
broken -- it concatenated the enum onto a str, so the caller saw
"can only concatenate str (not \"SoftwareType\") to str" instead of the
intended message, logged as an error for every such file.
"""

import json
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image.image_data_extractor import SoftwareType, image_data_extractor


def _make_png(tmp_dir, key: str, value: str) -> str:
    path = Path(tmp_dir) / "test.png"
    pnginfo = PngInfo()
    pnginfo.add_text(key, value)
    Image.new("RGB", (8, 8)).save(str(path), pnginfo=pnginfo)
    return str(path)


CHECKPOINT_PROMPT = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "somemodel.safetensors"},
    },
    "2": {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "somelora.safetensors", "strength_model": 0.8},
    },
}


class TestGetModelsSoftwareTypes:
    def test_comfyui_models_and_loras_extracted(self, tmp_path):
        path = _make_png(tmp_path, "prompt", json.dumps(CHECKPOINT_PROMPT))
        models, loras = image_data_extractor.get_models(path)
        assert models == ["somemodel"]
        assert loras == ["somelora"]

    def test_non_json_prompt_returns_empty_without_raising(self, tmp_path):
        """The reported failure: a 'prompt' key holding text that isn't JSON."""
        path = _make_png(tmp_path, "prompt", "this is not json")
        # Precondition -- this input is what produces SoftwareType.OTHER.
        _prompt, software_type = image_data_extractor.extract_prompt(path)
        assert software_type == SoftwareType.OTHER

        assert image_data_extractor.get_models(path) == ([], [])

    def test_no_metadata_returns_empty(self, tmp_path):
        path = Path(tmp_path) / "plain.png"
        Image.new("RGB", (8, 8)).save(str(path))
        assert image_data_extractor.get_models(str(path)) == ([], [])

    def test_unknown_software_type_raises_readable_error(self, tmp_path, monkeypatch):
        """A genuinely unhandled type must still raise -- but legibly.

        Guards the formatting itself: concatenating a non-str type onto a str
        raised TypeError and lost the message entirely.
        """
        path = _make_png(tmp_path, "prompt", json.dumps(CHECKPOINT_PROMPT))
        monkeypatch.setattr(
            image_data_extractor, "extract_prompt",
            lambda *args, **kwargs: ({}, "a-future-software-type"),
        )
        with pytest.raises(Exception) as excinfo:
            image_data_extractor.get_models(path)
        assert not isinstance(excinfo.value, TypeError)
        assert "Unhandled software type" in str(excinfo.value)
        assert "a-future-software-type" in str(excinfo.value)

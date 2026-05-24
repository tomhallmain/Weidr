"""Unit tests for rerun positive prompt extraction."""

import tempfile
from pathlib import Path

from image.image_data_extractor import ImageDataExtractor, image_data_extractor
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def test_extract_positive_prompt_for_rerun_prefers_sdr_original_tags():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.png"
        pnginfo = PngInfo()
        pnginfo.add_text(
            ImageDataExtractor.SDR_ORIGINAL_POSITIVE_TAGS_KEY,
            "original tags from sdr",
        )
        Image.new("RGB", (8, 8), color="red").save(path, pnginfo=pnginfo)

        positive, negative = image_data_extractor.extract_positive_prompt_for_rerun(
            str(path)
        )

        assert positive == "original tags from sdr"
        assert negative == ""


def test_extract_positive_prompt_for_rerun_falls_back_without_sdr_key():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.png"
        Image.new("RGB", (8, 8), color="blue").save(path)

        positive, negative = image_data_extractor.extract_positive_prompt_for_rerun(
            str(path)
        )

        assert isinstance(positive, str)
        assert isinstance(negative, str)

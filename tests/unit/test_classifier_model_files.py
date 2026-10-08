"""
Unit tests for image/classifier_model_files.py: which repo files count as model
weights and which one the Model Manager pre-selects.
"""
from __future__ import annotations

from image.classifier_model_files import MODEL_FILE_EXTENSIONS, is_model_file, model_file_choices


def test_tflite_is_a_model_file_extension():
    assert ".tflite" in MODEL_FILE_EXTENSIONS
    assert is_model_file("person_classification_flash(448x640).tflite")


def test_extension_check_is_case_insensitive():
    assert is_model_file("Model.SafeTensors")
    assert not is_model_file(".gitattributes")
    assert not is_model_file("README.md")


def test_repo_without_model_files_offers_everything_and_preselects_nothing():
    files = [".gitattributes", "README.md", "inference_example.py"]
    choices, default = model_file_choices(files)
    assert choices == files
    assert default is None


def test_root_level_model_file_is_preferred_over_checkpoints():
    files = [
        ".gitattributes",
        "checkpoint-489/model.safetensors",
        "checkpoint-489/optimizer.pt",
        "config.json",
        "model.safetensors",
        "training_args.bin",
    ]
    choices, default = model_file_choices(files)
    assert default == "model.safetensors"
    assert choices[:2] == ["model.safetensors", "training_args.bin"]
    assert "config.json" not in choices


def test_custom_extension_set_is_respected():
    choices, default = model_file_choices(["a.onnx", "b.h5"], extensions={".h5"})
    assert choices == ["b.h5"]
    assert default == "b.h5"


def test_empty_file_list():
    assert model_file_choices([]) == ([], None)

"""Choosing which file in a Hugging Face model repo is the classifier's weights file."""

from __future__ import annotations

import os
from typing import Iterable, Optional

MODEL_FILE_EXTENSIONS = frozenset({
    ".safetensors",
    ".ckpt",
    ".bin",
    ".onnx",
    ".pt",
    ".pth",
    ".h5",
    ".keras",
    ".tflite",
})


def is_model_file(filename: str, extensions: Iterable[str] = MODEL_FILE_EXTENSIONS) -> bool:
    """Whether *filename* has one of *extensions* (case-insensitive)."""
    return os.path.splitext(filename)[1].lower() in set(extensions)


def model_file_choices(
    files: Iterable[str],
    extensions: Iterable[str] = MODEL_FILE_EXTENSIONS,
) -> tuple[list[str], Optional[str]]:
    """(choices to offer, file to pre-select) for a repo's file list.

    Model files are offered with repo-root files first, since subdirectories can
    hold training checkpoints or alternate exports. When the repo has no model
    file, every file is offered and nothing is pre-selected, so a README or
    .gitattributes is never picked by default.
    """
    files = list(files)
    model_files = [f for f in files if is_model_file(f, extensions)]
    if not model_files:
        return files, None
    model_files.sort(key=lambda f: (f.count("/"), f.lower()))
    return model_files, model_files[0]

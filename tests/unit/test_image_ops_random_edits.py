"""Output paths for the random edits.

An explicit output_path is honoured (the UI preview flow renders into a temp
file that way), and the default destination never overwrites an earlier
result -- the behaviour the box / background-box edits already had.
"""

import os
import random

from PIL import Image

from image.image_ops import ImageOps


def _png(path, size=(64, 64), color=(90, 120, 150)) -> str:
    Image.new("RGB", size, color).save(str(path), format="PNG")
    return str(path)


def test_scramble_writes_to_an_explicit_output_path(tmp_path):
    source = _png(tmp_path / "a.png")
    out = str(tmp_path / "explicit.png")

    result = ImageOps.scramble_image(source, output_path=out)

    assert result == out
    assert os.path.isfile(out)


def test_scramble_twice_keeps_both_results(tmp_path):
    source = _png(tmp_path / "a.png")

    first = ImageOps.scramble_image(source)
    second = ImageOps.scramble_image(source)

    assert os.path.basename(first) == "a" + ImageOps.SCRAMBLE_SUFFIX + ".png"
    assert first != second
    assert os.path.isfile(first)
    assert os.path.isfile(second)


def test_semi_scramble_twice_keeps_both_results(tmp_path):
    source = _png(tmp_path / "a.png")

    first = ImageOps.semi_scramble_image(source)
    second = ImageOps.semi_scramble_image(source)

    assert os.path.basename(first) == "a" + ImageOps.SEMI_SCRAMBLE_SUFFIX + ".png"
    assert first != second
    assert os.path.isfile(first)
    assert os.path.isfile(second)


def test_random_modification_writes_to_an_explicit_output_path(tmp_path):
    random.seed(20260912)
    source = _png(tmp_path / "b.png")
    out = str(tmp_path / "explicit_edit.png")

    result = ImageOps.randomly_modify_image(source, output_path=out)

    assert result == out
    assert os.path.isfile(out)


def test_random_modification_twice_keeps_both_results(tmp_path):
    random.seed(20260912)
    source = _png(tmp_path / "b.png")

    first = ImageOps.randomly_modify_image(source)
    second = ImageOps.randomly_modify_image(source)

    assert os.path.basename(first) == "b" + ImageOps.RANDOM_EDIT_SUFFIX + ".png"
    assert first != second
    assert os.path.isfile(first)
    assert os.path.isfile(second)

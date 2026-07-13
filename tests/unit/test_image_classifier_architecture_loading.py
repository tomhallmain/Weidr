"""
Unit tests for BaseImageClassifier._import_from_file (image.image_classifier).

Covers the case where multiple third-party classifier architecture files share
a generic basename (e.g. two different HF repos each shipping their own
"model_architecture.py") — the loader must resolve each to its own distinct
class, and never let the second load stomp on the first via a shared
sys.modules / sys.path identity, regardless of load order.

Uses only plain files on disk; no torch/tensorflow classifiers are
instantiated, so this does not require those optional heavy dependencies.
"""
import os
import sys

from image.image_classifier import BaseImageClassifier

_MODULE_SOURCE = """
class Foo:
    tag = {tag!r}
"""


def _write_module(directory, tag):
    path = os.path.join(directory, "model_architecture.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_MODULE_SOURCE.format(tag=tag))
    return path


def test_same_basename_in_different_directories_resolve_independently(tmp_path):
    dir_a = tmp_path / "repo_a"
    dir_b = tmp_path / "repo_b"
    dir_a.mkdir()
    dir_b.mkdir()

    path_a = _write_module(str(dir_a), "from_a")
    path_b = _write_module(str(dir_b), "from_b")

    class_a = BaseImageClassifier._import_from_file(path_a, "Foo")
    class_b = BaseImageClassifier._import_from_file(path_b, "Foo")

    assert class_a.tag == "from_a"
    assert class_b.tag == "from_b"
    assert class_a is not class_b


def test_load_order_does_not_change_result(tmp_path):
    """Loading B before A must not change what A resolves to (no shared cache key)."""
    dir_a = tmp_path / "repo_a"
    dir_b = tmp_path / "repo_b"
    dir_a.mkdir()
    dir_b.mkdir()

    path_a = _write_module(str(dir_a), "from_a")
    path_b = _write_module(str(dir_b), "from_b")

    class_b = BaseImageClassifier._import_from_file(path_b, "Foo")
    class_a = BaseImageClassifier._import_from_file(path_a, "Foo")

    assert class_b.tag == "from_b"
    assert class_a.tag == "from_a"


def test_reloading_same_file_is_stable(tmp_path):
    """Loading the exact same file twice should keep returning a working, correct class."""
    dir_a = tmp_path / "repo_a"
    dir_a.mkdir()
    path_a = _write_module(str(dir_a), "from_a")

    first = BaseImageClassifier._import_from_file(path_a, "Foo")
    second = BaseImageClassifier._import_from_file(path_a, "Foo")

    assert first.tag == "from_a"
    assert second.tag == "from_a"


def test_each_load_gets_a_unique_sys_modules_entry(tmp_path):
    dir_a = tmp_path / "repo_a"
    dir_b = tmp_path / "repo_b"
    dir_a.mkdir()
    dir_b.mkdir()

    path_a = _write_module(str(dir_a), "from_a")
    path_b = _write_module(str(dir_b), "from_b")

    before = set(sys.modules.keys())
    BaseImageClassifier._import_from_file(path_a, "Foo")
    BaseImageClassifier._import_from_file(path_b, "Foo")
    after = set(sys.modules.keys())

    new_keys = [k for k in (after - before) if k.startswith("model_architecture_")]
    assert len(new_keys) == 2, f"Expected two distinct module_architecture_* entries, got {new_keys}"

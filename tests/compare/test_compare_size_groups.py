"""SIZE group runs (CompareSize.run_comparison), headless with real images.

Files group by dimensions within the pixel tolerance (the compare threshold),
measured against each group's first member; groups of one are left out.
"""

import os

from PIL import Image

from compare.compare_args import CompareArgs
from compare.compare_manager import CompareManager
from compare.compare_result import CompareResult
from utils.constants import CompareMode, Mode
from utils.headless_app_actions import build_headless_app_actions
from utils.ui_responsiveness import NullResponsiveness


def _png(directory, name, size) -> str:
    path = os.path.join(directory, name)
    Image.new("RGB", size, (100, 100, 100)).save(path, format="PNG")
    return path


def _run(base_dir, tolerance, store_checkpoints=False):
    app_actions = build_headless_app_actions({"get_base_dir": lambda: base_dir})
    manager = CompareManager(
        master=None, app_actions=app_actions,
        get_base_dir=lambda: base_dir, responsiveness=NullResponsiveness(),
    )
    manager.set_compare_mode(CompareMode.SIZE)
    manager.set_threshold(tolerance)
    manager.set_store_checkpoints(store_checkpoints)
    args = CompareArgs(base_dir=base_dir, mode=Mode.GROUP, compare_mode=CompareMode.SIZE,
                       recursive=False, app_actions=app_actions)
    manager.apply_settings_to_args(args)
    manager.run(args)
    return [set(g) for g in manager._primary_wrapper().file_groups.values()]


def test_exact_sizes_group_and_a_single_size_is_left_out(tmp_path):
    d = str(tmp_path)
    small = [_png(d, "a.png", (48, 48)), _png(d, "b.png", (48, 48))]
    wide = [_png(d, "c.png", (64, 32)), _png(d, "d.png", (64, 32))]
    lone = _png(d, "e.png", (96, 96))

    groups = _run(d, tolerance=0)

    assert set(small) in groups
    assert set(wide) in groups
    assert not any(lone in g for g in groups)


def test_tolerance_is_measured_from_the_first_member(tmp_path):
    """48 and 56 are within 10 of each other, 56 and 64 too, but 64 is 16
    from 48: it does not join their group, and is left alone."""
    d = str(tmp_path)
    a = _png(d, "a.png", (48, 48))
    b = _png(d, "b.png", (56, 56))
    c = _png(d, "c.png", (64, 64))

    groups = _run(d, tolerance=10)

    assert {a, b} in groups
    assert not any(c in g for g in groups)


def test_a_completed_run_stores_a_checkpoint(tmp_path):
    d = str(tmp_path)
    _png(d, "a.png", (48, 48))
    _png(d, "b.png", (48, 48))

    _run(d, tolerance=0, store_checkpoints=True)

    path = CompareResult.cache_path(d, CompareMode.SIZE)
    loaded = CompareResult.load(d, sorted([os.path.join(d, "a.png"), os.path.join(d, "b.png")]),
                                mode=CompareMode.SIZE, threshold=0)
    assert os.path.isfile(path)
    assert loaded.is_complete


def test_search_works_in_a_directory_without_a_size_cache(tmp_path):
    """save_data() frees compare_data.file_data_dict after writing new data;
    search must not depend on it. Its first run in a directory writes the
    cache, so this is the case that failed."""
    d = str(tmp_path)
    query = _png(d, "a.png", (48, 48))
    same = _png(d, "b.png", (48, 48))
    other = _png(d, "c.png", (64, 64))
    app_actions = build_headless_app_actions({"get_base_dir": lambda: d})
    manager = CompareManager(
        master=None, app_actions=app_actions,
        get_base_dir=lambda: d, responsiveness=NullResponsiveness(),
    )
    manager.set_compare_mode(CompareMode.SIZE)
    args = CompareArgs(base_dir=d, mode=Mode.SEARCH, compare_mode=CompareMode.SIZE,
                       recursive=False, app_actions=app_actions, search_media_path=query)
    manager.apply_settings_to_args(args)

    manager.run(args)

    matched = manager._primary_wrapper().files_matched
    assert same in matched
    assert other not in matched

"""A stored, complete compare checkpoint survives deleting a grouped file.

End to end with no Qt: a real COLOR_MATCHING run that stores its checkpoint,
then the steps FileOpsController takes when the user deletes a file in group
mode (remove the file, CompareWrapper.remove_from_groups via the compare,
_update_groups_for_removed_file, _sync_result_after_deletion), then a reload.

files_grouped's keys index the stored file list. The deleted file is the
highest-indexed grouped one, so keys left unshifted after the list shrinks
would include one equal to the new list length, and the reload would discard
the checkpoint.
"""

import os

from compare.compare_args import CompareArgs
from compare.compare_manager import CompareManager
from compare.compare_result import CompareResult
from utils.config import config
from utils.constants import CompareMode, Mode
from utils.headless_app_actions import build_headless_app_actions
from utils.ui_responsiveness import NullResponsiveness


def _run_with_checkpoint(base_dir):
    app_actions = build_headless_app_actions({"get_base_dir": lambda: base_dir})
    manager = CompareManager(
        master=None, app_actions=app_actions,
        get_base_dir=lambda: base_dir, responsiveness=NullResponsiveness(),
    )
    manager.set_compare_mode(CompareMode.COLOR_MATCHING)
    args = CompareArgs(
        base_dir=base_dir, mode=Mode.GROUP, compare_mode=CompareMode.COLOR_MATCHING,
        recursive=False, app_actions=app_actions,
    )
    manager.set_store_checkpoints(True)
    manager.apply_settings_to_args(args)
    manager.run(args)
    return manager


def _delete_in_group_mode(wrapper, path):
    """What FileOpsController does for a delete while groups are shown."""
    display_index = next(
        i for i, g in enumerate(wrapper.group_indexes) if path in wrapper.file_groups[g]
    )
    wrapper.current_group_index = display_index
    wrapper.set_current_group()
    match_index = wrapper.files_matched.index(path)

    os.remove(path)
    wrapper._compare.remove_from_groups([path])
    wrapper._update_groups_for_removed_file(Mode.GROUP, display_index, match_index, set_group=True)
    wrapper._sync_result_after_deletion(path)


def test_checkpoint_reloads_complete_and_consistent_after_a_delete(compare_colors_dir):
    base_dir = compare_colors_dir["dir"]
    manager = _run_with_checkpoint(base_dir)
    wrapper = manager._primary_wrapper()
    compare = wrapper._compare
    cache_path = CompareResult.cache_path(base_dir, CompareMode.COLOR_MATCHING)
    assert os.path.isfile(cache_path), "the run did not store a checkpoint"

    files = list(compare.compare_data.files_found)
    grouped_indexes = sorted(compare.compare_result.files_grouped)
    # The highest-indexed grouped file, in a group that survives losing it.
    victim = next(
        files[i] for i in reversed(grouped_indexes)
        if sum(files[i] in g for g in wrapper.file_groups.values()) == 1
        and len(next(g for g in wrapper.file_groups.values() if files[i] in g)) >= 3
    )
    assert files.index(victim) == grouped_indexes[-1], (
        "fixture changed: the highest-indexed grouped file is no longer in a large group"
    )

    _delete_in_group_mode(wrapper, victim)

    remaining = [f for f in files if f != victim]
    reloaded = CompareResult.load(base_dir, remaining, mode=CompareMode.COLOR_MATCHING)
    assert reloaded.is_complete, "the checkpoint was discarded on reload"
    assert reloaded.files_grouped, "the reloaded checkpoint lost its groups"
    for file_index, (group_index, _score) in reloaded.files_grouped.items():
        path = remaining[file_index]
        assert path in reloaded.file_groups.get(group_index, {}), (
            f"files_grouped[{file_index}] names {os.path.basename(path)}, "
            f"which is not in group {group_index}"
        )
    assert all(victim not in g for g in reloaded.file_groups.values())


def test_store_checkpoints_follows_the_manager_setting(compare_colors_dir, monkeypatch):
    """The setting reaches the run through apply_settings_to_args."""
    monkeypatch.setattr(config, "store_checkpoints", True)
    base_dir = compare_colors_dir["dir"]
    app_actions = build_headless_app_actions({"get_base_dir": lambda: base_dir})
    manager = CompareManager(
        master=None, app_actions=app_actions,
        get_base_dir=lambda: base_dir, responsiveness=NullResponsiveness(),
    )
    manager.set_compare_mode(CompareMode.COLOR_MATCHING)
    manager.set_store_checkpoints(False)
    args = CompareArgs(base_dir=base_dir, mode=Mode.GROUP, compare_mode=CompareMode.COLOR_MATCHING,
                       recursive=False, app_actions=app_actions)
    manager.apply_settings_to_args(args)
    manager.run(args)
    assert not os.path.exists(CompareResult.cache_path(base_dir, CompareMode.COLOR_MATCHING))

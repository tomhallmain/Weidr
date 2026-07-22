"""
UI tests for GROUP_COMPLEMENT mode.

TestGroupComplementModePipeline runs a real COLOR_MATCHING group compare
(same fixture/pattern as test_compare_pipeline.py's TestCompareGroupPipeline --
no ML models required, deterministic solid-color PNGs) and exercises the
feature end to end through the real sidebar buttons: entering complement
mode, masonry integration, deletion, and returning to group mode.

TestFileOpsControllerGroupComplementDelete isolates a single regression risk
found while designing this feature: FileOpsController.delete_media's generic
(non-BROWSE) delete path resolves the file via CompareManager.current_group_index
and calls _update_groups_for_removed_file, which indexes into group_indexes /
file_groups assuming the current file belongs to the current group. In
GROUP_COMPLEMENT mode, current_group_index still refers to whatever real
group was last being browsed (deliberately left untouched so
return_to_group_mode() can restore it) -- so that generic path must never
run for this mode, or it would silently corrupt an unrelated group's data.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from tests.ui.app_window_fixtures import _teardown_app_window
from ui.app_window.app_window import AppWindow
from utils.config import config
from utils.constants import CompareMode, Mode


# ---------------------------------------------------------------------------
# Shared fixture / helpers (mirrors tests/ui/test_compare_pipeline.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def window_with_compare_dir(qtbot, compare_colors_dir):
    """AppWindow loaded with the solid-color compare directory."""
    win = AppWindow()
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    win.set_base_dir(compare_colors_dir["dir"])
    qtbot.waitUntil(lambda: win.base_dir == compare_colors_dir["dir"], timeout=3000)
    yield win, compare_colors_dir
    _teardown_app_window(win)


def _run_group_compare(win, qtbot, immediate_compare_debounce):
    """Clear sidebar search fields and trigger a group compare; wait for completion."""
    immediate_compare_debounce(win.search_ctrl)
    win.sidebar_panel.search_media_path_box.clear()
    win.sidebar_panel.search_text_box.clear()
    win.search_ctrl.set_search()
    qtbot.waitUntil(lambda: not win.search_ctrl.is_compare_running(), timeout=20000)


def _click_dynamic_button(sidebar_panel, name):
    """Trigger a dynamically-added sidebar button's clicked signal directly.

    Uses QPushButton.click() rather than qtbot.mouseClick(), which depends on
    real widget geometry and is fragile for offscreen/headless test runs.
    """
    sidebar_panel._dynamic_buttons[name].click()


def _grouped_paths(compare_manager) -> set:
    return {path for group in compare_manager.file_groups.values() for path in group}


# ---------------------------------------------------------------------------
# End-to-end pipeline: real group compare -> complement -> delete -> return
# ---------------------------------------------------------------------------

class TestGroupComplementModePipeline:
    def test_enter_complement_mode_shows_only_ungrouped_files(
        self, window_with_compare_dir, qtbot, bypass_password, immediate_compare_debounce
    ):
        """The three color-distinct outlier images never join any of the
        red/blue/green groups (see compare_image_fixtures's _OUTLIER_COLORS),
        so they are exactly what enter_complement_mode() should surface."""
        win, colors = window_with_compare_dir
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        grouped_paths = _grouped_paths(win.compare_manager)
        assert set(colors["outliers"]) & grouped_paths == set(), (
            "test fixture assumption broken: an outlier ended up in a group"
        )

        assert "view_ungrouped_btn" in win.sidebar_panel._dynamic_buttons
        _click_dynamic_button(win.sidebar_panel, "view_ungrouped_btn")
        qtbot.waitUntil(lambda: win.mode == Mode.GROUP_COMPLEMENT, timeout=2000)

        assert set(win.compare_manager.files_matched) == set(colors["outliers"])
        # The complement and the grouped set are always disjoint by construction.
        assert set(win.compare_manager.files_matched) & grouped_paths == set()

    def test_complement_mode_swaps_sidebar_buttons(
        self, window_with_compare_dir, qtbot, bypass_password, immediate_compare_debounce
    ):
        win, colors = window_with_compare_dir
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        _click_dynamic_button(win.sidebar_panel, "view_ungrouped_btn")
        qtbot.waitUntil(lambda: win.mode == Mode.GROUP_COMPLEMENT, timeout=2000)

        dynamic = win.sidebar_panel._dynamic_buttons
        assert "return_to_groups_btn" in dynamic
        for stale in ("prev_group_btn", "next_group_btn", "random_purge_btn", "view_ungrouped_btn"):
            assert stale not in dynamic, f"{stale} should have been removed on mode switch"

    def test_masonry_files_are_the_complement_list(
        self, window_with_compare_dir, qtbot, bypass_password, immediate_compare_debounce
    ):
        win, colors = window_with_compare_dir
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        _click_dynamic_button(win.sidebar_panel, "view_ungrouped_btn")
        qtbot.waitUntil(lambda: win.mode == Mode.GROUP_COMPLEMENT, timeout=2000)

        assert set(win._get_masonry_files()) == set(colors["outliers"])

    def test_return_to_group_mode_restores_previous_group(
        self, window_with_compare_dir, qtbot, bypass_password, immediate_compare_debounce
    ):
        win, colors = window_with_compare_dir
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        group_before = list(win.compare_manager.files_matched)
        group_index_before = win.compare_manager.current_group_index

        _click_dynamic_button(win.sidebar_panel, "view_ungrouped_btn")
        qtbot.waitUntil(lambda: win.mode == Mode.GROUP_COMPLEMENT, timeout=2000)

        _click_dynamic_button(win.sidebar_panel, "return_to_groups_btn")
        qtbot.waitUntil(lambda: win.mode == Mode.GROUP, timeout=2000)

        assert win.compare_manager.current_group_index == group_index_before
        assert list(win.compare_manager.files_matched) == group_before
        assert "prev_group_btn" in win.sidebar_panel._dynamic_buttons
        assert "return_to_groups_btn" not in win.sidebar_panel._dynamic_buttons

    def test_delete_in_complement_mode_removes_file_and_leaves_groups_untouched(
        self, window_with_compare_dir, qtbot, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, colors = window_with_compare_dir
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        file_groups_before = {k: dict(v) for k, v in win.compare_manager.file_groups.items()}

        win.compare_manager.enter_complement_mode()
        qtbot.waitUntil(lambda: win.mode == Mode.GROUP_COMPLEMENT, timeout=2000)
        assert len(win.compare_manager.files_matched) == 3

        monkeypatch.setattr(config, "delete_instantly", True)
        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append(path),
        )
        target = win.compare_manager.files_matched[0]

        win.file_ops_ctrl.delete_media()

        assert removed == [target]
        assert target not in win.compare_manager.files_matched
        assert len(win.compare_manager.files_matched) == 2
        assert win.mode == Mode.GROUP_COMPLEMENT
        # The real groups from before entering complement mode must be untouched.
        assert win.compare_manager.file_groups == file_groups_before

    def test_deleting_every_complement_file_returns_to_group_mode(
        self, window_with_compare_dir, qtbot, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, colors = window_with_compare_dir
        win.compare_manager.set_compare_mode(CompareMode.COLOR_MATCHING)
        win.compare_manager.set_threshold(15)
        _run_group_compare(win, qtbot, immediate_compare_debounce)

        win.compare_manager.enter_complement_mode()
        qtbot.waitUntil(lambda: win.mode == Mode.GROUP_COMPLEMENT, timeout=2000)

        monkeypatch.setattr(config, "delete_instantly", True)
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: None,
        )

        remaining = len(win.compare_manager.files_matched)
        for _i in range(remaining):
            win.file_ops_ctrl.delete_media()

        qtbot.waitUntil(lambda: win.mode == Mode.GROUP, timeout=2000)


# ---------------------------------------------------------------------------
# FileOpsController.delete_media -- isolated regression test for the
# group-index-corruption risk, without needing a real compare run.
# ---------------------------------------------------------------------------

class TestFileOpsControllerGroupComplementDelete:
    def test_delete_media_bypasses_update_groups_for_removed_file(
        self, window_with_dir, bypass_password, monkeypatch
    ):
        win, media_dir = window_with_dir
        monkeypatch.setattr(config, "delete_instantly", True)
        target = os.path.join(media_dir, "img02.png")
        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append(path),
        )
        monkeypatch.setattr(win, "release_media_canvas", lambda: None)
        monkeypatch.setattr(win.media_navigator, "get_active_media_filepath", lambda: target)
        monkeypatch.setattr(win.compare_manager, "has_compare", lambda: True)

        group_removed = []
        monkeypatch.setattr(
            win.compare_manager,
            "compare",
            lambda: SimpleNamespace(remove_from_groups=lambda files: group_removed.extend(files)),
        )
        complement_removed = []
        monkeypatch.setattr(
            win.compare_manager,
            "remove_from_complement",
            lambda f: complement_removed.append(f),
        )
        update_groups_calls = []
        monkeypatch.setattr(
            win.compare_manager,
            "_update_groups_for_removed_file",
            lambda *a, **k: update_groups_calls.append((a, k)),
        )

        win.set_mode(Mode.GROUP_COMPLEMENT)
        win.file_ops_ctrl.delete_media()

        assert removed == [target]
        assert group_removed == [target]
        assert complement_removed == [target]
        # The key regression this guards against: the generic delete path
        # (which assumes the file belongs to current_group_index) must never
        # run for GROUP_COMPLEMENT.
        assert update_groups_calls == []

    def test_delete_media_noop_when_no_active_file(
        self, window_with_dir, bypass_password, monkeypatch
    ):
        win, _media_dir = window_with_dir
        monkeypatch.setattr(config, "delete_instantly", True)
        removed = []
        monkeypatch.setattr(
            "ui.app_window.file_ops_controller.Utils.remove_path",
            lambda path, **kwargs: removed.append(path),
        )
        monkeypatch.setattr(win.media_navigator, "get_active_media_filepath", lambda: None)
        complement_removed = []
        monkeypatch.setattr(
            win.compare_manager,
            "remove_from_complement",
            lambda f: complement_removed.append(f),
        )

        win.set_mode(Mode.GROUP_COMPLEMENT)
        win.file_ops_ctrl.delete_media()

        assert removed == []
        assert complement_removed == []


# ---------------------------------------------------------------------------
# FileOpsController.handle_remove_files_from_groups -- the move-flow analog
# of delete_media's GROUP_COMPLEMENT branch. Reached via
# AppWindow.refresh(removed_files=...) after a move-out completes (Ctrl+Z-able,
# unlike delete). Before this fix it always went through
# _get_file_group_map/_update_groups_for_removed_file, which assume the
# removed file belongs to the group at current_group_index -- corrupting
# either the live complement list or the real (untouched) group data.
# ---------------------------------------------------------------------------

class TestFileOpsControllerGroupComplementMoveRemoval:
    def test_bypasses_group_index_machinery_for_group_complement(
        self, window_with_dir, monkeypatch
    ):
        win, media_dir = window_with_dir
        monkeypatch.setattr(
            win.compare_manager, "capture_removal_undo_snapshot", lambda files, mode: None
        )
        monkeypatch.setattr(win.compare_manager, "current_match", lambda: None)
        removed_calls = []
        monkeypatch.setattr(
            win.compare_manager, "remove_from_complement", lambda f: removed_calls.append(f)
        )
        group_map_calls = []
        monkeypatch.setattr(
            win.compare_manager,
            "_get_file_group_map",
            lambda mode: group_map_calls.append(mode) or {},
        )
        update_groups_calls = []
        monkeypatch.setattr(
            win.compare_manager,
            "_update_groups_for_removed_file",
            lambda *a, **k: update_groups_calls.append((a, k)),
        )

        win.set_mode(Mode.GROUP_COMPLEMENT)
        target1 = os.path.join(media_dir, "img01.png")
        target2 = os.path.join(media_dir, "img02.png")
        win.file_ops_ctrl.handle_remove_files_from_groups([target1, target2])

        assert removed_calls == [target1, target2]
        # The key regression this guards against: the group-index-based path
        # (which assumes each removed file belongs to current_group_index)
        # must never run for GROUP_COMPLEMENT.
        assert group_map_calls == []
        assert update_groups_calls == []

    def test_group_mode_still_uses_group_index_machinery(
        self, window_with_dir, monkeypatch
    ):
        """Sanity check: the new branch is GROUP_COMPLEMENT-specific -- GROUP
        mode's existing removal path must be unaffected."""
        win, media_dir = window_with_dir
        monkeypatch.setattr(
            win.compare_manager, "capture_removal_undo_snapshot", lambda files, mode: None
        )
        monkeypatch.setattr(win.compare_manager, "current_match", lambda: None)
        removed_calls = []
        monkeypatch.setattr(
            win.compare_manager, "remove_from_complement", lambda f: removed_calls.append(f)
        )
        target = os.path.join(media_dir, "img01.png")
        group_map_calls = []
        monkeypatch.setattr(
            win.compare_manager,
            "_get_file_group_map",
            lambda mode: group_map_calls.append(mode) or {},
        )

        win.set_mode(Mode.GROUP)
        win.file_ops_ctrl.handle_remove_files_from_groups([target])

        assert group_map_calls == [Mode.GROUP]
        assert removed_calls == []


# ---------------------------------------------------------------------------
# AppWindow.restore_compare_state_for_undone_move -- GROUP_COMPLEMENT branch.
# Before this fix, an undone move restored files_matched to the snapshot's
# complement list via maybe_restore_removal_undo_snapshot(), only to have it
# immediately clobbered again by the unconditional set_current_group() call,
# which rebuilds files_matched from file_groups (the real, untouched groups).
# ---------------------------------------------------------------------------

class TestRestoreCompareStateForUndoneMoveGroupComplement:
    def test_group_complement_snapshot_is_not_clobbered_by_set_current_group(
        self, window_with_dir, monkeypatch
    ):
        win, _media_dir = window_with_dir
        snapshot = SimpleNamespace(app_mode=Mode.GROUP_COMPLEMENT, removed_files=["/a.jpg"], match_index=0)
        monkeypatch.setattr(
            win.compare_manager, "maybe_restore_removal_undo_snapshot", lambda: snapshot
        )
        monkeypatch.setattr(win.compare_manager, "current_match", lambda: "/restored.jpg")
        created = []
        monkeypatch.setattr(win.media_navigator, "create_media", lambda f: created.append(f))
        set_current_group_calls = []
        monkeypatch.setattr(
            win.compare_manager,
            "set_current_group",
            lambda **k: set_current_group_calls.append(k),
        )

        win.set_mode(Mode.GROUP_COMPLEMENT)
        win.restore_compare_state_for_undone_move()

        assert created == ["/restored.jpg"]
        # The regression this guards against: set_current_group() would
        # rebuild files_matched from the real (untouched) file_groups, wiping
        # out the complement list that maybe_restore_removal_undo_snapshot()
        # just restored.
        assert set_current_group_calls == []

    def test_group_mode_snapshot_still_uses_set_current_group(
        self, window_with_dir, monkeypatch
    ):
        """Sanity check: the new branch is GROUP_COMPLEMENT-specific -- GROUP
        mode's existing restore path must be unaffected."""
        win, _media_dir = window_with_dir
        snapshot = SimpleNamespace(app_mode=Mode.GROUP, removed_files=["/a.jpg"], match_index=2)
        monkeypatch.setattr(
            win.compare_manager, "maybe_restore_removal_undo_snapshot", lambda: snapshot
        )
        set_current_group_calls = []
        monkeypatch.setattr(
            win.compare_manager,
            "set_current_group",
            lambda **k: set_current_group_calls.append(k),
        )

        win.set_mode(Mode.GROUP)
        win.restore_compare_state_for_undone_move()

        assert set_current_group_calls == [{"start_match_index": 2}]

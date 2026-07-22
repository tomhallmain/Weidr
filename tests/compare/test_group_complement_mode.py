"""
Unit tests for GROUP_COMPLEMENT mode's CompareWrapper methods.

enter_complement_mode() / return_to_group_mode() / remove_from_complement()
deliberately never touch file_groups / group_indexes / current_group_index --
only files_matched / match_index are reassigned, the same flat-list trick
run_search() already uses for SEARCH mode. These tests exercise that
contract directly, without a real compare run or a real AppWindow.
"""
from __future__ import annotations

from types import SimpleNamespace

from compare.compare_wrapper import CompareWrapper
from utils.constants import CompareMode, Mode


class _RecordingAppActions:
    """Stub app actions that records every call for assertions."""

    def __init__(self):
        self.alerts = []
        self.modes = []
        self.created_media = []
        self.add_buttons_calls = 0
        self.refresh_masonry_calls = 0
        self.toasts = []

    def alert(self, title, message, *a, **k):
        self.alerts.append((title, message))

    def set_mode(self, mode, do_update=True):
        self.modes.append((mode, do_update))

    def _add_buttons_for_mode(self):
        self.add_buttons_calls += 1

    def create_media(self, filepath):
        self.created_media.append(filepath)

    def refresh_masonry(self):
        self.refresh_masonry_calls += 1

    def toast(self, message, *a, **k):
        self.toasts.append(message)

    # Methods hit only if a test happens to exercise _display_current_match /
    # _load_current_group via return_to_group_mode()'s set_current_group() call.
    def _set_label_state(self, *a, **k):
        pass


def _wrapper_with_groups(
    files_found, file_groups, group_indexes=None, current_group_index=0,
    file_browser_files=None, app_mode=Mode.GROUP,
):
    app_actions = _RecordingAppActions()
    # _display_current_match() (hit via return_to_group_mode -> set_current_group)
    # calls self._master.update() unconditionally, so master=None would raise.
    # enter_complement_mode() also reads self._master.file_browser.get_files()
    # to order the complement by the user's active sort; default it to the
    # scan order itself so tests that don't care about sorting are unaffected.
    # _master.mode is read by _in_group_complement_mode() -- the guard that
    # keeps group/supergroup navigation from corrupting the live complement
    # list -- so tests exercising that guard pass app_mode=Mode.GROUP_COMPLEMENT.
    master = SimpleNamespace(
        update=lambda: None,
        mode=app_mode,
        file_browser=SimpleNamespace(
            get_files=lambda: list(file_browser_files if file_browser_files is not None else files_found)
        ),
    )
    wrapper = CompareWrapper(master=master, compare_mode=CompareMode.CLIP_EMBEDDING, app_actions=app_actions)
    wrapper.file_groups = file_groups
    wrapper.group_indexes = group_indexes if group_indexes is not None else sorted(file_groups.keys())
    wrapper.current_group_index = current_group_index
    wrapper._compare = SimpleNamespace(
        compare_data=SimpleNamespace(files_found=list(files_found)),
        # Hit by _display_current_match -> _supergroup_label_suffix whenever
        # return_to_group_mode()'s set_current_group() call displays a match.
        compare_result=SimpleNamespace(has_meaningful_supergroups=lambda: False),
    )
    return wrapper, app_actions


class TestEnterComplementMode:
    def test_complement_excludes_grouped_files_preserves_scan_order(self):
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg", "/e.jpg"]
        file_groups = {0: {"/a.jpg": 0.1, "/b.jpg": 0.2}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups)

        wrapper.enter_complement_mode()

        assert wrapper.files_matched == ["/c.jpg", "/d.jpg", "/e.jpg"]
        assert wrapper.match_index == 0

    def test_complement_ordered_by_file_browser_sort_not_scan_order(self):
        """The complement has no similarity-derived order of its own, so it
        should follow the file browser's active sort (e.g. name/date/size),
        not the compare engine's raw scan order."""
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}}
        # File browser sorted descending by name, say -- scan order is ascending.
        file_browser_files = ["/d.jpg", "/c.jpg", "/b.jpg", "/a.jpg"]
        wrapper, app_actions = _wrapper_with_groups(
            files_found, file_groups, file_browser_files=file_browser_files
        )

        wrapper.enter_complement_mode()

        assert wrapper.files_matched == ["/d.jpg", "/c.jpg", "/b.jpg"]

    def test_complement_keeps_files_missing_from_file_browser_list(self):
        """A file the file browser's own filters happen to exclude (e.g. a
        type/hidden filter mismatch with the compare scan) must still show up
        in the complement, appended in scan order, rather than silently
        disappearing."""
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}}
        # file_browser's list is missing "/c.jpg" entirely.
        file_browser_files = ["/b.jpg"]
        wrapper, app_actions = _wrapper_with_groups(
            files_found, file_groups, file_browser_files=file_browser_files
        )

        wrapper.enter_complement_mode()

        assert wrapper.files_matched == ["/b.jpg", "/c.jpg"]

    def test_switches_mode_and_displays_first_file(self):
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups)

        wrapper.enter_complement_mode()

        assert app_actions.modes == [(Mode.GROUP_COMPLEMENT, False)]
        assert app_actions.add_buttons_calls == 1
        assert app_actions.created_media == ["/b.jpg"]
        assert app_actions.refresh_masonry_calls == 1
        assert len(app_actions.toasts) == 1

    def test_does_not_touch_group_state(self):
        """file_groups / group_indexes / current_group_index survive unchanged --
        return_to_group_mode() depends on this to restore the exact group."""
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}, 1: {"/b.jpg": 0.2}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups, current_group_index=1)

        wrapper.enter_complement_mode()

        assert wrapper.file_groups == file_groups
        assert wrapper.group_indexes == [0, 1]
        assert wrapper.current_group_index == 1

    def test_empty_complement_alerts_and_does_not_switch_mode(self):
        files_found = ["/a.jpg", "/b.jpg"]
        file_groups = {0: {"/a.jpg": 0.1, "/b.jpg": 0.2}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups)
        original_files_matched = list(wrapper.files_matched)

        wrapper.enter_complement_mode()

        assert len(app_actions.alerts) == 1
        assert app_actions.modes == []
        assert wrapper.files_matched == original_files_matched


class TestReturnToGroupMode:
    def test_restores_the_group_being_browsed(self):
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}, 1: {"/b.jpg": 0.2, "/c.jpg": 0.1}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups, current_group_index=1)
        wrapper.set_current_group()
        group_files_matched = list(wrapper.files_matched)
        assert group_files_matched  # sanity: group 1 has members

        wrapper.enter_complement_mode()
        assert wrapper.files_matched == ["/d.jpg"]

        wrapper.return_to_group_mode()

        assert app_actions.modes[-1] == (Mode.GROUP, False)
        assert wrapper.files_matched == group_files_matched
        assert wrapper.current_group_index == 1


class TestRemoveFromComplement:
    def test_removes_file_and_advances_to_next(self):
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups)
        wrapper.enter_complement_mode()
        assert wrapper.files_matched == ["/b.jpg", "/c.jpg"]

        wrapper.remove_from_complement("/b.jpg")

        assert wrapper.files_matched == ["/c.jpg"]
        assert wrapper.match_index == 0
        assert app_actions.created_media[-1] == "/c.jpg"
        assert app_actions.refresh_masonry_calls >= 2  # once on entry, once on removal

    def test_missing_filepath_is_a_noop_removal(self):
        files_found = ["/a.jpg", "/b.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups)
        wrapper.enter_complement_mode()

        wrapper.remove_from_complement("/not-present.jpg")

        assert wrapper.files_matched == ["/b.jpg"]

    def test_removing_last_file_returns_to_group_mode(self):
        files_found = ["/a.jpg", "/b.jpg"]
        file_groups = {0: {"/a.jpg": 0.1}}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups, current_group_index=0)
        wrapper.set_current_group()
        group_files_matched = list(wrapper.files_matched)
        wrapper.enter_complement_mode()
        assert wrapper.files_matched == ["/b.jpg"]

        wrapper.remove_from_complement("/b.jpg")

        # files_matched is empty only transiently, inside remove_from_complement,
        # right before it falls back to return_to_group_mode() -- which
        # immediately rebuilds files_matched from the restored group. So by the
        # time this method returns, files_matched already holds the group again,
        # not [].
        assert len(app_actions.alerts) == 1
        # return_to_group_mode() was invoked as part of exhausting the complement
        assert app_actions.modes[-1] == (Mode.GROUP, False)
        # set_current_group() (called by return_to_group_mode) rebuilt the group
        assert wrapper.files_matched == group_files_matched

    def test_match_index_clamped_when_last_entry_removed(self):
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg"]
        file_groups = {}
        wrapper, app_actions = _wrapper_with_groups(files_found, file_groups)
        wrapper.enter_complement_mode()
        wrapper.match_index = 2  # pointing at /c.jpg, the last entry

        wrapper.remove_from_complement("/c.jpg")

        assert wrapper.files_matched == ["/a.jpg", "/b.jpg"]
        assert wrapper.match_index == 0


class TestCompareManagerDelegation:
    """CompareManager.enter_complement_mode/return_to_group_mode/remove_from_complement
    must delegate to the primary wrapper, like every other wrapper method."""

    def test_delegates_to_primary_wrapper(self):
        from compare.compare_manager import CompareManager

        calls = []
        fake_wrapper = SimpleNamespace(
            enter_complement_mode=lambda: calls.append("enter"),
            return_to_group_mode=lambda: calls.append("return"),
            remove_from_complement=lambda f: calls.append(("remove", f)),
        )
        cm = CompareManager.__new__(CompareManager)
        cm._primary_wrapper = lambda: fake_wrapper

        cm.enter_complement_mode()
        cm.return_to_group_mode()
        cm.remove_from_complement("/a.jpg")

        assert calls == ["enter", "return", ("remove", "/a.jpg")]

    def test_delegation_is_a_noop_without_a_primary_wrapper(self):
        from compare.compare_manager import CompareManager

        cm = CompareManager.__new__(CompareManager)
        cm._primary_wrapper = lambda: None

        # None of these should raise even though there is no wrapper yet.
        cm.enter_complement_mode()
        cm.return_to_group_mode()
        cm.remove_from_complement("/a.jpg")


class TestGroupNavigationGuardedInComplementMode:
    """Shift+Left/Right, Ctrl+Shift+Left/Right, and Home/End are global
    keybindings that stay active regardless of app mode. Before adding
    _in_group_complement_mode(), calling any of them while GROUP_COMPLEMENT
    was showing would silently rebuild files_matched from file_groups (the
    real, untouched groups) via set_current_group()/_load_current_group(),
    swapping the complement list out for a real group's members while the
    app still displayed "Group Complement Mode" and current_group_index
    still pointed at whatever group was last being browsed."""

    def _complement_wrapper(self):
        files_found = ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg", "/e.jpg"]
        file_groups = {0: {"/a.jpg": 0.1, "/b.jpg": 0.2}, 1: {"/x.jpg": 0.1, "/y.jpg": 0.2}}
        wrapper, app_actions = _wrapper_with_groups(
            files_found, file_groups, current_group_index=0, app_mode=Mode.GROUP_COMPLEMENT
        )
        # show_boundary_match()'s skip loop would otherwise call the real
        # ClassifierActionsManager.prevalidate_media() (config.enable_prevalidations
        # defaults True in the test config) -- irrelevant to what these tests
        # check, and _RecordingAppActions deliberately doesn't stub that heavy,
        # unrelated path. Disabling it here isolates the test the same way
        # CompareWrapper.__init__ lets any caller do per-instance.
        wrapper.prevalidations_running = False
        # Simulate having already entered complement mode.
        wrapper.files_matched = ["/c.jpg", "/d.jpg", "/e.jpg"]
        wrapper.match_index = 0
        return wrapper, app_actions

    def test_show_prev_group_is_a_noop(self):
        wrapper, app_actions = self._complement_wrapper()
        complement_before = list(wrapper.files_matched)

        wrapper.show_prev_group()

        assert wrapper.files_matched == complement_before
        assert wrapper.current_group_index == 0
        assert app_actions.created_media == []

    def test_show_next_group_is_a_noop(self):
        wrapper, app_actions = self._complement_wrapper()
        complement_before = list(wrapper.files_matched)

        wrapper.show_next_group()

        assert wrapper.files_matched == complement_before
        assert wrapper.current_group_index == 0

    def test_show_prev_supergroup_is_a_noop(self):
        wrapper, app_actions = self._complement_wrapper()
        complement_before = list(wrapper.files_matched)

        wrapper.show_prev_supergroup()

        assert wrapper.files_matched == complement_before
        assert wrapper.current_supergroup_index == 0

    def test_show_next_supergroup_is_a_noop(self):
        wrapper, app_actions = self._complement_wrapper()
        complement_before = list(wrapper.files_matched)

        wrapper.show_next_supergroup()

        assert wrapper.files_matched == complement_before

    def test_show_boundary_match_jumps_within_complement_not_real_groups(self):
        wrapper, app_actions = self._complement_wrapper()

        wrapper.show_boundary_match(last_file=True)

        assert wrapper.files_matched == ["/c.jpg", "/d.jpg", "/e.jpg"]  # untouched
        assert wrapper.match_index == 2
        assert app_actions.created_media[-1] == "/e.jpg"
        assert wrapper.current_group_index == 0  # untouched

        wrapper.show_boundary_match(last_file=False)

        assert wrapper.match_index == 0
        assert app_actions.created_media[-1] == "/c.jpg"

    def test_show_boundary_match_noop_when_complement_empty(self):
        wrapper, app_actions = self._complement_wrapper()
        wrapper.files_matched = []

        wrapper.show_boundary_match(last_file=True)  # must not raise

        assert wrapper.files_matched == []
        assert app_actions.created_media == []

"""SearchController: current-media search, directory image gen, compare refresh."""

import os

from compare.compare_args import CompareArgs
from ui.app_window.window_manager import WindowManager
from utils.constants import ImageGenerationType, Mode


def _search_mocks(win, monkeypatch, run_calls):
    """Common mocks for code paths that end in set_search / compare run."""
    monkeypatch.setattr(win.search_ctrl._cm, "validate_compare_mode", lambda *a, **k: None)
    monkeypatch.setattr(win.media_navigator, "show_searched_media", lambda: None)
    monkeypatch.setattr(win.media_frame, "setFocus", lambda: None)
    monkeypatch.setattr(
        win.search_ctrl._cm,
        "run",
        lambda args: run_calls.append(args),
    )


class TestSetCurrentMediaRunSearch:
    def test_set_current_media_run_search_sets_box_and_runs_compare(
        self,
        window_with_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        _search_mocks(win, monkeypatch, run_calls)

        filepath = win.file_browser.get_files()[1]
        win.media_path = filepath
        monkeypatch.setattr(
            win.media_navigator,
            "get_active_media_filepath",
            lambda: filepath,
        )

        win.search_ctrl.set_current_media_run_search(base_dir=media_dir)

        assert win.sidebar_panel.search_media_path_box.text() == "img02.png"
        assert win.mode == Mode.SEARCH
        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].search_media_path is not None
        assert os.path.normcase(run_calls[0].search_media_path) == os.path.normcase(
            filepath
        )

    def test_set_media_run_search_uses_relative_path_in_sidebar(
        self, window_with_dir, monkeypatch
    ):
        win, media_dir = window_with_dir
        searched = []
        monkeypatch.setattr(win.search_ctrl, "set_search", lambda: searched.append(True))

        filepath = os.path.join(media_dir, "img01.png")
        win.search_ctrl._set_media_run_search(filepath)

        assert win.sidebar_panel.search_media_path_box.text() == "img01.png"
        assert searched == [True]


class TestImageGenerationOnDirectory:
    def test_run_image_generation_on_directory_invokes_sd_runner(
        self, window_with_dir, qtbot, bypass_password, monkeypatch
    ):
        win, media_dir = window_with_dir
        media_path = win.file_browser.get_files()[0]
        dir_calls = []

        class FakeSDRunner:
            def run_on_directory(self, gen_type, directory_path):
                dir_calls.append((gen_type, directory_path))

        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient",
            FakeSDRunner,
        )
        monkeypatch.setattr(
            "ui.image.media_details.MediaDetails.get_image_specific_generation_mode",
            lambda: ImageGenerationType.LAST_SETTINGS,
        )
        monkeypatch.setattr(win.search_ctrl, "_get_media_path", lambda: media_path)

        win.search_ctrl.run_image_generation_on_directory()

        qtbot.waitUntil(lambda: len(dir_calls) == 1, timeout=5000)
        gen_type, directory_path = dir_calls[0]
        assert gen_type == ImageGenerationType.LAST_SETTINGS
        assert os.path.normcase(directory_path) == os.path.normcase(media_dir)
        qtbot.waitUntil(lambda: win.search_ctrl._img_gen_workers == [], timeout=5000)


class TestCompareRefreshPaths:
    def test_run_compare_find_duplicates_sets_flag_on_args(
        self,
        window_with_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        win, _ = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        _search_mocks(win, monkeypatch, run_calls)
        monkeypatch.setattr(win.search_ctrl, "_validate_run", lambda: True)

        win.search_ctrl.run_compare(find_duplicates=True)

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].find_duplicates is True

    def test_refresh_compare_schedules_compare_run(
        self,
        window_with_dir,
        qtbot,
        bypass_password,
        immediate_compare_debounce,
        monkeypatch,
    ):
        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        _search_mocks(win, monkeypatch, run_calls)
        monkeypatch.setattr(win.search_ctrl, "_validate_run", lambda: True)

        custom = CompareArgs()
        custom.search_text = "refresh-me"
        win.search_ctrl.refresh_compare(compare_args=custom)

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].search_text == "refresh-me"
        assert run_calls[0].base_dir == media_dir

    def test_refresh_all_compares_calls_refresh_compare_on_active_windows(
        self, window_with_dir, monkeypatch
    ):
        win, _ = window_with_dir
        calls = []
        monkeypatch.setattr(
            win.search_ctrl,
            "refresh_compare",
            lambda compare_args=CompareArgs(): calls.append(compare_args),
        )
        monkeypatch.setattr(win.compare_manager, "has_compare", lambda: True)

        WindowManager.refresh_all_compares()

        assert len(calls) == 1


class TestFileListCompareArgs:
    """A file-list search (e.g. FileActionsWindow's "Search in New Window",
    via SearchController.set_search(file_list=...)) has no real base_dir --
    its corpus spans wherever the files actually live. _run_compare() must
    not fall back to self._app.get_base_dir() in that case (it may be None,
    and the compare engine's own self.base_dir is used pervasively as a real
    path and would raise on None) -- it substitutes a designated cache
    directory instead, without ever touching AppWindow.base_dir itself."""

    def test_file_list_overrides_base_dir(
        self, window_with_dir, qtbot, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        from utils.utils import Utils

        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        _search_mocks(win, monkeypatch, run_calls)
        monkeypatch.setattr(win.search_ctrl, "_validate_run", lambda: True)

        file_list = [os.path.join(media_dir, "img01.png"), os.path.join(media_dir, "img02.png")]
        args = CompareArgs()
        args.file_list = list(file_list)

        win.search_ctrl.run_compare(compare_args=args)

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].file_list == file_list
        assert run_calls[0].base_dir == Utils.get_no_directory_compare_cache_dir()
        # AppWindow's own base_dir is untouched -- prevalidation profile
        # matching (which reads get_base_dir(), not args.base_dir) must keep
        # seeing whatever directory the window actually has (or none).
        assert win.get_base_dir() == media_dir

    def test_no_file_list_uses_normal_base_dir(
        self, window_with_dir, qtbot, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        """Sanity check: the substitution is file-list-specific -- an
        ordinary compare run must be unaffected."""
        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        _search_mocks(win, monkeypatch, run_calls)
        monkeypatch.setattr(win.search_ctrl, "_validate_run", lambda: True)

        win.search_ctrl.run_compare(compare_args=CompareArgs())

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].file_list == []
        assert run_calls[0].base_dir == media_dir


class TestSetSearchFileList:
    """SearchController.set_search(file_list=...) -- the direct entry point
    AppWindow.__init__'s do_search path (and add_secondary_window's
    reuse-existing-window branch) use to run a file-list search, rather than
    relying on any separate staged/pending state on AppWindow."""

    def test_file_list_flows_through_to_compare_args(
        self, window_with_dir, qtbot, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        _search_mocks(win, monkeypatch, run_calls)

        query = os.path.join(media_dir, "img01.png")
        win.sidebar_panel.search_media_path_box.setText(query)
        file_list = [
            os.path.join(media_dir, "img01.png"),
            os.path.join(media_dir, "img02.png"),
        ]

        win.search_ctrl.set_search(file_list=file_list)

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].file_list == file_list
        assert run_calls[0].search_media_path == query
        assert win.mode == Mode.SEARCH

    def test_no_file_list_defaults_to_empty(
        self, window_with_dir, qtbot, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        _search_mocks(win, monkeypatch, run_calls)
        win.sidebar_panel.search_media_path_box.setText(os.path.join(media_dir, "img01.png"))

        win.search_ctrl.set_search()

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].file_list == []

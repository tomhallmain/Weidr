"""SearchController media search setup (set_search_for_media, negative media)."""

import os

from utils.constants import Mode


class TestSearchControllerMedia:
    def test_set_search_for_media_uses_sidebar_basename(
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
        monkeypatch.setattr(win.search_ctrl._cm, "validate_compare_mode", lambda *a, **k: None)
        monkeypatch.setattr(win.search_ctrl._cm, "run", lambda args: run_calls.append(args))
        monkeypatch.setattr(win.media_navigator, "show_searched_media", lambda: None)
        monkeypatch.setattr(win.media_frame, "setFocus", lambda: None)

        win.sidebar_panel.search_media_path_box.setText("img02.png")
        win.search_ctrl.set_search_for_media()
        qtbot.waitUntil(lambda: len(run_calls) >= 1, timeout=5000)

        assert win.mode == Mode.SEARCH
        assert len(run_calls) == 1
        assert run_calls[0].search_media_path is not None
        assert os.path.normcase(run_calls[0].search_media_path) == os.path.normcase(
            os.path.join(media_dir, "img02.png")
        )

    def test_get_negative_search_media_path_resolves_basename(
        self, window_with_dir
    ):
        win, media_dir = window_with_dir
        win.sidebar_panel.search_media_negative_path_box.setText("img01.png")
        resolved = win.search_ctrl.get_negative_search_media_path()
        assert resolved is not None
        assert os.path.normcase(resolved) == os.path.normcase(
            os.path.join(media_dir, "img01.png")
        )

    def test_negative_media_search_sets_box_and_runs_search(
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
        monkeypatch.setattr(win.search_ctrl._cm, "validate_compare_mode", lambda *a, **k: None)
        monkeypatch.setattr(win.search_ctrl._cm, "run", lambda args: run_calls.append(args))
        monkeypatch.setattr(win.media_navigator, "show_searched_media", lambda: None)
        monkeypatch.setattr(win.media_frame, "setFocus", lambda: None)

        target = os.path.join(media_dir, "img03.png")
        win.search_ctrl.negative_media_search(target)
        qtbot.waitUntil(lambda: len(run_calls) >= 1, timeout=5000)

        assert win.sidebar_panel.search_media_negative_path_box.text() == "img03.png"
        assert win.mode == Mode.SEARCH
        assert len(run_calls) == 1
        assert os.path.normcase(run_calls[0].negative_search_media_path) == os.path.normcase(
            target
        )

    def test_set_search_for_media_falls_back_to_current_media_path(
        self, window_with_dir, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        monkeypatch.setattr(win.search_ctrl._cm, "validate_compare_mode", lambda *a, **k: None)
        monkeypatch.setattr(win.search_ctrl._cm, "run", lambda _args: None)
        monkeypatch.setattr(win.media_navigator, "show_searched_media", lambda: None)
        monkeypatch.setattr(win.media_frame, "setFocus", lambda: None)

        win.media_path = win.file_browser.get_files()[0]
        win.sidebar_panel.search_media_path_box.clear()
        win.search_ctrl.set_search_for_media()

        text = win.sidebar_panel.search_media_path_box.text()
        assert text
        assert os.path.basename(win.media_path) in text

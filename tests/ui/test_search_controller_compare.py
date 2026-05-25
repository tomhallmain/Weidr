"""SearchController compare scheduling and mode switches (mocked CompareManager.run)."""

from utils.constants import Mode


class TestSearchControllerCompare:
    def test_set_search_with_text_enters_search_mode(
        self, window_with_dir, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, _ = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        monkeypatch.setattr(win.search_ctrl._cm, "validate_compare_mode", lambda *a, **k: None)
        monkeypatch.setattr(win.media_navigator, "show_searched_media", lambda: None)
        monkeypatch.setattr(win.media_frame, "setFocus", lambda: None)
        monkeypatch.setattr(win.search_ctrl._cm, "run", lambda _args: None)

        win.sidebar_panel.search_text_box.setText("cat")
        win.search_ctrl.set_search()

        assert win.mode == Mode.SEARCH

    def test_set_search_empty_keeps_browse_mode(
        self, window_with_dir, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, _ = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        monkeypatch.setattr(win.search_ctrl._cm, "validate_compare_mode", lambda *a, **k: None)
        monkeypatch.setattr(win.media_frame, "setFocus", lambda: None)
        monkeypatch.setattr(win.search_ctrl._cm, "run", lambda _args: None)

        win.sidebar_panel.search_text_box.clear()
        win.sidebar_panel.search_text_negative_box.clear()
        win.sidebar_panel.search_media_path_box.clear()
        win.set_mode(Mode.BROWSE)
        win.search_ctrl.set_search()

        assert win.mode == Mode.BROWSE

    def test_debounced_compare_calls_compare_manager_run(
        self, window_with_dir, qtbot, bypass_password, immediate_compare_debounce, monkeypatch
    ):
        win, media_dir = window_with_dir
        immediate_compare_debounce(win.search_ctrl)
        run_calls = []
        monkeypatch.setattr(
            win.search_ctrl._cm,
            "run",
            lambda args: run_calls.append(args),
        )
        monkeypatch.setattr(win.search_ctrl._cm, "validate_compare_mode", lambda *a, **k: None)
        monkeypatch.setattr(win.media_navigator, "show_searched_media", lambda: None)
        monkeypatch.setattr(win.media_frame, "setFocus", lambda: None)

        win.sidebar_panel.search_text_box.setText("cat")
        win.search_ctrl.set_search()

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        assert run_calls[0].search_text == "cat"
        assert run_calls[0].base_dir == media_dir
        qtbot.waitUntil(lambda: not win.search_ctrl.is_compare_running(), timeout=5000)

    def test_is_compare_running_while_worker_active(
        self, window_with_dir, qtbot, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        started = []

        def slow_run(_args):
            started.append(True)

        monkeypatch.setattr(win.search_ctrl._cm, "run", slow_run)
        monkeypatch.setattr(win.search_ctrl, "_validate_run", lambda: True)

        from compare.compare_args import CompareArgs

        win.search_ctrl._run_with_progress(win.search_ctrl._run_compare, args=[CompareArgs()])
        qtbot.waitUntil(lambda: win.search_ctrl.is_compare_running(), timeout=2000)
        qtbot.waitUntil(
            lambda: len(started) > 0 and not win.search_ctrl.is_compare_running(),
            timeout=5000,
        )

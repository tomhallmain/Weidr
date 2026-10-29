"""Show-new-media queue: queued files must pass prevalidation like any other navigation."""


class TestShowNextQueuedMediaPrevalidation:

    def _prime(self, win, monkeypatch, files, skipped):
        win._new_media_queue.clear()
        win._new_media_queue_set.clear()
        win._new_media_queue.extend(files)
        win._new_media_queue_set.update(files)
        checked = []

        def fake_skip(path):
            checked.append(path)
            return path in skipped

        shown = []
        monkeypatch.setattr(win.compare_manager, "skip_media", fake_skip)
        monkeypatch.setattr(win.media_navigator, "create_media", lambda p, *a, **kw: shown.append(p))
        return checked, shown

    def test_skipped_file_is_not_shown_and_next_is(self, window_with_dir, monkeypatch):
        win, _ = window_with_dir
        files = win.file_browser.get_files()[:2]
        checked, shown = self._prime(win, monkeypatch, files, skipped={files[0]})

        assert win._show_next_queued_media() is True

        assert checked == files
        assert shown == [files[1]]
        assert not win._new_media_queue

    def test_all_skipped_shows_nothing(self, window_with_dir, monkeypatch):
        win, _ = window_with_dir
        files = win.file_browser.get_files()[:2]
        checked, shown = self._prime(win, monkeypatch, files, skipped=set(files))

        assert win._show_next_queued_media() is False

        assert checked == files
        assert shown == []
        assert not win._new_media_queue_set

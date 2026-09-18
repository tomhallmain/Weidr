"""Switching to a directory whose cached recursive setting differs from the
current one."""

import os

from tests.ui.app_window_fixtures import make_png


def test_switch_from_recursive_to_non_recursive_dir(window, tmp_path, qtbot):
    from utils.app_info_cache import app_info_cache

    # Previous directory: media only in a subdirectory, so a non-recursive
    # listing of it is empty.
    recursive_dir = tmp_path / "recursive_dir"
    (recursive_dir / "sub").mkdir(parents=True)
    make_png(str(recursive_dir / "sub" / "a.png"))
    flat_dir = tmp_path / "flat_dir"
    flat_dir.mkdir()
    flat_file = str(flat_dir / "b.png")
    make_png(flat_file)
    app_info_cache.set(str(recursive_dir), "recursive", True)
    app_info_cache.set(str(flat_dir), "recursive", False)

    win = window
    win.set_base_dir(str(recursive_dir))
    qtbot.waitUntil(lambda: win.base_dir == str(recursive_dir), timeout=2000)
    assert win.file_browser.recursive
    qtbot.waitUntil(lambda: win.media_path is not None, timeout=3000)

    win.set_base_dir(str(flat_dir))
    qtbot.waitUntil(lambda: win.base_dir == str(flat_dir), timeout=2000)

    assert not win.file_browser.recursive
    assert not win.sidebar_panel.recursive_check.isChecked()
    qtbot.waitUntil(
        lambda: win.media_path is not None
        and os.path.normpath(win.media_path) == os.path.normpath(flat_file),
        timeout=3000,
    )


def test_toggle_recursive_during_incremental_load(window_with_dir, qtbot, monkeypatch):
    """The rescan after a toggle can be an incremental load whose list is still
    empty when the toggle handler returns; the first file shows once loaded."""
    import threading

    from files.file_browser import FileBrowser

    win, media_dir = window_with_dir
    qtbot.waitUntil(lambda: win.media_path is not None, timeout=3000)

    release = threading.Event()
    original_worker = FileBrowser._incremental_load_worker

    def held_worker(self):
        release.wait(timeout=5)
        original_worker(self)

    monkeypatch.setattr("files.file_browser.is_slow_file_count", lambda *a, **k: True)
    monkeypatch.setattr(FileBrowser, "_incremental_load_worker", held_worker)

    win.sidebar_panel.recursive_check.setChecked(
        not win.sidebar_panel.recursive_check.isChecked()
    )
    assert win.file_browser.is_incremental_loading
    assert win.media_path is None

    release.set()
    qtbot.waitUntil(
        lambda: not win.file_browser.is_incremental_loading and win.media_path is not None,
        timeout=5000,
    )
    assert os.path.dirname(win.media_path) == media_dir


def test_toggle_recursive_off_with_no_top_level_files(window, tmp_path, qtbot):
    from utils.app_info_cache import app_info_cache

    base = tmp_path / "base"
    (base / "sub").mkdir(parents=True)
    make_png(str(base / "sub" / "a.png"))
    app_info_cache.set(str(base), "recursive", True)

    win = window
    win.set_base_dir(str(base))
    qtbot.waitUntil(lambda: win.base_dir == str(base), timeout=2000)
    qtbot.waitUntil(lambda: win.media_path is not None, timeout=3000)

    win.sidebar_panel.recursive_check.setChecked(False)

    assert not win.file_browser.recursive
    assert not win.file_browser.has_files()
    assert win.media_path is None

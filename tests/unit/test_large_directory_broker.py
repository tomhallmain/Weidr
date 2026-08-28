"""Unit tests for the bulk large-directory confirmation broker.

Restoring N large directories used to raise N modal prompts from inside window
construction. The broker settles every confirmation up front, in one dialog,
before any window exists.

The crash it fixes is a Qt re-entrancy race and will not reproduce
deterministically, so these target the invariants that make it unreachable
instead: the pre-flight and the per-window check must agree on what "large"
means, and a confirmed directory must short-circuit the per-window check
without gathering (which is both the double-scan fix and the guarantee that no
modal can be raised mid-construction).
"""

import os
from types import SimpleNamespace

import pytest
from PIL import Image

from files.file_browser import FileBrowser, count_matching_files, is_slow_file_count
from utils.config import config
from utils.utils import Utils


def _make_png(path) -> None:
    Image.new("RGB", (4, 4), (128, 128, 128)).save(str(path), format="PNG")


@pytest.fixture(autouse=True)
def _isolate_shared_state(monkeypatch):
    monkeypatch.setattr(config, "file_types", [".png"])
    FileBrowser.have_confirmed_directories.clear()
    yield
    FileBrowser.have_confirmed_directories.clear()


@pytest.fixture
def dir_with_files(tmp_path):
    def _build(name: str, count: int, nested: int = 0):
        root = tmp_path / name
        root.mkdir()
        for i in range(count):
            _make_png(root / f"f{i}.png")
        (root / "notes.txt").write_text("skip", encoding="utf-8")
        if nested:
            sub = root / "sub"
            sub.mkdir()
            for i in range(nested):
                _make_png(sub / f"n{i}.png")
        return str(root)
    return _build


class TestCountMatchingFiles:
    def test_counts_only_configured_extensions(self, dir_with_files):
        assert count_matching_files(dir_with_files("d", 3)) == 3

    def test_non_recursive_excludes_subdirectories(self, dir_with_files):
        path = dir_with_files("d", 2, nested=5)
        assert count_matching_files(path, recursive=False) == 2

    def test_recursive_includes_subdirectories(self, dir_with_files):
        path = dir_with_files("d", 2, nested=5)
        assert count_matching_files(path, recursive=True) == 7

    def test_unreadable_directory_counts_zero(self, tmp_path):
        """A directory that cannot be scanned is not one worth warning about."""
        assert count_matching_files(str(tmp_path / "does_not_exist")) == 0

    def test_extension_match_is_case_insensitive(self, tmp_path):
        root = tmp_path / "d"
        root.mkdir()
        Image.new("RGB", (4, 4)).save(str(root / "UPPER.PNG"), format="PNG")
        assert count_matching_files(str(root)) == 1


class TestSizeRuleIsShared:
    """The pre-flight and the per-window check must never disagree: a
    directory cleared by one but flagged by the other puts a modal back inside
    a window constructor."""

    def test_agrees_with_file_browser_for_local_dir(self, dir_with_files, monkeypatch):
        monkeypatch.setattr(Utils, "is_external_drive", staticmethod(lambda p: False))
        path = dir_with_files("d", 6)
        fb = FileBrowser(path, recursive=False)
        fb.set_directory(path)

        for threshold in (3, 5, 6, 7, 100):
            assert is_slow_file_count(
                count_matching_files(path), path, threshold=threshold
            ) == fb.is_slow_total_files(threshold=threshold)

    def test_agrees_with_file_browser_for_external_dir(self, dir_with_files, monkeypatch):
        monkeypatch.setattr(Utils, "is_external_drive", staticmethod(lambda p: True))
        path = dir_with_files("d", 6)
        fb = FileBrowser(path, recursive=False)
        fb.set_directory(path)

        for threshold in (10, 29, 30, 31, 100):
            assert is_slow_file_count(
                count_matching_files(path), path, threshold=threshold
            ) == fb.is_slow_total_files(threshold=threshold)

    def test_external_drive_applies_the_five_times_factor(self, monkeypatch):
        monkeypatch.setattr(Utils, "is_external_drive", staticmethod(lambda p: True))
        assert is_slow_file_count(201, "E:/x", threshold=1000) is True
        assert is_slow_file_count(200, "E:/x", threshold=1000) is False

    def test_local_drive_uses_the_raw_count(self, monkeypatch):
        monkeypatch.setattr(Utils, "is_external_drive", staticmethod(lambda p: False))
        assert is_slow_file_count(1001, "C:/x", threshold=1000) is True
        assert is_slow_file_count(1000, "C:/x", threshold=1000) is False


class TestRestorePreflight:
    """_confirm_large_directories_for_restore is exercised against a stub self:
    it touches only self.notification_ctrl, so a real window is not needed."""

    #: Sentinel for "the user left every box checked" -- resolved from the
    #: labels the dialog was actually given, so the test never has to rebuild
    #: the label format the broker uses.
    CHECK_ALL = object()

    def _run(self, dirs, alert_result, recorded=None):
        from ui.app_window.app_window import AppWindow

        def _alert(title, message, **kwargs):
            if recorded is not None:
                recorded.append(kwargs)
            if alert_result is self.CHECK_ALL:
                return [label for label, _checked in kwargs["items"]]
            return alert_result

        stub = SimpleNamespace(notification_ctrl=SimpleNamespace(alert=_alert))
        return AppWindow._confirm_large_directories_for_restore(stub, dirs)

    @pytest.fixture(autouse=True)
    def _clear_counts(self):
        from ui.app_window.app_window import AppWindow
        AppWindow._preflight_file_counts.clear()
        yield
        AppWindow._preflight_file_counts.clear()

    @pytest.fixture
    def _local_drive(self, monkeypatch):
        monkeypatch.setattr(Utils, "is_external_drive", staticmethod(lambda p: False))

    def test_no_large_directories_asks_nothing(self, dir_with_files, _local_drive):
        dirs = [dir_with_files("a", 2), dir_with_files("b", 3)]
        recorded: list = []
        assert self._run(dirs, alert_result=False, recorded=recorded) == dirs
        assert recorded == []  # no dialog raised at all

    def test_empty_input_is_returned_unchanged(self, _local_drive):
        assert self._run([], alert_result=False) == []

    def test_confirmed_directories_are_marked_and_restored(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        big = dir_with_files("big", 5)
        small = dir_with_files("small", 1)
        recorded: list = []

        result = self._run(
            [big, small], alert_result=self.CHECK_ALL, recorded=recorded
        )

        assert big in result and small in result
        assert big in FileBrowser.have_confirmed_directories
        # Only the large one is offered; the small one was never in question.
        assert len(recorded[0]["items"]) == 1
        # Every box starts checked -- restoring what was open is the default.
        assert all(checked for _label, checked in recorded[0]["items"])

    def test_declined_directories_are_dropped_from_the_restore_list(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        big = dir_with_files("big", 5)
        small = dir_with_files("small", 1)

        result = self._run([big, small], alert_result=[])  # accepted, none checked

        assert big not in result
        assert small in result  # never large, never offered
        assert big not in FileBrowser.have_confirmed_directories

    def test_partial_selection_keeps_only_the_checked_directories(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        """The per-directory case: keep one large directory, drop the other."""
        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        keep = dir_with_files("keep", 5)
        drop = dir_with_files("drop", 6)
        recorded: list = []

        def _check_only_keep(title, message, **kwargs):
            recorded.append(kwargs)
            return [
                label for label, _checked in kwargs["items"]
                if label.startswith(keep)
            ]

        from ui.app_window.app_window import AppWindow
        stub = SimpleNamespace(
            notification_ctrl=SimpleNamespace(alert=_check_only_keep)
        )
        result = AppWindow._confirm_large_directories_for_restore(stub, [keep, drop])

        assert result == [keep]
        assert keep in FileBrowser.have_confirmed_directories
        assert drop not in FileBrowser.have_confirmed_directories

    def test_cancelling_skips_every_large_directory(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        big_a = dir_with_files("a", 5)
        big_b = dir_with_files("b", 6)

        result = self._run([big_a, big_b], alert_result=False)

        assert result == []
        assert FileBrowser.have_confirmed_directories == []

    def test_already_confirmed_directories_are_not_re_asked(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        big = dir_with_files("big", 5)
        FileBrowser.have_confirmed_directories.append(big)
        recorded: list = []

        assert self._run([big], alert_result=False, recorded=recorded) == [big]
        assert recorded == []

    def test_missing_directories_are_not_offered(self, tmp_path, _local_drive):
        absent = str(tmp_path / "gone")
        recorded: list = []
        assert self._run([absent], alert_result=False, recorded=recorded) == [absent]
        assert recorded == []

    def test_counts_are_cached_for_every_directory_counted(
        self, dir_with_files, _local_drive
    ):
        """Not just the large ones: a small directory would otherwise be
        gathered again purely to be re-measured."""
        from ui.app_window.app_window import AppWindow

        small = dir_with_files("small", 2)
        self._run([small], alert_result=False)
        assert AppWindow._preflight_file_counts[small] == 2

    def test_primary_directory_is_included_in_the_one_dialog(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        """The primary window's own directory must not prompt separately: it
        loads from its own timer, so leaving it out of the pre-flight gives the
        user two dialogs -- the consolidated one plus the primary's."""
        from ui.app_window.app_window import AppWindow

        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        primary = dir_with_files("primary", 5)
        secondary = dir_with_files("secondary", 6)
        monkeypatch.setattr(
            "utils.app_info_cache.app_info_cache.get_meta",
            lambda key, default_val=None: [secondary] if key == "secondary_base_dirs" else default_val,
        )

        recorded: list = []
        loaded: list = []
        armed: list = []
        monkeypatch.setattr(
            "ui.app_window.app_window.QTimer",
            SimpleNamespace(singleShot=lambda ms, fn: armed.append(fn)),
        )
        stub = SimpleNamespace(
            notification_ctrl=SimpleNamespace(alert=None),
            _pending_restore_dirs=None,
            set_base_dir=lambda bd: loaded.append(bd),
            _restore_secondary_windows=lambda: None,
        )

        def _alert(title, message, **kwargs):
            recorded.append(kwargs)
            # The restore must not be armed yet: a modal pumps the event loop,
            # so an already-armed timer fires here, before the decision exists.
            assert armed == []
            return [label for label, _checked in kwargs["items"]]

        stub.notification_ctrl.alert = _alert
        stub._confirm_large_directories_for_restore = (
            lambda dirs: AppWindow._confirm_large_directories_for_restore(stub, dirs)
        )

        AppWindow._startup_load(stub, primary)
        assert armed == [stub._restore_secondary_windows]

        assert len(recorded) == 1
        offered = " ".join(label for label, _c in recorded[0]["items"])
        assert primary in offered and secondary in offered
        assert loaded == [primary]
        assert stub._pending_restore_dirs == [secondary]

    def test_declined_primary_directory_is_not_loaded(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        from ui.app_window.app_window import AppWindow

        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        primary = dir_with_files("primary", 5)
        monkeypatch.setattr(
            "utils.app_info_cache.app_info_cache.get_meta",
            lambda key, default_val=None: [],
        )

        loaded: list = []
        monkeypatch.setattr(
            "ui.app_window.app_window.QTimer",
            SimpleNamespace(singleShot=lambda ms, fn: None),
        )
        stub = SimpleNamespace(
            notification_ctrl=SimpleNamespace(alert=lambda *a, **k: []),
            _pending_restore_dirs=None,
            set_base_dir=lambda bd: loaded.append(bd),
            _restore_secondary_windows=lambda: None,
        )
        stub._confirm_large_directories_for_restore = (
            lambda dirs: AppWindow._confirm_large_directories_for_restore(stub, dirs)
        )

        AppWindow._startup_load(stub, primary)

        assert loaded == []
        assert primary not in FileBrowser.have_confirmed_directories

    def test_restore_consumes_the_preflight_without_asking_again(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        """The second dialog the user saw came from restore re-running the
        pre-flight; it must consume what _startup_load already decided."""
        from ui.app_window.app_window import AppWindow

        opened: list = []
        monkeypatch.setattr(
            "ui.app_window.window_manager.WindowManager.add_secondary_window",
            staticmethod(lambda *a, **k: opened.append(a[0] if a else None)),
        )
        # Replace the module-level name, not QTimer.singleShot: attributes on
        # PySide6 types are not writable, and no event loop is running here.
        monkeypatch.setattr(
            "ui.app_window.app_window.QTimer",
            SimpleNamespace(singleShot=lambda ms, fn: None),
        )

        def _boom(dirs):
            raise AssertionError("restore re-ran the pre-flight")

        stub = SimpleNamespace(
            _pending_restore_dirs=["/x/a", "/x/b"],
            _confirm_large_directories_for_restore=_boom,
            _refocus_primary=lambda: None,
        )
        AppWindow._restore_secondary_windows(stub)

        assert opened == ["/x/a", "/x/b"]
        assert stub._pending_restore_dirs is None  # consumed, not reused

    def test_restore_without_a_decision_opens_nothing_and_does_not_ask(
        self, monkeypatch
    ):
        """Firing before _startup_load decided must not fall back to its own
        confirmation -- that is the second dialog, listing only secondaries."""
        from ui.app_window.app_window import AppWindow

        opened: list = []
        monkeypatch.setattr(
            "ui.app_window.window_manager.WindowManager.add_secondary_window",
            staticmethod(lambda *a, **k: opened.append(a[0] if a else None)),
        )
        monkeypatch.setattr(
            "ui.app_window.app_window.QTimer",
            SimpleNamespace(singleShot=lambda ms, fn: None),
        )

        def _boom(dirs):
            raise AssertionError("restore raised a second confirmation")

        stub = SimpleNamespace(
            _pending_restore_dirs=None,
            _confirm_large_directories_for_restore=_boom,
            _refocus_primary=lambda: None,
        )
        AppWindow._restore_secondary_windows(stub)

        assert opened == []

    def test_dialog_is_raised_once_for_many_large_directories(
        self, dir_with_files, monkeypatch, _local_drive
    ):
        monkeypatch.setattr(
            "files.file_browser.is_slow_file_count", lambda c, d, threshold=5000: c > 1
        )
        dirs = [dir_with_files(f"d{i}", 5) for i in range(4)]
        recorded: list = []

        self._run(dirs, alert_result=[], recorded=recorded)

        assert len(recorded) == 1
        assert len(recorded[0]["items"]) == 4
        assert recorded[0]["severity"] == "high"

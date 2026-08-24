"""A headless caller must be able to open a directory and mark/move/delete
files, not only run comparisons.

FileBrowser and MarkedFiles are Qt-free, so this drives the same "manual"
mark-then-move and delete paths a human takes through the UI, using only those
plus the headless AppActions -- no compare involved.

Delete goes through MarkedFiles.delete_file_static(), which
FileOpsController._handle_delete also delegates to, so the GUI and a headless
caller share one implementation instead of two.

Lives in tests/unit/ rather than in tests/compare/test_headless_compare_run.py
because nothing here runs a compare.
"""

from PIL import Image

from files.file_browser import FileBrowser
from files.marked_files import MarkedFiles
from utils.config import config
from utils.headless_app_actions import build_headless_app_actions


def _make_png(path) -> None:
    Image.new("RGB", (4, 4), (128, 128, 128)).save(path, format="PNG")


class TestHeadlessFileMarking:
    def test_open_directory_mark_and_move_files(self, tmp_path, monkeypatch):
        # FileBrowser filters by config.file_types; a plain .txt file never
        # shows up regardless of directory contents.
        monkeypatch.setattr(config, "file_types", [".png"])

        source = tmp_path / "src"
        target = tmp_path / "dst"
        source.mkdir()
        target.mkdir()
        paths = []
        for i in range(3):
            p = source / f"file{i}.png"
            _make_png(p)
            paths.append(str(p))

        browser = FileBrowser(str(source))
        browser.set_directory(str(source))
        assert set(browser.get_files()) == set(paths)

        app_actions = build_headless_app_actions({
            "get_base_dir": lambda: str(source),
            "is_compare_running": lambda: False,
        })
        for path in browser.get_files():
            assert MarkedFiles.add_mark_if_not_present(path, app_actions)
        assert set(MarkedFiles.file_marks) == set(paths)

        some_files_already_present, exceptions_present = MarkedFiles.move_marks_to_dir_static(
            app_actions,
            target_dir=str(target),
        )

        assert some_files_already_present is False
        assert exceptions_present is False
        assert sorted(p.name for p in target.iterdir()) == ["file0.png", "file1.png", "file2.png"]
        assert list(source.iterdir()) == []
        assert MarkedFiles.file_marks == []

    def test_delete_file(self, tmp_path, monkeypatch):
        # delete_instantly removes directly via os.remove -- deterministic,
        # no dependency on send2trash or a configured trash folder.
        monkeypatch.setattr(config, "delete_instantly", True)

        target_file = tmp_path / "unwanted.png"
        _make_png(target_file)
        alerts = []
        app_actions = build_headless_app_actions({
            "_alert": lambda title, message, **kw: alerts.append((title, message)) or False,
        })

        success = MarkedFiles.delete_file_static(str(target_file), app_actions)

        assert success is True
        assert not target_file.exists()
        assert alerts == []

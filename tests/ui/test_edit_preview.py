"""preview_and_confirm_op: nothing is written until the user accepts.

The dialog itself is stubbed -- these tests are about what reaches the disk
and how often the op runs, not about the widget.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from PIL import Image

from image.image_ops import ImageOps
from ui.image import edit_preview


def _png(path, color=(10, 20, 30)) -> str:
    Image.new("RGB", (16, 16), color).save(str(path), format="PNG")
    return str(path)


def _writer(calls, color=(1, 2, 3)):
    """A stand-in op: records each call and writes a real file."""
    def _run(out_path):
        calls.append(out_path)
        Image.new("RGB", (16, 16), color).save(out_path, format="PNG")
    return _run


def _preview_path(ext=".png") -> str:
    return os.path.join(tempfile.gettempdir(), edit_preview._PREVIEW_BASENAME + ext)


@pytest.fixture(autouse=True)
def _isolated_preview_dir(tmp_path, monkeypatch):
    """Render previews into a per-test directory.

    The preview basename is fixed and sits in the system temp directory, so it
    is shared with a running app and with every other test: a leftover or a
    concurrent run would otherwise pass for this test's candidate. Kept out of
    tmp_path itself so the assertions about the source directory's contents
    still mean something.
    """
    preview_dir = tmp_path / "preview_temp"
    preview_dir.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(preview_dir))
    return preview_dir


class TestPreviewAndConfirmOp:
    def test_accept_writes_the_candidate_beside_the_source(self, tmp_path, qtbot, monkeypatch):
        source = _png(tmp_path / "a.png")
        calls = []
        monkeypatch.setattr(
            edit_preview, "show_preview_confirm_dialog", lambda *a, **kw: True
        )

        result = edit_preview.preview_and_confirm_op(
            None, MagicMock(), source, _writer(calls),
            suffix=ImageOps.SCRAMBLE_SUFFIX, title="Preview",
        )

        assert result == str(tmp_path / ("a" + ImageOps.SCRAMBLE_SUFFIX + ".png"))
        assert os.path.isfile(result)
        assert len(calls) == 1
        assert not os.path.isfile(_preview_path())

    def test_cancel_writes_nothing(self, tmp_path, qtbot, monkeypatch):
        source = _png(tmp_path / "a.png")
        calls = []
        monkeypatch.setattr(
            edit_preview, "show_preview_confirm_dialog", lambda *a, **kw: False
        )

        result = edit_preview.preview_and_confirm_op(
            None, MagicMock(), source, _writer(calls),
            suffix=ImageOps.SCRAMBLE_SUFFIX, title="Preview",
        )

        assert result is None
        assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["a.png"]
        assert not os.path.isfile(_preview_path())

    def test_reroll_runs_the_op_again_into_the_same_temp_file(self, tmp_path, qtbot, monkeypatch):
        source = _png(tmp_path / "a.png")
        calls = []

        def _dialog(master, preview_path, on_reroll, **kwargs):
            on_reroll()
            on_reroll()
            return False

        monkeypatch.setattr(edit_preview, "show_preview_confirm_dialog", _dialog)

        edit_preview.preview_and_confirm_op(
            None, MagicMock(), source, _writer(calls),
            suffix=ImageOps.SCRAMBLE_SUFFIX, title="Preview",
        )

        # One initial render plus two rerolls, all to the one temp path.
        assert len(calls) == 3
        assert set(calls) == {_preview_path()}

    def test_accepting_twice_keeps_both_results(self, tmp_path, qtbot, monkeypatch):
        source = _png(tmp_path / "a.png")
        monkeypatch.setattr(
            edit_preview, "show_preview_confirm_dialog", lambda *a, **kw: True
        )

        first = edit_preview.preview_and_confirm_op(
            None, MagicMock(), source, _writer([]),
            suffix=ImageOps.SCRAMBLE_SUFFIX, title="Preview",
        )
        second = edit_preview.preview_and_confirm_op(
            None, MagicMock(), source, _writer([]),
            suffix=ImageOps.SCRAMBLE_SUFFIX, title="Preview",
        )

        assert first != second
        assert os.path.isfile(first)
        assert os.path.isfile(second)

    def test_a_failed_render_reports_and_writes_nothing(self, tmp_path, qtbot, monkeypatch):
        source = _png(tmp_path / "a.png")
        app_actions = MagicMock()
        monkeypatch.setattr(
            edit_preview, "show_preview_confirm_dialog", _fail_if_called
        )

        def _boom(out_path):
            raise RuntimeError("op failed")

        result = edit_preview.preview_and_confirm_op(
            None, app_actions, source, _boom,
            suffix=ImageOps.SCRAMBLE_SUFFIX, title="Preview",
        )

        assert result is None
        app_actions.warn.assert_called_once()
        assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["a.png"]


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("the dialog must not open when the first render fails")

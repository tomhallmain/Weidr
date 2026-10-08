"""
UI tests for HfModelManagerWindow's model-file handling: repo-file combo
pre-selection, backend combo choices, and the confirmation shown before
installing a file without a model extension.
"""

from unittest.mock import MagicMock

from PySide6.QtWidgets import QComboBox, QTreeWidgetItem

from image.image_classifier import BackendType
from ui.compare.hf_model_manager_window_qt import HfModelManagerWindow, _populate_repo_file_combo
from utils.translations import _


def _make_window(qtbot):
    win = HfModelManagerWindow(parent=None, app_actions=MagicMock())
    qtbot.addWidget(win)
    return win


def _editable_combo(qtbot, text=""):
    combo = QComboBox()
    combo.setEditable(True)
    qtbot.addWidget(combo)
    combo.setEditText(text)
    return combo


class TestPopulateRepoFileCombo:
    def test_repo_without_model_files_preselects_nothing(self, qtbot):
        combo = _editable_combo(qtbot)
        _populate_repo_file_combo(combo, [".gitattributes", "README.md"], {".tflite"})
        assert combo.count() == 2
        assert combo.currentText() == ""

    def test_root_model_file_is_preselected(self, qtbot):
        combo = _editable_combo(qtbot)
        files = ["checkpoint-1/model.safetensors", "model.safetensors"]
        _populate_repo_file_combo(combo, files, {".safetensors"})
        assert combo.currentText() == "model.safetensors"

    def test_current_text_kept_when_still_offered(self, qtbot):
        combo = _editable_combo(qtbot, "b.onnx")
        _populate_repo_file_combo(combo, ["a.onnx", "b.onnx"], {".onnx"})
        assert combo.currentText() == "b.onnx"

    def test_no_files_falls_back(self, qtbot):
        combo = _editable_combo(qtbot)
        _populate_repo_file_combo(combo, [], {".onnx"}, fallback_text="model.safetensors")
        assert combo.currentText() == "model.safetensors"


def test_backend_combo_offers_every_backend(qtbot):
    win = _make_window(qtbot)
    items = [win._backend_combo.itemText(i) for i in range(win._backend_combo.count())]
    assert items == BackendType.config_values()
    assert "tflite" in items


class TestNonModelFileInstallGuard:
    def _prepare(self, qtbot, monkeypatch, filename, auto_model):
        win = _make_window(qtbot)
        api_mock = MagicMock()
        api_mock.list_model_files.return_value = []
        api_mock.download_snapshot.side_effect = RuntimeError("stop after guard")
        monkeypatch.setattr(win, "_api", lambda: api_mock)
        QTreeWidgetItem(win._search_tree, ["org/model", "image-classification", "0", "0", "unknown", "no"])
        win._search_tree.topLevelItem(0).setSelected(True)
        win._filename_combo.setEditText(filename)
        win._use_transformers_auto_model_cb.setChecked(auto_model)
        return win, api_mock

    def test_declining_the_confirmation_skips_download(self, qtbot, monkeypatch):
        win, api_mock = self._prepare(qtbot, monkeypatch, ".gitattributes", auto_model=False)
        win._app_actions.alert.return_value = False
        win._download_and_install_selected()
        assert win._app_actions.alert.call_args.args[0] == _("Not a Model File?")
        api_mock.download_snapshot.assert_not_called()

    def test_confirming_proceeds_to_download(self, qtbot, monkeypatch):
        win, api_mock = self._prepare(qtbot, monkeypatch, ".gitattributes", auto_model=False)
        win._app_actions.alert.return_value = True
        win._download_and_install_selected()
        api_mock.download_snapshot.assert_called_once()

    def test_model_file_needs_no_confirmation(self, qtbot, monkeypatch):
        win, api_mock = self._prepare(qtbot, monkeypatch, "model.tflite", auto_model=False)
        win._download_and_install_selected()
        titles = [c.args[0] for c in win._app_actions.alert.call_args_list]
        assert _("Not a Model File?") not in titles
        api_mock.download_snapshot.assert_called_once()

    def test_transformers_auto_model_needs_no_confirmation(self, qtbot, monkeypatch):
        win, api_mock = self._prepare(qtbot, monkeypatch, "config.json", auto_model=True)
        win._download_and_install_selected()
        titles = [c.args[0] for c in win._app_actions.alert.call_args_list]
        assert _("Not a Model File?") not in titles
        api_mock.download_snapshot.assert_called_once()

"""
Targeted UI tests for the audio-classification support added to
HfModelManagerWindow: the HF Hub Search tab's install flow, and the Installed
Models tab's visibility + Remove for audio entries (Edit/Test/Preload
deliberately stay image-only -- guarded with a clear warning instead of a
confusing failure). Does not attempt to cover the window's pre-existing
image-classifier behavior beyond what these changes touch.
"""

from unittest.mock import MagicMock

from PySide6.QtWidgets import QTreeWidgetItem

from ui.compare.hf_model_manager_window_qt import HfModelManagerWindow
from utils.config import config


def _make_window(qtbot):
    win = HfModelManagerWindow(parent=None, app_actions=MagicMock())
    qtbot.addWidget(win)
    return win


def _add_search_row(win, repo_id: str, task: str) -> None:
    QTreeWidgetItem(win._search_tree, [repo_id, task, "0", "0", "unknown", "no"])


class TestSelectedRepoIsAudio:
    def test_false_when_nothing_selected(self, qtbot):
        win = _make_window(qtbot)
        assert win._selected_repo_is_audio() is False

    def test_true_for_audio_classification_task(self, qtbot):
        win = _make_window(qtbot)
        _add_search_row(win, "org/audio-model", "audio-classification")
        win._search_tree.topLevelItem(0).setSelected(True)
        assert win._selected_repo_is_audio() is True

    def test_true_for_zero_shot_audio_classification_task(self, qtbot):
        win = _make_window(qtbot)
        _add_search_row(win, "org/clap-model", "zero-shot-audio-classification")
        win._search_tree.topLevelItem(0).setSelected(True)
        assert win._selected_repo_is_audio() is True

    def test_false_for_image_classification_task(self, qtbot):
        win = _make_window(qtbot)
        _add_search_row(win, "org/image-model", "image-classification")
        win._search_tree.topLevelItem(0).setSelected(True)
        assert win._selected_repo_is_audio() is False

    def test_mixed_results_use_the_selected_row_not_the_search_filter(self, qtbot):
        """A search run under 'All tasks' can return mixed rows -- the check must
        read the selected row's own task, not whatever filter was used to find it."""
        win = _make_window(qtbot)
        _add_search_row(win, "org/image-model", "image-classification")
        _add_search_row(win, "org/audio-model", "audio-classification")
        win._search_tree.topLevelItem(0).setSelected(True)
        assert win._selected_repo_is_audio() is False
        win._search_tree.topLevelItem(0).setSelected(False)
        win._search_tree.topLevelItem(1).setSelected(True)
        assert win._selected_repo_is_audio() is True


class TestInstallSelectedAudioSearchResult:
    def test_empty_model_name_warns_and_does_not_persist(self, qtbot, monkeypatch):
        win = _make_window(qtbot)
        win._model_name_edit.setText("")
        monkeypatch.setattr(win, "_default_model_name", lambda repo_id: "")
        persist_mock = MagicMock()
        monkeypatch.setattr(win, "_persist_model_details", persist_mock)
        win._install_selected_audio_search_result("org/audio-model")
        persist_mock.assert_not_called()
        win._app_actions.warn.assert_called_once()

    def test_empty_categories_warns_and_does_not_persist(self, qtbot, monkeypatch):
        win = _make_window(qtbot)
        win._model_name_edit.setText("my_audio_model")
        win._categories_edit.setText("")
        persist_mock = MagicMock()
        monkeypatch.setattr(win, "_persist_model_details", persist_mock)
        win._install_selected_audio_search_result("org/audio-model")
        persist_mock.assert_not_called()
        win._app_actions.warn.assert_called_once()

    def test_valid_input_persists_audio_shaped_details(self, qtbot, monkeypatch):
        win = _make_window(qtbot)
        win._model_name_edit.setText("my_audio_model")
        win._categories_edit.setText("safe, explicit")
        persist_mock = MagicMock(return_value=True)
        monkeypatch.setattr(win, "_persist_model_details", persist_mock)
        win._install_selected_audio_search_result("org/audio-model")

        persist_mock.assert_called_once()
        (model_details,), kwargs = persist_mock.call_args
        assert model_details["model_name"] == "my_audio_model"
        assert model_details["model_location"] == "org/audio-model"
        assert model_details["model_categories"] == ["safe", "explicit"]
        assert model_details["hf_repo_id"] == "org/audio-model"
        # No image-only fields should leak into an audio install.
        assert "backend" not in model_details
        assert "hf_selected_filename" not in model_details
        assert kwargs["classifier_type"] == "audio"

    def test_invalid_config_alerts_and_does_not_persist(self, qtbot, monkeypatch):
        win = _make_window(qtbot)
        win._model_name_edit.setText("my_audio_model")
        win._categories_edit.setText("safe")
        monkeypatch.setattr(
            "ui.compare.hf_model_manager_window_qt.AudioClassifierModelConfig.from_dict",
            MagicMock(side_effect=ValueError("boom")),
        )
        persist_mock = MagicMock()
        monkeypatch.setattr(win, "_persist_model_details", persist_mock)
        win._install_selected_audio_search_result("org/audio-model")
        persist_mock.assert_not_called()
        win._app_actions.alert.assert_called_once()


class TestDownloadAndInstallSelectedRoutesAudioAway:
    def test_audio_selection_delegates_to_audio_installer_not_image_flow(self, qtbot, monkeypatch):
        win = _make_window(qtbot)
        _add_search_row(win, "org/audio-model", "audio-classification")
        win._search_tree.topLevelItem(0).setSelected(True)

        audio_install_mock = MagicMock()
        monkeypatch.setattr(win, "_install_selected_audio_search_result", audio_install_mock)
        api_mock = MagicMock()
        monkeypatch.setattr(win, "_api", lambda: api_mock)

        win._download_and_install_selected()

        audio_install_mock.assert_called_once_with("org/audio-model")
        api_mock.download_snapshot.assert_not_called()


def _valid_image_model(name="my_image_model"):
    return {
        "model_name": name,
        "model_location": __file__,  # any real path so os.path.exists() passes
        "model_categories": ["a", "b"],
        "backend": "pytorch",
    }


def _valid_audio_model(name="my_audio_model", repo_id="org/audio-model"):
    return {
        "model_name": name,
        "model_location": repo_id,  # bare repo id, deliberately not a local path
        "model_categories": ["safe", "explicit"],
        "hf_repo_id": repo_id,
    }


class TestRefreshInstalledModelsShowsBothTypes:
    def test_image_and_audio_entries_both_appear(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [_valid_image_model()])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        names = {win._installed_tree.topLevelItem(i).text(0) for i in range(win._installed_tree.topLevelItemCount())}
        assert names == {"my_image_model", "my_audio_model"}

    def test_type_column_stores_correct_raw_data(self, qtbot, monkeypatch):
        from PySide6.QtCore import Qt

        monkeypatch.setattr(config, "image_classifier_models", [_valid_image_model()])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        by_name = {
            win._installed_tree.topLevelItem(i).text(0): win._installed_tree.topLevelItem(i)
            for i in range(win._installed_tree.topLevelItemCount())
        }
        assert by_name["my_audio_model"].data(1, Qt.ItemDataRole.UserRole) == "audio"
        assert by_name["my_image_model"].data(1, Qt.ItemDataRole.UserRole) == "image"

    def test_audio_entry_valid_without_local_path(self):
        """Audio model_location is a bare repo id, never a local path -- must not
        be penalized by the image path's os.path.exists() requirement. Tests the
        static method directly rather than parsing the (translated,
        locale-dependent) notice label text."""
        model = _valid_audio_model()
        assert HfModelManagerWindow._is_valid_installed_audio_model(model) is True


class TestSelectedInstalledModelClassifierType:
    def test_defaults_to_image_when_nothing_selected(self, qtbot):
        win = _make_window(qtbot)
        assert win._selected_installed_model_classifier_type() == "image"

    def test_returns_audio_for_audio_row(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        win._installed_tree.topLevelItem(0).setSelected(True)
        assert win._selected_installed_model_classifier_type() == "audio"


class TestEditTestPreloadGuardAudioSelections:
    def _select_audio_row(self, win):
        win._installed_tree.topLevelItem(0).setSelected(True)

    def test_edit_warns_and_does_not_open_dialog(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        self._select_audio_row(win)
        win._edit_selected_installed_model()
        win._app_actions.warn.assert_called_once()
        assert "audio" in win._app_actions.warn.call_args[0][0].lower()

    def test_test_on_current_image_warns(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        self._select_audio_row(win)
        win._test_selected_model_on_current_image()
        win._app_actions.warn.assert_called_once()

    def test_preload_warns(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        self._select_audio_row(win)
        win._preload_selected_model()
        win._app_actions.warn.assert_called_once()


class TestRemoveSelectedInstalledModel:
    def _mock_api(self, win):
        api_mock = MagicMock()
        api_mock.has_connection.return_value = False  # skip HF cache-deletion branch entirely
        # is_repo_hosted() is called unconditionally whenever api_backend + repo_id
        # are truthy (not gated on has_connection) -- an unconfigured MagicMock
        # return value isn't unpackable into (bool, str), so this must be explicit.
        api_mock.is_repo_hosted.return_value = (True, "")
        win._hf_api = api_mock
        return api_mock

    def _select_row_named(self, win, name: str) -> None:
        count = win._installed_tree.topLevelItemCount()
        names = [win._installed_tree.topLevelItem(i).text(0) for i in range(count)]
        win._installed_tree.topLevelItem(names.index(name)).setSelected(True)

    def test_removes_audio_model_from_audio_config_not_image(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [_valid_image_model()])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        win._app_actions.alert.return_value = True
        self._mock_api(win)
        self._select_row_named(win, "my_audio_model")

        win._remove_selected_installed_model()

        assert config.audio_classifier_models == []
        assert len(config.image_classifier_models) == 1  # untouched

    def test_removes_image_model_from_image_config_not_audio(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [_valid_image_model()])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        win._app_actions.alert.return_value = True
        self._mock_api(win)
        self._select_row_named(win, "my_image_model")

        win._remove_selected_installed_model()

        assert config.image_classifier_models == []
        assert len(config.audio_classifier_models) == 1  # untouched

    def test_declining_confirmation_leaves_config_untouched(self, qtbot, monkeypatch):
        monkeypatch.setattr(config, "image_classifier_models", [])
        monkeypatch.setattr(config, "audio_classifier_models", [_valid_audio_model()])
        win = _make_window(qtbot)
        win._app_actions.alert.return_value = False
        self._mock_api(win)
        self._select_row_named(win, "my_audio_model")

        win._remove_selected_installed_model()

        assert len(config.audio_classifier_models) == 1

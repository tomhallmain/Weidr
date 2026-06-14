"""Tests for TypeConfigurationWindow (persistence, apply logic, UI smoke)."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from ui.files.type_configuration_window_qt import TypeConfigurationWindow
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.constants import CompareMediaType
from utils.translations import _

_tr = _

_TOGGLEABLE_TYPES = tuple(
    mt for mt in CompareMediaType if mt != CompareMediaType.UNCONFIGURED
)


def _reset_type_configuration_window() -> None:
    if TypeConfigurationWindow._instance is not None:
        try:
            TypeConfigurationWindow.on_closing()
        except RuntimeError:
            pass
    TypeConfigurationWindow._instance = None
    TypeConfigurationWindow._pending_changes.clear()
    TypeConfigurationWindow._original_config.clear()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture(autouse=True)
def _type_configuration_cleanup():
    yield
    _reset_type_configuration_window()


class TestTypeConfigurationPersistence:
    def test_load_pending_changes_reads_cache(self):
        app_info_cache.set_meta(
            "file_type_configuration",
            {"AUDIO": False, "VIDEO": True},
        )
        TypeConfigurationWindow.load_pending_changes()
        assert TypeConfigurationWindow._pending_changes[CompareMediaType.AUDIO] is False
        assert TypeConfigurationWindow._pending_changes[CompareMediaType.VIDEO] is True

    def test_save_pending_changes_merges_original_and_pending(self):
        TypeConfigurationWindow._original_config = {
            CompareMediaType.VIDEO: True,
            CompareMediaType.AUDIO: True,
        }
        TypeConfigurationWindow._pending_changes = {
            CompareMediaType.AUDIO: False,
        }
        TypeConfigurationWindow.save_pending_changes()
        stored = app_info_cache.get_meta("file_type_configuration", default_val={})
        assert stored["AUDIO"] is False
        assert stored["VIDEO"] is True


class TestTypeConfigurationApplyLogic:
    def test_get_initial_value_reflects_config_flags(self):
        config.enable_images = True
        config.enable_audio = False
        config.enable_videos = True
        assert TypeConfigurationWindow._get_initial_value(CompareMediaType.IMAGE) is True
        assert TypeConfigurationWindow._get_initial_value(CompareMediaType.AUDIO) is False
        assert TypeConfigurationWindow._get_initial_value(CompareMediaType.VIDEO) is True
        config.enable_images = False
        assert TypeConfigurationWindow._get_initial_value(CompareMediaType.IMAGE) is False

    def test_has_changes_detects_audio_toggle(self):
        TypeConfigurationWindow._original_config = {CompareMediaType.AUDIO: True}
        TypeConfigurationWindow._pending_changes = {CompareMediaType.AUDIO: True}
        assert TypeConfigurationWindow._has_changes() is False
        TypeConfigurationWindow._pending_changes[CompareMediaType.AUDIO] = False
        assert TypeConfigurationWindow._has_changes() is True

    def test_apply_changes_disables_audio_and_strips_extensions(self):
        config.enable_audio = True
        for ext in config.audio_types:
            if ext not in config.file_types:
                config.file_types.append(ext)

        TypeConfigurationWindow._original_config = {CompareMediaType.AUDIO: True}
        TypeConfigurationWindow._pending_changes = {CompareMediaType.AUDIO: False}
        TypeConfigurationWindow.apply_changes()

        assert config.enable_audio is False
        assert not set(config.audio_types) & set(config.file_types)

    def test_apply_changes_enables_audio_and_adds_extensions(self):
        config.enable_audio = False
        audio_set = set(config.audio_types)
        config.file_types = [e for e in config.file_types if e not in audio_set]

        TypeConfigurationWindow._pending_changes = {CompareMediaType.AUDIO: True}
        TypeConfigurationWindow.apply_changes()

        assert config.enable_audio is True
        for ext in config.audio_types:
            assert ext in config.file_types

    def test_apply_changes_disables_images_and_strips_extensions(self):
        config.enable_images = True
        for ext in config.image_types:
            if ext not in config.file_types:
                config.file_types.append(ext)

        TypeConfigurationWindow._original_config = {CompareMediaType.IMAGE: True}
        TypeConfigurationWindow._pending_changes = {CompareMediaType.IMAGE: False}
        TypeConfigurationWindow.apply_changes()

        assert config.enable_images is False
        assert not set(config.image_types) & set(config.file_types)

    def test_apply_changes_enables_images_and_adds_extensions(self):
        config.enable_images = False
        image_set = set(config.image_types)
        config.file_types = [e for e in config.file_types if e not in image_set]

        TypeConfigurationWindow._pending_changes = {CompareMediaType.IMAGE: True}
        TypeConfigurationWindow.apply_changes()

        assert config.enable_images is True
        for ext in config.image_types:
            assert ext in config.file_types

    def test_audio_has_description(self):
        desc = TypeConfigurationWindow.MEDIA_TYPE_DESCRIPTIONS[CompareMediaType.AUDIO]
        assert isinstance(desc, str) and len(desc) > 0


class TestTypeConfigurationWindowUI:
    def test_open_via_launcher_shows_all_media_checkboxes(
        self, window_with_dir, qtbot
    ):
        win, _ = window_with_dir
        win.window_launcher.open_type_configuration_window()
        qtbot.waitUntil(
            lambda: TypeConfigurationWindow._instance is not None,
            timeout=5000,
        )
        dlg = TypeConfigurationWindow._instance
        qtbot.addWidget(dlg)
        qtbot.waitExposed(dlg, timeout=3000)

        assert dlg.isVisible()
        assert set(dlg._checkboxes.keys()) == set(_TOGGLEABLE_TYPES)

        titles = [lbl.text() for lbl in dlg.findChildren(QLabel)]
        assert _tr("Configure Media Types") in titles

    def test_image_checkbox_enabled_and_reflects_config(
        self, window_with_dir, qtbot
    ):
        config.enable_images = True
        win, _ = window_with_dir
        win.window_launcher.open_type_configuration_window()
        qtbot.waitUntil(
            lambda: TypeConfigurationWindow._instance is not None,
            timeout=5000,
        )
        image_cb = TypeConfigurationWindow._instance._checkboxes[CompareMediaType.IMAGE]
        assert image_cb.isChecked()
        assert image_cb.isEnabled()

    def test_unchecking_image_stores_pending_change(self, window_with_dir, qtbot):
        config.enable_images = True
        win, _ = window_with_dir
        win.window_launcher.open_type_configuration_window()
        qtbot.waitUntil(
            lambda: TypeConfigurationWindow._instance is not None,
            timeout=5000,
        )
        image_cb = TypeConfigurationWindow._instance._checkboxes[CompareMediaType.IMAGE]
        assert image_cb.isChecked()

        with qtbot.waitSignal(image_cb.stateChanged):
            image_cb.setChecked(False)

        assert TypeConfigurationWindow._pending_changes.get(CompareMediaType.IMAGE) is False

    def test_audio_checkbox_matches_config(self, window_with_dir, qtbot):
        config.enable_audio = False
        win, _ = window_with_dir
        win.window_launcher.open_type_configuration_window()
        qtbot.waitUntil(
            lambda: TypeConfigurationWindow._instance is not None,
            timeout=5000,
        )
        audio_cb = TypeConfigurationWindow._instance._checkboxes[CompareMediaType.AUDIO]
        assert audio_cb.isChecked() is False
        assert audio_cb.isEnabled()

    def test_unchecking_audio_stores_pending_change(self, window_with_dir, qtbot):
        config.enable_audio = True
        win, _ = window_with_dir
        win.window_launcher.open_type_configuration_window()
        qtbot.waitUntil(
            lambda: TypeConfigurationWindow._instance is not None,
            timeout=5000,
        )
        audio_cb = TypeConfigurationWindow._instance._checkboxes[CompareMediaType.AUDIO]
        assert audio_cb.isChecked()

        with qtbot.waitSignal(audio_cb.stateChanged):
            audio_cb.setChecked(False)

        assert TypeConfigurationWindow._pending_changes.get(CompareMediaType.AUDIO) is False

    def test_second_open_reuses_singleton(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.window_launcher.open_type_configuration_window()
        qtbot.waitUntil(
            lambda: TypeConfigurationWindow._instance is not None,
            timeout=5000,
        )
        first_id = id(TypeConfigurationWindow._instance)

        win.window_launcher.open_type_configuration_window()
        assert id(TypeConfigurationWindow._instance) == first_id

    def test_apply_changes_refreshes_compares(self, window_with_dir, qtbot, monkeypatch):
        config.enable_audio = True
        refreshed: list[bool] = []

        class _RecordingActions:
            def refresh_all_compares(self):
                refreshed.append(True)

            def toast(self, *_args, **_kwargs):
                pass

        win, _ = window_with_dir
        win.window_launcher.open_type_configuration_window()
        qtbot.waitUntil(
            lambda: TypeConfigurationWindow._instance is not None,
            timeout=5000,
        )
        dlg = TypeConfigurationWindow._instance
        TypeConfigurationWindow._original_config = dict(
            TypeConfigurationWindow._original_config
        )
        TypeConfigurationWindow._pending_changes[CompareMediaType.AUDIO] = False

        monkeypatch.setattr(
            dlg._app_actions,
            "find_window_with_compare",
            lambda: None,
        )
        TypeConfigurationWindow.apply_changes(_RecordingActions())

        assert refreshed == [True]
        assert config.enable_audio is False
        assert TypeConfigurationWindow._instance is None

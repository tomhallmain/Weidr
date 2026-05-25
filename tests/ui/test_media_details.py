"""UI smoke tests for MediaDetails (metadata labels, refresh)."""

import os

import pytest
from PySide6.QtWidgets import QApplication

from ui.image.media_details import MediaDetails


def _close_media_details(app_win) -> None:
    details = app_win.app_actions.media_details_window()
    if details is not None:
        try:
            details.close_windows()
            details.close()
            details.deleteLater()
        except RuntimeError:
            pass
        app_win.app_actions.set_media_details_window(None)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture
def mock_media_metadata(monkeypatch):
    """Stable prompt/model extraction for deterministic label text."""
    monkeypatch.setattr(
        "image.image_data_extractor.image_data_extractor.get_image_prompts_and_models",
        lambda _path: (
            "test positive prompt",
            "test negative prompt",
            ["model-alpha"],
            ["lora-beta"],
            False,
        ),
    )
    monkeypatch.setattr(
        "ui.image.media_details.MediaDetails.get_related_image_text",
        lambda _self: "related: (none)",
    )


@pytest.fixture
def media_details_window(window_with_dir, qtbot, bypass_password, mock_media_metadata):
    """Open MediaDetails for the first PNG in the loaded directory."""
    app_win, media_dir = window_with_dir
    media_path = app_win.file_browser.get_files()[0]

    app_win.window_launcher.open_media_details(media_path=media_path)
    qtbot.waitUntil(
        lambda: app_win.app_actions.media_details_window() is not None,
        timeout=5000,
    )
    details = app_win.app_actions.media_details_window()
    qtbot.addWidget(details)
    qtbot.waitExposed(details, timeout=3000)

    yield details, app_win, media_path, media_dir

    _close_media_details(app_win)


@pytest.fixture(autouse=True)
def _cleanup_media_details_singleton():
    yield
    MediaDetails.temp_media_canvas = None


class TestMediaDetailsWindow:
    def test_open_populates_path_dimensions_and_prompt_labels(
        self, media_details_window
    ):
        details, _app_win, media_path, _media_dir = media_details_window

        assert details._lbl_path.text() == media_path
        assert details._lbl_dims.text() == "10x10"
        assert details._lbl_mode.text() == "RGB"
        assert "test positive prompt" in details._lbl_positive.text()
        assert "model-alpha" in details._lbl_models.text()
        assert "lora-beta" in details._lbl_loras.text()
        assert "related:" in details._lbl_related_image.text()

    def test_update_media_details_refreshes_labels(
        self, media_details_window, qtbot
    ):
        details, app_win, _first_path, media_dir = media_details_window
        second_path = os.path.join(media_dir, "img02.png")
        index_text = app_win.file_browser.get_index_details()

        details.update_media_details(second_path, index_text)
        qtbot.waitUntil(
            lambda: details._lbl_path.text() == second_path,
            timeout=2000,
        )

        assert details._lbl_dims.text() == "10x10"
        assert details._lbl_path.text() == second_path

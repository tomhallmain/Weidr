"""SearchController image generation worker (mocked SD runner)."""

import os

from utils.constants import ImageGenerationType


class TestSearchControllerImageGen:
    def test_run_image_generation_invokes_sd_runner_in_worker(
        self, window_with_dir, qtbot, bypass_password, monkeypatch
    ):
        win, media_dir = window_with_dir
        media_path = win.file_browser.get_files()[0]
        run_calls = []

        class FakeSDRunner:
            def run(self, gen_type, path, **kwargs):
                run_calls.append((gen_type, path, kwargs))

        monkeypatch.setattr(
            "extensions.sd_runner_client.SDRunnerClient",
            FakeSDRunner,
        )
        monkeypatch.setattr(
            "ui.image.media_details.MediaDetails.get_image_specific_generation_mode",
            lambda: ImageGenerationType.LAST_SETTINGS,
        )
        monkeypatch.setattr(win.search_ctrl, "_get_media_path", lambda: media_path)

        win.search_ctrl.run_image_generation()

        qtbot.waitUntil(lambda: len(run_calls) == 1, timeout=5000)
        gen_type, path, kwargs = run_calls[0]
        assert path == media_path
        assert gen_type == ImageGenerationType.LAST_SETTINGS
        assert kwargs.get("append") is False
        qtbot.waitUntil(lambda: win.search_ctrl._img_gen_workers == [], timeout=5000)

    def test_run_image_generation_no_op_without_media_path(
        self, window_with_dir, bypass_password, monkeypatch
    ):
        win, _ = window_with_dir
        monkeypatch.setattr(win.search_ctrl, "_get_media_path", lambda: None)

        win.search_ctrl.run_image_generation()

        assert win.search_ctrl._img_gen_workers == []

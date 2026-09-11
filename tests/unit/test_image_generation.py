"""files.image_generation.request_image_generation: the Qt-free core both
sessions' run_image_generation actions call."""

from unittest.mock import patch

import pytest

from files import image_generation
from utils.constants import ImageGenerationType


def _record_cache_clears(monkeypatch):
    cleared = []
    monkeypatch.setattr(image_generation, "clear_generate_gate_cache",
                        lambda d: cleared.append(("gate", d)))
    monkeypatch.setattr(image_generation, "clear_base_stem_dir_cache",
                        lambda d: cleared.append(("stem", d)))
    return cleared


def test_sends_the_request_then_clears_the_source_directory_caches(monkeypatch):
    cleared = _record_cache_clears(monkeypatch)

    with patch("extensions.sd_runner_client.SDRunnerClient") as client_cls:
        image_generation.request_image_generation(
            ImageGenerationType.CONTROL_NET, "/media/a.png", append=True,
            prompt_overrides=("pos", "neg"), edit_suffix="_x", target_dir="/out",
        )

    client_cls.return_value.run.assert_called_once_with(
        ImageGenerationType.CONTROL_NET, "/media/a.png", append=True,
        positive_prompt="pos", negative_prompt="neg", edit_suffix="_x", target_dir="/out",
    )
    assert cleared == [("gate", "/media"), ("stem", "/media")]


def test_without_prompt_overrides_no_prompts_are_sent(monkeypatch):
    _record_cache_clears(monkeypatch)

    with patch("extensions.sd_runner_client.SDRunnerClient") as client_cls:
        image_generation.request_image_generation(ImageGenerationType.CONTROL_NET, "/media/a.png")

    kwargs = client_cls.return_value.run.call_args.kwargs
    assert kwargs["positive_prompt"] is None
    assert kwargs["negative_prompt"] is None
    assert kwargs["append"] is False


def test_caches_are_left_alone_when_the_request_fails(monkeypatch):
    cleared = _record_cache_clears(monkeypatch)

    with patch("extensions.sd_runner_client.SDRunnerClient") as client_cls:
        client_cls.return_value.run.side_effect = RuntimeError("sd-runner unreachable")
        with pytest.raises(RuntimeError):
            image_generation.request_image_generation(ImageGenerationType.CONTROL_NET, "/media/a.png")

    assert cleared == []

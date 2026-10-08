"""
ImageClassifierWrapper's handling of non-raster media (SVG, PDF, video, ...):
predict_image and discard_cached_prediction work on the FrameCache render of
the media, so every caller (Test button, pipeline conditions, classifier
actions, compare filters) classifies the same image. FrameCache rendering is
stubbed; the backend is a recording stand-in.
"""
from __future__ import annotations

import numpy as np
import pytest

from image.frame_cache import FrameCache
from image.image_classifier import ImageClassifierWrapper
from image.image_classifier_model_config import ImageClassifierModelConfig


class _RecordingBackend:
    is_loaded = True

    def __init__(self):
        self.paths = []

    def predict_image(self, image_path):
        self.paths.append(image_path)
        return np.array([[0.2, 0.8]])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    svg = tmp_path / "drawing.svg"
    svg.write_text("<svg/>")
    render = tmp_path / "drawing_render.png"
    render.write_bytes(b"png")
    renders = {str(svg): str(render)}
    monkeypatch.setattr(FrameCache, "get_image_path", classmethod(lambda cls, p: renders.get(p, p)))

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"stub")
    wrapper = ImageClassifierWrapper(ImageClassifierModelConfig(
        model_name="m", model_location=str(model_path), model_categories=["a", "b"]))
    backend = _RecordingBackend()
    wrapper.classifier = backend
    wrapper.can_run = True
    return wrapper, backend, str(svg), str(render)


def test_backend_reads_the_rendered_image(setup):
    wrapper, backend, svg, render = setup
    assert wrapper.classify_image(svg) == "b"
    assert backend.paths == [render]


def test_media_path_and_render_share_one_cached_prediction(setup):
    wrapper, backend, svg, render = setup
    wrapper.predict_image(svg)
    wrapper.predict_image(render)
    assert backend.paths == [render]


def test_discard_by_media_path_forgets_the_render_prediction(setup):
    wrapper, backend, svg, render = setup
    wrapper.predict_image(svg)
    wrapper.discard_cached_prediction(svg)
    wrapper.predict_image(svg)
    assert backend.paths == [render, render]


def test_raster_paths_pass_through(setup, tmp_path):
    wrapper, backend, _svg, _render = setup
    png = tmp_path / "photo.png"
    png.write_bytes(b"png")
    wrapper.predict_image(str(png))
    assert backend.paths == [str(png)]


def test_input_image_path_uses_frame_cache(setup):
    _wrapper, _backend, svg, render = setup
    assert ImageClassifierWrapper.input_image_path(svg) == render

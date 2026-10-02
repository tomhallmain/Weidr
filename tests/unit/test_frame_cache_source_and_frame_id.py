"""FrameCache.source_and_frame_id: tracing a render back to its source."""

import os

from image.frame_cache import FrameCache, _stable_media_path_hash


def test_render_names_resolve_to_source_and_frame(tmp_path):
    source = os.path.normpath(str(tmp_path / "clip.mp4"))
    h = _stable_media_path_hash(source)
    for suffix in ("_first.jpg", "_sample_120.jpg", "_page_0.jpg", "_sample_page_3.jpg", ".png"):
        assert FrameCache.source_and_frame_id(f"/any/tmp/dir/{h}{suffix}") == (source, suffix)


def test_unknown_hash_and_foreign_names_return_none(tmp_path):
    assert FrameCache.source_and_frame_id("/tmp/" + "0" * 64 + "_first.jpg") is None
    assert FrameCache.source_and_frame_id(str(tmp_path / "photo.jpg")) is None
    h = _stable_media_path_hash(str(tmp_path / "clip.mp4"))
    assert FrameCache.source_and_frame_id(f"/tmp/{h}") is None
    assert FrameCache.source_and_frame_id(f"/tmp/{h}x.jpg") is None

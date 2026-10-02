"""
Tests for image/classifier_prediction_cache.py, the epoch-free
FileMtimeInvalidationCache it builds on (lib/file_invalidation_cache.py), and
the image classifier wrapper's use of it.

Encryption is replaced by plain file I/O so no key material is touched.
"""

import json
import os
import time

import numpy as np
import pytest

import lib.file_invalidation_cache as fic
from image import classifier_prediction_cache as cpc_module
from image.classifier_prediction_cache import ClassifierPredictionCache, model_signature
from lib.file_invalidation_cache import FileKeyedInvalidationCache, FileMtimeInvalidationCache


@pytest.fixture
def files(tmp_path):
    media = tmp_path / "a.png"
    media.write_bytes(b"img")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    return str(media), str(model)


@pytest.fixture
def plain_encryption(monkeypatch):
    """Store/load the cache file as plain bytes."""
    import utils.encryptor as enc

    def _encrypt(data, service_name, app_identifier, output_path, *a, **kw):
        with open(output_path, "wb") as f:
            f.write(data)

    def _decrypt(path, service_name, app_identifier):
        with open(path, "rb") as f:
            return f.read()

    monkeypatch.setattr(enc, "encrypt_data_to_file", _encrypt)
    monkeypatch.setattr(enc, "decrypt_data_from_file", _decrypt)


def _touch_later(path):
    later = time.time() + 10
    os.utime(path, (later, later))


SCORES = {"photo": 0.9, "drawing": 0.1}


# ===========================================================================
# FileMtimeInvalidationCache
# ===========================================================================

class TestFileMtimeInvalidationCache:
    @pytest.fixture(autouse=True)
    def _isolate_module_state(self, monkeypatch):
        monkeypatch.setattr(fic, "_policy_epoch", fic._policy_epoch)
        monkeypatch.setattr(fic, "_signature_memo", fic._signature_memo)
        monkeypatch.setattr(fic, "_file_buckets", {})

    def test_survives_policy_invalidation(self, files):
        row = FileMtimeInvalidationCache()
        row.set(files, "v", "sig")
        fic.invalidate_policy_caches()
        assert row.try_get(files) == (True, "v")

    def test_base_class_still_invalidated_by_policy(self, files):
        row = FileKeyedInvalidationCache()
        row.set(files, "v", "sig")
        fic.invalidate_policy_caches()
        assert row.try_get(files) == (False, None)

    def test_invalidated_by_file_change(self, files):
        row = FileMtimeInvalidationCache()
        row.set(files, "v", "sig")
        _touch_later(files[1])
        assert row.try_get(files) == (False, None)

    def test_snapshot_round_trip_ignores_stored_epoch(self, files):
        row = FileMtimeInvalidationCache()
        row.set(files, "v", "sig")
        snap = row.snapshot_for_persistence()
        fic.invalidate_policy_caches()
        restored = FileMtimeInvalidationCache()
        restored.load_from_snapshot(snap["path_mtimes"], "v", snap["signature"], epoch_at_set=7,
                                    cached_at_unix=snap["cached_at_unix"])
        assert restored.try_get(files) == (True, "v")


# ===========================================================================
# ClassifierPredictionCache — access
# ===========================================================================

class TestPredictionCacheAccess:
    def test_put_then_get(self, files):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        assert cache.get("image:m", "sig", model, media) == SCORES

    def test_get_returns_copy(self, files):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        cache.get("image:m", "sig", model, media)["photo"] = 0.0
        assert cache.get("image:m", "sig", model, media) == SCORES

    def test_other_model_misses(self, files):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        assert cache.get("image:other", "sig", model, media) is None

    def test_signature_change_misses(self, files):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        assert cache.get("image:m", "sig2", model, media) is None

    def test_media_change_misses(self, files):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        _touch_later(media)
        assert cache.get("image:m", "sig", model, media) is None

    def test_model_file_change_misses(self, files):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        _touch_later(model)
        assert cache.get("image:m", "sig", model, media) is None

    def test_hub_repo_id_tracks_media_only(self, files):
        media, _model = files
        cache = ClassifierPredictionCache()
        cache.put("audio:m", "sig", "org/model", media, SCORES)
        assert cache.get("audio:m", "sig", "org/model", media) == SCORES

    def test_missing_media_is_not_cached(self, files, tmp_path):
        _media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, str(tmp_path / "gone.png"), SCORES)
        assert cache.entry_count() == 0

    def test_unknown_render_not_cached(self, files, monkeypatch):
        """A FrameCache render whose source is unknown has nothing to key on."""
        import utils.media_utils as mu
        media, model = files
        monkeypatch.setattr(mu, "is_frame_cache_path", lambda p: True)
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        assert cache.entry_count() == 0
        assert cache.get("image:m", "sig", model, media) is None

    def test_render_keyed_on_source_and_frame(self, tmp_path, monkeypatch):
        """Scores for a rendered frame are found again under a new temp
        directory (as after a restart), are separate per frame, and are
        invalidated by a change to the source file."""
        import utils.media_utils as mu
        from image.frame_cache import _stable_media_path_hash
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video")
        h = _stable_media_path_hash(str(video))
        session_a, session_b = tmp_path / "frames_a", tmp_path / "frames_b"
        monkeypatch.setattr(mu, "is_frame_cache_path", lambda p: "frames_" in p)

        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", "org/model", str(session_a / f"{h}_sample_120.jpg"), SCORES)

        assert cache.get("image:m", "sig", "org/model", str(session_b / f"{h}_sample_120.jpg")) == SCORES
        assert cache.get("image:m", "sig", "org/model", str(session_b / f"{h}_sample_240.jpg")) is None
        assert cache.get("image:m", "sig", "org/model", str(video)) is None

        _touch_later(str(video))
        assert cache.get("image:m", "sig", "org/model", str(session_b / f"{h}_sample_120.jpg")) is None

    def test_render_of_temp_source_not_cached(self, tmp_path, monkeypatch):
        """HTML is rendered through a temp PDF; frames of that PDF have no
        lasting source."""
        import utils.media_utils as mu
        from image.frame_cache import _stable_media_path_hash
        temp_pdf = tmp_path / "frames_tmp" / "x_from_html.pdf"
        temp_pdf.parent.mkdir()
        temp_pdf.write_bytes(b"pdf")
        h = _stable_media_path_hash(str(temp_pdf))
        monkeypatch.setattr(mu, "is_frame_cache_path", lambda p: "frames_" in p)
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", "org/model", str(tmp_path / "frames_tmp" / f"{h}_first.jpg"), SCORES)
        assert cache.entry_count() == 0

    def test_discard_and_clear(self, files):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        cache.discard("image:m", media)
        assert cache.get("image:m", "sig", model, media) is None
        cache.put("image:m", "sig", model, media, SCORES)
        cache.clear()
        assert cache.entry_count() == 0

    def test_model_signature_stable_and_sensitive(self):
        a = model_signature({"location": "/m", "categories": ["x", "y"]})
        assert a == model_signature({"categories": ["x", "y"], "location": "/m"})
        assert a != model_signature({"location": "/m", "categories": ["y", "x"]})


# ===========================================================================
# ClassifierPredictionCache — persistence
# ===========================================================================

class TestPredictionCachePersistence:
    def test_store_and_load_round_trip(self, files, plain_encryption):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.load()
        cache.put("image:m", "sig", model, media, SCORES)
        cache.store()
        assert os.path.exists(cpc_module._cache_file_path())

        fresh = ClassifierPredictionCache()
        fresh.load()
        assert fresh.get("image:m", "sig", model, media) == SCORES

    def test_file_lives_in_cache_dir(self):
        assert os.path.dirname(cpc_module._cache_file_path()) == os.environ["WEIDR_CACHE_DIR"]

    def test_store_before_load_writes_nothing(self, files, plain_encryption):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        cache.store()
        assert not os.path.exists(cpc_module._cache_file_path())

    def test_store_without_changes_writes_nothing(self, plain_encryption):
        cache = ClassifierPredictionCache()
        cache.load()
        cache.store()
        assert not os.path.exists(cpc_module._cache_file_path())

    def test_failed_encryption_keeps_changes_and_writes_no_plaintext(self, files, monkeypatch):
        import utils.encryptor as enc

        def _fail(*a, **kw):
            raise RuntimeError("no keys")

        monkeypatch.setattr(enc, "encrypt_data_to_file", _fail)
        media, model = files
        cache = ClassifierPredictionCache()
        cache.load()
        cache.put("image:m", "sig", model, media, SCORES)
        cache.store()
        cache_dir = os.path.dirname(cpc_module._cache_file_path())
        leftovers = [n for n in os.listdir(cache_dir) if "classifier_prediction_cache" in n]
        assert leftovers == []
        assert cache._dirty is True

    def test_stale_rows_dropped_on_load(self, files, plain_encryption):
        media, model = files
        cache = ClassifierPredictionCache()
        cache.load()
        cache.put("image:m", "sig", model, media, SCORES)
        cache.store()
        path = cpc_module._cache_file_path()
        with open(path, "rb") as f:
            payload = json.loads(f.read())
        payload["models"]["image:m"][0]["cached_at_unix"] = 0.0
        with open(path, "wb") as f:
            f.write(json.dumps(payload).encode("utf-8"))

        fresh = ClassifierPredictionCache()
        fresh.load()
        assert fresh.entry_count() == 0

    def test_unknown_format_ignored(self, plain_encryption):
        with open(cpc_module._cache_file_path(), "wb") as f:
            f.write(json.dumps({"version": 999, "models": {}}).encode("utf-8"))
        cache = ClassifierPredictionCache()
        cache.load()
        assert cache.entry_count() == 0

    def test_store_trims_oldest_rows(self, tmp_path, plain_encryption, monkeypatch):
        monkeypatch.setattr(ClassifierPredictionCache, "MAX_ENTRIES", 2)
        cache = ClassifierPredictionCache()
        cache.load()
        paths = []
        for i in range(3):
            p = tmp_path / f"m{i}.png"
            p.write_bytes(b"x")
            paths.append(str(p))
            cache.put("image:m", "sig", "org/model", str(p), SCORES)
            time.sleep(0.01)
        cache.store()
        assert cache.entry_count() == 2
        assert cache.get("image:m", "sig", "org/model", paths[0]) is None
        assert cache.get("image:m", "sig", "org/model", paths[2]) == SCORES

    def test_load_keeps_rows_added_before_load(self, files, plain_encryption):
        media, model = files
        writer = ClassifierPredictionCache()
        writer.load()
        writer.put("image:m", "sig", model, media, {"photo": 0.1})
        writer.store()

        cache = ClassifierPredictionCache()
        cache.put("image:m", "sig", model, media, SCORES)
        cache.load()
        assert cache.get("image:m", "sig", model, media) == SCORES


# ===========================================================================
# ImageClassifierWrapper integration
# ===========================================================================

class _CountingBackend:
    def __init__(self):
        self.calls = 0

    def predict_image(self, path):
        self.calls += 1
        return np.array([[0.8, 0.2]])


def _wrapper(model_location):
    from image.image_classifier import BackendType, ImageClassifierWrapper
    w = object.__new__(ImageClassifierWrapper)
    w.model_name = "m"
    w.model_location = model_location
    w.model_categories = ["photo", "drawing"]
    w.backend = BackendType.ONNX
    w.model_kwargs = {}
    w.input_shape = None
    w.predictions_cache = {}
    w._prediction_signature = None
    w.can_run = True
    w.classifier = _CountingBackend()
    return w


class TestImageWrapperUsesPersistentCache:
    def test_new_session_reuses_persisted_scores(self, files):
        media, model = files
        first = _wrapper(model)
        assert first.predict_image(media) == {"photo": 0.8, "drawing": 0.2}

        second = _wrapper(model)  # empty session cache, as after a restart
        assert second.predict_image(media) == {"photo": pytest.approx(0.8), "drawing": pytest.approx(0.2)}
        assert second.classifier.calls == 0

    def test_changed_categories_recompute(self, files):
        media, model = files
        _wrapper(model).predict_image(media)
        other = _wrapper(model)
        other.model_categories = ["cat", "dog"]
        other.predict_image(media)
        assert other.classifier.calls == 1

    def test_discard_cached_prediction_forces_recompute(self, files):
        media, model = files
        w = _wrapper(model)
        w.predict_image(media)
        w.discard_cached_prediction(media)
        w.predict_image(media)
        assert w.classifier.calls == 2
        assert _wrapper(model).predict_image(media) is not None

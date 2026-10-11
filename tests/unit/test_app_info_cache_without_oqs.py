"""AppInfoCache when the stored keys are quantum (OQS) keys and the process
has no OQS: it starts, never touches the encrypted cache, and saves to a
separate plaintext file."""

import json
import os

import pytest

import utils.app_info_cache as aic


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    # Its own folder: the root conftest's isolated_singletons fills tmp_path.
    cache_dir = tmp_path / "no_oqs_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("WEIDR_CACHE_DIR", str(cache_dir))
    # A key store must exist for the check to run; its type is patched, so
    # the file is never read.
    store = aic.key_store_path(aic.AppInfo.SERVICE_NAME, aic.AppInfo.APP_IDENTIFIER)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    with open(store, "w", encoding="utf-8") as f:
        f.write("{}")
    monkeypatch.setattr(aic, "stored_keys_need_unavailable_oqs", lambda service, app: True)

    def fail(*args, **kwargs):
        raise AssertionError("must not encrypt or decrypt without OQS")

    monkeypatch.setattr(aic, "encrypt_data_to_file", fail)
    monkeypatch.setattr(aic, "decrypt_data_from_file", fail)
    return cache_dir


def test_starts_without_reading_the_encrypted_cache(cache_dir):
    encrypted = cache_dir / "app_info_cache.enc"
    encrypted.write_bytes(b"encrypted with OQS keys")
    cache = aic.AppInfoCache()
    assert cache.get_meta("anything") is None
    assert encrypted.read_bytes() == b"encrypted with OQS keys"
    assert not (cache_dir / "app_info_cache.no_oqs.json").exists()
    assert not (cache_dir / "app_info_cache.json").exists()
    assert not (cache_dir / "app_info_cache.enc.bak").exists()


def test_saves_to_the_separate_file_and_reloads_it(cache_dir):
    cache = aic.AppInfoCache()
    cache.set_meta("key", "value")
    assert cache.store() is False
    saved = cache_dir / "app_info_cache.no_oqs.json"
    assert json.loads(saved.read_text(encoding="utf-8"))[aic.AppInfoCache.META_INFO_KEY]["key"] == "value"
    assert not (cache_dir / "app_info_cache.json").exists()
    assert aic.AppInfoCache().get_meta("key") == "value"

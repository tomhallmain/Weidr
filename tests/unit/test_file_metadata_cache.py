"""
Persistent file metadata cache (files/file_metadata_cache.py) and the
SortableFile write-through paths that fill it.

The cache trades freshness for I/O, so what is pinned here is the shape of that
trade: a hit serves without touching the disk, a miss populates, a failed read
is never written, and the load/store round trip neither loses entries nor
persists over a cache that was never loaded.

Encryption is replaced by plain file I/O so no key material is touched.
"""

from datetime import datetime
import json
import os
import threading

import pytest
from PIL import Image

import files.file_metadata_cache as fmc_mod
from files.file_metadata_cache import CACHE_KEY, FileMetadataCache
from files.sortable_file import SortableFile
from utils.app_info_cache import app_info_cache
from utils.config import config

# Distinct from any real file's stat, and far enough past the epoch that the
# local-time conversion stays in range on every platform.
PLANTED_CTIME = 1_700_000_000.0
PLANTED_MTIME = 1_700_000_123.0


@pytest.fixture(autouse=True)
def enable_cache(monkeypatch):
    monkeypatch.setattr(config, "enable_file_metadata_cache", True)


@pytest.fixture(autouse=True)
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


def _stored_entries() -> dict:
    with open(fmc_mod._cache_file_path(), "rb") as f:
        payload = json.loads(f.read().decode("utf-8"))
    assert payload["version"] == fmc_mod.FORMAT_VERSION
    return payload["entries"]


@pytest.fixture
def cache(monkeypatch):
    """A fresh cache instance, also installed as the module singleton so
    SortableFile writes into it."""
    instance = FileMetadataCache()
    instance.load()
    monkeypatch.setattr(fmc_mod, "file_metadata_cache", instance)
    import files.sortable_file as sf_mod
    monkeypatch.setattr(sf_mod, "file_metadata_cache", instance)
    return instance


@pytest.fixture
def png(tmp_path):
    path = str(tmp_path / "pic.png")
    Image.new("RGB", (7, 3), (10, 20, 30)).save(path, format="PNG")
    return path


class TestDisabledByDefault:
    def test_the_flag_is_off_in_the_shipped_config(self):
        """The cache serves stale metadata by design, so it is opt-in."""
        from utils.config import Config
        assert Config().enable_file_metadata_cache is False

    def test_nothing_is_recorded_while_disabled(self, monkeypatch, png):
        monkeypatch.setattr(config, "enable_file_metadata_cache", False)
        instance = FileMetadataCache()
        import files.sortable_file as sf_mod
        monkeypatch.setattr(sf_mod, "file_metadata_cache", instance)

        SortableFile(png)

        assert instance.entry_count() == 0
        assert instance.get(png) is None


class TestStatWriteThrough:
    def test_a_first_build_records_the_stat(self, cache, png):
        SortableFile(png)

        entry = cache.get(png)
        stat_obj = os.stat(png)
        assert entry["size"] == stat_obj.st_size
        assert entry["mtime"] == pytest.approx(stat_obj.st_mtime)
        assert entry["ctime"] == pytest.approx(stat_obj.st_ctime)

    def test_a_second_build_reads_the_cache_instead_of_the_disk(self, cache, png):
        SortableFile(png)
        cache.update_stat(png, ctime=PLANTED_CTIME, mtime=PLANTED_MTIME, size=4242)

        sortable = SortableFile(png)

        # The planted values, not the real ones -- proof the stat was skipped.
        assert sortable.size == 4242
        assert sortable.mtime == datetime.fromtimestamp(PLANTED_MTIME)
        assert sortable.ctime == datetime.fromtimestamp(PLANTED_CTIME)

    def test_a_dir_entry_is_not_consulted_on_a_hit(self, cache, png):
        """The cache short-circuits ahead of the scan's DirEntry, so a hit
        costs nothing even where DirEntry.stat() would have been free."""
        class _ExplodingEntry:
            path = png

            def stat(self):
                raise AssertionError("stat() called despite a cache hit")

        SortableFile(png)

        sortable = SortableFile(png, dir_entry=_ExplodingEntry())
        assert sortable.size == os.stat(png).st_size

    def test_a_failed_stat_is_not_recorded(self, cache, tmp_path):
        """The zeroed fallback would otherwise be served as real metadata for
        every later session."""
        missing = str(tmp_path / "gone.png")

        sortable = SortableFile(missing)

        assert sortable.size == 0
        assert cache.get(missing) is None


class TestMalformedEntries:
    """The cache blob persists across sessions, so a bad entry in it must cost
    one file's metadata rather than the whole directory load."""

    def test_an_entry_with_no_stat_falls_back_to_the_disk(self, cache, png):
        cache.update_related_image_path(png, "")

        sortable = SortableFile(png)

        assert sortable.size == os.stat(png).st_size

    @pytest.mark.parametrize(
        "bad", [{"size": 1, "ctime": "x", "mtime": 2.0}, {"size": 1, "ctime": 1e30, "mtime": 1e30}]
    )
    def test_an_unusable_timestamp_falls_back_to_the_disk(self, cache, png, bad):
        with cache._lock:
            cache._data[png] = dict(bad)

        sortable = SortableFile(png)

        assert sortable.size == os.stat(png).st_size
        assert sortable.mtime.timestamp() == pytest.approx(os.stat(png).st_mtime)


class TestDimensionWriteThrough:
    def test_dimensions_are_recorded_once_read(self, cache, png):
        sortable = SortableFile(png)

        assert sortable.get_image_dimensions() == (7, 3)

        entry = cache.get(png)
        assert entry["image_width"] == 7
        assert entry["image_height"] == 3

    def test_a_later_build_takes_dimensions_from_the_cache(self, cache, png):
        SortableFile(png).get_image_dimensions()
        cache.update_dimensions(png, 111, 222)

        assert SortableFile(png).get_image_dimensions() == (111, 222)

    def test_an_unreadable_file_is_not_recorded_as_zero_sized(self, cache, tmp_path):
        """Caching (0, 0) would sort the file as zero-pixel for good; leaving
        it out lets a later session retry."""
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a png")

        sortable = SortableFile(str(broken))
        assert sortable.get_image_dimensions() == (0, 0)

        entry = cache.get(str(broken))
        assert entry is not None
        assert "image_width" not in entry


class TestRelatedImageWriteThrough:
    def test_the_lookup_result_is_recorded(self, cache, png, monkeypatch):
        import files.sortable_file as sf_mod
        monkeypatch.setattr(
            sf_mod.image_data_extractor, "get_related_image_path", lambda p: None
        )

        sortable = SortableFile(png)
        sortable.set_related_image_path()

        assert sortable.related_image_path == ""
        assert cache.get(png)["related_image_path"] == ""

    def test_a_later_build_starts_from_the_cached_result(self, cache, png):
        cache.update_stat(png, ctime=PLANTED_CTIME, mtime=PLANTED_MTIME, size=3)
        cache.update_related_image_path(png, "/elsewhere/origin.png")

        sortable = SortableFile(png)

        assert sortable.related_image_path == "/elsewhere/origin.png"
        assert sortable.get_related_image_or_self() == "origin.png"


class TestPersistence:
    def test_a_round_trip_preserves_entries(self, cache, png):
        SortableFile(png)
        cache.store()

        reloaded = FileMetadataCache()
        reloaded.load()

        assert reloaded.get(png)["size"] == os.stat(png).st_size

    def test_the_file_lives_in_the_cache_dir(self):
        assert os.path.dirname(fmc_mod._cache_file_path()) == os.environ["WEIDR_CACHE_DIR"]

    def test_storing_leaves_app_info_cache_alone(self, cache, png):
        """The entries go to their own file; the main cache neither holds them
        nor needs a store because of them."""
        app_info_cache._has_changes = False
        SortableFile(png)

        cache.store()

        assert app_info_cache.has_changes is False
        assert app_info_cache.get_meta(CACHE_KEY) is None
        assert _stored_entries()[png]["size"] == os.stat(png).st_size

    def test_a_copy_left_in_app_info_cache_is_dropped_on_load(self, png):
        app_info_cache.set_meta(CACHE_KEY, {png: {"size": 1, "ctime": 0.0, "mtime": 0.0}})

        instance = FileMetadataCache()
        instance.load()

        assert app_info_cache.get_meta(CACHE_KEY) is None
        assert instance.get(png) is None

    def test_a_second_load_keeps_what_is_already_in_memory(self, cache, png):
        """Every window calls load(); a later one must not drop what an
        earlier one collected."""
        SortableFile(png)

        cache.load()

        assert cache.get(png) is not None

    def test_entries_recorded_before_load_win_over_the_file(self, cache, png):
        cache.update_stat(png, ctime=0.0, mtime=0.0, size=1)
        cache.store()
        fresh = FileMetadataCache()
        fresh.update_stat(png, ctime=0.0, mtime=0.0, size=2)

        fresh.load()

        assert fresh.get(png)["size"] == 2

    def test_an_unloaded_cache_refuses_to_store(self, cache, png):
        """A partial dict written over the full one would lose entries."""
        cache.update_stat(png, ctime=0.0, mtime=0.0, size=1)
        cache.store()
        never_loaded = FileMetadataCache()
        never_loaded.update_stat("/other.png", ctime=0.0, mtime=0.0, size=2)

        never_loaded.store()

        assert _stored_entries() == {png: {"ctime": 0.0, "mtime": 0.0, "size": 1}}

    def test_a_clean_cache_does_not_write(self, cache):
        cache.store()

        assert not os.path.exists(fmc_mod._cache_file_path())

    def test_a_failed_encryption_keeps_changes_and_writes_no_plaintext(self, cache, png, monkeypatch):
        import utils.encryptor as enc

        def _fail(*a, **kw):
            raise RuntimeError("no keys")

        monkeypatch.setattr(enc, "encrypt_data_to_file", _fail)
        SortableFile(png)

        cache.store()

        cache_dir = os.path.dirname(fmc_mod._cache_file_path())
        assert [n for n in os.listdir(cache_dir) if "file_metadata_cache" in n] == []
        assert cache._dirty is True

    def test_an_unreadable_file_starts_empty(self, png):
        path = fmc_mod._cache_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"not json")

        instance = FileMetadataCache()
        instance.load()

        assert instance.entry_count() == 0
        instance.update_stat(png, ctime=0.0, mtime=0.0, size=3)
        instance.store()
        assert _stored_entries()[png]["size"] == 3

    def test_an_unknown_format_is_ignored(self):
        path = fmc_mod._cache_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(json.dumps({"version": 999, "entries": {"/a.png": {"size": 1}}}).encode("utf-8"))

        instance = FileMetadataCache()
        instance.load()

        assert instance.entry_count() == 0

    def test_the_entry_ceiling_is_enforced_on_store(self, cache, monkeypatch):
        monkeypatch.setattr(cache, "MAX_ENTRIES", 10)
        for i in range(25):
            cache.update_stat(f"/f{i:02d}.png", ctime=0.0, mtime=0.0, size=i)

        cache.store()

        stored = _stored_entries()
        assert len(stored) == 10
        # The tail is kept: insertion order stands in for recency.
        assert "/f24.png" in stored
        assert "/f00.png" not in stored

    def test_clear_empties_the_cache(self, cache, png):
        SortableFile(png)

        cache.clear()

        assert cache.entry_count() == 0
        assert cache.get(png) is None


class TestConcurrency:
    def test_a_store_during_an_active_scan_does_not_raise(self, cache, monkeypatch):
        """SortableFile is built on the incremental-load thread, so new entries
        arrive while the GUI thread is snapshotting the dict for persistence.
        Unlocked, the snapshot raises "dictionary changed size during
        iteration" partway through a scan.

        The file write is stubbed out so the test measures the snapshot alone.
        """
        monkeypatch.setattr(fmc_mod, "_write_encrypted", lambda data, destination: None)

        stop = threading.Event()
        errors = []

        def _writer():
            i = 0
            while not stop.is_set() and i < 20_000:
                cache.update_stat(f"/scan/{i}.png", ctime=0.0, mtime=0.0, size=i)
                i += 1

        writer = threading.Thread(target=_writer, daemon=True)
        writer.start()
        try:
            for _ in range(200):
                try:
                    cache.store()
                except Exception as e:  # pragma: no cover - the failure this guards
                    errors.append(e)
                    break
        finally:
            stop.set()
            writer.join(timeout=5)

        assert errors == []
        assert cache.entry_count() > 0

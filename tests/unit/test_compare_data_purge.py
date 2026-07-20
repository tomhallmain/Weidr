"""Unit tests for CompareData stale-entry purge (media data cache churn cleanup).

Covers docs/media-data-cache-churn-cleanup.md: existence-based purge,
inflation trigger with floor, extreme-scale bail-out with external-drive
penalty factor, repeat-scan guard, and persistence of the purged dict.
"""

import os
import pickle

import pytest

from compare.compare_data import CompareData
from tests.helpers import isolated_app_info_cache
from utils.utils import Utils


def _touch(path) -> str:
    with open(path, "wb") as f:
        f.write(b"x")
    return str(path)


def _make_data(tmp_path, existing=0, missing=0, subdir_existing=0):
    """Build a CompareData with real, missing, and subdirectory-file keys."""
    cd = CompareData(base_dir=str(tmp_path), data_filename="test_data.pkl")
    for i in range(existing):
        cd.file_data_dict[_touch(tmp_path / f"real_{i}.png")] = [float(i)]
    for i in range(missing):
        cd.file_data_dict[str(tmp_path / f"gone_{i}.png")] = [float(i)]
    if subdir_existing:
        sub = tmp_path / "sub"
        sub.mkdir(exist_ok=True)
        for i in range(subdir_existing):
            cd.file_data_dict[_touch(sub / f"nested_{i}.png")] = [float(i)]
    return cd


@pytest.fixture(autouse=True)
def _zero_floor(monkeypatch):
    """Most tests want the trigger governed by the ratio alone."""
    monkeypatch.setattr(CompareData, "PURGE_FLOOR", 0)


class TestPurgeStaleEntries:
    def test_removes_only_missing_files(self, tmp_path):
        cd = _make_data(tmp_path, existing=2, missing=3)
        assert cd.purge_stale_entries() == 3
        assert len(cd.file_data_dict) == 2
        assert all(os.path.isfile(k) for k in cd.file_data_dict)

    def test_subdir_keys_for_existing_files_survive(self, tmp_path):
        """Recursion concern: entries under subdirectories stay valid as long
        as the files exist, regardless of any recursive-option change."""
        cd = _make_data(tmp_path, existing=1, missing=2, subdir_existing=2)
        cd.purge_stale_entries()
        nested = [k for k in cd.file_data_dict if os.sep + "sub" + os.sep in k]
        assert len(nested) == 2


class TestPurgeTrigger:
    def test_skips_below_floor(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CompareData, "PURGE_FLOOR", 100)
        cd = _make_data(tmp_path, missing=10)
        cd._maybe_purge_stale_entries()
        assert len(cd.file_data_dict) == 10  # untouched

    def test_skips_when_ratio_not_inflated(self, tmp_path):
        cd = _make_data(tmp_path, existing=5, missing=5)
        cd.files_found = ["f"] * 6  # 10 <= 2 * 6
        cd._maybe_purge_stale_entries()
        assert len(cd.file_data_dict) == 10

    def test_purges_when_inflated(self, tmp_path):
        cd = _make_data(tmp_path, existing=5, missing=5)
        cd.files_found = ["f"] * 2  # 10 > 2 * 2
        cd._maybe_purge_stale_entries()
        assert len(cd.file_data_dict) == 5
        assert cd.has_new_file_data is True
        state = isolated_app_info_cache().get(
            str(tmp_path), CompareData.PURGE_STATE_KEY, default_val={})
        assert state.get("test_data.pkl") == 5

    def test_extreme_scale_bailout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CompareData, "LARGE_CACHE_CEILING", 10)
        monkeypatch.setattr(CompareData, "LARGE_FOUND_CEILING", 4)
        cd = _make_data(tmp_path, missing=12)
        cd.files_found = ["f"] * 5  # both ceilings met -> user handles it
        cd._maybe_purge_stale_entries()
        assert len(cd.file_data_dict) == 12

    def test_extreme_cache_with_few_found_still_purges(self, tmp_path, monkeypatch):
        """True churn: huge cache over few remaining files must still clean."""
        monkeypatch.setattr(CompareData, "LARGE_CACHE_CEILING", 10)
        monkeypatch.setattr(CompareData, "LARGE_FOUND_CEILING", 4)
        cd = _make_data(tmp_path, existing=1, missing=12)
        cd.files_found = ["f"]  # below found ceiling
        cd._maybe_purge_stale_entries()
        assert len(cd.file_data_dict) == 1

    def test_external_drive_hits_ceilings_at_reduced_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CompareData, "LARGE_CACHE_CEILING", 50)
        monkeypatch.setattr(CompareData, "LARGE_FOUND_CEILING", 20)
        cd = _make_data(tmp_path, missing=12)
        cd.files_found = ["f"] * 5
        # Local drive: 12 < 50 and 5 < 20 -> would purge. External drive:
        # 5 * 12 = 60 >= 50 and 5 * 5 = 25 >= 20 -> bail out.
        monkeypatch.setattr(Utils, "is_external_drive", lambda p: True)
        cd._maybe_purge_stale_entries()
        assert len(cd.file_data_dict) == 12

    def test_repeat_scan_guard(self, tmp_path):
        cd = _make_data(tmp_path, existing=3, missing=2)
        cd._maybe_purge_stale_entries()
        assert len(cd.file_data_dict) == 3

        calls = []
        cd.purge_stale_entries = lambda: calls.append(1) or 0
        # Same size as recorded post-purge state -> no rescan.
        cd._maybe_purge_stale_entries()
        assert not calls

        # Cache grew past the recorded size -> scan runs again.
        cd.file_data_dict[str(tmp_path / "gone_new.png")] = [1.0]
        cd.file_data_dict[str(tmp_path / "gone_new2.png")] = [2.0]
        cd._maybe_purge_stale_entries()
        assert calls


class TestPurgePersistence:
    def test_save_data_persists_purged_dict(self, tmp_path):
        cd = _make_data(tmp_path, existing=2, missing=4)
        real_keys = [k for k in cd.file_data_dict if os.path.isfile(k)]
        cd.files_found = ["f"] * 2  # 6 > 2 * 2 -> purge on save
        cd.save_data()
        with open(str(tmp_path / "test_data.pkl"), "rb") as f:
            stored = pickle.load(f)
        assert sorted(stored.keys()) == sorted(real_keys)

    def test_save_data_survives_purge_failure(self, tmp_path, monkeypatch):
        cd = _make_data(tmp_path, existing=1, missing=3)
        cd.files_found = ["f"]
        cd.has_new_file_data = True
        monkeypatch.setattr(
            cd, "_maybe_purge_stale_entries",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        cd.save_data()  # must not raise
        assert os.path.isfile(str(tmp_path / "test_data.pkl"))

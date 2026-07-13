"""
Unit tests for ClassifierActionsManager.gather_sorted_media_paths.

Covers: per-directory sort_by resolution from app_info_cache (falling back to
config.sort_by when nothing is cached), skipping missing directories via
Utils.isdir_with_retry (not a bare os.path.isdir), and that the resulting
file list actually reflects what's on disk for real directories.
"""
from compare.classifier_actions_manager import ClassifierActionsManager
from utils.app_info_cache import app_info_cache
from utils.config import config
from utils.constants import SortBy
from utils.utils import Utils


class _FakeFileBrowser:
    """Captures constructor args instead of touching the filesystem."""
    instances = []

    def __init__(self, directory, recursive=False, sort_by=None):
        self.directory = directory
        self.recursive = recursive
        self.sort_by = sort_by
        self.is_incremental_loading = False
        self._files = []
        _FakeFileBrowser.instances.append(self)

    def set_directory(self, directory):
        pass

    def get_files(self):
        return self._files


def _patch_file_browser(monkeypatch):
    _FakeFileBrowser.instances = []
    monkeypatch.setattr("files.file_browser.FileBrowser", _FakeFileBrowser)


# ---------------------------------------------------------------------------
# sort_by resolution: cache first, config.sort_by fallback
# ---------------------------------------------------------------------------

class TestSortByResolution:
    def test_uses_cached_sort_by_when_present(self, tmp_path, monkeypatch):
        directory = str(tmp_path)
        _patch_file_browser(monkeypatch)
        app_info_cache.set(directory, "sort_by", SortBy.CREATION_TIME)

        ClassifierActionsManager.gather_sorted_media_paths([directory])

        assert _FakeFileBrowser.instances[0].sort_by == SortBy.CREATION_TIME

    def test_falls_back_to_config_sort_by_when_not_cached(self, tmp_path, monkeypatch):
        directory = str(tmp_path)
        _patch_file_browser(monkeypatch)
        monkeypatch.setattr(config, "sort_by", SortBy.NAME)

        ClassifierActionsManager.gather_sorted_media_paths([directory])

        assert _FakeFileBrowser.instances[0].sort_by == SortBy.NAME

    def test_invalid_cached_sort_by_falls_back_to_config_default(self, tmp_path, monkeypatch):
        directory = str(tmp_path)
        _patch_file_browser(monkeypatch)
        monkeypatch.setattr(config, "sort_by", SortBy.NAME)
        app_info_cache.set(directory, "sort_by", "not a real sort")

        ClassifierActionsManager.gather_sorted_media_paths([directory])

        assert _FakeFileBrowser.instances[0].sort_by == SortBy.NAME

    def test_string_cached_sort_by_is_resolved_via_sortby_get(self, tmp_path, monkeypatch):
        directory = str(tmp_path)
        _patch_file_browser(monkeypatch)
        app_info_cache.set(directory, "sort_by", SortBy.CREATION_TIME.value)

        ClassifierActionsManager.gather_sorted_media_paths([directory])

        assert _FakeFileBrowser.instances[0].sort_by == SortBy.CREATION_TIME

    def test_different_directories_can_have_different_cached_sort(self, tmp_path, monkeypatch):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        _patch_file_browser(monkeypatch)
        app_info_cache.set(str(dir1), "sort_by", SortBy.CREATION_TIME)
        app_info_cache.set(str(dir2), "sort_by", SortBy.NAME_LENGTH)

        ClassifierActionsManager.gather_sorted_media_paths([str(dir1), str(dir2)])

        by_dir = {inst.directory: inst.sort_by for inst in _FakeFileBrowser.instances}
        assert by_dir[str(dir1)] == SortBy.CREATION_TIME
        assert by_dir[str(dir2)] == SortBy.NAME_LENGTH

    def test_recursive_true_passed_to_file_browser(self, tmp_path, monkeypatch):
        directory = str(tmp_path)
        _patch_file_browser(monkeypatch)

        ClassifierActionsManager.gather_sorted_media_paths([directory])

        assert _FakeFileBrowser.instances[0].recursive is True


# ---------------------------------------------------------------------------
# Missing directories
# ---------------------------------------------------------------------------

class TestMissingDirectorySkipped:
    def test_nonexistent_directory_is_skipped(self, tmp_path, monkeypatch):
        _patch_file_browser(monkeypatch)
        missing = str(tmp_path / "does_not_exist")

        result = ClassifierActionsManager.gather_sorted_media_paths([missing])

        assert result == []
        assert _FakeFileBrowser.instances == []

    def test_mix_of_missing_and_valid_directories(self, tmp_path, monkeypatch):
        _patch_file_browser(monkeypatch)
        valid = str(tmp_path)
        missing = str(tmp_path / "does_not_exist")

        ClassifierActionsManager.gather_sorted_media_paths([missing, valid])

        assert len(_FakeFileBrowser.instances) == 1
        assert _FakeFileBrowser.instances[0].directory == valid

    def test_uses_isdir_with_retry_not_bare_isdir(self, tmp_path, monkeypatch):
        """Directory existence check must go through Utils.isdir_with_retry
        (handles sleeping external drives), not a bare os.path.isdir."""
        directory = str(tmp_path)
        _patch_file_browser(monkeypatch)
        calls = []
        monkeypatch.setattr(
            Utils, "isdir_with_retry", staticmethod(lambda path, *a, **kw: calls.append(path) or True)
        )

        ClassifierActionsManager.gather_sorted_media_paths([directory])

        assert calls == [directory]


# ---------------------------------------------------------------------------
# Real filesystem gathering (no FileBrowser mocking)
# ---------------------------------------------------------------------------

class TestRealFileGathering:
    def test_returns_files_in_directory(self, tmp_path):
        (tmp_path / "b.jpg").write_bytes(b"1")
        (tmp_path / "a.jpg").write_bytes(b"2")

        result = ClassifierActionsManager.gather_sorted_media_paths([str(tmp_path)])

        assert set(result) == {str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")}

    def test_empty_directory_returns_empty_list(self, tmp_path):
        result = ClassifierActionsManager.gather_sorted_media_paths([str(tmp_path)])
        assert result == []

    def test_multiple_directories_combined(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "x.jpg").write_bytes(b"1")
        (dir2 / "y.jpg").write_bytes(b"2")

        result = ClassifierActionsManager.gather_sorted_media_paths([str(dir1), str(dir2)])

        assert set(result) == {str(dir1 / "x.jpg"), str(dir2 / "y.jpg")}

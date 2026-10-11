"""Tests for utils.repo_paths' build-related directories: the app data dir,
the logs dir, and where user_root()/resource_root() point in a build."""

import os

import pytest

from utils import repo_paths


@pytest.fixture
def compiled(monkeypatch):
    monkeypatch.setattr(repo_paths, "_COMPILED", True)


def test_app_data_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv(repo_paths.APP_DATA_DIR_ENV, str(tmp_path))
    assert repo_paths.app_data_dir() == str(tmp_path)


def test_app_data_dir_default_is_named_for_the_app(monkeypatch):
    monkeypatch.delenv(repo_paths.APP_DATA_DIR_ENV, raising=False)
    assert os.path.basename(repo_paths.app_data_dir()) == repo_paths.APP_DIR_NAME


def test_logs_dir_is_created_under_the_app_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(repo_paths.APP_DATA_DIR_ENV, str(tmp_path / "data"))
    logs = repo_paths.logs_dir()
    assert logs == str(tmp_path / "data" / "logs")
    assert os.path.isdir(logs)


def test_checkout_writes_to_the_app_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(repo_paths.APP_DATA_DIR_ENV, str(tmp_path))
    assert repo_paths.user_root() == str(tmp_path)
    assert repo_paths.resource_root() == repo_paths.repo_root()


def test_build_writes_to_the_app_data_dir(compiled, monkeypatch, tmp_path):
    monkeypatch.setenv(repo_paths.APP_DATA_DIR_ENV, str(tmp_path))
    assert repo_paths.user_root() == str(tmp_path)


def test_build_resources_sit_beside_the_utils_package(compiled):
    # A build has no src/ level: utils/ is directly in the build folder.
    utils_dir = os.path.dirname(os.path.abspath(repo_paths.__file__))
    assert repo_paths.resource_root() == os.path.dirname(utils_dir)


def _write(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def legacy_repo(tmp_path):
    """A repo root holding shipped files and user files at their old places."""
    repo = tmp_path / "repo"
    for name in ("config_example.json", "suggested_classifier_models.json", "config.json", "config1.json"):
        _write(repo / "configs" / name, name)
    for name in ("app_info_cache.enc", "app_info_cache.enc.bak", "app_info_cache.enc.bak2",
                 "classifier_prediction_cache.enc", ".classifier_prediction_cache_1.tmp",
                 "file_paths.json"):
        _write(repo / name, name)
    _write(repo / "assets" / "pipelines" / "example.json")
    _write(repo / "assets" / "icon.png")
    _write(repo / "src" / "utils" / "app_info_cache.py")
    return repo


def test_migration_moves_user_files_and_leaves_shipped_ones(legacy_repo, tmp_path):
    target = tmp_path / "app_data"
    repo_paths._move_legacy_user_files(str(legacy_repo), str(target))
    for relative in ("configs/config.json", "configs/config1.json", "app_info_cache.enc",
                     "app_info_cache.enc.bak", "app_info_cache.enc.bak2",
                     "classifier_prediction_cache.enc", "file_paths.json",
                     "assets/pipelines/example.json"):
        assert (target / relative).is_file(), relative
        assert not (legacy_repo / relative).exists(), relative
    for relative in ("configs/config_example.json", "configs/suggested_classifier_models.json",
                     "assets/icon.png", ".classifier_prediction_cache_1.tmp",
                     "src/utils/app_info_cache.py"):
        assert (legacy_repo / relative).is_file(), relative
        assert not (target / relative).exists(), relative


def _age(path, seconds_ago):
    """Set *path*'s modification time *seconds_ago* seconds in the past."""
    t = os.path.getmtime(path) - seconds_ago
    os.utime(path, (t, t))


def test_migration_replaces_newer_target_files(legacy_repo, tmp_path):
    target = tmp_path / "app_data"
    _write(target / "configs" / "config.json", "existing")
    _age(legacy_repo / "configs" / "config.json", 3600)
    repo_paths._move_legacy_user_files(str(legacy_repo), str(target))
    assert (target / "configs" / "config.json").read_text(encoding="utf-8") == "config.json"
    assert not (legacy_repo / "configs" / "config.json").exists()


def test_migration_keeps_target_when_the_legacy_file_is_newer(legacy_repo, tmp_path):
    target = tmp_path / "app_data"
    _write(target / "app_info_cache.enc", "existing")
    _age(target / "app_info_cache.enc", 3600)
    repo_paths._move_legacy_user_files(str(legacy_repo), str(target))
    assert (target / "app_info_cache.enc").read_text(encoding="utf-8") == "existing"
    assert (legacy_repo / "app_info_cache.enc").is_file()
    # Files without a counterpart still move.
    assert (target / "app_info_cache.enc.bak").is_file()


def test_migration_merges_folders_file_by_file(legacy_repo, tmp_path):
    target = tmp_path / "app_data"
    pipelines = legacy_repo / "assets" / "pipelines"
    _write(pipelines / "older.json", "legacy")
    _age(pipelines / "older.json", 3600)
    _write(pipelines / "newer.json", "legacy")
    _write(target / "assets" / "pipelines" / "older.json", "target")
    _write(target / "assets" / "pipelines" / "newer.json", "target")
    _age(target / "assets" / "pipelines" / "newer.json", 3600)
    repo_paths._move_legacy_user_files(str(legacy_repo), str(target))
    assert (target / "assets" / "pipelines" / "older.json").read_text(encoding="utf-8") == "legacy"
    assert (target / "assets" / "pipelines" / "newer.json").read_text(encoding="utf-8") == "target"
    assert (target / "assets" / "pipelines" / "example.json").is_file()
    # The newer legacy file stays, so its folder does too.
    assert (pipelines / "newer.json").is_file()
    assert not (pipelines / "older.json").exists()


def test_migration_removes_a_fully_merged_source_folder(legacy_repo, tmp_path):
    target = tmp_path / "app_data"
    _write(target / "assets" / "pipelines" / "other.json", "target")
    repo_paths._move_legacy_user_files(str(legacy_repo), str(target))
    assert not (legacy_repo / "assets" / "pipelines").exists()
    assert (target / "assets" / "pipelines" / "example.json").is_file()
    assert (target / "assets" / "pipelines" / "other.json").is_file()


@pytest.fixture
def migration_not_done(monkeypatch, tmp_path):
    """Re-arms the once-per-process migration with both movers stubbed and
    the home directory pointed at tmp_path: these tests remove the override
    variables, so a real move would act on the actual repo and home
    directory. Request it before anything that removes them."""
    monkeypatch.setattr(repo_paths, "_migration_done", False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    moves = []
    monkeypatch.setattr(repo_paths, "_move_legacy_user_files",
                        lambda src, dst: moves.append(("repo", src, dst)))
    monkeypatch.setattr(repo_paths, "_move_legacy_home_files",
                        lambda home, dst: moves.append(("home", home, dst)))
    return moves


def _without_overrides(monkeypatch, app_data):
    for name in repo_paths._OVERRIDE_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(repo_paths, "default_app_data_dir", lambda: str(app_data))


def test_checkout_migrates_once_without_overrides(migration_not_done, monkeypatch, tmp_path):
    _without_overrides(monkeypatch, tmp_path / "app_data")
    repo_paths.user_root()
    repo_paths.user_root()
    assert migration_not_done == [
        ("repo", repo_paths.repo_root(), str(tmp_path / "app_data")),
        ("home", str(tmp_path / "home"), str(tmp_path / "app_data")),
    ]


@pytest.mark.parametrize("name", ["WEIDR_APP_DATA_DIR", "WEIDR_CONFIGS_DIR", "WEIDR_CACHE_DIR"])
def test_no_migration_under_an_override(migration_not_done, monkeypatch, tmp_path, name):
    _without_overrides(monkeypatch, tmp_path / "app_data")
    monkeypatch.setenv(name, str(tmp_path / "override"))
    repo_paths.user_root()
    assert migration_not_done == []


def test_a_build_migrates_only_from_the_home_directory(migration_not_done, compiled, monkeypatch, tmp_path):
    _without_overrides(monkeypatch, tmp_path / "app_data")
    repo_paths.user_root()
    assert migration_not_done == [("home", str(tmp_path / "home"), str(tmp_path / "app_data"))]


def test_home_migration_moves_the_no_directory_compare_cache(tmp_path):
    legacy = tmp_path / "home" / ".weidr"
    _write(legacy / repo_paths.NO_DIRECTORY_COMPARE_CACHE_NAME / "embeddings.pkl")
    target = tmp_path / "app_data"
    repo_paths._move_legacy_home_files(str(tmp_path / "home"), str(target))
    assert (target / repo_paths.NO_DIRECTORY_COMPARE_CACHE_NAME / "embeddings.pkl").is_file()
    assert not legacy.exists()


def test_home_migration_keeps_the_folder_while_other_entries_remain(tmp_path):
    legacy = tmp_path / "home" / ".weidr"
    _write(legacy / repo_paths.NO_DIRECTORY_COMPARE_CACHE_NAME / "embeddings.pkl")
    _write(legacy / "something_else.txt")
    repo_paths._move_legacy_home_files(str(tmp_path / "home"), str(tmp_path / "app_data"))
    assert (legacy / "something_else.txt").is_file()
    assert not (legacy / repo_paths.NO_DIRECTORY_COMPARE_CACHE_NAME).exists()


def test_home_migration_keeps_target_when_the_legacy_file_is_newer(tmp_path):
    legacy = tmp_path / "home" / ".weidr"
    legacy_file = legacy / repo_paths.NO_DIRECTORY_COMPARE_CACHE_NAME / "embeddings.pkl"
    _write(legacy_file, "legacy")
    target = tmp_path / "app_data"
    target_file = target / repo_paths.NO_DIRECTORY_COMPARE_CACHE_NAME / "embeddings.pkl"
    _write(target_file, "target")
    _age(target_file, 3600)
    repo_paths._move_legacy_home_files(str(tmp_path / "home"), str(target))
    assert target_file.read_text(encoding="utf-8") == "target"
    assert legacy_file.is_file()


def test_relative_file_paths_json_path_resolves_under_the_user_root():
    import utils.config as cfg
    # The per-test config is the example config, which gives a relative path.
    assert cfg.config.dict["file_paths_json_path"] == "file_paths.json"
    assert cfg.config.file_paths_json_path == os.path.join(repo_paths.user_root(), "file_paths.json")
    # The config file keeps what the user wrote.
    assert cfg.config._build_persisted_config_dict()["file_paths_json_path"] == "file_paths.json"

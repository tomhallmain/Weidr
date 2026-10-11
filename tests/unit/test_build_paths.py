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


def test_checkout_writes_to_the_repo(monkeypatch, tmp_path):
    monkeypatch.setenv(repo_paths.APP_DATA_DIR_ENV, str(tmp_path))
    assert repo_paths.user_root() == repo_paths.repo_root()
    assert repo_paths.resource_root() == repo_paths.repo_root()


def test_build_writes_to_the_app_data_dir(compiled, monkeypatch, tmp_path):
    monkeypatch.setenv(repo_paths.APP_DATA_DIR_ENV, str(tmp_path))
    assert repo_paths.user_root() == str(tmp_path)


def test_build_resources_sit_beside_the_utils_package(compiled):
    # A build has no src/ level: utils/ is directly in the build folder.
    utils_dir = os.path.dirname(os.path.abspath(repo_paths.__file__))
    assert repo_paths.resource_root() == os.path.dirname(utils_dir)

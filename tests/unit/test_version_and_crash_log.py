"""utils.version (build info, git commit lookup, source drift) and
utils.crash_log (the faulthandler log)."""

import json

import pytest

from utils import crash_log, repo_paths, version


@pytest.fixture
def resources(monkeypatch, tmp_path):
    monkeypatch.setattr(repo_paths, "resource_root", lambda: str(tmp_path))
    return tmp_path


def test_checkout_has_no_build_info(resources):
    assert version.build_info() == {}
    assert version.describe() == f"Weidr {version.APP_VERSION} (source checkout)"
    assert version.source_drift() is None


def test_describe_names_build_and_commit(resources):
    (resources / version.BUILD_INFO_FILE).write_text(json.dumps({
        "version": version.APP_VERSION, "build_id": "20261010-120000", "python": "3.11.13",
        "packages": {"torch": "2.14.1"}, "source": {"commit": "a" * 40, "changes": "x"},
    }), encoding="utf-8")
    line = version.describe()
    assert "build 20261010-120000" in line
    assert "aaaaaaaaaa with uncommitted changes" in line
    assert "torch 2.14.1" in line


def test_read_git_commit_follows_a_branch_ref(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (git / "refs" / "heads" / "master").write_text("b" * 40 + "\n", encoding="utf-8")
    assert version.read_git_commit(str(tmp_path)) == "b" * 40


def test_read_git_commit_falls_back_to_packed_refs(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (git / "packed-refs").write_text("c" * 40 + " refs/heads/master\n", encoding="utf-8")
    assert version.read_git_commit(str(tmp_path)) == "c" * 40


def test_read_git_commit_without_git_dir(tmp_path):
    assert version.read_git_commit(str(tmp_path)) == ""


def test_source_drift_reports_a_missing_checkout(resources, tmp_path):
    (resources / version.BUILD_INFO_FILE).write_text(json.dumps({
        "source": {"repo": str(tmp_path / "gone"), "commit": "d" * 40, "changes": ""},
    }), encoding="utf-8")
    drifted, message = version.source_drift()
    assert drifted is False
    assert "not found" in message


def test_fault_log_writes_a_start_line(monkeypatch, tmp_path):
    enabled = []
    monkeypatch.setattr(crash_log, "get_log_dir", lambda: tmp_path)
    monkeypatch.setattr(crash_log.faulthandler, "enable", lambda **kwargs: enabled.append(kwargs))
    monkeypatch.setattr(crash_log, "_fault_log", None)
    try:
        path = crash_log.enable_fault_log()
        assert path == str(tmp_path / crash_log.FAULT_LOG_NAME)
        assert enabled and enabled[0]["all_threads"] is True
    finally:
        crash_log._fault_log.close()
    assert (tmp_path / crash_log.FAULT_LOG_NAME).read_text(encoding="utf-8").startswith("--- started ")


def test_fault_log_restarts_when_too_large(monkeypatch, tmp_path):
    log = tmp_path / crash_log.FAULT_LOG_NAME
    log.write_text("x" * (crash_log.MAX_FAULT_LOG_BYTES + 1), encoding="utf-8")
    monkeypatch.setattr(crash_log, "get_log_dir", lambda: tmp_path)
    monkeypatch.setattr(crash_log.faulthandler, "enable", lambda **kwargs: None)
    monkeypatch.setattr(crash_log, "_fault_log", None)
    try:
        crash_log.enable_fault_log()
    finally:
        crash_log._fault_log.close()
    assert log.read_text(encoding="utf-8").startswith("--- started ")

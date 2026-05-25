"""
Proof tests: assert that no test touches the production config or cache files.

These tests verify the isolation contract enforced by the autouse fixtures in
tests/conftest.py.  They are the CI guardrail — if isolation breaks (e.g. the
bootstrap env vars are removed or the autouse fixture is deleted), these tests
fail before any test has a chance to corrupt user state.
"""

import os
import pytest

import utils.config as cfg
import utils.app_info_cache as aic


# ---------------------------------------------------------------------------
# Session fixture: record production file mtimes before any test runs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def prod_file_mtimes():
    """
    Snapshot mtime of production config and cache files at session start.
    Skips silently for files that do not exist (fresh checkout, CI).
    Returns a dict of {path: mtime_ns} for files that were present.
    """
    candidates = [
        cfg.Config.CONFIGS_DIR_LOC and os.path.join(cfg.Config.CONFIGS_DIR_LOC, "config.json"),
        aic.AppInfoCache.CACHE_LOC,
    ]
    return {
        p: os.stat(p).st_mtime_ns
        for p in candidates
        if p and os.path.isfile(p)
    }


# ---------------------------------------------------------------------------
# Per-test isolation proofs
# ---------------------------------------------------------------------------

class TestConfigIsolation:
    def test_config_path_is_under_tmp_path(self, tmp_path):
        """config.config_path must resolve to the per-test temp dir, not the repo."""
        configs_dir = str(tmp_path / "configs")
        assert cfg.config.config_path.startswith(configs_dir), (
            f"config.config_path ({cfg.config.config_path!r}) is not under the "
            f"per-test configs dir ({configs_dir!r}). "
            "isolated_singletons fixture may not be running."
        )

    def test_config_path_not_in_repo_configs(self):
        """config.config_path must not point into the real repo configs/ directory."""
        assert cfg.Config.CONFIGS_DIR_LOC not in cfg.config.config_path, (
            f"config.config_path ({cfg.config.config_path!r}) points into the "
            f"production configs directory ({cfg.Config.CONFIGS_DIR_LOC!r})."
        )

    def test_weidr_configs_dir_env_var_is_set(self, tmp_path):
        """WEIDR_CONFIGS_DIR must be set to a per-test path during tests."""
        configs_dir = os.environ.get("WEIDR_CONFIGS_DIR", "")
        assert configs_dir, "WEIDR_CONFIGS_DIR is not set — bootstrap did not run."
        assert configs_dir.startswith(str(tmp_path)), (
            f"WEIDR_CONFIGS_DIR ({configs_dir!r}) is not under tmp_path ({tmp_path})."
        )


class TestCacheIsolation:
    def test_cache_loc_is_under_weidr_cache_dir(self):
        """app_info_cache._cache_loc must be under WEIDR_CACHE_DIR."""
        cache_dir = os.environ.get("WEIDR_CACHE_DIR", "")
        assert cache_dir, "WEIDR_CACHE_DIR is not set — bootstrap did not run."
        assert aic.app_info_cache._cache_loc.startswith(cache_dir), (
            f"app_info_cache._cache_loc ({aic.app_info_cache._cache_loc!r}) is not "
            f"under WEIDR_CACHE_DIR ({cache_dir!r})."
        )

    def test_cache_loc_not_at_production_path(self):
        """app_info_cache._cache_loc must not equal the production cache file."""
        assert aic.app_info_cache._cache_loc != aic.AppInfoCache.CACHE_LOC, (
            "app_info_cache._cache_loc points to the production cache file. "
            "isolated_singletons fixture may not be running."
        )

    def test_weidr_cache_dir_env_var_is_set(self, tmp_path):
        """WEIDR_CACHE_DIR must be set to a per-test path during tests."""
        cache_dir = os.environ.get("WEIDR_CACHE_DIR", "")
        assert cache_dir, "WEIDR_CACHE_DIR is not set — bootstrap did not run."
        assert cache_dir.startswith(str(tmp_path)), (
            f"WEIDR_CACHE_DIR ({cache_dir!r}) is not under tmp_path ({tmp_path})."
        )


class TestCachePersistenceFingerprint:
    def test_written_key_survives_reload(self, tmp_path):
        """
        Write a unique key into the per-test cache, reload from disk in a new
        instance, and assert the key is present — proving store/load works
        against the isolated directory, not a shared location.
        """
        unique_key = f"proof_{id(tmp_path)}"
        sentinel = "isolation_ok"

        aic.app_info_cache.set_meta(unique_key, sentinel)
        aic.app_info_cache.store()

        reloaded = aic.AppInfoCache()
        assert reloaded.get_meta(unique_key) == sentinel, (
            "Reloaded cache did not contain the key written in this test. "
            "store()/load() may be targeting a different directory."
        )

    def test_key_absent_in_sibling_instance(self, tmp_path):
        """
        A key written in one cache instance must not appear in a second instance
        that uses a different temp directory — cross-test bleed would surface here.
        """
        unique_key = f"sibling_{id(tmp_path)}"
        aic.app_info_cache.set_meta(unique_key, "should_not_leak")
        aic.app_info_cache.store()

        # Instance pointing at a completely different temp directory
        other_dir = tmp_path / "other_cache"
        other_dir.mkdir()
        original_env = os.environ.get("WEIDR_CACHE_DIR")
        try:
            os.environ["WEIDR_CACHE_DIR"] = str(other_dir)
            other = aic.AppInfoCache()
        finally:
            if original_env is not None:
                os.environ["WEIDR_CACHE_DIR"] = original_env
            else:
                del os.environ["WEIDR_CACHE_DIR"]

        assert other.get_meta(unique_key) is None, (
            f"Key {unique_key!r} appeared in a cache instance pointing at a "
            "different directory — possible shared-state bleed."
        )


class TestProductionFilesUntouched:
    def test_production_config_mtime_unchanged(self, prod_file_mtimes):
        """
        Production config.json mtime must not have changed since session start.
        Skipped when the file does not exist (CI / fresh checkout).
        """
        prod_config = os.path.join(cfg.Config.CONFIGS_DIR_LOC, "config.json")
        if prod_config not in prod_file_mtimes:
            pytest.skip("No production config.json present — skipping mtime check.")
        current = os.stat(prod_config).st_mtime_ns
        assert current == prod_file_mtimes[prod_config], (
            "Production configs/config.json was modified during the test session. "
            "A test wrote to the real config file — isolation is broken."
        )

    def test_production_cache_mtime_unchanged(self, prod_file_mtimes):
        """
        Production app_info_cache.enc mtime must not have changed since session start.
        Skipped when the file does not exist.
        """
        prod_cache = aic.AppInfoCache.CACHE_LOC
        if prod_cache not in prod_file_mtimes:
            pytest.skip("No production app_info_cache.enc present — skipping mtime check.")
        current = os.stat(prod_cache).st_mtime_ns
        assert current == prod_file_mtimes[prod_cache], (
            "Production app_info_cache.enc was modified during the test session. "
            "A test wrote to the real cache file — isolation is broken."
        )

"""
conftest for tests/unit/.

Mirrors the module-level env var setup from the root conftest.py. This is
necessary because pytest loads each directory's conftest before collecting
tests in that directory, and the singletons may not yet be imported when the
root conftest runs in some collection orders.
"""

import os

# Re-apply the env vars at module load. setdefault means the root conftest
# values win if already set; this just ensures they're present either way.
if "WEIDR_CACHE_DIR" not in os.environ:
    import tempfile, shutil, atexit
    _tmp = tempfile.mkdtemp(prefix="weidr_unit_")
    os.environ["WEIDR_CACHE_DIR"] = os.path.join(_tmp, "cache")
    os.environ["WEIDR_CONFIGS_DIR"] = os.path.join(_tmp, "configs")
    os.environ["WEIDR_APP_DATA_DIR"] = os.path.join(_tmp, "app_data")
    os.makedirs(os.environ["WEIDR_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["WEIDR_CONFIGS_DIR"], exist_ok=True)
    _src = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config_example.json")
    shutil.copy(_src, os.path.join(os.environ["WEIDR_CONFIGS_DIR"], "config.json"))
    atexit.register(shutil.rmtree, _tmp, True)

import os

from tests.fixtures.compare_image_fixtures import compare_colors_dir  # noqa: F401

if "WEIDR_CACHE_DIR" not in os.environ:
    import atexit, shutil, tempfile
    _tmp = tempfile.mkdtemp(prefix="weidr_compare_")
    os.environ["WEIDR_CACHE_DIR"] = os.path.join(_tmp, "cache")
    os.environ["WEIDR_CONFIGS_DIR"] = os.path.join(_tmp, "configs")
    os.makedirs(os.environ["WEIDR_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["WEIDR_CONFIGS_DIR"], exist_ok=True)
    _src = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config_example.json")
    shutil.copy(_src, os.path.join(os.environ["WEIDR_CONFIGS_DIR"], "config.json"))
    atexit.register(shutil.rmtree, _tmp, True)

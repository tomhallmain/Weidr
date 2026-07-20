"""
Shared test utilities: asset paths and reusable helper objects.
"""

import os
import threading
import time

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))


def asset(filename: str) -> str:
    """Return the absolute path to a file inside tests/assets/."""
    return os.path.join(ASSETS_DIR, filename)


MALFORMED_WEBM = asset("example_malformed_absent_cues.webm")


class HangingVlcPlayer:
    """Stand-in for a VLC MediaPlayer whose stop() sleeps indefinitely.

    Wraps a real player so every other attribute still delegates to it.
    Used to reliably exercise the video_stop() timeout path without depending
    on a specific VLC version or file triggering the hang.
    """

    def __init__(self, real_player):
        self._real = real_player
        self.stop_called = threading.Event()

    def stop(self):
        self.stop_called.set()
        # Sleep past the 3-second timeout in video_stop() -- just enough to
        # reliably trigger the abandon-and-replace path, not so long that the
        # thread (and the real player it wraps) stays alive across dozens of
        # later, unrelated tests and their Qt teardown.
        time.sleep(10)

    def set_media(self, _media):
        pass

    def __getattr__(self, name):
        return getattr(self._real, name)


def isolated_app_info_cache():
    """Return the per-test isolated app_info_cache instance.

    The autouse isolated_singletons fixture (tests/conftest.py) installs a
    fresh AppInfoCache as utils.app_info_cache.app_info_cache for each test.
    Looking the attribute up at call time yields that isolated instance — the
    same one production code reaches via deferred imports. A module-level
    `from utils.app_info_cache import app_info_cache` in a test file would
    instead bind the pre-isolation original and read stale state.
    """
    import utils.app_info_cache as aic
    return aic.app_info_cache

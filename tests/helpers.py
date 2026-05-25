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
        # Sleep well beyond the 3-second timeout in video_stop().
        time.sleep(30)

    def set_media(self, _media):
        pass

    def __getattr__(self, name):
        return getattr(self._real, name)

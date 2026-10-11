"""Flags for starting console programs from a GUI process.

On Windows, a process with no console of its own (Weidr started from
Explorer, ``pythonw``, or the compiled build) gives each console program it
starts a new console window, which flashes up for the program's lifetime.
Pass ``creationflags=NO_CONSOLE_WINDOW`` to ``subprocess`` calls that run a
console program (ffmpeg, ffprobe, ``gimp --version``) and capture or discard
its output. Programs meant to open a window of their own do not need it.
"""

import subprocess
import sys

NO_CONSOLE_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

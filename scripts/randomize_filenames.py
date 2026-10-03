#!/usr/bin/env python3
"""Command-line entry for files/randomize_filenames.py; see that module for
the options and behavior.

    python scripts/randomize_filenames.py "C:\\path\\to\\files" [--execute] [--verbose]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from files.randomize_filenames import main

if __name__ == "__main__":
    raise SystemExit(main())

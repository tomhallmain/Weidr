"""The project root, resolved once here instead of independently at each
call site that needs it.

Several modules need to find top-level siblings that are never imported as
packages -- configs/, locale/, assets/, scripts/ -- so they locate them by
walking up from their own __file__ instead. Six of those had each grown
their own hop count, which is fragile: a hop count is only correct for the
exact directory depth the computing file happens to live at, and silently
wrong (pointing one directory too high or low) the moment that file moves.
Routing all of them through one function here means a future move only
needs this file's hop count corrected, not six independent ones scattered
across the codebase.

This file's own hop count is exactly that kind of assumption, and is not
exempt from the problem it exists to centralize: it is currently correct
for src/utils/repo_paths.py, three directories under the project root. If
this module is ever relocated again, this is the one place that needs its
hop count bumped to match the new depth.
"""

import os


def repo_root() -> str:
    """The project root -- three directories up from this file today."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

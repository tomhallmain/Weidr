"""Preload the Windows ICU for the Nuitka process and its compile-time children.

Nuitka queries PySide6 in a fresh ``python -c`` interpreter. That interpreter
inherits PYTHONPATH, so this module runs at startup when its directory is
first on that path. conda-forge cairo's ICU in ``Library\\bin`` is otherwise
the ``icuuc.dll`` Qt loads, and the import fails.
"""

import os
import sys

_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

from utils.windows_dll_compat import prefer_system_icu

prefer_system_icu()

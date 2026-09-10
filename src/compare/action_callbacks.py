from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ActionCallbacks:
    """Bundles the optional side-effect callbacks passed through the classifier
    and pipeline execution stack so they can be forwarded as a single argument."""

    hide_callback: Optional[Callable] = None
    notify_callback: Optional[Callable] = None
    add_mark_callback: Optional[Callable] = None
    blur_callback: Optional[Callable] = None
    generate_callback: Optional[Callable] = None
    scramble_callback: Optional[Callable] = None

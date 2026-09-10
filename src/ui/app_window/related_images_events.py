"""Qt signal bridge for related-image action results.

Owned by AppActions (one per app window, created lazily) and outliving any
related images window. Chosen over a plain Python callback for two
properties the callback lacked:

- Thread bridge: results may be reported from whatever thread an action
  completes on; Qt AutoConnection delivers to the receiver's (main) thread
  as a queued event, so GUI updates are always main-thread.
- Automatic lifetime: connections whose receiver is a QObject are
  disconnected by Qt when the receiver is destroyed — no manual
  register/unregister bookkeeping in the window.
"""

from PySide6.QtCore import QObject, Signal


class RelatedImagesResultSignals(QObject):
    # message, action_label, data (dict or None)
    result = Signal(str, object, object)

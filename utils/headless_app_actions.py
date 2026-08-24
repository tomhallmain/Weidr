"""Qt-free implementation of the AppActions contract.

AppActions is overwhelmingly a presentation port: the Qt-free packages call it
to tell the user something or to change what is displayed. Those operations
have no meaning without a GUI, so here they become logging or no-ops.

A minority of the contract is not presentation at all -- it navigates, reads
state, or touches the filesystem. Silently stubbing those would let a caller
believe work happened when it did not, and for `delete` / `hide_current_media`
would hide destructive intent. They must therefore be supplied by the caller;
an unsupplied one raises when called rather than returning a plausible-looking
None. The set is listed in DOMAIN_ACTIONS below and doubles as the worklist for
moving those calls off AppActions entirely.

This module must not import PySide6, directly or transitively. AppActions
itself is already safe to import here: its only non-stdlib import is
ui.app_style, which is a pure constants module with no imports of its own.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from utils.app_actions import AppActions
from utils.logging_setup import get_logger

logger = get_logger("headless_app_actions")


class HeadlessActionUnavailable(RuntimeError):
    """A domain action was called that the caller never supplied."""


# ---------------------------------------------------------------------------
# Contract partition
# ---------------------------------------------------------------------------

# Message-bearing: the text is the whole point, so it is logged rather than
# dropped -- the Qt implementation logs alerts too.
MESSAGE_ACTIONS = (
    "toast",
    "title_notify",
    "_alert",
    "title",
)

# Display-only: changing, refreshing, or opening something on screen. Nothing
# to do without a screen, and nothing a caller can observe.
NOOP_ACTIONS = (
    "new_window",
    "refresh",
    "refocus",
    "refresh_all_compares",
    "refresh_masonry",
    "start_loading_spinner",
    "stop_loading_spinner",
    "start_progress_bar",
    "stop_progress_bar",
    "create_media",
    "release_media_canvas",
    "request_media_blur",
    "copy_media_path",
    "get_media_details",
    "open_move_marks_window",
    "open_password_admin_window",
    "open_file_action_sets_window",
    "play_media",
    "pause_media",
    "toggle_media_play_pause",
    "seek_media",
    "stop_media",
    "set_media_volume",
    "toggle_media_mute",
    "set_media_mute",
    "restart_slideshow_timer_after_interaction",
    "_set_toggled_view_matches",
    "_set_label_state",
    "_add_buttons_for_mode",
    # Begins with one domain call and is otherwise UI reaction: toast, restore
    # the previous mode, redisplay. Callers use it to announce that moved-out
    # files came back, not to request the restore. The domain half is reachable
    # on its own -- CompareManager.maybe_restore_removal_undo_snapshot() -- so a
    # caller that needs it does not go through here.
    "restore_compare_state_for_undone_move",
    # set_mode carries both display and state meaning; treated as display here
    # because nothing headless reads back a mode set this way. Revisit when the
    # domain calls move off AppActions.
    "set_mode",
)

# Queried for a value that must not be invented. Each returns the value that
# means "nothing / no", matching what the Qt side yields when there is no
# window, no selection, or a dismissed dialog.
NEUTRAL_RETURN_ACTIONS: Dict[str, Any] = {
    "get_window": None,
    "get_open_windows": (),
    "find_window_with_compare": None,
    "is_fullscreen": False,
    "get_media_volume": 0,
    "is_media_muted": False,
    # Navigation across windows: find the file and put it on screen, opening a
    # temporary canvas if no window holds it. Nothing is created, moved or
    # persisted, so there is nothing to do without a screen. Both report
    # whether they navigated, and False -- "did not" -- is the truthful answer
    # here; the pdf_creator call sites already treat a failure to navigate as
    # non-fatal and carry on.
    "go_to_file": False,
    "go_to_file_by_index": False,
    # Advance-and-render: moves the browser cursor, then displays what it
    # landed on. The cursor half stands on its own as FileBrowser.next_file(),
    # so a caller that wants to advance does not come through here. Its one
    # caller in a Qt-free package is the "no groups left" fallback, which
    # ignores the return value.
    "show_next_media": False,
}

# Not presentation: these navigate, mutate, persist, or report real state.
# Must be supplied by the caller.
DOMAIN_ACTIONS = (
    "get_active_media_filepath",
    "get_base_dir",
    "set_base_dir",
    "delete",
    "hide_current_media",
    "store_info_cache",
    "is_compare_running",
    "run_image_generation",
    "set_marks_from_downstream_related_images",
    "run_compare",
    "get_compare_mode",
)


# ---------------------------------------------------------------------------
# Qt-free stand-in for the related-images signal bridge
# ---------------------------------------------------------------------------

class _NullSignal:
    """Accepts connect/disconnect/emit and does nothing."""

    def connect(self, *_args, **_kwargs) -> None:
        return None

    def disconnect(self, *_args, **_kwargs) -> None:
        return None

    def emit(self, *args, **_kwargs) -> None:
        logger.debug("related images result (no receivers headless): %s", args)


class NullRelatedImagesSignals:
    """Stands in for the Qt signal bridge AppActions creates lazily.

    Seeded into the action dict up front so AppActions.related_images_signals()
    finds it and never reaches its lazy import of the QObject version.
    """

    def __init__(self) -> None:
        self.result = _NullSignal()


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------

def _message_stub(name: str) -> Callable[..., Any]:
    def _log_message(*args, **kwargs):
        text = args[0] if args else kwargs.get("message", "")
        if name == "_alert":
            title = args[0] if args else kwargs.get("title", "")
            body = args[1] if len(args) > 1 else kwargs.get("message", "")
            logger.warning('Alert - Title: "%s" Message: %s', title, body)
            # The contract is that callers may write `if not result:` for every
            # button mode, with False meaning rejected. Returning False is
            # therefore the answer that declines rather than silently consents.
            return False
        logger.info("%s: %s", name, text)
        return None
    return _log_message


def _noop_stub(name: str) -> Callable[..., Any]:
    def _noop(*_args, **_kwargs):
        logger.debug("no-op display action: %s", name)
        return None
    return _noop


def _neutral_stub(name: str, value: Any) -> Callable[..., Any]:
    def _neutral(*_args, **_kwargs):
        logger.debug("neutral return for %s: %r", name, value)
        return list(value) if isinstance(value, tuple) else value
    return _neutral


def _unavailable_stub(name: str) -> Callable[..., Any]:
    def _unavailable(*_args, **_kwargs):
        raise HeadlessActionUnavailable(
            f"Action '{name}' is not a display operation and has no headless "
            f"default. Pass it to build_headless_app_actions()."
        )
    return _unavailable


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_headless_app_actions(
    domain_actions: Optional[Dict[str, Callable[..., Any]]] = None,
    *,
    master: Optional[object] = None,
) -> AppActions:
    """Build an AppActions usable with no QApplication and no display.

    *domain_actions* supplies the entries in DOMAIN_ACTIONS. Anything omitted
    raises HeadlessActionUnavailable when called, naming the action -- omitting
    one is only a problem for a caller that actually reaches it.

    Passing a name outside the known contract is rejected: it would otherwise
    sit unused in the dict and read as wired up.
    """
    supplied = dict(domain_actions or {})

    known = (
        set(MESSAGE_ACTIONS)
        | set(NOOP_ACTIONS)
        | set(NEUTRAL_RETURN_ACTIONS)
        | set(DOMAIN_ACTIONS)
    )
    unknown = sorted(set(supplied) - known)
    if unknown:
        raise ValueError(
            f"Unknown action names passed to build_headless_app_actions: {unknown}"
        )

    actions: Dict[str, Callable[..., Any]] = {}
    for name in MESSAGE_ACTIONS:
        actions[name] = _message_stub(name)
    for name in NOOP_ACTIONS:
        actions[name] = _noop_stub(name)
    for name, value in NEUTRAL_RETURN_ACTIONS.items():
        actions[name] = _neutral_stub(name, value)
    for name in DOMAIN_ACTIONS:
        actions[name] = supplied.get(name) or _unavailable_stub(name)

    # Overriding a display action is allowed but not the common case; it lets a
    # caller capture toasts or alerts for assertions.
    for name, func in supplied.items():
        actions[name] = func

    # Pre-seeded so the lazy QObject bridge is never constructed.
    actions["_related_images_signals"] = NullRelatedImagesSignals()

    return AppActions(actions=actions, master=master)


def missing_domain_actions(app_actions: AppActions) -> list:
    """Domain actions that would raise if called. Useful as a pre-flight check."""
    missing = []
    for name in DOMAIN_ACTIONS:
        func = app_actions._actions.get(name)
        if getattr(func, "__name__", "") == "_unavailable":
            missing.append(name)
    return missing

"""Preview-and-confirm wrapper for the random image edits.

Random modification and scrambling produce a different result on every run, so
the UI renders one candidate into a temp file, shows it through the shared
preview dialog, and writes it beside the source only once the user accepts. A
reroll runs the op again; a cancel writes nothing.

Only interactive callers come through here: pipeline runs and file-action
replay call ImageOps directly and must stay non-interactive.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable, Optional, Sequence, Tuple

from lib.fill_preview_dialog_qt import show_preview_confirm_dialog
from utils.config import config
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

logger = get_logger("edit_preview")

_PREVIEW_BASENAME = "weidr_op_preview"


def preview_enabled() -> bool:
    """Whether random edits are previewed before being written."""
    return bool(config.preview_random_edits)


def palette_toggle() -> Tuple[str, bool, Callable[[bool], None]]:
    """The "Match image palette" checkbox, for ops that consume a fill palette.

    Its value is session state on FillPalette, shared by every window and
    never persisted; the next start follows the configured value again.
    """
    from image.fill_palette import FillPalette

    return (
        _("Match image palette"),
        FillPalette.match_enabled(),
        FillPalette.set_session_match_enabled,
    )


def _remove_preview(path: str) -> None:
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not remove preview file %s", path)


def preview_and_confirm_op(
    master,
    app_actions,
    source_path: str,
    run_op: Callable[[str], None],
    *,
    suffix: str,
    title: str,
    toggles: Sequence[Tuple[str, bool, Callable[[bool], None]]] = (),
) -> Optional[str]:
    """Render a candidate with *run_op*, confirm it, and save it on accept.

    *run_op* takes the path to write to, and is called again for every reroll --
    each call produces a different candidate, which is the reason for
    previewing at all. On accept the candidate is moved next to *source_path*
    with *suffix*, the same destination the op would have chosen itself.

    Returns the saved path, or None if the user cancelled or nothing rendered.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    ext = os.path.splitext(source_path)[1] or ".png"
    preview_path = os.path.join(tempfile.gettempdir(), _PREVIEW_BASENAME + ext)
    # Clear any leftover from an interrupted run: _render reports success by
    # the file being there, so a stale one would pass for this run's candidate
    # if an op wrote nothing without raising.
    _remove_preview(preview_path)

    def _render() -> bool:
        # Full-image operations -- scramble_image's FFT phase pass most of all
        # -- run on the GUI thread while the dialog is modal.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            run_op(preview_path)
        except Exception as e:
            logger.exception("Preview render failed for %s", source_path)
            app_actions.warn(_("Could not render preview: {0}").format(str(e)))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        return os.path.isfile(preview_path)

    def _reroll() -> None:
        # A failed reroll leaves the previous candidate on screen.
        _render()

    def _rendering(on_toggled: Callable[[bool], None]) -> Callable[[bool], None]:
        """Re-render after a toggle: a rendered result cannot be changed in
        place, so the setting only shows once the op runs again."""
        def _handle(checked: bool) -> None:
            on_toggled(checked)
            _render()
        return _handle

    if not _render():
        return None

    try:
        accepted = show_preview_confirm_dialog(
            master, preview_path, _reroll,
            title=title,
            hint=_("Enter to accept, Escape to cancel, R to reroll a different result"),
            toggles=tuple(
                (label, initial, _rendering(on_toggled))
                for label, initial, on_toggled in toggles
            ),
        )
        if not accepted:
            return None
        final_path = Utils.unique_sibling_path(source_path, suffix)
        # shutil.move, not os.replace: the temp directory is regularly on a
        # different filesystem than the media.
        shutil.move(preview_path, final_path)
        return final_path
    finally:
        _remove_preview(preview_path)

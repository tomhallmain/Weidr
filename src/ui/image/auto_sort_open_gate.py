"""
Confirmation gate for viewing the most recently auto-sorted media.

Kept out of ``ui.image.media_details`` on purpose: this is the only place that
needs to know about both the file-action history and the temp media canvas, and
the gate applies solely to the "view last system-moved media" action -- not to
the other callers of ``open_temp_media_canvas``, which stay ungated.
"""

from __future__ import annotations

from files.auto_sort_confirmation import AutoSortConfirmation
from files.file_action import FileAction
from utils.logging_setup import get_logger
from utils.translations import _

logger = get_logger("auto_sort_open_gate")


def open_last_auto_sorted_media(master, app_actions) -> None:
    """Open the last system-moved file, confirming first for gated categories.

    Declining is a plain no-op: the gate stands before the display, so there is
    nothing to undo when the user says no.
    """
    from ui.image.media_details import MediaDetails

    action = FileAction.get_history_action(auto=True)
    media_path = action.new_files[0] if action is not None and action.new_files else None
    if media_path is None:
        return

    category = AutoSortConfirmation.category_for_target_dir(action.target)
    if AutoSortConfirmation.is_confirm_required(category):
        logger.info(
            f"Confirmation required before viewing auto-sorted media in category "
            f"{category!r}: {media_path}"
        )
        confirmed = app_actions.alert(
            _("Confirm Auto-Sorted Category"),
            _(
                'This file was automatically sorted into "{0}".\n\n'
                "View it now?"
            ).format(category),
            kind="askyesno",
        )
        if not confirmed:
            return

    MediaDetails.open_temp_media_canvas(master, media_path, app_actions)

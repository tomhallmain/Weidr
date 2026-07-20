"""
RelatedImagesWindow -- the related-image action family in one place (PySide6).

Collects the related-image key-chord actions behind labeled buttons with
their chords displayed alongside (doubling as the family's cheat sheet),
plus a persistent result area fed by the related-images result signal on
app_actions — toasts expire, the result area doesn't.

Results arrive through the RelatedImagesResultSignals bridge (see
ui/app_window/related_images_events.py): thread-safe queued delivery for reports from
worker threads, and Qt disconnects the connection automatically when this
window is destroyed — no manual registration lifecycle. The window sets
WA_DeleteOnClose so closing destroys it (each open is a fresh window).

Opened via Ctrl+Shift+Y (which previously ran the mark-all-downstream bulk
action directly; that action remains available as a button here).
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout

from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.translations import _


class RelatedImagesWindow(SmartDialog):
    """Modeless dialog of related-image actions with a rolling result label.

    Buttons invoke the exact same controller methods the key bindings call —
    this window is pure exposure, no action logic of its own. It deliberately
    stays open across actions so their outcomes stay readable in the result
    area after the corresponding toasts have expired.
    """

    _MAX_RESULT_LINES = 3

    def __init__(self, app_window, geometry: str = "620x460") -> None:
        super().__init__(
            parent=app_window,
            position_parent=app_window,
            title=_("Related Images"),
            geometry=geometry,
            respect_title_bar=True,
        )
        self._app = app_window
        self._results: list[str] = []
        # Close destroys the window: Qt then auto-disconnects the result
        # signal (receiver gone) and the next open builds a fresh window.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        header = QLabel(_("Related Image Actions"))
        header.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold; font-size: 12pt;"
        )
        outer.addWidget(header)

        app = self._app
        actions = [
            (_("View related image"), "Shift+R",
             lambda: app.window_launcher.show_related_media()),
            (_("Next downstream image in open window"), "Shift+T",
             lambda: app.search_ctrl.find_related_media_in_open_window()),
            (_("Set marks from downstream images"), "Shift+Y",
             lambda: app.file_marks_ctrl.set_marks_from_downstream_related_images()),
            (_("Mark sources with downstream files in current directory"), "Ctrl+Y",
             lambda: app.file_marks_ctrl.mark_sources_with_downstream_in_dir()),
            (_("Mark all downstream files in current directory"), "—",
             lambda: app.file_marks_ctrl.mark_downstream_files_in_dir()),
            (_("Search all open windows for downstream images"), "Alt+Y",
             lambda: app.file_marks_ctrl.set_marks_from_downstream_related_images_all_windows()),
        ]

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setColumnStretch(0, 4)
        grid.setColumnStretch(1, 1)
        for row, (label, chord, func) in enumerate(actions):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, f=func: f())
            grid.addWidget(btn, row, 0)
            chord_lbl = QLabel(chord)
            chord_lbl.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
            chord_lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(chord_lbl, row, 1)
        outer.addLayout(grid)

        results_header = QLabel(_("Results"))
        results_header.setStyleSheet(
            f"color: {AppStyle.FG_COLOR}; font-weight: bold;"
        )
        outer.addWidget(results_header)

        self._result_label = QLabel(_("(no results yet)"))
        self._result_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._result_label.setWordWrap(True)
        self._result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        outer.addWidget(self._result_label, 1)

        self._app.app_actions.related_images_signals().result.connect(self._on_result)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.close)

    # ------------------------------------------------------------------
    # Result reporting
    # ------------------------------------------------------------------

    def _on_result(self, message: str, action_label=None, data=None) -> None:
        """Append an action outcome to the rolling result area.

        Delivered via the RelatedImagesResultSignals bridge — deferred flows
        (actions that detour through the recent-directory picker) report long
        after the button handler returned, and land here the same way; Qt
        queues cross-thread reports onto the main thread and drops the
        connection when this window is destroyed.
        """
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}]  {action_label}\n{message}" if action_label else f"[{stamp}]  {message}"
        self._results.insert(0, line)
        del self._results[self._MAX_RESULT_LINES:]
        self._result_label.setText("\n\n".join(self._results))

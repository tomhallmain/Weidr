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

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout

from lib.multi_display_qt import SmartDialog
from ui.app_style import AppStyle
from utils.translations import _, format_shortcut
from utils.utils import Utils


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

        # The window that opened this dialog is where "current media" comes
        # from for every action below — name its directory so that's never
        # ambiguous when several windows are open. get_base_dir is a required
        # AppActions action, so this is the stable API for it.
        source_dir = app_window.app_actions.get_base_dir() or _("(no directory)")
        self._source_label = QLabel(_("Current media source: {0}").format(source_dir))
        self._source_label.setStyleSheet(f"color: {AppStyle.FG_COLOR};")
        self._source_label.setWordWrap(True)
        self._source_label.setToolTip(
            _("Actions in this window use the current media of the window "
              "that opened it.")
        )
        outer.addWidget(self._source_label)

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
            (_("Mark files without a related image in current directory"), "—",
             lambda: app.file_marks_ctrl.mark_files_without_related_images_in_dir()),
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
            chord_lbl = QLabel(format_shortcut(chord))
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
    _MAX_LISTED_FILES = 3

    @staticmethod
    def _outcome_label(outcome: str) -> str:
        """Translated display text for an outcome token.

        The tokens themselves are internal identifiers (see
        AppActions.report_related_images), so they never reach the user
        untranslated; an unrecognised one falls back to its own text rather
        than being dropped.
        """
        return {
            "blocked": _("cancelled"),
            "deferred": _("waiting"),
            "no_media": _("no media"),
            "no_files": _("no files"),
            "too_many_files": _("too many files"),
        }.get(outcome, outcome)

    @staticmethod
    def _mechanism_label(mechanism: str) -> str:
        """Translated display text for a match-mechanism token.

        Internal identifiers, as with the outcome tokens -- an unrecognised
        one falls back to its own text.
        """
        return {
            "metadata": _("metadata"),
            "basename": _("filename"),
            "stem": _("name prefix"),
        }.get(mechanism, mechanism)

    @staticmethod
    def _short_dir(directory: str) -> str:
        return Utils.get_relative_dirpath(directory, levels=2)

    @classmethod
    def _format_detail(cls, data) -> list:
        """Render the structured payload from AppActions.report_related_images
        as indented detail lines under the headline.

        Every key is optional; each block is skipped when its key is absent,
        so an action reports only what it actually knows.
        """
        if not isinstance(data, dict):
            return []
        lines = []

        source = data.get("source")
        if source:
            lines.append(_("source: {0}").format(os.path.basename(source)))

        found_by_dir = data.get("found_by_dir")
        searched_dirs = data.get("searched_dirs")
        base_dir = data.get("base_dir")
        if found_by_dir:
            lines.append("   ".join(
                f"{cls._short_dir(d)}: {n}" for d, n in found_by_dir.items()
            ))
        elif searched_dirs:
            lines.append(_("searched {0} director(ies)").format(len(searched_dirs)))
        elif base_dir:
            lines.append(_("in {0}").format(cls._short_dir(base_dir)))

        scanned = data.get("scanned")
        found = data.get("found")
        if scanned is not None:
            if found is not None:
                lines.append(_("{0} of {1} file(s) scanned").format(found, scanned))
            else:
                lines.append(_("{0} file(s) scanned").format(scanned))

        position, total = data.get("position"), data.get("total")
        if position is not None and total is not None:
            lines.append(_("{0} of {1} in cycle").format(position, total))

        by_mechanism = data.get("by_mechanism")
        if by_mechanism:
            lines.append(_("matched by: ") + ", ".join(
                f"{cls._mechanism_label(k)} {n}" for k, n in by_mechanism.items()
            ))

        if data.get("cached"):
            lines.append(_("from a cached scan"))

        files = data.get("files")
        if files:
            shown = ", ".join(os.path.basename(f) for f in files[:cls._MAX_LISTED_FILES])
            remaining = len(files) - cls._MAX_LISTED_FILES
            if remaining > 0:
                shown += _(", and {0} more").format(remaining)
            lines.append(shown)

        skipped_dirs = data.get("skipped_dirs")
        if skipped_dirs:
            lines.append(_("{0} large unconfirmed director(ies) skipped").format(
                len(skipped_dirs)))

        return lines

    def _on_result(self, message: str, action_label=None, data=None) -> None:
        """Append an action outcome to the rolling result area.

        Delivered via the RelatedImagesResultSignals bridge — deferred flows
        (actions that detour through the recent-directory picker) report long
        after the button handler returned, and land here the same way; Qt
        queues cross-thread reports onto the main thread and drops the
        connection when this window is destroyed.
        """
        stamp = time.strftime("%H:%M:%S")
        outcome = data.get("outcome") if isinstance(data, dict) else None
        header = action_label
        if outcome:
            tag = f"({self._outcome_label(outcome)})"
            header = f"{tag} {action_label}" if action_label else tag
        parts = [f"[{stamp}]  {header}\n{message}" if header else f"[{stamp}]  {message}"]
        parts.extend("    " + detail for detail in self._format_detail(data))
        self._results.insert(0, "\n".join(parts))
        del self._results[self._MAX_RESULT_LINES:]
        self._result_label.setText("\n\n".join(self._results))

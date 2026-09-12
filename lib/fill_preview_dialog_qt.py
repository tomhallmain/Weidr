"""
Preview-and-confirm dialog for PySide6.

Shows a candidate result already rendered to a file and lets the user reroll
(regenerate it and refresh the preview) before accepting or cancelling --
without a full settings window. Used for box / background-box fills and for
the random image edits (modify / scramble), where every run produces something
different and seeing one candidate before it is written is the point.

Callers supply the title, the hint line, any extra buttons and any checkboxes;
nothing here interprets them, so this module stays free of app domain
knowledge like the rest of lib/.
"""

from typing import Callable, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from utils.translations import _

_MAX_PREVIEW_FRACTION = 0.7  # fraction of available screen size used as a cap


def show_preview_confirm_dialog(
    master: Optional[QWidget],
    preview_path: str,
    on_reroll: Callable[[], None],
    *,
    title: str,
    hint: str,
    extra_buttons: Sequence[Tuple[str, Callable[[], None]]] = (),
    toggles: Sequence[Tuple[str, bool, Callable[[bool], None]]] = (),
) -> bool:
    """
    Show *preview_path* -- which must already exist and hold the candidate
    result -- with the option to reroll, accept or cancel.

    *on_reroll* regenerates the candidate and rewrites *preview_path* in place.
    *extra_buttons* are ``(label, callback)`` pairs placed before Cancel/Accept;
    *toggles* are ``(label, initial_value, on_toggled)`` triples shown as
    checkboxes above the button row. The preview is reloaded after a reroll, an
    extra button or a toggle: none of them can change an already-rendered
    result in place, so each one re-renders.

    Returns True if the user accepted, False if they cancelled.
    """
    dialog = QDialog(master)
    dialog.setWindowTitle(title)
    dialog.setModal(True)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(10)

    image_label = QLabel()
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(image_label)

    hint_label = QLabel(hint)
    hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(hint_label)

    def _load_preview() -> None:
        pixmap = QPixmap(preview_path)
        if pixmap.isNull():
            return
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            max_w = int(available.width() * _MAX_PREVIEW_FRACTION)
            max_h = int(available.height() * _MAX_PREVIEW_FRACTION)
        else:
            max_w, max_h = 800, 600
        scaled = pixmap.scaled(
            max_w, max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        image_label.setPixmap(scaled)
        dialog.adjustSize()

    _load_preview()

    _accepted = [False]

    def _handle_reroll() -> None:
        on_reroll()
        _load_preview()

    def _handle_accept() -> None:
        _accepted[0] = True
        dialog.accept()

    def _handle_cancel() -> None:
        _accepted[0] = False
        dialog.reject()

    for toggle_label, toggle_initial, on_toggled in toggles:
        checkbox = QCheckBox(toggle_label)
        checkbox.setChecked(bool(toggle_initial))

        # callback bound as a default argument: the loop variable would
        # otherwise be read at click time, when it holds the last toggle.
        def _handle_toggled(checked: bool, callback=on_toggled) -> None:
            callback(bool(checked))
            _load_preview()

        checkbox.toggled.connect(_handle_toggled)
        layout.addWidget(checkbox)

    btn_layout = QHBoxLayout()
    btn_layout.addStretch()

    reroll_btn = QPushButton(_("Reroll"))
    reroll_btn.clicked.connect(_handle_reroll)
    btn_layout.addWidget(reroll_btn)

    for button_label, on_clicked in extra_buttons:
        extra_btn = QPushButton(button_label)

        def _handle_extra(_checked=False, callback=on_clicked) -> None:
            callback()
            _load_preview()

        extra_btn.clicked.connect(_handle_extra)
        btn_layout.addWidget(extra_btn)

    cancel_btn = QPushButton(_("Cancel"))
    cancel_btn.clicked.connect(_handle_cancel)
    btn_layout.addWidget(cancel_btn)

    accept_btn = QPushButton(_("Accept"))
    accept_btn.setDefault(True)
    accept_btn.clicked.connect(_handle_accept)
    btn_layout.addWidget(accept_btn)

    layout.addLayout(btn_layout)

    # Escape-to-cancel and Enter/Return-to-accept are already QDialog/QPushButton
    # defaults (reject() on Escape, click the default button on Enter); only
    # Reroll needs an explicit shortcut.
    QShortcut(QKeySequence("R"), dialog, activated=_handle_reroll)

    dialog.adjustSize()
    dialog.exec()
    return _accepted[0]


def show_fill_preview_dialog(
    master: Optional[QWidget],
    preview_path: str,
    on_reroll: Callable[[], None],
    on_solid: Callable[[tuple], None],
    toggles: Sequence[Tuple[str, bool, Callable[[bool], None]]] = (),
) -> bool:
    """
    The fill flavour of the dialog: adds plain black/white fills to the
    reroll/accept/cancel set.

    *on_solid* regenerates the fill with a plain ``(r, g, b)`` colour instead
    of a fresh random one, rewriting *preview_path* the way *on_reroll* does.
    A toggle's callback must re-render too, for the same reason.
    """
    return show_preview_confirm_dialog(
        master, preview_path, on_reroll,
        title=_("Preview Fill"),
        hint=_("Enter to accept, Escape to cancel, R to reroll a different fill"),
        extra_buttons=(
            (_("Black"), lambda: on_solid((0, 0, 0))),
            (_("White"), lambda: on_solid((255, 255, 255))),
        ),
        toggles=toggles,
    )

"""
Fill-preview confirmation dialog for PySide6.

Shows a candidate box / background-box fill already composited into the
target image, letting the user reroll (regenerate the random fill and
refresh the preview) before accepting or cancelling -- without a full
settings window.
"""

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from utils.translations import _

_MAX_PREVIEW_FRACTION = 0.7  # fraction of available screen size used as a cap


def show_fill_preview_dialog(
    master: Optional[QWidget],
    preview_path: str,
    on_reroll: Callable[[], None],
    on_solid: Callable[[tuple], None],
) -> bool:
    """
    Show a preview of the fill about to be painted into an image, with the
    option to reroll (regenerate the fill and refresh the preview), switch to
    a plain black or white fill, or accept/cancel.

    *preview_path* must already exist and contain the candidate result.
    *on_reroll* regenerates the fill and rewrites *preview_path* in place;
    *on_solid* does the same but with a plain ``(r, g, b)`` color instead of a
    fresh random fill. This function reloads *preview_path* after either.

    Returns True if the user accepted, False if they cancelled.
    """
    dialog = QDialog(master)
    dialog.setWindowTitle(_("Preview Fill"))
    dialog.setModal(True)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(15, 15, 15, 15)
    layout.setSpacing(10)

    image_label = QLabel()
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(image_label)

    hint_label = QLabel(_("Enter to accept, Escape to cancel, R to reroll a different fill"))
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

    def _handle_black() -> None:
        on_solid((0, 0, 0))
        _load_preview()

    def _handle_white() -> None:
        on_solid((255, 255, 255))
        _load_preview()

    def _handle_accept() -> None:
        _accepted[0] = True
        dialog.accept()

    def _handle_cancel() -> None:
        _accepted[0] = False
        dialog.reject()

    btn_layout = QHBoxLayout()
    btn_layout.addStretch()

    reroll_btn = QPushButton(_("Reroll"))
    reroll_btn.clicked.connect(_handle_reroll)
    btn_layout.addWidget(reroll_btn)

    black_btn = QPushButton(_("Black"))
    black_btn.clicked.connect(_handle_black)
    btn_layout.addWidget(black_btn)

    white_btn = QPushButton(_("White"))
    white_btn.clicked.connect(_handle_white)
    btn_layout.addWidget(white_btn)

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

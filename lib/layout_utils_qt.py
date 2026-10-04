"""Small helpers for Qt layouts."""

from PySide6.QtWidgets import QLayout


def clear_layout(layout: QLayout) -> None:
    """Remove every item from *layout*, deleting its widgets and emptying
    nested layouts."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        sub = item.layout()
        if sub is not None:
            clear_layout(sub)

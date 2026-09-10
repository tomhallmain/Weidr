"""
Which auto-sorted categories require confirmation before their media is opened.

An automatic move records its destination directory on the FileAction, and for
classifier-driven sorts that directory's basename *is* the category the file was
sorted into. This holds the user's set of categories that should prompt before
the sorted file is displayed.

Categories are matched case-insensitively and stored case-folded; no other
normalization is applied.
"""

import os
from typing import Optional

from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger

logger = get_logger("auto_sort_confirmation")


class AutoSortConfirmation:

    CONFIRM_CATEGORIES_KEY = "auto_sort_confirm_categories"

    confirm_categories: set[str] = set()

    @staticmethod
    def _normalize(category: Optional[str]) -> str:
        if not category or not isinstance(category, str):
            return ""
        return category.casefold()

    @staticmethod
    def load() -> None:
        stored = app_info_cache.get_meta(
            AutoSortConfirmation.CONFIRM_CATEGORIES_KEY, default_val=[]
        )
        if not isinstance(stored, list):
            stored = []
        AutoSortConfirmation.confirm_categories = {
            normalized
            for normalized in (AutoSortConfirmation._normalize(c) for c in stored)
            if normalized
        }

    @staticmethod
    def save() -> None:
        app_info_cache.set_meta(
            AutoSortConfirmation.CONFIRM_CATEGORIES_KEY,
            sorted(AutoSortConfirmation.confirm_categories),
        )

    @staticmethod
    def is_confirm_required(category: Optional[str]) -> bool:
        normalized = AutoSortConfirmation._normalize(category)
        if normalized == "":
            return False
        return normalized in AutoSortConfirmation.confirm_categories

    @staticmethod
    def set_confirm_required(category: str, required: bool) -> None:
        normalized = AutoSortConfirmation._normalize(category)
        if normalized == "":
            return
        if required:
            AutoSortConfirmation.confirm_categories.add(normalized)
        else:
            AutoSortConfirmation.confirm_categories.discard(normalized)

    @staticmethod
    def set_categories(categories) -> None:
        """Replace the whole set, e.g. from the editor's freeform text field."""
        AutoSortConfirmation.confirm_categories = {
            normalized
            for normalized in (AutoSortConfirmation._normalize(c) for c in categories)
            if normalized
        }

    @staticmethod
    def get_categories() -> list[str]:
        return sorted(AutoSortConfirmation.confirm_categories)

    @staticmethod
    def category_for_target_dir(target_dir: Optional[str]) -> Optional[str]:
        """The category an auto-sorted file landed in, i.e. its target's basename."""
        if not target_dir:
            return None
        return os.path.basename(os.path.normpath(target_dir)) or None

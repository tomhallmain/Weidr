import os

from utils.app_info_cache import app_info_cache
from utils.logging_setup import get_logger

logger = get_logger("target_directories")


class TargetDirectories:
    """
    Non-UI data management for target directories used in related-file searches.

    This is separate from ``RecentDirectories`` (which tracks the main app's
    browsing directories).  Persisted under a distinct cache key so the two
    lists do not interfere with each other.
    """

    recent_directories: list[str] = []
    MAX_RECENT_DIRECTORIES = 50
    RECENT_DIRECTORIES_KEY = "target_directory_window.recent_directories"

    @staticmethod
    def load_recent_directories() -> None:
        """Load recent directories from app cache."""
        dirs = app_info_cache.get_meta(
            TargetDirectories.RECENT_DIRECTORIES_KEY, default_val=[]
        )
        if not isinstance(dirs, list):
            dirs = []
        # Filter out any paths that are no longer valid directories
        filtered_dirs = [
            os.path.normpath(d) for d in dirs
            if isinstance(d, str) and os.path.isdir(d)
        ]
        TargetDirectories.recent_directories = filtered_dirs
        # Persist the filtered list back so stale entries are removed
        if filtered_dirs != dirs:
            app_info_cache.set_meta(
                TargetDirectories.RECENT_DIRECTORIES_KEY, filtered_dirs
            )

    @staticmethod
    def save_recent_directories() -> None:
        """Save recent directories to app cache."""
        app_info_cache.set_meta(
            TargetDirectories.RECENT_DIRECTORIES_KEY,
            TargetDirectories.recent_directories,
        )

    @staticmethod
    def add_recent_directory(directory: str) -> None:
        """Add a directory to the recent list (most recent first)."""
        if not directory or not os.path.isdir(directory):
            return

        normalized_dir = os.path.normpath(os.path.abspath(directory))

        # Remove if already exists
        if normalized_dir in TargetDirectories.recent_directories:
            TargetDirectories.recent_directories.remove(normalized_dir)

        # Add to beginning
        TargetDirectories.recent_directories.insert(0, normalized_dir)

        # Enforce maximum limit
        if len(TargetDirectories.recent_directories) > TargetDirectories.MAX_RECENT_DIRECTORIES:
            TargetDirectories.recent_directories = (
                TargetDirectories.recent_directories[:TargetDirectories.MAX_RECENT_DIRECTORIES]
            )

        TargetDirectories.save_recent_directories()

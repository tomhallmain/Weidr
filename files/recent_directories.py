import os
from typing import Optional

from utils.app_info_cache import app_info_cache
from utils.translations import _
from utils.utils import Utils
from utils.logging_setup import get_logger

logger = get_logger("recent_directories")


class RecentDirectories:
    directories: list[str] = []
    directory_history: list[str] = []

    MAX_RECENT_DIRECTORIES = 100

    last_set_directory: Optional[str] = None
    last_comparison_directory: Optional[str] = None

    @staticmethod
    def store_recent_directories():
        app_info_cache.set_meta("recent_directories", RecentDirectories.directories)

    @staticmethod
    def load_recent_directories():
        dirs = app_info_cache.get_meta("recent_directories", default_val=[])
        if not isinstance(dirs, list):
            dirs = []
        # Filter out any paths that are no longer valid directories
        filtered_dirs = [os.path.normpath(d) for d in dirs if isinstance(d, str) and os.path.isdir(d)]
        RecentDirectories.directories = filtered_dirs
        # Persist the filtered list back into the cache so stale entries are removed
        if filtered_dirs != dirs:
            app_info_cache.set_meta("recent_directories", filtered_dirs)

    @staticmethod
    def set_recent_directories(directories):
        RecentDirectories.directories = list(directories)

    @staticmethod
    def set_recent_directory(_dir=dir):
        if len(RecentDirectories.directories) > 0:
            if RecentDirectories.directories[0] == _dir:
                return
            if _dir in RecentDirectories.directories:
                RecentDirectories.directories.remove(_dir)
        RecentDirectories.directories.insert(0, _dir)
        # Enforce the maximum limit by removing excess directories from the end
        if len(RecentDirectories.directories) > RecentDirectories.MAX_RECENT_DIRECTORIES:
            RecentDirectories.directories = RecentDirectories.directories[:RecentDirectories.MAX_RECENT_DIRECTORIES]

    @staticmethod
    def remove_directory(directory: str) -> None:
        """Remove a directory from all recent directory caches and persist the changes."""
        try:
            # Remove from main recent directories list
            if directory in RecentDirectories.directories:
                RecentDirectories.directories.remove(directory)
                app_info_cache.set_meta("recent_directories", RecentDirectories.directories)
            
            # Remove from in-memory history trackers
            try:
                RecentDirectories.directory_history = [d for d in RecentDirectories.directory_history if d != directory]
                if RecentDirectories.last_comparison_directory == directory:
                    RecentDirectories.last_comparison_directory = None
                if RecentDirectories.last_set_directory == directory:
                    RecentDirectories.last_set_directory = None
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Error updating recent directories during delete: {e}")

    @staticmethod
    def find_replacement_directory(current_base_dir: str, open_window_directories: list[str]) -> str:
        """
        Find a valid replacement directory from recent directories that is not currently open.
        Returns the most recent valid directory, or home directory if none found.
        
        Args:
            current_base_dir: The directory being deleted (to exclude from consideration)
            open_window_directories: List of directories currently open in other windows
            
        Returns:
            A valid directory path to use as replacement
        """
        # Quick validation - check if current directory should be deletable
        if current_base_dir in open_window_directories:
            # This directory is open in another window, shouldn't delete
            raise ValueError(f"Directory {current_base_dir} is currently open in another window and cannot be deleted")
        
        # Look for a valid replacement from recent directories
        for directory in RecentDirectories.directories:
            if (directory != current_base_dir and 
                directory not in open_window_directories and 
                os.path.isdir(directory)):
                return directory
        
        # If no recent directory is suitable, fall back to home directory
        try:
            home_dir = Utils.get_home_directory()
            if os.path.isdir(home_dir):
                return home_dir
        except Exception as e:
            logger.error(f"Error getting home directory: {e}")
        
        # Final fallback - use current working directory
        try:
            cwd = os.getcwd()
            if os.path.isdir(cwd) and cwd != current_base_dir:
                return cwd
        except Exception as e:
            logger.error(f"Error getting current working directory: {e}")
        
        # If all else fails, raise an error
        raise ValueError("No suitable replacement directory found")
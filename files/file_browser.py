import fnmatch
import glob
import json
import os
from random import choice, randint, shuffle
import re
import threading
from time import sleep
from typing import Dict, List, Optional

from files.sortable_file import SortableFile
from utils.config import config
from utils.constants import Sort, SortBy, Direction
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils
logger = get_logger("file_browser")


class FileBrowser:
    have_confirmed_directories: List[str] = []

    def __init__(self, directory: str = ".", recursive: bool = False,
                 filter: Optional[str] = None, sort_by: SortBy = SortBy.NAME) -> None:
        self.directory = directory
        self.recursive = recursive
        self.filter = filter
        self._files_cache = {}
        self._files = []
        self._new_files = []
        self.filepaths = []
        self.sort_by = sort_by
        self.sort = Sort.ASC
        self.file_cursor = -1
        self.cursor_lock = threading.Lock()
        self.checking_files = False
        self.use_file_paths_json = config.use_file_paths_json
        self._preferred_initial_file: Optional[str] = None
        self.is_incremental_loading = False
        self._incremental_thread: Optional[threading.Thread] = None
        self._incremental_stop_event = threading.Event()
        self._incremental_scanned_dirs = 0
        self._incremental_files_discovered = 0
        self._incremental_batch_size = 200
        self._use_incremental_on_full_refresh = False
        # One-shot arming: incremental loading should run after explicit
        # directory/recurse changes, not on every later refresh.
        self._incremental_refresh_armed = False
        self._is_external_drive_directory = Utils.is_external_drive(self.directory)

    def has_files(self) -> bool:
        return len(self._files) > 0

    def has_file(self, _file: str) -> bool:
        return _file in self.filepaths

    def count(self) -> int:
        return len(self._files)

    def is_slow_total_files(self, threshold: int = 5000, use_sortable_files: bool = False) -> bool:
        factor = 5 if self._is_external_drive_directory else 1
        file_count = len(self._files) if use_sortable_files else len(self.filepaths)
        return factor * file_count > threshold

    def has_confirmed_dir(self) -> bool:
        return self.directory in FileBrowser.have_confirmed_directories

    def set_dir_confirmed(self) -> None:
        if not self.has_confirmed_dir():
            FileBrowser.have_confirmed_directories.append(self.directory)

    def set_filter(self, filter: str) -> None:
        if config.debug:
            logger.debug(f"File browser set filter: {filter}")
        self.filter = filter

    @staticmethod
    def _parse_filter(raw: str) -> tuple:
        """Split raw filter string into (inclusions, exclusions) term lists.

        Terms are separated by ';'. A term prefixed with '!' is an exclusion;
        the rest are inclusions. Strips whitespace from each term.
        """
        terms = [t.strip() for t in raw.split(";") if t.strip()]
        inclusions = [t for t in terms if not t.startswith("!")]
        exclusions = [t[1:].strip() for t in terms if t.startswith("!") and t[1:].strip()]
        return inclusions, exclusions

    def set_recursive(self, recursive: bool) -> None:
        if config.debug:
            logger.debug(f"File browser set recursive: {recursive}")
        self.recursive = recursive
        self._recompute_incremental_decision()
        self._incremental_refresh_armed = self._use_incremental_on_full_refresh
        self.refresh()

    def is_recursive(self) -> bool:
        return self.recursive

    def get_cursor(self) -> int:
        with self.cursor_lock:
            return self.file_cursor

    def refresh(self, refresh_cursor: bool = True, file_check: bool = False,
                removed_files: List[str] = [], direction: Direction = Direction.FORWARD) -> List[str]:
        last_files = self.get_files() if file_check else []
        if config.use_file_paths_json:
            self.update_json_for_removed_files(removed_files)
        if refresh_cursor:
            with self.cursor_lock:
                self.file_cursor = direction.get_correction(backward_value=1)
        current_file = self.current_file() if file_check else None
        self.filepaths = []
        self._stop_incremental_load()
        if self._should_use_incremental_load(file_check=file_check, removed_files=removed_files):
            self._start_incremental_load()
        else:
            self._get_sortable_files()
        files = self.get_files()
        self.checking_files = len(files) > 0 and len(files) < config.file_check_skip_if_n_files_over # Avoid rechecking in directories with many files
        if file_check and current_file and os.path.isfile(current_file) and current_file in files:
            with self.cursor_lock:
                self.file_cursor = files.index(current_file)
                if len(removed_files) > 0:
                    if self.file_cursor > -1:
                        self.file_cursor += direction.get_correction()
            self._new_files = list(set(files) - set(last_files))
        elif file_check and current_file and os.path.isfile(current_file):
            # current_file still exists on disk but was filtered out (e.g. its
            # media type was just disabled in configuration) -- fall back to a
            # bounds-safe cursor instead of crashing on files.index().
            with self.cursor_lock:
                if len(files) - 1 < self.file_cursor:
                    self.file_cursor = direction.get_correction()
                else:
                    self.file_cursor += direction.get_correction()
            self._new_files = list(set(files) - set(last_files))
        elif not refresh_cursor:
            with self.cursor_lock:
                if len(files) - 1 < self.file_cursor:
                    self.file_cursor = direction.get_correction()
                else:
                    self.file_cursor += direction.get_correction()
        return files

    def update_cursor_to_new_media(self) -> bool:
        if len(self._new_files) == 0:
            return False
        with self.cursor_lock:
            self.file_cursor = self.filepaths.index(self._new_files[0]) - 1
        return True

    def set_directory(self, directory: str) -> List[str]:
        self.directory = directory
        self._update_drive_characteristics()
        self.checking_files = True
        self._files_cache = {}
        self._preferred_initial_file = None
        self._recompute_incremental_decision()
        self._incremental_refresh_armed = self._use_incremental_on_full_refresh
        logger.info(f"Setting base directory: {directory}")
        return self.refresh()

    def set_directory_with_preferred_file(
        self, directory: str, preferred_file: Optional[str] = None
    ) -> List[str]:
        self.directory = directory
        self._update_drive_characteristics()
        self.checking_files = True
        self._files_cache = {}
        self._preferred_initial_file = preferred_file
        self._recompute_incremental_decision()
        self._incremental_refresh_armed = self._use_incremental_on_full_refresh
        logger.info(f"Setting base directory: {directory}")
        return self.refresh()

    def get_sort_by(self) -> SortBy:
        return self.sort_by

    def set_sort_by(self, sort_by: SortBy) -> List[str]:
        self.sort_by = sort_by
        if self.sort_by == SortBy.RANDOMIZE:
            self.sort = Sort.RANDOM
        self.refresh()
        return self.get_files()

    def set_sort(self, sort: Sort) -> List[str]:
        self.sort = sort
        return self.get_files()

    def current_file(self) -> Optional[str]:
        if self.has_files():
            try:
                cursor = 0
                with self.cursor_lock:
                    cursor = self.file_cursor
                return self.get_files()[cursor]
            except Exception:
                with self.cursor_lock:
                    self.file_cursor = 0
                    return self.get_files()[self.file_cursor]
        else:
            return None

    def previous_file(self) -> str:
        files = self.get_files()
        if len(files) == 0:
            recursive_str = "" if self.recursive else _(" (try setting recursive to True)")
            raise Exception(_("No files found for current browsing settings.") + recursive_str)
        with self.cursor_lock:
            if self.file_cursor == 0:
                self.file_cursor = len(files) - 1
            else:
                self.file_cursor -= 1
            return files[self.file_cursor]

    def next_file(self) -> str:
        files = self.get_files()
        if len(files) == 0:
            recursive_str = "" if self.recursive else _(" (try setting recursive to True)")
            raise Exception(_("No files found for current browsing settings.") + recursive_str)
        with self.cursor_lock:
            if len(files) > self.file_cursor + 1:
                self.file_cursor += 1
            else:
                self.file_cursor = 0
        return files[self.file_cursor]

    def last_file(self) -> str:
        files = self.get_files()
        if len(files) == 0:
            recursive_str = "" if self.recursive else _(" (try setting recursive to True)")
            raise Exception(_("No files found for current browsing settings.") + recursive_str)
        with self.cursor_lock:
            self.file_cursor = len(files) - 1
        return files[self.file_cursor]

    def load_file_paths_json(self) -> List[str]:
        logger.info(f"Loading external file paths from JSON: {config.file_paths_json_path}")
        try:
            with open(config.file_paths_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load file paths JSON from %s: %s", config.file_paths_json_path, e)
            raise

    def update_json_for_removed_files(self, removed_file_paths: List[str] = []) -> None:
        if len(removed_file_paths) == 0:
            return

        files = list(self.get_files())
        for removed_filepath in removed_file_paths:
            files.remove(removed_filepath)

        with open(config.file_paths_json_path,"w") as f:
            json.dump(files, f, indent=4)
            logger.info(f"JSON file updated: {config.file_paths_json_path}")

    def get_index_details(self):
        files = self.get_files()
        return _("FILE_BROWSER_INDEX_DETAILS").format(self.file_cursor+1, len(files), self.sort_by.get_text(), self.sort.get_text())

    def go_to_file(self, filepath: str) -> None:
        files = self.get_files()
        if filepath in files:
            self.file_cursor = files.index(filepath)

    def go_to_index(self, index: int) -> Optional[str]:
        """
        Go to file at the specified index (1-based).
        Returns the file path if successful, None if index is invalid.
        """
        files = self.get_files()
        if len(files) == 0:
            raise Exception(_("No files found for current browsing settings."))
        
        # Convert 1-based index to 0-based
        if index < 1:
            raise ValueError(_("Index must be 1 or greater."))
        
        zero_based_index = index - 1
        if zero_based_index >= len(files):
            raise ValueError(_("Index {0} is out of range. There are {1} files.").format(index, len(files)))
        
        with self.cursor_lock:
            self.file_cursor = zero_based_index
        
        return files[self.file_cursor]

    def random_file(self) -> str:
        files = self.get_files()
        random_file = choice(files)
        self.file_cursor = files.index(random_file)
        return random_file

    def select_series(self, start_file: str, end_file: str) -> List[str]:
        files = self.get_files()
        selected = []
        if start_file in files and end_file in files:
            start_index = files.index(start_file)
            end_index = files.index(end_file)
            if start_index > end_index:
                selected.extend(files[end_index:start_index+1])
            else:
                selected.extend(files[start_index:end_index+1])
        return selected

    def find(self, search_text: Optional[str] = None, retry_with_delay: int = 0,
             exact_match: bool = False, closest_sort_by: Optional[SortBy] = None) -> Optional[str]:
        if not search_text or search_text.strip() == "":
            raise Exception(_("Search text provided to file_browser.find() was invalid."))
        files = self.get_files_with_retry(retry_with_delay)
        # First try to match filename
        if search_text in files:
            self.file_cursor = files.index(search_text)
            if config.debug:
                logger.debug(f"Index of {search_text}: {self.file_cursor}")
            return search_text
        filenames = [os.path.basename(f) for f in files]
        if search_text in filenames:
            self.file_cursor = filenames.index(search_text)
            if config.debug:
                logger.debug(f"Index of {search_text}: {self.file_cursor}")
            return files[self.file_cursor]
        if exact_match and closest_sort_by is None:
            return None
        search_text = search_text.lower()
        # If that fails, match string to the start of file name
        for i in range(len(filenames)):
            filename = filenames[i]
            if filename.lower().startswith(search_text):
                if config.debug:
                    logger.debug(f"Index of {filename}: {i}")
                self.file_cursor = i
                return files[self.file_cursor]
        # Finally try to match string anywhere within file name
        for i in range(len(filenames)):
            filename = files[i]
            if search_text in filename:
                if config.debug:
                    logger.debug(f"Index of {filename}: {i}")
                self.file_cursor = i
                return files[self.file_cursor]
        
        # If no match found and closest_sort_by is specified, find the closest file based on specified sorting
        if closest_sort_by is not None and len(files) > 0:
            return self._find_closest_file_by_position(search_text, files, closest_sort_by)
        
        return None

    def _find_closest_file_by_position(self, search_text: str, files: List[str], closest_sort_by: SortBy) -> Optional[str]:
        """
        Find the closest file to the search text based on positional distance in a sorted list.
        This finds files that would be positioned closest to the search text in the specified sort order.
        """
        if not files:
            return None
        
        if config.debug:
            logger.debug(f"Finding closest file by position: {search_text}, closest sort by: {closest_sort_by}")
        
        # Create a properly sorted list for the requested sort type
        sorted_files = self._get_sorted_files_for_sort_type(files, closest_sort_by)
        if not sorted_files:
            return None
        
        # Route to appropriate handler based on sort type
        if closest_sort_by in [SortBy.NAME, SortBy.FULL_PATH]:
            return self._find_closest_by_name(search_text, sorted_files, files)
        elif closest_sort_by in [SortBy.CREATION_TIME, SortBy.MODIFY_TIME]:
            return self._find_closest_by_time(search_text, sorted_files, files, closest_sort_by)
        elif closest_sort_by == SortBy.SIZE:
            return self._find_closest_by_size(search_text, sorted_files, files)
        elif closest_sort_by == SortBy.TYPE:
            return self._find_closest_by_type(search_text, sorted_files, files)
        elif closest_sort_by == SortBy.NAME_LENGTH:
            return self._find_closest_by_name_length(search_text, sorted_files, files)
        elif closest_sort_by in [SortBy.IMAGE_PIXELS, SortBy.IMAGE_HEIGHT, SortBy.IMAGE_WIDTH]:
            return self._find_closest_by_image_property(search_text, sorted_files, files, closest_sort_by)
        elif closest_sort_by == SortBy.RELATED_IMAGE:
            return self._find_closest_by_related_image(search_text, sorted_files, files)
        elif closest_sort_by == SortBy.RANDOMIZE:
            return self._find_closest_by_random(search_text, sorted_files, files)
        elif closest_sort_by == SortBy.SUFFIX:
            return self._find_closest_by_suffix(search_text, sorted_files, files)
        else:
            # Unknown sort type, return first file
            return self._handle_find_closest_failure_message(sorted_files, files, "unknown sort type", f"sort type {closest_sort_by}")

    def _get_sorted_files_for_sort_type(self, files: List[str], sort_by: SortBy) -> List[SortableFile]:
        """
        Create a properly sorted list of SortableFile objects for the specified sort type.
        This ensures we're working with the correct sort order for closest file finding
        and avoids re-creating SortableFile objects.
        """
        try:
            # Create a temporary file browser with the specified sorting
            temp_browser = FileBrowser(
                directory=self.directory, 
                recursive=self.recursive, 
                filter=self.filter, 
                sort_by=sort_by
            )
            temp_browser._files = self._files.copy()  # Use the same file cache
            temp_browser._get_sortable_files()
            # Return SortableFile objects to avoid re-creating them in _find_closest methods
            return temp_browser.get_sorted_files(temp_browser._files, return_sortable_files=True)
        except Exception as e:
            if config.debug:
                logger.debug(f"Error creating sorted files for {sort_by}: {e}")
            # Fallback: convert file paths to SortableFile objects
            return [SortableFile(f) for f in files]

    def _handle_find_closest_failure_message(self, sort_by_or_description: str, message: str) -> None:
        """Unified method to return None when no closer match is found."""
        if config.debug:
            logger.debug(f"Closest file by {sort_by_or_description}: {message}, no match found")
        return None

    def _alphanumeric_key(self, text: str) -> List:
        """Convert text to alphanumeric key for comparison, matching the logic in Utils.alphanumeric_sort."""
        def convert(text): return int(text) if text.isdigit() else text
        return [convert(c) for c in re.split('([0-9]+)', text.lower())]

    def _find_closest_by_name(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str]) -> Optional[str]:
        """
        Simple name-based closest file finder.
        Finds the file whose name would be closest alphabetically to the search text.
        Uses alphanumeric comparison to match the sorting logic.
        """
        search_text_key = self._alphanumeric_key(os.path.basename(search_text))
        
        # Find the position where search_text would fit alphabetically using alphanumeric comparison
        for sortable_file in sorted_files:
            filename_key = self._alphanumeric_key(sortable_file.basename)
            if filename_key >= search_text_key:
                closest_file = sortable_file.full_file_path
                # Map back to original file index
                self.file_cursor = original_files.index(closest_file)
                if config.debug:
                    logger.debug(f"Closest file by name: position {self.file_cursor}")
                return closest_file
        
        return self._handle_find_closest_failure_message("name", "no match")

    def _find_closest_by_time(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str], sort_by: SortBy) -> Optional[str]:
        """Find closest file by creation or modification time."""
        # Get the time value of the file if it exists
        try:
            search_sortable = SortableFile(search_text)
            if sort_by == SortBy.CREATION_TIME:
                target_time = search_sortable.ctime
            else:  # MODIFY_TIME
                target_time = search_sortable.mtime
            
            # Find the closest file by time
            closest_file = None
            closest_time_diff = float('inf')
            for sortable_file in sorted_files:
                try:
                    if sort_by == SortBy.CREATION_TIME:
                        file_time = sortable_file.ctime
                    else:  # MODIFY_TIME
                        file_time = sortable_file.mtime
                    
                    time_diff = abs((file_time - target_time).total_seconds())
                    if time_diff < closest_time_diff:
                        closest_time_diff = time_diff
                        closest_file = sortable_file.full_file_path
                except Exception:
                    continue
            
            if closest_file:
                self.file_cursor = original_files.index(closest_file)
                if config.debug:
                    logger.debug(f"Closest file by time: position {self.file_cursor}")
                return closest_file
        except Exception as e:
            if config.debug:
                logger.debug(f"Error finding closest file by time: {e}")
            pass  # Fall through to return first file
        
        return self._handle_find_closest_failure_message(sort_by.get_text(), "non-numeric search")

    def _find_closest_by_size(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str]) -> Optional[str]:
        """Find closest file by size."""
        try:
            target_size = int(search_text)
            closest_file = None
            closest_size_diff = float('inf')
            
            for sortable_file in sorted_files:
                try:
                    file_size = sortable_file.size
                    size_diff = abs(file_size - target_size)
                    if size_diff < closest_size_diff:
                        closest_size_diff = size_diff
                        closest_file = sortable_file.full_file_path
                except Exception:
                    continue
            
            if closest_file:
                self.file_cursor = original_files.index(closest_file)
                if config.debug:
                    logger.debug(f"Closest file by size: position {self.file_cursor}")
                return closest_file
        except ValueError:
            pass  # Search text is not numeric
        
        # Fallback: return first file
        return self._handle_find_closest_failure_message("size", "non-numeric search")

    def _find_closest_by_type(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str]) -> Optional[str]:
        """Find closest file by file type/extension."""
        _unused, search_ext = os.path.splitext(search_text)
        search_text_lower = search_ext.lower()
        
        # Try to find files with matching extension
        for sortable_file in sorted_files:
            if sortable_file.extension.lower() == search_text_lower:
                closest_file = sortable_file.full_file_path
                self.file_cursor = original_files.index(closest_file)
                if config.debug:
                    logger.debug(f"Closest file by type: position {self.file_cursor}")
                return closest_file
        
        # Fallback: return first file
        return self._handle_find_closest_failure_message("type", "no match")

    def _find_closest_by_suffix(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str]) -> Optional[str]:
        """Find closest file by filename suffix (the variant marker after the base stem)."""
        search_sortable = SortableFile(search_text)
        search_suffix = search_sortable.get_suffix().lower()

        # Try to find files with matching suffix
        for sortable_file in sorted_files:
            if sortable_file.get_suffix().lower() == search_suffix:
                closest_file = sortable_file.full_file_path
                self.file_cursor = original_files.index(closest_file)
                if config.debug:
                    logger.debug(f"Closest file by suffix: position {self.file_cursor}")
                return closest_file

        # Fallback: return first file
        return self._handle_find_closest_failure_message("suffix", "no match")

    def _find_closest_by_name_length(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str]) -> Optional[str]:
        """Find closest file by name length."""
        try:
            target_length = len(os.path.basename(search_text))
            closest_file = None
            closest_length_diff = float('inf')
            
            for sortable_file in sorted_files:
                length_diff = abs(sortable_file.name_length - target_length)
                if length_diff < closest_length_diff:
                    closest_length_diff = length_diff
                    closest_file = sortable_file.full_file_path
            
            if closest_file:
                self.file_cursor = original_files.index(closest_file)
                if config.debug:
                    logger.debug(f"Closest file by name length: position {self.file_cursor}")
                return closest_file
        except Exception:
            pass
        
        # Fallback: return first file
        return self._handle_find_closest_failure_message("name length", "error")

    def _find_closest_by_image_property(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str], sort_by: SortBy) -> Optional[str]:
        """Find closest file by image properties (pixels, height, width)."""
        try:
            target_value = int(search_text)
            closest_file = None
            closest_diff = float('inf')
            
            for sortable_file in sorted_files:
                try:
                    # Get the appropriate image property
                    if sort_by == SortBy.IMAGE_PIXELS:
                        current_value = sortable_file.get_image_pixels()
                    elif sort_by == SortBy.IMAGE_HEIGHT:
                        current_value = sortable_file.get_image_height()
                    else:  # IMAGE_WIDTH
                        current_value = sortable_file.get_image_width()
                    
                    if current_value is not None:
                        diff = abs(current_value - target_value)
                        if diff < closest_diff:
                            closest_diff = diff
                            closest_file = sortable_file.full_file_path
                except Exception:
                    continue
            
            if closest_file:
                self.file_cursor = original_files.index(closest_file)
                if config.debug:
                    logger.debug(f"Closest file by {sort_by.get_text()}: position {self.file_cursor}")
                return closest_file
        except ValueError:
            pass  # Search text is not numeric
        
        return self._handle_find_closest_failure_message(sort_by.get_text(), "non-numeric search")

    def _find_closest_by_related_image(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str]) -> Optional[str]:
        """Find closest file by related image."""
        basename_lookup = {sf.basename: sf for sf in sorted_files}
        try:
            sortable_file = SortableFile(search_text)
            target_related_path = sortable_file.get_origin_image_or_self(basename_lookup)
            target_key = self._alphanumeric_key(target_related_path)
        except Exception:
            return self._handle_find_closest_failure_message("related image", "error getting related image")

        for i, sortable_file in enumerate(sorted_files):
            try:
                file_related_path = sortable_file.get_origin_image_or_self(basename_lookup)
                file_key = self._alphanumeric_key(file_related_path)
                if file_key >= target_key:
                    closest_file = sortable_file.full_file_path
                    self.file_cursor = original_files.index(closest_file)
                    if config.debug:
                        logger.debug(f"Closest file by related image: position {self.file_cursor}")
                    return closest_file
            except Exception:
                continue

        return self._handle_find_closest_failure_message("related image", "no match")

    def _find_closest_by_random(self, search_text: str, sorted_files: List[SortableFile], original_files: List[str]) -> str:
        """Find closest file for random sorting."""
        # For random sorting, just return a random file
        random_index = randint(0, len(sorted_files) - 1)
        closest_file = sorted_files[random_index].full_file_path
        self.file_cursor = original_files.index(closest_file)
        if config.debug:
            logger.debug(f"Closest file by random: position {self.file_cursor}")
        return closest_file

    def page_down(self, half_length: bool = False) -> str:
        paging_length = self._get_paging_length(half_length=half_length)
        test_cursor = self.file_cursor + paging_length
        if test_cursor > len(self._files):
            test_cursor = 1
        with self.cursor_lock:
            self.file_cursor = test_cursor - 1
        return self.next_file()

    def page_up(self, half_length: bool = False) -> str:
        paging_length = self._get_paging_length(half_length=half_length)
        test_cursor = self.file_cursor - paging_length
        if test_cursor < 0:
            test_cursor = -1
        with self.cursor_lock:
            self.file_cursor = test_cursor + 1
        return self.previous_file()

    def _get_paging_length(self, half_length: bool = False) -> int:
        divisor = 20 if half_length else 10
        paging_length = int(len(self._files) / divisor)
        if paging_length > 200:
            return 200
        if paging_length == 0:
            return 1
        return paging_length

    def _get_sortable_files(self) -> List[SortableFile]:
        self._files = []
        if not self.use_file_paths_json and not os.path.exists(self.directory):
            return self._files

        files = []
        self._gather_files(files)

        # NOTE using a cache may result in incorrect sorting on refresh if files were renamed to the same as a previous file
        def cache_fileinfo(f):
            sortable_file = SortableFile(f)
            self._files_cache[f] = sortable_file
            return sortable_file
        self._files = [self._files_cache[f] if f in self._files_cache else cache_fileinfo(f) for f in files]
        return self._files

    def get_files(self) -> List[str]:
        if self.filepaths is None or len(self.filepaths) == 0:
            self.filepaths = self.get_sorted_files(self._files)
        return self.filepaths

    def get_incremental_progress_text(self) -> str:
        if not self.is_incremental_loading:
            return ""
        return _(
            "Loading directory... {0} files currently available (scanned {1} folders; total unknown)"
        ).format(len(self.filepaths), self._incremental_scanned_dirs)

    def _should_use_incremental_load(
        self, *, file_check: bool, removed_files: List[str]
    ) -> bool:
        if file_check or len(removed_files) > 0:
            return False
        if self.use_file_paths_json:
            return False
        return self._use_incremental_on_full_refresh and self._incremental_refresh_armed

    def _recompute_incremental_decision(self, threshold: int = 5000) -> None:
        if self.use_file_paths_json or not os.path.isdir(self.directory):
            self._use_incremental_on_full_refresh = False
            return
        probe_files: List[str] = []
        self._gather_files(probe_files)
        # External drives are slower overall; keep incremental batches smaller
        # so files appear earlier while scanning.
        self._incremental_batch_size = 100 if self._is_external_drive_directory else 250
        factor = 5 if self._is_external_drive_directory else 1
        self._use_incremental_on_full_refresh = factor * len(probe_files) > threshold

    def _update_drive_characteristics(self) -> None:
        self._is_external_drive_directory = Utils.is_external_drive(self.directory)

    def _stop_incremental_load(self) -> None:
        if self._incremental_thread is not None and self._incremental_thread.is_alive():
            self._incremental_stop_event.set()
        self._incremental_thread = None
        self._incremental_stop_event = threading.Event()
        self.is_incremental_loading = False
        self._incremental_scanned_dirs = 0
        self._incremental_files_discovered = 0

    def _start_incremental_load(self) -> None:
        self._incremental_refresh_armed = False
        self.is_incremental_loading = True
        self._incremental_scanned_dirs = 0
        self._incremental_files_discovered = 0
        self._files = []
        self.filepaths = []

        seed = self._preferred_initial_file
        self._preferred_initial_file = None
        if seed and os.path.isfile(seed):
            ext = os.path.splitext(seed)[1].lower()
            if ext in set(config.file_types):
                sf = self._files_cache.get(seed) or SortableFile(seed)
                self._files_cache[seed] = sf
                self._files = [sf]
                self.filepaths = self.get_sorted_files(list(self._files))

        thread = threading.Thread(
            target=self._incremental_load_worker,
            name="file_browser_incremental_load",
            daemon=True,
        )
        self._incremental_thread = thread
        thread.start()

    def _incremental_load_worker(self) -> None:
        """
        Scan and merge files in batches for large directories.

        Important behavior note:
        during incremental load, `get_files()` returns a sorted view of the
        currently discovered subset, not the final complete directory set.
        Until loading completes, "first/last" reflects that subset only.
        """
        allowed_extensions = set(config.file_types)
        batch: List[str] = []
        to_scan = [self.directory]
        seen_paths = set(self.filepaths)
        stop_event = self._incremental_stop_event
        while to_scan and not stop_event.is_set():
            current_dir = to_scan.pop()
            self._incremental_scanned_dirs += 1
            try:
                with os.scandir(current_dir) as entries:
                    for entry in entries:
                        if stop_event.is_set():
                            break
                        if self.recursive and entry.is_dir(follow_symlinks=False):
                            to_scan.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in allowed_extensions and entry.path not in seen_paths:
                                batch.append(entry.path)
                                seen_paths.add(entry.path)
                                self._incremental_files_discovered += 1
                                if len(batch) >= self._incremental_batch_size:
                                    self._merge_incremental_batch(batch)
                                    batch = []
            except PermissionError:
                continue
            except OSError:
                continue

        if batch and not stop_event.is_set():
            self._merge_incremental_batch(batch)
        self.is_incremental_loading = False

    def _merge_incremental_batch(self, batch: List[str]) -> None:
        if len(batch) == 0:
            return
        for path in batch:
            if path in self._files_cache:
                sf = self._files_cache[path]
            else:
                sf = SortableFile(path)
                self._files_cache[path] = sf
            self._files.append(sf)
        self.filepaths = self.get_sorted_files(list(self._files))

    def get_files_sorted_for_operation(
        self, sort_by: SortBy, sort: Sort = Sort.ASC
    ) -> List[str]:
        """
        Return files in a caller-requested sort order without changing the
        browser's persistent sort settings.

        Uses cached SortableFile instances, so time-based sorts can reuse
        existing stat-derived metadata.

        The returned paths are exactly the browser's current file list: when
        ``self.recursive`` is True, that list includes files in subfolders
        (same scope as navigation); each path keeps its directory.
        """
        if len(self._files) == 0:
            return []

        original_sort_by = self.sort_by
        original_sort = self.sort
        try:
            self.sort_by = sort_by
            self.sort = sort
            return list(self.get_sorted_files(list(self._files)))
        finally:
            self.sort_by = original_sort_by
            self.sort = original_sort

    def get_files_with_retry(self, retry_with_delay: int = 0) -> List[str]:
        files = self.get_files()
        if len(files) == 0 and retry_with_delay > 0:
            if retry_with_delay > 3:
                return files
            logger.warning(f"No files found, sleeping for {retry_with_delay} seconds and trying again...")
            sleep(retry_with_delay)
            return self.get_files_with_retry(retry_with_delay=retry_with_delay+1)
        return files

    def get_sorted_files(self, sortable_files: List[SortableFile], return_sortable_files: bool = False):
        if self.sort == Sort.RANDOM:
            sortable_files = list(sortable_files)
            shuffle(sortable_files)  # TODO technically should be caching the random sort state somehow
        else:
            reverse = self.sort == Sort.DESC

            if self.sort_by == SortBy.FULL_PATH:
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
            elif self.sort_by == SortBy.NAME:
                sortable_files = Utils.alphanumeric_sort(sortable_files, text_lambda=lambda sf: sf.basename.lower(), reverse=reverse)
            elif self.sort_by == SortBy.CREATION_TIME:
                sortable_files.sort(key=lambda sf: sf.ctime, reverse=reverse)
            elif self.sort_by == SortBy.MODIFY_TIME:
                sortable_files.sort(key=lambda sf: sf.mtime, reverse=reverse)
            elif self.sort_by == SortBy.TYPE:
                # Sort by full path first, then by extension
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.extension.lower(), reverse=reverse)
            elif self.sort_by == SortBy.SIZE:
                # Sort by full path first, then by size
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.size, reverse=reverse)
            elif self.sort_by == SortBy.NAME_LENGTH:
                # Sort by full path first, then by name length
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.name_length, reverse=reverse)
            elif self.sort_by == SortBy.IMAGE_PIXELS:
                # Sort by full path first, then by total pixels
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.get_image_pixels(), reverse=reverse)
            elif self.sort_by == SortBy.IMAGE_HEIGHT:
                # Sort by full path first, then by height
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.get_image_height(), reverse=reverse)
            elif self.sort_by == SortBy.IMAGE_WIDTH:
                # Sort by full path first, then by width
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.get_image_width(), reverse=reverse)
            elif self.sort_by == SortBy.RELATED_IMAGE:
                basename_lookup = {sf.basename: sf for sf in sortable_files}
                sortable_files.sort(key=lambda sf: (sf.get_origin_image_or_self(basename_lookup), sf.basename.lower()), reverse=reverse)
            elif self.sort_by == SortBy.SUFFIX:
                # Sort by full path first, then by basename, then by suffix
                sortable_files.sort(key=lambda sf: sf.full_file_path.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.basename.lower(), reverse=reverse)
                sortable_files.sort(key=lambda sf: sf.get_suffix().lower(), reverse=reverse)

        # Return either SortableFile objects or filepaths
        if return_sortable_files:
            return sortable_files
        else:
            # Extract the file path only
            filepaths = [sf.full_file_path for sf in sortable_files]
            return filepaths

    def _gather_files(self, files: Optional[List[str]] = None) -> None:
        allowed_extensions = config.file_types
        if files is None:  # This is not the standard use case, only used in methods where we don't care about sorting
            self.filepaths.clear()
            files = self.filepaths

        inclusions, exclusions = FileBrowser._parse_filter(self.filter or "")

        if inclusions:
            if self.use_file_paths_json:
                filepaths = self.load_file_paths_json()
                candidate_files = [f for f in filepaths if any(re.search(term, f) for term in inclusions)]
                if exclusions:
                    candidate_files = [f for f in candidate_files if not any(re.search(term, f) for term in exclusions)]
            else:
                seen: set = set()
                candidate_files = []
                for term in inclusions:
                    pattern = "**/" + term if self.recursive else term
                    if "*" not in pattern or (pattern.startswith("**") and not pattern.endswith("*")):
                        pattern += "*"
                    recursive = self.recursive or term.startswith("**/")
                    for f in glob.glob(os.path.join(self.directory, pattern), recursive=recursive):
                        if f not in seen:
                            seen.add(f)
                            candidate_files.append(f)
                if exclusions:
                    candidate_files = [f for f in candidate_files if not any(fnmatch.fnmatch(os.path.basename(f), term) for term in exclusions)]
            for f in candidate_files:
                for ext in allowed_extensions:
                    if f.lower().endswith(ext):
                        files.append(f)
        elif exclusions:
            # No inclusions — gather full file set then subtract exclusions.
            if self.use_file_paths_json:
                filepaths = self.load_file_paths_json()
                for f in filepaths:
                    if any(re.search(term, f) for term in exclusions):
                        continue
                    for ext in allowed_extensions:
                        if f.lower().endswith(ext):
                            files.append(f)
            else:
                pattern = "**/*" if self.recursive else "*"
                candidate_files = glob.glob(os.path.join(self.directory, pattern), recursive=self.recursive)
                candidate_files = [f for f in candidate_files if not any(fnmatch.fnmatch(os.path.basename(f), term) for term in exclusions)]
                for f in candidate_files:
                    for ext in allowed_extensions:
                        if f.lower().endswith(ext):
                            files.append(f)
        elif self.use_file_paths_json:
            filepaths = self.load_file_paths_json()
            _pat = "**/" if self.recursive else ""
            for ext in allowed_extensions:
                if self.use_file_paths_json:
                    files.extend([f for f in filepaths if f.lower().endswith(ext)])
                else:
                    files.extend(glob.glob(os.path.join(self.directory, _pat + "*" + ext), recursive=self.recursive))
        else:
            with Utils.file_operation_lock:
                to_scan = [self.directory]
                while to_scan:
                    current_dir = to_scan.pop()
                    try:
                        with os.scandir(current_dir) as it:
                            for entry in it:
                                if entry.is_dir(follow_symlinks=False) and self.recursive:
                                    to_scan.append(entry.path)
                                elif entry.is_file(follow_symlinks=False):
                                    ext = os.path.splitext(entry.name)[1].lower()
                                    if ext in allowed_extensions:
                                        files.append(entry.path)
                    except PermissionError:
                        logger.warning(f"Permission denied: {current_dir}")

    def count_files_by_type_in_directory(self, recursive: bool = True) -> Dict[str, int]:
        """
        Count ALL files by type in a specified directory, including unsupported file types.
        This is used for delete confirmation to show users exactly what they're deleting.
        
        Args:
            directory_path: Path to the directory to scan
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            Dictionary mapping file extensions to counts, plus 'total' and 'subdirectories' keys
        """
        if not os.path.exists(self.directory) or not os.path.isdir(self.directory):
            return {}
            
        file_counts = {}
        total_count = 0
        subdirectory_count = 0
        
        try:
            with Utils.file_operation_lock:
                to_scan = [self.directory]
                while to_scan:
                    current_dir = to_scan.pop()
                    try:
                        with os.scandir(current_dir) as it:
                            for entry in it:
                                if entry.is_dir(follow_symlinks=False) and recursive:
                                    to_scan.append(entry.path)
                                    subdirectory_count += 1
                                elif entry.is_file(follow_symlinks=False):
                                    ext = os.path.splitext(entry.name)[1].lower()
                                    if ext == "":  # Files with no extension
                                        ext = "(no extension)"
                                    file_counts[ext] = file_counts.get(ext, 0) + 1
                                    total_count += 1
                    except PermissionError:
                        logger.warning(f"Permission denied: {current_dir}")
        except Exception as e:
            logger.error(f"Error counting files in directory {self.directory}: {e}")
            return {}
            
        file_counts['__total'] = total_count
        file_counts['__subdirectories'] = subdirectory_count
        return file_counts

    def get_file_type_summary_for_directory(self, recursive: bool = True) -> str:
        """
        Get a human-readable summary of ALL file types in a directory.
        This shows users exactly what they're about to delete.
        
        Args:
            directory_path: Path to the directory to scan
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            Formatted string showing file type counts and subdirectory count
        """
        file_counts = self.count_files_by_type_in_directory(recursive)
        
        if not file_counts or file_counts.get('__total', 0) == 0:
            return _("No files found")

        total = file_counts.pop('__total', 0)
        subdirectories = file_counts.pop('__subdirectories', 0)
        
        # Sort by count (descending) and then by extension
        sorted_counts = sorted(file_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Build the summary string
        summary_parts = []
        for ext, count in sorted_counts:
            if count == 1:
                summary_parts.append(f"{ext}: {count} file")
            else:
                summary_parts.append(f"{ext}: {count} files")
                
        summary = "\n".join(summary_parts)
        # Add totals
        if subdirectories > 0:
            if subdirectories == 1:
                summary += f"\n\n{_('Subdirectories')}: {subdirectories}"
            else:
                summary += f"\n\n{_('Subdirectories')}: {subdirectories}"
        
        summary += f"\n{_('Total files')}: {total}"
        
        return summary

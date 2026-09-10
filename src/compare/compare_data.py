import os
import pickle
import sys

from utils.logging_setup import get_logger

logger = get_logger("compare_data")


class CompareData:

    # Stale-entry purge for churny directories (files get cached, then moved
    # out, and their entries linger forever). Triggered at save time when the
    # cache holds PURGE_FACTOR times more entries than files found this run
    # (subject to PURGE_FLOOR), except at extreme scale where the scan is
    # left to the user. External drives hit the ceilings at 1/EXTERNAL_DRIVE_FACTOR
    # of the counts, since os.path.isfile is far slower there (same convention
    # as FileBrowser.is_slow_total_files).
    PURGE_FACTOR = 2.0
    PURGE_FLOOR = 100
    LARGE_CACHE_CEILING = 40000
    LARGE_FOUND_CEILING = 20000
    EXTERNAL_DRIVE_FACTOR = 5
    PURGE_STATE_KEY = "media_cache_purge_state"

    def __init__(self, base_dir=".", data_filename="image_data.pkl"):
        self.base_dir = base_dir
        self.has_new_file_data = False
        self.files_found = []
        self.n_files_found = 0
        self.file_data_dict = {}
        self._data_filename = data_filename
        self._file_data_filepath = os.path.join(base_dir, data_filename)

    def load_data(self, overwrite=False):
        if overwrite or not os.path.exists(self._file_data_filepath):
            if not os.path.exists(self._file_data_filepath):
                logger.info("Image data not found so creating new cache"
                      + " - this may take a while.")
            elif overwrite:
                logger.info("Overwriting image data caches - this may take a while.")
            self.file_data_dict = {}
        else:
            with open(self._file_data_filepath, "rb") as f:
                self.file_data_dict = pickle.load(f)

    def purge_stale_entries(self) -> int:
        """Remove entries whose file no longer exists on disk; return count.

        Existence-based on purpose (not listing-membership-based): entries for
        subdirectory files survive a recursive-option change as long as the
        files exist — they're still valid data — so only genuine churn
        (files moved out or deleted) is ever removed.
        """
        stale = [f for f in self.file_data_dict if not os.path.isfile(f)]
        for f in stale:
            del self.file_data_dict[f]
        return len(stale)

    def _maybe_purge_stale_entries(self) -> None:
        """Run purge_stale_entries() when the cache looks inflated for churn.

        See the constants above for the gates: inflation trigger with floor,
        extreme-scale bail-out with external-drive penalty, and a repeat-scan
        guard so an inflated-but-clean cache (e.g. recursive-built entries
        during non-recursive runs) doesn't rescan on every save.
        """
        if not self.file_data_dict:
            return
        cache_size = len(self.file_data_dict)
        files_found = len(self.files_found)
        if cache_size <= max(self.PURGE_FLOOR, self.PURGE_FACTOR * files_found):
            return

        from utils.utils import Utils
        drive_factor = (self.EXTERNAL_DRIVE_FACTOR
                        if Utils.is_external_drive(self.base_dir) else 1)
        if (drive_factor * cache_size >= self.LARGE_CACHE_CEILING
                and drive_factor * files_found >= self.LARGE_FOUND_CEILING):
            logger.debug(
                "Skipping stale-entry purge for %s: too large to scan automatically"
                " (%d entries, %d files found, drive factor %d)",
                self._file_data_filepath, cache_size, files_found, drive_factor)
            return

        from utils.app_info_cache import app_info_cache
        purge_state = app_info_cache.get(self.base_dir, self.PURGE_STATE_KEY, default_val={})
        last_size = purge_state.get(self._data_filename)
        if last_size is not None and cache_size <= last_size:
            return

        removed = self.purge_stale_entries()
        if removed > 0:
            self.has_new_file_data = True
            logger.info("Purged %d stale cache entries from %s",
                        removed, self._file_data_filepath)
        purge_state = dict(purge_state)
        purge_state[self._data_filename] = len(self.file_data_dict)
        app_info_cache.set(self.base_dir, self.PURGE_STATE_KEY, purge_state)

    def save_data(self, overwrite=False, verbose=False):
        try:
            self._maybe_purge_stale_entries()
        except Exception:
            logger.exception("Stale-entry purge failed; saving cache unmodified")
        if self.has_new_file_data or overwrite:
            with open(self._file_data_filepath, "wb") as store:
                pickle.dump(self.file_data_dict, store)
            # Free memory after persist: comparison/search must use in-memory data
            # (e.g. embedding mode uses _file_embeddings; prompts_exact uses _file_pos_texts/_file_neg_texts).
            self.file_data_dict = None
            if verbose:
                if overwrite:
                    logger.info("Overwrote any pre-existing image data at:")
                else:
                    logger.info("Updated image data saved to: ")
                logger.info(self._file_data_filepath)

        self.n_files_found = len(self.files_found)

        if self.n_files_found == 0:
            raise AssertionError("No image data found for comparison with"
                                 + " current params - checked"
                                 + " in base dir = \"" + self.base_dir + "\"")
        elif verbose:
            logger.info("Data from " + str(self.n_files_found)
                  + " files compiled for comparison.")
    
    def estimate_memory_size(self) -> int:
        """
        Estimate the memory size of this CompareData instance in bytes.
        
        Returns:
            Estimated memory size in bytes
        """
        total_size = sys.getsizeof(self)
        # Size of base_dir string
        total_size += sys.getsizeof(self.base_dir)
        # Size of files_found list
        if self.files_found is not None:
            total_size += sys.getsizeof(self.files_found)
            for file_path in self.files_found:
                total_size += sys.getsizeof(file_path)
        # Size of file_data_dict (dictionary of embeddings)
        if self.file_data_dict is not None:
            total_size += sys.getsizeof(self.file_data_dict)
            # Estimate size of embeddings (each embedding is a list of floats)
            # CLIP embeddings are typically 512 floats = ~2KB per embedding
            for rel_path, embedding in self.file_data_dict.items():
                total_size += sys.getsizeof(rel_path)  # Path string
                total_size += sys.getsizeof(embedding)  # List of floats
                if isinstance(embedding, list):
                    total_size += len(embedding) * sys.getsizeof(0.0)  # Float size
        return total_size
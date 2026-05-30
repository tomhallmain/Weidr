import os
import pickle

from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils
logger = get_logger("compare_result")


class CompareResult:
    # TODO: Re-enable file output in a more usable form — JSON, written only when
    # a config setting (e.g. `save_compare_output`) is enabled, with a filename
    # scoped per mode so concurrent runs on the same directory don't clash.
    RESULT_FILENAME_TEMPLATE = "weidr_result_{mode}.pkl"

    def __init__(self, base_dir=".", files=[], mode=None):
        self.base_dir = base_dir
        self._mode = mode
        self._dir_files_hash = CompareResult.hash_dir_files(files)
        self.file_groups = {}
        self.files_grouped = {}
        self.group_index = 0
        self.is_complete = False
        self.i = 1  # start at 1 because index 0 is identity comparison roll index

    def finalize_search_result(self, search_path, args=None, verbose=False, threshold_duplicate=0.99, threshold_related=0.95, is_embedding=False):
        if len(self.files_grouped) > 0:
            if verbose:
                if args is not None:
                    parts = []
                    if args.search_media_path:
                        parts.append(f"file={args.search_media_path}")
                    if args.search_text:
                        parts.append(f"text=\"{args.search_text}\"")
                    header = f"Possibly related images to ({', '.join(parts)}):"
                else:
                    header = f"Possibly related images to \"{search_path}\":"
                print(header)
                for f in self.files_grouped:
                    if f == search_path:
                        continue
                    if is_embedding:
                        similarity = self.files_grouped[f]
                        if similarity > threshold_duplicate:
                            line = f"DUPLICATE: {f} - similarity: {similarity}"
                        elif similarity > threshold_related:
                            line = f"PROBABLE MATCH: {f} - similarity: {similarity}"
                        else:
                            line = f"{f} - similarity: {similarity}"
                    else:
                        diff_score = int(self.files_grouped[f])
                        if diff_score < threshold_duplicate:
                            line = "DUPLICATE: " + f
                        elif diff_score < threshold_related:
                            line = "PROBABLE MATCH: " + f
                        else:
                            line = f + " - similarity: " + str(round(1000 / diff_score, 4))
                    print(line)
        elif verbose:
            logger.warning(f"No similar images to \"{search_path}\" identified with current params.")

    def finalize_group_result(self, verbose=False, store_checkpoints=False):
        if not verbose:
            print("")
        group_counter = 0
        group_print_cutoff = 5
        to_print_etc = True

        if len(self.files_grouped) > 0:
            print("")

            # TODO calculate group similarities and mark duplicates separately in this case

            for group_index in self.sort_groups(self.file_groups):
                group = self.file_groups[group_index]
                if len(group) < 2:
                    continue
                    # Technically this means losing some possible associations.
                    # TODO handle stranded group members
                group_counter += 1
                if group_counter <= group_print_cutoff:
                    print("Group " + str(group_counter))
                    for f in sorted(group, key=lambda f: group[f]):
                        print(f)
                elif to_print_etc:
                    print("(etc.)")
                    to_print_etc = False

            logger.info(f"Found {group_counter} image groups with current parameters.")
            if store_checkpoints:
                self.is_complete = True
                self.store()
        else:
            logger.warning("No similar images identified with current params.")
            if store_checkpoints:
                self.is_complete = True
                self.store()

    def sort_groups(self, file_groups):
        return sorted(file_groups,
                      key=lambda group_index: len(file_groups[group_index]))

    def store(self):
        save_path = CompareResult.cache_path(self.base_dir, self._mode)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
            logger.info(f"Stored compare result: {save_path}")

    def equals_hash(self, files):
        return self._dir_files_hash == CompareResult.hash_dir_files(files)

    @staticmethod
    def cache_path(base_dir, mode=None):
        mode_slug = mode.name.lower() if mode is not None else "default"
        filename = CompareResult.RESULT_FILENAME_TEMPLATE.format(mode=mode_slug)
        return os.path.join(base_dir, filename)

    @staticmethod
    def hash_dir_files(files):
        # Store paths directly rather than hash() values. Python's built-in
        # hash() for strings is randomised per-process (PYTHONHASHSEED), so
        # a pkl written in one session would never match in the next.
        return list(files)

    def validate_indices(self, files):
        """
        Validates that all indices in files_grouped are valid for the given files list.
        Returns True if all indices are valid, False otherwise.
        """
        valid_indices = [idx for idx in self.files_grouped if idx < len(files)]
        if len(valid_indices) != len(self.files_grouped):
            logger.error("Warning: Checkpoint data contains invalid indices. Discarding checkpoint data.")
            return False
        return True

    @staticmethod
    def load(base_dir, files, mode=None, overwrite=False):
        if overwrite:
            return CompareResult(base_dir, files, mode=mode)
        cache_path = CompareResult.cache_path(base_dir, mode)
        if not os.path.exists(cache_path):
            logger.info(f"No checkpoint found for {base_dir} - creating new compare result cache.")
            return CompareResult(base_dir, files, mode=mode)
        cached = None
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        except Exception:
            logger.error(f"Failed to load compare result from base dir {base_dir}")
            return CompareResult(base_dir, files, mode=mode)
        if not cached.equals_hash(files):
            # Old pkls used Python's hash() on strings, which is randomised per-process.
            # Those are always stale — discard silently and rebuild rather than surface a
            # misleading error. New pkls (path strings) that genuinely don't match raise.
            if (cached._dir_files_hash
                    and isinstance(cached._dir_files_hash[0], int)):
                logger.warning(f"Discarding {cache_path}: stored in legacy format, rebuilding checkpoint.")
                return CompareResult(base_dir, files, mode=mode)
            raise ValueError(f"{cache_path} does not match {files}")

        # Validate that all indices in files_grouped are valid
        if not cached.validate_indices(files):
            return CompareResult(base_dir, files, mode=mode)

        logger.info(f"Loaded compare result: {cache_path}")
        return cached

import os
import pickle

from utils.logging_setup import get_logger
from utils.translations import I18N
from utils.utils import Utils

_ = I18N._
logger = get_logger("compare_result")


class CompareResult:
    SEARCH_OUTPUT_FILE = "weidr_search_output.txt"
    GROUPS_OUTPUT_FILE = "weidr_file_groups_output.txt"
    RESULT_FILENAME = "weidr_result.pkl"

    def __init__(self, base_dir=".", files=[]):
        self.base_dir = base_dir
        self.search_output_path = os.path.join(
            base_dir, CompareResult.SEARCH_OUTPUT_FILE)
        self.groups_output_path = os.path.join(
            base_dir, CompareResult.GROUPS_OUTPUT_FILE)
        self._dir_files_hash = CompareResult.hash_dir_files(files)
        self.file_groups = {}
        self.files_grouped = {}
        self.group_index = 0
        self.is_complete = False
        self.i = 1  # start at 1 because index 0 is identity comparison roll index

    def finalize_search_result(self, search_path, args=None, verbose=False, threshold_duplicate=0.99, threshold_related=0.95, is_embedding=False):
        if len(self.files_grouped) > 0:
            with open(self.search_output_path, "w") as textfile:
                # Header
                if args is not None:
                    header = f"Possibly related images to ("
                    if args.search_media_path is not None:
                        header += f"search file {args.search_media_path}, "
                    if args.search_text is not None:
                        header += f"search text \"{args.search_text}\", "
                    if args.negative_search_media_path is not None:
                        header += f"negative search file {args.negative_search_media_path}, "
                    if args.search_text_negative is not None:
                        header += f"negative search text \"{args.search_text_negative}\", "
                    header = header[:-2] + "):\n"
                else:
                    header = f"Possibly related images to \"{search_path}\":\n"
                Utils.safe_write(textfile, header)
                if verbose:
                    print(header)

                # Content
                for f in self.files_grouped:
                    if not f == search_path:
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
                                similarity_score = str(
                                    round(1000/diff_score, 4))
                                line = f + " - similarity: " + similarity_score
                        Utils.safe_write(textfile, line + "\n")
                        if verbose:
                            print(line)
            if verbose:
                logger.info(f"This output data saved to file at {self.search_output_path}")
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

            with open(self.groups_output_path, "w") as textfile:
                for group_index in self.sort_groups(self.file_groups):
                    group = self.file_groups[group_index]
                    if len(group) < 2:
                        continue
                        # Technically this means losing some possible associations.
                        # TODO handle stranded group members
                    group_counter += 1
                    textfile.write("Group " + str(group_counter) + "\n")
                    if group_counter <= group_print_cutoff:
                        print("Group " + str(group_counter))
                    elif to_print_etc:
                        print("(etc.)")
                        to_print_etc = False
                    for f in sorted(group, key=lambda f: group[f]):
                        Utils.safe_write(textfile, f + "\n")
                        if group_counter <= group_print_cutoff:
                            print(f)

            logger.info(f"Found {group_counter} image groups with current parameters.")
            logger.info(f"Printed up to first {group_print_cutoff} groups identified. All group data saved to file at {self.groups_output_path}")
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
        save_path = CompareResult.cache_path(self.base_dir)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
            logger.info(f"Stored compare result: {save_path}")

    def equals_hash(self, files):
        return self._dir_files_hash == CompareResult.hash_dir_files(files)

    @staticmethod
    def cache_path(base_dir):
        return os.path.join(base_dir, CompareResult.RESULT_FILENAME)

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
    def load(base_dir, files, overwrite=False):
        if overwrite:
            return CompareResult(base_dir, files)
        cache_path = CompareResult.cache_path(base_dir)
        if not os.path.exists(cache_path):
            logger.info(f"No checkpoint found for {base_dir} - creating new compare result cache.")
            return CompareResult(base_dir, files)
        cached = None
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        except Exception:
            logger.error(f"Failed to load compare result from base dir {base_dir}")
            return CompareResult(base_dir, files)
        if not cached.equals_hash(files):
            # Old pkls used Python's hash() on strings, which is randomised per-process.
            # Those are always stale — discard silently and rebuild rather than surface a
            # misleading error. New pkls (path strings) that genuinely don't match raise.
            if (cached._dir_files_hash
                    and isinstance(cached._dir_files_hash[0], int)):
                logger.warning(f"Discarding {cache_path}: stored in legacy format, rebuilding checkpoint.")
                return CompareResult(base_dir, files)
            raise ValueError(f"{cache_path} does not match {files}")

        # Validate that all indices in files_grouped are valid
        if not cached.validate_indices(files):
            return CompareResult(base_dir, files)

        logger.info(f"Loaded compare result: {cache_path}")
        return cached

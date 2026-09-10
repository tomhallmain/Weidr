import os
import pickle

from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils
logger = get_logger("compare_result")


def _format_group_sizes(group_sizes: dict, summary_threshold: int = 8, top_n: int = 5) -> str:
    """Compact one-line representation of a {group_index: file_count} dict.
    Groups with more than *summary_threshold* entries are summarised to avoid
    flooding the log with hundreds of lines."""
    if len(group_sizes) <= summary_threshold:
        return str(group_sizes)
    top = dict(sorted(group_sizes.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
    return f"{len(group_sizes)} groups; largest: {top}"


class CheckpointCorruptedError(ValueError):
    """Raised when a checkpoint file exists but contains corrupt index data."""


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
        # Partition of group_index values into "supergroups" (clusters of
        # related groups based on group mean-embedding similarity). Populated
        # by BaseCompareEmbedding.compute_supergroups(); empty for compare
        # modes with no per-file embedding to average (color/size/models/exact-prompt).
        self.supergroups: list = []
        self.applied_group_sort = None
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
                    print(f"Group {group_counter} ({len(group)} files)")
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

    def sort_groups(self, file_groups, reverse=False):
        return sorted(file_groups,
                      key=lambda group_index: len(file_groups[group_index]),
                      reverse=reverse)

    def build_sorted_group_indexes(self, file_groups, reverse=False) -> list:
        """Return a group_indexes list sorted with supergroup membership as the
        primary key and individual group size as the secondary key (both in the
        same direction).  self.supergroups is reordered in-place to match the
        final display order and then logged.  Groups not covered by any
        supergroup are appended last, sorted by size."""
        if self.supergroups:
            # Primary sort: supergroups by aggregate file count.
            self.supergroups.sort(
                key=lambda cluster: sum(len(file_groups.get(g, {})) for g in cluster),
                reverse=reverse,
            )
            result = []
            covered: set = set()
            for cluster in self.supergroups:
                valid = sorted(
                    (g for g in cluster if g in file_groups),
                    key=lambda g: len(file_groups[g]),
                    reverse=reverse,
                )
                result.extend(valid)
                covered.update(valid)
            ungrouped = sorted(
                (g for g in file_groups if g not in covered),
                key=lambda g: len(file_groups[g]),
                reverse=reverse,
            )
            result.extend(ungrouped)
        else:
            logger.info("Supergroups: none formed")
            result = self.sort_groups(file_groups, reverse=reverse)

        self._log_supergroups(result)
        return result

    def _log_supergroups(self, group_indexes: list) -> None:
        if not self.supergroups:
            return
        # Map internal group_index → 1-based display position to match the UI.
        display_pos = {g: pos + 1 for pos, g in enumerate(group_indexes)}
        total = len(self.supergroups)
        for i, cluster in enumerate(self.supergroups):
            group_sizes = {
                display_pos[g]: len(self.file_groups.get(g, {}))
                for g in cluster if g in display_pos
            }
            total_files = sum(group_sizes.values())
            stranded = [pos for g, pos in display_pos.items() if g in cluster and len(self.file_groups.get(g, {})) == 1]
            logger.info(
                "Supergroup %d/%d: %s — %d files total%s",
                i + 1,
                total,
                _format_group_sizes(group_sizes),
                total_files,
                f" (stranded: {stranded})" if stranded else "",
            )

    def prune_stale_supergroups(self, active_group_indexes=None) -> None:
        '''
        Drop any group_index a supergroup referenced that no longer exists,
        and drop any cluster left empty by that. A cluster that shrinks to a
        single surviving group_index is left as-is.

        active_group_indexes: optional set of currently-live group_index keys.
        When omitted, falls back to set(self.file_groups.keys()) — callers that
        have already synced compare_result.file_groups can omit it; callers that
        haven't (e.g. CompareWrapper._update_groups_for_removed_file) should
        pass the live set explicitly.

        No-op when supergroups is empty or absent (CompareResult unpickled
        from before this feature existed).
        '''
        existing = getattr(self, "supergroups", None)
        if not existing:
            return
        if active_group_indexes is None:
            active_group_indexes = set(self.file_groups.keys())
        self.supergroups = [
            surviving for cluster in existing
            if (surviving := [g for g in cluster if g in active_group_indexes])
        ]

    def has_meaningful_supergroups(self) -> bool:
        """Return True only when at least one supergroup clusters multiple base
        groups together.  When every cluster contains exactly one group the
        supergroup partition is equivalent to the base groups and adds no new
        information worth surfacing in the UI."""
        return any(len(cluster) > 1 for cluster in getattr(self, "supergroups", []))

    def clear_supergroups(self) -> None:
        '''
        Wipe supergroups entirely -- for operations where group_index values
        are fully invalidated rather than just partially stale (e.g. random
        purge removing every group, or a composite-filter rebuild renumbering
        groups from scratch).
        '''
        self.supergroups = []

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

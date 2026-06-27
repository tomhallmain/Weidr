"""Compare images by the L1 distance between their HSV colour histogram signatures.

Unlike CompareColors (pixel-level LAB thumbnail diff), this mode compares the
overall colour *distribution* of each image, order-invariant.  A histogram is
computed once per file and cached; for GIF and video, multiple frames are sampled
and their histograms averaged so the signature represents the expected colour
distribution across the whole clip.
"""

from __future__ import annotations

import getopt
import os
import sys

import numpy as np
from PIL import Image

from compare.base_compare import BaseCompare, gather_files
from compare.compare_args import CompareArgs
from compare.compare_result import CompareResult
from image.frame_cache import FrameCache
from image.image_ops import ImageOps
from utils.config import config
from utils.constants import CompareMode
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

logger = get_logger("compare_color_histogram")

_HIST_LEN = 68  # 36 H + 16 S + 16 V bins (matches ImageOps.color_histogram defaults)


def _l1_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Normalized L1 distance between two per-channel-normalized HSV histograms.

    Result is in [0, 1]: 0 = identical distributions, 1 = maximally different.
    Dividing by 2 converts the raw L1 sum (whose max is 2 for unit-sum histograms)
    to a [0, 1] range.
    """
    return float(np.sum(np.abs(h1 - h2))) / 2.0


class CompareColorHistogram(BaseCompare):
    COMPARE_MODE = CompareMode.COLOR_HISTOGRAM
    CACHE_FILENAME = "image_color_histogram.pkl"

    # Distance thresholds (lower = more similar)
    THRESHOLD_POTENTIAL_DUPLICATE = 0.02
    THRESHOLD_PROBABLE_MATCH = 0.08
    THRESHOLD_GROUP_CUTOFF = 0.15

    def __init__(
        self,
        args: CompareArgs = CompareArgs(),
        gather_files_func=gather_files,
    ) -> None:
        super().__init__(args, gather_files_func)
        self.threshold = float(args.threshold) if args.threshold is not None else 0.2
        self._file_histograms: np.ndarray = np.empty((0, _HIST_LEN), dtype=np.float64)

    def print_settings(self) -> None:
        logger.info("|--------------------------------------------------------------------|")
        logger.info(" COMPARE COLOR HISTOGRAM SETTINGS:")
        logger.info(f" run search: {self.is_run_search}")
        if self.is_run_search:
            logger.info(f" search_media_path: {self.search_media_path}")
        logger.info(f" comparison files base directory: {self.base_dir}")
        logger.info(f" histogram distance threshold: {self.threshold}")
        logger.info(f" max file process limit: {self.args.counter_limit}")
        logger.info(f" recursive: {self.args.recursive}")
        logger.info(
            f" cache file: {self.compare_data._file_data_filepath}"
        )
        logger.info("|--------------------------------------------------------------------|\n\n")

    def get_similarity_threshold(self) -> float:
        return self.threshold

    def set_similarity_threshold(self, threshold) -> None:
        self.threshold = float(threshold)

    # ── Signature computation ─────────────────────────────────────────────────

    @staticmethod
    def _compute_signature(file_path: str) -> np.ndarray | None:
        """Return the HSV histogram signature for *file_path*, or None on failure.

        For GIF and video, samples multiple frames via FrameCache and returns the
        elementwise mean so the signature represents the expected colour
        distribution across the clip.
        """
        from utils.media_utils import get_media_type_for_path
        media_type = get_media_type_for_path(file_path)

        if media_type.is_unconfigured() or media_type.is_audio():
            return None

        try:
            if media_type.is_gif() or media_type.is_video():
                frame_paths = FrameCache.get_frame_samples(file_path)
                if not frame_paths:
                    frame_paths = [FrameCache.get_image_path(file_path)]
                hists = []
                for p in frame_paths:
                    try:
                        hists.append(ImageOps.color_histogram(p))
                    except Exception as e:
                        logger.debug("Histogram failed for frame %s: %s", p, e)
                if not hists:
                    return None
                return np.mean(hists, axis=0).astype(np.float64)
            else:
                image_path = FrameCache.get_image_path(file_path)
                return ImageOps.color_histogram(image_path)
        except Exception as e:
            logger.warning("color_histogram failed for %s: %s", file_path, e)
            return None

    # ── Data gathering ────────────────────────────────────────────────────────

    def get_data(self) -> None:
        """Compute or load the HSV histogram for every file in the base directory."""
        self.compare_data.load_data(overwrite=self.args.overwrite)

        if self.verbose:
            logger.info("Gathering histogram data...")
        else:
            print("Gathering histogram data", end="", flush=True)

        counter = 0
        for f in self.files:
            if self.is_cancelled():
                self.raise_cancellation_exception()
            if Utils.is_invalid_file(f, counter, self.is_run_search, self.args.file_filter):
                continue
            if counter > self.args.counter_limit:
                break

            if f in self.compare_data.file_data_dict:
                hist = self.compare_data.file_data_dict[f]
            else:
                hist = self._compute_signature(f)
                if hist is None:
                    continue
                self.compare_data.file_data_dict[f] = hist
                self.compare_data.has_new_file_data = True

            counter += 1
            self._file_histograms = np.vstack((self._file_histograms, [hist]))
            self.compare_data.files_found.append(f)
            self._handle_progress(counter, self.max_files_processed_even)

        self.compare_data.save_data(self.args.overwrite, verbose=self.verbose)

    # ── Search ────────────────────────────────────────────────────────────────

    def find_similars_to_media(
        self, search_path: str, search_file_index: int
    ) -> dict:
        """Return files ranked by histogram distance from *search_path*."""
        files_found = list(self.compare_data.files_found)
        files_found.pop(search_file_index)
        search_hist = self._file_histograms[search_file_index]
        other_hists = np.delete(self._file_histograms, search_file_index, 0)

        distances = np.sum(np.abs(other_hists - search_hist), axis=1) / 2.0

        if config.search_only_return_closest:
            files_grouped = {
                files_found[i]: float(distances[i])
                for i in range(len(files_found))
                if distances[i] <= self.threshold
            }
        else:
            ranked = sorted(range(len(files_found)), key=lambda i: distances[i])
            files_grouped = {}
            for i in ranked[: config.max_search_results]:
                files_grouped[files_found[i]] = float(distances[i])

        self.compare_result.files_grouped = dict(
            sorted(files_grouped.items(), key=lambda item: item[1])
        )
        self.compare_result.finalize_search_result(
            search_path,
            verbose=self.verbose,
            is_embedding=False,
            threshold_duplicate=CompareColorHistogram.THRESHOLD_POTENTIAL_DUPLICATE,
            threshold_related=CompareColorHistogram.THRESHOLD_PROBABLE_MATCH,
        )
        return {0: files_grouped}

    def run_search(self) -> dict:
        return self._run_search_on_path_histogram(self.search_media_path)

    def _run_search_on_path_histogram(self, search_media_path: str) -> dict:
        if search_media_path not in self.compare_data.files_found:
            if self.verbose:
                logger.info(
                    "Search path not in initial list — computing histogram: %s",
                    search_media_path,
                )
            hist = self._compute_signature(search_media_path)
            if hist is None:
                raise AssertionError(
                    "Could not compute colour histogram for the search file."
                )
            self._file_histograms = np.insert(
                self._file_histograms, 0, [hist], axis=0
            )
            self.compare_data.files_found.insert(0, search_media_path)

        idx = self.compare_data.files_found.index(search_media_path)
        return self.find_similars_to_media(search_media_path, idx)

    # ── Group comparison ──────────────────────────────────────────────────────

    def run_comparison(self, store_checkpoints: bool = False):
        """Compare all histogram pairs and group files with similar distributions."""
        overwrite = self.args.overwrite or not store_checkpoints
        self.compare_result = CompareResult.load(
            self.base_dir,
            self.compare_data.files_found,
            mode=self.COMPARE_MODE,
            overwrite=overwrite,
        )
        if self.compare_result.is_complete:
            return (self.compare_result.files_grouped, self.compare_result.file_groups)

        n_even = Utils.round_up(self.compare_data.n_files_found, 5)
        if self.compare_result.i > 1:
            self._handle_progress(
                self.compare_result.i, n_even, gathering_data=False
            )

        if self.verbose:
            logger.info("Comparing histogram data...")
        else:
            print("Comparing histogram data", end="", flush=True)

        for i in range(self.compare_data.n_files_found):
            if i == 0:
                continue
            if store_checkpoints:
                if i < self.compare_result.i:
                    continue
                if i % 250 == 0 and i > self.compare_result.i:
                    self.compare_result.store()
                self.compare_result.i = i
            self._handle_progress(i, n_even, gathering_data=False)

            rolled = np.roll(self._file_histograms, i, axis=0)
            distances = np.sum(np.abs(self._file_histograms - rolled), axis=1) / 2.0
            similars = np.nonzero(distances <= self.threshold)[0]

            for base_index in similars:
                diff_index = int(
                    (base_index - i) % self.compare_data.n_files_found
                )
                dist = float(distances[base_index])
                f1_grouped = base_index in self.compare_result.files_grouped
                f2_grouped = diff_index in self.compare_result.files_grouped

                if not f1_grouped and not f2_grouped:
                    self.compare_result.files_grouped[base_index] = (
                        self.compare_result.group_index, dist
                    )
                    self.compare_result.files_grouped[diff_index] = (
                        self.compare_result.group_index, dist
                    )
                    self.compare_result.group_index += 1
                elif f1_grouped:
                    existing_group, prev_dist = self.compare_result.files_grouped[
                        base_index
                    ]
                    if prev_dist - CompareColorHistogram.THRESHOLD_GROUP_CUTOFF > dist:
                        self.compare_result.files_grouped[base_index] = (
                            self.compare_result.group_index, dist
                        )
                        self.compare_result.files_grouped[diff_index] = (
                            self.compare_result.group_index, dist
                        )
                        self.compare_result.group_index += 1
                    else:
                        self.compare_result.files_grouped[diff_index] = (
                            existing_group, dist
                        )
                else:
                    existing_group, prev_dist = self.compare_result.files_grouped[
                        diff_index
                    ]
                    if prev_dist - CompareColorHistogram.THRESHOLD_GROUP_CUTOFF > dist:
                        self.compare_result.files_grouped[base_index] = (
                            self.compare_result.group_index, dist
                        )
                        self.compare_result.files_grouped[diff_index] = (
                            self.compare_result.group_index, dist
                        )
                        self.compare_result.group_index += 1
                    else:
                        self.compare_result.files_grouped[base_index] = (
                            existing_group, dist
                        )

        return_current, should_restart = self._validate_checkpoint_data()
        if should_restart:
            return self.run_comparison(store_checkpoints=store_checkpoints)
        if return_current:
            return (self.compare_result.files_grouped, self.compare_result.file_groups)

        for file_index in self.compare_result.files_grouped:
            _file = self.compare_data.files_found[file_index]
            group_index, dist = self.compare_result.files_grouped[file_index]
            file_group = self.compare_result.file_groups.get(group_index, {})
            file_group[_file] = dist
            self.compare_result.file_groups[group_index] = file_group

        self.compare_result.finalize_group_result()
        return (self.compare_result.files_grouped, self.compare_result.file_groups)

    def run(self, store_checkpoints: bool = False):
        if self.is_run_search:
            return self.run_search()
        return self.run_comparison(store_checkpoints=store_checkpoints)

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def remove_from_groups(self, removed_files: list = []) -> None:
        remove_indexes = []
        for f in removed_files:
            if f in self.compare_data.files_found:
                remove_indexes.append(self.compare_data.files_found.index(f))
        remove_indexes.sort()
        if len(self._file_histograms) > 0:
            self._file_histograms = np.delete(
                self._file_histograms, remove_indexes, axis=0
            )
        for f in removed_files:
            if f in self.compare_data.files_found:
                self.compare_data.files_found.remove(f)

    @staticmethod
    def is_related(media1: str, media2: str) -> bool:
        # Cached histogram lookup not yet implemented; always falls through.
        return False


def _usage():
    print("  Option                 Function                                 Default")
    print("      --dir=dirpath      Set base directory                       .      ")
    print("      --counter=int      Set counter cutoff for processing files  10000  ")
    print("  -h, --help             Print help                                      ")
    print("      --include=pattern  File inclusion pattern                          ")
    print("      --search=filepath  Search for similar files to file         None   ")
    print("  -o, --overwrite        Overwrite saved image data               False  ")
    print("      --threshold=float  Histogram distance threshold (0–1)       0.2    ")
    print("  -v                     Verbose                                         ")


if __name__ == "__main__":
    from utils.utils import Utils

    base_dir = "."
    overwrite = False
    search_media_path = None
    verbose = False
    include_gifs = False
    counter_limit = 10000
    file_filter = None
    threshold = None

    try:
        opts, _args = getopt.getopt(sys.argv[1:], "hov", [
            "help", "overwrite", "dir=", "counter=", "include=",
            "search=", "threshold="])
    except getopt.GetoptError as err:
        print(err)
        _usage()
        sys.exit(2)

    for o, a in opts:
        try:
            if o == "-v":
                verbose = True
            elif o in ("-h", "--help"):
                _usage()
                sys.exit()
            elif o == "--counter":
                counter_limit = int(a)
            elif o == "--dir":
                base_dir = a
                if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
                    raise ValueError("Invalid directory: " + base_dir)
            elif o == "--include":
                file_filter = a
            elif o in ("-o", "--overwrite"):
                overwrite = True
                confirm = input("Confirm overwriting image data (y/n): ")
                if confirm not in ("y", "Y"):
                    print("No change made.")
                    sys.exit()
            elif o == "--search":
                search_media_path = Utils.get_valid_file(base_dir, a)
                if search_media_path is None:
                    raise ValueError(
                        f'Search file "{a}" is invalid — ensure --dir is passed first '
                        "if not providing a full file path."
                    )
            elif o == "--threshold":
                threshold = float(a)
            else:
                raise ValueError("Unhandled option: " + o)
        except Exception as e:
            print(e)
            print("")
            _usage()
            sys.exit(1)

    args = CompareArgs(
        base_dir=base_dir,
        compare_threshold=threshold,
        search_media_path=search_media_path,
        counter_limit=counter_limit,
        file_filter=file_filter,
        overwrite=overwrite,
    )
    args.include_gifs = include_gifs
    compare = CompareColorHistogram(args=args)
    compare.get_files()
    compare.get_data()
    compare.run()

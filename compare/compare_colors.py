from collections import Counter
import getopt
import os
import sys

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
# from imutils import face_utils

from PIL import Image

from compare.base_compare import BaseCompare, gather_files
from compare.compare_args import CompareArgs
from compare.compare_result import CompareResult
from image.frame_cache import FrameCache
from utils.config import config
from utils.constants import CompareMode
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

logger = get_logger("compare_colors")


def usage():
    print("  Option                 Function                                 Default")
    print("      --dir=dirpath      Set base directory                       .      ")
    print("      --counter=int      Set counter cutoff for processing files  10000  ")
    print("  -h, --help             Print help                                      ")
    print("      --include=pattern  File inclusion pattern                          ")
    print("      --search=filepath  Search for similar files to file         None   ")
    print("      --use_thumb=bool   Set compare thumbs or color averages     True   ")
    print("  -o, --overwrite        Overwrite saved image data               False  ")
    print("      --threshold=int    Color diff threshold                     15     ")
    print("  -v                     Verbose                                         ")


def get_median_values(ndarray1d):
    medians = []
    for i in range(0, 3):
        if len(ndarray1d) == 0:
            return medians
        medians.append(np.median(ndarray1d))
        ndarray1d = np.take(ndarray1d, np.nonzero(ndarray1d != medians[i])[0])
    return medians


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(
        int(color[0]), int(color[1]), int(color[2]))


def get_image_thumb_colors(image, thumb_dim):
    '''
    Normalize and reduce the size of the image array, and return the colors in LAB
    '''
    modified_image = cv2.resize(
        image, (thumb_dim, thumb_dim), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(
        modified_image.shape[0]*modified_image.shape[1], 3)
    return rgb2lab(np.uint8(modified_image))


def get_image_colors(image, clf, show_chart=False):
    '''
    Get the set of the most significant colors in the image array
    '''
    modified_image = cv2.resize(
        image, (300, 300), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(
        modified_image.shape[0]*modified_image.shape[1], 3)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    lab_colors = [rgb2lab(np.uint8(np.asarray(ordered_colors[i])))
                  for i in counts.keys()]
    if show_chart:
        hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        plt.figure(figsize=(8, 6))
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    return lab_colors


# TODO improve this comparison alg for non-thumb case


def is_any_x_true_weighted(bool_list, x_threshold):
    '''
    Given a list of boolean values. return 1 (true) only if ... ?
    '''
    count_true = 1
    _count = 0

    for i in range(len(bool_list)):
        if _count == x_threshold and count_true > x_threshold:
            return 1
        if _count > x_threshold and count_true + 1 > x_threshold:
            return 1

    return 0


def is_any_x_true_consecutive(bool_list, x_threshold):
    '''
    Given a list of boolean values, return 1 (true) only if at least
    x_threshold of them are consecutively true.
    '''
    count_true = 1
    prior_bool = False
    consecutive_count_true = 1
    consecutive_runs_true = 0
    consecutive_threshold = 10
    consecutive_run_threshold = 10

    for _bool in bool_list:
        if _bool:
            count_true += 1
            if prior_bool:
                consecutive_count_true += 1
                if consecutive_count_true > consecutive_threshold:
                    consecutive_runs_true += 1
        prior_bool = _bool
        if (count_true > x_threshold
                and consecutive_runs_true > consecutive_run_threshold):
            return 1

    return 0


def get_image_array(filepath):
    '''
    If this is a GIF or video file, return the array from the first frame only.

    If the image is grayscale, raise a ValueError. If the image has more
    dimensions than standard RGB, reshape it to RGB.
    '''
    try:
        image = imageio.imread(filepath)
    except Image.DecompressionBombError:
        raise
    except Exception as e:
        filepath = FrameCache.get_image_path(filepath)
        image = imageio.imread(filepath)
    image_shape = np.shape(image)
    if (len(image_shape) < 3 or image_shape[0] < 1 or image_shape[1] < 1
            or image_shape[2] < 1):
        raise ValueError
    if image_shape[2] > 3:
        image = image[:, :, 0:3]
    return image


class CompareColors(BaseCompare):
    COMPARE_MODE = CompareMode.COLOR_MATCHING
    CACHE_FILENAME_TOP   = "image_top_colors.pkl"
    CACHE_FILENAME_THUMB = "image_thumb_colors.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = 50
    THRESHHOLD_PROBABLE_MATCH = 1000
    THRESHHOLD_GROUP_CUTOFF = 4500

    def __init__(self, args=CompareArgs(), use_thumb=True, gather_files_func=gather_files):
        self.use_thumb = use_thumb
        self.CACHE_FILENAME = (CompareColors.CACHE_FILENAME_THUMB if use_thumb
                               else CompareColors.CACHE_FILENAME_TOP)
        super().__init__(args, gather_files_func)
        if self.use_thumb:
            self.thumb_dim = 15
            self.n_colors = self.thumb_dim ** 2
            self.colors_below_threshold = int(self.n_colors / 2)
            self.color_diff_threshold = self.args.threshold
            if self.color_diff_threshold is None:
                self.color_diff_threshold = 15
            self.modifier = self.thumb_dim
            self.color_getter = get_image_thumb_colors
            self.color_diff_alg = is_any_x_true_consecutive
        else:
            self.n_colors = 8
            self.colors_below_threshold = int(self.n_colors * 4 / 8)
            self.color_diff_threshold = self.args.threshold
            if self.color_diff_threshold is None:
                self.color_diff_threshold = 15
            self.modifier = KMeans(n_clusters=self.n_colors)
            self.color_getter = get_image_colors
            self.color_diff_alg = is_any_x_true_weighted
        self._file_colors = np.empty((0, self.n_colors, 3))
        self.settings_updated = False
        self._probable_duplicates = []

    def print_settings(self):
        logger.info("|--------------------------------------------------------------------|")
        logger.info(" CONFIGURATION SETTINGS:")
        logger.info(f" run search: {self.is_run_search}")
        if self.is_run_search:
            logger.info(f" search_media_path: {self.search_media_path}")
        logger.info(f" comparison files base directory: {self.base_dir}")
        logger.info(f" use thumb: {self.use_thumb}")
        logger.info(f" max file process limit: {self.args.counter_limit}")
        logger.info(
            f" max files processable for base dir: {self.max_files_processed}")
        logger.info(f" recursive: {self.args.recursive}")
        logger.info(f" file glob pattern: {self.args.file_filter}")
        logger.info(f" include gifs: {self.args.include_videos}")
        logger.info(f" n colors: {self.n_colors}")
        logger.info(f" colors below threshold: {self.colors_below_threshold}")
        logger.info(f" color diff threshold: {self.color_diff_threshold}")
        logger.info(
            f" file colors filepath: {self.compare_data._file_data_filepath}")
        logger.info(f" modifier: {self.modifier}")
        logger.info(f" color getter: {self.color_getter}")
        logger.info(f" color diff alg: {self.color_diff_alg}")
        logger.info(f" overwrite image data: {self.args.overwrite}")
        logger.info("|--------------------------------------------------------------------|\n\n")

    def get_similarity_threshold(self):
        return self.color_diff_threshold

    def set_similarity_threshold(self, threshold):
        self.color_diff_threshold = threshold

    def get_data(self):
        '''
        For all the found files in the base directory, either load the cached
        image data or extract new data and add it to the cache.
        '''
        self.compare_data.load_data(overwrite=self.args.overwrite)

        # Gather image file data from directory

        if self.verbose:
            logger.info("Gathering image data...")
        else:
            print("Gathering image data", end="", flush=True)

        counter = 0

        for f in self.files:
            if self.is_cancelled():
                self.raise_cancellation_exception()
            
            if Utils.is_invalid_file(f, counter, self.is_run_search, self.args.file_filter):
                continue

            if counter > self.args.counter_limit:
                break

            if f in self.compare_data.file_data_dict:
                colors = self.compare_data.file_data_dict[f]
            else:
                image_file_path = self.get_image_path(f)
                try:
                    image_array = get_image_array(image_file_path)
                except Image.DecompressionBombError as e:
                    logger.warning(f"{f} - skipping, image too large: {e}")
                    continue
                except OSError as e:
                    logger.error(f"{f} - {e}")
                    continue
                except ValueError:
                    continue
                except SyntaxError as e:
                    if self.verbose:
                        logger.error(f"{f} - {e}")
                    # i.e. broken PNG file (bad header checksum in b'tEXt')
                    continue

                try:
                    colors = self.color_getter(image_array, self.modifier)
                except ValueError as e:
                    if self.verbose:
                        logger.error(e)
                        logger.error(f)
                    continue

                self.compare_data.file_data_dict[f] = colors
                self.compare_data.has_new_file_data = True

            counter += 1
            self._file_colors = np.vstack((self._file_colors, [colors]))
            self.compare_data.files_found.append(f)
            self._handle_progress(counter, self.max_files_processed_even)

        # Save image file data
        self.compare_data.save_data(self.args.overwrite, verbose=self.verbose)

    def _compute_color_diff(self, base_array, compare_array,
                            return_diff_scores=False):
        '''
        Perform an elementwise diff between two image color arrays using the
        selected color difference algorithm.
        '''
        lab_diff_squares = np.square(base_array - compare_array)
        deltaE_cie76s = np.sqrt(np.sum(lab_diff_squares, 2)).astype(int)
        similars = np.apply_along_axis(
            self.color_diff_alg, 1, deltaE_cie76s < self.color_diff_threshold,
            self.colors_below_threshold)
        if return_diff_scores:
            return similars, np.sum(deltaE_cie76s, axis=1)
        else:
            return similars

    def find_similars_to_image(self, search_path, search_file_index):
        '''
        Search the numpy array of all known image arrays for similar
        characteristics to the provide image.
        '''
        files_grouped = {}
        _files_found = list(self.compare_data.files_found)

        if self.verbose:
            logger.info("Identifying similar image files...")
        _files_found.pop(search_file_index)
        search_file_colors = self._file_colors[search_file_index]
        file_colors = np.delete(self._file_colors, search_file_index, 0)
        color_similars = self._compute_color_diff(
            file_colors, search_file_colors, True)

        similars = np.nonzero(color_similars[0])

        if config.search_only_return_closest:
            for _index in similars[0]:
                files_grouped[_files_found[_index]] = color_similars[1][_index]
        else:
            temp = {}
            count = 0
            for i in range(len(_files_found)):
                temp[_files_found[i]] = color_similars[1][i]
            for file, difference in dict(sorted(temp.items(), key=lambda item: item[1])).items():
                if count == config.max_search_results:
                    break
                files_grouped[file] = difference
                count += 1

        # Sort results by increasing difference score
        self.compare_result.files_grouped = dict(
            sorted(files_grouped.items(), key=lambda item: item[1]))
        self.compare_result.finalize_search_result(
            self.search_media_path, verbose=self.verbose, is_embedding=False,
            threshold_duplicate=CompareColors.THRESHHOLD_POTENTIAL_DUPLICATE,
            threshold_related=CompareColors.THRESHHOLD_PROBABLE_MATCH)
        return {0: files_grouped}

    def _run_search_on_path(self, search_media_path):
        '''
        Prepare and begin a search for a provided image file path.
        '''
        if (search_media_path is None or search_media_path == ""
                or search_media_path == self.base_dir):
            while search_media_path is None:
                search_media_path = input(
                    "\nEnter a new file path to search for similars "
                    + "(enter \"exit\" or press Ctrl-C to quit): \n\n  > ")
                if search_media_path is not None and search_media_path == "exit":
                    break
                search_media_path = Utils.get_valid_file(
                    self.base_dir, search_media_path)
                if search_media_path is None:
                    logger.error("Invalid filepath provided.")
                else:
                    logger.info("")

        # Gather new image data if it was not in the initial list

        if search_media_path not in self.compare_data.files_found:
            if self.verbose:
                logger.info("Filepath not found in initial list - gathering new file data")
            try:
                image = get_image_array(search_media_path)
            except OSError as e:
                if self.verbose:
                    logger.error(f"{search_media_path} - {e}")
                raise AssertionError(
                    "Encountered an error accessing the provided file path in the file system.")

            try:
                colors = self.color_getter(image, self.modifier)
            except ValueError as e:
                if self.verbose:
                    logger.error(e)
                raise AssertionError(
                    "Encountered an error gathering colors from the file provided.")
            self._file_colors = np.insert(self._file_colors, 0, [colors], 0)
            self.compare_data.files_found.insert(0, search_media_path)

        files_grouped = self.find_similars_to_image(
            search_media_path, self.compare_data.files_found.index(search_media_path))
        search_media_path = None
        return files_grouped

    def run_search(self):
        return self._run_search_on_path(self.search_media_path)

    def run_comparison(self, store_checkpoints=False):
        '''
        Compare all found image arrays to each other by starting with the
        base numpy array containing all image data and moving each array to
        the next index.

        For example, if there are three images [X, Y, Z], there are two steps:
            Step 1: [X, Y, Z] -> [Z, X, Y] (elementwise comparison)
            Step 2: [X, Y, Z] -> [Y, Z, X] (elementwise comparison)
            ^ At this point, all arrays have been compared.

        files_grouped - Keys are the file indexes, values are tuple of the group index and diff score.
        file_groups - Keys are the group indexes, values are dicts with keys as the file in the group, values the diff score
        '''
        overwrite = self.args.overwrite or not store_checkpoints
        self.compare_result = CompareResult.load(
            self.base_dir, self.compare_data.files_found, mode=self.COMPARE_MODE, overwrite=overwrite)
        if self.compare_result.is_complete:
            return (self.compare_result.files_grouped, self.compare_result.file_groups)
        n_files_found_even = Utils.round_up(self.compare_data.n_files_found, 5)
        if self.compare_result.i > 1:
            self._handle_progress(self.compare_result.i, n_files_found_even, gathering_data=False)

        if self.compare_data.n_files_found > 5000:
            logger.warning("\nWARNING: Large image file set found, comparison between all"
                  + " images may take a while.\n")
        if self.verbose:
            logger.info("Identifying groups of similar image files...")
        else:
            print("Identifying groups of similar image files", end="", flush=True)

        # Ensure we have correct counts of data compared to files found
        if len(self.compare_data.files_found) != len(self._file_colors):
            logger.error(f"Warning: Mismatch between files_found ({len(self.compare_data.files_found)}) and file_colors ({len(self._file_colors)})")

        for i in range(self.compare_data.n_files_found):
            if i == 0:  # At this roll index the data would compare to itself
                continue
            if store_checkpoints:
                if i < self.compare_result.i:
                    continue
                if i % 250 == 0 and i != len(self.compare_data.files_found) and i > self.compare_result.i:
                    self.compare_result.store()
                self.compare_result.i = i
            self._handle_progress(i, n_files_found_even, gathering_data=False)

            compare_file_colors = np.roll(self._file_colors, i, 0)
            color_similars = self._compute_color_diff(
                self._file_colors, compare_file_colors, True)

            similars = np.nonzero(color_similars[0])

            for base_index in similars[0]:
                diff_index = ((base_index - i) %
                              self.compare_data.n_files_found)
                diff_score = color_similars[1][base_index]
                f1_grouped = base_index in self.compare_result.files_grouped
                f2_grouped = diff_index in self.compare_result.files_grouped

                if diff_score < CompareColors.THRESHHOLD_POTENTIAL_DUPLICATE:
                    base_file = self.compare_data.files_found[base_index]
                    diff_file = self.compare_data.files_found[diff_index]
                    if ((base_file, diff_file) not in self._probable_duplicates
                            and (diff_file, base_file) not in self._probable_duplicates):
                        self._probable_duplicates.append(
                            (base_file, diff_file))

                if not f1_grouped and not f2_grouped:
                    self.compare_result.files_grouped[base_index] = (
                        self.compare_result.group_index, diff_score)
                    self.compare_result.files_grouped[diff_index] = (
                        self.compare_result.group_index, diff_score)
                    self.compare_result.group_index += 1
                elif f1_grouped:
                    existing_group_index, previous_diff_score = self.compare_result.files_grouped[
                        base_index]
                    if previous_diff_score - CompareColors.THRESHHOLD_GROUP_CUTOFF > diff_score:
                        # logger.debug(f"Previous: {previous_diff_score} , New: {diff_score}")
                        self.compare_result.files_grouped[base_index] = (
                            self.compare_result.group_index, diff_score)
                        self.compare_result.files_grouped[diff_index] = (
                            self.compare_result.group_index, diff_score)
                        self.compare_result.group_index += 1
                    else:
                        self.compare_result.files_grouped[diff_index] = (
                            existing_group_index, diff_score)
                else:
                    existing_group_index, previous_diff_score = self.compare_result.files_grouped[
                        diff_index]
                    if previous_diff_score - CompareColors.THRESHHOLD_GROUP_CUTOFF > diff_score:
                        # logger.debug(f"Previous: {previous_diff_score} , New: {diff_score}")
                        self.compare_result.files_grouped[base_index] = (
                            self.compare_result.group_index, diff_score)
                        self.compare_result.files_grouped[diff_index] = (
                            self.compare_result.group_index, diff_score)
                        self.compare_result.group_index += 1
                    else:
                        self.compare_result.files_grouped[base_index] = (
                            existing_group_index, diff_score)

        # Validate indices before accessing files_found
        return_current_results, should_restart = self._validate_checkpoint_data()
        if should_restart:
            return self.run_comparison(store_checkpoints=store_checkpoints)
        if return_current_results:
            return (self.compare_result.files_grouped, self.compare_result.file_groups)

        for file_index in self.compare_result.files_grouped:
            _file = self.compare_data.files_found[file_index]
            group_index, diff_score = self.compare_result.files_grouped[file_index]
            if group_index in self.compare_result.file_groups:
                file_group = self.compare_result.file_groups[group_index]
            else:
                file_group = {}
            file_group[_file] = diff_score
            self.compare_result.file_groups[group_index] = file_group

        self.compare_result.finalize_group_result()
        return (self.compare_result.files_grouped, self.compare_result.file_groups)

    def run(self, store_checkpoints=False):
        '''
        Runs the specified operation on this Compare.
        '''
        if self.is_run_search:
            return self.run_search()
        else:
            return self.run_comparison(store_checkpoints=store_checkpoints)

    def _sort_groups(self, file_groups):
        return sorted(file_groups,
                      key=lambda group_index: len(file_groups[group_index]))

    def get_probable_duplicates(self):
        return self._probable_duplicates

    def remove_from_groups(self, removed_files=[]):
        # TODO technically it would be better to refresh the file and data lists every time a compare is done
        remove_indexes = []
        for f in removed_files:
            if f in self.compare_data.files_found:
                remove_indexes.append(self.compare_data.files_found.index(f))
        remove_indexes.sort()

        if len(self._file_colors) > 0:
            self._file_colors = np.delete(
                self._file_colors, remove_indexes, axis=0)

        for f in removed_files:
            if f in self.compare_data.files_found:
                self.compare_data.files_found.remove(f)

    @staticmethod
    def is_related(image1, image2):
        # TODO implement this method for this compare mode
        return False


if __name__ == "__main__":
    base_dir = "."
    run_search = False
    overwrite = False
    search_file_index = None
    search_media_path = None
    verbose = False
    include_gifs = False
    use_thumb = True
    counter_limit = 10000
    file_filter = None
    color_diff_threshold = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "bcist:hov", [
            "help",  "overwrite", "dir=", "counter=", "include=",
            "search=", "use_thumb=", "threshold="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for o, a in opts:
        try:
            if o == "-v":
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o == "--counter":
                counter_limit = int(a)
            elif o == "--dir":
                base_dir = a
                if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
                    assert False, "Invalid directory: " + base_dir
            elif o == "--gifs":
                include_gifs = True
            elif o == "--include":
                file_filter = a
            elif o in ("-o", "--overwrite"):
                overwrite = True
                confirm = input("Confirm overwriting image data (y/n): ")
                if confirm != "y" and confirm != "Y":
                    print("No change made.")
                    exit()
            elif o == "--search":
                search_media_path = Utils.get_valid_file(base_dir, a)
                run_search = True
                if search_media_path is None:
                    assert False, "Search file provided \"" + str(a) \
                        + "\" is invalid - please ensure \"dir\" is passed first" \
                        + " if not providing full file path."
            elif o == "--threshold":
                color_diff_threshold = int(a)
            elif o == "--use_thumb":
                use_thumb = a != "False" and a != "false" and a != "f"
            else:
                assert False, "unhandled option " + o
        except Exception as e:
            print(e)
            print("")
            usage()
            exit(1)

    compare = CompareColors(base_dir,
                      search_media_path=search_media_path,
                      counter_limit=counter_limit,
                      use_thumb=use_thumb,
                      color_diff_threshold=color_diff_threshold,
                      file_filter=file_filter,
                      overwrite=overwrite, verbose=verbose,
                      include_gifs=include_gifs)
    compare.get_files()
    compare.get_data()
    compare.run()

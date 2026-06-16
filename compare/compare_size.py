import os
import re
from typing import Tuple, Optional

from PIL import Image

from compare.base_compare import BaseCompare, gather_files
from utils.pillow_plugins import ensure_pillow_plugins_registered

ensure_pillow_plugins_registered()
from compare.compare_args import CompareArgs
from compare.compare_data import CompareData
from compare.compare_result import CompareResult
from image.frame_cache import FrameCache
from utils.config import config
from utils.constants import CompareMode
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

logger = get_logger("compare_size")


def extract_size_from_media(media_path: str) -> Optional[Tuple[int, int]]:
    """
    Extract width and height from a media file.
    Returns (width, height) tuple or None if extraction fails.
    """
    try:
        image_path = FrameCache.get_image_path(media_path)
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        logger.error(f"Error extracting size from {media_path}: {e}")
        return None


def parse_size_search(search_text: str) -> Optional[Tuple[int, int]]:
    """
    Parse size search text in formats like "512x512", "1024,768", "512 512", etc.
    Returns (width, height) tuple or None if parsing fails.
    """
    if not search_text or not search_text.strip():
        return None
    
    # Try various formats: "512x512", "1024,768", "512 512", etc.
    patterns = [
        r'(\d+)\s*[xX]\s*(\d+)',  # "512x512" or "512 x 512"
        r'(\d+)\s*,\s*(\d+)',      # "1024,768" or "1024, 768"
        r'(\d+)\s+(\d+)',          # "512 512"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, search_text.strip())
        if match:
            try:
                width = int(match.group(1))
                height = int(match.group(2))
                return (width, height)
            except ValueError:
                continue
    
    return None


def size_similarity(size1: Tuple[int, int], size2: Tuple[int, int], tolerance: int = 0) -> float:
    """
    Calculate similarity between two sizes.
    Returns 1.0 if sizes match exactly (within tolerance), 0.0 otherwise.
    """
    if size1 is None or size2 is None:
        return 0.0
    
    width_diff = abs(size1[0] - size2[0])
    height_diff = abs(size1[1] - size2[1])
    
    if width_diff <= tolerance and height_diff <= tolerance:
        # Exact match within tolerance
        if tolerance == 0:
            return 1.0
        # Scale similarity based on tolerance
        width_sim = 1.0 - (width_diff / max(tolerance, 1))
        height_sim = 1.0 - (height_diff / max(tolerance, 1))
        return (width_sim + height_sim) / 2.0
    
    return 0.0


class CompareSize(BaseCompare):
    COMPARE_MODE = CompareMode.SIZE
    CACHE_FILENAME = "image_sizes.pkl"
    THRESHOLD_MATCH = 0.95  # For exact size matches

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self.threshold_match = CompareSize.THRESHOLD_MATCH
        self.threshold_tolerance = 0  # Pixel tolerance for size matching
        self.settings_updated = False
        # Initialize compare_data for size comparison
        self.compare_data = CompareData(base_dir=self.base_dir, data_filename=CompareSize.CACHE_FILENAME)
        # Set initial tolerance from args
        if hasattr(args, 'threshold'):
            self.set_similarity_threshold(args.threshold)

    def set_base_dir(self, base_dir):
        '''
        Set the base directory and prepare cache file references.
        '''
        self.base_dir = base_dir
        self.compare_data = CompareData(base_dir=base_dir, data_filename=CompareSize.CACHE_FILENAME)

    def set_search_media_path(self, search_media_path):
        '''
        Set the search file path. If it is already in the found data, move the
        reference to it to the first index in the list.
        '''
        self.search_media_path = search_media_path
        self.is_run_search = search_media_path is not None
        if self.is_run_search and self.files is not None:
            if self.search_media_path in self.files:
                self.files.remove(self.search_media_path)
            self.search_file_index = 0
            self.files.insert(self.search_file_index, self.search_media_path)

    def get_files(self):
        '''
        Get all image files in the base dir as requested by the parameters.
        '''
        self._files_found = []
        if self.gather_files_func:
            exts = config.image_types
            if self.args.include_gifs:
                exts.append(".gif")
            self.files = self.gather_files_func(base_dir=self.base_dir, exts=exts, recursive=self.args.recursive)
        else:
            raise Exception("No gather files function found.")
        self.files.sort()
        self.has_new_file_data = False
        self.max_files_processed = min(self.args.counter_limit, len(self.files))
        self.max_files_processed_even = Utils.round_up(self.max_files_processed, 200)

        if self.is_run_search:
            if self.search_media_path in self.files:
                self.files.remove(self.search_media_path)
            self.search_file_index = 0
            self.files.insert(self.search_file_index, self.search_media_path)

        if self.verbose:
            self.print_settings()

    def print_settings(self):
        logger.info("|--------------------------------------------------------------------|")
        logger.info(" CONFIGURATION SETTINGS:")
        logger.info(f" run search: {self.is_run_search}")
        if self.is_run_search:
            logger.info(f" search_media_path: {self.search_media_path}")
        logger.info(f" comparison files base directory: {self.base_dir}")
        logger.info(f" max file process limit: {self.args.counter_limit}")
        logger.info(f" max files processable for base dir: {self.max_files_processed}")
        logger.info(f" recursive: {self.args.recursive}")
        logger.info(f" file glob pattern: {self.args.file_filter}")
        logger.info(f" include gifs: {self.args.include_gifs}")
        logger.info(f" file sizes filepath: {self.compare_data._file_data_filepath}")
        logger.info(f" overwrite image data: {self.args.overwrite}")
        logger.info("|--------------------------------------------------------------------|\n\n")

    def get_similarity_threshold(self):
        return self.threshold_match

    def set_similarity_threshold(self, threshold):
        # For size comparison, threshold is used as pixel tolerance
        # Convert to int if it's a string
        if isinstance(threshold, str):
            try:
                self.threshold_tolerance = int(threshold)
            except ValueError:
                self.threshold_tolerance = 0
        else:
            self.threshold_tolerance = int(threshold) if isinstance(threshold, (int, float)) else 0
        self.threshold_match = CompareSize.THRESHOLD_MATCH  # Keep match threshold for similarity scoring

    def get_data(self):
        '''
        For all the found files in the base directory, either load the cached
        size data or extract new data and add it to the cache.
        '''
        self.compare_data.load_data(overwrite=self.args.overwrite)

        if self.verbose:
            logger.info("Gathering size data...")
        else:
            print("Gathering size data", end="", flush=True)

        counter = 0

        for f in self.files:
            if self.is_cancelled():
                self.raise_cancellation_exception()
            
            if Utils.is_invalid_file(f, counter, self.is_run_search, self.args.file_filter):
                continue

            if counter > self.args.counter_limit:
                break

            if f in self.compare_data.file_data_dict:
                size = self.compare_data.file_data_dict[f]
            else:
                size = extract_size_from_media(f)
                if size is None:
                    # Skip files where size extraction fails
                    if self.verbose:
                        logger.debug(f"Could not extract size from {f}, skipping")
                    continue
                self.compare_data.file_data_dict[f] = size
                self.compare_data.has_new_file_data = True

            counter += 1
            self.compare_data.files_found.append(f)
            self._handle_progress(counter, self.max_files_processed_even)

        # Save size data
        self.compare_data.save_data(self.args.overwrite, verbose=self.verbose)

    def find_similars_to_media(self, search_path, search_file_index):
        '''
        Search for media with similar sizes to the provided media file.
        '''
        files_grouped = {}
        _files_found = list(self.compare_data.files_found)

        if self.verbose:
            logger.info("Identifying similar size files...")

        # Get the search media's size
        if search_path in self.compare_data.file_data_dict:
            search_size = self.compare_data.file_data_dict[search_path]
        else:
            search_size = extract_size_from_media(search_path)
            if search_size is None:
                if self.verbose:
                    logger.warning(f"Could not extract size from search image {search_path}")
                return {0: {}}
            self.compare_data.file_data_dict[search_path] = search_size

        # Remove search file from comparison list
        if search_file_index < len(_files_found):
            _files_found.pop(search_file_index)

        # Use tolerance from set_similarity_threshold
        tolerance = getattr(self, 'threshold_tolerance', 0)

        # Compare with all other files
        for file_path in _files_found:
            if file_path in self.compare_data.file_data_dict:
                file_size = self.compare_data.file_data_dict[file_path]
            else:
                continue  # Skip files without size data
            
            # Calculate similarity score
            similarity = size_similarity(search_size, file_size, tolerance=tolerance)
            
            if similarity >= self.threshold_match:
                files_grouped[file_path] = similarity

        # Sort results by decreasing similarity score
        self.compare_result.files_grouped = dict(
            sorted(files_grouped.items(), key=lambda item: item[1], reverse=True))
        self.compare_result.finalize_search_result(
            self.search_media_path, verbose=self.verbose, is_embedding=False,
            threshold_duplicate=self.threshold_match,
            threshold_related=self.threshold_match)
        return {0: files_grouped}

    def search_multimodal(self):
        '''
        Search for images matching the provided size search.
        Supports search via search_media_path (extract size from image) or search_text (parse size string).
        '''
        files_grouped = {0: {}}

        search_size = None

        # If a search media file is provided, extract its size
        if self.args.search_media_path is not None:
            search_size = extract_size_from_media(self.args.search_media_path)
            if search_size is None:
                logger.error(f"Could not extract size from search media: {self.args.search_media_path}")
                return files_grouped

        # If search text is provided, parse it as a size
        if search_size is None and self.args.search_text is not None and self.args.search_text.strip() != "":
            search_size = parse_size_search(self.args.search_text)
            if search_size is None:
                logger.error(f"Could not parse size from search text: {self.args.search_text}")
                logger.error("Expected format: '512x512', '1024,768', or '512 512'")
                return files_grouped

        if search_size is None:
            logger.error("No size search criteria provided. Use search_media_path or search_text with size format.")
            return files_grouped

        # Use tolerance from set_similarity_threshold
        tolerance = getattr(self, 'threshold_tolerance', 0)

        # Compute similarity against each file's stored size
        temp_scores = {}
        for file_path in self.compare_data.files_found:
            if file_path in self.compare_data.file_data_dict:
                file_size = self.compare_data.file_data_dict[file_path]
            else:
                continue
            
            similarity = size_similarity(search_size, file_size, tolerance=tolerance)
            
            if similarity >= self.threshold_match:
                temp_scores[file_path] = similarity

        # Order and cap results
        sorted_items = sorted(temp_scores.items(), key=lambda item: item[1], reverse=True)
        if config.search_only_return_closest:
            files_grouped[0] = dict(sorted_items)
        else:
            files_grouped_limited = {}
            for idx, (fp, score) in enumerate(sorted_items):
                if idx == config.max_search_results:
                    break
                files_grouped_limited[fp] = score
            files_grouped[0] = files_grouped_limited

        # Finalize
        self.compare_result.files_grouped = files_grouped[0]
        self.compare_result.finalize_search_result(
            self.args.search_media_path, verbose=self.verbose, is_embedding=False,
            threshold_duplicate=self.threshold_match,
            threshold_related=self.threshold_match)
        return files_grouped

    def run_search(self):
        return self.search_multimodal()

    def run_comparison(self, store_checkpoints=False):
        '''
        Group comparison is not supported for size comparison.
        '''
        raise Exception("Group comparison is not supported for size comparison mode. Use search mode instead.")

    def run(self, store_checkpoints=False):
        '''
        Runs the specified operation on this Compare.
        '''
        # Treat presence of search text or search file as a search request
        has_search = (
            (self.args.search_text is not None and self.args.search_text.strip() != "") or
            (self.args.search_media_path is not None and self.args.search_media_path.strip() != "")
        )

        if has_search or self.is_run_search:
            return self.run_search()
        else:
            return self.run_comparison(store_checkpoints=store_checkpoints)

    def remove_from_groups(self, removed_files=[]):
        for f in removed_files:
            if f in self.compare_data.files_found:
                self.compare_data.files_found.remove(f)
            if f in self.compare_data.file_data_dict:
                del self.compare_data.file_data_dict[f]

    @staticmethod
    def is_related(media1, media2):
        """
        Determine relation by comparing media sizes.
        Media files are considered related if they have the same dimensions.
        """
        try:
            size1 = extract_size_from_media(media1)
            size2 = extract_size_from_media(media2)
        except OSError as e:
            logger.error(f"{media1} or {media2} - {e}")
            raise AssertionError(
                "Encountered an error accessing the provided file paths in the file system.")
        except Exception as e:
            logger.error(e)
            return False

        if size1 is None or size2 is None:
            return False

        return size1 == size2


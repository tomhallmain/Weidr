from glob import glob
import os
import random
import sys

import cv2
import numpy as np

from compare.compare_args import CompareArgs
from compare.compare_data import CompareData
from compare.compare_result import CompareResult
from image.frame_cache import FrameCache
from utils.config import config
from utils.constants import CompareMode
from utils.logging_setup import get_logger
from utils.translations import I18N
from utils.utils import Utils

_ = I18N._

logger = get_logger("base_compare")


def gather_files(base_dir=".", exts=config.image_types, recursive=True, include_videos=False, include_gifs=False, include_pdfs=False):
    files = []
    recursive_str = "**/" if recursive else ""
    exts = exts[:]
    
    # Add video types if enabled (excluding GIFs)
    if include_videos:
        for ext in config.video_types:
            if ext != '.gif' and ext not in exts:
                exts.append(ext)
    else:
        exts = [e for e in exts if e not in config.video_types or e == '.gif']
    
    # Add GIF if enabled
    if include_gifs and '.gif' not in exts:
        exts.append('.gif')
    elif not include_gifs and '.gif' in exts:
        exts.remove('.gif')
    
    # Add PDF if enabled
    if include_pdfs and '.pdf' not in exts:
        exts.append('.pdf')
    elif not include_pdfs and '.pdf' in exts:
        exts.remove('.pdf')
    
    for ext in exts:
        pattern = os.path.join(base_dir, recursive_str + "*" + ext)
        files.extend(glob(pattern, recursive=recursive))
    return files


class CompareCancelled(Exception):
    pass


class BaseCompare:
    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        self.args = args
        self.files = []
        self.set_base_dir(self.args.base_dir)
        self.set_search_media_path(self.args.search_media_path)
        # Keep search-mode state centralized from full args (positive/negative
        # image and text inputs), not just positive search image path.
        self.sync_search_state()
        self.compare_faces = self.args.compare_faces
        # self.args.match_dims = match_dims
        self.verbose = self.args.verbose
        self.progress_listener = self.args.listener
        self._faceCascade = None
        self._cancelled = False
        if self.compare_faces:
            self._set_face_cascade()
        self.gather_files_func = gather_files_func
        self.compare_result = CompareResult(base_dir=self.args.base_dir, mode=self.args.compare_mode)

    def is_runnable(self):
        return True

    def cancel(self):
        """Signal that the compare operation should be cancelled."""
        self._cancelled = True

    def is_cancelled(self):
        """Check if the compare operation has been cancelled."""
        return self._cancelled

    def raise_cancellation_exception(self):
        """Raise CompareCancelled exception."""
        raise CompareCancelled("Compare cancelled by user")

    @staticmethod
    def calculate_chunk_size(embeddings, max_mem_gb=None):
        """
        Calculate the number of rows (M) to process per chunk.
        :param embeddings: N x D numpy array of embeddings.
        :param max_mem_gb: Maximum memory to allocate for a chunk (e.g., 4 GB).
        """
        if max_mem_gb is None or max_mem_gb < 0:
            max_mem_gb = Utils.calculate_available_ram() - 1.0
        n, d = embeddings.shape
        bytes_per_row = d * embeddings.dtype.itemsize  # e.g., 512 * 4 bytes (float32)
        max_rows_per_chunk = int((max_mem_gb * 1e9) / (n * bytes_per_row))
        return max(1, max_rows_per_chunk)  # Ensure at least 1 row per chunk

    @staticmethod
    def chunked_similarity(embeddings, threshold=0.9):
        """
        Compute pairwise similarities in memory-efficient chunks.
        NOTE: This is a performance-inefficient implementation for documentation purposes.
        :returns: List of (i, j, similarity) tuples where similarity > threshold.
        """
        n = embeddings.shape[0]
        memory = Utils.calculate_available_ram()
        chunk_size = BaseCompare.calculate_chunk_size(embeddings, max_mem_gb=(memory / 2))
        similar_pairs = []

        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            chunk = embeddings[i_start:i_end]  # M x D

            # Compute similarities for this chunk against all embeddings
            chunk_similarity = chunk @ embeddings.T  # M x N

            # Find indices where similarity exceeds threshold (upper triangle only)
            for i in range(i_start, i_end):
                for j in range(i + 1, n):  # Upper triangle (i < j)
                    sim = chunk_similarity[i - i_start, j]
                    if sim > threshold:
                        similar_pairs.append((i, j, sim))

        return similar_pairs

    @staticmethod
    def chunked_similarity_vectorized(embeddings, threshold=0.9, decimals=9):
        """
        Compute pairwise similarities in memory-efficient chunks using vectorized operations.
        :returns: List of (i, j, similarity) tuples where similarity > threshold.
        """
        n = embeddings.shape[0]
        memory = Utils.calculate_available_ram()
        chunk_size = BaseCompare.calculate_chunk_size(embeddings, max_mem_gb=(memory / 2))
        similar_pairs = []

        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            chunk = embeddings[i_start:i_end]  # M x D

            # Compute all similarities for this chunk (M x N)
            chunk_sim = chunk @ embeddings.T

            # Create mask for upper triangle (i < j) and threshold
            j_indices = np.arange(n)
            global_i = i_start + np.arange(chunk.shape[0])[:, None]  # Local to global row indices
            mask = (j_indices > global_i) & (chunk_sim > threshold)

            # Extract valid (i, j, sim) triplets
            local_i, j = np.where(mask)
            global_i = i_start + local_i
            similarities = np.round(chunk_sim[mask], decimals=decimals)

            # Extend results
            similar_pairs.extend(zip(global_i, j, similarities))

        return similar_pairs

    def get_similarity_threshold(self):
        return -1.0 # overridden method

    def set_similarity_threshold(self, threshold):
        return None

    def set_base_dir(self, base_dir):
        '''
        Set the base directory and prepare cache file references.
        '''
        self.base_dir = base_dir
        cache_filename = getattr(self, 'CACHE_FILENAME', None)
        if cache_filename is None:
            raise AttributeError(
                f"{type(self).__name__} must define a CACHE_FILENAME class attribute"
            )
        self.compare_data = CompareData(base_dir=base_dir, data_filename=cache_filename)
        self.compare_result = CompareResult(base_dir=base_dir, mode=self.args.compare_mode)

    def set_search_media_path(self, search_media_path):
        '''
        Set the search file path. If it is already in the found data, move the
        reference to it to the first index in the list.
        '''
        self.search_media_path = search_media_path
        self.args.search_media_path = search_media_path
        self.sync_search_state()

    def _has_search_inputs(self) -> bool:
        def _non_empty(value):
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip() != ""
            return True

        return any(
            _non_empty(v)
            for v in (
                self.args.search_media_path,
                self.args.search_text,
                self.args.search_text_negative,
                self.args.negative_search_media_path,
            )
        )

    def sync_search_state(self) -> None:
        """
        Recompute and apply search-mode state from current ``self.args``.
        """
        self.search_media_path = self.args.search_media_path
        self.is_run_search = self._has_search_inputs()
        if (
            self.search_media_path is not None
            and self.files is not None
            and self.is_run_search
        ):
            if self.search_media_path in self.files:
                self.files.remove(self.search_media_path)
            self.search_file_index = 0
            self.files.insert(self.search_file_index, self.search_media_path)

    def get_files(self):
        '''
        Get all image files in the base dir as requested by the parameters.

        To override the default file inclusion behavior, pass a gather_files_func to the Compare object.
        '''
        if self.gather_files_func:
            exts = config.image_types
            self.files = self.gather_files_func(
                base_dir=self.base_dir, exts=exts, recursive=self.args.recursive, include_videos=self.args.include_videos, include_gifs=self.args.include_gifs, include_pdfs=self.args.include_pdfs)
        else:
            raise Exception("No gather files function found.")
        self.files.sort()

        data_filter = getattr(self.args, 'data_filter', None)
        if data_filter and data_filter.is_active():
            from compare.compare_filters import apply_filter
            before = len(self.files)
            self.files = apply_filter(self.files, data_filter)
            # The search file is a reference, not a candidate — keep it even if the
            # filter would exclude it (e.g. size filter, but reference is a diff size)
            if self.is_run_search and self.search_media_path not in self.files:
                logger.debug(
                    "Data filter excluded the search reference file — "
                    "re-adding it so search can still run"
                )
                self.files.insert(0, self.search_media_path)
            logger.debug(
                f"Data filter reduced file list from {before} to {len(self.files)}"
            )

        self.compare_data.has_new_file_data = False
        self.max_files_processed = min(
            self.args.counter_limit, len(self.files))
        self.max_files_processed_even = Utils.round_up(
            self.max_files_processed, 200)

        if self.is_run_search:
            if self.search_media_path in self.files:
                self.files.remove(self.search_media_path)
            self.search_file_index = 0
            self.files.insert(self.search_file_index, self.search_media_path)

        if self.verbose:
            self.print_settings()

    def get_image_path(self, path: str) -> str:
        """
        Get the image path for a file, using FrameCache if needed.
        Returns the original path if no frame extraction is needed.
        """
        return FrameCache.get_image_path(path)

    def get_data(self):
        pass

    def _handle_progress(self, counter, total, gathering_data=True):
        if self.is_cancelled():
            self.raise_cancellation_exception()
        
        percent_complete = counter / total * 100
        if percent_complete % 10 == 0 or counter % 500 == 0:
            if self.verbose:
                desc1 = "data gathered" if gathering_data else "compared"
                logger.info(str(int(percent_complete)) + "% " + desc1)
            else:
                print(".", end="", flush=True)
            if self.progress_listener and sys.platform != "darwin":
                # TODO there is a bug with updating master here on OSX for some reason.
                desc2 = _("Image data collection") if gathering_data else _(
                    "Image comparison")
                self.progress_listener.update(desc2, percent_complete)

    def print_settings(self):
        pass

    def _set_face_cascade(self):
        '''
        Load the face recognition model if compare_faces option was requested.
        '''
        cascPath = ""
        for minor_version in range(10, 5, -1):
            cascPath = "/usr/local/lib/python3." + \
                str(minor_version) + \
                "/site-packages/cv2/data/haarcascade_frontalface_default.xml"
            if os.path.exists(cascPath):
                break
        if not os.path.exists(cascPath):
            logger.warning("WARNING: Face cascade model not found (cv2 package,"
                  + " Python version 3.6 or greater expected)")
            logger.warning("Run with flag --faces=False to avoid this warning.")
            self.compare_faces = False
        else:
            self._faceCascade = cv2.CascadeClassifier(cascPath)

    def _get_faces_count(self, filepath):
        '''
        Try to get the number of faces in the image using the face model.
        '''
        n_faces = random.random() * 10000 + 6  # Set to a value unlikely to match
        try:
            gray = cv2.imread(filepath, 0)
            faces = self._faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            n_faces = len(faces)
        except Exception as e:
            if self.verbose:
                logger.error(e)
        return n_faces

    def run(self):
        pass

    def run_search(self):
        pass

    def run_comparison(self, store_checkpoints=False):
        pass

    def find_similars_to_image(self, search_path, search_file_index):
        pass

    def remove_from_groups(self, removed_files=[]):
        pass

    def readd_files(self, filepaths=[]):
        pass

    def _validate_checkpoint_data(self):
        """
        Validates checkpoint data and handles restart if needed.
        Returns tuple of (return_current_results, should_restart) where:
        - return_current_results is True if validation failed and user cancelled
        - should_restart is True if validation failed and user wants to restart
        """
        if not self.compare_result.validate_indices(self.compare_data.files_found):
            if hasattr(self.args, 'listener') and self.args.listener is not None:
                self.args.listener.display_progress(_("Invalid Checkpoint Data - Confirm Restart"))
            if hasattr(self.args, 'app_actions') and self.args.app_actions is not None:
                if not self.args.app_actions.alert(_("Invalid Checkpoint Data"), 
                    _("Checkpoint data contains invalid indices. Would you like to restart the comparison?"), 
                    kind="askokcancel"):
                    return (True, False)
            self.args.overwrite = True # Ensure the next run will overwrite the checkpoint
            if hasattr(self.args, 'listener') and self.args.listener is not None:
                self.args.listener.display_progress(_("Restarting Comparison"))
            return (False, True)
        return (False, False)
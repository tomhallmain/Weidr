import os
from typing import List, Tuple

from compare.base_compare import BaseCompare, gather_files
from compare.compare_args import CompareArgs
from compare.compare_data import CompareData
from compare.compare_result import CompareResult
from image.image_data_extractor import image_data_extractor
from utils.config import config
from utils.constants import CompareMode
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

logger = get_logger("compare_models")


def extract_models_from_media(media_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract models and loras from a media file.
    Returns (models, loras) tuple.
    """
    try:
        models, loras = image_data_extractor.get_models(media_path)
        return models, loras
    except Exception as e:
        logger.error(f"Error extracting models from {media_path}: {e}")
        return [], []


def model_similarity(models1: List[str], loras1: List[str], 
                     models2: List[str], loras2: List[str]) -> float:
    """
    Calculate similarity between two sets of models and loras.
    Returns value between 0.0 and 1.0, where 1.0 is identical.
    """
    if not models1 and not models2 and not loras1 and not loras2:
        return 1.0  # Both have no models/loras
    
    if not models1 and not loras1:
        return 0.0  # First has no models/loras
    if not models2 and not loras2:
        return 0.0  # Second has no models/loras
    
    # Convert to sets for comparison
    models1_set = set(models1)
    models2_set = set(models2)
    loras1_set = set(loras1)
    loras2_set = set(loras2)
    
    # Calculate model overlap
    model_intersection = len(models1_set & models2_set)
    model_union = len(models1_set | models2_set)
    model_sim = model_intersection / model_union if model_union > 0 else 0.0
    
    # Calculate lora overlap
    lora_intersection = len(loras1_set & loras2_set)
    lora_union = len(loras1_set | loras2_set)
    lora_sim = lora_intersection / lora_union if lora_union > 0 else 0.0
    
    # Weight models more heavily than loras
    combined_sim = (model_sim * 0.7) + (lora_sim * 0.3)
    
    return combined_sim


class CompareModels(BaseCompare):
    COMPARE_MODE = CompareMode.MODELS
    CACHE_FILENAME = "image_models.pkl"
    THRESHOLD_MATCH = 0.7  # Default threshold for model matching

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self.threshold_match = CompareModels.THRESHOLD_MATCH
        self.settings_updated = False
        # Initialize compare_data for model comparison
        self.compare_data = CompareData(base_dir=self.base_dir, data_filename=CompareModels.CACHE_FILENAME)
        # Set initial threshold from args
        if hasattr(args, 'threshold'):
            self.set_similarity_threshold(args.threshold)

    def set_base_dir(self, base_dir):
        '''
        Set the base directory and prepare cache file references.
        '''
        self.base_dir = base_dir
        self.compare_data = CompareData(base_dir=base_dir, data_filename=CompareModels.CACHE_FILENAME)

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
        logger.info(f" file models filepath: {self.compare_data._file_data_filepath}")
        logger.info(f" overwrite image data: {self.args.overwrite}")
        logger.info("|--------------------------------------------------------------------|\n\n")

    def get_similarity_threshold(self):
        return self.threshold_match

    def set_similarity_threshold(self, threshold):
        self.threshold_match = threshold

    def get_data(self):
        '''
        For all the found files in the base directory, either load the cached
        model data or extract new data and add it to the cache.
        '''
        self.compare_data.load_data(overwrite=self.args.overwrite)

        if self.verbose:
            logger.info("Gathering model data...")
        else:
            print("Gathering model data", end="", flush=True)

        counter = 0

        for f in self.files:
            if self.is_cancelled():
                self.raise_cancellation_exception()
            
            if Utils.is_invalid_file(f, counter, self.is_run_search, self.args.file_filter):
                continue

            if counter > self.args.counter_limit:
                break

            if f in self.compare_data.file_data_dict:
                models_data = self.compare_data.file_data_dict[f]
                models, loras = models_data if isinstance(models_data, tuple) and len(models_data) == 2 else ([], [])
            else:
                models, loras = extract_models_from_media(f)
                # Store as tuple (models, loras)
                self.compare_data.file_data_dict[f] = (models, loras)
                self.compare_data.has_new_file_data = True

            counter += 1
            self.compare_data.files_found.append(f)
            self._handle_progress(counter, self.max_files_processed_even)

        # Save model data
        self.compare_data.save_data(self.args.overwrite, verbose=self.verbose)

    def find_similars_to_media(self, search_path, search_file_index):
        '''
        Search for media with similar models to the provided media file.
        '''
        files_grouped = {}
        _files_found = list(self.compare_data.files_found)

        if self.verbose:
            logger.info("Identifying similar model files...")
        
        # Get the search media's models
        if search_path in self.compare_data.file_data_dict:
            search_models_data = self.compare_data.file_data_dict[search_path]
            search_models, search_loras = search_models_data if isinstance(search_models_data, tuple) and len(search_models_data) == 2 else ([], [])
        else:
            search_models, search_loras = extract_models_from_media(search_path)
            self.compare_data.file_data_dict[search_path] = (search_models, search_loras)

        # Remove search file from comparison list
        if search_file_index < len(_files_found):
            _files_found.pop(search_file_index)

        # Compare with all other files
        for i, file_path in enumerate(_files_found):
            if file_path in self.compare_data.file_data_dict:
                file_models_data = self.compare_data.file_data_dict[file_path]
                file_models, file_loras = file_models_data if isinstance(file_models_data, tuple) and len(file_models_data) == 2 else ([], [])
            else:
                continue  # Skip files without model data
            
            # Calculate model similarity only
            similarity = model_similarity(
                search_models, search_loras,
                file_models, file_loras
            )
            
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
        Search for images matching the provided model search.
        Supports search via search_media_path (extract models from image) or search_text (model names).
        Empty search text searches for images with no models.
        '''
        files_grouped = {0: {}}

        search_models = []
        search_loras = []
        search_for_no_models = False

        # If a search media file is provided, extract its models
        if self.args.search_media_path is not None:
            search_models, search_loras = extract_models_from_media(self.args.search_media_path)

        # If search text is provided and not empty, parse it as model names (comma-separated)
        if self.args.search_text is not None and self.args.search_text.strip() != "":
            for model_name in self.args.search_text.split(","):
                model_name = model_name.strip()
                if model_name:
                    search_models.append(model_name)

        # If no models found (empty search text or search media had no models), search for media without models
        if not search_models and not search_loras:
            # Check if we have any search criteria at all
            if self.args.search_media_path is None and (self.args.search_text is None or self.args.search_text.strip() == ""):
                logger.error("No model search criteria provided. Use search_media_path or search_text with model names, or empty search_text to search for images without models.")
                return files_grouped
            # User provided search criteria but it resulted in no models - search for images without models
            search_for_no_models = True

        # Compute similarity against each file's stored models
        temp_scores = {}
        for i, file_path in enumerate(self.compare_data.files_found):
            if file_path in self.compare_data.file_data_dict:
                file_models_data = self.compare_data.file_data_dict[file_path]
                file_models, file_loras = file_models_data if isinstance(file_models_data, tuple) and len(file_models_data) == 2 else ([], [])
            else:
                continue
            
            if search_for_no_models:
                # Search for images with no models/loras
                if not file_models and not file_loras:
                    temp_scores[file_path] = 1.0  # Perfect match - no models
            else:
                # Calculate model similarity only
                similarity = model_similarity(
                    search_models, search_loras,
                    file_models, file_loras
                )
                
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
        Compare all found models to each other.
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
            logger.info("Identifying groups of similar model files...")
        else:
            print("Identifying groups of similar model files", end="", flush=True)

        for i in range(self.compare_data.n_files_found):
            if i == 0:  # Skip self-comparison
                continue
            if store_checkpoints:
                if i < self.compare_result.i:
                    continue
                if i % 250 == 0 and i != len(self.compare_data.files_found) and i > self.compare_result.i:
                    self.compare_result.store()
                self.compare_result.i = i
            self._handle_progress(i, n_files_found_even, gathering_data=False)

            # Get models for current file
            current_file = self.compare_data.files_found[i]
            if current_file not in self.compare_data.file_data_dict:
                continue
            current_models_data = self.compare_data.file_data_dict[current_file]
            current_models, current_loras = current_models_data if isinstance(current_models_data, tuple) and len(current_models_data) == 2 else ([], [])

            # Compare with all other files
            for j in range(i + 1, self.compare_data.n_files_found):
                if self.is_cancelled():
                    self.raise_cancellation_exception()
                
                compare_file = self.compare_data.files_found[j]
                if compare_file not in self.compare_data.file_data_dict:
                    continue
                compare_models_data = self.compare_data.file_data_dict[compare_file]
                compare_models, compare_loras = compare_models_data if isinstance(compare_models_data, tuple) and len(compare_models_data) == 2 else ([], [])

                # Calculate model similarity only
                similarity = model_similarity(
                    current_models, current_loras,
                    compare_models, compare_loras
                )

                # Group similar files
                if similarity >= self.threshold_match:
                    f1_grouped = i in self.compare_result.files_grouped
                    f2_grouped = j in self.compare_result.files_grouped

                    if not f1_grouped and not f2_grouped:
                        self.compare_result.files_grouped[i] = (
                            self.compare_result.group_index, similarity)
                        self.compare_result.files_grouped[j] = (
                            self.compare_result.group_index, similarity)
                        self.compare_result.group_index += 1
                    elif f1_grouped:
                        existing_group_index, previous_similarity = self.compare_result.files_grouped[i]
                        if similarity > previous_similarity:
                            self.compare_result.files_grouped[i] = (
                                self.compare_result.group_index, similarity)
                            self.compare_result.files_grouped[j] = (
                                self.compare_result.group_index, similarity)
                            self.compare_result.group_index += 1
                        else:
                            self.compare_result.files_grouped[j] = (
                                existing_group_index, similarity)
                    else:
                        existing_group_index, previous_similarity = self.compare_result.files_grouped[j]
                        if similarity > previous_similarity:
                            self.compare_result.files_grouped[i] = (
                                self.compare_result.group_index, similarity)
                            self.compare_result.files_grouped[j] = (
                                self.compare_result.group_index, similarity)
                            self.compare_result.group_index += 1
                        else:
                            self.compare_result.files_grouped[i] = (
                                existing_group_index, similarity)

        # Validate indices before accessing files_found
        return_current_results, should_restart = self._validate_checkpoint_data()
        if should_restart:
            return self.run_comparison(store_checkpoints=store_checkpoints)
        if return_current_results:
            return (self.compare_result.files_grouped, self.compare_result.file_groups)

        for file_index in self.compare_result.files_grouped:
            _file = self.compare_data.files_found[file_index]
            group_index, similarity = self.compare_result.files_grouped[file_index]
            if group_index in self.compare_result.file_groups:
                file_group = self.compare_result.file_groups[group_index]
            else:
                file_group = {}
            file_group[_file] = similarity
            self.compare_result.file_groups[group_index] = file_group

        self.compare_result.finalize_group_result()
        return (self.compare_result.files_grouped, self.compare_result.file_groups)

    def run(self, store_checkpoints=False):
        '''
        Runs the specified operation on this Compare.
        '''
        # Treat presence of any search text (or search file) as a search request
        has_text_search = (
            (self.args.search_text is not None and self.args.search_text.strip() != "") or
            (self.args.search_media_path is not None and self.args.search_media_path.strip() != "")
        )

        if self.is_run_search and self.args.search_media_path:
            return self.run_search()
        elif has_text_search:
            return self.search_multimodal()
        else:
            return self.run_comparison(store_checkpoints=store_checkpoints)

    def remove_from_groups(self, removed_files=[]):
        remove_indexes = []
        for f in removed_files:
            if f in self.compare_data.files_found:
                remove_indexes.append(self.compare_data.files_found.index(f))
        remove_indexes.sort()

        for f in removed_files:
            if f in self.compare_data.files_found:
                self.compare_data.files_found.remove(f)
            if f in self.compare_data.file_data_dict:
                del self.compare_data.file_data_dict[f]

    @staticmethod
    def is_related(media1, media2):
        """
        Determine relation by comparing extracted models only.
        """
        try:
            models1, loras1 = extract_models_from_media(media1)
            models2, loras2 = extract_models_from_media(media2)
        except OSError as e:
            logger.error(f"{media1} or {media2} - {e}")
            raise AssertionError(
                "Encountered an error accessing the provided file paths in the file system.")
        except Exception as e:
            logger.error(e)
            return False

        similarity = model_similarity(
            models1, loras1,
            models2, loras2
        )

        # Threshold for considering images related
        return similarity > 0.7


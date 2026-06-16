import os
import re
from typing import List, Tuple, Optional, Any

from compare.base_compare import BaseCompare, gather_files
from compare.compare_args import CompareArgs
from compare.compare_data import CompareData
from compare.compare_result import CompareResult
from image.image_data_extractor import ImageDataExtractor
from utils.config import config
from utils.constants import CompareMode
from utils.logging_setup import get_logger
from utils.translations import _
from utils.utils import Utils

logger = get_logger("compare_prompts_exact")


def _ensure_str(val: Any) -> str:
    """Normalize a value to a string safe for text comparison. Handles None, bytes, dict, and other types."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(val, dict):
        # Old cache format or A1111-style object; avoid using dict as string
        for key in ("positive", "Positive prompt", "text"):
            if key in val and isinstance(val[key], str):
                return val[key]
        return ""
    try:
        return str(val)
    except Exception:
        return ""


def _normalize_search_text_list(val: Any) -> List[str]:
    """Return a list of non-empty stripped strings from search text input. Safe for None, dict, bytes."""
    if val is None:
        return []
    s = _ensure_str(val)
    if not s or not s.strip():
        return []
    return [t.strip() for t in s.split(",") if t and t.strip()]


def _parse_cached_prompts(prompts: Any) -> Tuple[str, str]:
    """Parse cached prompt value (tuple or dict) into (positive_str, negative_str)."""
    if isinstance(prompts, tuple) and len(prompts) == 2:
        return _ensure_str(prompts[0]), _ensure_str(prompts[1])
    if isinstance(prompts, dict):
        pos = prompts.get("positive") or prompts.get("Positive prompt") or ""
        neg = prompts.get("negative") or prompts.get("Negative prompt") or ""
        return _ensure_str(pos), _ensure_str(neg)
    return "", ""


_image_data_extractor = None

def get_image_data_extractor():
    global _image_data_extractor
    if _image_data_extractor is None:
        _image_data_extractor = ImageDataExtractor()
    return _image_data_extractor

def extract_prompts_from_media(media_path):
    try:
        extractor = get_image_data_extractor()
        positive, negative = extractor.extract_prompts_all_strategies(media_path)
        if positive is not None:
            return positive, negative
        return None, None
    except Exception as e:
        logger.error(f"Error extracting prompt from {media_path}: {e}")
        return None, None

def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def _compute_fuzzy_word_similarity(words1: set, words2: set) -> float:
    exact_matches = words1 & words2
    substring_matches = set()
    fuzzy_matches = set()

    for w1 in words1:
        for w2 in words2:
            if w1 != w2 and len(w1) >= 3 and len(w2) >= 3:
                if w1 in w2 or w2 in w1:
                    substring_matches.add((w1, w2))

    for w1 in words1:
        for w2 in words2:
            if w1 != w2 and w1 not in exact_matches and w2 not in exact_matches:
                is_substring_match = any((w1 in w2 or w2 in w1) for w1_check, w2_check in substring_matches 
                                       if (w1_check == w1 and w2_check == w2) or (w1_check == w2 and w2_check == w1))
                if not is_substring_match and len(w1) >= 3 and len(w2) >= 3:
                    distance = _levenshtein_distance(w1, w2)
                    if distance <= 2:
                        fuzzy_matches.add((w1, w2))

    exact_score = len(exact_matches)
    substring_score = len(substring_matches) * 0.8
    fuzzy_score = len(fuzzy_matches) * 0.6

    total_matches = exact_score + substring_score + fuzzy_score
    total_words = len(words1 | words2)

    return total_matches / total_words if total_words > 0 else 0.0

def compute_text_similarity(text1: str, text2: str) -> float:
    text1 = _ensure_str(text1)
    text2 = _ensure_str(text2)
    if not text1 or not text2:
        return 0.0
    text1_lower = text1.lower().strip()
    text2_lower = text2.lower().strip()
    if text1_lower == text2_lower:
        return 1.0
    if text1_lower in text2_lower or text2_lower in text1_lower:
        return 0.9

    def split_structured_elements(text):
        elements = []
        for separator in [',', '\n', '\r\n']:
            elements.extend([elem.strip() for elem in text.split(separator)])
        return [elem for elem in elements if elem]

    elements1 = set(split_structured_elements(text1_lower))
    elements2 = set(split_structured_elements(text2_lower))
    if elements1 and elements2:
        element_intersection = elements1.intersection(elements2)
        element_union = elements1.union(elements2)
        element_similarity = len(element_intersection) / len(element_union) if element_union else 0.0
        if element_similarity > 0:
            if len(element_intersection) >= 2:
                element_similarity = min(0.95, element_similarity * 1.3)
            return element_similarity

    words1 = set(text1_lower.split())
    words2 = set(text2_lower.split())
    if not words1 or not words2:
        return 0.0
    fuzzy_similarity = _compute_fuzzy_word_similarity(words1, words2)
    return min(0.7, fuzzy_similarity)


class ComparePromptsExact(BaseCompare):
    COMPARE_MODE = CompareMode.PROMPTS_EXACT
    CACHE_FILENAME = "image_prompts_exact.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = 0.95  # High similarity threshold for exact matches
    THRESHHOLD_PROBABLE_MATCH = 0.85
    THRESHHOLD_GROUP_CUTOFF = 0.75

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self.threshold_duplicate = ComparePromptsExact.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = ComparePromptsExact.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = ComparePromptsExact.THRESHHOLD_GROUP_CUTOFF
        self._probable_duplicates = []
        self.settings_updated = False
        # Initialize compare_data for prompt comparison
        self.compare_data = CompareData(base_dir=self.base_dir, data_filename=ComparePromptsExact.CACHE_FILENAME)
        # In-memory prompt caches aligned with compare_data.files_found order
        self._file_pos_texts = []
        self._file_neg_texts = []

    def set_base_dir(self, base_dir):
        '''
        Set the base directory and prepare cache file references.
        '''
        self.base_dir = base_dir
        self.compare_data = CompareData(base_dir=base_dir, data_filename=ComparePromptsExact.CACHE_FILENAME)

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

        To override the default file inclusion behavior, pass a gather_files_func to the Compare object.
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
        logger.info(f" file prompts filepath: {self.compare_data._file_data_filepath}")
        logger.info(f" overwrite image data: {self.args.overwrite}")
        logger.info("|--------------------------------------------------------------------|\n\n")

    def get_similarity_threshold(self):
        return self.threshold_probable_match

    def set_similarity_threshold(self, threshold):
        self.threshold_probable_match = threshold

    def get_data(self):
        '''
        For all the found files in the base directory, either load the cached
        prompt data or extract new data and add it to the cache.
        '''
        self.compare_data.load_data(overwrite=self.args.overwrite)

        if self.verbose:
            logger.info("Gathering prompt data...")
        else:
            print("Gathering prompt data", end="", flush=True)

        counter = 0

        for f in self.files:
            if self.is_cancelled():
                self.raise_cancellation_exception()
            
            if Utils.is_invalid_file(f, counter, self.is_run_search, self.args.file_filter):
                continue

            if counter > self.args.counter_limit:
                break

            if f in self.compare_data.file_data_dict:
                prompts = self.compare_data.file_data_dict[f]
            else:
                positive_prompt, negative_prompt = extract_prompts_from_media(f)
                if positive_prompt is None and negative_prompt is None:
                    # Skip files with no prompt data - this is normal for many images
                    # if self.verbose:
                        # logger.debug(f"No prompt data found in {f}, skipping")
                    continue
                prompts = (positive_prompt, negative_prompt)
                self.compare_data.file_data_dict[f] = prompts
                self.compare_data.has_new_file_data = True

            counter += 1
            # For exact matching, we just store the prompt text, no embeddings needed
            self.compare_data.files_found.append(f)
            # Maintain in-memory prompt arrays aligned with files_found; normalize to str
            pos, neg = _parse_cached_prompts(prompts)
            self._file_pos_texts.append(pos)
            self._file_neg_texts.append(neg)
            self._handle_progress(counter, self.max_files_processed_even)

        # Save prompt data
        self.compare_data.save_data(self.args.overwrite, verbose=self.verbose)

    def _populate_text_arrays_from_cache(self):
        '''
        Populate in-memory text arrays from cached compare_data when available.
        Ensures alignment with compare_data.files_found order.
        '''
        self._file_pos_texts = []
        self._file_neg_texts = []
        if self.compare_data is None:
            return
        if getattr(self.compare_data, 'file_data_dict', None) is None:
            # Try to load without regenerating
            self.compare_data.load_data(overwrite=False)
        data = self.compare_data.file_data_dict or {}
        for f in self.compare_data.files_found or []:
            prompts = data.get(f, ("", ""))
            pos, neg = _parse_cached_prompts(prompts)
            self._file_pos_texts.append(pos)
            self._file_neg_texts.append(neg)

    def find_similars_to_media(self, search_path, search_file_index):
        '''
        Search for media with similar prompts to the provided media file using exact text matching.
        Uses in-memory _file_pos_texts / _file_neg_texts when aligned with files_found;
        otherwise falls back to file_data_dict or extraction (e.g. when search file was just added).
        '''
        # Ensure in-memory prompt arrays are populated
        if (len(self._file_pos_texts) != len(self.compare_data.files_found) or
                len(self._file_neg_texts) != len(self.compare_data.files_found)):
            self._populate_text_arrays_from_cache()

        files_grouped = {}
        _files_found = list(self.compare_data.files_found)

        if self.verbose:
            logger.info("Identifying similar prompt files using exact text matching...")

        # Get the search image's prompts from arrays (by index) or extract if missing
        if search_file_index < len(self._file_pos_texts) and search_file_index < len(self._file_neg_texts):
            search_positive = self._file_pos_texts[search_file_index]
            search_negative = self._file_neg_texts[search_file_index]
        else:
            search_positive, search_negative = extract_prompts_from_media(search_path)
            if search_positive is None and search_negative is None:
                if self.verbose:
                    logger.warning(f"No prompt data found in search image {search_path}")
                return {0: {}}
            search_positive = _ensure_str(search_positive)
            search_negative = _ensure_str(search_negative)

        # Remove search file from comparison list
        if search_file_index < len(_files_found):
            _files_found.pop(search_file_index)

        # Compare with all other files using in-memory arrays by index
        for file_path in _files_found:
            try:
                idx = self.compare_data.files_found.index(file_path)
            except ValueError:
                continue
            if idx >= len(self._file_pos_texts) or idx >= len(self._file_neg_texts):
                continue
            file_positive = self._file_pos_texts[idx]
            file_negative = self._file_neg_texts[idx]

            # Calculate similarity scores
            positive_similarity = compute_text_similarity(search_positive, file_positive)
            negative_similarity = compute_text_similarity(search_negative, file_negative)
            
            # Combined similarity (weighted average)
            combined_similarity = (positive_similarity * 0.7) + (negative_similarity * 0.3)
            
            if combined_similarity >= self.threshold_probable_match:
                files_grouped[file_path] = combined_similarity

        # Sort results by decreasing similarity score
        self.compare_result.files_grouped = dict(
            sorted(files_grouped.items(), key=lambda item: item[1], reverse=True))
        self.compare_result.finalize_search_result(
            self.search_media_path, verbose=self.verbose, is_embedding=False,
            threshold_duplicate=self.threshold_duplicate,
            threshold_related=self.threshold_probable_match)
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
                search_media_path = Utils.get_valid_file(self.base_dir, search_media_path)
                if search_media_path is None:
                    logger.error("Invalid filepath provided.")
                else:
                    logger.info("")

        # For exact matching, we don't need to preprocess the search image
        # We just need to make sure it's in our file list and in-memory arrays stay in sync
        if search_media_path not in self.compare_data.files_found:
            if self.verbose:
                logger.info("Filepath not found in initial list - adding to comparison list")
            self.compare_data.files_found.insert(0, search_media_path)
            
            # Extract and store prompts for the search image
            positive_prompt, negative_prompt = extract_prompts_from_media(search_media_path)
            if positive_prompt is None and negative_prompt is None:
                raise AssertionError("No prompt data found in the provided image. This image may not contain prompt metadata.")
            # Keep in-memory arrays in sync (primary source for comparison); update dict only if still in memory
            self._file_pos_texts.insert(0, _ensure_str(positive_prompt))
            self._file_neg_texts.insert(0, _ensure_str(negative_prompt))
            if self.compare_data.file_data_dict is not None:
                self.compare_data.file_data_dict[search_media_path] = (positive_prompt, negative_prompt)

        files_grouped = self.find_similars_to_media(
            search_media_path, self.compare_data.files_found.index(search_media_path))
        search_media_path = None
        return files_grouped

    def run_search(self):
        return self.search_multimodal()
        # return self._run_search_on_path(self.search_media_path)

    def search_multimodal(self):
        '''
        Search for provided search images and text (exact/string-based variant).
        Mirrors the embedding flow but uses text similarity instead of embeddings.
        '''
        files_grouped = {0: {}}

        positive_texts: List[str] = []
        negative_texts: List[str] = []

        # If a search image is provided, use its positive as positive and its negative as negative
        if self.args.search_media_path is not None:
            pos, neg = extract_prompts_from_media(self.args.search_media_path)
            if pos is not None:
                positive_texts.append(_ensure_str(pos))
            if neg is not None:
                negative_texts.append(_ensure_str(neg))

        # If a negative search image is provided, treat its positive as negative context
        if self.args.negative_search_media_path is not None:
            pos, neg = extract_prompts_from_media(self.args.negative_search_media_path)
            if pos:
                negative_texts.append(pos)
            if neg:
                negative_texts.append(neg)

        # Add explicit positive texts
        if self.args.search_text is not None and self.args.search_text.strip() != "":
            for text in self.args.search_text.split(","):
                t = text.strip()
                if t:
                    positive_texts.append(t)

        # Add explicit positive and negative texts (safe for None/dict/bytes)
        positive_texts.extend(_normalize_search_text_list(self.args.search_text))
        negative_texts.extend(_normalize_search_text_list(self.args.search_text_negative))

        if len(positive_texts) == 0 and len(negative_texts) == 0:
            logger.error(
                "Failed to prepare texts for search.\n"
                f"search image = {self.args.search_media_path}\n"
                f"negative search image = {self.args.negative_search_media_path}\n"
                f"search text = {self.args.search_text}\n"
                f"search text negative = {self.args.search_text_negative}")
            return files_grouped

        if (not hasattr(self, '_file_pos_texts') or not hasattr(self, '_file_neg_texts')
                or len(self._file_pos_texts) != len(self.compare_data.files_found)
                or len(self._file_pos_texts) == 0):
            self._populate_text_arrays_from_cache()

        # Compute similarity against each file's stored prompts
        temp_scores = {}
        for idx, file_path in enumerate(self.compare_data.files_found):
            if idx >= len(self._file_pos_texts):
                continue
            file_pos = _ensure_str(self._file_pos_texts[idx] if idx < len(self._file_pos_texts) else "")
            file_neg = _ensure_str(self._file_neg_texts[idx] if idx < len(self._file_neg_texts) else "")

            # Positive score: best match of any positive text against file's positive prompt
            pos_score = 0.0
            for t in positive_texts:
                pos_score = max(pos_score, compute_text_similarity(t, file_pos))

            # Negative score: best match of any negative text against file's positive/negative prompts
            # Note the negative weight is 0.5 compared to the positive weight
            neg_score_candidates = [compute_text_similarity(t, file_pos) for t in negative_texts]
            neg_score_candidates += [compute_text_similarity(t, file_neg) for t in negative_texts]
            neg_score = max(neg_score_candidates) if neg_score_candidates else 0.0

            # Combine: favor positives, penalize negatives
            combined = max(0.0, min(1.0, pos_score - 0.5 * neg_score))

            if combined >= self.threshold_probable_match:
                temp_scores[file_path] = combined

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
            threshold_duplicate=self.threshold_duplicate,
            threshold_related=self.threshold_probable_match)
        return files_grouped

    def run_comparison(self, store_checkpoints=False):
        '''
        Compare all found prompt texts to each other using exact text matching.
        Uses in-memory _file_pos_texts / _file_neg_texts (aligned with files_found), not
        file_data_dict, because save_data() clears file_data_dict to free memory after persist.
        '''
        # Ensure in-memory prompt arrays are populated (from get_data() or from cache)
        if (len(self._file_pos_texts) != len(self.compare_data.files_found) or
                len(self._file_neg_texts) != len(self.compare_data.files_found)):
            self._populate_text_arrays_from_cache()

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
            logger.info("Identifying groups of similar prompt files using exact text matching...")
        else:
            print("Identifying groups of similar prompt files using exact text matching", end="", flush=True)

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

            # Get prompts from in-memory arrays (same pattern as embedding mode using _file_embeddings)
            if i >= len(self._file_pos_texts) or i >= len(self._file_neg_texts):
                continue
            current_positive = self._file_pos_texts[i]
            current_negative = self._file_neg_texts[i]

            # Compare with all other files
            for j in range(i + 1, self.compare_data.n_files_found):
                if self.is_cancelled():
                    self.raise_cancellation_exception()
                
                if j >= len(self._file_pos_texts) or j >= len(self._file_neg_texts):
                    continue
                compare_positive = self._file_pos_texts[j]
                compare_negative = self._file_neg_texts[j]

                # Calculate similarity
                positive_similarity = compute_text_similarity(current_positive, compare_positive)
                negative_similarity = compute_text_similarity(current_negative, compare_negative)
                combined_similarity = (positive_similarity * 0.7) + (negative_similarity * 0.3)

                if combined_similarity >= self.threshold_duplicate:
                    base_file = self.compare_data.files_found[i]
                    diff_file = self.compare_data.files_found[j]
                    if ((base_file, diff_file) not in self._probable_duplicates
                            and (diff_file, base_file) not in self._probable_duplicates):
                        self._probable_duplicates.append((base_file, diff_file))

                # Group similar files
                if combined_similarity >= self.threshold_probable_match:
                    f1_grouped = i in self.compare_result.files_grouped
                    f2_grouped = j in self.compare_result.files_grouped

                    if not f1_grouped and not f2_grouped:
                        self.compare_result.files_grouped[i] = (
                            self.compare_result.group_index, combined_similarity)
                        self.compare_result.files_grouped[j] = (
                            self.compare_result.group_index, combined_similarity)
                        self.compare_result.group_index += 1
                    elif f1_grouped:
                        existing_group_index, previous_similarity = self.compare_result.files_grouped[i]
                        if combined_similarity - previous_similarity > self.threshold_group_cutoff:
                            self.compare_result.files_grouped[i] = (
                                self.compare_result.group_index, combined_similarity)
                            self.compare_result.files_grouped[j] = (
                                self.compare_result.group_index, combined_similarity)
                            self.compare_result.group_index += 1
                        else:
                            self.compare_result.files_grouped[j] = (
                                existing_group_index, combined_similarity)
                    else:
                        existing_group_index, previous_similarity = self.compare_result.files_grouped[j]
                        if combined_similarity - previous_similarity > self.threshold_group_cutoff:
                            self.compare_result.files_grouped[i] = (
                                self.compare_result.group_index, combined_similarity)
                            self.compare_result.files_grouped[j] = (
                                self.compare_result.group_index, combined_similarity)
                            self.compare_result.group_index += 1
                        else:
                            self.compare_result.files_grouped[i] = (
                                existing_group_index, combined_similarity)

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
        # Treat presence of any search text (or negative) as a search request (safe for None/dict)
        has_text_search = bool(
            _normalize_search_text_list(getattr(self.args, "search_text", None))
            or _normalize_search_text_list(getattr(self.args, "search_text_negative", None))
        )

        if self.is_run_search and self.args.search_media_path:
            return self.run_search()
        elif has_text_search:
            return self.search_multimodal()
        else:
            return self.run_comparison(store_checkpoints=store_checkpoints)

    def get_probable_duplicates(self):
        return self._probable_duplicates
    
    def remove_from_groups(self, removed_files=[]):
        # TODO technically it would be better to refresh the file and data lists every time a compare is done
        remove_indexes = []
        for f in removed_files:
            if f in self.compare_data.files_found:
                remove_indexes.append(self.compare_data.files_found.index(f))
        remove_indexes.sort()

        for f in removed_files:
            if f in self.compare_data.files_found:
                self.compare_data.files_found.remove(f)
        # Keep in-memory arrays aligned
        for i in reversed(remove_indexes):
            if 0 <= i < len(self._file_pos_texts):
                self._file_pos_texts.pop(i)
            if 0 <= i < len(self._file_neg_texts):
                self._file_neg_texts.pop(i)

    @staticmethod
    def is_related(media1, media2):
        """
        Determine relation by comparing extracted prompt texts directly.
        Uses the same text similarity as search (exact/substring/structured/word) and
        considers media related if the weighted similarity exceeds a threshold.
        """
        try:
            pos1, neg1 = extract_prompts_from_media(media1)
            pos2, neg2 = extract_prompts_from_media(media2)
        except OSError as e:
            logger.error(f"{media1} or {media2} - {e}")
            raise AssertionError(
                "Encountered an error accessing the provided file paths in the file system.")
        except Exception as e:
            logger.error(e)
            return False

        # Special handling for absent prompts (None indicates no prompt present in metadata)
        has_prompts_1 = not (pos1 is None and neg1 is None)
        has_prompts_2 = not (pos2 is None and neg2 is None)

        # If neither image has prompts at all, consider them related
        if not has_prompts_1 and not has_prompts_2:
            return True

        # If exactly one image has prompts, treat them as unrelated
        if has_prompts_1 != has_prompts_2:
            return False

        # Fallback to empty strings for missing prompts
        pos1 = pos1 or ""
        neg1 = neg1 or ""
        pos2 = pos2 or ""
        neg2 = neg2 or ""

        # Use module-level similarity implementation
        pos_sim = compute_text_similarity(pos1, pos2)
        neg_sim = compute_text_similarity(neg1, neg2)
        combined = (pos_sim * 0.7) + (neg_sim * 0.3)

        # Threshold analogous to embedding case
        return combined > 0.8

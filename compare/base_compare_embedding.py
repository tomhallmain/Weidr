import getopt
import os
import sys

import numpy as np

from compare.base_compare import BaseCompare, gather_files
from compare.compare_args import CompareArgs
from compare.compare_result import CompareResult
from compare.model import embedding_similarity
from utils.config import config
from utils.logging_setup import get_logger
from utils.utils import Utils

logger = get_logger("base_compare_embedding")


class BaseCompareEmbedding(BaseCompare):
    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self.embedding_similarity_threshold = self.args.threshold
        self.settings_updated = False
        self._probable_duplicates = []
        self.segregation_map = {}
        self.image_embeddings_func = None
        self.text_embeddings_func = None
        self.threshold_duplicate = None
        self.threshold_probable_match = None
        self.threshold_group_cutoff = None
        self.text_embedding_cache = {}
        self._file_embeddings = np.empty((0, 512))
        self._file_faces = np.empty((0))

    def get_similarity_threshold(self):
        return self.embedding_similarity_threshold

    def set_similarity_threshold(self, threshold):
        self.embedding_similarity_threshold = threshold

    def print_settings(self):
        logger.info("|--------------------------------------------------------------------|")
        logger.info(" CONFIGURATION SETTINGS:")
        logger.info(f" run search: {self.is_run_search}")
        if self.is_run_search:
            logger.info(f" search_media_path: {self.search_media_path}")
        logger.info(f" comparison files base directory: {self.base_dir}")
        logger.info(f" compare faces: {self.compare_faces}")
        logger.info(f" embedding similarity threshold: {self.embedding_similarity_threshold}")
        logger.info(f" max file process limit: {self.args.counter_limit}")
        logger.info(f" max files processable for base dir: {self.max_files_processed}")
        logger.info(f" recursive: {self.args.recursive}")
        logger.info(f" file glob pattern: {self.args.inclusion_pattern}")
        logger.info(f" include videos: {self.args.include_videos}")
        logger.info(f" file embeddings filepath: {self.compare_data._file_data_filepath}")
        logger.info(f" overwrite image data: {self.args.overwrite}")
        logger.info(f" compare mode: {self.COMPARE_MODE}")
        logger.info("|--------------------------------------------------------------------|\n\n")

    def get_data(self):
        '''
        For all the found files in the base directory, either load the cached
        image data or extract new data and add it to the cache.
        '''
        self.compare_data.load_data(overwrite=self.args.overwrite,
                                    compare_faces=self.compare_faces)

        # Gather image file data from directory

        if self.verbose:
            logger.info("Gathering image data...")
        else:
            print("Gathering image data", end="", flush=True)

        counter = 0

        for f in self.files:
            # Check for cancellation during data gathering
            if self.is_cancelled():
                self.raise_cancellation_exception()
            
            if Utils.is_invalid_file(f, counter, self.is_run_search, self.args.inclusion_pattern):
                continue

            if counter > self.args.counter_limit:
                break

            if f in self.compare_data.file_data_dict:
                embedding = self.compare_data.file_data_dict[f]
                if self.compare_faces:
                    if f in self.compare_data.file_faces_dict:
                        n_faces = self.compare_data.file_faces_dict[f]
                    else:
                        image_file_path = self.get_image_path(f)
                        n_faces = self._get_faces_count(image_file_path)
                        self.compare_data.file_faces_dict[f] = n_faces
            else:
                image_file_path = self.get_image_path(f)
                try:
                    embedding = self.image_embeddings_func(image_file_path)
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
                self.compare_data.file_data_dict[f] = embedding
                if self.compare_faces:
                    if f in self.compare_data.file_faces_dict:
                        n_faces = self.compare_data.file_faces_dict[f]
                    else:
                        n_faces = self._get_faces_count(image_file_path)
                        self.compare_data.file_faces_dict[f] = n_faces
                self.compare_data.has_new_file_data = True

            counter += 1
            self._file_embeddings = np.vstack((self._file_embeddings, [embedding]))
            if self.compare_faces:
                self._file_faces = np.vstack((self._file_faces, [n_faces]))
            self.compare_data.files_found.append(f)
            self._handle_progress(counter, self.max_files_processed_even)

        # Save image file data
        self.compare_data.save_data(self.args.overwrite, verbose=self.verbose,
                                    compare_faces=self.compare_faces)

    def _compute_embedding_diff(self, base_array, compare_array,
                                return_diff_scores=False, threshold=None):
        '''
        Perform an elementwise diff between two image color arrays using the
        selected color difference algorithm.
        '''
        vectorized = np.vectorize(np.dot, signature="(m),(n)->()")
        simlarities = vectorized(base_array, compare_array)
        if threshold is None:
            similars = simlarities > self.embedding_similarity_threshold
        else:
            similars = simlarities > threshold
        if return_diff_scores:
            return similars, simlarities
        else:
            return similars

    def run(self, store_checkpoints=False):
        '''
        Runs the specified operation on this Compare.
        '''
        if self.is_run_search:
            return self.run_search()
        else:
            return self.run_comparison(store_checkpoints=store_checkpoints)

    def run_search(self):
        if self.args.compare_faces:
            return self._run_search_on_path(self.search_media_path)
        else:
            return self.search_multimodal()

    def run_comparison(self, store_checkpoints=False):
        '''
        Compare all found embeddings to each other using either matrix-based or
        iterative comparison based on the use_matrix_comparison flag.

        For matrix comparison:
            Group the embeddings E = [X, Y, Z]
            Calculate L2-norm: N = L2(E)
            If available RAM, simply multiply the normalized matrix by its transpose:
                S = N * N.T
            Otherwise, use chunking to compute the similarity matrix.
                S = concat(chunk(N) * N.T) for each chunk(N)
            Extract similars from the upper triangle:
                i, j = np.triu_indices_from(S, k=1)
            Group the similars by their similarity.

        For iterative comparison:
            Compare all found image arrays to each other by starting with the
            base numpy array containing all image data and moving each array to
            the next index.

            For example, if there are three images [X, Y, Z], there are two steps:
                Step 1: [X, Y, Z] -> [Z, X, Y] (elementwise comparison)
                Step 2: [X, Y, Z] -> [Y, Z, X] (elementwise comparison)
                ^ At this point, all arrays have been compared.
                  Note it is inefficient as pairs are compared twice.

        files_grouped - Keys are the file indexes, values are tuple of the group index and diff score.
        file_groups - Keys are the group indexes, values are dicts with keys as the file in the group, values the diff score
        '''
        overwrite = self.args.overwrite or not store_checkpoints
        logger.debug(f"Store checkpoints: {store_checkpoints}")
        self.compare_result = CompareResult.load(self.base_dir, self.compare_data.files_found, mode=self.COMPARE_MODE, overwrite=overwrite)
        if self.compare_result.is_complete:
            return (self.compare_result.files_grouped, self.compare_result.file_groups)

        # Ensure we have correct counts of data compared to files found
        if len(self.compare_data.files_found) != len(self._file_embeddings):
            logger.error(f"Warning: Mismatch between files_found ({len(self.compare_data.files_found)}) and file_embeddings ({len(self._file_embeddings)})")

        if self.compare_faces and len(self.compare_data.files_found) != len(self._file_faces):
            logger.error(f"Warning: Mismatch between files_found ({len(self.compare_data.files_found)}) and file_faces ({len(self._file_faces)})")

        if self.verbose:
            logger.info("Identifying groups of similar image files...")
        else:
            print("Identifying groups of similar image files", end="", flush=True)

        if self.args.use_matrix_comparison:
            similarity_matrix, _i, _j = self._compute_matrix_similarities()
            for i, j in zip(_i, _j):
                if i == j:  # exclude diagonal (self-comparisons)
                    continue
                base_index = i
                diff_index = j
                diff_score = similarity_matrix[base_index, diff_index]
                if diff_score < self.embedding_similarity_threshold:
                    continue
                self._process_similarity_results(base_index, diff_index, diff_score)
        else:
            n_files_found_even = Utils.round_up(self.compare_data.n_files_found, 5)
            if self.compare_result.i > 1:
                self._handle_progress(self.compare_result.i, n_files_found_even, gathering_data=False)

            if self.compare_data.n_files_found > 5000:
                logger.warning("\nWARNING: Large image file set found, comparison between all"
                                 + " images may take a while.\n")

            for i in range(self.compare_data.n_files_found):
                if i == 0:  # At this roll index the data would compare to itself
                    continue
                if store_checkpoints:
                    if i < self.compare_result.i:
                        continue
                    if i % 250 == 0 and i != self.compare_data.n_files_found and i > self.compare_result.i:
                        self.compare_result.store()
                    self.compare_result.i = i
                self._handle_progress(i, n_files_found_even, gathering_data=False)

                similars, diff_scores = self._compute_iterative_similarities(i)
                for base_index in similars[0]:
                    diff_index = ((base_index - i) % self.compare_data.n_files_found)
                    diff_score = diff_scores[base_index]
                    self._process_similarity_results(base_index, diff_index, diff_score)

        # Validate indices before accessing files_found
        return_current_results, should_restart = self._validate_checkpoint_data()
        if should_restart:
            return self.run_comparison(store_checkpoints=store_checkpoints)
        if return_current_results:
            return (self.compare_result.files_grouped, self.compare_result.file_groups)

        for file_index in self.compare_result.files_grouped:
            _file = self.compare_data.files_found[file_index]
            group_index, diff_score = self.compare_result.files_grouped[file_index]
            file_group = self.compare_result.file_groups[group_index] if group_index in self.compare_result.file_groups else {}
            file_group[_file] = diff_score
            self.compare_result.file_groups[group_index] = file_group

        self.compare_result.finalize_group_result(store_checkpoints=store_checkpoints)
        return (self.compare_result.files_grouped, self.compare_result.file_groups)

    def _compute_matrix_similarities(self):
        '''
        Compute all pairwise similarities in one step using matrix multiplication.
        Returns a tuple of (similarity_matrix, indices_i, indices_j) where indices
        are the upper triangle indices of the matrix.
        '''
        similarity_matrix = BaseCompare.chunked_similarity_vectorized(self._file_embeddings, threshold=self.embedding_similarity_threshold)
        _i, _j = np.triu_indices_from(similarity_matrix, k=1)
        return similarity_matrix, _i, _j

    def _compute_iterative_similarities(self, i):
        '''
        Compute similarities using iterative comparison with np.roll.
        Returns a tuple of (similars, diff_scores) where similars is the array of
        indices that meet the similarity threshold.
        '''
        compare_file_embeddings = np.roll(self._file_embeddings, i, 0)
        color_similars = self._compute_embedding_diff(
            self._file_embeddings, compare_file_embeddings, True)

        if self.compare_faces:
            compare_file_faces = np.roll(self._file_faces, i, 0)
            face_comparisons = self._file_faces - compare_file_faces
            face_similars = face_comparisons == 0
            similars = np.nonzero(color_similars[0] * face_similars)
        else:
            similars = np.nonzero(color_similars[0])
        
        return similars, color_similars[1]

    def _process_similarity_results(self, base_index, diff_index, diff_score):
        '''
        Process the results of a similarity comparison, updating the grouping
        and duplicate detection.
        '''
        f1_grouped = base_index in self.compare_result.files_grouped
        f2_grouped = diff_index in self.compare_result.files_grouped

        if diff_score > self.threshold_duplicate:
            base_file = self.compare_data.files_found[base_index]
            diff_file = self.compare_data.files_found[diff_index]
            if ((base_file, diff_file) not in self._probable_duplicates
                    and (diff_file, base_file) not in self._probable_duplicates):
                self._probable_duplicates.append((base_file, diff_file))

        if not f1_grouped and not f2_grouped:
            self.compare_result.files_grouped[base_index] = (self.compare_result.group_index, diff_score)
            self.compare_result.files_grouped[diff_index] = (self.compare_result.group_index, diff_score)
            self.compare_result.group_index += 1
        elif f1_grouped:
            existing_group_index, previous_diff_score = self.compare_result.files_grouped[base_index]
            if previous_diff_score - self.threshold_group_cutoff > diff_score:
                self.compare_result.files_grouped[base_index] = (self.compare_result.group_index, diff_score)
                self.compare_result.files_grouped[diff_index] = (self.compare_result.group_index, diff_score)
                self.compare_result.group_index += 1
            else:
                self.compare_result.files_grouped[diff_index] = (
                    existing_group_index, diff_score)
        else:
            existing_group_index, previous_diff_score = self.compare_result.files_grouped[diff_index]
            if previous_diff_score - self.threshold_group_cutoff > diff_score:
                self.compare_result.files_grouped[base_index] = (self.compare_result.group_index, diff_score)
                self.compare_result.files_grouped[diff_index] = (self.compare_result.group_index, diff_score)
                self.compare_result.group_index += 1
            else:
                self.compare_result.files_grouped[base_index] = (existing_group_index, diff_score)


    def find_similars_to_image(self, search_path, search_file_index):
        '''
        Search the numpy array of all known image arrays for similar
        characteristics to the provide image.
        NOTE Legacy method to allow for compare_faces boolean to be respected.
        '''
        files_grouped = {}
        _files_found = list(self.compare_data.files_found)

        if self.verbose:
            logger.info("Identifying similar image files...")
        _files_found.pop(search_file_index)
        search_file_embedding = self._file_embeddings[search_file_index]
        file_embeddings = np.delete(self._file_embeddings, search_file_index, 0)
        embedding_similars = self._compute_embedding_diff(
            file_embeddings, search_file_embedding, True)

        if self.compare_faces:
            search_file_faces = self._file_faces[search_file_index]
            file_faces = np.delete(self._file_faces, search_file_index)
            face_comparisons = file_faces - search_file_faces
            face_similars = face_comparisons == 0
            similars = np.nonzero(embedding_similars[0] * face_similars)
        else:
            similars = np.nonzero(embedding_similars[0])

        if config.search_only_return_closest:
            for _index in similars[0]:
                files_grouped[_files_found[_index]] = embedding_similars[1][_index]
            # Sort results by increasing difference score
            self.compare_result.files_grouped = dict(sorted(files_grouped.items(), key=lambda item: item[1]))
        else:
            temp = {}
            count = 0
            for i in range(len(_files_found)):
                temp[_files_found[i]] = embedding_similars[1][i]
            for file, similarity in dict(sorted(temp.items(), key=lambda item: item[1], reverse=True)).items():
                if count == config.max_search_results:
                    break
                files_grouped[file] = similarity
                count += 1
            self.compare_result.files_grouped = dict(sorted(files_grouped.items(), key=lambda item: item[1], reverse=True))

        self.compare_result.finalize_search_result(
            self.search_media_path, verbose=self.verbose, is_embedding=True,
            threshold_duplicate=self.threshold_duplicate,
            threshold_related=self.threshold_probable_match)
        return {0: self.compare_result.files_grouped}

    def _run_search_on_path(self, search_media_path):
        '''
        Prepare and begin a search for a provided image file path.
        NOTE Legacy method to allow for compare_faces boolean to be respected.
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

        # Gather new image data if it was not in the initial list

        if search_media_path not in self.compare_data.files_found:
            if self.verbose:
                logger.info("Filepath not found in initial list - gathering new file data")
            try:
                embedding = self.image_embeddings_func(search_media_path)
            except OSError as e:
                if self.verbose:
                    logger.error(f"{search_media_path} - {e}")
                raise AssertionError(
                    "Encountered an error accessing the provided file path in the file system.")

            self._file_embeddings = np.insert(self._file_embeddings, 0, [embedding], 0)
            if self.compare_faces:
                n_faces = self._get_faces_count(search_media_path)
                self._file_faces = np.insert(self._file_faces, 0, [n_faces], 0)
            self.compare_data.files_found.insert(0, search_media_path)

        files_grouped = self.find_similars_to_image(
            search_media_path, self.compare_data.files_found.index(search_media_path))
        search_media_path = None
        return files_grouped

    def _compute_multiembedding_diff(self, positive_embeddings=[], negative_embeddings=[], threshold=0.0):
        files_grouped = {}

        if config.search_only_return_closest:
            _files_found = list(self.compare_data.files_found)
            embedding_similars = self._compute_embedding_diff(
                self._file_embeddings, positive_embeddings[0], True, threshold=threshold)
            similars = np.nonzero(embedding_similars[0])
            for _index in similars[0]:
                files_grouped[_files_found[_index]] = embedding_similars[1][_index]
            # Sort results by increasing difference score
            self.compare_result.files_grouped = dict(sorted(files_grouped.items(), key=lambda item: item[1]))
            return self.compare_result.files_grouped

        '''
        Generate embedding_similars arrays for both positive and negative embedding
        sets. For the positives, multiply the similarities together. For the negatives
        successively divide the results from the positive multiplications. The end
        result should reflect a combined similarity in the appropriate direction for
        each set of requested text embeddings.
        '''

        combined_similars = None
        positive_similarities = []
        negative_similarities = []

        for p_emb in positive_embeddings:
            embedding_similars = self._compute_embedding_diff(
                self._file_embeddings, p_emb, True, threshold=threshold)
            positive_similarities.append(embedding_similars[1])

        for n_emb in negative_embeddings:
            embedding_similars = self._compute_embedding_diff(
                self._file_embeddings, n_emb, True, threshold=threshold)
            negative_similarities.append(embedding_similars[1])

        avg_positive = np.mean(positive_similarities, axis=0) if positive_similarities else 0
        avg_negative = np.mean(negative_similarities, axis=0) if negative_similarities else 0
        combined_scores = avg_positive - avg_negative
        sorted_indices = np.argsort(combined_scores)[::-1] # descending order
        combined_similars = combined_scores[sorted_indices]

        if combined_similars is None or len(combined_similars) == 0:
            raise Exception('No results found.')

        logger.info(f"len files_found: {len(self.compare_data.files_found)}")
        logger.info(f"len combined_similars: {len(combined_similars)}")

        files_grouped = {}
        temp = {}
        count = 0
        sorted_files = [self.compare_data.files_found[i] for i in sorted_indices]
        sorted_scores = combined_scores[sorted_indices]

        for file, score in zip(sorted_files, sorted_scores):
            temp[file] = score

        for file, similarity in temp.items():  # Already in sorted order
            if count == config.max_search_results:
                break
            files_grouped[file] = similarity
            count += 1

        self.compare_result.files_grouped = files_grouped

    def find_similars_to_embeddings(self, positive_embeddings, negative_embeddings):
        '''
        Search the numpy array of all known image embeddings for similar
        characteristics to the provided images and texts.
        '''
        if self.verbose:
            logger.info("Identifying similar image files...")

        if self.args.search_media_path is None and self.args.negative_search_media_path is None:
            # NOTE It is much less likely for text to match exactly
            adjusted_threshold = self.embedding_similarity_threshold / 3
        else:
            adjusted_threshold = self.embedding_similarity_threshold
        self._compute_multiembedding_diff(positive_embeddings, negative_embeddings, adjusted_threshold)

        self.compare_result.finalize_search_result(
            self.search_media_path, args=self.args, verbose=self.verbose, is_embedding=True,
            threshold_duplicate=self.threshold_duplicate,
            threshold_related=self.threshold_probable_match)
        return {0: self.compare_result.files_grouped}

    def search_multimodal(self):
        '''
        Search for provided search images and text.
        '''

        if config.text_embedding_search_presets_exclusive \
                and self.args.search_text in config.text_embedding_search_presets:
            return self.segregate_by_text_with_domain(self.args.search_text)

        files_grouped = {0: {}}
        positive_embeddings = []
        negative_embeddings = []

        if self.args.search_media_path is not None:
            self._tokenize_image(self.args.search_media_path, positive_embeddings)

        if self.args.negative_search_media_path is not None:
            self._tokenize_image(self.args.negative_search_media_path, negative_embeddings, "negative search image")

        if self.args.search_text is not None and self.args.search_text.strip() != "":
            for text in self.args.search_text.split(","):
                self._tokenize_text(text.strip(), positive_embeddings, "positive search text")

        if self.args.search_text_negative is not None and self.args.search_text_negative.strip() != "":
            for text in self.args.search_text_negative.split(","):
                self._tokenize_text(text.strip(), negative_embeddings, "negative search text")

        if len(positive_embeddings) == 0 and len(negative_embeddings) == 0:
            logger.error(f"Failed to generate embeddings.\n"
                  f"search image = {self.args.search_media_path}\n"
                  f"negative search image = {self.args.negative_search_media_path}\n"
                  f"search text = {self.args.search_text}\n"
                  f"search text negative = {self.args.search_text_negative}")
            return files_grouped  # TODO better exception handling

        files_grouped = self.find_similars_to_embeddings(positive_embeddings, negative_embeddings)
        return files_grouped

    def segregate_by_text_with_domain(self, search_text, search_text_negative=None, threshold=0.0):
        #### TODO refactor this to work with negative search text
        '''
        Optionally we may want to find the matches that are most exclusive to the
        search text within the domain of the provided search presets.
        '''
        files_grouped = {}
        temp = {}
        embeddings = []
        search_text_index = config.text_embedding_search_presets.index(search_text)
        count = 0

        if len(self.segregation_map) == 0 or self.args.overwrite:  # TODO different boolean for this cache
            for preset in config.text_embedding_search_presets:
                self._tokenize_text(preset, embeddings)

            for f in self.compare_data.files_found:
                self.segregation_map[f] = []

            for embedding in embeddings:
                embedding_similars = self._compute_embedding_diff(
                    self._file_embeddings, embedding, True, threshold=threshold)
                normalized = embedding_similars[1] / min(embedding_similars[1])
                for i in range(len(normalized)):
                    self.segregation_map[self.compare_data.files_found[i]].append(normalized[i])

        for f, similarities in self.segregation_map.items():
            max_similarity_index = similarities.index(max(similarities))
            if search_text_index == max_similarity_index:
                temp[f] = similarities[max_similarity_index]

        # TODO need some type of way to massage the results so that the clusters formed by the texts with
        # strong signals don't cannibalize the results from the other search terms

        for file, similarity in dict(sorted(temp.items(), key=lambda item: item[1], reverse=True)).items():
            if count == config.max_search_results:
                break
            files_grouped[file] = similarity
            count += 1
        files_grouped = dict(sorted(files_grouped.items(), key=lambda item: item[1], reverse=True))

        return {0: files_grouped}

    def _tokenize_text(self, text, embeddings=[], descriptor="search text"):
        if text in self.text_embedding_cache:
            text_embedding = self.text_embedding_cache[text]
            if text_embedding is not None:
                embeddings.append(self.text_embedding_cache[text])
                return
        if self.verbose:
            logger.info(f"Tokenizing {descriptor}: \"{text}\"")
        try:
            text_embedding = self.text_embeddings_func(text)
            embeddings.append(text_embedding)
            self.text_embedding_cache[text] = text_embedding
        except OSError as e:
            if self.verbose:
                logger.error(f"{text} - {e}")
            raise AssertionError(
                f"Encountered an error generating token embedding for {descriptor}")

    def _tokenize_image(self, image_path, embeddings=[], descriptor="search image"):
        if self.verbose:
            logger.info(f"Tokenizing {descriptor}: \"{image_path}\"")
        try:
            embedding = self.image_embeddings_func(image_path)
            embeddings.append(embedding)
        except OSError as e:
            if self.verbose:
                logger.error(f"{image_path} - {e}")
            raise AssertionError(
                "Encountered an error accessing the provided file path in the file system.")

    def get_probable_duplicates(self):
        return self._probable_duplicates

    def remove_from_groups(self, removed_files=[]):
        # TODO technically it would be better to refresh the file and data lists every time a compare is done
        # If not, will need to add a way to re-add the removed file data in case the remove action was undone
        remove_indexes = []
        for f in removed_files:
            if f in self.compare_data.files_found:
                remove_indexes.append(self.compare_data.files_found.index(f))
        remove_indexes.sort()

        if len(self._file_embeddings) > 0:
            self._file_embeddings = np.delete(self._file_embeddings, remove_indexes, axis=0)
        if len(self._file_faces) > 0:
            self._file_faces = np.delete(self._file_faces, remove_indexes, axis=0)

        for f in removed_files:
            if f in self.compare_data.files_found:
                self.compare_data.files_found.remove(f)

    def readd_files(self, filepaths=[]):
        readded_indexes = []
        for f in filepaths:
            if f not in self.compare_data.files_found:
                readded_indexes.append(len(self.compare_data.files_found))
                self.compare_data.files_found.append(f)
                try:
                    embedding = self.image_embeddings_func(f)
                except OSError as e:
                    logger.error(f"Error generating embedding from file {f}: {e}")
                    continue
                self.file_embeddings_dict[f] = embedding
                self._file_embeddings = np.vstack((self._file_embeddings, [embedding]))
                if self.compare_faces:
                    n_faces = self.get_faces_count(f)
                    self.compare_data.file_faces_dict = n_faces
                    self._file_faces = np.vstack((self._file_faces, [n_faces]))
                if self.verbose:
                    logger.info(f"Readded file to compare: {f}")


    @staticmethod
    def _get_text_embedding_from_cache(text, text_cache, text_embeddings_func):
        if text in text_cache:
            text_embedding = text_cache[text]
        else:
            try:
                text_embedding = text_embeddings_func(text)
                text_cache[text] = text_embedding
            except OSError as e:
                logger.error(f"{text} - {e}")
                raise AssertionError("Encountered an error generating text embedding.")
        return text_embedding

    @staticmethod
    def single_text_compare(image_path, texts_dict, image_embeddings_func, text_cache, text_embeddings_func):
        logger.info(f"Running text comparison for \"{image_path}\" - text = {texts_dict}")
        similarities = {}
        try:
            image_embedding = image_embeddings_func(image_path)
        except OSError as e:
            logger.error(f"{image_path} - {e}")
            raise AssertionError(
                f"Encountered an error accessing the provided file path {image_path} in the file system.")
        for key, text in texts_dict.items():
            similarities[key] = embedding_similarity(image_embedding, BaseCompareEmbedding._get_text_embedding_from_cache(text, text_cache, text_embeddings_func))
        return similarities

    @staticmethod
    def multi_text_compare(image_path, positives, negatives, image_embeddings_func, text_cache, text_embeddings_func, multi_cache, threshold=0.3):
        key = (image_path, "::p", tuple(positives), "::n", tuple(negatives))
        if key in multi_cache:
            return bool(multi_cache[key] > threshold)
        positive_similarities = []
        negative_similarities = []
        try:
            image_embedding = image_embeddings_func(image_path)
        except OSError as e:
            logger.error(f"{image_path} - {e}")
            raise AssertionError(
                f"Encountered an error accessing the provided file path {image_path} in the file system.")

        for text in positives:
            similarity = embedding_similarity(image_embedding, BaseCompareEmbedding._get_text_embedding_from_cache(text, text_cache, text_embeddings_func))
            positive_similarities.append(float(similarity[0]))
        for text in negatives:
            similarity = embedding_similarity(image_embedding, BaseCompareEmbedding._get_text_embedding_from_cache(text, text_cache, text_embeddings_func))
            negative_similarities.append(1/float(similarity[0]))

        combined_positive_similarity = sum(positive_similarities)/max(len(positive_similarities),1)
        combined_negative_similarity = sum(negative_similarities)/max(len(negative_similarities),1)
        if combined_positive_similarity > 0 and combined_negative_similarity > 0:
            combined_similarity = combined_positive_similarity / combined_negative_similarity
        elif combined_positive_similarity > 0:
            combined_similarity = combined_positive_similarity
        else:
            combined_similarity = 1 / combined_negative_similarity
        multi_cache[key] = combined_similarity
        return combined_similarity > threshold

    @staticmethod
    def is_related(image1, image2, image_embeddings_func):
        try:
            emb1 = image_embeddings_func(image1)
            emb2 = image_embeddings_func(image2)
        except OSError as e:
            logger.error(f"{image1} - {e}")
            raise AssertionError(
                "Encountered an error accessing the provided file paths in the file system.")
        similarity = embedding_similarity(emb1, emb2)[0]
        return similarity > 0.8


def usage():
    print("  Option                 Function                                 Default")
    print("      --dir=dirpath      Set base directory                       .      ")
    print("      --counter=int      Set counter cutoff for processing files  10000  ")
    print("      --faces=bool       Set use strict face matching             True   ")
    print("  -h, --help             Print help                                      ")
    print("      --include=pattern  File inclusion pattern                          ")
    print("      --search=filepath  Search for similar files to file         None   ")
    print("  -o, --overwrite        Overwrite saved image data               False  ")
    print("      --threshold=float  Embedding similarity threshold           0.9    ")
    print("  -v                     Verbose                                         ")

def main(compare_class):
    base_dir = "."
    overwrite = False
    search_media_path = None
    verbose = False
    compare_faces = True
    include_gifs = False
    counter_limit = 10000
    inclusion_pattern = None
    embedding_similarity_threshold = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "bcfist:hov", [
            "help",  "overwrite", "dir=", "counter=", "faces=", "include=",
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
                inclusion_pattern = a
            elif o == "--faces":
                compare_faces = a == "True" or a == "true" or a == "t"
            elif o in ("-o", "--overwrite"):
                overwrite = True
                confirm = input("Confirm overwriting image data (y/n): ")
                if confirm != "y" and confirm != "Y":
                    print("No change made.")
                    exit()
            elif o == "--search":
                search_media_path = Utils.get_valid_file(base_dir, a)
                if search_media_path is None:
                    assert False, "Search file provided \"" + str(a) \
                        + "\" is invalid - please ensure \"dir\" is passed first" \
                        + " if not providing full file path."
            elif o == "--threshold":
                embedding_similarity_threshold = float(a)
            else:
                assert False, "unhandled option " + o
        except Exception as e:
            print(e)
            print("")
            usage()
            exit(1)

    compare = compare_class.__init__(base_dir,
                               search_media_path=search_media_path,
                               counter_limit=counter_limit,
                               embedding_similarity_threshold=embedding_similarity_threshold,
                               compare_faces=compare_faces,
                               inclusion_pattern=inclusion_pattern,
                               overwrite=overwrite,
                               verbose=verbose,
                               include_gifs=include_gifs)
    compare.get_files()
    compare.get_data()
    compare.run()

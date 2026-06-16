import getopt
import os
import pprint
import sys

import numpy as np
from PIL import Image

from compare.base_compare import BaseCompare, gather_files
from compare.compare_args import CompareArgs
from compare.compare_result import CompareResult
from compare.model import embedding_similarity
from image.frame_cache import FrameCache
from utils.config import config
from utils.logging_setup import get_logger
from utils.media_utils import is_classifier_dynamic_media_path
from utils.utils import Utils

logger = get_logger("base_compare_embedding")


def cluster_group_indexes(centroids: dict, threshold: float) -> list:
    '''
    Greedy single-link clustering of group_index values by centroid cosine
    similarity (centroids are already unit vectors, so dot product is cosine
    similarity). O(g^2) over the number of groups, not files -- group counts
    are typically far smaller than file counts, so this is cheap.

    Returns a list of clusters, each a list of group_index values. Order is
    arbitrary; callers that care about ordering (see compute_supergroups)
    sort the result themselves.
    '''
    indexes = list(centroids.keys())
    parent = {i: i for i in indexes}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(indexes):
        for b in indexes[i + 1:]:
            if np.dot(centroids[a], centroids[b]) >= threshold:
                union(a, b)

    clusters: dict = {}
    for i in indexes:
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


class BaseCompareEmbedding(BaseCompare):
    # Set True on subclasses whose image_embeddings_func already samples video
    # frames itself (e.g. V-JEPA 2) -- compute_embedding_for_path() then skips
    # its own multi-frame sampling/averaging to avoid double-sampling and
    # diluting the model's own temporal encoding with repeated-still-frame passes.
    EMBEDS_DYNAMIC_MEDIA_NATIVELY = False

    # Supergrouping: clusters of related groups based on group mean-embedding
    # similarity (see compute_supergroups / cluster_group_indexes below). The
    # similarity threshold for clustering group mean embeddings is derived
    # from this run's own embedding_similarity_threshold rather than a fixed
    # constant, scaled by this ratio.
    SUPERGROUP_THRESHOLD_RATIO = 0.9
    # Below this, embedding_similarity_threshold is already loose enough that
    # supergrouping would over-merge unrelated groups -- skip it entirely.
    SUPERGROUP_MIN_VIABLE_THRESHOLD = 0.5

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
        logger.info(f" embedding similarity threshold: {self.embedding_similarity_threshold}")
        logger.info(f" max file process limit: {self.args.counter_limit}")
        logger.info(f" max files processable for base dir: {self.max_files_processed}")
        logger.info(f" recursive: {self.args.recursive}")
        logger.info(f" file glob pattern: {self.args.file_filter}")
        logger.info(f" include videos: {self.args.include_videos}")
        logger.info(f" file embeddings filepath: {self.compare_data._file_data_filepath}")
        logger.info(f" overwrite image data: {self.args.overwrite}")
        logger.info(f" compare mode: {self.COMPARE_MODE}")
        logger.info("|--------------------------------------------------------------------|\n\n")

    @staticmethod
    def _get_dynamic_media_sample_paths(path: str) -> list:
        '''
        Sampled frame/page paths for video, GIF, or PDF media, reusing the same
        FrameCache sampling primitives as prevalidation's frame-trigger scan
        (compare/classifier_action.py). Capped by the compare-specific
        ``compare_embedding_dynamic_media_max_samples`` rather than the
        (larger) prevalidation cap, since each sample here costs a full
        embedding-model forward pass instead of a lighter classifier call.

        Falls back to a single-item list (the resolved still frame) on any
        sampling failure so callers can treat the result uniformly.
        '''
        try:
            _planned, sample_iter = FrameCache.stream_frame_samples(
                path,
                sample_ratio=config.compare_embedding_dynamic_media_sample_ratio,
                max_samples=config.compare_embedding_dynamic_media_max_samples,
            )
            sample_paths = list(sample_iter)
            return sample_paths if sample_paths else [FrameCache.get_image_path(path)]
        except Exception as e:
            logger.debug(f"Dynamic-media sampling failed for {path}, using single frame: {e}")
            return [FrameCache.get_image_path(path)]

    @classmethod
    def compute_embedding_for_path(cls, path: str, image_embeddings_func, sample_dynamic_media: bool = True):
        '''
        Compute an embedding for *path*.

        For video/GIF/PDF media, combines embeddings from multiple sampled
        frames/pages (mean-pooled, then re-normalized so the result remains a
        valid unit vector for cosine similarity) instead of relying on a
        single first-frame snapshot -- the rest of a video's duration or a
        PDF's pages otherwise never contributes to compare/search results.

        Falls back to the single resolved still frame for plain images, when
        *sample_dynamic_media* is False (set by subclasses whose
        image_embeddings_func already samples video natively, e.g. V-JEPA 2),
        or when sampling yields nothing usable.

        Per-sample failures (exceptions or a None embedding, e.g. no face
        detected in that frame for face embeddings) are skipped rather than
        failing the whole file; the caller's own exception handling still
        applies to the always-present single-frame fallback path.
        '''
        if not sample_dynamic_media or not is_classifier_dynamic_media_path(path):
            return image_embeddings_func(FrameCache.get_image_path(path))

        sample_paths = cls._get_dynamic_media_sample_paths(path)
        if len(sample_paths) <= 1:
            return image_embeddings_func(sample_paths[0] if sample_paths else FrameCache.get_image_path(path))

        embeddings = []
        for sample_path in sample_paths:
            try:
                sample_embedding = image_embeddings_func(sample_path)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                continue
            if sample_embedding is not None:
                embeddings.append(sample_embedding)

        if not embeddings:
            return image_embeddings_func(FrameCache.get_image_path(path))
        if len(embeddings) == 1:
            return embeddings[0]

        mean_embedding = np.mean(np.array(embeddings), axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        return mean_embedding.tolist()

    def supports_supergrouping(self) -> bool:
        return True

    def compute_group_centroids(self) -> dict:
        '''
        Mean-pooled, re-normalized embedding per group in self.compare_result.file_groups
        (same mean+renormalize pattern as compute_embedding_for_path / EmbeddingPrototype).

        Single-member ("stranded") groups are deliberately included -- the
        centroid of one file is just that file's own embedding. Stranding
        happens in _process_similarity_results: when a file already assigned
        to group G is later matched to a different file at a meaningfully
        worse score (the previous_diff_score - threshold_group_cutoff >
        diff_score branch), both get moved into a brand-new group, abandoning
        the first file's original groupmate -- which was almost certainly
        still similar to *something*. Excluding it from clustering would
        permanently strand it; including it gives it a real chance to land in
        a supergroup with its old group (or any other related one) instead of
        looking orphaned forever. compare_wrapper.py's "Stranded Group Members
        Found" alert and run_group()'s initial-view skip only affect which
        group is auto-displayed first -- stranded groups stay fully present in
        file_groups/group_indexes regardless, so including them here is
        consistent with how they already behave everywhere else.

        Members not found in compare_data.files_found (e.g. since removed)
        are silently skipped; a group left with no resolvable members is omitted.
        '''
        path_to_index = {p: i for i, p in enumerate(self.compare_data.files_found)}
        centroids = {}
        for group_index, members in self.compare_result.file_groups.items():
            idxs = [path_to_index[p] for p in members if p in path_to_index]
            if not idxs:
                continue
            mean = np.mean(self._file_embeddings[idxs], axis=0)
            norm = np.linalg.norm(mean)
            centroids[group_index] = mean / norm if norm > 0 else mean
        return centroids

    def compute_supergroups(self) -> list:
        '''
        Cluster groups by mean-embedding similarity into "supergroups" and store
        the partition on self.compare_result.supergroups (list of lists of
        group_index, ascending by total member-file count -- mirrors
        CompareResult.sort_groups's existing ascending-by-size convention for
        ordinary groups).

        This is a navigation-only layer: file_groups / files_grouped are never
        rewritten, so marks/purge/composite-filter code that keys off
        group_index is unaffected. Computed once per run_comparison() call
        (cache-hit and freshly-computed paths both call this) and is NOT
        re-clustered afterward if file_groups later mutates (random purge,
        single-file removal, composite-filter rebuild) -- callers that read
        compare_result.supergroups should tolerate a group_index that no
        longer exists in file_groups.

        Skipped (supergroups set to []) when embedding_similarity_threshold is
        already below SUPERGROUP_MIN_VIABLE_THRESHOLD -- a loose base threshold
        means an already-derived, even-looser supergroup threshold would merge
        unrelated groups together. Also skipped when fewer than 2 groups have
        a usable centroid (compute_group_centroids only omits a group when
        none of its members can be resolved to an embedding at all, e.g. every
        member was since removed -- a stranded single-file group still gets one).
        '''
        if self.embedding_similarity_threshold < self.SUPERGROUP_MIN_VIABLE_THRESHOLD:
            logger.info(
                "Supergroups skipped: embedding_similarity_threshold %.3f is below "
                "SUPERGROUP_MIN_VIABLE_THRESHOLD %.3f",
                self.embedding_similarity_threshold, self.SUPERGROUP_MIN_VIABLE_THRESHOLD,
            )
            self.compare_result.clear_supergroups()
            return []

        centroids = self.compute_group_centroids()
        if len(centroids) < 2:
            logger.info("Supergroups skipped: only %d group(s) have a usable centroid", len(centroids))
            self.compare_result.clear_supergroups()
            return []

        threshold = self.embedding_similarity_threshold * self.SUPERGROUP_THRESHOLD_RATIO
        clusters = cluster_group_indexes(centroids, threshold)
        clusters.sort(key=lambda cluster: sum(len(self.compare_result.file_groups[g]) for g in cluster))
        self.compare_result.supergroups = clusters
        self._log_supergroups(clusters)
        return clusters

    def _log_supergroups(self, clusters: list) -> None:
        if not clusters:
            logger.info("Supergroups: none formed")
            return
        for supergroup_index, cluster in enumerate(clusters):
            stranded = [g for g in cluster if len(self.compare_result.file_groups.get(g, {})) == 1]
            logger.info(
                "Supergroup %d: groups %s%s",
                supergroup_index,
                pprint.pformat(cluster),
                f" (stranded: {pprint.pformat(stranded)})" if stranded else "",
            )

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
            # Check for cancellation during data gathering
            if self.is_cancelled():
                self.raise_cancellation_exception()

            if Utils.is_invalid_file(f, counter, self.is_run_search, self.args.file_filter):
                continue

            if counter > self.args.counter_limit:
                break

            if f in self.compare_data.file_data_dict:
                embedding = self.compare_data.file_data_dict[f]
            else:
                try:
                    embedding = self.compute_embedding_for_path(
                        f, self.image_embeddings_func,
                        sample_dynamic_media=not self.EMBEDS_DYNAMIC_MEDIA_NATIVELY,
                    )
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
                self.compare_data.file_data_dict[f] = embedding
                self.compare_data.has_new_file_data = True

            if embedding is None:
                continue

            counter += 1
            self._file_embeddings = np.vstack((self._file_embeddings, [embedding]))
            self.compare_data.files_found.append(f)
            self._handle_progress(counter, self.max_files_processed_even)

        # Save image file data
        self.compare_data.save_data(self.args.overwrite, verbose=self.verbose)

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
            self.compute_supergroups()
            return (self.compare_result.files_grouped, self.compare_result.file_groups)

        # Ensure we have correct counts of data compared to files found
        if len(self.compare_data.files_found) != len(self._file_embeddings):
            logger.error(f"Warning: Mismatch between files_found ({len(self.compare_data.files_found)}) and file_embeddings ({len(self._file_embeddings)})")

        if self.verbose:
            logger.info("Identifying groups of similar image files...")
        else:
            print("Identifying groups of similar image files", end="", flush=True)

        if self.args.use_matrix_comparison:
            for base_index, diff_index, diff_score in self._compute_matrix_similarities():
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
        self.compute_supergroups()
        return (self.compare_result.files_grouped, self.compare_result.file_groups)

    def _compute_matrix_similarities(self):
        '''
        Upper-triangle embedding pairs at or above the similarity threshold.

        Uses chunked matrix multiply (see ``BaseCompare.chunked_similarity_vectorized``)
        so large libraries do not allocate a full N×N similarity matrix in RAM.

        Returns:
            List of ``(base_index, diff_index, similarity_score)`` with ``base_index < diff_index``.
        '''
        return BaseCompare.chunked_similarity_vectorized(
            self._file_embeddings,
            threshold=self.embedding_similarity_threshold,
        )

    def _compute_iterative_similarities(self, i):
        '''
        Compute similarities using iterative comparison with np.roll.
        Returns a tuple of (similars, diff_scores) where similars is the array of
        indices that meet the similarity threshold.
        '''
        compare_file_embeddings = np.roll(self._file_embeddings, i, 0)
        color_similars = self._compute_embedding_diff(
            self._file_embeddings, compare_file_embeddings, True)

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


    def find_similars_to_media(self, search_path, search_file_index):
        '''
        Search the numpy array of all known media embeddings for similar
        characteristics to the provided media file.
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
                embedding = self.compute_embedding_for_path(
                    search_media_path, self.image_embeddings_func,
                    sample_dynamic_media=not self.EMBEDS_DYNAMIC_MEDIA_NATIVELY,
                )
            except OSError as e:
                if self.verbose:
                    logger.error(f"{search_media_path} - {e}")
                raise AssertionError(
                    "Encountered an error accessing the provided file path in the file system.")

            if embedding is None:
                raise AssertionError(
                    "No embedding could be produced for the search file. "
                    "For face mode this means no face was detected in the image.")

            self._file_embeddings = np.insert(self._file_embeddings, 0, [embedding], 0)
            self.compare_data.files_found.insert(0, search_media_path)

        files_grouped = self.find_similars_to_media(
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
            self._tokenize_media(self.args.search_media_path, positive_embeddings)

        if self.args.negative_search_media_path is not None:
            self._tokenize_media(self.args.negative_search_media_path, negative_embeddings, "negative search media")

        if self.args.search_text is not None and self.args.search_text.strip() != "":
            for text in self.args.search_text.split(","):
                self._tokenize_text(text.strip(), positive_embeddings, "positive search text")

        if self.args.search_text_negative is not None and self.args.search_text_negative.strip() != "":
            for text in self.args.search_text_negative.split(","):
                self._tokenize_text(text.strip(), negative_embeddings, "negative search text")

        if len(positive_embeddings) == 0 and len(negative_embeddings) == 0:
            logger.error(f"Failed to generate embeddings.\n"
                  f"search media = {self.args.search_media_path}\n"
                  f"negative search media = {self.args.negative_search_media_path}\n"
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

    def _tokenize_media(self, media_path, embeddings=[], descriptor="search media"):
        if self.verbose:
            logger.info(f"Tokenizing {descriptor}: \"{media_path}\"")
        try:
            embedding = self.compute_embedding_for_path(
                media_path, self.image_embeddings_func,
                sample_dynamic_media=not self.EMBEDS_DYNAMIC_MEDIA_NATIVELY,
            )
            embeddings.append(embedding)
        except OSError as e:
            if self.verbose:
                logger.error(f"{media_path} - {e}")
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
                    embedding = self.compute_embedding_for_path(
                        f, self.image_embeddings_func,
                        sample_dynamic_media=not self.EMBEDS_DYNAMIC_MEDIA_NATIVELY,
                    )
                except OSError as e:
                    logger.error(f"Error generating embedding from file {f}: {e}")
                    continue
                self.file_embeddings_dict[f] = embedding
                self._file_embeddings = np.vstack((self._file_embeddings, [embedding]))
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
    def single_text_compare(media_path, texts_dict, image_embeddings_func, text_cache, text_embeddings_func, sample_dynamic_media=True):
        logger.info(f"Running text comparison for \"{media_path}\" - text = {texts_dict}")
        similarities = {}
        try:
            media_embedding = BaseCompareEmbedding.compute_embedding_for_path(
                media_path, image_embeddings_func, sample_dynamic_media=sample_dynamic_media)
        except OSError as e:
            logger.error(f"{media_path} - {e}")
            raise AssertionError(
                f"Encountered an error accessing the provided file path {media_path} in the file system.")
        for key, text in texts_dict.items():
            similarities[key] = embedding_similarity(media_embedding, BaseCompareEmbedding._get_text_embedding_from_cache(text, text_cache, text_embeddings_func))
        return similarities

    @staticmethod
    def multi_text_compare(media_path, positives, negatives, image_embeddings_func, text_cache, text_embeddings_func, multi_cache, threshold=0.3, sample_dynamic_media=True):
        key = (media_path, "::p", tuple(positives), "::n", tuple(negatives))
        if key in multi_cache:
            return bool(multi_cache[key] > threshold)
        positive_similarities = []
        negative_similarities = []
        try:
            media_embedding = BaseCompareEmbedding.compute_embedding_for_path(
                media_path, image_embeddings_func, sample_dynamic_media=sample_dynamic_media)
        except OSError as e:
            logger.error(f"{media_path} - {e}")
            raise AssertionError(
                f"Encountered an error accessing the provided file path {media_path} in the file system.")

        for text in positives:
            similarity = embedding_similarity(media_embedding, BaseCompareEmbedding._get_text_embedding_from_cache(text, text_cache, text_embeddings_func))
            positive_similarities.append(float(similarity[0]))
        for text in negatives:
            similarity = embedding_similarity(media_embedding, BaseCompareEmbedding._get_text_embedding_from_cache(text, text_cache, text_embeddings_func))
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
    def is_related(media1, media2, image_embeddings_func, sample_dynamic_media=True):
        try:
            emb1 = BaseCompareEmbedding.compute_embedding_for_path(
                media1, image_embeddings_func, sample_dynamic_media=sample_dynamic_media)
            emb2 = BaseCompareEmbedding.compute_embedding_for_path(
                media2, image_embeddings_func, sample_dynamic_media=sample_dynamic_media)
        except OSError as e:
            logger.error(f"{media1} - {e}")
            raise AssertionError(
                "Encountered an error accessing the provided file paths in the file system.")
        similarity = embedding_similarity(emb1, emb2)[0]
        return similarity > 0.8


def usage():
    print("  Option                 Function                                 Default")
    print("      --dir=dirpath      Set base directory                       .      ")
    print("      --counter=int      Set counter cutoff for processing files  10000  ")
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
    include_gifs = False
    counter_limit = 10000
    file_filter = None
    embedding_similarity_threshold = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "bcfist:hov", [
            "help", "overwrite", "dir=", "counter=", "include=",
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
                               file_filter=file_filter,
                               overwrite=overwrite,
                               verbose=verbose,
                               include_gifs=include_gifs)
    compare.get_files()
    compare.get_data()
    compare.run()

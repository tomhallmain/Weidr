import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_clip, text_embeddings_clip
from utils.config import config
from utils.constants import CompareMode


class CompareEmbeddingClip(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.CLIP_EMBEDDING
    CACHE_FILENAME = "image_embeddings_clip.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = config.threshold_potential_duplicate_embedding
    THRESHHOLD_PROBABLE_MATCH = 0.98
    THRESHHOLD_GROUP_CUTOFF = 4500  # TODO fix this for Embedding case
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {} # keys are tuples of the filename + any text embedding search combination, values are combined similarity

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self._file_embeddings = np.empty((0, 512))
        self._file_faces = np.empty((0))
        self.threshold_duplicate = CompareEmbeddingClip.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingClip.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingClip.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_clip
        self.text_embeddings_func = text_embeddings_clip
        self.text_embedding_cache = CompareEmbeddingClip.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingClip.MULTI_EMBEDDING_CACHE

    @staticmethod
    def _get_text_embedding_from_cache(text):
        return BaseCompareEmbedding._get_text_embedding_from_cache(
            text, 
            CompareEmbeddingClip.TEXT_EMBEDDING_CACHE,
            text_embeddings_clip
        )

    @staticmethod
    def single_text_compare(image_path, texts_dict):
        return BaseCompareEmbedding.single_text_compare(
            image_path,
            texts_dict,
            image_embeddings_clip,
            CompareEmbeddingClip.TEXT_EMBEDDING_CACHE,
            text_embeddings_clip
        )

    @staticmethod
    def multi_text_compare(image_path, positives, negatives, threshold=0.3):
        return BaseCompareEmbedding.multi_text_compare(
            image_path,
            positives,
            negatives,
            image_embeddings_clip,
            CompareEmbeddingClip.TEXT_EMBEDDING_CACHE,
            text_embeddings_clip,
            CompareEmbeddingClip.MULTI_EMBEDDING_CACHE,
            threshold
        )

    @staticmethod
    def is_related(image1, image2):
        return BaseCompareEmbedding.is_related(
            image1,
            image2,
            image_embeddings_clip
        )


if __name__ == "__main__":
    main(CompareEmbeddingClip)


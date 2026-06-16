import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_laion, text_embeddings_laion
from utils.config import config
from utils.constants import CompareMode


class CompareEmbeddingLaion(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.LAION_EMBEDDING
    CACHE_FILENAME = "image_embeddings_laion.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = config.threshold_potential_duplicate_embedding
    THRESHHOLD_PROBABLE_MATCH = 0.98
    THRESHHOLD_GROUP_CUTOFF = 4500  # TODO fix this for Embedding case
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {} # keys are tuples of the filename + any text embedding search combination, values are combined similarity

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self._file_embeddings = np.empty((0, 1024))  # LAION ViT-H/14 has 1024-dimensional embeddings
        self.threshold_duplicate = CompareEmbeddingLaion.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingLaion.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingLaion.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_laion
        self.text_embeddings_func = text_embeddings_laion
        self.text_embedding_cache = CompareEmbeddingLaion.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingLaion.MULTI_EMBEDDING_CACHE

    @staticmethod
    def _get_text_embedding_from_cache(text):
        return BaseCompareEmbedding._get_text_embedding_from_cache(
            text, 
            CompareEmbeddingLaion.TEXT_EMBEDDING_CACHE,
            text_embeddings_laion
        )

    @staticmethod
    def single_text_compare(media_path, texts_dict):
        return BaseCompareEmbedding.single_text_compare(
            media_path,
            texts_dict,
            image_embeddings_laion,
            CompareEmbeddingLaion.TEXT_EMBEDDING_CACHE,
            text_embeddings_laion
        )

    @staticmethod
    def multi_text_compare(media_path, positives, negatives, threshold=0.3):
        return BaseCompareEmbedding.multi_text_compare(
            media_path,
            positives,
            negatives,
            image_embeddings_laion,
            CompareEmbeddingLaion.TEXT_EMBEDDING_CACHE,
            text_embeddings_laion,
            CompareEmbeddingLaion.MULTI_EMBEDDING_CACHE,
            threshold
        )

    @staticmethod
    def is_related(media1, media2):
        return BaseCompareEmbedding.is_related(
            media1,
            media2,
            image_embeddings_laion
        )


if __name__ == "__main__":
    main(CompareEmbeddingLaion) 
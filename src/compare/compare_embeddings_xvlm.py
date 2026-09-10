import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import xvlm_loaded, image_embeddings_xvlm, text_embeddings_xvlm
from utils.config import config
from utils.constants import CompareMode


class CompareEmbeddingXVLM(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.XVLM_EMBEDDING
    CACHE_FILENAME = "image_embeddings_xvlm.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = config.threshold_potential_duplicate_embedding
    THRESHHOLD_PROBABLE_MATCH = 0.98
    THRESHHOLD_GROUP_CUTOFF = 4500  # TODO fix this for Embedding case
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        # X-VLM projects to embed_dim=256 for both 4m and 16m configs
        self._file_embeddings = np.empty((0, 256))
        self.threshold_duplicate = CompareEmbeddingXVLM.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingXVLM.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingXVLM.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_xvlm
        self.text_embeddings_func = text_embeddings_xvlm
        self.text_embedding_cache = CompareEmbeddingXVLM.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingXVLM.MULTI_EMBEDDING_CACHE

    def is_runnable(self):
        return xvlm_loaded

    @staticmethod
    def _get_text_embedding_from_cache(text):
        return BaseCompareEmbedding._get_text_embedding_from_cache(
            text,
            CompareEmbeddingXVLM.TEXT_EMBEDDING_CACHE,
            text_embeddings_xvlm,
        )

    @staticmethod
    def single_text_compare(media_path, texts_dict):
        return BaseCompareEmbedding.single_text_compare(
            media_path,
            texts_dict,
            image_embeddings_xvlm,
            CompareEmbeddingXVLM.TEXT_EMBEDDING_CACHE,
            text_embeddings_xvlm,
        )

    @staticmethod
    def multi_text_compare(media_path, positives, negatives, threshold=0.3):
        return BaseCompareEmbedding.multi_text_compare(
            media_path,
            positives,
            negatives,
            image_embeddings_xvlm,
            CompareEmbeddingXVLM.TEXT_EMBEDDING_CACHE,
            text_embeddings_xvlm,
            CompareEmbeddingXVLM.MULTI_EMBEDDING_CACHE,
            threshold,
        )

    @staticmethod
    def is_related(media1, media2):
        return BaseCompareEmbedding.is_related(
            media1,
            media2,
            image_embeddings_xvlm,
        )


if __name__ == "__main__":
    main(CompareEmbeddingXVLM)

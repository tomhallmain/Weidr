import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_siglip, text_embeddings_siglip
from utils.config import config
from utils.constants import CompareMode


class CompareEmbeddingSiglip(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.SIGLIP_EMBEDDING
    CACHE_FILENAME = "image_embeddings_siglip.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = config.threshold_potential_duplicate_embedding
    THRESHHOLD_PROBABLE_MATCH = 0.98
    THRESHHOLD_GROUP_CUTOFF = 4500  # TODO fix this for Embedding case
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {} # keys are tuples of the filename + any text embedding search combination, values are combined similarity

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self._file_embeddings = np.empty((0, 1024 if config.siglip_enable_large_model else 768))
        self.threshold_duplicate = CompareEmbeddingSiglip.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingSiglip.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingSiglip.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_siglip
        self.text_embeddings_func = text_embeddings_siglip
        self.text_embedding_cache = CompareEmbeddingSiglip.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingSiglip.MULTI_EMBEDDING_CACHE


if __name__ == "__main__":
    main(CompareEmbeddingSiglip)

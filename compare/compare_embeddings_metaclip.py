import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_metaclip, text_embeddings_metaclip
from utils.config import config
from utils.constants import CompareMode


class CompareEmbeddingMetaClip(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.METACLIP_EMBEDDING
    CACHE_FILENAME = "image_embeddings_metaclip.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = config.threshold_potential_duplicate_embedding
    THRESHHOLD_PROBABLE_MATCH = 0.98
    THRESHHOLD_GROUP_CUTOFF = 4500  # TODO fix this for Embedding case
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        # Default model (facebook/metaclip-fullcc2.5b-h14-400m) is ViT-H/14 → 1024-dim.
        # Smaller variants: ViT-L/14 → 768, ViT-B/16 or B/32 → 512.
        self._file_embeddings = np.empty((0, 1024))
        self.threshold_duplicate = CompareEmbeddingMetaClip.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingMetaClip.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingMetaClip.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_metaclip
        self.text_embeddings_func = text_embeddings_metaclip
        self.text_embedding_cache = CompareEmbeddingMetaClip.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingMetaClip.MULTI_EMBEDDING_CACHE

    @staticmethod
    def _get_text_embedding_from_cache(text):
        return BaseCompareEmbedding._get_text_embedding_from_cache(
            text,
            CompareEmbeddingMetaClip.TEXT_EMBEDDING_CACHE,
            text_embeddings_metaclip,
        )

    @staticmethod
    def single_text_compare(media_path, texts_dict):
        return BaseCompareEmbedding.single_text_compare(
            media_path,
            texts_dict,
            image_embeddings_metaclip,
            CompareEmbeddingMetaClip.TEXT_EMBEDDING_CACHE,
            text_embeddings_metaclip,
        )

    @staticmethod
    def multi_text_compare(media_path, positives, negatives, threshold=0.3):
        return BaseCompareEmbedding.multi_text_compare(
            media_path,
            positives,
            negatives,
            image_embeddings_metaclip,
            CompareEmbeddingMetaClip.TEXT_EMBEDDING_CACHE,
            text_embeddings_metaclip,
            CompareEmbeddingMetaClip.MULTI_EMBEDDING_CACHE,
            threshold,
        )

    @staticmethod
    def is_related(media1, media2):
        return BaseCompareEmbedding.is_related(
            media1,
            media2,
            image_embeddings_metaclip,
        )


if __name__ == "__main__":
    main(CompareEmbeddingMetaClip)

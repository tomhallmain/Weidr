import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_vjepa2
from utils.config import config
from utils.constants import CompareMode

# Default hidden_size for ViT-L; ViT-H → 1280, ViT-G → 1408.
# If the user switches vjepa2_model to a larger variant the cache must be rebuilt.
_VJEPA2_DEFAULT_DIM = 1024


class CompareEmbeddingVJepa2(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.VJEPA2_EMBEDDING
    CACHE_FILENAME = "image_embeddings_vjepa2.pkl"
    # image_embeddings_vjepa2 already samples video frames itself (_sample_vjepa2_frames);
    # skip BaseCompareEmbedding's own multi-frame averaging to avoid double-sampling.
    EMBEDS_DYNAMIC_MEDIA_NATIVELY = True
    THRESHHOLD_POTENTIAL_DUPLICATE = config.threshold_potential_duplicate_embedding
    THRESHHOLD_PROBABLE_MATCH = 0.98
    THRESHHOLD_GROUP_CUTOFF = 4500
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self._file_embeddings = np.empty((0, _VJEPA2_DEFAULT_DIM))
        self.threshold_duplicate = CompareEmbeddingVJepa2.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingVJepa2.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingVJepa2.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_vjepa2
        # V-JEPA 2 has no joint text-image embedding space — text search is unsupported.
        self.text_embeddings_func = None
        self.text_embedding_cache = CompareEmbeddingVJepa2.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingVJepa2.MULTI_EMBEDDING_CACHE

    @staticmethod
    def is_related(image1, image2):
        return BaseCompareEmbedding.is_related(
            image1,
            image2,
            image_embeddings_vjepa2,
            sample_dynamic_media=False,
        )


if __name__ == "__main__":
    main(CompareEmbeddingVJepa2)

from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_vjepa2
from utils.constants import CompareMode

# Default hidden_size for ViT-L; ViT-H → 1280, ViT-G → 1408.
# If the user switches vjepa2_model to a larger variant the cache must be rebuilt.
_VJEPA2_DEFAULT_DIM = 1024


class CompareEmbeddingVJepa2(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.VJEPA2_EMBEDDING
    CACHE_FILENAME = "image_embeddings_vjepa2.pkl"
    # image_embeddings_vjepa2 already samples video frames itself (_sample_vjepa2_frames);
    # skip BaseCompareEmbedding's own multi-frame averaging to avoid double-sampling.
    EMBEDS_DYNAMIC_MEDIA_NATIVELY = True
    EMBEDDING_DIM = _VJEPA2_DEFAULT_DIM
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_vjepa2)
    # V-JEPA 2 has no joint text-image embedding space — text search is unsupported.
    TEXT_EMBEDDINGS_FUNC = None
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}


if __name__ == "__main__":
    main(CompareEmbeddingVJepa2)

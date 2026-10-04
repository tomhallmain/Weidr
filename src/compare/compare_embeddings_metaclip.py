from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_metaclip, text_embeddings_metaclip
from utils.constants import CompareMode


class CompareEmbeddingMetaClip(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.METACLIP_EMBEDDING
    CACHE_FILENAME = "image_embeddings_metaclip.pkl"
    # Default model (facebook/metaclip-fullcc2.5b-h14-400m) is ViT-H/14 → 1024-dim.
    # Smaller variants: ViT-L/14 → 768, ViT-B/16 or B/32 → 512.
    EMBEDDING_DIM = 1024
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_metaclip)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_metaclip)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}


if __name__ == "__main__":
    main(CompareEmbeddingMetaClip)

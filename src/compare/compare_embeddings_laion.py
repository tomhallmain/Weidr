from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_laion, text_embeddings_laion
from utils.constants import CompareMode


class CompareEmbeddingLaion(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.LAION_EMBEDDING
    CACHE_FILENAME = "image_embeddings_laion.pkl"
    # LAION ViT-H/14 has 1024-dimensional embeddings.
    EMBEDDING_DIM = 1024
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_laion)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_laion)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}


if __name__ == "__main__":
    main(CompareEmbeddingLaion)

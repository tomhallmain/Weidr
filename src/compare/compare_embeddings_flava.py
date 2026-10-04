from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_flava, text_embeddings_flava
from utils.constants import CompareMode


class CompareEmbeddingFlava(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.FLAVA_EMBEDDING
    CACHE_FILENAME = "image_embeddings_flava.pkl"
    EMBEDDING_DIM = 768
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_flava)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_flava)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}


if __name__ == "__main__":
    main(CompareEmbeddingFlava)

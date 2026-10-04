from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_align, text_embeddings_align
from utils.constants import CompareMode


class CompareEmbeddingAlign(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.ALIGN_EMBEDDING
    CACHE_FILENAME = "image_embeddings_align.pkl"
    EMBEDDING_DIM = 640
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_align)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_align)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}


if __name__ == "__main__":
    main(CompareEmbeddingAlign)

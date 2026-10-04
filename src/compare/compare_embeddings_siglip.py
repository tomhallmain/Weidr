from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_siglip, text_embeddings_siglip
from utils.config import config
from utils.constants import CompareMode


class CompareEmbeddingSiglip(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.SIGLIP_EMBEDDING
    CACHE_FILENAME = "image_embeddings_siglip.pkl"
    EMBEDDING_DIM = 768
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_siglip)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_siglip)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    @classmethod
    def embedding_dim(cls) -> int:
        return 1024 if config.siglip_enable_large_model else cls.EMBEDDING_DIM


if __name__ == "__main__":
    main(CompareEmbeddingSiglip)

from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import xvlm_loaded, image_embeddings_xvlm, text_embeddings_xvlm
from utils.constants import CompareMode


class CompareEmbeddingXVLM(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.XVLM_EMBEDDING
    CACHE_FILENAME = "image_embeddings_xvlm.pkl"
    # X-VLM projects to embed_dim=256 for both 4m and 16m configs.
    EMBEDDING_DIM = 256
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_xvlm)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_xvlm)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def is_runnable(self):
        return xvlm_loaded


if __name__ == "__main__":
    main(CompareEmbeddingXVLM)

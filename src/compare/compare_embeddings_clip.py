from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_clip, text_embeddings_clip
from utils.constants import CompareMode


class CompareEmbeddingClip(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.CLIP_EMBEDDING
    CACHE_FILENAME = "image_embeddings_clip.pkl"
    EMBEDDING_DIM = 512
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_clip)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_clip)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}


if __name__ == "__main__":
    main(CompareEmbeddingClip)

from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import eva_clip_loaded, image_embeddings_eva_clip, text_embeddings_eva_clip
from utils.constants import CompareMode


class CompareEmbeddingEvaClip(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.EVA_CLIP_EMBEDDING
    CACHE_FILENAME = "image_embeddings_eva_clip.pkl"
    # EVA01-g-14 produces 1024-dimensional embeddings; update if using a
    # smaller EVA02 variant (EVA02-B-16 → 512, EVA02-L-14 → 768).
    EMBEDDING_DIM = 1024
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_eva_clip)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_eva_clip)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def is_runnable(self):
        return eva_clip_loaded


if __name__ == "__main__":
    main(CompareEmbeddingEvaClip)

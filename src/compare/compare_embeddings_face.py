from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.model import image_embeddings_face, insightface_loaded
from utils.constants import CompareMode


class CompareEmbeddingFace(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.FACE_EMBEDDING
    CACHE_FILENAME = "image_embeddings_face.pkl"
    EMBEDDING_DIM = 512
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_face)
    # Face identity embeddings have no text counterpart.
    TEXT_EMBEDDINGS_FUNC = None
    THRESHHOLD_POTENTIAL_DUPLICATE = 0.6
    THRESHHOLD_PROBABLE_MATCH = 0.85
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def is_runnable(self):
        return insightface_loaded


if __name__ == "__main__":
    main(CompareEmbeddingFace)

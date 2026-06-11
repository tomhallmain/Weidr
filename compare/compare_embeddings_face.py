import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_face, insightface_loaded
from utils.constants import CompareMode


class CompareEmbeddingFace(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.FACE_EMBEDDING
    CACHE_FILENAME = "image_embeddings_face.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = 0.6
    THRESHHOLD_PROBABLE_MATCH = 0.85
    THRESHHOLD_GROUP_CUTOFF = 4500
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def __init__(self, args=CompareArgs(), gather_files_func=gather_files):
        super().__init__(args, gather_files_func)
        self._file_embeddings = np.empty((0, 512))
        self.threshold_duplicate = CompareEmbeddingFace.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingFace.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingFace.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_face
        # Face identity embeddings have no text counterpart.
        self.text_embeddings_func = None
        self.text_embedding_cache = CompareEmbeddingFace.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingFace.MULTI_EMBEDDING_CACHE

    def is_runnable(self):
        return insightface_loaded

    @staticmethod
    def is_related(image1, image2):
        return BaseCompareEmbedding.is_related(
            image1,
            image2,
            image_embeddings_face,
        )


if __name__ == "__main__":
    main(CompareEmbeddingFace)

"""
CLAP audio embedding compare mode -- audio-to-audio similarity/dedup/search,
the audio analog of the existing CLIP-family modes.

CLAP is not "the CLIP of audio" to the same degree CLIP dominates vision-
language: audio embedding models are fragmented by sub-task (speech-specific
Wav2Vec2/HuBERT, general sound-event PANNs/YAMNet with no text encoder,
music-specific MERT), and CLAP itself is younger and less battle-tested than
CLIP, with LAION's and Microsoft's variants not directly interchangeable.
It remains the right choice here specifically because it's the only real
open-source option pairing a text encoder with the audio encoder in a shared
space, which is what this mode's text search needs -- not because it won a
field-wide consensus the way CLIP did. Expect more rough edges than the
CLIP-family modes this codebase already has experience with.
"""

import numpy as np

from compare.base_compare import gather_files
from compare.base_compare_embedding import BaseCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_clap, text_embeddings_clap
from utils.config import config
from utils.constants import CompareMode


def gather_audio_files(base_dir=".", exts=None, recursive=True, include_videos=False, include_gifs=False, include_pdfs=False):
    """gather_files_func override: audio files only, never the image_types
    default -- CLAP has no notion of processing an image file, unlike the
    video/GIF/PDF flags on the other embedding modes, which add to an image
    file set that a visual model can still process every member of."""
    return gather_files(base_dir=base_dir, exts=config.audio_types, recursive=recursive)


class CompareEmbeddingAudioClap(BaseCompareEmbedding):
    COMPARE_MODE = CompareMode.AUDIO_CLAP_EMBEDDING
    CACHE_FILENAME = "audio_embeddings_clap.pkl"
    THRESHHOLD_POTENTIAL_DUPLICATE = config.threshold_potential_duplicate_embedding
    THRESHHOLD_PROBABLE_MATCH = 0.98
    THRESHHOLD_GROUP_CUTOFF = 4500  # TODO fix this for Embedding case
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {} # keys are tuples of the filename + any text embedding search combination, values are combined similarity

    def __init__(self, args=CompareArgs(), gather_files_func=gather_audio_files):
        super().__init__(args, gather_files_func)
        self._file_embeddings = np.empty((0, 512))  # ClapModel default projection_dim
        self.threshold_duplicate = CompareEmbeddingAudioClap.THRESHHOLD_POTENTIAL_DUPLICATE
        self.threshold_probable_match = CompareEmbeddingAudioClap.THRESHHOLD_PROBABLE_MATCH
        self.threshold_group_cutoff = CompareEmbeddingAudioClap.THRESHHOLD_GROUP_CUTOFF
        self.image_embeddings_func = image_embeddings_clap
        self.text_embeddings_func = text_embeddings_clap
        self.text_embedding_cache = CompareEmbeddingAudioClap.TEXT_EMBEDDING_CACHE
        self.multi_embedding_cache = CompareEmbeddingAudioClap.MULTI_EMBEDDING_CACHE

    @staticmethod
    def _get_text_embedding_from_cache(text):
        return BaseCompareEmbedding._get_text_embedding_from_cache(
            text,
            CompareEmbeddingAudioClap.TEXT_EMBEDDING_CACHE,
            text_embeddings_clap
        )

    @staticmethod
    def single_text_compare(media_path, texts_dict):
        return BaseCompareEmbedding.single_text_compare(
            media_path,
            texts_dict,
            image_embeddings_clap,
            CompareEmbeddingAudioClap.TEXT_EMBEDDING_CACHE,
            text_embeddings_clap
        )

    @staticmethod
    def multi_text_compare(media_path, positives, negatives, threshold=0.3):
        return BaseCompareEmbedding.multi_text_compare(
            media_path,
            positives,
            negatives,
            image_embeddings_clap,
            CompareEmbeddingAudioClap.TEXT_EMBEDDING_CACHE,
            text_embeddings_clap,
            CompareEmbeddingAudioClap.MULTI_EMBEDDING_CACHE,
            threshold
        )

    @staticmethod
    def is_related(media1, media2):
        return BaseCompareEmbedding.is_related(
            media1,
            media2,
            image_embeddings_clap
        )


if __name__ == "__main__":
    main(CompareEmbeddingAudioClap)

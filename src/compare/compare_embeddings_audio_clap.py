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

from compare.base_compare import gather_files
from compare.base_compare_embedding import ModelCompareEmbedding, main
from compare.compare_args import CompareArgs
from compare.model import image_embeddings_clap, text_embeddings_clap
from utils.config import config
from utils.constants import CompareMode


def gather_audio_files(base_dir=".", exts=None, recursive=True, include_videos=False, include_gifs=False, include_pdfs=False, include_epubs=False):
    """gather_files_func override: audio files only, never the image_types
    default -- CLAP has no notion of processing an image file, unlike the
    video/GIF/PDF/ePub flags on the other embedding modes, which add to an image
    file set that a visual model can still process every member of."""
    return gather_files(base_dir=base_dir, exts=config.audio_types, recursive=recursive)


class CompareEmbeddingAudioClap(ModelCompareEmbedding):
    COMPARE_MODE = CompareMode.AUDIO_CLAP_EMBEDDING
    CACHE_FILENAME = "audio_embeddings_clap.pkl"
    EMBEDDING_DIM = 512  # ClapModel default projection_dim
    IMAGE_EMBEDDINGS_FUNC = staticmethod(image_embeddings_clap)
    TEXT_EMBEDDINGS_FUNC = staticmethod(text_embeddings_clap)
    TEXT_EMBEDDING_CACHE = {}
    MULTI_EMBEDDING_CACHE = {}

    def __init__(self, args=CompareArgs(), gather_files_func=gather_audio_files):
        super().__init__(args, gather_files_func)


if __name__ == "__main__":
    main(CompareEmbeddingAudioClap)

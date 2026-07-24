"""
embedding_capture -- ad-hoc single-file embedding computation for the
embedding seed library (docs/embedding-seed-library.md, section 5.4).

Deliberately independent of CompareWrapper/CompareManager and any active
compare instance: a seed can be captured using a different embedding
architecture than whatever compare mode happens to be currently loaded.
"""
from __future__ import annotations

import os
from typing import List, Optional

from compare.compare_args import CompareArgs
from compare.compare_embeddings_align import CompareEmbeddingAlign
from compare.compare_embeddings_clip import CompareEmbeddingClip
from compare.compare_embeddings_eva_clip import CompareEmbeddingEvaClip
from compare.compare_embeddings_face import CompareEmbeddingFace
from compare.compare_embeddings_flava import CompareEmbeddingFlava
from compare.compare_embeddings_laion import CompareEmbeddingLaion
from compare.compare_embeddings_metaclip import CompareEmbeddingMetaClip
from compare.compare_embeddings_siglip import CompareEmbeddingSiglip
from compare.compare_embeddings_vjepa2 import CompareEmbeddingVJepa2
from compare.compare_embeddings_xvlm import CompareEmbeddingXVLM
from utils.constants import CompareMode
from utils.logging_setup import get_logger

logger = get_logger("embedding_capture")

# Mode -> embedding-compare class, for ad-hoc single-file embedding capture
# independent of whichever compare instance/mode happens to be currently active.
_EMBEDDING_MODE_CLASSES = {
    CompareMode.CLIP_EMBEDDING: CompareEmbeddingClip,
    CompareMode.SIGLIP_EMBEDDING: CompareEmbeddingSiglip,
    CompareMode.FLAVA_EMBEDDING: CompareEmbeddingFlava,
    CompareMode.ALIGN_EMBEDDING: CompareEmbeddingAlign,
    CompareMode.XVLM_EMBEDDING: CompareEmbeddingXVLM,
    CompareMode.LAION_EMBEDDING: CompareEmbeddingLaion,
    CompareMode.EVA_CLIP_EMBEDDING: CompareEmbeddingEvaClip,
    CompareMode.METACLIP_EMBEDDING: CompareEmbeddingMetaClip,
    CompareMode.VJEPA2_EMBEDDING: CompareEmbeddingVJepa2,
    CompareMode.FACE_EMBEDDING: CompareEmbeddingFace,
}


def embedding_capture_modes() -> List[CompareMode]:
    '''
    CompareMode values that support ad-hoc single-file embedding capture
    for the embedding seed library -- independent of whatever compare
    mode/instance is currently active, so a seed can be captured in a
    different architecture than the one currently loaded.
    See docs/embedding-seed-library.md, section 5.4.
    '''
    return list(_EMBEDDING_MODE_CLASSES.keys())


def compute_media_embedding(media_path: str, compare_mode: CompareMode):
    '''
    Compute an embedding for *media_path* using *compare_mode*'s own
    architecture directly -- no active CompareWrapper/compare instance
    required. Constructs a throwaway instance of the mode's class (cheap:
    model loading is lazy, deferred to the first actual inference call
    inside image_embeddings_func, so this doesn't preload anything extra).
    Returns None for an unrecognized/non-embedding mode, a missing file,
    or a failed computation.

    Callers wanting a responsive UI should run this off the main thread
    (e.g. a QThread + spinner, mirroring
    CompareWrapper._run_dynamic_prevalidation_with_spinner) since
    dynamic-media multi-frame sampling can be slow.
    See docs/embedding-seed-library.md, section 5.4.
    '''
    cls = _EMBEDDING_MODE_CLASSES.get(compare_mode)
    if cls is None or not media_path or not os.path.isfile(media_path):
        return None
    try:
        instance = cls(args=CompareArgs())
        return instance.compute_embedding_for_path(
            media_path,
            instance.image_embeddings_func,
            sample_dynamic_media=not instance.EMBEDS_DYNAMIC_MEDIA_NATIVELY,
        )
    except Exception as e:
        logger.warning(f"Failed to compute {compare_mode.name} embedding for {media_path}: {e}")
        return None

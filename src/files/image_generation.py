"""Sending one media file to Teikr for image generation."""

import os
from typing import Optional, Tuple

from files.related_image import clear_base_stem_dir_cache, clear_generate_gate_cache


def request_image_generation(
    generation_type,
    media_path: str,
    *,
    append: bool = False,
    prompt_overrides: Optional[Tuple[str, str]] = None,
    edit_suffix: Optional[str] = None,
    target_dir: Optional[str] = None,
) -> None:
    """Send *media_path* to Teikr as a *generation_type* request, then
    invalidate the related-image caches for its directory so the next
    generate-gate check rescans it.

    *prompt_overrides* is a (positive, negative) prompt pair. Blocks until
    TeikrClient.run returns; callers that must stay responsive run this
    on a worker thread.
    """
    # Imported here rather than at module level: tests patch
    # extensions.teikr_client.TeikrClient, which a module-level
    # import would already have bound.
    from extensions.teikr_client import TeikrClient

    positive_prompt, negative_prompt = prompt_overrides or (None, None)
    TeikrClient().run(
        generation_type, media_path, append=append,
        positive_prompt=positive_prompt, negative_prompt=negative_prompt,
        edit_suffix=edit_suffix, target_dir=target_dir,
    )
    output_dir = os.path.dirname(media_path)
    clear_generate_gate_cache(output_dir)
    clear_base_stem_dir_cache(output_dir)

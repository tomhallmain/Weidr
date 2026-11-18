"""
The file references one run_pipeline call works with, each under its own name.

A pipeline run touches up to three files of a stem group: the file it was
called with (*reached*), the group's *seed* (stem equal to the base stem), and
with ``preferred_target_category`` set, the *preferred* file (stem equal to
base stem + that category's suffix), which replaces the reached file as the
*evaluated* file. Keeping each as a separate field means no variable stands for
"the seed or whatever replaced it".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from files import related_image
from utils.translations import _


@dataclass(frozen=True)
class PipelineContext:
    reached_path: str
    base_stem: Optional[str]
    # Filename fact about the reached file.
    reached_is_seed_by_name: bool
    # Caller intent ("Run Current"): treat the reached file as its group's seed
    # for the seed-category GENERATE guard. Not a statement of which file the
    # seed is.
    assume_seed: bool = False
    seed_category: str = ""
    seed_suffix: str = ""
    preferred_target_category: str = ""
    preferred_suffix: str = ""
    preferred_path: Optional[str] = None
    # Translated reason the group is skipped without evaluation; None otherwise.
    skip_reason: Optional[str] = None

    @property
    def working_dir(self) -> str:
        return os.path.dirname(os.path.abspath(self.reached_path))

    @property
    def substitution_enabled(self) -> bool:
        return bool(self.preferred_suffix)

    @property
    def substituted(self) -> bool:
        return self.preferred_path is not None

    @property
    def evaluated_path(self) -> str:
        """The file nodes and actions run on."""
        return self.preferred_path if self.preferred_path is not None else self.reached_path

    @property
    def evaluated_is_seed_by_name(self) -> bool:
        if not self.substituted:
            return self.reached_is_seed_by_name
        return _stem_equals(self.preferred_path, self.base_stem)

    @property
    def seed_guard_applies(self) -> bool:
        """Whether GENERATE for the seed category's suffix is skipped.

        With substitution the seed still represents its category for the whole
        group, whichever file is evaluated.
        """
        if not self.seed_suffix:
            return False
        return self.substituted or self.assume_seed or self.reached_is_seed_by_name

    def skips_generate(self, action_modifier: Optional[str]) -> Optional[str]:
        """Which guard skips a GENERATE with *action_modifier*: "seed",
        "preferred", or None."""
        if not action_modifier:
            return None
        if self.seed_guard_applies and action_modifier == self.seed_suffix:
            return "seed"
        if self.substituted and action_modifier == self.preferred_suffix:
            return "preferred"
        return None

    @cached_property
    def seed_path(self) -> Optional[str]:
        """The group's seed among the working directory's own files, or None.

        Resolved on first access only, so a run that never asks pays no
        directory scan. With the seed present in several formats, the first by
        name.
        """
        if self.reached_is_seed_by_name:
            return self.reached_path
        if not self.base_stem:
            return None
        matches = _direct_matches(self.working_dir, self.base_stem, self.base_stem)
        return matches[0] if matches else None


def _stem_equals(path: str, stem: Optional[str]) -> bool:
    if not stem:
        return False
    return os.path.splitext(os.path.basename(path))[0].lower() == stem.lower()


def _direct_matches(directory: str, base_stem: str, wanted_stem: str) -> list[str]:
    """Files directly in *directory* whose stem is exactly *wanted_stem*.

    find_files_by_base_stem walks subdirectories as well, so its result is
    narrowed to the directory's own entries.
    """
    norm_dir = os.path.normcase(os.path.normpath(directory))
    return [
        f for f in related_image.find_files_by_base_stem([directory], base_stem, use_cache=True)
        if os.path.normcase(os.path.normpath(os.path.dirname(os.path.abspath(f)))) == norm_dir
        and _stem_equals(f, wanted_stem)
    ]


def build_pipeline_context(
    pipeline,
    reached_path: str,
    base_stem: Optional[str],
    *,
    assume_seed: bool = False,
) -> PipelineContext:
    """Resolve the references for one run of *pipeline* on *reached_path*.

    *base_stem* comes from the caller, which has already extracted it. With
    the pipeline's preferred_target_category set, looks up the preferred file
    and records a skip_reason when there is none, several, or one of a media
    type the pipeline does not apply to.
    """
    reached_is_seed_by_name = _stem_equals(reached_path, base_stem)
    seed_category = pipeline.seed_category or ""
    fields = dict(
        reached_path=reached_path,
        base_stem=base_stem,
        reached_is_seed_by_name=reached_is_seed_by_name,
        assume_seed=assume_seed,
        seed_category=seed_category,
        seed_suffix=pipeline.category_map.get(seed_category, "") if seed_category else "",
        preferred_target_category=pipeline.preferred_target_category or "",
        preferred_suffix=pipeline.preferred_target_suffix(),
    )
    preferred_suffix = fields["preferred_suffix"]
    if not preferred_suffix:
        return PipelineContext(**fields)

    name = os.path.basename(reached_path)
    if not base_stem:
        return PipelineContext(
            **fields,
            skip_reason=_("No base stem could be determined for {0}; preferred source file "
                          "lookup skipped.").format(name),
        )

    working_dir = os.path.dirname(os.path.abspath(reached_path))
    wanted_stem = base_stem + preferred_suffix
    matches = _direct_matches(working_dir, base_stem, wanted_stem)
    if not matches:
        return PipelineContext(
            **fields,
            skip_reason=_("No preferred source file {0} for {1}; stem group skipped.").format(
                wanted_stem, name
            ),
        )
    if len(matches) > 1:
        return PipelineContext(
            **fields,
            skip_reason=_("Several preferred source files {0} ({1}); stem group skipped.").format(
                wanted_stem, ", ".join(os.path.basename(m) for m in matches)
            ),
        )
    preferred_path = matches[0]
    if not pipeline.media_type_allowed(preferred_path):
        return PipelineContext(
            **fields,
            skip_reason=_("Preferred source file {0} has a media type this pipeline does not "
                          "apply to; stem group skipped.").format(os.path.basename(preferred_path)),
        )
    return PipelineContext(**fields, preferred_path=preferred_path)


@dataclass(frozen=True)
class PipelineFileResult:
    """What run_pipeline_with_context hands back.

    *evaluated* is True when the pipeline's nodes ran. *context* is None when
    the run ended before one was built (inactive pipeline, media type gate,
    stem group already done).
    """
    action: Optional[object] = None
    context: Optional[PipelineContext] = None
    evaluated: bool = False

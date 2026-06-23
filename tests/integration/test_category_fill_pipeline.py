"""
Integration tests for the category-fill demo pipeline.

Exercises the full run_pipeline() path against real files on disk, verifying
that the pipeline makes correct GENERATE decisions for each image state
described in §4.5 of docs/generation-pipeline-category-fill.md.

Base-stem naming convention
────────────────────────────
Real-world images have stems like "rose_17820172251234" — a label followed by
a long numeric timestamp.  extract_filename_base_stem() keeps this full string
as the base stem and strips only trailing segments whose length is ≤ 4 chars
(e.g. "_a", "_b") OR when the combined stem already reaches 10+ characters
before the next segment (the timestamp satisfies this immediately).

Suffixes "_apple" (5), "_banana" (6), "_cherry" (6) are longer than 4 chars,
so they are correctly stripped when the base stem already has ≥ 10 characters
(the timestamp path).  Using bare "rose" would NOT work because "rose_apple"
(10 chars) gets treated as one unit rather than base+suffix.

Stems used:
  STEM_A = "rose_17820172251234"
  STEM_B = "lily_29134560782345"

Directory layout
────────────────
  <tmp>/working/       ← working directory (base_directory for run_pipeline)
  <tmp>/target/A/      ← apple category target
  <tmp>/target/B/      ← banana category target
  <tmp>/target/C/      ← cherry category target
  <tmp>/sources/       ← filed seed archive (processed_stems validity checks)
"""

import os
from unittest.mock import patch

import pytest

from compare.action_callbacks import ActionCallbacks
from compare.classifier_pipeline import ClassifierPipelines
from compare.classifier_pipeline_runner import run_pipeline
from files.related_image import (
    clear_base_stem_dir_cache,
    clear_generate_gate_cache,
)
from utils.config import config
from utils.constants import ClassifierActionType

STEM_A = "rose_17820172251234"
STEM_B = "lily_29134560782345"


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def layout(tmp_path, monkeypatch):
    """
    Create the standard directory layout and configure
    directories_to_search_for_related_images to cover sources/ and the three
    target dirs (sources/ is where seeds are "filed" for processed_stems tests).
    Returns (working, target_a, target_b, target_c, sources).
    """
    working  = tmp_path / "working"
    target_a = tmp_path / "target" / "A"
    target_b = tmp_path / "target" / "B"
    target_c = tmp_path / "target" / "C"
    sources  = tmp_path / "sources"
    for d in (working, target_a, target_b, target_c, sources):
        d.mkdir(parents=True)

    monkeypatch.setattr(
        config, "directories_to_search_for_related_images",
        [str(sources), str(target_a), str(target_b), str(target_c)],
    )

    clear_base_stem_dir_cache()
    clear_generate_gate_cache()

    yield working, target_a, target_b, target_c, sources

    clear_base_stem_dir_cache()
    clear_generate_gate_cache()


def _pipeline(layout, *, active=True):
    """Build the category-fill pipeline wired to real target dirs."""
    _, ta, tb, tc, _ = layout
    p = ClassifierPipelines.build_category_fill_pipeline(
        target_dir_apple=str(ta),
        target_dir_banana=str(tb),
        target_dir_cherry=str(tc),
    )
    p.is_active = active
    return p


def _callbacks():
    """Return (ActionCallbacks, generated list).  generated = [(path, modifier)]."""
    generated = []
    cb = ActionCallbacks(
        generate_callback=lambda image_path, modifier, *a, **kw:
            generated.append((image_path, modifier)),
    )
    return cb, generated


# ---------------------------------------------------------------------------
# Pipeline meta-tests
# ---------------------------------------------------------------------------

class TestPipelineMeta:
    def test_pipeline_is_inactive_by_default(self, layout):
        """build_category_fill_pipeline() must ship inactive."""
        p = ClassifierPipelines.build_category_fill_pipeline()
        assert p.is_active is False

    def test_inactive_pipeline_returns_none(self, layout):
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        cb, generated = _callbacks()
        result = run_pipeline(_pipeline(layout, active=False), str(seed), cb,
                              base_directory=str(working))
        assert result is None
        assert generated == []

    def test_node_names_present(self, layout):
        p = _pipeline(layout)
        names = [n.name for n in p.nodes]
        assert "Unknown-suffix guard"   in names
        assert "Stem uniqueness check"  in names
        assert "Generate apple"         in names
        assert "Generate banana"        in names
        assert "Generate cherry"        in names


# ---------------------------------------------------------------------------
# §4.5 row 1 — Seed, no variants anywhere → GENERATE apple
# ---------------------------------------------------------------------------

class TestFreshSeed:
    def test_generates_all_missing_categories(self, layout):
        """
        Seed with no variants in working dir or any target dir.
        Pipeline should fire GENERATE for apple, banana, and cherry — all three
        nodes use EXECUTE_AND_CONTINUE so the run does not halt after the first.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        cb, generated = _callbacks()
        result = run_pipeline(_pipeline(layout), str(seed), cb,
                              base_directory=str(working))

        assert result == ClassifierActionType.GENERATE
        assert len(generated) == 3
        modifiers = [m for _, m in generated]
        assert "_apple"  in modifiers
        assert "_banana" in modifiers
        assert "_cherry" in modifiers
        assert all(path == str(seed) for path, _ in generated)

    def test_different_stems_each_generate_all_categories(self, layout):
        """Two seeds with distinct stems each produce three GENERATEs."""
        working, *_ = layout
        seed_a = working / f"{STEM_A}.jpg"
        seed_b = working / f"{STEM_B}.jpg"
        seed_a.touch()
        seed_b.touch()

        p = _pipeline(layout)
        cb, generated = _callbacks()
        run_pipeline(p, str(seed_a), cb, base_directory=str(working))
        run_pipeline(p, str(seed_b), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert modifiers.count("_apple")  == 2
        assert modifiers.count("_banana") == 2
        assert modifiers.count("_cherry") == 2


# ---------------------------------------------------------------------------
# §4.5 rows 2/3 — Target partially or fully covered
# ---------------------------------------------------------------------------

class TestPartiallyFilledTarget:
    def test_apple_covered_generates_remaining_categories(self, layout):
        """Apple variant in target/A/ → skip apple, generate banana AND cherry."""
        working, ta, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (ta / f"{STEM_A}_apple.jpg").touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  not in modifiers
        assert "_banana" in     modifiers
        assert "_cherry" in     modifiers
        assert len(generated) == 2

    def test_apple_and_banana_covered_generates_cherry(self, layout):
        working, ta, tb, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (ta / f"{STEM_A}_apple.jpg").touch()
        (tb / f"{STEM_A}_banana.jpg").touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        assert len(generated) == 1
        assert generated[0][1] == "_cherry"

    def test_all_covered_no_generation(self, layout):
        """All three target dirs already have a file for this stem → no GENERATE."""
        working, ta, tb, tc, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (ta / f"{STEM_A}_apple.jpg").touch()
        (tb / f"{STEM_A}_banana.jpg").touch()
        (tc / f"{STEM_A}_cherry.jpg").touch()

        cb, generated = _callbacks()
        result = run_pipeline(_pipeline(layout), str(seed), cb,
                              base_directory=str(working))

        assert generated == []

    def test_wrong_suffix_in_correct_dir_still_covers_category(self, layout):
        """
        §5.1: BaseStemMatchCondition uses directory as the primary discriminant.
        A file with an unexpected suffix living in target/A/ still signals that
        the apple category is covered for this stem.

        The guard uses use_base_directory=True so it only scans base_directory
        (the working dir). The wrong-suffix file in target/A/ is invisible to the
        guard; the apple node's BaseStemMatchCondition(search_directory=target/A/)
        finds it directly and correctly marks apple as covered.
        """
        working, ta, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        # Wrong suffix, right directory — apple should still be considered covered.
        (ta / f"{STEM_A}_wrong.jpg").touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  not in modifiers  # covered by wrong-suffix file in A/
        assert "_banana" in     modifiers  # banana still missing


# ---------------------------------------------------------------------------
# §4.5 row 2 — Category variant present in working dir → that category skipped
# ---------------------------------------------------------------------------

class TestVariantInWorkingDir:
    def test_apple_in_working_dir_skips_apple(self, layout):
        """
        RelatedImageCondition detects an existing apple variant in the working
        dir and returns False → pipeline skips apple, generates banana.
        """
        working, *_ = layout
        seed  = working / f"{STEM_A}.jpg"
        local = working / f"{STEM_A}_apple.jpg"
        seed.touch()
        local.touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  not in modifiers
        assert "_banana" in     modifiers

    def test_double_underscore_truncated_suffix_skips_apple(self, layout):
        """
        SD-runner style names use a double-underscore separator and a truncated
        category word (e.g. __appl for _apple) with no related-image metadata.
        RelatedImageCondition must still treat the working dir as covered.
        """
        working, *_ = layout
        seed  = working / f"{STEM_A}.jpg"
        local = working / f"{STEM_A}__appl.jpg"
        seed.touch()
        local.touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  not in modifiers
        assert "_banana" in     modifiers
        assert "_cherry" in     modifiers

    def test_double_underscore_variant_index_skips_cherry(self, layout):
        """Truncated suffix plus variant index (__cher_2) still covers cherry."""
        working, *_ = layout
        seed  = working / f"{STEM_A}.jpg"
        local = working / f"{STEM_A}__cher_2.jpg"
        seed.touch()
        local.touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_cherry" not in modifiers
        assert "_apple"  in     modifiers
        assert "_banana" in     modifiers

    def test_sd_runner_variants_in_working_dir_generate_only_missing(self, layout):
        """
        Manual test layout (operations5): seed with __appl, __cher, and __cher_2
        in the working dir should generate banana only.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}__appl.jpg").touch()
        (working / f"{STEM_A}__cher.jpg").touch()
        (working / f"{STEM_A}__cher_2.jpg").touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        assert len(generated) == 1
        assert generated[0][1] == "_banana"

    def test_intermediate_token_before_double_underscore_suffix_skips_apple(self, layout):
        """
        Generator may insert a label between the base stem and the category
        suffix (e.g. stem_upscaled_2x__appl). RelatedImageCondition must still
        recognise this as an apple variant in the working dir.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        local = working / f"{STEM_A}_upscaled_2x__appl.jpg"
        seed.touch()
        local.touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  not in modifiers
        assert "_banana" in     modifiers
        assert "_cherry" in     modifiers


# ---------------------------------------------------------------------------
# §4.5 row 5 — Type-3 derivative (upstream in working dir) → no generation
# ---------------------------------------------------------------------------

class TestType3Derivative:
    def test_derivative_with_upstream_metadata_blocked(self, layout):
        """
        A derivative whose related-image metadata points to a file present in
        the working directory is classified as type-3.  RelatedImageCondition
        returns False for every category node → no GENERATE fires.
        """
        working, *_ = layout
        seed  = working / f"{STEM_A}.jpg"
        deriv = working / f"{STEM_A}_apple.jpg"
        seed.touch()
        deriv.touch()

        def fake_related(path, *args, **kwargs):
            if os.path.normpath(path) == os.path.normpath(str(deriv)):
                return str(seed), False
            return None, False

        cb, generated = _callbacks()
        with patch("files.related_image.get_related_image_path",
                   side_effect=fake_related):
            clear_generate_gate_cache()
            run_pipeline(_pipeline(layout), str(deriv), cb,
                         base_directory=str(working))

        assert generated == [], (
            "Type-3 derivative must not trigger any GENERATE action"
        )


# ---------------------------------------------------------------------------
# §4.5 row 6 — Unknown-suffix guard → REJECT
# ---------------------------------------------------------------------------

class TestUnknownSuffixGuard:
    def test_unrecognised_suffix_rejects(self, layout):
        """
        Guard uses use_base_directory=True, so it scans base_directory (working dir).
        An unrecognised-suffix file there blocks generation — CompositeCondition(NOT)
        fires REJECT without any category node running.

        generated == [] proves the guard blocked: without the guard, the pipeline
        would have generated _apple (seed present, no target variants).
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}_zzz.jpg").touch()   # unrecognised suffix in working dir

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        assert generated == []

    def test_known_suffixes_pass_guard(self, layout):
        """
        All files in the working dir have recognised suffixes → guard passes.
        apple variant in working dir causes RelatedImageCondition to skip apple;
        banana is still missing → pipeline generates banana.
        """
        working, *_ = layout
        seed  = working / f"{STEM_A}.jpg"
        local = working / f"{STEM_A}_apple.jpg"
        seed.touch()
        local.touch()

        cb, generated = _callbacks()
        run_pipeline(_pipeline(layout), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_banana" in modifiers


# ---------------------------------------------------------------------------
# §5.13 — processed_stems batch skip
# ---------------------------------------------------------------------------

class TestProcessedStemsBatch:
    """
    Verify the stem-group classification and skip logic when processed_stems
    is passed.  sources/ acts as the seed archive so _resolve_stem_group
    classifies the seed as Type 1 valid (seed found in a configured dir).
    """

    def test_derivative_skipped_after_seed_processed(self, layout):
        """
        Seed evaluated first → stem marked done.
        Derivative evaluated next → skipped immediately, no conditions run.
        """
        working, ta, tb, tc, sources = layout
        seed  = working / f"{STEM_A}.jpg"
        deriv = working / f"{STEM_A}_apple.jpg"
        seed.touch()
        deriv.touch()
        # File seed into sources/ so _resolve_stem_group sees Type 1 valid.
        (sources / f"{STEM_A}.jpg").touch()

        p = _pipeline(layout)
        stems = set()
        cb, generated = _callbacks()
        run_pipeline(p, str(seed), cb, base_directory=str(working),
                     processed_stems=stems)
        assert STEM_A in stems

        cb2, generated2 = _callbacks()
        result2 = run_pipeline(p, str(deriv), cb2, base_directory=str(working),
                               processed_stems=stems)
        assert result2 is None
        assert generated2 == []

    def test_malformed_c_seed_not_in_sources_no_generation(self, layout):
        """
        Malformed C: seed in working dir but not in any configured dir.
        _resolve_stem_group returns (False, True) → pipeline skips, marks done.
        Subsequent derivatives for the same stem also skip.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        # sources/ is empty — seed not filed anywhere

        p = _pipeline(layout)
        stems = set()
        cb, generated = _callbacks()
        result = run_pipeline(p, str(seed), cb, base_directory=str(working),
                              processed_stems=stems)

        assert result is None
        assert generated == []
        assert STEM_A in stems   # marked done to suppress derivatives

    def test_two_distinct_stems_both_evaluated(self, layout):
        """
        Two different base stems must each be evaluated independently;
        the first being done must not suppress the second.
        """
        working, ta, tb, tc, sources = layout
        seed_a = working / f"{STEM_A}.jpg"
        seed_b = working / f"{STEM_B}.jpg"
        seed_a.touch()
        seed_b.touch()
        (sources / f"{STEM_A}.jpg").touch()
        (sources / f"{STEM_B}.jpg").touch()

        p = _pipeline(layout)
        stems = set()
        cb, generated = _callbacks()
        run_pipeline(p, str(seed_a), cb, base_directory=str(working),
                     processed_stems=stems)
        run_pipeline(p, str(seed_b), cb, base_directory=str(working),
                     processed_stems=stems)

        modifiers = [m for _, m in generated]
        assert modifiers.count("_apple")  == 2   # three GENERATEs per stem
        assert modifiers.count("_banana") == 2
        assert modifiers.count("_cherry") == 2


# ---------------------------------------------------------------------------
# seed_category guard — integration tests
# ---------------------------------------------------------------------------

def _pipeline_with_seed_category(layout):
    """Demo pipeline with seed_category='Apple' added for seed-guard tests."""
    p = _pipeline(layout)
    p.seed_category = "Apple"
    return p


class TestSeedCategoryGuardIntegration:
    """
    Integration tests for the seed_category guard using a modified form of the
    category-fill demo pipeline.  seed_category='Apple' is added to the demo
    pipeline so the runner skips 'Generate apple' for seed images without
    evaluating its conditions.

    The original demo pipeline (no seed_category) is used as a control in
    test_without_seed_category_generates_all_three to prove that apple IS
    generated when the guard is absent.
    """

    def test_seed_category_suppresses_apple_for_fresh_seed(self, layout):
        """
        Fresh seed, no variants anywhere, seed_category='Apple'.
        The runner skips the apple node via the guard; banana and cherry are
        still generated normally.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        p = _pipeline_with_seed_category(layout)
        cb, generated = _callbacks()
        run_pipeline(p, str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  not in modifiers
        assert "_banana" in     modifiers
        assert "_cherry" in     modifiers
        assert len(generated) == 2

    def test_without_seed_category_generates_all_three(self, layout):
        """
        Control: same fresh-seed setup but pipeline has no seed_category.
        Apple IS generated because the guard is disabled.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        p = _pipeline(layout)   # no seed_category
        cb, generated = _callbacks()
        run_pipeline(p, str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  in modifiers
        assert "_banana" in modifiers
        assert "_cherry" in modifiers
        assert len(generated) == 3

    def test_seed_category_and_apple_in_target_both_suppress_apple(self, layout):
        """
        Apple already covered in target/A/ AND seed_category='Apple'.
        Result is the same as without the guard: only banana+cherry generated.
        The guard fires first (bypasses condition evaluation) but the outcome
        is identical.
        """
        working, ta, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (ta / f"{STEM_A}_apple.jpg").touch()

        p = _pipeline_with_seed_category(layout)
        cb, generated = _callbacks()
        run_pipeline(p, str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert "_apple"  not in modifiers
        assert "_banana" in     modifiers
        assert "_cherry" in     modifiers

    def test_seed_category_does_not_affect_non_seed_images(self, layout):
        """
        A file that is not a seed (its stem ≠ base stem) is not affected by
        seed_category.  The apple node evaluates its condition normally.

        The derivative has no metadata and no apple variant in target/A/, so
        RelatedImageCondition and BaseStemMatchCondition both return True →
        apple IS generated.  This proves the seed guard did not suppress it
        (contrast with test_seed_category_suppresses_apple_for_fresh_seed where
        the seed has apple suppressed by the guard).
        """
        working, *_ = layout
        deriv = working / f"{STEM_A}_apple.jpg"
        deriv.touch()

        p = _pipeline_with_seed_category(layout)
        cb, generated = _callbacks()
        run_pipeline(p, str(deriv), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        # Derivative → is_seed=False → guard does not fire → apple generated.
        assert "_apple" in modifiers

    def test_seed_category_with_multiple_stems(self, layout):
        """
        Two distinct fresh seeds with seed_category='Apple': each generates
        banana and cherry, never apple.
        """
        working, *_ = layout
        seed_a = working / f"{STEM_A}.jpg"
        seed_b = working / f"{STEM_B}.jpg"
        seed_a.touch()
        seed_b.touch()

        p = _pipeline_with_seed_category(layout)
        cb, generated = _callbacks()
        run_pipeline(p, str(seed_a), cb, base_directory=str(working))
        run_pipeline(p, str(seed_b), cb, base_directory=str(working))

        modifiers = [m for _, m in generated]
        assert modifiers.count("_apple")  == 0
        assert modifiers.count("_banana") == 2
        assert modifiers.count("_cherry") == 2


# ---------------------------------------------------------------------------
# Generate-gate cache clearing between pipeline runs
# ---------------------------------------------------------------------------

class TestGenerateGateCacheClearing:
    """
    Validates that the generate-gate cache is cleared between pipeline runs so
    that RelatedImageCondition correctly detects variants written to the working
    directory after the first run and skips generation on subsequent runs.

    Without clearing _generate_gate_dir_cache, the second run sees the stale
    pre-generation directory scan and fires GENERATE again even though the
    variant files now exist.
    """

    def test_stale_cache_causes_duplicate_generate(self, layout):
        """
        Demonstrates the bug: if the gate cache is NOT cleared between runs,
        the second run generates even though the variant files are already present.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        p = _pipeline(layout)

        # First run — variants not yet present, all three generated.
        cb, generated = _callbacks()
        run_pipeline(p, str(seed), cb, base_directory=str(working))
        assert len(generated) == 3

        # Simulate generation: place variant files in the working directory.
        for modifier in ("_apple", "_banana", "_cherry"):
            (working / f"{STEM_A}{modifier}.jpg").touch()

        # Second run WITHOUT clearing the gate cache — stale scan still shows
        # no variants, so the pipeline incorrectly generates again.
        cb2, generated2 = _callbacks()
        run_pipeline(p, str(seed), cb2, base_directory=str(working))
        assert len(generated2) == 3, (
            "Stale cache bug: expected duplicate generate without cache clear"
        )

    def test_clearing_cache_skips_already_generated(self, layout):
        """
        After clearing the gate cache (as the pipeline worker now does at the
        start of each run), the second run correctly detects the existing variant
        files in the working directory and skips all three GENERATE actions.
        """
        working, *_ = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        p = _pipeline(layout)

        # First run — variants not yet present, all three generated.
        cb, generated = _callbacks()
        run_pipeline(p, str(seed), cb, base_directory=str(working))
        assert len(generated) == 3

        # Simulate generation: place variant files in the working directory.
        for modifier in ("_apple", "_banana", "_cherry"):
            (working / f"{STEM_A}{modifier}.jpg").touch()

        # Clear the gate cache as the pipeline worker now does between runs.
        clear_generate_gate_cache()

        # Second run — fresh scan finds the variants, all three skipped.
        cb2, generated2 = _callbacks()
        run_pipeline(p, str(seed), cb2, base_directory=str(working))
        assert generated2 == [], (
            "After cache clear, variants in working dir must suppress all GENERATEs"
        )

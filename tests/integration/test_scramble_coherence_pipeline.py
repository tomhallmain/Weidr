"""
Integration tests for the scramble-coherence demo pipeline.

Exercises the full run_pipeline() path against real files on disk, verifying
that the pipeline makes correct GENERATE / SCRAMBLE decisions for each image
state as documented in ClassifierPipelines.build_scramble_coherence_pipeline().

Suffix reference
────────────────
  _coherent  (8 chars) → GENERATE action
  _semiinco  (8 chars) → SCRAMBLE action  (semi_scramble_image)
  _inco      (4 chars) → SCRAMBLE action  (scramble_image)

Stems used
──────────
Long stems with numeric timestamps ensure extract_filename_base_stem() returns
the stem before the category suffix, matching the same convention used by the
category-fill pipeline tests.  Short stems (e.g. "photo") would cause suffixes
longer than 4 chars to be absorbed into the base stem.

  STEM_A = "rose_17820172251234"
  STEM_B = "lily_29134560782345"

Directory layout
────────────────
  <tmp>/working/   ← working directory (base_directory for run_pipeline)
"""

import os

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
    Create working directory and configure search paths.
    Returns (working,).
    """
    working = tmp_path / "working"
    working.mkdir(parents=True)

    monkeypatch.setattr(
        config, "directories_to_search_for_related_images", [str(working)],
    )

    clear_base_stem_dir_cache()
    clear_generate_gate_cache()

    yield (working,)

    clear_base_stem_dir_cache()
    clear_generate_gate_cache()


def _pipeline(*, active=True):
    p = ClassifierPipelines.build_scramble_coherence_pipeline()
    p.is_active = active
    return p


def _callbacks():
    """Return (ActionCallbacks, generated list, scrambled list)."""
    generated = []
    scrambled = []
    cb = ActionCallbacks(
        generate_callback=lambda path, modifier, *a, **kw:
            generated.append((path, modifier)),
        scramble_callback=lambda path, modifier=None:
            scrambled.append((path, modifier)),
    )
    return cb, generated, scrambled


# ---------------------------------------------------------------------------
# Pipeline meta-tests
# ---------------------------------------------------------------------------

class TestPipelineMeta:
    def test_pipeline_is_inactive_by_default(self):
        p = ClassifierPipelines.build_scramble_coherence_pipeline()
        assert p.is_active is False

    def test_inactive_pipeline_returns_none(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        cb, generated, scrambled = _callbacks()
        result = run_pipeline(_pipeline(active=False), str(seed), cb,
                              base_directory=str(working))
        assert result is None
        assert generated == []
        assert scrambled == []

    def test_node_names_present(self):
        p = _pipeline()
        names = [n.name for n in p.nodes]
        assert "Unknown-suffix guard"              in names
        assert "Stem uniqueness check"             in names
        assert "Generate coherent variant"         in names
        assert "Scramble semi-incoherent variant"  in names
        assert "Scramble incoherent variant"       in names

    def test_category_map_keys(self):
        p = _pipeline()
        assert set(p.category_map.keys()) == {"Coherent", "Semi-incoherent", "Incoherent"}

    def test_category_map_suffixes(self):
        p = _pipeline()
        assert p.category_map["Coherent"]        == "_coherent"
        assert p.category_map["Semi-incoherent"] == "_semiinco"
        assert p.category_map["Incoherent"]      == "_inco"

    def test_pipeline_validates_cleanly(self):
        p = _pipeline()
        errors = p.validate()
        assert errors == [], errors

    def test_pipeline_has_no_category_warnings(self):
        p = _pipeline()
        warnings = p.validate_warnings()
        assert warnings == [], warnings

    def test_coherent_node_uses_generate_action(self):
        p = _pipeline()
        node = next(n for n in p.nodes if n.name == "Generate coherent variant")
        assert node.on_match.action_type == ClassifierActionType.GENERATE
        assert node.on_match.action_modifier == "_coherent"

    def test_semiinco_node_uses_scramble_action(self):
        p = _pipeline()
        node = next(n for n in p.nodes if n.name == "Scramble semi-incoherent variant")
        assert node.on_match.action_type == ClassifierActionType.SCRAMBLE
        assert node.on_match.action_modifier == "_semiinco"

    def test_inco_node_uses_scramble_action(self):
        p = _pipeline()
        node = next(n for n in p.nodes if n.name == "Scramble incoherent variant")
        assert node.on_match.action_type == ClassifierActionType.SCRAMBLE
        assert node.on_match.action_modifier == "_inco"


# ---------------------------------------------------------------------------
# Case: Fresh seed — all three variants missing → all three actions fire
# ---------------------------------------------------------------------------

class TestFreshSeed:
    def test_generates_all_missing_variants(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        cb, generated, scrambled = _callbacks()
        result = run_pipeline(_pipeline(), str(seed), cb,
                              base_directory=str(working))

        assert result == ClassifierActionType.SCRAMBLE   # last EXECUTE_AND_CONTINUE
        assert len(generated) == 1
        assert generated[0][1] == "_coherent"
        assert len(scrambled) == 2
        modifiers = [m for _, m in scrambled]
        assert "_semiinco" in modifiers
        assert "_inco"     in modifiers

    def test_all_three_paths_are_the_seed(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        for path, _ in generated + scrambled:
            assert path == str(seed)

    def test_two_distinct_stems_each_fire_all_three(self, layout):
        (working,) = layout
        seed_a = working / f"{STEM_A}.jpg"
        seed_b = working / f"{STEM_B}.jpg"
        seed_a.touch()
        seed_b.touch()

        p = _pipeline()
        cb, generated, scrambled = _callbacks()
        run_pipeline(p, str(seed_a), cb, base_directory=str(working))
        run_pipeline(p, str(seed_b), cb, base_directory=str(working))

        assert len(generated) == 2
        assert len(scrambled) == 4


# ---------------------------------------------------------------------------
# Case: Some variants already present → those categories skipped
# ---------------------------------------------------------------------------

class TestPartiallyFilledVariants:
    def test_coherent_present_skips_generate(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}_coherent.jpg").touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        assert generated == []                      # coherent already exists
        assert len(scrambled) == 2                  # semiinco + inco still fire

    def test_semiinco_present_skips_semiinco_scramble(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}_semiinco.jpg").touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in scrambled]
        assert "_semiinco" not in modifiers
        assert "_inco"     in     modifiers
        assert len(generated) == 1                  # coherent still fires

    def test_inco_present_skips_inco_scramble(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}_inco.jpg").touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        modifiers = [m for _, m in scrambled]
        assert "_inco"     not in modifiers
        assert "_semiinco" in     modifiers

    def test_all_present_no_actions(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}_coherent.jpg").touch()
        (working / f"{STEM_A}_semiinco.jpg").touch()
        (working / f"{STEM_A}_inco.jpg").touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        assert generated == []
        assert scrambled == []


# ---------------------------------------------------------------------------
# Case: Unknown-suffix guard
# ---------------------------------------------------------------------------

class TestUnknownSuffixGuard:
    def test_unrecognised_suffix_rejects(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}_zzz.jpg").touch()    # unrecognised suffix

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        assert generated == []
        assert scrambled == []

    def test_known_suffixes_pass_guard(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        (working / f"{STEM_A}_coherent.jpg").touch()   # recognised → guard passes

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        # coherent already present; semiinco and inco still fire
        assert generated == []
        assert len(scrambled) == 2


# ---------------------------------------------------------------------------
# Case: Scramble callback receives correct modifier
# ---------------------------------------------------------------------------

class TestCallbackModifiers:
    def test_semiinco_modifier_passed_to_callback(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        # Pre-fill coherent and inco so only semiinco fires
        (working / f"{STEM_A}_coherent.jpg").touch()
        (working / f"{STEM_A}_inco.jpg").touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        assert len(scrambled) == 1
        assert scrambled[0] == (str(seed), "_semiinco")

    def test_inco_modifier_passed_to_callback(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        # Pre-fill coherent and semiinco so only inco fires
        (working / f"{STEM_A}_coherent.jpg").touch()
        (working / f"{STEM_A}_semiinco.jpg").touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        assert len(scrambled) == 1
        assert scrambled[0] == (str(seed), "_inco")

    def test_coherent_modifier_passed_to_generate_callback(self, layout):
        (working,) = layout
        seed = working / f"{STEM_A}.jpg"
        seed.touch()
        # Pre-fill scramble variants so only coherent fires
        (working / f"{STEM_A}_semiinco.jpg").touch()
        (working / f"{STEM_A}_inco.jpg").touch()

        cb, generated, scrambled = _callbacks()
        run_pipeline(_pipeline(), str(seed), cb, base_directory=str(working))

        assert len(generated) == 1
        assert generated[0] == (str(seed), "_coherent")
        assert scrambled == []

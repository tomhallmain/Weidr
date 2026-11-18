"""
Unit tests for preferred_target_category: the stem group's file of that
category is evaluated in place of the reached file (compare.pipeline_context,
run_pipeline_with_context).

Stem groups are real files in tmp_path; the runner's base-stem extraction is
patched to a fixed stem so the tests do not depend on its heuristics.
"""

import os

import pytest
from unittest.mock import patch

from compare.action_callbacks import ActionCallbacks
from compare.classifier_pipeline import (
    AlwaysCondition,
    ClassifierPipeline,
    FilenameContainsCondition,
    NodeOutcome,
    OutcomeType,
    PipelineNode,
    RelatedImageCondition,
)
from compare.classifier_pipeline_batch import _seed_summary
from compare.classifier_pipeline_runner import run_pipeline, run_pipeline_with_context
from compare.pipeline_context import build_pipeline_context
from compare.pipeline_run_report import PipelineRunReport
from files.related_image import (
    clear_base_stem_dir_cache,
    clear_generate_gate_cache,
    find_files_by_exact_stem,
)
from utils.constants import ClassifierActionType
from utils.translations import _

BASE = "seed12345678901234"
CAT_MAP = {"Apple": "_apple", "Coherent": "_coherent", "Inco": "_inco"}


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_base_stem_dir_cache()
    clear_generate_gate_cache()
    yield
    clear_base_stem_dir_cache()
    clear_generate_gate_cache()


@pytest.fixture
def fixed_base_stem():
    with patch("compare.classifier_pipeline_runner.extract_filename_base_stem",
               lambda p: BASE):
        yield


def _files(directory, *names):
    paths = []
    for name in names:
        path = os.path.join(str(directory), name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x")
        paths.append(path)
    return paths


def _generate_node(name, suffix, condition=None):
    return PipelineNode(
        name=name,
        condition=condition or AlwaysCondition(),
        on_match=NodeOutcome(OutcomeType.EXECUTE_AND_CONTINUE,
                             action_type=ClassifierActionType.GENERATE, action_modifier=suffix),
        on_no_match=NodeOutcome.continue_(),
    )


def _pipeline(*nodes, preferred="Coherent", seed_category="", **kwargs):
    return ClassifierPipeline(
        name="p", category_map=dict(CAT_MAP), seed_category=seed_category,
        preferred_target_category=preferred, nodes=list(nodes), **kwargs,
    )


def _record_dispatch():
    """Patch _dispatch_action; return (patcher, calls) with (action, modifier, path) per call."""
    calls = []

    def fake(action_type, modifier, _pipeline_name, path, *a, **kw):
        if action_type is not None:
            calls.append((action_type, modifier, path))

    return patch("compare.classifier_pipeline_runner._dispatch_action", side_effect=fake), calls


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_configuration_has_no_preferred_errors(self):
        assert _pipeline(_generate_node("g", "_inco")).validate() == []

    def test_category_not_in_map(self):
        errors = _pipeline(_generate_node("g", "_inco"), preferred="Pear").validate()
        assert _("preferred_target_category '{0}' is not a key in category_map.").format("Pear") in errors

    def test_empty_category_map(self):
        p = ClassifierPipeline(name="p", preferred_target_category="Coherent",
                               nodes=[_generate_node("g", "_inco")])
        assert (_("preferred_target_category '{0}' is set but category_map is empty.").format("Coherent")
                in p.validate())

    def test_equal_to_seed_category(self):
        errors = _pipeline(_generate_node("g", "_inco"), seed_category="Coherent").validate()
        assert _("preferred_target_category '{0}' must differ from seed_category.").format("Coherent") in errors

    def test_requires_dedupe_stem_groups(self):
        errors = _pipeline(_generate_node("g", "_inco"), dedupe_stem_groups=False).validate()
        assert _("preferred_target_category '{0}' requires dedupe_stem_groups.").format("Coherent") in errors


class TestSerialization:
    def test_round_trip(self):
        p = _pipeline(_generate_node("g", "_inco"))
        assert ClassifierPipeline.from_dict(p.to_dict()).preferred_target_category == "Coherent"

    def test_omitted_when_unset(self):
        p = _pipeline(_generate_node("g", "_inco"), preferred="")
        assert "preferred_target_category" not in p.to_dict()
        assert ClassifierPipeline.from_dict(p.to_dict()).preferred_target_category == ""

    def test_suffix_follows_category_map(self):
        assert _pipeline().preferred_target_suffix() == "_coherent"
        assert _pipeline(preferred="").preferred_target_suffix() == ""


# ---------------------------------------------------------------------------
# Exact-stem index
# ---------------------------------------------------------------------------

class TestFindFilesByExactStem:
    def test_direct_entries_only_case_insensitive(self, tmp_path):
        upper, lower = _files(tmp_path, BASE + "_Coherent.png", BASE + "_coherent.jpg")
        _files(tmp_path, os.path.join("sub", BASE + "_coherent.webp"), BASE + "_coherent_2.png")
        assert sorted(find_files_by_exact_stem(str(tmp_path), BASE + "_COHERENT")) == sorted([upper, lower])

    def test_index_reused_until_cleared(self, tmp_path):
        assert find_files_by_exact_stem(str(tmp_path), BASE) == []
        (seed,) = _files(tmp_path, BASE + ".png")
        assert find_files_by_exact_stem(str(tmp_path), BASE) == []
        clear_base_stem_dir_cache(str(tmp_path))
        assert find_files_by_exact_stem(str(tmp_path), BASE) == [seed]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestBuildPipelineContext:
    def test_exact_match_from_seed(self, tmp_path):
        seed, preferred = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")
        ctx = build_pipeline_context(_pipeline(), seed, BASE)
        assert ctx.skip_reason is None
        assert ctx.preferred_path == preferred
        assert ctx.evaluated_path == preferred
        assert ctx.reached_path == seed
        assert ctx.seed_path == seed
        assert ctx.evaluated_is_seed_by_name is False

    def test_seed_reference_kept_when_derivative_reached(self, tmp_path):
        seed, preferred, inco = _files(
            tmp_path, BASE + ".png", BASE + "_coherent.png", BASE + "_inco.png")
        ctx = build_pipeline_context(_pipeline(), inco, BASE)
        assert ctx.reached_path == inco
        assert ctx.preferred_path == preferred
        assert ctx.seed_path == seed

    def test_numbered_variant_alone_does_not_count(self, tmp_path):
        (seed,) = _files(tmp_path, BASE + ".png", BASE + "_coherent_2.png")[:1]
        ctx = build_pipeline_context(_pipeline(), seed, BASE)
        assert ctx.preferred_path is None
        assert ctx.skip_reason is not None
        assert ctx.evaluated_path == seed

    def test_no_match(self, tmp_path):
        (seed,) = _files(tmp_path, BASE + ".png")
        ctx = build_pipeline_context(_pipeline(), seed, BASE)
        assert ctx.preferred_path is None
        assert ctx.skip_reason is not None

    def test_several_matches(self, tmp_path):
        seed = _files(tmp_path, BASE + ".png", BASE + "_coherent.png", BASE + "_coherent.jpg")[0]
        ctx = build_pipeline_context(_pipeline(), seed, BASE)
        assert ctx.preferred_path is None
        assert ctx.skip_reason is not None

    def test_match_in_subdirectory_only_is_ignored(self, tmp_path):
        seed = _files(tmp_path, BASE + ".png", os.path.join("sub", BASE + "_coherent.png"))[0]
        ctx = build_pipeline_context(_pipeline(), seed, BASE)
        assert ctx.preferred_path is None
        assert ctx.skip_reason is not None

    def test_disallowed_media_type_skips(self, tmp_path):
        seed = _files(tmp_path, BASE + ".png", BASE + "_coherent.mp4")[0]
        p = _pipeline()
        p.media_type_allowed = lambda path: not path.endswith(".mp4")
        ctx = build_pipeline_context(p, seed, BASE)
        assert ctx.preferred_path is None
        assert ctx.skip_reason is not None

    def test_feature_off_keeps_reached_file(self, tmp_path):
        seed = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")[0]
        ctx = build_pipeline_context(_pipeline(preferred=""), seed, BASE)
        assert not ctx.substitution_enabled
        assert ctx.skip_reason is None
        assert ctx.evaluated_path == seed
        assert ctx.seed_path == seed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("fixed_base_stem")
class TestRunPipelineSubstitution:
    def test_seed_reached_generates_from_preferred_file(self, tmp_path):
        seed, preferred = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")
        patcher, calls = _record_dispatch()
        stems = set()
        with patcher:
            outcome = run_pipeline_with_context(
                _pipeline(_generate_node("g", "_inco")), seed, ActionCallbacks(),
                base_directory=str(tmp_path), processed_stems=stems,
            )
        assert calls == [(ClassifierActionType.GENERATE, "_inco", preferred)]
        assert outcome.evaluated
        assert outcome.context.seed_path == seed
        assert BASE in stems

    def test_other_derivative_reached_first(self, tmp_path):
        _seed, preferred, inco = _files(
            tmp_path, BASE + ".png", BASE + "_coherent.png", BASE + "_inco.png")
        patcher, calls = _record_dispatch()
        stems = set()
        with patcher:
            run_pipeline(_pipeline(_generate_node("g", "_apple")), inco, ActionCallbacks(),
                         base_directory=str(tmp_path), processed_stems=stems)
        assert calls == [(ClassifierActionType.GENERATE, "_apple", preferred)]
        assert BASE in stems

    def test_conditions_see_the_preferred_file(self, tmp_path):
        seed, preferred = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")
        node = _generate_node("g", "_inco", condition=FilenameContainsCondition(["_coherent"]))
        patcher, calls = _record_dispatch()
        with patcher:
            run_pipeline(_pipeline(node), seed, ActionCallbacks(),
                         base_directory=str(tmp_path), processed_stems=set())
        assert calls == [(ClassifierActionType.GENERATE, "_inco", preferred)]

    def test_group_without_preferred_file_is_skipped_and_marked_done(self, tmp_path):
        seed, inco = _files(tmp_path, BASE + ".png", BASE + "_inco.png")
        report = PipelineRunReport()
        patcher, calls = _record_dispatch()
        stems = set()
        with patcher:
            outcome = run_pipeline_with_context(
                _pipeline(_generate_node("g", "_inco")), seed, ActionCallbacks(),
                base_directory=str(tmp_path), processed_stems=stems, report=report,
            )
            # A later member of the group is skipped by the done mark.
            later = run_pipeline_with_context(
                _pipeline(_generate_node("g", "_inco")), inco, ActionCallbacks(),
                base_directory=str(tmp_path), processed_stems=stems, report=report,
            )
        assert calls == []
        assert not outcome.evaluated and outcome.context.skip_reason is not None
        assert BASE in stems
        assert later.context is None
        infos = report.messages_by_severity("INFO")
        assert len(infos) == 1 and infos[0].data == {"preferred_source_skip": True}

    def test_media_type_of_preferred_file_decides(self, tmp_path):
        seed = _files(tmp_path, BASE + ".mp4", BASE + "_coherent.png")[0]
        p = _pipeline(_generate_node("g", "_inco"))
        p.media_type_allowed = lambda path: not path.endswith(".mp4")
        patcher, calls = _record_dispatch()
        with patcher:
            run_pipeline(p, seed, ActionCallbacks(), base_directory=str(tmp_path),
                         processed_stems=set())
        assert [c[2] for c in calls] == [os.path.join(str(tmp_path), BASE + "_coherent.png")]


@pytest.mark.usefixtures("fixed_base_stem")
class TestGenerateGuards:
    def test_seed_and_preferred_categories_both_skipped(self, tmp_path):
        seed, preferred = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")
        p = _pipeline(
            _generate_node("apple", "_apple"),
            _generate_node("coherent", "_coherent"),
            _generate_node("inco", "_inco"),
            seed_category="Apple",
        )
        report = PipelineRunReport()
        patcher, calls = _record_dispatch()
        with patcher:
            run_pipeline(p, seed, ActionCallbacks(), base_directory=str(tmp_path),
                         processed_stems=set(), report=report)
        assert calls == [(ClassifierActionType.GENERATE, "_inco", preferred)]
        data = [m.data for m in report.messages_by_severity("INFO")]
        assert {"seed_category_skip": True} in data
        assert {"preferred_target_skip": True} in data

    def test_seed_guard_covers_group_when_derivative_reached(self, tmp_path):
        _seed, preferred, inco = _files(
            tmp_path, BASE + ".png", BASE + "_coherent.png", BASE + "_inco.png")
        p = _pipeline(_generate_node("apple", "_apple"), seed_category="Apple")
        patcher, calls = _record_dispatch()
        with patcher:
            run_pipeline(p, inco, ActionCallbacks(), base_directory=str(tmp_path),
                         processed_stems=set())
        assert calls == []


@pytest.mark.usefixtures("fixed_base_stem")
class TestRelatedImageGate:
    def _gate_node(self):
        # Searches base_directory, not the configured directories.
        condition = RelatedImageCondition(edit_suffix="_inco", use_configured_search_directories=False)
        return _generate_node("g", "_inco", condition=condition)

    def test_upstream_block_lifted_for_preferred_file(self, tmp_path):
        seed, preferred = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")
        seen = []

        def fake_gate(path, suffix, directory, threshold=1, **kw):
            seen.append((path, kw))
            return True

        patcher, calls = _record_dispatch()
        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate), patcher:
            run_pipeline(_pipeline(self._gate_node()), seed, ActionCallbacks(),
                         base_directory=str(tmp_path), processed_stems=set())
        assert seen == [(preferred, {"ignore_upstream_source": True})]
        assert calls == [(ClassifierActionType.GENERATE, "_inco", preferred)]

    def test_gate_unchanged_without_substitution(self, tmp_path):
        (seed,) = _files(tmp_path, BASE + ".png")
        seen = []

        def fake_gate(path, suffix, directory, threshold=1, **kw):
            seen.append((path, kw))
            return True

        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            run_pipeline(_pipeline(self._gate_node(), preferred=""), seed, ActionCallbacks(),
                         base_directory=str(tmp_path), dry_run=True)
        assert seen == [(seed, {})]


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("fixed_base_stem")
class TestTraceability:
    def test_decision_record_names_all_three_files(self, tmp_path):
        seed, preferred, inco = _files(
            tmp_path, BASE + ".png", BASE + "_coherent.png", BASE + "_inco.png")
        report = PipelineRunReport()
        p = _pipeline(_generate_node("g", "_apple"), record_node_verdicts=True)
        run_pipeline(p, inco, ActionCallbacks(), base_directory=str(tmp_path),
                     processed_stems=set(), report=report, dry_run=True)
        (record,) = report.decisions()
        assert record["path"] == preferred
        assert record["reached_path"] == inco
        assert record["seed_path"] == seed
        assert record["preferred_path"] == preferred

    def test_batch_summary_names_preferred_file_and_seed(self, tmp_path):
        seed, preferred = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")
        report = PipelineRunReport()
        outcome = run_pipeline_with_context(
            _pipeline(_generate_node("g", "_inco")), seed, ActionCallbacks(),
            base_directory=str(tmp_path), processed_stems=set(), report=report, dry_run=True,
        )
        summary = _seed_summary(report, seed, outcome, 0, True)
        assert summary.startswith(
            _("{0} (seed: {1})").format(os.path.basename(preferred), os.path.basename(seed))
        )

    def test_batch_summary_for_skipped_group(self, tmp_path):
        (seed,) = _files(tmp_path, BASE + ".png")
        report = PipelineRunReport()
        outcome = run_pipeline_with_context(
            _pipeline(_generate_node("g", "_inco")), seed, ActionCallbacks(),
            base_directory=str(tmp_path), processed_stems=set(), report=report,
        )
        summary = _seed_summary(report, seed, outcome, 0, True)
        assert outcome.context.skip_reason in summary

    def test_batch_summary_none_for_seed_skipped_by_done_mark(self, tmp_path):
        """A derivative reached first handled the group; the seed reached
        afterwards must not add a second summary."""
        seed, _preferred, inco = _files(
            tmp_path, BASE + ".png", BASE + "_coherent.png", BASE + "_inco.png")
        report = PipelineRunReport()
        stems = set()
        p = _pipeline(_generate_node("g", "_apple"))
        first = run_pipeline_with_context(p, inco, ActionCallbacks(), base_directory=str(tmp_path),
                                          processed_stems=stems, report=report, dry_run=True)
        later = run_pipeline_with_context(p, seed, ActionCallbacks(), base_directory=str(tmp_path),
                                          processed_stems=stems, report=report, dry_run=True)
        assert _seed_summary(report, inco, first, 0, True) is not None
        assert _seed_summary(report, seed, later, 0, True) is None


# ---------------------------------------------------------------------------
# Run Current
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("fixed_base_stem")
class TestRunCurrent:
    def test_substitutes_without_processed_stems(self, tmp_path):
        seed, preferred = _files(tmp_path, BASE + ".png", BASE + "_coherent.png")
        patcher, calls = _record_dispatch()
        with patcher:
            outcome = run_pipeline_with_context(
                _pipeline(_generate_node("g", "_inco")), seed, ActionCallbacks(),
                base_directory=str(tmp_path), assume_seed=True,
            )
        assert calls == [(ClassifierActionType.GENERATE, "_inco", preferred)]
        assert outcome.context.substituted

    def test_without_preferred_file_no_action_and_reason(self, tmp_path):
        (seed,) = _files(tmp_path, BASE + ".png")
        patcher, calls = _record_dispatch()
        with patcher:
            outcome = run_pipeline_with_context(
                _pipeline(_generate_node("g", "_inco")), seed, ActionCallbacks(),
                base_directory=str(tmp_path), assume_seed=True,
            )
        assert calls == []
        assert outcome.action is None
        assert outcome.context.skip_reason is not None

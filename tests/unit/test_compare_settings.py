"""Tests for compare/compare_settings.py (compare settings as plain data, for
the MCP sessions) and CompareManager.effective_threshold/_counter_limit, the
mode binding of its threshold and replace_mode_instances."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from compare import compare_filters
from compare.compare_filters import ClassifierFilter, SizeFilter
from compare.compare_args import CompareArgs
from compare.compare_manager import MAX_INSTANCES, CombinationLogic, CompareManager
from compare.compare_settings import (
    apply_compare_settings,
    describe_compare_settings,
    list_classifier_models,
    parse_filter_strict,
    resolve_run_compare_mode,
)
from utils.config import config
from utils.constants import CompareMode, Sort
from utils.headless_app_actions import build_headless_app_actions
from utils.ui_responsiveness import NullResponsiveness


@pytest.fixture
def cm(tmp_path):
    base_dir = str(tmp_path)
    manager = CompareManager(
        master=None,
        app_actions=build_headless_app_actions({"get_base_dir": lambda: base_dir}),
        get_base_dir=lambda: base_dir,
        responsiveness=NullResponsiveness(),
    )
    manager.set_compare_mode(CompareMode.CLIP_EMBEDDING)
    return manager


@pytest.fixture
def registered_classifier(monkeypatch):
    """One registered image classifier 'content' (photo/drawing/nsfw, positive group nsfw)."""
    cfg = SimpleNamespace(model_name="content", model_categories=["photo", "drawing", "nsfw"],
                          positive_groups=[["nsfw"]])
    manager = MagicMock()
    manager.resolve_registered_model_name.side_effect = lambda n: "content" if n == "content" else None
    manager.classifier_metadata = {"content": cfg}
    manager.get_model_configs.return_value = [cfg]
    monkeypatch.setattr(compare_filters, "_classifier_manager", lambda domain: manager)
    return manager


CLASSIFIER_FILTER = {"type": "classifier", "classifier_name": "content", "categories": ["photo"]}


class TestEffectiveSettings:
    def test_defaults_follow_mode_and_config(self, cm):
        assert cm.effective_threshold() == config.embedding_similarity_threshold
        cm.set_compare_mode(CompareMode.COLOR_MATCHING)
        assert cm.effective_threshold() == config.color_diff_threshold
        assert cm.effective_counter_limit() == config.file_counter_limit

    def test_color_histogram_default_is_its_distance_threshold(self, cm):
        from compare.compare_color_histogram import CompareColorHistogram
        cm.set_compare_mode(CompareMode.COLOR_HISTOGRAM)
        assert cm.effective_threshold() == CompareColorHistogram.DEFAULT_THRESHOLD == 0.2

    def test_set_values_win(self, cm):
        cm.set_threshold(0.7)
        cm.set_counter_limit(10)
        assert cm.effective_threshold() == 0.7
        assert cm.effective_counter_limit() == 10

    def test_models_mode_ignores_set_threshold(self, cm):
        from compare.compare_models import CompareModels
        cm.set_compare_mode(CompareMode.MODELS)
        cm.set_threshold(0.7)
        assert cm.effective_threshold() == CompareModels.THRESHOLD_MATCH


class TestDescribe:
    def test_reports_current_settings(self, cm):
        cm.set_threshold(0.75)
        cm.set_data_filter(SizeFilter(min_size=(10, 10)))
        settings = describe_compare_settings(cm)
        assert settings["compare_mode"] == "CLIP_EMBEDDING"
        assert settings["composite"] is False
        assert settings["threshold"] == 0.75
        assert settings["effective_threshold"] == 0.75
        assert settings["counter_limit"] is None
        assert settings["group_sort"] == config.compare_group_sort.name
        assert settings["data_filter"]["type"] == "size"
        assert [i["compare_mode"] for i in settings["instances"]] == ["CLIP_EMBEDDING"]


class TestApply:
    def test_applies_present_keys_only(self, cm):
        cm.set_counter_limit(50)
        apply_compare_settings(cm, {"threshold": 0.8, "overwrite": True})
        assert cm.get_threshold() == 0.8
        assert cm.get_overwrite() is True
        assert cm.get_counter_limit() == 50

    def test_null_resets_threshold_and_counter_limit(self, cm):
        cm.set_threshold(0.8)
        cm.set_counter_limit(50)
        apply_compare_settings(cm, {"threshold": None, "counter_limit": None})
        assert cm.get_threshold() is None
        assert cm.get_counter_limit() is None

    def test_round_trip_through_describe(self, cm, registered_classifier, monkeypatch):
        monkeypatch.setattr(config, "persist", lambda: None)
        apply_compare_settings(cm, {"data_filter": CLASSIFIER_FILTER, "counter_limit": 5})
        before = describe_compare_settings(cm)
        changeable = {k: before[k] for k in (
            "instances", "combination_logic", "threshold", "threshold_mode", "counter_limit",
            "overwrite", "store_checkpoints", "use_matrix_comparison",
            "search_only_return_closest", "group_sort", "data_filter")}
        apply_compare_settings(cm, changeable)
        assert describe_compare_settings(cm) == before

    def test_classifier_data_filter_set_and_cleared(self, cm, registered_classifier):
        apply_compare_settings(cm, {"data_filter": CLASSIFIER_FILTER})
        assert cm.get_data_filter() == ClassifierFilter(classifier_name="content", categories=["photo"])
        apply_compare_settings(cm, {"data_filter": None})
        assert cm.get_data_filter() is None

    def test_group_sort_is_persisted(self, cm, monkeypatch):
        persisted = []
        monkeypatch.setattr(config, "persist", lambda: persisted.append(config.compare_group_sort))
        apply_compare_settings(cm, {"group_sort": "ASC"})
        assert config.compare_group_sort == Sort.ASC
        assert persisted == [Sort.ASC]

    def test_search_only_return_closest(self, cm, monkeypatch):
        monkeypatch.setattr(config, "search_only_return_closest", False)
        apply_compare_settings(cm, {"search_only_return_closest": True})
        assert config.search_only_return_closest is True

    @pytest.mark.parametrize("changes", [
        {},
        {"bogus": 1},
        {"threshold": "high"},
        {"threshold": True},
        {"threshold": -1},
        {"counter_limit": 0},
        {"counter_limit": 2.5},
        {"overwrite": "yes"},
        {"group_sort": "RANDOM"},
    ])
    def test_invalid_input_rejected(self, cm, changes):
        with pytest.raises(ValueError):
            apply_compare_settings(cm, changes)

    def test_nothing_applied_when_any_key_is_invalid(self, cm):
        with pytest.raises(ValueError):
            apply_compare_settings(cm, {"threshold": 0.8, "counter_limit": 0})
        assert cm.get_threshold() is None

    def test_unregistered_classifier_rejected(self, cm, registered_classifier):
        with pytest.raises(ValueError, match="other"):
            apply_compare_settings(cm, {"data_filter": {**CLASSIFIER_FILTER, "classifier_name": "other"}})
        assert cm.get_data_filter() is None

    def test_unknown_category_rejected(self, cm, registered_classifier):
        with pytest.raises(ValueError, match="landscape"):
            apply_compare_settings(cm, {"data_filter": {**CLASSIFIER_FILTER, "categories": ["landscape"]}})

    def test_classifier_checked_without_loading_a_model(self, cm, registered_classifier):
        apply_compare_settings(cm, {"data_filter": CLASSIFIER_FILTER})
        registered_classifier.get_classifier.assert_not_called()


TWO_INSTANCES = [
    {"compare_mode": "CLIP_EMBEDDING", "weight": 2.0, "search_text": "cat"},
    {"compare_mode": "COLOR_MATCHING", "threshold": 12},
]


class TestThresholdMode:
    def test_threshold_applies_only_to_the_mode_it_was_set_for(self, cm):
        cm.set_threshold(0.8)
        cm.set_compare_mode(CompareMode.COLOR_MATCHING)
        assert cm.get_threshold() is None
        assert cm.effective_threshold() == config.color_diff_threshold
        cm.set_compare_mode(CompareMode.CLIP_EMBEDDING)
        assert cm.effective_threshold() == 0.8

    def test_threshold_mode_sets_it_for_another_mode(self, cm):
        apply_compare_settings(cm, {"threshold": 15, "threshold_mode": "COLOR_MATCHING"})
        assert cm.effective_threshold() == config.embedding_similarity_threshold
        cm.set_compare_mode(CompareMode.COLOR_MATCHING)
        assert cm.effective_threshold() == 15

    def test_describe_reports_the_mode(self, cm):
        apply_compare_settings(cm, {"threshold": 15, "threshold_mode": "COLOR_MATCHING"})
        settings = describe_compare_settings(cm)
        assert (settings["threshold"], settings["threshold_mode"]) == (15, "COLOR_MATCHING")
        assert settings["effective_threshold"] == config.embedding_similarity_threshold

    def test_threshold_without_mode_is_for_the_mode_after_instances_change(self, cm):
        apply_compare_settings(cm, {"instances": [{"compare_mode": "COLOR_MATCHING"}], "threshold": 15})
        assert cm.get_threshold_setting() == (15, CompareMode.COLOR_MATCHING)

    @pytest.mark.parametrize("changes", [
        {"threshold_mode": "CLIP_EMBEDDING"},
        {"threshold": 0.5, "threshold_mode": "NOT_A_MODE"},
    ])
    def test_invalid_threshold_mode_rejected(self, cm, changes):
        with pytest.raises(ValueError):
            apply_compare_settings(cm, changes)

    def test_snapshot_records_only_a_threshold_that_applies(self, cm):
        cm.set_threshold(0.8)
        cm.set_compare_mode(CompareMode.COLOR_MATCHING)
        assert cm.snapshot(CompareArgs(base_dir="/x")).run_settings.threshold is None

    def test_composite_instance_of_another_mode_uses_its_own_default(self, cm, monkeypatch):
        apply_compare_settings(cm, {"instances": [
            {"compare_mode": "CLIP_EMBEDDING"}, {"compare_mode": "COLOR_MATCHING"},
        ]})
        seen = {}

        def fake_wrapper(_iid, mode):
            wrapper = MagicMock(file_groups={})
            wrapper.run.side_effect = lambda a: seen.__setitem__(mode, a.threshold)
            return wrapper

        monkeypatch.setattr(cm, "_app_actions", MagicMock())
        monkeypatch.setattr(cm, "_ensure_wrapper", fake_wrapper)
        monkeypatch.setattr(cm, "_apply_combined_results_to_primary", lambda **_kw: None)
        args = CompareArgs()
        args.threshold = 0.8
        cm._run_composite(args)
        assert seen == {CompareMode.CLIP_EMBEDDING: 0.8,
                        CompareMode.COLOR_MATCHING: config.color_diff_threshold}


class TestInstances:
    def test_instances_replace_the_setup(self, cm):
        apply_compare_settings(cm, {"instances": TWO_INSTANCES, "combination_logic": "WEIGHTED"})
        assert cm.compare_mode == CompareMode.CLIP_EMBEDDING
        assert cm.is_composite_mode()
        assert cm.get_combination_logic() == CombinationLogic.WEIGHTED
        settings = describe_compare_settings(cm)
        assert settings["instances"] == [
            {"compare_mode": "CLIP_EMBEDDING", "enabled": True, "threshold": None, "weight": 2.0,
             "search_text": "cat", "search_text_negative": None},
            {"compare_mode": "COLOR_MATCHING", "enabled": True, "threshold": 12.0, "weight": 1.0,
             "search_text": None, "search_text_negative": None},
        ]

    def test_back_to_one_instance_is_not_composite(self, cm):
        apply_compare_settings(cm, {"instances": TWO_INSTANCES})
        apply_compare_settings(cm, {"instances": [{"compare_mode": "SIZE"}]})
        assert not cm.is_composite_mode()
        assert cm.compare_mode == CompareMode.SIZE

    def test_leading_instances_of_the_same_mode_are_kept(self, cm):
        apply_compare_settings(cm, {"instances": TWO_INSTANCES})
        ids = [c.instance_id for c in cm.get_mode_instances()]
        apply_compare_settings(cm, {"instances": [
            {"compare_mode": "CLIP_EMBEDDING", "weight": 3.0}, {"compare_mode": "SIZE"},
        ]})
        instances = cm.get_mode_instances()
        assert instances[0].instance_id == ids[0]
        assert (instances[0].weight, instances[0].search_text) == (3.0, None)
        assert instances[1].instance_id != ids[1]
        assert instances[1].compare_mode == CompareMode.SIZE

    def test_round_trip_of_a_composite_setup(self, cm, monkeypatch):
        monkeypatch.setattr(config, "persist", lambda: None)
        apply_compare_settings(cm, {"instances": TWO_INSTANCES, "combination_logic": "OR"})
        before = describe_compare_settings(cm)
        apply_compare_settings(cm, {k: before[k] for k in ("instances", "combination_logic")})
        assert describe_compare_settings(cm) == before

    @pytest.mark.parametrize("changes", [
        {"instances": []},
        {"instances": [{"compare_mode": "SIZE"}] * (MAX_INSTANCES + 1)},
        {"instances": [{}]},
        {"instances": [{"compare_mode": "NOT_A_MODE"}]},
        {"instances": [{"compare_mode": "SIZE", "bogus": 1}]},
        {"instances": [{"compare_mode": "SIZE", "search_text": "cat"}]},
        {"instances": [{"compare_mode": "CLIP_EMBEDDING", "enabled": False}, {"compare_mode": "SIZE"}]},
        {"instances": [{"compare_mode": "CLIP_EMBEDDING", "weight": None}]},
        {"instances": [{"compare_mode": "CLIP_EMBEDDING", "weight": -1}]},
        {"combination_logic": "XOR"},
    ])
    def test_invalid_input_rejected(self, cm, changes):
        with pytest.raises(ValueError):
            apply_compare_settings(cm, changes)

    def test_nothing_applied_when_instances_are_invalid(self, cm):
        with pytest.raises(ValueError):
            apply_compare_settings(cm, {"threshold": 0.8, "instances": [{"compare_mode": "NOT_A_MODE"}]})
        assert cm.get_threshold() is None
        assert [c.compare_mode for c in cm.get_mode_instances()] == [CompareMode.CLIP_EMBEDDING]


class TestResolveRunCompareMode:
    def test_without_mode_the_configured_mode(self, cm):
        assert resolve_run_compare_mode(cm, None, searching=False) == CompareMode.CLIP_EMBEDDING

    def test_with_mode_that_mode(self, cm):
        assert resolve_run_compare_mode(cm, "SIZE", searching=False) == CompareMode.SIZE

    def test_unknown_mode_rejected(self, cm):
        with pytest.raises(ValueError):
            resolve_run_compare_mode(cm, "NOT_A_MODE", searching=False)

    def test_composite_runs_only_without_mode(self, cm):
        apply_compare_settings(cm, {"instances": [{"compare_mode": "CLIP_EMBEDDING"}, {"compare_mode": "SIZE"}]})
        with pytest.raises(ValueError):
            resolve_run_compare_mode(cm, "CLIP_EMBEDDING", searching=False)
        assert resolve_run_compare_mode(cm, None, searching=False) == CompareMode.CLIP_EMBEDDING

    def test_composite_with_search_texts_runs_only_as_a_search(self, cm):
        apply_compare_settings(cm, {"instances": TWO_INSTANCES})
        with pytest.raises(ValueError):
            resolve_run_compare_mode(cm, None, searching=False)
        assert resolve_run_compare_mode(cm, None, searching=True) == CompareMode.CLIP_EMBEDDING


class TestParseFilterStrict:
    def test_nested_group_accepted(self):
        f = parse_filter_strict({"type": "group", "operator": "or", "filters": [
            {"type": "size", "min_size": [100, 100]},
            {**CLASSIFIER_FILTER, "mode": "exclude", "sample_ratio": 0.2, "positive_ratio": 0.5},
        ]})
        assert f.is_active()

    @pytest.mark.parametrize("d", [
        "size",
        {"type": "colour"},
        {"type": "size", "min_size": [100, 100], "extra": 1},
        {"type": "size", "min_size": [100]},
        {"type": "size", "size_tolerance": -1, "exact_size": [1, 1]},
        {"type": "model", "models": "sdxl"},
        {"type": "model", "models": ["sdxl"], "mode": "maybe"},
        {"type": "group", "operator": "xor", "filters": [{"type": "size", "min_size": [1, 1]}]},
        {"type": "group", "filters": []},
        {**CLASSIFIER_FILTER, "domain": "video"},
        {**CLASSIFIER_FILTER, "selection_mode": "auto"},
        {**CLASSIFIER_FILTER, "sample_ratio": 2},
        {**CLASSIFIER_FILTER, "classifier_name": ""},
        {"type": "classifier", "classifier_name": "content"},  # no categories: inactive
        {"type": "size"},  # no constraint: inactive
    ])
    def test_rejected(self, d):
        with pytest.raises(ValueError):
            parse_filter_strict(d)


def test_list_classifier_models(monkeypatch):
    import image.audio_classifier_manager as acm
    import image.image_classifier_manager as icm
    image_cfg = SimpleNamespace(model_name="content", model_categories=["photo", "nsfw"],
                                positive_groups=[["nsfw"]])
    audio_cfg = SimpleNamespace(model_name="speech", model_categories=["speech", "music"],
                                positive_groups=[])
    monkeypatch.setattr(icm.image_classifier_manager, "get_model_configs", lambda: [image_cfg])
    monkeypatch.setattr(acm.audio_classifier_manager, "get_model_configs", lambda: [audio_cfg])
    assert list_classifier_models() == {"classifiers": [
        {"domain": "image", "name": "content", "categories": ["photo", "nsfw"],
         "model_strategy_categories": ["nsfw"]},
        {"domain": "audio", "name": "speech", "categories": ["speech", "music"],
         "model_strategy_categories": []},
    ]}

"""Unit tests for interceptor rule matching, serialization, and apply()."""

import os
from unittest.mock import patch

import pytest

from files.file_interceptor_rule import (
    FileInterceptorRule,
    InterceptorAppliesTo,
    InterceptorBehavior,
    InterceptorTransformOp,
)
from files.file_interceptor_rules_manager import (
    FileInterceptorRulesManager,
    InterceptResult,
)
from files.marked_files import MarkedFiles
from utils.constants import CompareMediaType


@pytest.fixture(autouse=True)
def _clear_rules():
    saved = FileInterceptorRulesManager.rules[:]
    FileInterceptorRulesManager.rules = []
    yield
    FileInterceptorRulesManager.rules = saved


class _RecordingActions:
    """Minimal stand-in for AppActions capturing warn calls."""

    def __init__(self):
        self.warnings = []

    def warn(self, message, time_in_seconds=None):
        self.warnings.append(message)


def _block_rule(**kwargs) -> FileInterceptorRule:
    defaults = dict(name="block rule", behavior=InterceptorBehavior.BLOCK)
    defaults.update(kwargs)
    return FileInterceptorRule(**defaults)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_rule_with_no_conditions_matches_any_file():
    rule = _block_rule()
    assert FileInterceptorRulesManager.rule_matches(
        rule, "/src/anything.png", "/dest", is_moving=True
    )


def test_inactive_rule_never_matches():
    rule = _block_rule(is_active=False)
    assert not FileInterceptorRulesManager.rule_matches(
        rule, "/src/a.png", "/dest", is_moving=True
    )


def test_filename_pattern_matches_case_insensitively_by_default():
    rule = _block_rule(match_filename_patterns=["IMG_"])
    assert FileInterceptorRulesManager.rule_matches(
        rule, "/src/img_0042.png", "/dest", is_moving=True
    )


def test_filename_pattern_respects_case_sensitivity():
    rule = _block_rule(match_filename_patterns=["IMG_"], filename_case_sensitive=True)
    assert not FileInterceptorRulesManager.rule_matches(
        rule, "/src/img_0042.png", "/dest", is_moving=True
    )
    assert FileInterceptorRulesManager.rule_matches(
        rule, "/src/IMG_0042.png", "/dest", is_moving=True
    )


def test_target_dir_matches_exactly_but_not_subdirectory_by_default():
    target = os.path.abspath(os.path.join(os.sep, "photos"))
    subdir = os.path.join(target, "2024")
    rule = _block_rule(match_target_dirs=[target])
    assert FileInterceptorRulesManager.rule_matches(rule, "/src/a.png", target, True)
    assert not FileInterceptorRulesManager.rule_matches(rule, "/src/a.png", subdir, True)


def test_target_dir_matches_subdirectory_when_enabled():
    target = os.path.abspath(os.path.join(os.sep, "photos"))
    subdir = os.path.join(target, "2024", "raw")
    rule = _block_rule(match_target_dirs=[target], include_subdirectories=True)
    assert FileInterceptorRulesManager.rule_matches(rule, "/src/a.png", subdir, True)


def test_applies_to_filters_by_transfer_kind():
    move_only = _block_rule(applies_to=InterceptorAppliesTo.MOVE_ONLY)
    assert FileInterceptorRulesManager.rule_matches(move_only, "/s/a.png", "/d", True)
    assert not FileInterceptorRulesManager.rule_matches(move_only, "/s/a.png", "/d", False)

    copy_only = _block_rule(applies_to=InterceptorAppliesTo.COPY_ONLY)
    assert not FileInterceptorRulesManager.rule_matches(copy_only, "/s/a.png", "/d", True)
    assert FileInterceptorRulesManager.rule_matches(copy_only, "/s/a.png", "/d", False)


def test_conditions_combine_with_and():
    rule = _block_rule(
        match_filename_patterns=["draft"],
        match_media_types=[CompareMediaType.IMAGE],
    )
    with patch(
        "files.file_interceptor_rules_manager.get_media_type_for_path",
        return_value=CompareMediaType.IMAGE,
    ):
        assert FileInterceptorRulesManager.rule_matches(rule, "/s/draft.png", "/d", True)
        # Filename condition fails, so the rule does not fire despite the type match.
        assert not FileInterceptorRulesManager.rule_matches(rule, "/s/final.png", "/d", True)


def test_first_matching_rule_wins():
    first = _block_rule(name="first", match_media_types=[CompareMediaType.SVG])
    second = _block_rule(name="second")
    FileInterceptorRulesManager.rules = [first, second]
    with patch(
        "files.file_interceptor_rules_manager.get_media_type_for_path",
        return_value=CompareMediaType.SVG,
    ):
        matched = FileInterceptorRulesManager.find_matching_rule("/s/a.svg", "/d", True)
    assert matched is not None and matched.name == "first"


def test_rule_ordering_determines_precedence():
    svg_rule = _block_rule(name="svg", match_media_types=[CompareMediaType.SVG])
    dir_rule = _block_rule(name="dir", match_target_dirs=["/d"])
    FileInterceptorRulesManager.rules = [dir_rule, svg_rule]
    with patch(
        "files.file_interceptor_rules_manager.get_media_type_for_path",
        return_value=CompareMediaType.SVG,
    ):
        matched = FileInterceptorRulesManager.find_matching_rule(
            "/s/a.svg", os.path.abspath(os.path.join(os.sep, "d")), True
        )
    assert matched is not None and matched.name == "dir"


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

def test_apply_with_no_rules_is_inert():
    result = FileInterceptorRulesManager.apply("/s/a.png", "/d", True, None)
    assert result == InterceptResult()
    assert not result.blocked and result.transformed_source is None


def test_apply_block_warns_and_reports_blocked():
    FileInterceptorRulesManager.rules = [
        _block_rule(name="needs rename", block_message="Rename via RefacDir first")
    ]
    actions = _RecordingActions()
    result = FileInterceptorRulesManager.apply("/s/a.png", "/d", True, actions)
    assert result.blocked
    assert "Rename via RefacDir first" in result.block_message
    assert len(actions.warnings) == 1
    assert "Rename via RefacDir first" in actions.warnings[0]


def test_apply_block_falls_back_to_generic_message():
    FileInterceptorRulesManager.rules = [_block_rule(name="unnamed reason")]
    result = FileInterceptorRulesManager.apply("/s/a.png", "/d", True, None)
    assert result.blocked
    assert "unnamed reason" in result.block_message


def test_apply_transform_substitutes_source_and_flags_delete_on_move():
    FileInterceptorRulesManager.rules = [
        FileInterceptorRule(
            name="jpg",
            behavior=InterceptorBehavior.TRANSFORM,
            transform_op=InterceptorTransformOp.CONVERT_TO_JPG,
            delete_original_after_transform=True,
        )
    ]
    with patch.object(
        FileInterceptorRulesManager, "run_transform", return_value="/s/a.jpg"
    ):
        result = FileInterceptorRulesManager.apply("/s/a.png", "/d", True, None)
    assert result.transformed_source == "/s/a.jpg"
    assert result.delete_original


def test_apply_transform_never_deletes_original_on_copy():
    FileInterceptorRulesManager.rules = [
        FileInterceptorRule(
            name="jpg",
            behavior=InterceptorBehavior.TRANSFORM,
            transform_op=InterceptorTransformOp.CONVERT_TO_JPG,
            delete_original_after_transform=True,
        )
    ]
    with patch.object(
        FileInterceptorRulesManager, "run_transform", return_value="/s/a.jpg"
    ):
        result = FileInterceptorRulesManager.apply("/s/a.png", "/d", False, None)
    assert result.transformed_source == "/s/a.jpg"
    assert not result.delete_original


def test_apply_transform_failure_falls_back_to_untransformed():
    FileInterceptorRulesManager.rules = [
        FileInterceptorRule(
            name="jpg",
            behavior=InterceptorBehavior.TRANSFORM,
            transform_op=InterceptorTransformOp.CONVERT_TO_JPG,
        )
    ]
    with patch.object(
        FileInterceptorRulesManager, "run_transform", side_effect=OSError("boom")
    ):
        result = FileInterceptorRulesManager.apply("/s/a.png", "/d", True, None)
    # Not blocked and no substitute: the original still transfers.
    assert not result.blocked
    assert result.transformed_source is None


def test_apply_transform_noop_result_falls_back_to_untransformed():
    FileInterceptorRulesManager.rules = [
        FileInterceptorRule(
            name="jpg",
            behavior=InterceptorBehavior.TRANSFORM,
            transform_op=InterceptorTransformOp.CONVERT_TO_JPG,
        )
    ]
    with patch.object(FileInterceptorRulesManager, "run_transform", return_value=None):
        result = FileInterceptorRulesManager.apply("/s/a.png", "/d", True, None)
    assert not result.blocked
    assert result.transformed_source is None


# ---------------------------------------------------------------------------
# cleanup_after_move()
# ---------------------------------------------------------------------------
# It deletes via MarkedFiles.delete_file_static(), not app_actions.delete(), so
# these capture that call instead of touching the filesystem. What
# delete_file_static() itself does is covered in test_headless_file_marking.py.

def _capture_deletes(monkeypatch):
    deleted = []
    monkeypatch.setattr(
        MarkedFiles,
        "delete_file_static",
        lambda filepath, app_actions, **kw: deleted.append(filepath) or True,
    )
    return deleted


def test_cleanup_deletes_original_when_flagged(monkeypatch):
    deleted = _capture_deletes(monkeypatch)
    actions = _RecordingActions()
    result = InterceptResult(
        transformed_source="/s/a.jpg", delete_original=True, rule_name="jpg"
    )
    FileInterceptorRulesManager.cleanup_after_move(result, "/s/a.png", actions)
    assert deleted == ["/s/a.png"]


def test_cleanup_is_noop_without_delete_flag(monkeypatch):
    deleted = _capture_deletes(monkeypatch)
    actions = _RecordingActions()
    result = InterceptResult(transformed_source="/s/a.jpg", delete_original=False)
    FileInterceptorRulesManager.cleanup_after_move(result, "/s/a.png", actions)
    assert deleted == []


def test_cleanup_is_noop_without_transform(monkeypatch):
    deleted = _capture_deletes(monkeypatch)
    actions = _RecordingActions()
    FileInterceptorRulesManager.cleanup_after_move(InterceptResult(), "/s/a.png", actions)
    assert deleted == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_rule_round_trips_through_dict():
    rule = FileInterceptorRule(
        name="round trip",
        is_active=False,
        applies_to=InterceptorAppliesTo.MOVE_ONLY,
        match_target_dirs=["/a", "/b"],
        include_subdirectories=True,
        match_filename_patterns=["draft", "wip"],
        filename_case_sensitive=True,
        match_media_types=[CompareMediaType.SVG, CompareMediaType.IMAGE],
        behavior=InterceptorBehavior.TRANSFORM,
        block_message="nope",
        transform_op=InterceptorTransformOp.ENHANCE,
        delete_original_after_transform=False,
    )
    restored = FileInterceptorRule.from_dict(rule.to_dict())
    assert restored.to_dict() == rule.to_dict()
    assert restored.applies_to == InterceptorAppliesTo.MOVE_ONLY
    assert restored.behavior == InterceptorBehavior.TRANSFORM
    assert restored.transform_op == InterceptorTransformOp.ENHANCE
    assert restored.match_media_types == [CompareMediaType.SVG, CompareMediaType.IMAGE]


def test_unknown_enum_values_fall_back_to_safe_defaults():
    rule = FileInterceptorRule.from_dict(
        {
            "name": "legacy",
            "applies_to": "NOT_A_MODE",
            "behavior": "NOT_A_BEHAVIOR",
            "transform_op": "NOT_AN_OP",
            "match_media_types": ["image", "bogus"],
        }
    )
    assert rule.applies_to == InterceptorAppliesTo.MOVE_AND_COPY
    assert rule.behavior == InterceptorBehavior.BLOCK
    assert rule.transform_op is None
    assert rule.match_media_types == [CompareMediaType.IMAGE]


def test_empty_media_type_list_normalizes_to_any():
    rule = FileInterceptorRule(match_media_types=[])
    assert rule.match_media_types is None


def test_transform_behavior_without_op_is_not_a_transform():
    rule = FileInterceptorRule(behavior=InterceptorBehavior.TRANSFORM, transform_op=None)
    assert not rule.is_transform()

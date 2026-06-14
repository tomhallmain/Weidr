"""
Unit tests for ClassifierAction's GENERATE action type.

Covers:
  - related_image_edit_suffix field: defaults, serialization, backward-compat
  - run_action GENERATE branch: gating via should_run_generate_action,
    callback invocation, search_dir resolution from action_modifier / base_directory
"""
from unittest.mock import patch

import pytest

from compare.action_callbacks import ActionCallbacks
from compare.classifier_action import ClassifierAction
from utils.constants import ClassifierActionType

IMAGE = "/dir/source.jpg"
_NOOP = lambda *a, **kw: None


def _action(**kwargs) -> ClassifierAction:
    defaults = dict(
        name="test",
        action=ClassifierActionType.GENERATE,
        use_embedding=False,
        use_image_classifier=False,
        use_prompts=False,
        use_blacklist=False,
        use_prototype=False,
    )
    defaults.update(kwargs)
    return ClassifierAction(**defaults)


def _run(ca, generate_cb=None, base_directory=None):
    received = []
    if generate_cb is None:
        generate_cb = lambda path, suffix: received.append((path, suffix))
    result = ca.run_action(
        IMAGE,
        ActionCallbacks(notify_callback=_NOOP, generate_callback=generate_cb),
        base_directory=base_directory,
    )
    return result, received


# ---------------------------------------------------------------------------
# Field defaults and serialization
# ---------------------------------------------------------------------------

class TestRelatedImageEditSuffixField:
    def test_default_is_empty_string(self):
        assert _action().related_image_edit_suffix == ""

    def test_explicit_value_stored(self):
        assert _action(related_image_edit_suffix="_edit").related_image_edit_suffix == "_edit"

    def test_to_dict_includes_field(self):
        ca = _action(related_image_edit_suffix="_v2")
        assert ca.to_dict()["related_image_edit_suffix"] == "_v2"

    def test_from_dict_round_trip(self):
        ca = _action(related_image_edit_suffix="_edit")
        ca2 = ClassifierAction.from_dict(ca.to_dict())
        assert ca2.related_image_edit_suffix == "_edit"

    def test_from_dict_backward_compat_missing_key(self):
        d = _action().to_dict()
        d.pop("related_image_edit_suffix")
        ca = ClassifierAction.from_dict(d)
        assert ca.related_image_edit_suffix == ""


# ---------------------------------------------------------------------------
# GENERATE gate — no suffix set
# ---------------------------------------------------------------------------

class TestGenerateNoSuffix:
    def test_no_suffix_skips_gate_and_calls_generate(self):
        result, received = _run(_action())
        assert result == ClassifierActionType.GENERATE
        assert received == [(IMAGE, None)]

    def test_no_suffix_passes_none_to_callback(self):
        received = []
        _action().run_action(IMAGE, ActionCallbacks(
            notify_callback=_NOOP,
            generate_callback=lambda path, suffix: received.append(suffix),
        ))
        assert received == [None]


# ---------------------------------------------------------------------------
# GENERATE gate — suffix set, gate consulted
# ---------------------------------------------------------------------------

class TestGenerateWithSuffixGate:
    def test_gate_passes_calls_generate(self):
        ca = _action(related_image_edit_suffix="_edit")
        with patch("files.related_image.should_run_generate_action", return_value=True):
            result, received = _run(ca)
        assert result == ClassifierActionType.GENERATE
        assert received == [(IMAGE, "_edit")]

    def test_gate_fails_returns_none_and_skips_callback(self):
        ca = _action(related_image_edit_suffix="_edit")
        with patch("files.related_image.should_run_generate_action", return_value=False):
            result, received = _run(ca)
        assert result is None
        assert received == []

    def test_suffix_passed_to_generate_callback(self):
        ca = _action(related_image_edit_suffix="_v2")
        received = []
        with patch("files.related_image.should_run_generate_action", return_value=True):
            ca.run_action(IMAGE, ActionCallbacks(
                notify_callback=_NOOP,
                generate_callback=lambda path, suffix: received.append(suffix),
            ))
        assert received == ["_v2"]


# ---------------------------------------------------------------------------
# Search directory resolution
# ---------------------------------------------------------------------------

class TestGenerateSearchDirResolution:
    def _capture_search_dir(self, ca, base_directory=None):
        """Run run_action and return the search_dir that was passed to the gate."""
        captured = []

        def fake_gate(image_path, edit_suffix, search_dir, count_threshold=1):
            captured.append(search_dir)
            return False

        with patch("files.related_image.should_run_generate_action", side_effect=fake_gate):
            ca.run_action(
                IMAGE,
                ActionCallbacks(notify_callback=_NOOP),
                base_directory=base_directory,
            )

        return captured[0] if captured else None

    def test_action_modifier_used_as_primary_search_dir(self):
        ca = _action(related_image_edit_suffix="_edit", action_modifier="/custom/dir")
        assert self._capture_search_dir(ca, base_directory="/base") == "/custom/dir"

    def test_falls_back_to_base_directory_when_modifier_empty(self):
        ca = _action(related_image_edit_suffix="_edit", action_modifier="")
        assert self._capture_search_dir(ca, base_directory="/base") == "/base"

    def test_falls_back_to_image_directory_when_both_empty(self):
        ca = _action(related_image_edit_suffix="_edit", action_modifier="")
        assert self._capture_search_dir(ca, base_directory=None) == "/dir"

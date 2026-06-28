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
from files.related_image import clear_base_stem_dir_cache
from utils.config import config
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
        generate_cb = lambda path, suffix, target_dir=None: received.append((path, suffix))
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
            generate_callback=lambda path, suffix, target_dir=None: received.append(suffix),
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
                generate_callback=lambda path, suffix, target_dir=None: received.append(suffix),
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


# ---------------------------------------------------------------------------
# use_base_stem_match field and _check_base_stem_match
# ---------------------------------------------------------------------------

class TestBaseStemMatchField:
    def test_default_false(self):
        assert _action().use_base_stem_match is False

    def test_default_require_match_true(self):
        assert _action().base_stem_match_require_match is True

    def test_to_dict_includes_both_fields(self):
        ca = _action(use_base_stem_match=True, base_stem_match_require_match=False)
        d = ca.to_dict()
        assert d["use_base_stem_match"] is True
        assert d["base_stem_match_require_match"] is False

    def test_from_dict_round_trip(self):
        ca = _action(use_base_stem_match=True, base_stem_match_require_match=False)
        ca2 = ClassifierAction.from_dict(ca.to_dict())
        assert ca2.use_base_stem_match is True
        assert ca2.base_stem_match_require_match is False

    def test_from_dict_backward_compat_missing_keys(self):
        d = _action().to_dict()
        d.pop("use_base_stem_match", None)
        d.pop("base_stem_match_require_match", None)
        ca = ClassifierAction.from_dict(d)
        assert ca.use_base_stem_match is False
        assert ca.base_stem_match_require_match is True


class TestCheckBaseStemMatch:
    # extract_filename_base_stem and find_files_by_base_stem are lazy-imported
    # inside _check_base_stem_match, so patch the source module they come from.
    # config is the module-level singleton; patch its attribute directly.

    def setup_method(self):
        clear_base_stem_dir_cache()

    def teardown_method(self):
        clear_base_stem_dir_cache()

    def _ca(self, require_match=True):
        return _action(use_base_stem_match=True, base_stem_match_require_match=require_match)

    def test_returns_true_when_file_found_require_true(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_action.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_action.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_x.jpg"])
        assert self._ca()._check_base_stem_match(IMAGE) is True

    def test_returns_false_when_file_not_found_require_true(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_action.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_action.find_files_by_base_stem",
                            lambda dirs, stem, **kw: [])
        assert self._ca()._check_base_stem_match(IMAGE) is False

    def test_inverted_returns_true_when_not_found(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_action.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_action.find_files_by_base_stem",
                            lambda dirs, stem, **kw: [])
        assert self._ca(require_match=False)._check_base_stem_match(IMAGE) is True

    def test_no_base_stem_returns_false(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_action.extract_filename_base_stem", lambda p: None)
        assert self._ca()._check_base_stem_match(IMAGE) is False

    def test_empty_dirs_returns_false(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_action.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", [])
        assert self._ca()._check_base_stem_match(IMAGE) is False

    def test_evaluate_image_path_dispatches_base_stem(self, monkeypatch):
        monkeypatch.setattr("compare.classifier_action.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])
        monkeypatch.setattr("compare.classifier_action.find_files_by_base_stem",
                            lambda dirs, stem, **kw: ["/dir/stem_x.jpg"])
        ca = _action(use_base_stem_match=True, use_embedding=False)
        matched, _ = ca._evaluate_image_path_match(IMAGE)
        assert matched is True

    def test_use_cache_true_passed_to_find(self, monkeypatch):
        captured = []
        monkeypatch.setattr("compare.classifier_action.extract_filename_base_stem", lambda p: "stem")
        monkeypatch.setattr(config, "directories_to_search_for_related_images", ["/dir"])

        def fake_find(dirs, stem, **kw):
            captured.append(kw)
            return []

        monkeypatch.setattr("compare.classifier_action.find_files_by_base_stem", fake_find)
        self._ca()._check_base_stem_match(IMAGE)
        assert captured[0].get("use_cache") is True

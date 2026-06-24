"""Tests for Config.apply_and_persist() and Config.persist() atomic swap."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import utils.config as _cfg_module
from utils.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config() -> Config:
    """Return the per-test config instance (patched by isolated_singletons)."""
    return _cfg_module.config


# ---------------------------------------------------------------------------
# DIALOG_FIELDS registry
# ---------------------------------------------------------------------------

class TestDialogFieldsRegistry:
    def test_all_dialog_fields_are_real_attributes(self):
        """Every key in DIALOG_FIELDS must exist on a fresh Config instance."""
        c = _config()
        for key in Config.DIALOG_FIELDS:
            assert hasattr(c, key), f"DIALOG_FIELDS key {key!r} has no matching attribute"

    def test_all_dialog_fields_have_valid_type_sentinel(self):
        """Every value in DIALOG_FIELDS must be bool, int, float, str, or None."""
        allowed = {bool, int, float, str, None}
        for key, expected in Config.DIALOG_FIELDS.items():
            assert expected in allowed, (
                f"DIALOG_FIELDS[{key!r}] = {expected!r} is not an allowed type sentinel"
            )


# ---------------------------------------------------------------------------
# apply_and_persist — field type handling
# ---------------------------------------------------------------------------

class TestApplyAndPersistFieldTypes:
    def test_bool_field_set_true(self):
        c = _config()
        c.show_toasts = False
        errors = c.apply_and_persist({"show_toasts": True})
        assert errors == []
        assert c.show_toasts is True
        assert c.dict["show_toasts"] is True

    def test_bool_field_set_false(self):
        c = _config()
        c.show_toasts = True
        errors = c.apply_and_persist({"show_toasts": False})
        assert errors == []
        assert c.show_toasts is False

    def test_int_field_parsed_from_str(self):
        c = _config()
        errors = c.apply_and_persist({"font_size": "12"})
        assert errors == []
        assert c.font_size == 12
        assert c.dict["font_size"] == 12

    def test_int_field_invalid_str_returns_error(self):
        original = _config().font_size
        errors = _config().apply_and_persist({"font_size": "not_a_number"})
        assert len(errors) == 1
        assert "font_size" in errors[0]
        assert _config().font_size == original  # unchanged

    def test_float_field_parsed_from_str(self):
        c = _config()
        errors = c.apply_and_persist({"threshold_potential_duplicate_embedding": "0.95"})
        assert errors == []
        assert c.threshold_potential_duplicate_embedding == pytest.approx(0.95)
        assert c.dict["threshold_potential_duplicate_embedding"] == pytest.approx(0.95)

    def test_float_field_invalid_str_returns_error(self):
        c = _config()
        original = c.threshold_potential_duplicate_embedding
        errors = c.apply_and_persist({"threshold_potential_duplicate_embedding": "abc"})
        assert len(errors) == 1
        assert c.threshold_potential_duplicate_embedding == original  # unchanged

    def test_str_field_stored_stripped(self):
        c = _config()
        errors = c.apply_and_persist({"default_main_window_size": "  1920x1080  "})
        assert errors == []
        assert c.default_main_window_size == "1920x1080"
        assert c.dict["default_main_window_size"] == "1920x1080"

    def test_nullable_str_empty_becomes_none(self):
        c = _config()
        c.trash_folder = "/some/path"
        errors = c.apply_and_persist({"trash_folder": ""})
        assert errors == []
        assert c.trash_folder is None
        assert c.dict["trash_folder"] is None

    def test_nullable_str_whitespace_only_becomes_none(self):
        c = _config()
        errors = c.apply_and_persist({"trash_folder": "   "})
        assert errors == []
        assert c.trash_folder is None

    def test_nullable_str_nonempty_stored_as_str(self):
        c = _config()
        errors = c.apply_and_persist({"trash_folder": "/tmp/trash"})
        assert errors == []
        assert c.trash_folder == "/tmp/trash"
        assert c.dict["trash_folder"] == "/tmp/trash"


# ---------------------------------------------------------------------------
# apply_and_persist — error accumulation and atomicity
# ---------------------------------------------------------------------------

class TestApplyAndPersistErrorHandling:
    def test_multiple_errors_all_collected(self):
        errors = _config().apply_and_persist({
            "font_size": "bad",
            "toasts_persist_seconds": "also_bad",
        })
        assert len(errors) == 2
        keys_in_errors = " ".join(errors)
        assert "font_size" in keys_in_errors
        assert "toasts_persist_seconds" in keys_in_errors

    def test_on_validation_error_attr_not_modified(self):
        c = _config()
        original_font = c.font_size
        original_toasts = c.show_toasts
        c.apply_and_persist({"font_size": "bad", "show_toasts": False})
        # show_toasts is valid but nothing should be applied when any error exists
        assert c.font_size == original_font
        assert c.show_toasts == original_toasts

    def test_on_validation_error_dict_not_modified(self):
        c = _config()
        original_dict_font = c.dict.get("font_size")
        c.apply_and_persist({"font_size": "bad"})
        assert c.dict.get("font_size") == original_dict_font

    def test_on_validation_error_persist_not_called(self):
        c = _config()
        with patch.object(c, "persist") as mock_persist:
            c.apply_and_persist({"font_size": "bad"})
        mock_persist.assert_not_called()

    def test_unknown_key_is_ignored_not_applied(self):
        c = _config()
        errors = c.apply_and_persist({"__nonexistent_key__": "value"})
        assert errors == []
        assert not hasattr(c, "__nonexistent_key__")

    def test_unknown_key_does_not_pollute_dict(self):
        c = _config()
        c.apply_and_persist({"__nonexistent_key__": "value"})
        assert "__nonexistent_key__" not in c.dict


# ---------------------------------------------------------------------------
# apply_and_persist — successful persist interaction
# ---------------------------------------------------------------------------

class TestApplyAndPersistSuccess:
    def test_successful_apply_persists_to_disk(self, tmp_path):
        """Changes appear in the config file after a successful apply_and_persist."""
        c = _config()
        c.apply_and_persist({"font_size": "14"})
        data = json.loads(Path(c.config_path).read_text(encoding="utf-8"))
        assert data["font_size"] == 14

    def test_successful_apply_updates_config_dict(self):
        c = _config()
        c.apply_and_persist({"font_size": "14"})
        assert c.dict["font_size"] == 14

    def test_successful_apply_updates_attr(self):
        c = _config()
        c.apply_and_persist({"font_size": "14"})
        assert c.font_size == 14

    def test_multiple_fields_applied_together(self):
        c = _config()
        errors = c.apply_and_persist({
            "show_toasts": False,
            "font_size": "10",
            "default_main_window_size": "800x600",
        })
        assert errors == []
        assert c.show_toasts is False
        assert c.font_size == 10
        assert c.default_main_window_size == "800x600"

    def test_returns_empty_list_on_success(self):
        errors = _config().apply_and_persist({"show_toasts": True})
        assert errors == []

    def test_persist_exception_propagates(self):
        c = _config()
        with patch.object(c, "persist", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                c.apply_and_persist({"show_toasts": False})


# ---------------------------------------------------------------------------
# persist() — atomic swap
# ---------------------------------------------------------------------------

class TestPersistAtomicSwap:
    def test_writes_valid_json(self):
        c = _config()
        c.persist()
        data = json.loads(Path(c.config_path).read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_no_tmp_file_left_after_success(self):
        c = _config()
        c.persist()
        assert not Path(c.config_path + ".tmp").exists()

    def test_original_preserved_when_write_fails(self):
        c = _config()
        original_content = Path(c.config_path).read_text(encoding="utf-8")
        with patch("json.dump", side_effect=IOError("disk full")):
            with pytest.raises(IOError):
                c.persist()
        assert Path(c.config_path).read_text(encoding="utf-8") == original_content

    def test_no_tmp_file_left_after_write_failure(self):
        c = _config()
        with patch("json.dump", side_effect=IOError("disk full")):
            with pytest.raises(IOError):
                c.persist()
        assert not Path(c.config_path + ".tmp").exists()

    def test_updates_self_dict_after_successful_persist(self):
        c = _config()
        c.show_toasts = False
        c.dict["show_toasts"] = False
        c.persist()
        assert c.dict["show_toasts"] is False

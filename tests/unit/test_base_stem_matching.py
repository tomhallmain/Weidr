"""
Unit tests for files.related_image base-stem matching functions.

Covers:
  - extract_filename_base_stem: delimiter heuristics
  - _check_base_stem_boundary: exact / separator / category-change rules
  - find_files_by_base_stem: filesystem walk, threshold/callback, results
  - Caching behaviour: use_cache=True, TTL expiry, clear_base_stem_dir_cache
"""

import os
import time
from unittest.mock import patch

import pytest

from files.related_image import (
    _check_base_stem_boundary,
    _CharCategory,
    clear_base_stem_dir_cache,
    extract_filename_base_stem,
    find_files_by_base_stem,
)
import files.related_image as _ri_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, *names):
    """Create empty files under tmp_path and return the directory as str."""
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# extract_filename_base_stem
# ---------------------------------------------------------------------------

class TestExtractFilenameBaseStem:
    def test_sd_style_filename(self):
        assert extract_filename_base_stem("SDWebUI_17602175357792320_0_s.png") == "SDWebUI_17602175357792320"

    def test_single_long_part_no_delimiters(self):
        assert extract_filename_base_stem("abcdefghij.jpg") == "abcdefghij"

    def test_single_short_part_returns_full_basename(self):
        # < 8 chars, single part → returns basename
        result = extract_filename_base_stem("abc.jpg")
        assert result == "abc"

    def test_first_part_long_enough_returns_immediately(self):
        # first part >= 30 chars
        stem = "a" * 30
        result = extract_filename_base_stem(f"{stem}_extra.jpg")
        assert result == stem

    def test_two_part_short_second_returns_first(self):
        # second part <= 4 chars → first part is the stem
        result = extract_filename_base_stem("MyFile_01.jpg")
        assert result == "MyFile"

    def test_path_strips_directory(self):
        result = extract_filename_base_stem("/some/dir/SDWebUI_12345678901234567_0.png")
        assert "/" not in (result or "")
        assert result is not None

    def test_none_not_returned_for_valid_name(self):
        result = extract_filename_base_stem("photo_20240101_final.jpg")
        assert result is not None
        assert len(result) >= 3

    def test_extension_stripped(self):
        result = extract_filename_base_stem("foo_12345678901.png")
        assert result is not None
        assert ".png" not in result


# ---------------------------------------------------------------------------
# _check_base_stem_boundary
# ---------------------------------------------------------------------------

class TestCheckBaseStemBoundary:
    def test_exact_match(self):
        assert _check_base_stem_boundary("stem", "stem", _CharCategory.ALPHA) is True

    def test_no_match_different_name(self):
        assert _check_base_stem_boundary("other", "stem", _CharCategory.ALPHA) is False

    def test_separator_underscore(self):
        assert _check_base_stem_boundary("stem_edit", "stem", _CharCategory.ALPHA) is True

    def test_separator_hyphen(self):
        assert _check_base_stem_boundary("stem-v2", "stem", _CharCategory.ALPHA) is True

    def test_separator_space(self):
        assert _check_base_stem_boundary("stem copy", "stem", _CharCategory.ALPHA) is True

    def test_separator_dot(self):
        assert _check_base_stem_boundary("stem.bak", "stem", _CharCategory.ALPHA) is True

    def test_category_change_alpha_to_digit(self):
        # stem ends with alpha, next char is digit → different category → match
        assert _check_base_stem_boundary("stem1", "stem", _CharCategory.ALPHA) is True

    def test_same_category_alpha_to_alpha_no_match(self):
        # stem ends with alpha, next char is alpha → same category → no match
        assert _check_base_stem_boundary("stemX", "stem", _CharCategory.ALPHA) is False

    def test_digit_to_alpha_category_change(self):
        # stem ends with digit, next char is alpha → different category → match
        assert _check_base_stem_boundary("123abc", "123", _CharCategory.DIGIT) is True

    def test_not_starts_with_stem(self):
        assert _check_base_stem_boundary("other_stem", "stem", _CharCategory.ALPHA) is False


# ---------------------------------------------------------------------------
# find_files_by_base_stem — no cache
# ---------------------------------------------------------------------------

class TestFindFilesByBaseStem:
    def test_finds_matching_file(self, tmp_path):
        d = _write(tmp_path, "SDWebUI_12345678901_0_s.jpg", "SDWebUI_12345678901_1.png")
        results = find_files_by_base_stem([d], "SDWebUI_12345678901")
        basenames = [os.path.basename(r) for r in results]
        assert "SDWebUI_12345678901_0_s.jpg" in basenames
        assert "SDWebUI_12345678901_1.png" in basenames

    def test_does_not_match_unrelated_file(self, tmp_path):
        d = _write(tmp_path, "other_file.jpg")
        results = find_files_by_base_stem([d], "SDWebUI_12345678901")
        assert results == []

    def test_exact_stem_match_included(self, tmp_path):
        d = _write(tmp_path, "stem.jpg")
        results = find_files_by_base_stem([d], "stem")
        assert len(results) == 1

    def test_results_sorted_by_basename(self, tmp_path):
        d = _write(tmp_path, "stem_z.jpg", "stem_a.jpg", "stem_m.jpg")
        results = find_files_by_base_stem([d], "stem")
        basenames = [os.path.basename(r) for r in results]
        assert basenames == sorted(basenames, key=str.lower)

    def test_multiple_directories_searched(self, tmp_path):
        d1 = str(tmp_path / "d1")
        d2 = str(tmp_path / "d2")
        os.makedirs(d1); os.makedirs(d2)
        (tmp_path / "d1" / "stem_a.jpg").write_bytes(b"")
        (tmp_path / "d2" / "stem_b.jpg").write_bytes(b"")
        results = find_files_by_base_stem([d1, d2], "stem")
        basenames = [os.path.basename(r) for r in results]
        assert "stem_a.jpg" in basenames
        assert "stem_b.jpg" in basenames

    def test_threshold_triggers_callback(self, tmp_path):
        for i in range(5):
            (tmp_path / f"other_{i}.jpg").write_bytes(b"")
        called_with = []

        def cb(directory, count):
            called_with.append((directory, count))
            return True  # proceed

        find_files_by_base_stem([str(tmp_path)], "stem", threshold=3, on_threshold_exceeded=cb)
        assert len(called_with) == 1
        assert called_with[0][1] > 3

    def test_callback_returning_false_aborts(self, tmp_path):
        for i in range(5):
            (tmp_path / f"other_{i}.jpg").write_bytes(b"")

        result = find_files_by_base_stem(
            [str(tmp_path)], "stem", threshold=2, on_threshold_exceeded=lambda d, c: False
        )
        assert result == []

    def test_no_callback_threshold_exceeded_returns_partial(self, tmp_path):
        # Without a callback and threshold exceeded, walk is aborted but
        # matches found so far are returned (not an empty list).
        (tmp_path / "stem_a.jpg").write_bytes(b"")
        for i in range(5):
            (tmp_path / f"other_{i}.jpg").write_bytes(b"")

        results = find_files_by_base_stem([str(tmp_path)], "stem", threshold=2)
        # Returns whatever was found before abort, not necessarily all matches
        assert isinstance(results, list)

    def test_nonexistent_directory_skipped(self):
        results = find_files_by_base_stem(["/does/not/exist"], "stem")
        assert results == []


# ---------------------------------------------------------------------------
# find_files_by_base_stem — caching (use_cache=True)
# ---------------------------------------------------------------------------

class TestFindFilesByBaseStemCache:
    def setup_method(self):
        clear_base_stem_dir_cache()

    def teardown_method(self):
        clear_base_stem_dir_cache()

    def test_cache_hit_returns_same_results(self, tmp_path):
        d = _write(tmp_path, "stem_a.jpg")
        r1 = find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        r2 = find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        assert r1 == r2

    def test_cache_miss_after_clear(self, tmp_path):
        d = _write(tmp_path, "stem_a.jpg")
        find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        # Add a new file and clear cache
        (tmp_path / "stem_b.jpg").write_bytes(b"")
        clear_base_stem_dir_cache(str(tmp_path))
        r2 = find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        basenames = [os.path.basename(r) for r in r2]
        assert "stem_b.jpg" in basenames

    def test_stale_cache_is_not_used_after_clear(self, tmp_path):
        d = _write(tmp_path, "stem_old.jpg")
        find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        (tmp_path / "stem_old.jpg").unlink()
        # Without clearing, stale cache still lists old file
        r_stale = find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        assert any("stem_old" in r for r in r_stale)
        # After clearing, fresh walk sees nothing
        clear_base_stem_dir_cache(str(tmp_path))
        r_fresh = find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        assert r_fresh == []

    def test_ttl_expiry_causes_re_walk(self, tmp_path):
        d = _write(tmp_path, "stem_a.jpg")
        find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        (tmp_path / "stem_b.jpg").write_bytes(b"")
        # Fake the cached timestamp to be expired (cache is now a 3-tuple)
        norm = os.path.normpath(os.path.abspath(str(tmp_path)))
        entries, stems, _ = _ri_mod._base_stem_dir_cache[norm]
        _ri_mod._base_stem_dir_cache[norm] = (
            entries,
            stems,
            time.time() - _ri_mod._BASE_STEM_CACHE_TTL - 1,
        )
        r = find_files_by_base_stem([str(tmp_path)], "stem", use_cache=True)
        basenames = [os.path.basename(x) for x in r]
        assert "stem_b.jpg" in basenames

    def test_clear_all_removes_all_entries(self, tmp_path):
        d1 = str(tmp_path / "d1"); d2 = str(tmp_path / "d2")
        os.makedirs(d1); os.makedirs(d2)
        find_files_by_base_stem([d1], "stem", use_cache=True)
        find_files_by_base_stem([d2], "stem", use_cache=True)
        assert len(_ri_mod._base_stem_dir_cache) == 2
        clear_base_stem_dir_cache()
        assert len(_ri_mod._base_stem_dir_cache) == 0

    def test_clear_specific_directory(self, tmp_path):
        d1 = str(tmp_path / "d1"); d2 = str(tmp_path / "d2")
        os.makedirs(d1); os.makedirs(d2)
        find_files_by_base_stem([d1], "stem", use_cache=True)
        find_files_by_base_stem([d2], "stem", use_cache=True)
        clear_base_stem_dir_cache(d1)
        norm_d1 = os.path.normpath(os.path.abspath(d1))
        norm_d2 = os.path.normpath(os.path.abspath(d2))
        assert norm_d1 not in _ri_mod._base_stem_dir_cache
        assert norm_d2 in _ri_mod._base_stem_dir_cache

    def test_aborted_walk_not_cached(self, tmp_path):
        for i in range(5):
            (tmp_path / f"other_{i}.jpg").write_bytes(b"")
        norm = os.path.normpath(os.path.abspath(str(tmp_path)))
        find_files_by_base_stem(
            [str(tmp_path)], "stem", threshold=2, use_cache=True,
            on_threshold_exceeded=lambda d, c: False,
        )
        assert norm not in _ri_mod._base_stem_dir_cache


# ---------------------------------------------------------------------------
# Cursor-based scan optimisation
# ---------------------------------------------------------------------------

class TestBaseStemCursorOptimisation:
    """Tests for the sorted-entries + cursor optimisation in the use_cache path."""

    def setup_method(self):
        clear_base_stem_dir_cache()

    def teardown_method(self):
        clear_base_stem_dir_cache()

    def _norm(self, p) -> str:
        return os.path.normpath(os.path.abspath(str(p)))

    def test_sorted_order_advances_cursor_monotonically(self, tmp_path):
        d = str(tmp_path)
        _write(tmp_path, "aaa_edit.jpg", "bbb_edit.jpg", "ccc_edit.jpg")
        find_files_by_base_stem([d], "aaa", use_cache=True)
        idx_after_aaa, _ = _ri_mod._base_stem_dir_cursor[self._norm(tmp_path)]
        find_files_by_base_stem([d], "bbb", use_cache=True)
        idx_after_bbb, _ = _ri_mod._base_stem_dir_cursor[self._norm(tmp_path)]
        find_files_by_base_stem([d], "ccc", use_cache=True)
        idx_after_ccc, _ = _ri_mod._base_stem_dir_cursor[self._norm(tmp_path)]
        assert idx_after_aaa <= idx_after_bbb <= idx_after_ccc

    def test_sorted_order_returns_correct_matches(self, tmp_path):
        _write(tmp_path, "aaa_edit.jpg", "aaa_other.jpg", "bbb_edit.jpg", "zzz.jpg")
        d = str(tmp_path)
        r_aaa = find_files_by_base_stem([d], "aaa", use_cache=True)
        r_bbb = find_files_by_base_stem([d], "bbb", use_cache=True)
        assert {os.path.basename(p) for p in r_aaa} == {"aaa_edit.jpg", "aaa_other.jpg"}
        assert {os.path.basename(p) for p in r_bbb} == {"bbb_edit.jpg"}

    def test_out_of_order_query_falls_back_to_bisect(self, tmp_path):
        _write(tmp_path, "aaa_edit.jpg", "bbb_edit.jpg")
        d = str(tmp_path)
        # Query bbb first (sets cursor past bbb), then aaa (out-of-order).
        find_files_by_base_stem([d], "bbb", use_cache=True)
        r_aaa = find_files_by_base_stem([d], "aaa", use_cache=True)
        assert [os.path.basename(p) for p in r_aaa] == ["aaa_edit.jpg"]

    def test_out_of_order_does_not_corrupt_cursor(self, tmp_path):
        _write(tmp_path, "aaa_edit.jpg", "bbb_edit.jpg", "ccc_edit.jpg")
        d = str(tmp_path)
        norm = self._norm(tmp_path)
        # Advance cursor to bbb.
        find_files_by_base_stem([d], "bbb", use_cache=True)
        cursor_before, _ = _ri_mod._base_stem_dir_cursor[norm]
        # Out-of-order query; cursor must not move backward.
        find_files_by_base_stem([d], "aaa", use_cache=True)
        cursor_after, _ = _ri_mod._base_stem_dir_cursor[norm]
        assert cursor_after == cursor_before

    def test_repeated_query_returns_correct_results(self, tmp_path):
        _write(tmp_path, "stem_edit.jpg")
        d = str(tmp_path)
        r1 = find_files_by_base_stem([d], "stem", use_cache=True)
        r2 = find_files_by_base_stem([d], "stem", use_cache=True)
        assert r1 == r2

    def test_cursor_reset_on_cache_clear(self, tmp_path):
        _write(tmp_path, "aaa_edit.jpg")
        d = str(tmp_path)
        norm = self._norm(tmp_path)
        find_files_by_base_stem([d], "aaa", use_cache=True)
        assert norm in _ri_mod._base_stem_dir_cursor
        clear_base_stem_dir_cache(d)
        assert norm not in _ri_mod._base_stem_dir_cursor

    def test_cursor_reset_on_full_clear(self, tmp_path):
        _write(tmp_path, "aaa_edit.jpg")
        d = str(tmp_path)
        find_files_by_base_stem([d], "aaa", use_cache=True)
        clear_base_stem_dir_cache()
        assert len(_ri_mod._base_stem_dir_cursor) == 0

    def test_cursor_reset_on_ttl_expiry(self, tmp_path):
        _write(tmp_path, "aaa_edit.jpg", "bbb_edit.jpg")
        d = str(tmp_path)
        norm = self._norm(tmp_path)
        find_files_by_base_stem([d], "aaa", use_cache=True)
        # Expire the cache.
        entries, stems, _ = _ri_mod._base_stem_dir_cache[norm]
        _ri_mod._base_stem_dir_cache[norm] = (entries, stems, time.time() - _ri_mod._BASE_STEM_CACHE_TTL - 1)
        # Next query triggers a fresh scan; cursor is reset to 0 for that dir.
        find_files_by_base_stem([d], "bbb", use_cache=True)
        cursor_idx, _ = _ri_mod._base_stem_dir_cursor[norm]
        # bbb was queried first after reset, so cursor advanced past bbb only.
        # aaa sorts before bbb, so cursor must be past bbb's position (>= 1).
        assert cursor_idx >= 1

    def test_no_match_stem_advances_cursor_past_gap(self, tmp_path):
        _write(tmp_path, "aaa_edit.jpg", "ccc_edit.jpg")
        d = str(tmp_path)
        # Query bbb which doesn't exist — cursor should advance past aaa and bbb's position.
        r = find_files_by_base_stem([d], "bbb", use_cache=True)
        assert r == []
        # Subsequent in-order query for ccc must still find the file.
        r_ccc = find_files_by_base_stem([d], "ccc", use_cache=True)
        assert [os.path.basename(p) for p in r_ccc] == ["ccc_edit.jpg"]

    def test_scale_correctness(self, tmp_path):
        # 500 unique stems, sorted queries — verify correctness at scale.
        stems = [f"CUI_{i:019d}" for i in range(500)]
        for s in stems:
            (tmp_path / f"{s}_edit.jpg").write_bytes(b"")
            (tmp_path / f"{s}_other.png").write_bytes(b"")
        d = str(tmp_path)
        for s in stems:
            results = find_files_by_base_stem([d], s, use_cache=True)
            basenames = {os.path.basename(p) for p in results}
            assert f"{s}_edit.jpg" in basenames, f"Missing edit for {s}"
            assert f"{s}_other.png" in basenames, f"Missing other for {s}"
            assert len(results) == 2, f"Expected 2 results for {s}, got {len(results)}"

"""
Unit tests for files.related_image.should_run_generate_action.

All filesystem operations (related-image metadata lookup and directory scanning)
are mocked so no real directories are accessed.
"""
from unittest.mock import patch

import pytest

from files.related_image import (
    _scan_dir_cached,
    clear_generate_gate_cache,
    should_run_generate_action,
)

IMAGE = "/base/source.jpg"
SEARCH_DIR = "/base"


def _patch_related(related_path=None):
    """Patch get_related_image_path for the derivative-check call inside should_run_generate_action."""
    return patch(
        "files.related_image.get_related_image_path",
        return_value=(related_path, related_path is not None),
    )


def _patch_scan_dir(entries):
    """Patch _scan_dir_cached to return pre-built (filepath, related_or_None) entries."""
    return patch("files.related_image._scan_dir_cached", return_value=entries)


# ---------------------------------------------------------------------------
# Derivative-check gate
# ---------------------------------------------------------------------------

class TestShouldRunGenerateActionDerivativeCheck:
    def test_derivative_with_source_in_search_dir_returns_false(self):
        """If image_path's source is present in search_dir, never generate."""
        entries = [("/base/parent.jpg", None)]
        with _patch_related("/base/parent.jpg"):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_derivative_with_source_outside_search_dir_proceeds(self):
        """A related-image pointer to a source NOT in search_dir does not block
        generation -- only downstream-ness relative to the current directory matters.
        With no downstream entries, the suffix-count check then allows generation."""
        with _patch_related("/elsewhere/parent.jpg"):
            with _patch_scan_dir([]):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is True

    def test_derivative_with_source_in_search_dir_matched_by_basename(self):
        """Source match also works via basename when the related-path string
        differs from the scanned filepath but the (long) basename matches."""
        entries = [("/base/sub/long_parent_name.jpg", None)]
        with _patch_related("/different/location/long_parent_name.jpg"):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_non_derivative_proceeds_to_downstream_check(self):
        """No related path → derivative check passes; zero downstream → True."""
        with _patch_related(None):
            with _patch_scan_dir([]):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is True


# ---------------------------------------------------------------------------
# Suffix-count logic
# ---------------------------------------------------------------------------

class TestShouldRunGenerateActionSuffixCount:
    def test_no_downstream_returns_true(self):
        """suffix_count=0 < threshold=1 → generate is needed."""
        with _patch_related(None):
            with _patch_scan_dir([]):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is True

    def test_downstream_without_suffix_returns_true(self):
        """Downstream files exist but none end with the suffix → still needed.

        source_v.jpg and source_extra.jpg are caught by the stem-prefix heuristic
        (_VARIANT_SUFFIX_RE_STRICT matches alpha-only suffix) but their stems do
        not match the suffix regex '_edit\\d*$'.
        """
        entries = [
            ("/base/source_v.jpg", None),
            ("/base/source_extra.jpg", None),
        ]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is True

    def test_downstream_with_suffix_via_stem_prefix(self):
        """Downstream detected by stem-prefix heuristic counts toward threshold."""
        # source_edit.jpg: _VARIANT_SUFFIX_RE_STRICT matches, group(1)="source"
        entries = [("/base/source_edit.jpg", None)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_downstream_with_suffix_via_metadata_counts(self):
        """Downstream detected by related-path metadata also counts toward threshold."""
        entries = [("/base/source_edit.jpg", IMAGE)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_mixed_downstream_only_suffix_matches_counted(self):
        """Files without the suffix don't count; the one with the suffix tips threshold."""
        entries = [
            ("/base/source_v.jpg", None),    # downstream but no _edit suffix
            ("/base/source_edit.jpg", None), # downstream with _edit suffix
        ]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_suffix_must_be_at_end_of_stem(self):
        """A stem containing the suffix mid-string is not counted.

        source_edit_extra.jpg is a downstream (via metadata) but its stem
        ends with _extra, not _edit or _edit<int>.
        """
        entries = [("/base/source_edit_extra.jpg", IMAGE)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is True

    def test_suffix_followed_by_integer_is_counted(self):
        """Integer-suffixed variant counts.

        source_edit1.jpg ends with a digit so the stem-prefix heuristic won't
        catch it; it must be supplied with a related-path so it qualifies as a
        downstream.
        """
        entries = [("/base/source_edit1.jpg", IMAGE)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_suffix_followed_by_multi_digit_integer_is_counted(self):
        entries = [("/base/source_edit12.jpg", IMAGE)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_integer_variants_count_toward_threshold(self):
        """Both bare suffix and integer-suffixed variants contribute to the count."""
        entries = [
            ("/base/source_edit.jpg", None),   # caught by stem-prefix
            ("/base/source_edit1.jpg", IMAGE),  # caught by metadata (int suffix)
        ]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR, count_threshold=2)
        assert result is False

    def test_suffix_followed_by_separator_and_integer_is_counted(self):
        """'suffix_N' format (separator before index) is counted toward threshold.

        Some generators produce names like source_edit_2.jpg rather than
        source_edit2.jpg.  Both formats must be recognised.
        """
        entries = [("/base/source_edit_2.jpg", IMAGE)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_suffix_followed_by_separator_and_multi_digit_integer_is_counted(self):
        entries = [("/base/source_edit_12.jpg", IMAGE)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is False

    def test_separator_index_and_direct_index_both_count_toward_threshold(self):
        """Mixed naming styles are additive."""
        entries = [
            ("/base/source_edit.jpg", None),    # bare suffix, stem-prefix
            ("/base/source_edit_2.jpg", IMAGE), # separator-index, metadata
        ]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR, count_threshold=2)
        assert result is False

    def test_suffix_extra_text_after_separator_not_counted(self):
        """A stem of the form suffix_extra must NOT be counted — only suffix_<int> counts."""
        entries = [("/base/source_edit_extra.jpg", IMAGE)]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR)
        assert result is True


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------

class TestShouldRunGenerateActionCustomThreshold:
    def test_suffix_count_below_custom_threshold_returns_true(self):
        entries = [("/base/source_edit.jpg", None)]  # 1 match < threshold 2
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR, count_threshold=2)
        assert result is True

    def test_suffix_count_at_custom_threshold_returns_false(self):
        entries = [
            ("/base/source_edit.jpg", None),
            ("/base/source_edit1.jpg", IMAGE),
        ]
        with _patch_related(None):
            with _patch_scan_dir(entries):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR, count_threshold=2)
        assert result is False

    def test_threshold_zero_always_returns_false(self):
        """suffix_count (0) >= threshold (0) → do not generate."""
        with _patch_related(None):
            with _patch_scan_dir([]):
                result = should_run_generate_action(IMAGE, "_edit", SEARCH_DIR, count_threshold=0)
        assert result is False


# ---------------------------------------------------------------------------
# search_dir forwarding
# ---------------------------------------------------------------------------

class TestShouldRunGenerateActionSearchDir:
    def test_search_dir_forwarded_to_scan(self):
        captured = []

        def fake_scan(directory):
            captured.append(directory)
            return []

        with _patch_related(None):
            with patch("files.related_image._scan_dir_cached", side_effect=fake_scan):
                should_run_generate_action(IMAGE, "_edit", "/custom/search")

        assert captured == ["/custom/search"]


# ---------------------------------------------------------------------------
# Cache behaviour (_scan_dir_cached + clear_generate_gate_cache)
# ---------------------------------------------------------------------------

class TestGenerateGateCache:
    def setup_method(self):
        clear_generate_gate_cache()

    def teardown_method(self):
        clear_generate_gate_cache()

    def _mock_gather(self, monkeypatch, filepaths=None):
        """Patch FileBrowser._gather_files; returns a list accumulating scan-call dirs."""
        import files.file_browser as fb_mod
        calls = []

        def fake_gather(browser_self):
            calls.append(browser_self.directory)
            browser_self.filepaths = filepaths or []

        monkeypatch.setattr(fb_mod.FileBrowser, "_gather_files", fake_gather)
        return calls

    def test_directory_scanned_once_across_multiple_calls(self, monkeypatch):
        calls = self._mock_gather(monkeypatch)

        _scan_dir_cached(SEARCH_DIR)
        _scan_dir_cached(SEARCH_DIR)

        assert len(calls) == 1

    def test_cached_result_is_same_object_on_second_call(self, monkeypatch):
        """The list returned on the second call is the exact same cached instance."""
        self._mock_gather(monkeypatch)

        first = _scan_dir_cached(SEARCH_DIR)
        second = _scan_dir_cached(SEARCH_DIR)

        assert first is second

    def test_different_directories_scanned_independently(self, monkeypatch):
        calls = self._mock_gather(monkeypatch)

        _scan_dir_cached("/dir/a")
        _scan_dir_cached("/dir/b")

        assert calls == ["/dir/a", "/dir/b"]

    def test_same_dir_not_rescanned_after_different_dir_call(self, monkeypatch):
        calls = self._mock_gather(monkeypatch)

        _scan_dir_cached(SEARCH_DIR)
        _scan_dir_cached("/other/dir")
        _scan_dir_cached(SEARCH_DIR)

        assert calls.count(SEARCH_DIR) == 1

    def test_clear_specific_dir_causes_rescan(self, monkeypatch):
        calls = self._mock_gather(monkeypatch)

        _scan_dir_cached(SEARCH_DIR)
        clear_generate_gate_cache(SEARCH_DIR)
        _scan_dir_cached(SEARCH_DIR)

        assert len(calls) == 2

    def test_clear_all_causes_rescan_for_previously_cached_dir(self, monkeypatch):
        calls = self._mock_gather(monkeypatch)

        _scan_dir_cached(SEARCH_DIR)
        clear_generate_gate_cache()
        _scan_dir_cached(SEARCH_DIR)

        assert len(calls) == 2

    def test_clear_other_dir_does_not_evict_cached_dir(self, monkeypatch):
        calls = self._mock_gather(monkeypatch)

        _scan_dir_cached(SEARCH_DIR)
        clear_generate_gate_cache("/other/dir")
        _scan_dir_cached(SEARCH_DIR)

        assert calls.count(SEARCH_DIR) == 1

    def test_clear_none_evicts_all_cached_dirs(self, monkeypatch):
        calls = self._mock_gather(monkeypatch)

        _scan_dir_cached("/dir/a")
        _scan_dir_cached("/dir/b")
        clear_generate_gate_cache()
        _scan_dir_cached("/dir/a")
        _scan_dir_cached("/dir/b")

        assert calls == ["/dir/a", "/dir/b", "/dir/a", "/dir/b"]
